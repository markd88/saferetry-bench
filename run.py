#!/usr/bin/env python3
"""Run the SafeRetry-Bench grid and append one JSON line per run (runs already in --out are skipped).

Examples
  # scripted reference policies only (free, no API key) -- validates the oracle
  python run.py --scripted --models ""

  # the full experiment: 6 agent configurations with pinned providers (configs/full.json)
  export OPENAI_API_KEY=...                # an OpenRouter key
  python run.py --config configs/full.json --scripted

  # prompt experiment: neutral prompt + retry-safety instruction, aggregate-only lookup, baseline condition
  python run.py --config configs/full.json --prompt P1 --verif U --conditions C0 --out results/prompt.jsonl

  # a cheap smoke test first: 1 task per category, baseline only, 1 rep
  python run.py --config configs/full.json --tasks A1-V,B1-V,C1-V --conditions C0 --reps 1 --out results/smoke.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import datetime
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from saferetry.conditions import CONDITIONS
from saferetry.env import Env
from saferetry.oracle import label
from saferetry.scripted import POLICIES
from saferetry.tasks import FULL_BY_ID, full_grid

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def run_key(r: dict) -> tuple:
    return (r["agent"], r["task"], r["variant"], r["condition"], r["rep"], r.get("prompt", "P0"))


def one_run(agent: str, task, variant: str, condition: str, rep: int, client, args, model_cfg: dict,
            limiter=None) -> dict:
    env = Env(task, variant, condition)
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    extra = {}
    if agent.startswith("scripted:"):
        POLICIES[agent.split(":", 1)[1]](env)
    else:
        from saferetry.agent import openrouter_extra_body, run_llm
        body = openrouter_extra_body(model_cfg.get("provider"), model_cfg.get("reasoning"))
        if model_cfg.get("max_tokens"):
            body["max_tokens"] = model_cfg["max_tokens"]   # fixes the reasoning budget of budget-based models
        extra = run_llm(env, client, model_cfg.get("id", agent), max_steps=args.max_steps, extra_body=body,
                        limiter=limiter,
                        prompt_id=args.prompt)
    out = {"agent": agent, "task": task.id, "category": task.category, "variant": variant,
           "condition": condition, "rep": rep, "suite": args.suite,
           "prompt": args.prompt if not agent.startswith("scripted:") else "P0", **label(env),
           "model_config": model_cfg or None,
           "model": model_cfg.get("id", agent) if model_cfg else agent,
           "reasoning_level": ((model_cfg or {}).get("reasoning") or {}).get("effort")
           or ("off" if ((model_cfg or {}).get("reasoning") or {}).get("enabled") is False else None),
           "calls": env.calls, "effects": env.effects, "jobs": env.jobs,
           "usage": extra.get("usage"), "api_error": extra.get("api_error"),
           "parallel_turns": extra.get("parallel_turns", 0),
           "served": extra.get("served"),
           "hit_step_limit": extra.get("hit_step_limit", False), "max_steps": args.max_steps,
           "started_at": started_at, "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    if args.save_messages and "messages" in extra:
        out["messages"] = extra["messages"]
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="", help="JSON file with models, pinned providers, reasoning settings")
    p.add_argument("--models", default="", help="comma-separated model ids (no provider pinning)")
    p.add_argument("--scripted", action="store_true", help="also run the 3 scripted policies")
    p.add_argument("--tasks", default="", help="comma-separated task ids (default: all)")
    p.add_argument("--conditions", default=",".join(CONDITIONS))
    p.add_argument("--variants", default="fault,control")
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--probe-reps", type=int, default=None,
                   help="reps for probe tasks (default: 2 x --reps; the experiment used core 2, probe 4)")
    p.add_argument("--max-steps", type=int, default=None,
                   help="tool-call limit per run (default: 30)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--rpm", type=float, default=15,
                   help="max requests/minute per model (config 'rpm' overrides; new OpenRouter accounts: 20)")
    p.add_argument("--out", default="results/full.jsonl")
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    p.add_argument("--no-messages", dest="save_messages", action="store_false")
    p.add_argument("--verif", default="V,U", help="which verifiability versions of core tasks to run (probes always run)")
    p.add_argument("--prompt", default="P0", choices=["P0", "P1"],
                   help="system prompt: P0 neutral (main experiment), P1 adds a retry-safety instruction")
    p.add_argument("--dry-run", action="store_true", help="print the number of runs to do and exit")
    args = p.parse_args()

    args.suite = "full"
    if args.scripted and args.prompt != "P0":
        p.error("scripted policies ignore the prompt; run them with --prompt P0")
    if args.max_steps is None:
        args.max_steps = 30
    by_id = FULL_BY_ID
    want_tasks = set(args.tasks.split(",")) if args.tasks else None
    want_variants, want_conds = set(args.variants.split(",")), set(args.conditions.split(","))
    want_verif = set(args.verif.split(","))
    grid = [(t, v, c) for t, v, c in full_grid() if t.part == "probe" or t.verif in want_verif]
    grid = [(t, v, c) for t, v, c in grid if (want_tasks is None or t.id in want_tasks)
            and v in want_variants and c in want_conds]
    model_cfgs: dict[str, dict] = {m: {} for m in args.models.split(",") if m}
    if args.config:
        with open(args.config) as f:
            for m in json.load(f)["models"]:
                # "name" lets one model run under several settings (e.g. "openai/gpt-6-luna@high");
                # the name is the agent label in results, "id" is what is sent to the API
                model_cfgs[m.get("name", m["id"])] = m
    agents = list(model_cfgs)
    if args.scripted:
        agents += [f"scripted:{name}" for name in POLICIES]
    if not agents:
        p.error("give --config, --models and/or --scripted")

    client = None
    if args.dry_run:
        pass
    elif any(not a.startswith("scripted:") for a in agents):
        key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            p.error("set OPENAI_API_KEY (or OPENROUTER_API_KEY)")
        from openai import OpenAI
        client = OpenAI(base_url=args.base_url, api_key=key)

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            # runs that hit a model-API error are incomplete measurements: redo them
            done = {run_key(r) for r in map(json.loads, filter(str.strip, f)) if not r.get("api_error")}

    probe_reps = args.probe_reps if args.probe_reps is not None else 2 * args.reps
    jobs = []
    for agent in agents:
        reps = 1 if agent.startswith("scripted:") else args.reps
        prompt = "P0" if agent.startswith("scripted:") else args.prompt
        for task, variant, condition in grid:
            n = reps if (agent.startswith("scripted:") or getattr(task, "part", "core") != "probe") else probe_reps
            for rep in range(n):
                if (agent, task.id, variant, condition, rep, prompt) not in done:
                    jobs.append((agent, task.id, variant, condition, rep))
    # Scripted runs first (instant), then interleave models so every model's rate
    # limit is used at the same time instead of one model after another.
    agent_rank = {a: i for i, a in enumerate(agents)}
    jobs.sort(key=lambda k: (not k[0].startswith("scripted:"), k[4], k[1], k[2], k[3], agent_rank[k[0]]))
    print(f"{len(jobs)} runs to do ({len(done)} already in {args.out})", file=sys.stderr)
    if args.dry_run:
        from collections import Counter
        for a, n in Counter(j[0] for j in jobs).items():
            print(f"  {a}: {n}", file=sys.stderr)
        return 0

    from saferetry.agent import RateLimiter
    limiters = {a: RateLimiter(model_cfgs.get(a, {}).get("rpm", args.rpm)) for a in agents
                if not a.startswith("scripted:")}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    lock = threading.Lock()
    counts: dict[str, int] = {}
    with open(args.out, "a") as f, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(one_run, a, by_id[t], v, c, r, client, args, model_cfgs.get(a, {}),
                            limiters.get(a)): (a, t, v, c, r)
                for a, t, v, c, r in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                rec = fut.result()
            except Exception as e:  # keep going; a crashed run is simply retried next time
                print(f"run {futs[fut]} crashed: {e!r}", file=sys.stderr)
                continue
            if rec.get("api_error"):
                # not saved, so rerunning the same command retries it from scratch
                counts["API_ERROR"] = counts.get("API_ERROR", 0) + 1
                print(f"run {futs[fut]} not saved (model API error): {rec['api_error'][:160]}", file=sys.stderr)
                continue
            with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
            counts[rec["label"]] = counts.get(rec["label"], 0) + 1
            if i % 20 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] {counts}", file=sys.stderr)
    if counts.get("API_ERROR"):
        print(f"{counts['API_ERROR']} runs hit model-API errors and were not saved; "
              f"rerun the same command to retry them.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
