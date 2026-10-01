#!/usr/bin/env python3
"""Which reasoning settings does each model actually honour? Run BEFORE freezing the analysis plan.

  python3 scripts/check_reasoning.py configs/full.json | tee results/reasoning_check.txt
  python3 scripts/check_reasoning.py configs/full.json --write-config    # then also rewrite the config
  python3 scripts/check_reasoning.py configs/full.json --only deepseek/deepseek-v4.1-flash --write-config
      # re-check one model; the other models keep their recorded settings

Design rule (planned:
  * the five main models run at their LOWEST reasoning setting: switched off where the API allows it
    ("none" is accepted AND no reasoning is observed), otherwise "minimal" (OpenRouter maps a level a
    model does not support to the nearest supported one, so "minimal" = the model's lowest level);
  * one extra arm, GPT-6 Sol, runs at its HIGHEST setting: "max" (mapped down to the highest level
    the model supports).

For every model the same prompt is sent at every effort level (2 samples each). The table shows the
provider that served it, reasoning tokens / reasoning text length, or the error. A level that is
silently mapped to another shows up as identical reasoning use. Cost: well under $1.
"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openai import OpenAI  # noqa: E402

from saferetry.agent import openrouter_extra_body, response_meta  # noqa: E402

LEVELS = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]
LOW_MAX_TOKENS, HIGH_MAX_TOKENS = 4096, 32000   # same caps as the run: low arms 4096, max arm 32000
SAMPLES = 2
PROMPT = ("A warehouse ships 3 kinds of boxes. Small weighs 2 kg, medium 5 kg, large 9 kg. A truck carried "
          "exactly 50 kg using at least one of each kind and the fewest boxes possible. How many of each? "
          "Answer with three numbers.")
FRONTIER = "openai/gpt-6-sol"


def probe(client, m, effort):
    body = openrouter_extra_body(m.get("provider"), {"effort": effort})
    body["max_tokens"] = HIGH_MAX_TOKENS if effort in ("high", "xhigh", "max") else LOW_MAX_TOKENS
    out = []
    for _ in range(SAMPLES):
        try:
            r = client.chat.completions.create(model=m["id"], messages=[{"role": "user", "content": PROMPT}],
                                               extra_body=body)
            meta = response_meta(r)
            out.append({"ok": True, "provider": meta["provider"], "rt": meta["reasoning_tokens"],
                        "rc": meta["reasoning_chars"] or 0,
                        "ct": getattr(r.usage, "completion_tokens", None) if r.usage else None})
        except Exception as e:
            out.append({"ok": False, "err": str(e)[:100]})
    return out


def main() -> int:
    cfg_path = next((a for a in sys.argv[1:] if not a.startswith("--")), "configs/full.json")
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("set OPENAI_API_KEY (or OPENROUTER_API_KEY)")
    client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"), api_key=key)
    cfg = json.load(open(cfg_path))
    base = {}
    for m in cfg["models"]:                         # one entry per model id (drop the @max arm)
        base.setdefault(m["id"], {k: v for k, v in m.items() if k not in ("name", "reasoning", "max_tokens")})
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    previous = dict(cfg.get("_reasoning_check", {}).get("lowest", {}))
    print(f"config: {cfg_path}   prompt samples per level: {SAMPLES}" + (f"   only: {only}" if only else ""))
    print(f"{'model':30} {'effort':8} {'provider':16} {'ok':>3} {'reasoning_tok':>14} {'reasoning_chars':>15} "
          f"{'completion_tok':>14}  note")
    decisions = {}
    for mid, m in base.items():
        if only and mid != only:
            continue
        seen = {}
        for effort in LEVELS:
            res = probe(client, m, effort)
            ok = [r for r in res if r["ok"]]
            rt = [r["rt"] or 0 for r in ok]
            rc = [r["rc"] for r in ok]
            ct = [r["ct"] or 0 for r in ok]
            seen[effort] = {"ok": len(ok), "rt": rt, "rc": rc}
            note = "" if len(ok) == SAMPLES else "ERROR " + next(r["err"] for r in res if not r["ok"])
            prov = ok[0]["provider"] if ok else "-"
            print(f"{mid:30} {effort:8} {str(prov):16} {len(ok):>3} "
                  f"{(statistics.mean(rt) if rt else float('nan')):>14.0f} {(statistics.mean(rc) if rc else float('nan')):>15.0f} "
                  f"{(statistics.mean(ct) if ct else float('nan')):>14.0f}  {note}")
        none = seen["none"]
        can_off = none["ok"] == SAMPLES and not any(none["rt"]) and not any(none["rc"])
        lowest = "none" if can_off else "minimal"
        decisions[mid] = lowest
        print(f"  -> {mid}: lowest = {lowest!r} ({'reasoning switched off' if can_off else 'cannot switch off'})")
    decisions = {**{k: v for k, v in previous.items() if k in base}, **decisions}
    missing = [mid for mid in base if mid not in decisions]
    if missing:
        sys.exit(f"no recorded setting for {missing}; run without --only first")
    print("\nrun configuration (planned rule):")
    for mid, low in decisions.items():
        print(f"  {mid}: effort={low}")
    print(f"  {FRONTIER}@max: effort=max")

    if "--write-config" in sys.argv:
        models = []
        for mid, m in base.items():
            models.append(dict(m, reasoning={"effort": decisions[mid]}, max_tokens=LOW_MAX_TOKENS))
        if FRONTIER in base:
            models.append(dict(base[FRONTIER], name=f"{FRONTIER}@max", reasoning={"effort": "max"},
                               max_tokens=HIGH_MAX_TOKENS,
                               _role="strongest configuration: frontier model at its highest reasoning setting"))
        cfg["models"] = models
        cfg["_reasoning_check"] = {"lowest": decisions, "script": "scripts/check_reasoning.py",
                                   "log": "results/reasoning_check.txt"}
        json.dump(cfg, open(cfg_path, "w"), indent=2)
        print(f"\nwrote {cfg_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
