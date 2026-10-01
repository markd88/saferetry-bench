#!/usr/bin/env python3
"""Print one run as a readable transcript: what the model was told, which tools it had,
what it called, what the (mock) backend returned, how it ended, and how it was labelled.

  python3 scripts/show_run.py results/full.jsonl.gz --list                  # one line per run
  python3 scripts/show_run.py results/full.jsonl.gz --agent qwen --task A6-U
  python3 scripts/show_run.py results/full.jsonl.gz --label UNSAFE           # every matching run
  python3 scripts/show_run.py results/full.jsonl.gz --agent luna --task B8-U --tools   # + tool schemas
  python3 scripts/show_run.py results/full.jsonl.gz --index 7                # the 8th run in the file

Filters combine (AND). --agent matches a substring of the agent name; the others match exactly.
If a run has no saved messages (older files, or --no-messages), the call log is shown instead.
"""
import argparse
import json
import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WIDTH = 110
RULE = "-" * WIDTH


def wrap(text, indent="    "):
    text = "" if text is None else str(text)
    out = []
    for line in text.splitlines() or [""]:
        out += textwrap.wrap(line, WIDTH - len(indent), initial_indent=indent, subsequent_indent=indent) or [indent]
    return "\n".join(out)


def pretty_json(s):
    """Tool arguments / results arrive as JSON strings or objects; show them compactly but readably."""
    if isinstance(s, str):
        try:
            s = json.loads(s)
        except (ValueError, TypeError):
            return s
    return json.dumps(s, ensure_ascii=False, indent=None)


def get(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def header(r, i):
    served = r.get("served") or []
    rt = sum((m.get("reasoning_tokens") or 0) for m in served)
    providers = sorted({str(m.get("provider")) for m in served}) or ["-"]
    usage = r.get("usage") or {}
    lines = [
        "=" * WIDTH,
        f"#{i}  {r['agent']}   task={r['task']}  variant={r['variant']}  condition={r['condition']}  "
        f"rep={r.get('rep')}  prompt={r.get('prompt', 'P0')}",
        f"    LABEL: {r['label']}" + (f" ({r['unsafe_kind']})" if r.get("unsafe_kind") else "")
        + f"   ended with: {r.get('terminal')} {r.get('reported_outcome') or ''}"
        + f"   final state: {r.get('state')}",
        f"    part={r.get('part', 'core')}  verif={r.get('verif', 'V')}  steps={r.get('steps')}  "
        f"read_before_act={r.get('read_before_act')}  verified_before_next_action={r.get('verified_before_next_action')}"
        + (f"  key_used={r.get('key_used')} key_correct={r.get('key_correct')}" if "key_used" in r else ""),
        f"    provider={','.join(providers)}  reasoning_level={r.get('reasoning_level')}  reasoning_tokens={rt}  "
        f"tokens={usage.get('prompt_tokens')} in / {usage.get('completion_tokens')} out"
        + ("  HIT STEP LIMIT" if r.get("hit_step_limit") else ""),
        "=" * WIDTH,
    ]
    return "\n".join(lines)


def show_tools(r):
    """Rebuild the environment to print exactly the tool schemas the model was given."""
    try:
        from saferetry.env import Env
        from saferetry.tasks import FULL_BY_ID as by_id
        env = Env(by_id[r["task"]], r["variant"], r["condition"])
        print("TOOLS GIVEN TO THE MODEL:")
        for t in env.tool_schemas():
            f = t["function"]
            params = f.get("parameters", {}).get("properties", {})
            req = set(f.get("parameters", {}).get("required", []))
            args = ", ".join(f"{k}{'' if k in req else '?'}" for k in params)
            print(f"  - {f['name']}({args}): {f.get('description', '')}")
            if "idempotency_key" in params:
                print(wrap(f"idempotency_key: {params['idempotency_key'].get('description', '')}", "      "))
        print(RULE)
    except Exception as e:  # never let the viewer crash on an old record
        print(f"(could not rebuild tool schemas: {e!r})\n{RULE}")


def show_messages(msgs):
    for m in msgs:
        role = get(m, "role")
        content = get(m, "content")
        if isinstance(content, list):          # content parts
            content = " ".join(str(get(p, "text", p)) for p in content)
        if role == "system":
            print("[SYSTEM PROMPT]")
            print(wrap(content))
        elif role == "user":
            print("[USER / TASK]")
            print(wrap(content))
        elif role == "assistant":
            reasoning = get(m, "reasoning") or get(m, "reasoning_content")
            if reasoning:
                print("[MODEL - reasoning]")
                print(wrap(reasoning))
            if content:
                print("[MODEL - says]")
                print(wrap(content))
            for tc in get(m, "tool_calls") or []:
                fn = get(tc, "function") or {}
                print(f"[MODEL -> calls] {get(fn, 'name')}({pretty_json(get(fn, 'arguments'))})")
        elif role == "tool":
            print("[BACKEND -> returns]")
            print(wrap(pretty_json(content)))
        else:
            print(f"[{role}]")
            print(wrap(content))
        print()


def show_calls(r):
    print("(no saved messages; showing the call log)")
    for c in r.get("calls") or []:
        print(f"[step {c.get('step')}] MODEL -> {c.get('tool')}({pretty_json(c.get('args'))})")
        print(wrap("BACKEND -> " + pretty_json(c.get("result"))))
    print()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path")
    p.add_argument("--agent", help="substring of the agent name, e.g. qwen, luna, sol@max")
    p.add_argument("--task", help="exact task id, e.g. A6-U")
    p.add_argument("--variant", choices=["fault", "control"])
    p.add_argument("--condition", choices=["C0", "C1", "C2", "C3"])
    p.add_argument("--label", choices=["UNSAFE", "SUCCESS", "MISREPORT", "OVER_CAUTIOUS", "INCOMPLETE"])
    p.add_argument("--index", type=int, help="position in the file (0-based)")
    p.add_argument("--list", action="store_true", help="one summary line per matching run")
    p.add_argument("--tools", action="store_true", help="also print the tool schemas the model was given")
    p.add_argument("--calls", action="store_true", help="show the call log instead of the messages")
    p.add_argument("--max", type=int, default=20, help="stop after this many runs (default 20)")
    a = p.parse_args()

    import gzip
    opener = gzip.open if a.path.endswith(".gz") else open
    rows = [json.loads(l) for l in opener(a.path, "rt") if l.strip()]
    picked = []
    for i, r in enumerate(rows):
        if a.index is not None and i != a.index:
            continue
        if a.agent and a.agent.lower() not in r["agent"].lower():
            continue
        if a.task and r["task"] != a.task:
            continue
        if a.variant and r["variant"] != a.variant:
            continue
        if a.condition and r["condition"] != a.condition:
            continue
        if a.label and r["label"] != a.label:
            continue
        picked.append((i, r))
    if not picked:
        sys.exit("no run matches these filters (try --list without filters)")

    if a.list:
        for i, r in picked:
            print(f"#{i:<5} {r['agent']:32} {r['task']:7} {r['variant']:8} {r['condition']}  rep={r.get('rep')}  "
                  f"{r['label']:13} steps={r.get('steps')}")
        print(f"{len(picked)} runs")
        return

    for n, (i, r) in enumerate(picked):
        if n >= a.max:
            print(f"... {len(picked) - a.max} more runs match; narrow the filters or raise --max")
            break
        print(header(r, i))
        if a.tools:
            show_tools(r)
        if r.get("messages") and not a.calls:
            show_messages(r["messages"])
        else:
            show_calls(r)


if __name__ == "__main__":
    main()
