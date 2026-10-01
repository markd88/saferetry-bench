#!/usr/bin/env python3
"""Result tables (markdown) generated from the code and results/full.jsonl.gz: task list, exact wording,
agent configurations, all tests, per-cell outcomes, key use and diagnostics.

  python3 scripts/make_tables.py > results/tables.md
"""
import contextlib
import io
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import analyze as A  # noqa: E402
from saferetry.agent import SYSTEM_PROMPT  # noqa: E402
from saferetry.env import Env  # noqa: E402
from saferetry.tasks import CORE, FULL_TASKS  # noqa: E402

AGENTS = ["openai/gpt-6-luna", "openai/gpt-6-sol", "openai/gpt-6-sol@max", "qwen/qwen3.8-flash",
          "google/gemini-3.8-flash", "deepseek/deepseek-v4.1-flash"]
NAME = {"openai/gpt-6-luna": "GPT-6 Luna", "openai/gpt-6-sol": "GPT-6 Sol", "openai/gpt-6-sol@max": "GPT-6 Sol @max",
        "qwen/qwen3.8-flash": "Qwen3.8 Flash", "google/gemini-3.8-flash": "Gemini 3.8 Flash",
        "deepseek/deepseek-v4.1-flash": "DeepSeek V4.1 Flash"}
CAT = {"A": "single write", "B": "async job", "C": "batch"}
STOP = A.STOPPED


def pct(x):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.0f}%"


def fmt_p(p):
    return "<0.0001" if p < 0.0001 else f"{p:.4f}"


def tools(t):
    return [s["function"]["name"] for s in Env(t, "fault", "C0").tool_schemas() if s["function"]["name"] not in ("finish", "ask_user")]


def section_tasks():
    print("## A.1 Core tasks\n")
    print("Each base task has a V and a U version (same instruction and write tool; only the lookup differs), "
          "and a fault and a control variant.\n")
    print("Category = first letter of the id (A single write, B async job, C batch).\n")
    print("| Base | Domain | Instruction | Tools (V) | Lookup in U |")
    print("|---|---|---|---|---|")
    by = {t.id: t for t in CORE}
    bases = sorted({t.id.rsplit("-", 1)[0] for t in CORE}, key=lambda b: (b[0], int(b[1:])))
    for b in bases:
        v, u = by[f"{b}-V"], by[f"{b}-U"]
        tv, tu = tools(v), tools(u)
        lookup_u = (getattr(u, "lookup", None) or u.read).name     # the lookup whose response the U version changes
        only_u = [f"`{lookup_u}`" + ("" if lookup_u not in tv else " (aggregate view)")]
        print(f"| {b} | {v.domain} | {v.instruction} | "
              f"{', '.join(f'`{x}`' for x in tv)} | {', '.join(only_u)} |")
    print("\n## A.2 Probe tasks (definitive failure; retrying is safe)\n")
    print("| Probe | Domain | Instruction | Tools |")
    print("|---|---|---|---|")
    for t in [t for t in FULL_TASKS if t.part == "probe"]:
        print(f"| {t.id} | {t.domain} | {t.instruction} | {', '.join(f'`{x}`' for x in tools(t))} |")
    print()


def section_wording():
    from saferetry import conditions as c
    print("## A.3 Exact condition wording\n")
    L = "<lookup>"   # replaced by the task's own lookup tool name, e.g. list_orders / get_customer_balance
    rows = [
        ("Core A: write times out (504)", lambda k: c.timeout_message(k, L), ["C0", "C1", "C2", "C3"]),
        ("Core B: job record lost (404 on get_job_status)", lambda k: c.job_lost_message(k, L), ["C0", "C1", "C2", "C3"]),
        ("Core C: batch with items 3, 5 = `timeout` (207)", lambda k: c.batch_unknown_message(k), ["C0", "C1", "C2", "C3"]),
        ("Probe AF: write unavailable (503)", lambda k: c.unavailable_message(k), ["C0", "C2", "C3"]),
        ("Probe BF: submit accepted (202)", lambda k: c.async_accepted_message(k), ["C0", "C2", "C3"]),
        ("Probe BF: job ends `failed`", lambda k: c.async_failed_message(k), ["C0", "C2", "C3"]),
        ("Probe CF: all items failed (207)", lambda k: c.batch_all_failed_message(k), ["C0", "C2", "C3"]),
    ]
    print("| Situation | Condition | Exact message shown to the agent |")
    print("|---|---|---|")
    for name, msg, conds in rows:
        for k in conds:
            print(f"| {name} | {k} | {msg(k)} |")
    print()
    print(f"`{L}` is replaced by the task's own lookup tool (V: the record lookup; U: the aggregate lookup).")
    print("Core B accepts every submission with the constant message `Accepted.` in all conditions.")
    print()
    print(f"C1 adds an optional `idempotency_key` parameter to the operation whose outcome can be uncertain (the write, the job submission, or every item in batch tools), "
          f"with this description: \"{c.IDEMPOTENCY_DESCRIPTION}\"")
    print()


def section_agents():
    print("## B.1 System prompt (identical for all agents)\n")
    print("```\n" + SYSTEM_PROMPT + "\n```\n")
    print("## B.2 Agent configurations\n")
    cfg = json.load(open(os.path.join(HERE, "configs", "full.json")))
    print("| Agent | OpenRouter id | Provider (pinned, no fallback) | Reasoning effort | max_tokens |")
    print("|---|---|---|---|---|")
    for m in cfg["models"]:
        label = m.get("name", m["id"])
        print(f"| {NAME.get(label, label)} | `{m['id']}` | {m['provider']} | {m.get('reasoning', {}).get('effort', '-')} | "
              f"{m.get('max_tokens', '-')} |")
    print("\nTemperature, top_p and seed are not sent (provider defaults); `tool_choice` = auto; step limit 30. "
          "One tool-calling loop for all agents: every tool call in a turn is executed in order and its result returned.\n")


def section_stats(df):
    llm = df[~df.agent.str.startswith("scripted:") & (df.suite == "full")]
    with contextlib.redirect_stdout(io.StringIO()):
        res = A.hypotheses(df)
        rc = A.reasoning_check(df)
    print("## C.1 All hypothesis tests (all agents)\n")
    print("Unit = task (base task for V/U). p: paired sign-flip permutation (10,000); CI: paired t 95%; "
          "p_holm: Holm within family per agent (empty = estimate, no correction). "
          "H column: H2-H5 as in the paper; S1 = supplementary contrast (does the C0-C3 unsafe drop differ between V and U?).\n")
    order = {"H3": 0, "H4": 1, "H5a": 2, "H5b": 3, "S1": 4, "S6": 5}
    first = {t: i for i, t in enumerate(dict.fromkeys(res.test))}
    res = res.assign(_o=res.H.map(order), _t=res.test.map(first), _a=res.agent.map({a: i for i, a in enumerate(AGENTS)}))
    res = res.sort_values(["_o", "_t", "_a"])
    print("| H | Test | Agent | Units | Diff (pp) | 95% CI | Alt. | p | p_holm |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in res.itertuples():
        ph = "" if r.p_holm is None or (isinstance(r.p_holm, float) and np.isnan(r.p_holm)) else fmt_p(r.p_holm)
        print(f"| {'H2' if r.H == 'S6' else r.H} | {r.test.replace('H2 estimate: ', '')} | {NAME[r.agent]} | {r.n_units} | {r.diff_pp:+.1f} | {r.ci95_pp} | {r.alt} | {fmt_p(r.p)} | {ph} |")
    print()
    if rc is not None and len(rc):
        print("## C.2 H6: GPT-6 Sol, lowest vs max reasoning\n")
        print("| Test (max − lowest) | Units | Diff (pp) | 95% CI | Alt. | p | p_holm |")
        print("|---|---|---|---|---|---|---|")
        for r in rc.itertuples(index=False):
            print(f"| {r.test.replace(': max - lowest', '')} | {r.n_units} | {r.diff_pp:+.1f} | {r.ci95_pp} | {r.alt} | "
                  f"{fmt_p(r.p)} | {fmt_p(r.p_holm)} |")
        print()

    core = llm[llm.part == "core"]
    print("## C.3 Core outcomes by agent, verifiability and condition\n")
    print("Unsafe over fault runs; stopped and success over all runs (fault + control); Read first = read before the first write; "
          "Baseline + recheck = baseline before and recheck after (U: the strategy that makes the outcome knowable).\n")
    print("| Agent | Verif | Cond | Unsafe (fault) | Stopped | Success | Read first | Baseline + recheck |")
    print("|---|---|---|---|---|---|---|---|")
    for a in AGENTS:
        for v in ("V", "U"):
            for k in ("C0", "C1", "C2", "C3"):
                g = core[(core.agent == a) & (core.verif == v) & (core.condition == k)]
                f = g[g.variant == "fault"]
                s2 = g.read_before_act.dropna().astype(bool).mean()
                s2b = g.baseline_and_recheck.dropna().astype(bool).mean()
                print(f"| {NAME[a]} | {v} | {k} | {pct((f.label == 'UNSAFE').mean())} | {pct(g.label.isin(STOP).mean())} | "
                      f"{pct((g.label == 'SUCCESS').mean())} | {pct(s2)} | {pct(s2b)} |")
    print()

    print("## C.4 Unsafe rate by category (core fault runs, C0)\n")
    print("| Agent | V: A | V: B | V: C | U: A | U: B | U: C |")
    print("|---|---|---|---|---|---|---|")
    for a in AGENTS:
        cells = []
        for v in ("V", "U"):
            for c in "ABC":
                f = core[(core.agent == a) & (core.verif == v) & (core.condition == "C0") & (core.variant == "fault") & (core.category == c)]
                cells.append(f"{(f.label == 'UNSAFE').sum()}/{len(f)}")
        print(f"| {NAME[a]} | " + " | ".join(cells) + " |")
    print()

    print("## C.5 Fault × control patterns (same task version, condition and repetition)\n")
    print("verify = fault SUCCESS & control SUCCESS; blind retry = fault UNSAFE & control SUCCESS; "
          "timid = fault stopped & control stopped; inconsistent = fault UNSAFE & control stopped; other = anything else.\n")
    print("| Agent | Verif | Pairs | verify | blind retry | timid | inconsistent | other |")
    print("|---|---|---|---|---|---|---|---|")
    for a in AGENTS:
        for v in ("V", "U"):
            g = core[(core.agent == a) & (core.verif == v)]
            key = ["task", "condition", "rep"]
            f = g[g.variant == "fault"].set_index(key).label
            c = g[g.variant == "control"].set_index(key).label
            j = pd.concat([f.rename("f"), c.rename("c")], axis=1).dropna()
            def pat(r):
                if r.f == "SUCCESS" and r.c == "SUCCESS": return "verify"
                if r.f == "UNSAFE" and r.c == "SUCCESS": return "blind retry"
                if r.f in STOP and r.c in STOP: return "timid"
                if r.f == "UNSAFE" and r.c in STOP: return "inconsistent"
                return "other"
            cnt = Counter(pat(r) for r in j.itertuples())
            n = len(j)
            print(f"| {NAME[a]} | {v} | {n} | " + " | ".join(pct(cnt[p] / n) for p in ["verify", "blind retry", "timid", "inconsistent", "other"]) + " |")
    print()

    print("## C.6 Key use in C1 (core fault runs)\n")
    print("| Agent | Verif | Key sent with first write, reused | Key sent, no resend needed | Key only on retry / new key | No key | Unsafe |")
    print("|---|---|---|---|---|---|---|")
    c1 = core[(core.condition == "C1") & (core.variant == "fault")]
    for a in AGENTS:
        for v in ("V", "U"):
            g = c1[(c1.agent == a) & (c1.verif == v)]
            used = g.key_used.map(lambda x: bool(x) if x == x and x is not None else False)
            kc = g.key_correct
            correct = (used & (kc == True)).sum()  # noqa: E712
            wrong = (used & (kc == False)).sum()  # noqa: E712
            single = (used & kc.isna()).sum()
            nokey = (~used).sum()
            print(f"| {NAME[a]} | {v} | {correct} | {single} | {wrong} | {nokey} | {(g.label == 'UNSAFE').sum()}/{len(g)} |")
    print()

    print("## C.7 Exploratory: effect of offering the key in U\n")
    print("Unsafe rate, core fault runs in U, C1 − C0, paired over 30 base tasks; sign-flip p (two-sided), paired t 95% CI. "
          "p_holm = Holm across the six agents.\n")
    fu = core[(core.variant == "fault") & (core.verif == "U")]
    rows = []
    for a in AGENTS:
        g = fu[fu.agent == a]
        d = A._diff(g[g.condition == "C0"], g[g.condition == "C1"], "base", lambda x: x.label == "UNSAFE")
        m, lo, hi, p = A._boot(d, "two-sided")
        rows.append((a, m, lo, hi, p))
    ph = A.holm([r[4] for r in rows])
    print("| Agent | C0 unsafe | C1 unsafe | C1 − C0 (pp) | 95% CI | p | p_holm |")
    print("|---|---|---|---|---|---|---|")
    for (a, m, lo, hi, p), q in zip(rows, ph):
        g = fu[fu.agent == a]
        c0 = (g[g.condition == "C0"].label == "UNSAFE").mean(); c1 = (g[g.condition == "C1"].label == "UNSAFE").mean()
        print(f"| {NAME[a]} | {pct(c0)} | {pct(c1)} | {100*m:+.1f} | [{100*lo:.0f}, {100*hi:.0f}] | {fmt_p(p)} | {fmt_p(q)} |")
    print()

    print("## C.8 Other diagnostics\n")
    print("| Agent | Runs | Incomplete | of which false success | Success reported, state wrong: V / U | Unsafe kind: duplicate / follow-up | Mean reasoning tokens per response |")
    print("|---|---|---|---|---|---|---|")
    for a in AGENTS:
        g = llm[llm.agent == a]
        inc = g[g.label == "INCOMPLETE"]
        fs = g[g.false_success.fillna(False).astype(bool)]
        un = g[g.label == "UNSAFE"]
        served = [m for s in g.served for m in (s or [])]
        rt = np.mean([(m.get("reasoning_tokens") or 0) for m in served]) if served else float("nan")
        print(f"| {NAME[a]} | {len(g)} | {len(inc)} | {(inc.incomplete_kind == 'false_success').sum()} | "
              f"{(fs.verif == 'V').sum()} / {(fs.verif == 'U').sum()} | "
              f"{(un.unsafe_kind == 'duplicate').sum()} / {(un.unsafe_kind == 'followup').sum()} | {rt:.0f} |")
    print()


if __name__ == "__main__":
    df = A.load(os.path.join(HERE, "results", "full.jsonl.gz"))
    print("<!-- generated by scripts/make_tables.py; do not edit by hand -->\n")
    section_tasks()
    section_wording()
    section_agents()
    section_stats(df)
