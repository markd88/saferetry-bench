#!/usr/bin/env python3
"""Prompt experiment: does a one-sentence retry-safety instruction (system prompt P1) substitute for a
record lookup? Compares P1 with the neutral prompt P0 of the main experiment, with an aggregate-only
lookup (U) at the baseline condition C0, per agent.

  python3 scripts/prompt_experiment.py [results/full.jsonl.gz] [results/prompt.jsonl.gz]

Units are base tasks (core) or probe tasks; differences are P1 - P0 in percentage points with a paired-t
95% CI and a two-sided sign-flip permutation p-value; Holm correction across the six agents per outcome.
The P1 - V row compares P1 in U with the neutral prompt in V (how much of the V/U gap the prompt closes).
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import analyze as A  # noqa: E402

AGENTS = ["openai/gpt-6-luna", "openai/gpt-6-sol", "openai/gpt-6-sol@max", "qwen/qwen3.8-flash",
          "google/gemini-3.8-flash", "deepseek/deepseek-v4.1-flash"]
UNSAFE = lambda f: f.label == "UNSAFE"
STOPPED = lambda f: f.label.isin(A.STOPPED)
DONE = lambda f: f.label == "SUCCESS"
READ = lambda f: f.read_before_act.astype(float)   # undefined (no write) -> NaN, skipped in means
BASE = lambda f: f.baseline_and_recheck.astype(float)


def load(full, prompt):
    df = pd.concat([A.load(full), A.load(prompt)], ignore_index=True)
    df = df[~df.agent.str.startswith("scripted:") & (df.condition == "C0")]
    return df


def cells(df, agent):
    g = df[df.agent == agent]
    core = g[g.part == "core"]
    return {"P0U": core[(core.prompt == "P0") & (core.verif == "U")],
            "P1U": core[(core.prompt == "P1") & (core.verif == "U")],
            "P0V": core[(core.prompt == "P0") & (core.verif == "V")],
            "pr0": g[(g.part == "probe") & (g.prompt == "P0")],
            "pr1": g[(g.part == "probe") & (g.prompt == "P1")]}


OUTCOMES = [  # (name, which cells, hit, fault runs only, unit)
    ("unsafe (fault)", ("P0U", "P1U"), UNSAFE, True, "base"),
    ("stopped", ("P0U", "P1U"), STOPPED, False, "base"),
    ("completed", ("P0U", "P1U"), DONE, False, "base"),
    ("probe give-up", ("pr0", "pr1"), STOPPED, False, "task"),
    ("read before act", ("P0U", "P1U"), READ, False, "base"),
    ("baseline and recheck", ("P0U", "P1U"), BASE, False, "base"),
]


def analyse(df):
    rows = []
    for name, (a, b), hit, fault_only, unit in OUTCOMES:
        fam = []
        for ag in AGENTS:
            c = cells(df, ag)
            x, y = c[a], c[b]
            if fault_only:
                x, y = x[x.variant == "fault"], y[y.variant == "fault"]
            d = A._diff(x, y, unit, hit)
            m, lo, hi, p = A._boot(d, "two-sided")
            fam.append({"outcome": name, "agent": ag, "P0": 100 * hit(x).mean(), "P1": 100 * hit(y).mean(),
                        "n_runs": f"{len(x)}/{len(y)}", "diff_pp": 100 * m, "lo": 100 * lo, "hi": 100 * hi, "p": p})
        for r, ph in zip(fam, A.holm([r["p"] for r in fam])):
            r["p_holm"] = ph
        rows += fam
    return pd.DataFrame(rows)


def gap_to_v(df):
    rows = []
    for ag in AGENTS:
        c = cells(df, ag)
        r = {"agent": ag}
        for name, hit, fo in (("unsafe", UNSAFE, True), ("completed", DONE, False), ("stopped", STOPPED, False)):
            for k in ("P0V", "P0U", "P1U"):
                f = c[k][c[k].variant == "fault"] if fo else c[k]
                r[f"{name} {k}"] = 100 * hit(f).mean()
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    full = sys.argv[1] if len(sys.argv) > 1 else "results/full.jsonl.gz"
    prompt = sys.argv[2] if len(sys.argv) > 2 else ("results/prompt.jsonl.gz" if os.path.exists("results/prompt.jsonl.gz")
                                                     else "results/prompt.jsonl")
    df = load(full, prompt)
    n1 = df[df.prompt == "P1"].groupby("agent").size()
    print("P1 runs per agent:", n1.to_dict(), "\n")
    res = analyse(df)
    pd.set_option("display.width", 200)
    for name, g in res.groupby("outcome", sort=False):
        print(f"== {name}: P0 vs P1 (U, C0), % ; diff = P1 - P0 in pp ==")
        for r in g.itertuples():
            print(f"  {r.agent:30s} {r.P0:5.1f} -> {r.P1:5.1f}  ({r.n_runs:>7s})  {r.diff_pp:+6.1f} [{r.lo:+.0f},{r.hi:+.0f}]"
                  f"  p={r.p:.4f}  p_holm={r.p_holm:.4f}")
        print()
    print("== V/U gap: neutral prompt in V, neutral in U, P1 in U (C0), % ==")
    print(gap_to_v(df).round(1).to_string(index=False))
    p1 = df[(df.prompt == "P1") & (df.part == "core")]
    fs = p1[p1.label == "UNSAFE"].false_success == True  # noqa: E712
    print(f"\nP1 unsafe runs reporting success: {int(fs.sum())}/{len(fs)}")
    res.to_csv(os.path.join("results", "prompt_tests.csv"), index=False)


if __name__ == "__main__":
    main()
