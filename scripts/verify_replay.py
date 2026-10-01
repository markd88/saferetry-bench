#!/usr/bin/env python3
"""Replay every recorded run against the mock backend and check that each tool result and the final label
are reproduced exactly (the backend and oracle are deterministic). Also fingerprints the task suite.

  python3 scripts/verify_replay.py results/full.jsonl.gz
  python3 scripts/verify_replay.py results/full.jsonl.gz --fingerprint /tmp/suite.json   # write the suite fingerprint
"""
import argparse
import gzip
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from saferetry.env import Env  # noqa: E402
from saferetry.oracle import label  # noqa: E402
from saferetry.tasks import FULL_BY_ID, full_grid  # noqa: E402

LABEL_FIELDS = ["label", "unsafe_kind", "incomplete_kind", "false_success", "read_before_act", "baseline_and_recheck",
                "verified_before_next_action", "key_used", "key_correct", "state", "terminal", "reported_outcome"]


def fingerprint() -> dict:
    out = {}
    for task, variant, condition in full_grid():
        env = Env(task, variant, condition)
        blob = json.dumps({"instruction": task.instruction, "tools": env.tool_schemas(),
                           "state": env.state}, sort_keys=True, default=str)
        out[f"{task.id}|{variant}|{condition}"] = hashlib.sha256(blob.encode()).hexdigest()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--fingerprint", help="write the suite fingerprint (json) to this file")
    a = p.parse_args()
    opener = gzip.open if a.path.endswith(".gz") else open
    n = bad = 0
    for line in opener(a.path, "rt"):
        if not line.strip():
            continue
        r = json.loads(line)
        env = Env(FULL_BY_ID[r["task"]], r["variant"], r["condition"])
        ok = True
        for c in r["calls"]:
            if env.call(c["tool"], c["args"]) != c["result"]:
                ok = False
                break
        if ok:
            lab = json.loads(json.dumps(label(env), default=str))
            ok = all(lab.get(k) == r.get(k) for k in LABEL_FIELDS if k in lab or k in r)
        n += 1
        if not ok:
            bad += 1
            if bad <= 5:
                print("MISMATCH", r["agent"], r["task"], r["variant"], r["condition"], r["rep"])
    print(f"replayed {n} runs: {n - bad} reproduced exactly, {bad} mismatches")
    if a.fingerprint:
        json.dump(fingerprint(), open(a.fingerprint, "w"), indent=0, sort_keys=True)
        print("fingerprint written to", a.fingerprint)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
