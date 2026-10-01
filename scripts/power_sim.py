#!/usr/bin/env python3
"""Sensitivity analysis: minimum detectable effect (MDE) at 80% power
for each confirmatory test, under the full design and several between-task ICCs. No model calls.

  python3 scripts/power_sim.py | tee results/power_sim.txt            # MDE table
  python3 scripts/power_sim.py --type1 | tee results/type1_check.txt  # false-positive rate with no effect
  python3 scripts/power_sim.py --coverage | tee results/ci_coverage.txt  # 95% CI coverage

Model: each task (or base task) has its own propensity p_i ~ Beta(mean=p0, ICC=rho); a condition
shifts every task's propensity by the same amount (clipped to [0,1]); each run is Bernoulli(p_i).
Tests are the planned ones (analyze._boot: sign-flip permutation p-value over tasks, paired t CI),
one-sided, at the Holm first-step alpha of the test's family (conservative).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import analyze  # noqa: E402

SIMS, B = 200, 1000
RNG = np.random.default_rng(1)


def propensities(n, p0, rho):
    if rho <= 0:
        return np.full(n, p0)
    k = 1 / rho - 1
    return RNG.beta(p0 * k, (1 - p0) * k, n)


def rate(p, reps):
    return RNG.binomial(reps, np.clip(p, 0, 1)) / reps


def power(test, d, rho, alpha):
    hits = 0
    for _ in range(SIMS):
        diff = test(d, rho)
        _, _, _, p = analyze._boot(diff, "greater", B=B)
        hits += p < alpha
    return hits / SIMS


# each returns per-unit differences oriented so that the hypothesised effect is positive
def h2(d, rho):        # unsafe C0 - C2, core fault V: 30 tasks x 2 reps per condition, baseline 30%
    p = propensities(30, 0.30, rho)
    return rate(p, 2) - rate(p - d, 2)


def h3(d, rho):        # give-up C0 - C3 on probes: 15 tasks x 4 reps (planned), baseline 50%
    p = propensities(15, 0.50, rho)
    return rate(p, 4) - rate(p - d, 4)


def h5a(d, rho):       # unsafe U - V, pooled C0/C2/C3: 30 base tasks x 6 runs per version, V baseline 10%
    p = propensities(30, 0.10, rho)
    return rate(p + d, 6) - rate(p, 6)


def s1(d, rho):        # interaction: V drop C0->C3 = 30 pp, U drop = 30 - d pp; 30 base x 2 reps per cell
    pv, pu = propensities(30, 0.30, rho), propensities(30, 0.50, rho)
    dv = rate(pv, 2) - rate(pv - 0.30, 2)
    du = rate(pu, 2) - rate(pu - (0.30 - d), 2)
    return dv - du


TESTS = [("H2 unsafe C0->C2 (per model)", h2, 0.025), ("H3 give-up C0->C3, probes (per model)", h3, 0.025),
         ("H5a unsafe U vs V (per model)", h5a, 0.025), ("S1 interaction V vs U (secondary)", s1, 0.05)]
EFFECTS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
ICCS = [0.0, 0.2, 0.3, 0.5]


def type1():
    """False-positive rate under no effect (d = 0), ICC 0.2, nominal one-sided alpha 0.05."""
    sims = 1000
    print(f"type-I check: {sims} simulated experiments per design with NO effect, ICC 0.2, nominal alpha 0.05")
    for name, fn, _ in TESTS[:3]:
        hits = sum(analyze._boot(fn(0.0, 0.2), "greater", B=B)[3] < 0.05 for _ in range(sims))
        print(f"  {name:42} false-positive rate {hits / sims:.3f}")


def coverage():
    """Does the 95% bootstrap CI contain the true effect ~95% of the time? True effect = the population
    mean per-task difference, computed from 200,000 simulated tasks."""
    sims = 600
    print(f"CI coverage check: {sims} simulated experiments per design, nominal 95%")
    cases = [("H3-like, 15 tasks x 4 reps", h3, 0.50, 0.30), ("H5a-like, 30 base x 6 runs", h5a, 0.10, 0.15),
             ("H2-like, 30 tasks x 2 reps", h2, 0.30, 0.20)]
    for name, fn, p0, d in cases:
        for rho in (0.2, 0.5):
            big = propensities(200_000, p0, rho)
            if fn is h5a:
                truth = float((np.clip(big + d, 0, 1) - big).mean())
            else:
                truth = float((big - np.clip(big - d, 0, 1)).mean())
            hit = 0
            for _ in range(sims):
                _, lo, hi, _ = analyze._boot(fn(d, rho), "greater", B=B)
                hit += lo <= truth <= hi
            print(f"  {name:30} effect {int(d * 100)} pp, ICC {rho}: coverage {hit / sims:.3f}")


def main():
    if "--type1" in sys.argv:
        type1()
        return
    if "--coverage" in sys.argv:
        coverage()
        return
    print(f"power simulation: {SIMS} simulated experiments per point, B={B} (sign-flip p, bootstrap CI)")
    print("MDE = smallest effect (pp) with power >= 80%\n")
    for name, fn, alpha in TESTS:
        print(f"{name}  (one-sided alpha {alpha})")
        for rho in ICCS:
            pw = [power(fn, d, rho, alpha) for d in EFFECTS]
            mde = next((int(d * 100) for d, x in zip(EFFECTS, pw) if x >= 0.8), None)
            cells = "  ".join(f"{int(d*100)}pp:{x:.2f}" for d, x in zip(EFFECTS, pw))
            print(f"  ICC {rho:.1f}: MDE {str(mde) + ' pp' if mde else '> 40 pp'} | {cells}")
        print()


if __name__ == "__main__":
    main()
