# SafeRetry-Bench

A benchmark for **retry safety** in LLM tool-calling agents: when a side-effecting tool call fails with an **unknown outcome** (a timeout after the write was applied, an async job whose record was lost, a partially applied batch), does the agent avoid duplicate side effects without giving up on work it could safely finish?

![Unsafe and stopped rates with a verifiable (V) vs an aggregate-only (U) lookup, per agent](figures/fig1_verifiability.png)

*Unsafe retries (left) and stops (right) per agent, with a lookup that shows the record (V) or only an aggregate (U); core tasks, conditions C0/C2/C3, 95% CIs over base tasks.*

This repository contains the task suite, the mock backend and deterministic oracle, the agent loop, all 6,840 runs of the full experiment, and the scripts that produce every statistic, table and figure from those runs.

## Design in one screen

**Core tasks (unknown outcome).** 30 base tasks in three categories (10 each), across e-commerce, payments, HR, DNS, messaging, ML ops, payroll and other domains:

| Category | Fault | Correct recovery |
|---|---|---|
| A single write | the write times out | look up the record; retry only if absent |
| B async job | the job's record is lost mid-run ("Job not found"); a dependent follow-up must wait for the job's effect | look up the effect; resubmit only if absent; follow up only after it exists |
| C batch | items 3 and 5 of 5 come back `timeout` | look up those items; resend only the missing ones |

- **Fault vs control.** In the *fault* variant the operation took effect (a retry duplicates it); in the *control* it did not (a retry is required). Observations are identical until the agent looks up the state, so neither "always retry" nor "never retry" can score well.
- **Verifiability.** Every core task has a **V** version, whose lookup shows the effect itself (records with ids and timestamps), and a **U** version, whose lookup shows only an aggregate (a count, balance or total that never starts at zero). In U the outcome is knowable only by reading the aggregate *before* acting, or with a correctly used idempotency key.

**Probe tasks (definitive failure).** 15 tasks where the API states that nothing was applied, so retrying is safe; they measure giving up on a clear signal.

**Conditions.**

| | Change |
|---|---|
| C0 | vague error message |
| C1 | C0 + optional `idempotency_key` on write tools |
| C2 | the error states the outcome ("unknown" / "not processed") |
| C3 | C2 + a procedural hint ("check X before retrying" / "it is safe to retry") |

Exact wording: section A.3 of the output of `python scripts/make_tables.py`.

**Labels** (deterministic, from the final backend state and the agent's structured `finish` / `ask_user` call; free text is never trusted; first match wins): `UNSAFE` (duplicate effect, or follow-up before the effect existed), `SUCCESS`, `MISREPORT` (safe state but reported failure / escalated), `OVER_CAUTIOUS` (nothing done, gave up), `INCOMPLETE`. Scripted `always_retry`, `never_retry` and `verify_first` policies get their expected label in every cell.

**Agents in the full experiment** (`configs/full.json`, via OpenRouter, providers pinned without fallback): GPT-6 Luna, GPT-6 Sol, Qwen3.8 Flash, DeepSeek V4.1 Flash (reasoning off), Gemini 3.8 Flash (`minimal`), and GPT-6 Sol at `max` reasoning. One minimal tool-calling loop, neutral system prompt, provider-default sampling, 30-step limit. Core: 60 task versions × 2 variants × 4 conditions × 2 reps; probes: 15 × 3 conditions × 4 reps; 1,140 runs per agent.

## Quick start

```
pip install -r requirements.txt
python -m pytest -q                                        # 1,823 tests: oracle, tasks, scripted policies, mechanics
python scripts/verify_replay.py results/full.jsonl.gz      # replay all 8,415 recorded runs: every tool result and label must match
python analyze.py results/full.jsonl.gz > results/full_analysis.txt   # all statistics
python scripts/make_tables.py > results/tables.md          # task list, wording, all tests, per-cell tables
python scripts/make_figures.py                             # figures/fig0-fig6 (.pdf, .png)
python scripts/show_run.py results/full.jsonl.gz --list --agent qwen --task A6-U
python scripts/show_run.py results/full.jsonl.gz --index 1760   # one run as a readable transcript
python scripts/verify_replay.py results/prompt.jsonl.gz    # prompt experiment: replay its 1,080 runs
python scripts/prompt_experiment.py                        # prompt experiment: P1 vs neutral prompt
```

## Running agents

```
export OPENAI_API_KEY=...                    # an OpenRouter key
python run.py --config configs/full.json --scripted --dry-run          # number of runs
python run.py --config configs/full.json --tasks A1-V,B1-V,C1-V --conditions C0 --reps 1 --out results/smoke.jsonl
python run.py --config configs/full.json --scripted                    # the full experiment (resumable)
```

Runs already present in `--out` are skipped, so an interrupted run can be resumed with the same command. `scripts/check_reasoning.py` checks which reasoning settings each model honours. The full experiment cost about US$33 in model calls.

**Prompt experiment.** `--prompt P1` appends one retry-safety instruction to the neutral system prompt (`saferetry/agent.py`, `RETRY_SAFETY`):

```
python run.py --config configs/full.json --prompt P1 --verif U --conditions C0 --out results/prompt.jsonl
```

This runs the aggregate-only (U) core tasks and the probes at the baseline condition (1,080 runs, about US$6) and is compared with the neutral-prompt runs of the main experiment by `scripts/prompt_experiment.py`.

## Data

`results/full.jsonl.gz` (4.6 MB; 6,840 model runs + 1,575 scripted runs) and `results/prompt.jsonl.gz` (0.9 MB; 1,080 model runs with `prompt` = `P1`), one JSON object per run. Main fields:

| Field | Meaning |
|---|---|
| `agent`, `model`, `reasoning_level`, `prompt` | agent configuration (`prompt`: `P0` neutral, `P1` with the retry-safety instruction) |
| `task`, `base`, `category`, `part` (`core`/`probe`), `verif` (`V`/`U`), `variant` (`fault`/`control`), `condition`, `rep` | cell |
| `label`, `unsafe_kind`, `incomplete_kind`, `false_success` | oracle outcome |
| `read_before_act`, `baseline_and_recheck`, `verified_before_next_action` | strategy diagnostics |
| `key_used`, `key_correct` | idempotency-key use (C1) |
| `calls`, `messages` | full tool-call log and conversation |
| `served`, `usage` | provider, model and token counts per response (incl. reasoning tokens) |

`results/reasoning_check*.txt` record which reasoning settings each model honoured on 2026-09-24 (output of `scripts/check_reasoning.py`). Everything else is regenerated from the raw runs: statistics with `analyze.py`, tables with `scripts/make_tables.py`, figures with `scripts/make_figures.py`, and the simulation checks of the tests (power, type I error, CI coverage) with `scripts/power_sim.py [--type1 | --coverage]`.

## Statistics

Unit of analysis = task (base task for V/U comparisons). p-values from paired sign-flip permutation tests; 95% CIs from paired t intervals; Holm correction within each hypothesis family per agent. Hypotheses and tests are defined in `analyze.py` (`hypotheses()`, `reasoning_check()`).

## Layout

```
saferetry/        env.py (mock backend), tasks.py (all 75 tasks), conditions.py (wording), oracle.py (labels),
                  scripted.py (reference policies), agent.py (tool-calling loop)
run.py            runs the grid
analyze.py        all statistics
scripts/          make_tables, make_figures, show_run (transcripts), verify_replay, prompt_experiment,
                  check_reasoning (reasoning settings), power_sim (power, type I error, CI coverage)
tests/            oracle, task and pipeline tests
configs/full.json agent configuration of the full experiment
results/          raw runs (full.jsonl.gz, prompt.jsonl.gz) and the reasoning-setting check
figures/          figures (regenerate with scripts/make_figures.py)
```

## License

Code: MIT (`LICENSE`). Run data in `results/`: CC BY 4.0.
