"""Config plumbing, parallel-call logging and the analysis statistics."""
import json
import random
from types import SimpleNamespace as NS

import pandas as pd

import analyze
from saferetry.agent import openrouter_extra_body, run_llm
from saferetry.env import Env
from saferetry.oracle import label
from saferetry.scripted import POLICIES
from saferetry.tasks import CORE, FULL_BY_ID


def test_extra_body_pins_provider_and_reasoning():
    body = openrouter_extra_body("Alibaba", {"enabled": False})
    assert body == {"provider": {"order": ["Alibaba"], "allow_fallbacks": False}, "reasoning": {"enabled": False}}
    assert openrouter_extra_body(None, None) == {}


def test_extra_body_reaches_client_and_parallel_turns_counted():
    t = FULL_BY_ID["A1-V"]
    seen = {}

    def create(model, messages, tools, extra_body=None, **kw):
        seen["extra_body"] = extra_body
        if len(messages) == 2:   # one turn with two calls at once
            calls = [NS(id="a", function=NS(name="list_orders", arguments='{"customer_id": "C102"}')),
                     NS(id="b", function=NS(name="create_order", arguments=json.dumps(t.canonical_args)))]
        else:
            calls = [NS(id="c", function=NS(name="finish", arguments='{"outcome": "failed", "summary": "x"}'))]
        return NS(choices=[NS(message=NS(content="", tool_calls=calls))], usage=None)

    client = NS(chat=NS(completions=NS(create=create)))
    env = Env(t, "fault", "C0")
    out = run_llm(env, client, "m", extra_body=openrouter_extra_body("OpenAI", {"effort": "low"}))
    assert seen["extra_body"]["provider"]["order"] == ["OpenAI"]
    assert out["parallel_turns"] == 1
    assert [c["tool"] for c in env.calls] == ["list_orders", "create_order", "finish"]  # executed in order


def test_holm():
    assert analyze.holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def _fake_df(seed=0):
    """Two fake 'models' that pick a scripted policy at random per run."""
    rng = random.Random(seed)
    rows = []
    for agent, p_unsafe in [("fake/careless", 0.7), ("fake/careful", 0.1)]:
        for task in [t for t in CORE if t.verif == "V"]:
            for variant in ("fault", "control"):
                for cond in ("C0", "C2"):
                    for rep in range(3):
                        p = p_unsafe if cond == "C0" else p_unsafe / 3
                        pol = "always_retry" if rng.random() < p else "verify_first"
                        env = Env(task, variant, cond)
                        POLICIES[pol](env)
                        rows.append({"agent": agent, "task": task.id, "category": task.category,
                                     "variant": variant, "condition": cond, "rep": rep, **label(env),
                                     "usage": {"prompt_tokens": 1, "completion_tokens": 1}, "parallel_turns": 0})
    df = pd.DataFrame(rows)
    df["tokens"] = 2
    return df


def test_analysis_runs_and_detects_effect(capsys):
    df = _fake_df()
    analyze.comparisons(df)
    out = capsys.readouterr().out
    assert "p_holm" in out
    careless = df[(df.agent == "fake/careless") & (df.variant == "fault")]
    diff, lo, hi, p = analyze.paired_task_bootstrap(careless[careless.condition == "C0"],
                                                    careless[careless.condition == "C2"])
    assert diff < 0 and hi < 0   # C2 clearly lowers the unsafe rate in the fake data


def test_rate_limiter_spaces_requests():
    import time
    from saferetry.agent import RateLimiter
    lim = RateLimiter(rpm=600)          # one slot every 0.1 s
    t0 = time.monotonic()
    for _ in range(4):
        lim.wait()
    assert time.monotonic() - t0 >= 0.29


def test_retry_classification():
    from saferetry.agent import is_retryable
    assert is_retryable(NS(status_code=429)) and is_retryable(NS(status_code=503))
    assert not is_retryable(NS(status_code=404)) and not is_retryable(NS(status_code=400))


def test_non_retryable_error_stops_immediately():
    t = FULL_BY_ID["A1-V"]
    calls = {"n": 0}

    class NotFound(Exception):
        status_code = 404

    def create(**kw):
        calls["n"] += 1
        raise NotFound("no endpoint")

    client = NS(chat=NS(completions=NS(create=create)))
    out = run_llm(Env(t, "fault", "C0"), client, "m", backoff=0)
    assert calls["n"] == 1 and "no endpoint" in out["api_error"]


def test_rate_limited_run_is_retried_then_succeeds():
    t = FULL_BY_ID["A1-V"]
    state = {"n": 0}

    class RateLimited(Exception):
        status_code = 429

    def create(**kw):
        state["n"] += 1
        if state["n"] == 1:
            raise RateLimited("slow down")
        tc = NS(id="f", function=NS(name="finish", arguments='{"outcome": "failed", "summary": "x"}'))
        return NS(choices=[NS(message=NS(content="", tool_calls=[tc]))], usage=None)

    client = NS(chat=NS(completions=NS(create=create)))
    out = run_llm(Env(t, "fault", "C0"), client, "m", backoff=0)
    assert out["api_error"] is None and state["n"] == 2


def test_response_meta_captures_served_provider_and_reasoning():
    from saferetry.agent import response_meta
    msg = NS(content="", tool_calls=None, model_extra={"reasoning": "thinking..."})
    usage = NS(prompt_tokens=1, completion_tokens=2, completion_tokens_details=NS(reasoning_tokens=7))
    resp = NS(id="gen-1", model="qwen/qwen3.8-flash-20260826", model_extra={"provider": "Alibaba"},
              system_fingerprint="fp_abc", created=1790000000,
              choices=[NS(message=msg, finish_reason="tool_calls")], usage=usage)
    m = response_meta(resp)
    assert m == {"id": "gen-1", "model": "qwen/qwen3.8-flash-20260826", "provider": "Alibaba",
                 "system_fingerprint": "fp_abc", "created": 1790000000, "finish_reason": "tool_calls",
                 "reasoning_tokens": 7, "reasoning_chars": 11}


