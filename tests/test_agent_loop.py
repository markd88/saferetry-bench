"""Plumbing test for the LLM loop with a fake OpenAI-compatible client."""
import json
from types import SimpleNamespace as NS

from saferetry.agent import run_llm
from saferetry.env import Env
from saferetry.oracle import label
from saferetry.tasks import FULL_BY_ID


class FakeClient:
    """Replays a fixed list of assistant turns; each turn is a list of (tool, args)."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.seen = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, model, messages, tools, **kw):
        self.seen.append(messages[-1])
        turn = self.turns.pop(0) if self.turns else []
        calls = [NS(id=f"c{i}", function=NS(name=n, arguments=json.dumps(a))) for i, (n, a) in enumerate(turn)]
        msg = NS(content="" if calls else "done", tool_calls=calls or None)
        return NS(choices=[NS(message=msg)], usage=NS(prompt_tokens=10, completion_tokens=5))


def test_blind_retry_is_unsafe():
    t = FULL_BY_ID["A1-V"]
    client = FakeClient([[("create_order", t.canonical_args)], [("create_order", t.canonical_args)],
                         [("finish", {"outcome": "success", "summary": "ok"})]])
    env = Env(t, "fault", "C0")
    out = run_llm(env, client, "fake")
    assert label(env)["label"] == "UNSAFE"
    assert out["usage"]["prompt_tokens"] == 30
    # the tool result the model saw carries the timeout error
    assert "GATEWAY_TIMEOUT" in client.seen[1]["content"]


def test_nudge_then_stop():
    t = FULL_BY_ID["A1-V"]
    client = FakeClient([[], []])
    env = Env(t, "fault", "C0")
    run_llm(env, client, "fake")
    assert env.terminal is None and label(env)["label"] == "INCOMPLETE"


def test_bad_json_is_reported_not_crashed():
    t = FULL_BY_ID["C2-V"]
    client = FakeClient([])
    client.turns = [[]]
    env = Env(t, "control", "C2")

    def create(model, messages, tools, **kw):
        if len(messages) == 2:
            tc = NS(id="x", function=NS(name="adjust_inventory_batch", arguments="{not json"))
            return NS(choices=[NS(message=NS(content="", tool_calls=[tc]))], usage=None)
        tc = NS(id="y", function=NS(name="ask_user", arguments='{"question": "?"}'))
        return NS(choices=[NS(message=NS(content="", tool_calls=[tc]))], usage=None)

    client.chat = NS(completions=NS(create=create))
    out = run_llm(env, client, "fake")
    assert "Bad JSON" in out["messages"][3]["content"]
    assert label(env)["label"] == "OVER_CAUTIOUS"


def test_prompt_p1_is_sent_and_extends_p0():
    from saferetry.agent import SYSTEM_PROMPTS
    assert SYSTEM_PROMPTS["P1"].startswith(SYSTEM_PROMPTS["P0"])
    assert "idempotency" not in SYSTEM_PROMPTS["P1"].lower()
    t = FULL_BY_ID["A1-U"]
    sent = []
    client = FakeClient([[("finish", {"outcome": "success", "summary": "ok"})]])
    orig = client.chat.completions.create
    client.chat.completions.create = lambda model, messages, tools, **kw: (sent.append(messages[0]), orig(model, messages, tools, **kw))[1]
    run_llm(Env(t, "fault", "C0"), client, "fake", prompt_id="P1")
    assert sent[0]["role"] == "system" and sent[0]["content"] == SYSTEM_PROMPTS["P1"]
