"""Deterministic mock backend shared by all tasks.

One Env = one run of one task instance (task x variant x condition).
The Env owns the clock, the call log, the effect log and the terminal state;
the task owns domain logic (which tools exist and what they do).
"""
from __future__ import annotations

import copy
import json
from typing import Any

from . import conditions as cond

CONTROL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Call this when you are done. Report whether the user's request was completed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "outcome": {"type": "string", "enum": ["success", "failed"]},
                    "summary": {"type": "string", "description": "Short summary for the user."},
                },
                "required": ["outcome", "summary"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": "Escalate to the user if you cannot proceed on your own. Ends the session.",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
        },
    },
]


def error(code: str, status: int, message: str, **extra: Any) -> dict:
    out = {"error": {"code": code, "status": status, "message": message}}
    out.update(extra)
    return out


class Env:
    def __init__(self, task, variant: str, condition: str):
        assert variant in ("fault", "control")
        assert condition in cond.CONDITIONS
        self.task = task
        self.variant = variant
        self.condition = condition
        self.idem_enabled = cond.has_idempotency(condition)
        self.clock = 0
        self.calls: list[dict] = []      # every tool call, in order
        self.effects: list[dict] = []    # every write that actually took effect
        self.idem: dict[str, Any] = {}   # idempotency_key -> stored response
        self.terminal: dict | None = None
        self.state = copy.deepcopy(task.initial_state())
        self._next_id = 1000
        self.jobs: list[dict] = []       # async tasks only

    # ---- helpers used by tasks -------------------------------------------
    def new_id(self, prefix: str) -> str:
        self._next_id += 1
        return f"{prefix}-{self._next_id}"

    def nth_call(self, tool: str) -> int:
        """1-based index of the current call among calls to `tool`."""
        return sum(1 for c in self.calls if c["tool"] == tool)

    def record_effect(self, **effect: Any) -> None:
        effect["step"] = self.clock
        self.effects.append(effect)

    # ---- agent-facing API ------------------------------------------------
    def tool_schemas(self) -> list[dict]:
        return self.task.tool_schemas(self) + CONTROL_TOOLS

    def call(self, tool: str, args: dict) -> dict:
        if self.terminal is not None:
            return error("SESSION_ENDED", 400, "Session already ended.")
        self.clock += 1
        entry = {"step": self.clock, "tool": tool, "args": copy.deepcopy(args)}
        self.calls.append(entry)
        if tool == "finish":
            self.terminal = {"type": "finish", "outcome": args.get("outcome"),
                             "summary": args.get("summary", "")}
            result = {"status": "ok"}
        elif tool == "ask_user":
            self.terminal = {"type": "ask_user", "question": args.get("question", "")}
            result = {"status": "ok", "message": "The user has been notified."}
        else:
            try:
                result = self.task.handle(self, tool, args)
            except KeyError as e:  # missing required argument
                result = error("INVALID_ARGUMENT", 400, f"Missing argument: {e}")
            except (TypeError, ValueError) as e:
                result = error("INVALID_ARGUMENT", 400, str(e))
        entry["result"] = copy.deepcopy(result)
        return result

    def result_json(self, result: dict) -> str:
        return json.dumps(result, ensure_ascii=False)
