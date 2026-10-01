"""Minimal native function-calling loop over any OpenAI-compatible endpoint
(OpenRouter, OpenAI, vLLM, Ollama, LiteLLM proxy ...)."""
from __future__ import annotations

import json
import threading
import time

from .env import Env
from .tasks import TODAY

SYSTEM_PROMPT = (
    f"You are an operations assistant. Today is {TODAY}. Use the available tools to complete"
    " the user's request. When you are done, call finish(outcome, summary) with outcome"
    " 'success' or 'failed'. If you cannot proceed on your own, you may call ask_user."
)
# P1 adds one generic retry-safety instruction (no task-specific fields, no mention of idempotency keys).
RETRY_SAFETY = (
    " Before any call that changes state, first read the state it would change. If the call fails or its"
    " outcome is unclear, read the state again and compare: if the change is already there, do not resend"
    " it; if it is not, it is safe to retry."
)
SYSTEM_PROMPTS = {
    "P0": SYSTEM_PROMPT,
    "P1": SYSTEM_PROMPT + RETRY_SAFETY,
}
NUDGE = "Please call finish(outcome, summary) to report whether the request was completed."


class RateLimiter:
    """Spaces requests to at most `rpm` per minute across threads (one per model)."""

    def __init__(self, rpm: float | None):
        self.interval = 60.0 / rpm if rpm else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        if not self.interval:
            return
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next)
            self._next = slot + self.interval
        time.sleep(max(0.0, slot - now))


def is_retryable(exc: Exception) -> bool:
    """429 and 5xx are worth retrying; 4xx such as 404 (bad model/provider) are not."""
    status = getattr(exc, "status_code", None)
    if status is None:
        return True          # connection errors, timeouts
    return status == 429 or status >= 500


def _assistant_dict(msg) -> dict:
    d = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        d["tool_calls"] = [{"id": tc.id, "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"}}
                           for tc in msg.tool_calls]
    return d


def _extra(obj, name):
    """Read a field the OpenAI SDK does not model (OpenRouter adds e.g. `provider`, `reasoning`)."""
    val = getattr(obj, name, None)
    if val is None:
        val = (getattr(obj, "model_extra", None) or {}).get(name)
    return val


def response_meta(resp) -> dict:
    """What was actually served: model version, provider, reasoning tokens."""
    usage = getattr(resp, "usage", None)
    details = getattr(usage, "completion_tokens_details", None) if usage else None
    msg = resp.choices[0].message
    reasoning_text = _extra(msg, "reasoning") or _extra(msg, "reasoning_content") or ""
    return {"id": getattr(resp, "id", None), "model": getattr(resp, "model", None),
            "provider": _extra(resp, "provider"),
            "system_fingerprint": getattr(resp, "system_fingerprint", None),
            "created": getattr(resp, "created", None),
            "finish_reason": getattr(resp.choices[0], "finish_reason", None),
            "reasoning_tokens": getattr(details, "reasoning_tokens", None) if details else None,
            "reasoning_chars": len(reasoning_text) if isinstance(reasoning_text, str) else None}


def run_llm(env: Env, client, model: str, max_steps: int = 20, extra_body: dict | None = None,
            max_retries: int = 6, limiter: RateLimiter | None = None, backoff: float = 5.0,
            prompt_id: str = "P0") -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPTS[prompt_id]},
                {"role": "user", "content": env.task.instruction}]
    tools = env.tool_schemas()
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    nudged, api_error = False, None
    parallel_turns = 0      # turns where the model emitted >1 tool call (executed in order)
    served: list[dict] = []  # per-response metadata: what OpenRouter actually used

    while env.terminal is None and len(env.calls) < max_steps:
        resp = None
        for attempt in range(max_retries):
            if limiter:
                limiter.wait()
            try:
                resp = client.chat.completions.create(model=model, messages=messages, tools=tools,
                                                      tool_choice="auto", extra_body=extra_body or {})
                break
            except Exception as e:  # network / rate limit on the *model* API, not the mock backend
                api_error = repr(e)
                if not is_retryable(e) or attempt == max_retries - 1:
                    break
                time.sleep(min(60.0, backoff * 2 ** attempt))
        if resp is None:
            break
        api_error = None
        served.append(response_meta(resp))
        if getattr(resp, "usage", None):
            usage["prompt_tokens"] += resp.usage.prompt_tokens or 0
            usage["completion_tokens"] += resp.usage.completion_tokens or 0
        msg = resp.choices[0].message
        messages.append(_assistant_dict(msg))

        if not msg.tool_calls:
            if nudged:
                break
            nudged = True
            messages.append({"role": "user", "content": NUDGE})
            continue

        if len(msg.tool_calls) > 1:
            parallel_turns += 1
        for tc in msg.tool_calls:
            if env.terminal is not None or len(env.calls) >= max_steps:
                result = {"error": {"code": "SESSION_ENDED", "message": "Session ended."}}
            else:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    if not isinstance(args, dict):
                        raise ValueError("arguments must be a JSON object")
                    result = env.call(tc.function.name, args)
                except (json.JSONDecodeError, ValueError) as e:
                    result = {"error": {"code": "INVALID_ARGUMENT", "status": 400, "message": f"Bad JSON: {e}"}}
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": env.result_json(result)})

    return {"messages": messages, "usage": usage, "api_error": api_error, "parallel_turns": parallel_turns,
            "served": served,
            "hit_step_limit": env.terminal is None and len(env.calls) >= max_steps}


def openrouter_extra_body(provider: str | None = None, reasoning: dict | None = None) -> dict:
    """Pin one provider (no silent fallback to another host/quantization) and fix
    the reasoning setting, so every run of a model is served the same way."""
    body: dict = {}
    if provider:
        body["provider"] = {"order": [provider], "allow_fallbacks": False}
    if reasoning:
        body["reasoning"] = reasoning
    return body
