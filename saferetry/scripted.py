"""Three scripted reference policies (no LLM). They double as a test of the
oracle: their labels are known in advance.

always_retry  treat any non-final response as failure and resend
never_retry   one attempt; on any failure report failed
verify_first  check the real state before deciding (upper bound)
"""
from __future__ import annotations

from .env import Env


def _finish(env: Env, ok: bool) -> None:
    env.call("finish", {"outcome": "success" if ok else "failed", "summary": "scripted"})


def _poll(env: Env, job_id: str, limit: int = 10) -> dict:
    res = {}
    for _ in range(limit):
        res = env.call("get_job_status", {"job_id": job_id})
        if res.get("status") in ("succeeded", "failed") or "error" in res:
            break
    return res


def _is_lost_job(t) -> bool:
    return type(t).__name__ == "AsyncLostTask"


def _is_unknown_batch(t) -> bool:
    return t.category == "C" and getattr(t, "mode", "fail") == "unknown"


def _not_ok(res: dict) -> list[int]:
    return [r["index"] for r in res.get("results", []) if r.get("status") != "ok"]


# per-task item checks for the batch cells (verify-first uses the real lookup tools)
def _item_applied(env: Env, t, item: dict) -> bool:
    tid = t.id[-1]
    if tid == "1":
        return bool(env.call("list_sent_messages", {"to": item["to"]})["records"])
    if tid == "2":
        return env.call("get_inventory", {"sku": item["sku"]})["on_hand"] >= t.extra_state["inventory"][item["sku"]] + item["delta"]
    if tid == "3":
        return env.call("get_points", {"user_id": item["user_id"]})["balance"] >= t.extra_state["points"][item["user_id"]] + item["points"]
    if tid == "4":
        recs = env.call("list_events", {"date": item["start"][:10]})["records"]
        return any(item["attendees"][0] in r.get("attendees", []) for r in recs)
    return bool(env.call("get_ticket_by_feedback", {"feedback_id": item["feedback_id"]})["tickets"])


def _failed_indices(res: dict) -> list[int]:
    return [r["index"] for r in res.get("results", []) if r.get("status") != "ok"]


def always_retry(env: Env) -> None:
    t = env.task
    if _is_lost_job(t):
        sub = env.call(t.submit.name, dict(t.canonical_submit))
        res = _poll(env, sub["job_id"])
        if res.get("status") != "succeeded":
            sub = env.call(t.submit.name, dict(t.canonical_submit))
            res = _poll(env, sub["job_id"])
        env.call(t.dependent.name, t.canonical_dependent(res.get("result", {})))
        _finish(env, True)
        return
    if _is_unknown_batch(t):
        items = list(t.canonical_items)
        res = env.call(t.batch_tool, {t.list_param: items})
        if "error" in res:
            env.call(t.batch_tool, {t.list_param: [items[i] for i in _not_ok(res)]})
        _finish(env, True)
        return
    if t.category == "A":
        res = env.call(t.write.name, dict(t.canonical_args))
        if "error" in res:
            env.call(t.write.name, dict(t.canonical_args))
        _finish(env, True)
    elif t.category == "B":
        first = env.call(t.submit.name, dict(t.canonical_submit))
        second = env.call(t.submit.name, dict(t.canonical_submit))   # 202 != done -> resend
        res = _poll(env, second["job_id"])
        env.call(t.dependent.name, t.canonical_dependent(res.get("result", {})))
        _finish(env, True)
    else:
        res = env.call(t.batch_tool, {t.list_param: list(t.canonical_items)})
        if "error" in res:
            env.call(t.batch_tool, {t.list_param: list(t.canonical_items)})
        _finish(env, True)


def never_retry(env: Env) -> None:
    t = env.task
    if _is_lost_job(t):
        sub = env.call(t.submit.name, dict(t.canonical_submit))
        res = _poll(env, sub["job_id"])
        if res.get("status") == "succeeded":
            env.call(t.dependent.name, t.canonical_dependent(res.get("result", {})))
        _finish(env, res.get("status") == "succeeded")
        return
    if t.category == "A":
        res = env.call(t.write.name, dict(t.canonical_args))
        _finish(env, "error" not in res)
    elif t.category == "B":
        sub = env.call(t.submit.name, dict(t.canonical_submit))
        res = _poll(env, sub["job_id"])
        if res.get("status") == "succeeded":
            env.call(t.dependent.name, t.canonical_dependent(res.get("result", {})))
            _finish(env, True)
        else:
            _finish(env, False)
    else:
        res = env.call(t.batch_tool, {t.list_param: list(t.canonical_items)})
        _finish(env, "error" not in res)


def _u_value(env: Env, t, item=None) -> float:
    tool = t.lookup.name if _is_lost_job(t) else t.read.name
    return env.call(tool, t.u_args(item))[t.u_field]


def _verify_first_unverifiable(env: Env) -> None:
    """U version: the lookup only shows an aggregate, so read it BEFORE acting and compare after."""
    t = env.task
    import json
    if _is_lost_job(t):
        before = _u_value(env, t)
        sub = env.call(t.submit.name, dict(t.canonical_submit))
        res = _poll(env, sub["job_id"])
        if res.get("status") != "succeeded":
            if _u_value(env, t) != before:          # the effect happened
                env.call(t.dependent.name, t.canonical_dependent({}))
                _finish(env, True)
                return
            sub = env.call(t.submit.name, dict(t.canonical_submit))
            res = _poll(env, sub["job_id"])
        env.call(t.dependent.name, t.canonical_dependent(res.get("result", {})))
        _finish(env, True)
        return
    if t.category == "A":
        before = _u_value(env, t)
        res = env.call(t.write.name, dict(t.canonical_args))
        if "error" in res and _u_value(env, t) == before:
            env.call(t.write.name, dict(t.canonical_args))
        _finish(env, True)
        return
    # unknown-outcome batch: one baseline per distinct lookup
    items = list(t.canonical_items)
    gkey = lambda it: json.dumps(t.u_args(it), sort_keys=True)
    before = {}
    for it in items:
        if gkey(it) not in before:
            before[gkey(it)] = _u_value(env, t, it)
    res = env.call(t.batch_tool, {t.list_param: items})
    if "error" in res:
        bad = set(_not_ok(res))
        resend = []
        for g in dict.fromkeys(gkey(items[i]) for i in bad):
            ok_change = sum(t.u_step(it) for i, it in enumerate(items) if i not in bad and gkey(it) == g)
            now = _u_value(env, t, next(items[i] for i in bad if gkey(items[i]) == g))
            if abs(now - (before[g] + ok_change)) < 1e-9:    # timeout items were not applied
                resend += [items[i] for i in sorted(bad) if gkey(items[i]) == g]
        if resend:
            env.call(t.batch_tool, {t.list_param: resend})
    _finish(env, True)


def verify_first(env: Env) -> None:
    t = env.task
    if getattr(t, "verif", "V") == "U":
        _verify_first_unverifiable(env)
        return
    if _is_lost_job(t):
        sub = env.call(t.submit.name, dict(t.canonical_submit))
        res = _poll(env, sub["job_id"])
        if res.get("status") != "succeeded":
            seen = env.call(t.lookup.name, dict(t.canonical_lookup))
            if seen.get(t.effect_name):
                env.call(t.dependent.name, t.canonical_dependent(seen[t.effect_name][0]))
                _finish(env, True)
                return
            sub = env.call(t.submit.name, dict(t.canonical_submit))
            res = _poll(env, sub["job_id"])
        env.call(t.dependent.name, t.canonical_dependent(res.get("result", {})))
        _finish(env, True)
        return
    if _is_unknown_batch(t):
        items = list(t.canonical_items)
        res = env.call(t.batch_tool, {t.list_param: items})
        applied = ((lambda it: t.v_check(it, env.call(t.read.name, t.v_args(it)))) if t.v_args
                   else (lambda it: _item_applied(env, t, it)))
        missing = [items[i] for i in _not_ok(res) if not applied(items[i])] if "error" in res else []
        if missing:
            env.call(t.batch_tool, {t.list_param: missing})
        _finish(env, True)
        return
    if t.category == "A":
        res = env.call(t.write.name, dict(t.canonical_args))
        if "error" in res:
            seen = env.call(t.read.name, dict(t.canonical_read_args))
            if not t.found_in(seen):
                env.call(t.write.name, dict(t.canonical_args))
        _finish(env, True)
    elif t.category == "B":
        for _ in range(2):
            sub = env.call(t.submit.name, dict(t.canonical_submit))
            res = _poll(env, sub["job_id"])
            if res.get("status") == "succeeded":
                env.call(t.dependent.name, t.canonical_dependent(res.get("result", {})))
                _finish(env, True)
                return
        _finish(env, False)
    else:
        items = list(t.canonical_items)
        res = env.call(t.batch_tool, {t.list_param: items})
        if "error" in res:
            retry = [items[i] for i in _failed_indices(res)]
            env.call(t.batch_tool, {t.list_param: retry})
        _finish(env, True)


POLICIES = {"always_retry": always_retry, "never_retry": never_retry, "verify_first": verify_first}
