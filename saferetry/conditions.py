"""API conditions: what the agent sees changes, backend logic never does.

C0  baseline        vague messages, no idempotency key
C1  idempotency     C0 messages + optional idempotency_key on write tools
C2  outcome-unknown messages state that the outcome is unknown / current state
C3  next-step       C2 + one sentence suggesting what to do
"""

CONDITIONS = ["C0", "C1", "C2", "C3"]


def has_idempotency(condition: str) -> bool:
    return condition == "C1"


def _tier(condition: str) -> str:
    # C1 reuses the baseline wording so that only the key parameter differs.
    return "C0" if condition in ("C0", "C1") else condition


def timeout_message(condition: str, read_tool: str) -> str:
    t = _tier(condition)
    base = "Request timed out."
    unknown = base + " The operation may or may not have been applied."
    if t == "C0":
        return base
    if t == "C2":
        return unknown
    return unknown + f" Check {read_tool} before retrying to avoid a duplicate."


def async_accepted_message(condition: str) -> str:
    t = _tier(condition)
    if t == "C0":
        return "Accepted."
    unknown = "Accepted. The job has not completed yet."
    if t == "C2":
        return unknown
    return unknown + (" Poll get_job_status until it returns succeeded before"
                      " performing dependent actions. Do not resubmit while pending.")


def async_failed_message(condition: str) -> str:
    t = _tier(condition)
    if t == "C0":
        return "Job failed."
    unknown = "Job failed. No data was processed."
    if t == "C2":
        return unknown
    return unknown + " It is safe to resubmit."


def batch_message(condition: str) -> str:
    t = _tier(condition)
    if t == "C0":
        return "Batch request failed."
    unknown = "Some items failed. Items not listed as failed were applied."
    if t == "C2":
        return unknown
    return unknown + (" Retry only the failed items; resending applied items"
                      " will apply them again.")


IDEMPOTENCY_DESCRIPTION = (
    "Optional. Requests with the same idempotency_key within 24h are executed at"
    " most once; later requests return the original result."
)


# ---------------- core cells (async / batch) and probe ----------------
# Rule: C0 must not mention uncertainty; C2 states it; C3 adds one action.

def job_lost_message(condition: str, lookup_tool: str) -> str:
    t = _tier(condition)
    if t == "C0":
        return "Job not found."
    unknown = "Job record lost after a worker restart; the job may have completed."
    if t == "C2":
        return unknown
    return unknown + f" Check {lookup_tool} before resubmitting."


def batch_unknown_message(condition: str) -> str:
    t = _tier(condition)
    if t == "C0":
        return "Batch completed with errors."
    unknown = "Items marked timeout may or may not have been applied."
    if t == "C2":
        return unknown
    return unknown + " Check those items before resending."


# probe: definitive failure (nothing was processed; retrying is safe)
def unavailable_message(condition: str) -> str:
    t = _tier(condition)
    if t == "C0":
        return "Service unavailable."
    known = "Service unavailable. The request was not processed."
    if t == "C2":
        return known
    return known + " It is safe to retry."


def batch_all_failed_message(condition: str) -> str:
    t = _tier(condition)
    if t == "C0":
        return "Batch request failed."
    known = "All items failed. Nothing was applied."
    if t == "C2":
        return known
    return known + " It is safe to resend the batch."
