"""Opt-in feasibility of finishing an LLM-mediated evidence acquisition.

These are count budgets and an explicit wake-up contract, not expected value,
source publication probabilities, or a promise that an admitted query succeeds.
"""
from __future__ import annotations

from copy import deepcopy

from .inquiry_contract import utc


VERSION = "llm-continuation-budget-v1"
MINIMUMS = {
    "read_url": {"source_requests_including_current": 1, "llm_calls_after_current": 1},
    "search_web": {"source_requests_including_current": 2, "llm_calls_after_current": 2},
}


def remaining(run):
    budget = run["budget"]
    return {"source_requests": budget["max_requests"]-budget["http_attempts"]-budget["reserved_requests"],
            "llm_calls": budget["max_llm_calls"]-budget["llm_calls"]}


def discovery_needs_page(run):
    if not run["receipts"]:
        return False
    latest = run["receipts"][-1]
    doc = next((d for d in run["documents"] if d["document_id"] == latest["document_id"]), None)
    return bool(latest["tool"] == "search_web" and doc and doc["status"] == "OK" and doc.get("links"))


def duplicate_query_blocked(run, execution_key):
    previous = [r for r in run["receipts"] if r["execution_key"] == execution_key]
    if not previous:
        return False
    if not run["configuration"].get("continuation_policy"):
        return True
    wake = next((e for e in reversed(run["events"]) if e["type"] == "inquiry_wait_resumed"), None)
    if wake is None:
        return True
    # One explicit due wake-up permits a paid refresh of this URL once. It is
    # not evidence that a publication occurred or that the response will change.
    old_requests = {r["request_id"] for r in wake["snapshot"]["receipts"]}
    return any(r["request_id"] not in old_requests for r in previous)


class ContinuationPolicy:
    version = VERSION

    def configuration(self):
        return {"version": VERSION, "scope": "llm_mediated_source_claim_delivery",
                "minimum_paths": deepcopy(MINIMUMS), "wait_resume": "explicit_due_call_no_budget_refill",
                "source_publication_schedule": "not_configured", "numerical_value": None}

    def public(self, run):
        counts = remaining(run)
        following = counts["llm_calls"]-1
        paths = {name: {**need, "admissible": counts["source_requests"] >= need["source_requests_including_current"]
                  and following >= need["llm_calls_after_current"]} for name, need in MINIMUMS.items()}
        return {"version": VERSION, "budget_reference": "remaining_llm_calls_includes_this_proposal_call",
                "llm_calls_after_this_proposal": max(0, following), "paths": paths,
                "awaiting_discovered_page_selection": discovery_needs_page(run),
                "verified_source_release_schedule": "unknown_not_inferred_from_match_time",
                "revisit_at_meaning": "agent_proposed_reassessment_time_not_verified_publication_time",
                "deadline_duration_feasibility": "not_estimated_count_budgets_only",
                "instructions": "Choose only actions with capacity to process their returns. A factual search needs a later page read and another interpretation call. Use stop or defer on the final model call; preserve newly extracted evidence. With discovered links, avoid spending the page-read continuation budget on reflect. An admissible query may still be low value. Unknown publication timing is not evidence that information is absent. Waiting requires a future reassessment time and an explicit due resume; it creates no automatic schedule.",
                "value_status": "NOT_A_VALUE_ESTIMATE"}

    def apply(self, run, response, decision, *, now):
        counts = remaining(run)  # The current proposal has already been charged.
        kind, reason, need = decision["kind"], None, None
        if utc(now) >= utc(run["task"]["deadline"]):
            reason = "TASK_DEADLINE_REACHED_DURING_PROCESSING"
        elif kind == "query":
            candidate = next(c for c in response["candidates"] if c["candidate_id"] == decision["candidate_id"])
            need = MINIMUMS[candidate["tool"]]
            if counts["source_requests"] < need["source_requests_including_current"]:
                reason = "SOURCE_CONTINUATION_BUDGET_INSUFFICIENT"
            elif counts["llm_calls"] < need["llm_calls_after_current"]:
                reason = "RETURN_PROCESSING_BUDGET_INSUFFICIENT"
        elif kind in {"reflect", "wait"}:
            minimum = 2 if discovery_needs_page(run) else 1
            need = {"llm_calls_after_current": minimum}
            if counts["llm_calls"] < minimum:
                reason = "CONTINUATION_REASONING_BUDGET_INSUFFICIENT"
            elif kind == "wait" and not utc(now) < utc(decision["revisit_at"]) < utc(run["task"]["deadline"]):
                reason = "WAIT_TIME_ELAPSED_DURING_REASONING"
        check = {"version": VERSION, "admitted": reason is None, "requested_kind": kind,
                 "available_after_proposal": counts, "minimum_required": deepcopy(need),
                 "blocked_reason": reason, "value_estimated": False}
        change = {"continuation_check": check}
        if reason:
            change.update(kind="stop" if reason == "TASK_DEADLINE_REACHED_DURING_PROCESSING" else "defer", candidate_id=None, revisit_at=None,
                reason="Operational continuation constraint: "+reason+"; retain acquired evidence and unknown information value.",
                selection_basis="operational_continuation_feasibility", expected_gain=None, cost_in_loss_units=None,
                value_status="UNSUPPORTED_OPEN_TEXT",
                proposal_before_admission={k: deepcopy(decision[k]) for k in ("kind", "candidate_id", "reason", "revisit_at")})
        return change


def get_continuation(config):
    if config is None:
        return None
    result = ContinuationPolicy()
    if config != result.configuration():
        raise ValueError("Unknown or changed continuation policy configuration")
    return result
