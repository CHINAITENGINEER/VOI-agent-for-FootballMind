"""Semantic audit for matched policy replays.

The finite replay tells us what a policy queried.  This module connects those
actions back to the state-query board so the evaluator can report which state
variables remained unresolved at the stopping point.  It keeps the agent view
and evaluator view separate: a private pre-query annotation can be known to
the evaluator while remaining an actionable unknown for the agent.
"""
from __future__ import annotations

from collections import Counter
import math
from typing import Any, Iterable, Mapping

from .state_query_contract import (
    ACTIONABLE_UNKNOWN, STATE_CONTESTED, StateQueryBoard,
)


VERSION = "policy-semantic-audit-v1"
_ACTIONABLE = frozenset({ACTIONABLE_UNKNOWN, STATE_CONTESTED})
_AVAILABILITY = frozenset({"available", "late", "failed", "duplicate"})


def _availability(value: str) -> str:
    if value not in _AVAILABILITY:
        raise ValueError(f"Unsupported candidate availability: {value}")
    return value


def audit_episode(
    board: StateQueryBoard,
    episode: Mapping[str, Any],
    *,
    view: str = "agent",
    candidate_availability: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Audit selected actions and residual state unknowns for one replay."""
    if not isinstance(board, StateQueryBoard):
        raise ValueError("board must be a StateQueryBoard")
    if not isinstance(episode, Mapping) or not isinstance(episode.get("trace"), list):
        raise ValueError("episode must contain a replay trace")
    raw_resolved = episode.get("resolved_state_variables", ())
    if raw_resolved is None:
        raw_resolved = ()
    if isinstance(raw_resolved, (str, bytes)):
        raise ValueError("resolved_state_variables must be a sequence")
    try:
        resolved_states = set(raw_resolved)
    except TypeError as exc:
        raise ValueError("resolved_state_variables must be a sequence") from exc
    if any(not isinstance(state, str) or not state.strip() for state in resolved_states):
        raise ValueError("resolved_state_variables must contain nonempty strings")
    availability = {candidate.candidate_id: "available" for candidate in board.candidates}
    if candidate_availability is not None:
        for candidate_id, status in candidate_availability.items():
            if candidate_id not in availability:
                raise ValueError(f"Unknown candidate availability ID: {candidate_id}")
            availability[candidate_id] = _availability(status)

    assessments = {row["candidate_id"]: row for row in board.assess_candidates(view=view)}
    declared_states = {row["state_variable"] for row in assessments.values()}
    if not resolved_states <= declared_states:
        raise ValueError("resolved_state_variables must refer to declared candidate states")
    action_ids = [candidate.action_id for candidate in board.candidates if candidate.action_id is not None]
    if len(action_ids) != len(set(action_ids)):
        raise ValueError("Candidate action IDs must be unique for semantic audit")
    by_action = {candidate.action_id: candidate for candidate in board.candidates if candidate.action_id is not None}

    queried_ids: list[str] = []
    queried_actions: list[str] = []
    for event in episode["trace"]:
        if not isinstance(event, Mapping):
            raise ValueError("Replay trace events must be mappings")
        action_id = event.get("selected_action")
        if action_id is None:
            continue
        candidate = by_action.get(action_id)
        if candidate is None:
            raise ValueError(f"Replay action is absent from the state-query board: {action_id}")
        if candidate.candidate_id in queried_ids:
            raise ValueError(f"Replay queried a candidate more than once: {candidate.candidate_id}")
        queried_ids.append(candidate.candidate_id)
        queried_actions.append(action_id)

    actionable_ids = [candidate_id for candidate_id, row in assessments.items()
                      if row["status"] in _ACTIONABLE]
    status_by_candidate: dict[str, str] = {}
    for candidate_id, row in assessments.items():
        if availability[candidate_id] != "available":
            status_by_candidate[candidate_id] = availability[candidate_id]
        elif candidate_id in queried_ids:
            status_by_candidate[candidate_id] = "queried"
        elif row["status"] in _ACTIONABLE:
            status_by_candidate[candidate_id] = "residual_unknown"
        else:
            status_by_candidate[candidate_id] = "state_not_actionable"

    residual = [candidate_id for candidate_id in actionable_ids if candidate_id not in queried_ids
                and status_by_candidate[candidate_id] == "residual_unknown"]
    queried_states = [assessments[candidate_id]["state_variable"] for candidate_id in queried_ids]
    residual_states = list(dict.fromkeys(assessments[candidate_id]["state_variable"] for candidate_id in residual))
    state_candidates: dict[str, list[str]] = {}
    for candidate_id, row in assessments.items():
        if row["status"] in _ACTIONABLE:
            state_candidates.setdefault(row["state_variable"], []).append(candidate_id)
    unknown_state_status: dict[str, str] = {}
    for state_variable, candidate_ids in state_candidates.items():
        if state_variable in resolved_states:
            unknown_state_status[state_variable] = "resolved"
        elif any(availability[candidate_id] != "available" for candidate_id in candidate_ids):
            route_statuses = {availability[candidate_id] for candidate_id in candidate_ids
                              if availability[candidate_id] != "available"}
            unknown_state_status[state_variable] = "unresolved_" + "+".join(sorted(route_statuses))
        elif state_variable in queried_states:
            unknown_state_status[state_variable] = "queried_unresolved"
        else:
            unknown_state_status[state_variable] = "residual_unknown"
    unresolved_states = [state for state, status in unknown_state_status.items() if status != "resolved"]

    return {
        "schema_version": VERSION,
        "view": view,
        "policy_id": episode.get("policy_id"),
        "seed": episode.get("seed"),
        "queried_candidate_ids": queried_ids,
        "queried_action_ids": queried_actions,
        "queried_state_variables": queried_states,
        "actionable_candidate_ids": actionable_ids,
        "residual_unknown_candidate_ids": residual,
        "residual_unknown_state_variables": residual_states,
        "residual_unknown_count": len(residual),
        "resolved_state_variables": sorted(resolved_states & set(state_candidates)),
        "state_resolution_status": unknown_state_status,
        "unresolved_state_variables": unresolved_states,
        "unresolved_state_count": len(unresolved_states),
        "status_by_candidate": status_by_candidate,
        "availability_by_candidate": availability,
        "stop_reason": episode.get("stop_reason"),
        "spent": episode.get("spent"),
        "query_count": episode.get("query_count"),
        "hidden_result_included": False,
    }


def audit_comparison(
    board: StateQueryBoard,
    comparison: Mapping[str, Any],
    *,
    view: str = "agent",
    candidate_availability: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Attach semantic unknown-state summaries to a matched comparison."""
    if not isinstance(comparison, Mapping) or not isinstance(comparison.get("episodes"), Mapping):
        raise ValueError("comparison must contain policy episodes")
    audits: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for policy_id, episodes in comparison["episodes"].items():
        rows = [audit_episode(board, episode, view=view, candidate_availability=candidate_availability)
                for episode in episodes]
        if not rows:
            raise ValueError("Every policy needs at least one episode")
        audits[policy_id] = rows
        summaries[policy_id] = {
            "episodes": len(rows),
            "mean_residual_unknown_count": math.fsum(row["residual_unknown_count"] for row in rows) / len(rows),
            "mean_unresolved_state_count": math.fsum(row["unresolved_state_count"] for row in rows) / len(rows),
            "mean_query_count": math.fsum(row["query_count"] for row in rows) / len(rows),
            "residual_unknown_counts": dict(sorted(Counter(row["residual_unknown_count"] for row in rows).items())),
            "stop_reasons": dict(sorted(Counter(row["stop_reason"] for row in rows).items())),
            "queried_state_variables": dict(sorted(Counter(
                state for row in rows for state in row["queried_state_variables"]
            ).items())),
            "availability_status_counts": dict(sorted(Counter(
                status for row in rows for status in row["status_by_candidate"].values()
            ).items())),
            "state_resolution_status_counts": dict(sorted(Counter(
                status for row in rows for status in row["state_resolution_status"].values()
            ).items())),
        }
    return {
        "schema_version": VERSION,
        "view": view,
        "visibility": "evaluator_only",
        "board_manifest": board.manifest(view=view),
        "summaries": summaries,
        "audits": audits,
        "hidden_result_included": False,
    }


def audit_episode_with_platform_evidence(
    board: StateQueryBoard,
    episode: Mapping[str, Any],
    evidence_rows: Iterable[Mapping[str, Any]],
    *,
    cutoff: str,
    view: str = "agent",
) -> dict[str, Any]:
    """Audit one replay after binding platform timestamps and quality."""
    from .evidence_binding import bind_candidate_evidence

    binding = bind_candidate_evidence(board, evidence_rows, cutoff=cutoff)
    result = audit_episode(board, episode, view=view,
                           candidate_availability=binding.availability_by_candidate)
    result["evidence_binding"] = binding.as_dict()
    return result


def audit_comparison_with_platform_evidence(
    board: StateQueryBoard,
    comparison: Mapping[str, Any],
    evidence_rows: Iterable[Mapping[str, Any]],
    *,
    cutoff: str,
    view: str = "agent",
) -> dict[str, Any]:
    """Attach one frozen platform evidence snapshot to every policy episode."""
    from .evidence_binding import bind_candidate_evidence

    binding = bind_candidate_evidence(board, evidence_rows, cutoff=cutoff)
    result = audit_comparison(board, comparison, view=view,
                              candidate_availability=binding.availability_by_candidate)
    result["evidence_binding"] = binding.as_dict()
    return result


__all__ = [
    "VERSION", "audit_episode", "audit_comparison",
    "audit_episode_with_platform_evidence", "audit_comparison_with_platform_evidence",
]
