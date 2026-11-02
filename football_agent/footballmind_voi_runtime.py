"""Match-only offline entry point for FootballMind's autonomous VoI loop.

This module is deliberately a thin integration boundary.  It owns no source
transport and makes no outcome prediction.  A caller supplies a match and a
public, as-of state; the adapter builds the candidate information menu and
delegates the decision to :mod:`autonomous_acquisition_step`.  Values are
accepted only when they are explicitly bound to the resulting state digest.

The returned trace is suitable for a later provider/execution layer: it says
which candidate was selected, deferred, or stopped, why, and what evidence is
still unresolved.  It never turns an unknown value into zero and never claims
numeric VoI calibration.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from typing import Any, Mapping

from .autonomous_acquisition_step import prepare_acquisition_state, select_acquisition_action
from .autonomous_match_contract import build_autonomous_initial_state, normalize_match_request

VERSION = "footballmind-voi-runtime-v1"

_FORBIDDEN_PUBLIC_KEYS = {
    "result", "outcome", "score", "scores", "label", "labels", "gold",
    "winner", "actual_result", "future_result", "realized_gain",
    "realized_loss", "settlement", "post_match", "evaluator",
    "probability", "probabilities", "prediction", "predictions",
}


def _digest(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha256(body).hexdigest()


def _scan_public(value: Any, path: str = "public_state") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).casefold() in _FORBIDDEN_PUBLIC_KEYS:
                raise ValueError(f"{path}.{key} contains outcome/evaluator data")
            _scan_public(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _scan_public(child, f"{path}[{index}]")


def _time(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an ISO timestamp string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _as_list(value: Any, name: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list")
    return [deepcopy(item) for item in value]


def _default_candidates(initial_state: Mapping[str, Any], public_state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Translate the match contract's information families to acquisition rows.

    The default rows intentionally have unknown utility values.  A source
    layer can provide ``candidate_menu`` with costs and availability, but this
    adapter never invents them from topic names.
    """
    configured = public_state.get("candidate_menu")
    if configured is not None:
        if not isinstance(configured, (list, tuple)):
            raise ValueError("public_state.candidate_menu must be a list")
        return _as_list(configured, "candidate_menu")
    rows = []
    for item in initial_state["candidate_catalog"]:
        rows.append({
            "candidate_id": item["action_id"],
            "kind": item["information_need"],
            "action_kind": "query",
            "source_ids": list(item["source_capabilities"]),
            "requires_source_ids": [],
            "aggregation_mode": "any_usable",
            # Unknown keeps the acquisition boundary honest.  A provider
            # registry may explicitly mark a route available in its menu.
            "action_availability": "unknown",
            "resource_cost": None,
            "utility_cost": None,
            "contract_prerequisites": list(item["prerequisites"]),
        })
    return rows


def build_match_voi_state(
    match: str | Mapping[str, Any],
    public_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a state-bound acquisition menu from match-only input.

    ``public_state`` may contain ``prediction_cutoff``, ``source_observations``
    and a provider-owned ``candidate_menu``.  A missing cutoff is retained as
    an explicit unresolved prerequisite rather than inferred from the clock.
    """
    if public_state is None:
        public_state = {}
    if not isinstance(public_state, Mapping):
        raise ValueError("public_state must be a mapping")
    _scan_public(public_state)
    normalized = normalize_match_request(match)
    cutoff = public_state.get("prediction_cutoff") or normalized["match"].get("prediction_cutoff")
    if cutoff is None:
        raise ValueError("prediction_cutoff is required for an as-of VoI state")
    cutoff = _time(cutoff, "prediction_cutoff")
    initial = build_autonomous_initial_state(normalized)
    match_id = str(public_state.get("match_id") or normalized["target_key"])
    snapshot = {
        "match_id": match_id,
        "prediction_cutoff": cutoff,
        "source_observations": _as_list(public_state.get("source_observations"), "source_observations"),
        "known_public_fields": [
            {"field": "home_team", "value": normalized["match"]["home"]},
            {"field": "away_team", "value": normalized["match"]["away"]},
        ],
    }
    actor_context = {
        "match_id": match_id,
        "prediction_cutoff": cutoff,
        "information_mode": "match_only_open_world",
        "snapshot": snapshot,
        "context_digest": _digest(snapshot),
    }
    candidates = _default_candidates(initial, public_state)
    state = prepare_acquisition_state(
        actor_context,
        candidates,
        candidate_generation_complete=bool(public_state.get("candidate_generation_complete", False)),
    )
    return {
        "schema_version": VERSION,
        "match_request": deepcopy(normalized),
        "contract_state": deepcopy(initial),
        "actor_context": actor_context,
        "acquisition_state": state,
        "audit": {
            "network_requests": 0,
            "model_calls": 0,
            "outcome_fields_consumed": False,
            "user_scope_question": None,
            "selection_owner": "agent",
            "numeric_voi_calibrated": False,
            "empirical_g_status": "deferred_timestamped_stream",
        },
    }


def _bind_estimates(
    state: Mapping[str, Any],
    estimates: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """Bind caller supplied estimates to this exact state digest.

    The caller still supplies the value and timestamps; the adapter only adds
    the candidate and state identity needed by the strict selector.
    """
    if estimates is None:
        return {}
    if not isinstance(estimates, Mapping):
        raise ValueError("estimates must be a mapping")
    candidate_ids = {row["candidate_id"] for row in state["candidates"]}
    bound: dict[str, dict[str, Any]] = {}
    for candidate_id, raw in estimates.items():
        if candidate_id not in candidate_ids:
            raise ValueError(f"estimate candidate absent from menu: {candidate_id}")
        if not isinstance(raw, Mapping):
            raise ValueError("each estimate must be a mapping")
        row = dict(raw)
        if row.get("candidate_id") not in (None, candidate_id):
            raise ValueError(f"estimate candidate mismatch: {candidate_id}")
        if row.get("state_digest") not in (None, state["state_digest"]):
            raise ValueError(f"estimate state digest mismatch: {candidate_id}")
        row["candidate_id"] = candidate_id
        row["state_digest"] = state["state_digest"]
        bound[candidate_id] = row
    return bound


def select_match_voi_step(
    bundle: Mapping[str, Any],
    *,
    estimates: Mapping[str, Mapping[str, Any]] | None = None,
    remaining_budget: Mapping[str, float] | None = None,
    loss_unit: str = "decision_loss",
) -> dict[str, Any]:
    """Select one acquisition, defer, or stop for a prepared match bundle."""
    if not isinstance(bundle, Mapping) or bundle.get("schema_version") != VERSION:
        raise ValueError("bundle must be a footballmind-voi-runtime-v1 object")
    state = bundle.get("acquisition_state")
    if not isinstance(state, Mapping):
        raise ValueError("bundle acquisition_state is required")
    trace = select_acquisition_action(
        state,
        loss_unit=loss_unit,
        estimates=_bind_estimates(state, estimates),
        remaining_budget=remaining_budget,
    )
    output = {
        "schema_version": VERSION,
        "match_request": deepcopy(bundle["match_request"]),
        "decision_trace": trace,
        "candidate_values": [
            {
                "candidate_id": row["candidate_id"],
                "net_value": row["assessment"].get("net_value"),
                "expected_value": row["assessment"].get("expected_value"),
                "cost": row["assessment"].get("cost"),
                "status": row["blocked_reason"] or row["assessment"].get("reason"),
            }
            for row in trace["evaluations"]
        ],
        "audit": deepcopy(bundle["audit"]),
    }
    output["audit"].update({
        "decision": trace["decision"],
        "selected_candidate_id": trace["selected_candidate_id"],
        "stop_or_defer_reason": trace["query_value_decision"].get("selection_basis")
        or trace["query_value_decision"].get("reason"),
        "trace_digest": trace["trace_digest"],
    })
    return output


def run_match_voi_step(
    match: str | Mapping[str, Any],
    public_state: Mapping[str, Any] | None = None,
    *,
    estimates: Mapping[str, Mapping[str, Any]] | None = None,
    remaining_budget: Mapping[str, float] | None = None,
    loss_unit: str = "decision_loss",
) -> dict[str, Any]:
    """Convenience entry point: match + public state -> one audited decision."""
    bundle = build_match_voi_state(match, public_state)
    return select_match_voi_step(
        bundle, estimates=estimates, remaining_budget=remaining_budget, loss_unit=loss_unit,
    )


__all__ = [
    "VERSION", "build_match_voi_state", "select_match_voi_step", "run_match_voi_step",
]
