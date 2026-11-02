"""Join public platform evidence, provenance and the existing query-value contract.

Candidates are machine-generated declarations supplied through a separate input.
This boundary neither proposes football questions nor estimates their value: an
estimator must bind each estimate to the *current* public state. Resource budgets
remain independent of the declared cost in decision-loss units. No executor or
sidecar is read here; an acquisition decision is only a reservation proposal.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from .query_value_contract import (
    BUDGET_EXHAUSTED, DEFER, MODEL_SCOPED_STOP, NO_FEASIBLE_REGISTERED_ACTION,
    QUERY, STOP, WITHHELD_OPEN_WORLD_VALUE_UNKNOWN,
    QueryValueAssessment, QueryValueDecision, StopCertificate,
)
from .round135_provenance_gate import build_provenance_gate

VERSION = "autonomous-acquisition-step-v1"
_USABLE = {"available", "duplicate"}


def _evidence_satisfied(provenance: Mapping[str, Any]) -> bool:
    """Whether the action's declared target evidence is already usable.

    ``source_ids`` are the claims a query would obtain.  A query for a claim
    that is already usable is a duplicate acquisition, even if its numeric
    estimate is positive.  ``any_usable`` candidates need one usable source;
    ``all_required`` candidates need all of them.  Explicit ``refresh`` is
    handled by the caller because it may be valuable despite fresh evidence.
    """
    states = list((provenance.get("source_states") or {}).values())
    if not states:
        return False
    mode = provenance.get("aggregation_mode", "all_required")
    if mode == "any_usable":
        return any(state in _USABLE for state in states)
    return all(state in _USABLE for state in states)


def _digest(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _number(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def _time(value: Any, name: str) -> datetime:
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError(f"{name} must be a timezone-aware timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include timezone")
    return parsed.astimezone(timezone.utc)


def _ids(value: Any, name: str) -> list[str]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a sequence")
    ids = [_text(item, name) for item in value]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name} must be unique")
    return sorted(ids)


def _resources(value: Any, name: str) -> dict[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    result = {_text(key, name): _number(amount, f"{name}.{key}")
              for key, amount in value.items()}
    if len(result) != len(value):
        raise ValueError(f"{name} has duplicate normalized resource names")
    return dict(sorted(result.items()))


def prepare_acquisition_state(
    actor_context: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]],
    *, candidate_generation_complete: bool = False,
) -> dict[str, Any]:
    """Project a machine candidate catalog and bind it to public evidence.

    ``source_ids`` names evidence an action would acquire. In contrast,
    ``requires_source_ids`` names evidence already needed to run it. Missing
    target evidence is therefore never mistaken for a missing prerequisite.
    ``kind`` is a topic (injury/weather/etc.); only ``action_kind`` controls
    query versus refresh. Unspecified action availability stays unknown.
    """
    if type(candidate_generation_complete) is not bool:
        raise ValueError("candidate_generation_complete must be boolean")
    if not isinstance(actor_context, Mapping) or "actor_context" in actor_context:
        raise ValueError("pass only the public actor_context, not its envelope")
    snapshot = actor_context.get("snapshot")
    if not isinstance(snapshot, Mapping) or actor_context.get("context_digest") != _digest(snapshot):
        raise ValueError("actor context snapshot digest mismatch")
    match_id = _text(actor_context.get("match_id"), "match_id")
    cutoff = _time(actor_context.get("prediction_cutoff"), "prediction_cutoff")
    if snapshot.get("match_id") != match_id or _time(snapshot.get("prediction_cutoff"), "snapshot cutoff") != cutoff:
        raise ValueError("actor context identity mismatch")
    if not isinstance(candidates, (list, tuple)):
        raise ValueError("candidates must be a sequence")
    rows, seen = [], set()
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("candidate must be a mapping")
        cid = _text(candidate.get("candidate_id"), "candidate_id")
        if cid in seen:
            raise ValueError("duplicate candidate_id")
        seen.add(cid)
        kind = candidate.get("action_kind", "auto")
        if kind not in {"auto", "query", "refresh"}:
            raise ValueError("action_kind must be auto, query or refresh")
        cost = candidate.get("utility_cost")
        if cost is not None:
            if not isinstance(cost, Mapping):
                raise ValueError("utility_cost must be a mapping")
            cost = {"amount": _number(cost.get("amount"), "utility_cost.amount"),
                    "unit": _text(cost.get("unit"), "utility_cost.unit"),
                    "source": _text(cost.get("source"), "utility_cost.source")}
        availability = candidate.get("action_availability", "unknown")
        if type(availability) is bool:
            availability = "available" if availability else "unavailable"
        if availability not in {"available", "unavailable", "unknown"}:
            raise ValueError("action_availability is invalid")
        rows.append({"candidate_id": cid, "action_kind": kind,
                     "source_ids": _ids(candidate.get("source_ids", []), "source_ids"),
                     "requires_source_ids": _ids(candidate.get("requires_source_ids", []), "requires_source_ids"),
                     "aggregation_mode": candidate.get("aggregation_mode", "all_required"),
                     "action_availability": availability,
                     "resource_cost": _resources(candidate.get("resource_cost"), "resource_cost"),
                     "utility_cost": cost})
    rows.sort(key=lambda item: item["candidate_id"])
    gate = None
    if rows:
        # The frozen gate's audit is excluded from actor input and state hash.
        report = build_provenance_gate(
            {"prediction_cutoff": cutoff.isoformat()}, candidates=rows,
            observations=snapshot.get("source_observations", []))
        gate = {key: value for key, value in report.items() if key != "audit"}
    state = {"schema_version": VERSION, "match_id": match_id,
             "prediction_cutoff": cutoff.isoformat(),
             "information_mode": actor_context.get("information_mode"),
             "context_digest": actor_context["context_digest"],
             "snapshot": deepcopy(dict(snapshot)), "candidates": rows,
             "candidate_generation_complete": candidate_generation_complete,
             "provenance_gate": gate}
    state["state_digest"] = _digest(state)
    return state


def _estimate(candidate_id: str, state: Mapping[str, Any], value: Any,
              loss_unit: str, cost: Mapping[str, Any] | None) -> tuple[QueryValueAssessment, dict[str, Any]]:
    cost_amount = cost["amount"] if cost else None
    reason = "estimate_missing"
    if value is not None:
        if not isinstance(value, Mapping):
            raise ValueError("estimate must be a mapping")
        if value.get("candidate_id") != candidate_id or value.get("state_digest") != state["state_digest"]:
            reason = "estimate_state_or_candidate_mismatch"
        else:
            for field in ("source", "unit", "training_cutoff", "available_at"):
                _text(value.get(field), f"estimate.{field}")
            expected = _number(value.get("expected_value"), "estimate.expected_value")
            cutoff = _time(state["prediction_cutoff"], "prediction_cutoff")
            trained = _time(value["training_cutoff"], "estimate.training_cutoff")
            available = _time(value["available_at"], "estimate.available_at")
            if trained > cutoff or available > cutoff or available < trained:
                reason = "estimate_not_available_at_cutoff"
            elif value["unit"] != loss_unit:
                raise ValueError("estimate unit differs from decision loss unit")
            elif cost is None:
                reason = "utility_cost_unknown"
            else:
                assessment = QueryValueAssessment.modeled(
                    candidate_id, expected_value=expected, cost=cost_amount,
                    reason="state_bound_declared_estimate")
                return assessment, {"source": value["source"], "unit": loss_unit,
                                    "training_cutoff": trained.isoformat(),
                                    "available_at": available.isoformat(),
                                    "state_digest": state["state_digest"]}
    return QueryValueAssessment.unmeasured(candidate_id, cost=cost_amount, reason=reason), {}


def select_acquisition_action(
    state: Mapping[str, Any], *, loss_unit: str,
    estimates: Mapping[str, Mapping[str, Any]] | None = None,
    remaining_budget: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Select one query/refresh, or issue a bounded defer/stop decision.

    Menu incompleteness limits stopping claims but does not prevent a supported
    positive action. Values lacking a current state binding stay unmeasured.
    A fresh or duplicate target is treated as already satisfied for query/auto
    actions; only an explicit refresh can spend resources to seek a revision.
    Physical resources are checked independently from cost already translated
    into the declared decision-loss unit.
    """
    if not isinstance(state, Mapping) or state.get("schema_version") != VERSION:
        raise ValueError("invalid acquisition state")
    if state.get("state_digest") != _digest({key: value for key, value in state.items() if key != "state_digest"}):
        raise ValueError("acquisition state digest mismatch")
    loss_unit = _text(loss_unit, "loss_unit")
    budget = _resources(remaining_budget, "remaining_budget")
    estimates = {} if estimates is None else estimates
    if not isinstance(estimates, Mapping):
        raise ValueError("estimates must be a mapping")
    if set(estimates) - {row["candidate_id"] for row in state["candidates"]}:
        raise ValueError("estimate candidate absent from menu")
    gate = state["provenance_gate"] or {"candidates": [], "sources": {}}
    gates = {row["candidate_id"]: row for row in gate["candidates"]}
    evaluations, assessments = [], []
    for row in state["candidates"]:
        cid = row["candidate_id"]
        provenance = gates[cid]
        action_kind = row["action_kind"]
        if action_kind == "auto":
            action_kind = "refresh" if "stale" in provenance["source_states"].values() else "query"
        cost = row["utility_cost"]
        if cost is not None and cost["unit"] != loss_unit:
            raise ValueError("utility cost unit differs from decision loss unit")
        unmet = [sid for sid in row["requires_source_ids"]
                 if gate["sources"].get(sid, {}).get("state") not in _USABLE]
        blocked = None
        if provenance["action_eligibility"] != "executable":
            blocked = "action_" + provenance["action_eligibility"]
        elif unmet:
            blocked = "prerequisites_unresolved"
        # A query/auto action is a request to obtain target evidence.  Once
        # that evidence is already usable, calling the provider again would
        # spend resources without changing the actor's information state.
        # An explicit refresh remains executable because its value may be the
        # arrival of a new revision before the cutoff.
        elif action_kind != "refresh" and _evidence_satisfied(provenance):
            blocked = "evidence_already_usable"
        resources = row["resource_cost"]
        if blocked is None:
            if resources is None:
                blocked = "resource_cost_unknown"
            elif any(amount > 0 and (budget is None or key not in budget) for key, amount in resources.items()):
                blocked = "resource_budget_unknown"
            elif any(amount > (budget or {}).get(key, 0) for key, amount in resources.items()):
                blocked = "resource_budget_exhausted"
        assessment, support = _estimate(cid, state, estimates.get(cid), loss_unit, cost)
        # Even when no query is needed, an explicitly supplied estimate must
        # still be state-bound.  A caller carrying an estimate from an older
        # snapshot gets a defer rather than a seemingly successful stop; this
        # prevents stale VOI evidence from being silently accepted on a new
        # public state.
        if (blocked == "evidence_already_usable"
                and assessment.reason == "estimate_state_or_candidate_mismatch"):
            blocked = "estimate_state_or_candidate_mismatch"
        assessments.append(assessment)
        evaluations.append({"candidate_id": cid, "action_kind": action_kind,
                            "evidence_eligibility": provenance["eligibility"],
                            "source_states": provenance["source_states"],
                            "action_eligibility": provenance["action_eligibility"],
                            "unmet_prerequisites": unmet, "blocked_reason": blocked,
                            "resource_cost": resources, "utility_cost": cost,
                            "assessment": assessment.as_dict(), "estimate_support": support})
    positive = [row for row in evaluations if row["blocked_reason"] is None
                and row["assessment"]["net_value"] is not None
                and row["assessment"]["net_value"] > 1e-12]
    unresolved = [row for row in evaluations
                  if row["blocked_reason"] not in {
                      "action_unavailable", "resource_budget_exhausted",
                      "evidence_already_usable",
                  }
                  and (row["blocked_reason"] is not None or row["assessment"]["net_value"] is None)]
    feasible = [row for row in evaluations if row["blocked_reason"] is None]
    selected = None
    if positive:
        selected = min(positive, key=lambda row: (-row["assessment"]["net_value"], row["candidate_id"]))
        decision = QueryValueDecision(QUERY, selected["candidate_id"],
                                     "best_supported_state_bound_net_value", tuple(assessments), None)
    else:
        evaluated = tuple(row["candidate_id"] for row in feasible
                          if row["assessment"]["net_value"] is not None)
        maximum = max((row["assessment"]["net_value"] for row in feasible
                       if row["assessment"]["net_value"] is not None), default=None)
        if unresolved or not state["candidate_generation_complete"]:
            action, kind, scope = DEFER, WITHHELD_OPEN_WORLD_VALUE_UNKNOWN, "open_world"
            reason = "unresolved_action_cost_or_value" if unresolved else "candidate_menu_incomplete"
        elif feasible:
            action, kind, scope, reason = STOP, MODEL_SCOPED_STOP, "declared_finite_model", "all_feasible_modeled_net_values_nonpositive"
        elif any(row["blocked_reason"] == "resource_budget_exhausted" for row in evaluations):
            action, kind, scope, reason = DEFER, BUDGET_EXHAUSTED, "operational_budget", "resource_budget_exhausted"
        elif any(row["blocked_reason"] == "evidence_already_usable" for row in evaluations):
            # There is no acquisition action left in the registered menu.  A
            # bounded registry stop states that the current evidence is
            # sufficient without claiming anything about unregistered tools.
            action, kind, scope, reason = (
                STOP, NO_FEASIBLE_REGISTERED_ACTION, "declared_candidate_registry",
                "all_registered_actions_already_satisfied",
            )
        else:
            action, kind, scope, reason = DEFER, NO_FEASIBLE_REGISTERED_ACTION, "declared_candidate_registry", "no_executable_registered_action"
        certificate = StopCertificate(kind, scope, False, reason, len(unresolved), evaluated,
                                      budget or {}, maximum)
        decision = QueryValueDecision(action, None, reason, tuple(assessments), certificate)
    reservation = None if selected is None else {
        "candidate_id": selected["candidate_id"], "state_digest": state["state_digest"],
        "status": "proposed", "resource_cost": deepcopy(selected["resource_cost"]),
        "utility_cost": deepcopy(selected["utility_cost"]), "execution_receipt_id": None,
        "actual_spend_observed": False,
    }
    trace = {"schema_version": VERSION, "state_digest": state["state_digest"],
             "context_digest": state["context_digest"], "match_id": state["match_id"],
             "prediction_cutoff": state["prediction_cutoff"],
             "decision": {QUERY: "acquire", STOP: "stop", DEFER: "defer"}[decision.action],
             "selected_candidate_id": decision.candidate_id,
             "selected_action_kind": selected["action_kind"] if selected else None,
             "candidate_generation_complete": state["candidate_generation_complete"],
             "loss_unit": loss_unit, "remaining_budget": budget,
             "evaluations": evaluations, "query_value_decision": decision.as_dict(),
             "reservation": reservation, "numeric_voi_calibrated": False,
             "global_optimum_claim": False}
    trace["trace_digest"] = _digest(trace)
    if selected is not None:
        # Stable identity for the whole acquisition action.  Transport-level
        # request ids are assigned later and must not become the action id.
        trace["acquisition_id"] = "acq-" + _digest({
            "trace_digest": trace["trace_digest"],
            "candidate_id": selected["candidate_id"],
            "action_kind": selected["action_kind"],
        })[:24]
        trace["trace_digest"] = _digest({key: value for key, value in trace.items()
                                          if key != "trace_digest"})
    return trace


__all__ = ["VERSION", "prepare_acquisition_state", "select_acquisition_action"]
