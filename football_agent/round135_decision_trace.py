"""Deterministic autonomous query/refresh/stop decision trace (Round135-C).

This module is the machine gate between a public candidate menu and an executor.
It never asks a user what to search, reads outcome/evaluator fields, or treats an
unknown value estimate as zero.  A decision is a reservation only: actual spend
stays zero until an execution receipt is recorded by a separate layer.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

VERSION = "round135-decision-trace-v1"
EPS = 1e-12
GATE_STATUSES = frozenset({"available", "fresh", "stale", "missing", "unknown", "failed", "blocked", "future", "conflict", "duplicate", "late"})
ACTION_ELIGIBILITY = frozenset({"executable", "unavailable", "unknown"})
DEFER_REASONS = frozenset({"defer_unknown_estimate", "provenance_deferred", "defer_incomplete_candidate_menu", "defer_budget"})
_FORBIDDEN = frozenset({"result", "score", "outcome", "goals", "home_score", "away_score", "final_loss", "label", "gold", "sidecar", "evaluator"})


def _text(v: Any, name: str) -> str:
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return v.strip()


def _finite(v: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise ValueError(f"{name} must be finite")
    value = float(v)
    if minimum is not None and value < minimum - EPS:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _iso(v: Any, name: str) -> str:
    if isinstance(v, datetime):
        dt = v
    elif isinstance(v, str):
        raw = v.strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be an ISO timestamp") from exc
    else:
        raise ValueError(f"{name} must be an ISO timestamp")
    if dt.tzinfo is None:
        raise ValueError(f"{name} must include timezone")
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _cutoff_leq(training: str, prediction: str) -> bool:
    def parse(s: str) -> datetime:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    return parse(training) <= parse(prediction)


def _digest(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _gate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("provenance_gate", "gate", "provenance"):
        value = candidate.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _gate_status(candidate: Mapping[str, Any]) -> str:
    gate = _gate(candidate)
    status = gate.get("status", gate.get("eligibility", candidate.get("provenance_status", "unknown")))
    if isinstance(status, Mapping):
        status = status.get("status", status.get("state", "unknown"))
    status = str(status).strip().lower() if status is not None else "unknown"
    return status if status in GATE_STATUSES else "unknown"


def _action_eligibility(candidate: Mapping[str, Any]) -> str:
    gate = _gate(candidate)
    raw = gate.get("action_eligibility", candidate.get("action_eligibility", "executable"))
    raw = str(raw).strip().lower() if raw is not None else "unknown"
    return raw if raw in ACTION_ELIGIBILITY else "unknown"


def _dependencies(candidate: Mapping[str, Any]) -> tuple[str, ...]:
    raw = candidate.get("dependencies", candidate.get("dependency_keys", ()))
    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)):
        raise ValueError("dependencies must be a sequence")
    out = tuple(_text(v, "dependency") for v in raw)
    if len(set(out)) != len(out):
        raise ValueError("dependencies must be unique")
    return out


def _candidate_value(candidate: Mapping[str, Any]) -> tuple[float | None, str | None, str | None, str | None]:
    estimate = candidate.get("value_estimate")
    if isinstance(estimate, Mapping):
        raw = estimate.get("value", estimate.get("expected_value"))
        source = estimate.get("source", estimate.get("value_estimate_source", candidate.get("value_estimate_source", candidate.get("estimate_source"))))
        training = estimate.get("training_cutoff", estimate.get("value_training_cutoff", candidate.get("value_training_cutoff", candidate.get("estimate_training_cutoff"))))
        unit = estimate.get("unit", estimate.get("value_unit", candidate.get("value_unit", candidate.get("loss_unit"))))
    else:
        raw = candidate.get("expected_value", estimate)
        source = candidate.get("value_estimate_source", candidate.get("estimate_source"))
        training = candidate.get("value_training_cutoff", candidate.get("estimate_training_cutoff"))
        unit = candidate.get("value_unit", candidate.get("loss_unit"))
    if raw is None:
        return None, None if source is None else str(source), None if training is None else str(training), None if unit is None else str(unit)
    value = _finite(raw, "expected_value", minimum=0)
    if source is None or not isinstance(source, str) or not source.strip():
        raise ValueError("numeric expected_value requires value_estimate_source")
    if training is None:
        raise ValueError("numeric expected_value requires value_training_cutoff")
    training_iso = _iso(training, "value_training_cutoff")
    return value, source.strip(), training_iso, None if unit is None else str(unit).strip()


def _cost(candidate: Mapping[str, Any]) -> tuple[float | None, str | None]:
    raw = candidate.get("cost")
    unit = candidate.get("cost_unit", candidate.get("loss_unit"))
    if raw is None:
        return None, None if unit is None else str(unit).strip()
    value = _finite(raw, "cost", minimum=0)
    return value, None if unit is None else str(unit).strip()


def _kind(candidate: Mapping[str, Any], status: str) -> str:
    kind = candidate.get("action_kind", candidate.get("kind"))
    if kind is None:
        kind = "refresh" if status in {"stale", "missing", "failed", "late"} else "query"
    kind = str(kind).strip().lower()
    if kind not in {"query", "refresh"}:
        raise ValueError("action_kind must be query or refresh")
    return kind


def _normalize_candidates(candidates: Sequence[Mapping[str, Any]], *, prediction_cutoff: str, value_unit: str | None, cost_unit: str | None) -> tuple[list[dict[str, Any]], str | None, str | None]:
    if not isinstance(candidates, (list, tuple)) or isinstance(candidates, (str, bytes)) or not candidates:
        raise ValueError("candidates must be a nonempty sequence")
    seen: set[str] = set(); rows: list[dict[str, Any]] = []
    units_value: str | None = value_unit; units_cost: str | None = cost_unit
    for index, raw in enumerate(candidates):
        if not isinstance(raw, Mapping):
            raise ValueError(f"candidate[{index}] must be a mapping")
        cid = _text(raw.get("candidate_id", raw.get("action_id")), f"candidate[{index}].candidate_id")
        if cid in seen:
            raise ValueError(f"duplicate candidate: {cid}")
        seen.add(cid)
        status = _gate_status(raw); eligibility = _action_eligibility(raw)
        kind = _kind(raw, status); deps = _dependencies(raw)
        value, source, training, vu = _candidate_value(raw)
        cost, cu = _cost(raw)
        if vu is not None:
            if units_value is None: units_value = vu
            elif units_value != vu: raise ValueError("value units must match")
        if cu is not None:
            if units_cost is None: units_cost = cu
            elif units_cost != cu: raise ValueError("cost units must match")
        if value is not None and training is not None and not _cutoff_leq(training, prediction_cutoff):
            raise ValueError(f"{cid} value estimate is trained after prediction cutoff")
        if value is not None and units_value is None:
            raise ValueError(f"{cid} numeric value needs value_unit")
        if cost is not None and units_cost is None:
            raise ValueError(f"{cid} cost needs cost_unit")
        if value is not None and cost is not None and units_value != units_cost:
            raise ValueError(f"{cid} value and cost units differ")
        feasible = eligibility == "executable" and not deps
        # An action remains a candidate when evidence is stale/missing.  The
        # gate status describes evidence, while action_eligibility describes
        # whether a refresh/query operation itself can run.
        blocked_reason = None
        if eligibility != "executable": blocked_reason = "action_eligibility_" + eligibility
        elif deps: blocked_reason = "dependencies_unmet"
        if status in {"future", "conflict", "duplicate"}:
            # Future/conflicting evidence cannot justify a current gate.  The
            # action may still be proposed by a later refresh planner, but
            # this decision trace defers it until provenance is resolved.
            feasible = False; blocked_reason = "provenance_deferred"
        if cost is None: feasible = False; blocked_reason = blocked_reason or "cost_unknown"
        rows.append({"candidate_id": cid, "action_kind": kind, "gate_status": status,
                     "action_eligibility": eligibility, "dependencies": list(deps),
                     "expected_value": value, "value_estimate_source": source,
                     "value_training_cutoff": training, "value_unit": units_value,
                     "cost": cost, "cost_unit": units_cost, "feasible": feasible,
                     "blocked_reason": blocked_reason})
    if units_value is not None and units_cost is not None and units_value != units_cost:
        raise ValueError("value and cost units must match")
    return rows, units_value, units_cost


def _rank(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    def key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        value = row.get("expected_value"); cost = row.get("cost")
        numeric = value is not None and cost is not None and row.get("feasible")
        net = value - cost if numeric else None
        gate_rank = 0 if row.get("gate_status") in {"stale", "missing", "failed", "late"} else 1
        return (0 if numeric else 1, -(net if net is not None else float("-inf")), -(value if value is not None else float("-inf")), cost if cost is not None else float("inf"), gate_rank, row["candidate_id"])
    ranked = sorted(rows, key=key)
    return [{"rank": i + 1, "candidate_id": r["candidate_id"], "action_kind": r["action_kind"],
             "feasible": r["feasible"], "gate_status": r["gate_status"],
             "net_utility": (r["expected_value"] - r["cost"] if r["expected_value"] is not None and r["cost"] is not None and r["feasible"] else None),
             "selection_basis": "numeric_net_utility" if r["expected_value"] is not None and r["cost"] is not None and r["feasible"] else "unknown_or_infeasible"} for i, r in enumerate(ranked)]


def decide_next_action(*, episode_id: str, prediction_cutoff: str | datetime, candidates: Sequence[Mapping[str, Any]], budget: float | None = None, actual_spent: float = 0.0, candidate_generation_complete: bool = False, value_unit: str | None = None, cost_unit: str | None = None, prior_cost_ledger: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Return one deterministic machine decision and an auditable trace."""
    episode_id = _text(episode_id, "episode_id"); cutoff = _iso(prediction_cutoff, "prediction_cutoff")
    budget_value = None if budget is None else _finite(budget, "budget", minimum=0)
    spent = _finite(actual_spent, "actual_spent", minimum=0)
    rows, vu, cu = _normalize_candidates(candidates, prediction_cutoff=cutoff, value_unit=value_unit, cost_unit=cost_unit)
    if budget_value is not None and spent > budget_value + EPS: raise ValueError("actual_spent exceeds budget")
    available_budget = None if budget_value is None else max(0.0, budget_value - spent)
    for row in rows:
        if available_budget is not None and row["cost"] is not None and row["cost"] > available_budget + EPS:
            row["feasible"] = False; row["blocked_reason"] = "budget_unavailable"
    ranked = _rank(rows)
    feasible = [r for r in rows if r["feasible"]]
    unknown = [r for r in feasible if r["expected_value"] is None]
    numeric = [r for r in feasible if r["expected_value"] is not None and r["cost"] is not None]
    positive = [r for r in numeric if r["expected_value"] - r["cost"] > EPS]
    selected = None; decision = "stop"; stop_reason = None
    if not candidate_generation_complete:
        decision = "defer"; stop_reason = "defer_incomplete_candidate_menu"
    elif unknown and not positive:
        decision = "defer"; stop_reason = "defer_unknown_estimate"
    elif positive:
        selected = sorted(positive, key=lambda r: (-(r["expected_value"] - r["cost"]), -r["expected_value"], r["cost"], r["candidate_id"]))[0]
        decision = "acquire"; stop_reason = "numeric_net_utility_gate"
    elif unknown:
        decision = "defer"; stop_reason = "defer_unknown_estimate"
    elif not feasible:
        decision = "defer"; stop_reason = "provenance_deferred" if any(r["blocked_reason"] == "provenance_deferred" or r["action_eligibility"] != "executable" for r in rows) else "defer_budget"
    else:
        decision = "stop"; stop_reason = "stop_all_feasible_net_utility_nonpositive"
    stop_certificate = None
    if decision == "stop":
        nets = [r["expected_value"] - r["cost"] for r in numeric]
        stop_certificate = {
            "kind": "MODEL_SCOPED_STOP",
            "scope": "complete_public_candidate_menu_and_finite_value_estimates",
            "global_claim": False,
            "reason": stop_reason,
            "actionable_unknown_count": len(unknown),
            "evaluated_candidate_ids": sorted(r["candidate_id"] for r in numeric),
            "maximum_modeled_net_value": max(nets) if nets else None,
        }
    ledger = []
    if prior_cost_ledger is not None:
        if not isinstance(prior_cost_ledger, (list, tuple)): raise ValueError("prior_cost_ledger must be a sequence")
        for item in prior_cost_ledger:
            if not isinstance(item, Mapping): raise ValueError("prior ledger entry must be a mapping")
            ledger.append({"kind": str(item.get("kind", "execution")), "candidate_id": item.get("candidate_id"), "reserved_cost": _finite(item.get("reserved_cost", 0), "reserved_cost", minimum=0), "actual_spent": _finite(item.get("actual_spent", 0), "actual_spent", minimum=0), "receipt_id": item.get("receipt_id")})
    if selected is not None:
        ledger.append({"kind": "cost_reservation", "candidate_id": selected["candidate_id"], "reserved_cost": selected["cost"], "actual_spent": 0.0, "receipt_id": None})
    public_candidates = [{k: r[k] for k in ("candidate_id", "action_kind", "gate_status", "action_eligibility", "dependencies", "expected_value", "value_estimate_source", "value_training_cutoff", "value_unit", "cost", "cost_unit", "feasible", "blocked_reason")} for r in rows]
    payload = {"episode_id": episode_id, "prediction_cutoff": cutoff, "candidate_generation_complete": bool(candidate_generation_complete), "budget": budget_value, "actual_spent_before_decision": spent, "candidates": public_candidates, "decision": decision, "selected_candidate_id": None if selected is None else selected["candidate_id"], "stop_reason": stop_reason, "stop_certificate": stop_certificate, "candidate_ranking": ranked, "cost_ledger": ledger, "units": {"value": vu, "cost": cu}, "actual_spent_for_reservation": 0.0}
    trace_hash = _digest(payload)
    return {"schema_version": VERSION, "trace_id": trace_hash, "episode_id": episode_id, "prediction_cutoff": cutoff, "decision": decision, "selected_candidate_id": None if selected is None else selected["candidate_id"], "selected_action_kind": None if selected is None else selected["action_kind"], "stop_reason": stop_reason, "stop_certificate": stop_certificate, "candidate_ranking": ranked, "cost_ledger": ledger, "reservation": None if selected is None else {"candidate_id": selected["candidate_id"], "reserved_cost": selected["cost"], "actual_spent": 0.0, "status": "reserved"}, "actual_spent": 0.0, "actual_spent_before_decision": spent, "budget": budget_value, "remaining_budget": available_budget, "candidate_generation_complete": bool(candidate_generation_complete), "public_candidates": public_candidates, "units": {"value": vu, "cost": cu}, "replay": {"input_digest": _digest({"episode_id": episode_id, "prediction_cutoff": cutoff, "candidate_generation_complete": bool(candidate_generation_complete), "budget": budget_value, "actual_spent": spent, "candidates": public_candidates}), "step": 1, "previous_trace_id": None}, "outcome_labels_excluded": True}


def build_decision_trace(**kwargs: Any) -> dict[str, Any]:
    return decide_next_action(**kwargs)


def choose_next_action(**kwargs: Any) -> dict[str, Any]:
    return decide_next_action(**kwargs)


__all__ = ["VERSION", "decide_next_action", "build_decision_trace", "choose_next_action", "DEFER_REASONS"]
