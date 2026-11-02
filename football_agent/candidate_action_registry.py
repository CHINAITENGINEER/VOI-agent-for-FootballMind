"""Bind proposal menus to explicit actions and receipt-driven replanning."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from typing import Any, Iterable, Mapping

from .autonomous_acquisition_step import prepare_acquisition_state
from .candidate_menu_compiler import VERSION as MENU_VERSION, compile_source_backed_menu
from .execution_receipt_bridge import COMPLETED, rebuild_context_after_receipt
from .source_context_adapter import build_source_backed_context

VERSION = "candidate-action-registry-v1"

_FORBIDDEN = frozenset({
    "result", "outcome", "score", "scores", "goals", "home_score",
    "away_score", "home_goals", "away_goals", "final_score", "label",
    "evaluator", "sidecar", "realized_loss", "decision_loss",
    "actual_result", "future_result", "sequence_losses", "action_returns",
})


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _time(value: Any, name: str) -> datetime:
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a timezone-aware timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware timestamp")
    return parsed.astimezone(timezone.utc)


def _digest(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _finite(value: Any, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _resources(value: Any, name: str) -> dict[str, float]:
    raw = _mapping(value, name)
    result = {}
    for key, amount in raw.items():
        field = _text(str(key), f"{name}.key")
        result[field] = _finite(amount, f"{name}.{field}")
    if not result:
        raise ValueError(f"{name} must not be empty")
    return dict(sorted(result.items()))


def _safe(value: Any, path: str = "parameters") -> Any:
    if isinstance(value, Mapping):
        result = {}
        for key, child in value.items():
            name = _text(str(key), f"{path}.key")
            if name.casefold() in _FORBIDDEN:
                raise ValueError(f"{path} contains an evaluator field")
            result[name] = _safe(child, f"{path}.{name}")
        return result
    if isinstance(value, (list, tuple)):
        return [_safe(child, f"{path}[]") for child in value]
    if value is None or isinstance(value, (str, bool, int)):
        return deepcopy(value)
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(f"{path} contains an unsupported value")


def _action_availability(value: Any, name: str) -> str:
    if value not in {"available", "unavailable", "unknown", "stale", "duplicate"}:
        raise ValueError(f"{name} is invalid")
    if value in {"available", "stale", "duplicate"}:
        return "available"
    return value


def _verify_registry(registry: Mapping[str, Any]) -> None:
    if not isinstance(registry, Mapping) or registry.get("schema_version") != VERSION:
        raise ValueError("invalid action registry version")
    body = {key: value for key, value in registry.items() if key != "registry_digest"}
    if registry.get("registry_digest") != _digest(body):
        raise ValueError("action registry digest mismatch")


def build_action_registry(
    menu_report: Mapping[str, Any],
    *,
    actor_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile a proposal-only menu into explicit state-bound action routes."""
    report = _mapping(menu_report, "menu_report")
    if report.get("schema_version") != MENU_VERSION:
        raise ValueError("menu_report schema_version mismatch")
    contract = _mapping(report.get("contract"), "menu_report.contract")
    if contract.get("proposal_only") is not True or contract.get("numeric_voi_estimated") is not False:
        raise ValueError("menu_report must remain proposal-only")
    cutoff = _time(report.get("prediction_cutoff"), "menu_report.prediction_cutoff")
    if actor_context is not None:
        actor = _mapping(actor_context, "actor_context")
        actor_cutoff = _time(actor.get("prediction_cutoff"), "actor_context.prediction_cutoff")
        if actor_cutoff != cutoff:
            raise ValueError("menu and actor cutoffs differ")
        match_id = _text(actor.get("match_id"), "actor_context.match_id")
        context_digest = _text(actor.get("context_digest"), "actor_context.context_digest")
    else:
        match_id = None
        context_digest = None
    menu = report.get("candidate_menu")
    routes = _mapping(report.get("routes"), "menu_report.routes")
    visibility = _mapping(report.get("source_visibility", {}), "menu_report.source_visibility")
    if not isinstance(menu, (list, tuple)) or not menu:
        raise ValueError("menu_report.candidate_menu must be nonempty")
    entries = []
    acquisition_candidates = []
    by_candidate = {}
    by_action = {}
    for index, raw in enumerate(menu):
        item = _mapping(raw, f"candidate_menu[{index}]")
        candidate_id = _text(item.get("candidate_id"), f"candidate_menu[{index}].candidate_id")
        if candidate_id in by_candidate:
            raise ValueError(f"duplicate candidate_id: {candidate_id}")
        route = _mapping(routes.get(candidate_id), f"routes[{candidate_id}]")
        action_id = _text(route.get("action_id"), f"routes[{candidate_id}].action_id")
        if action_id in by_action:
            raise ValueError(f"duplicate action_id: {action_id}")
        source_ids = item.get("source_ids")
        if not isinstance(source_ids, (list, tuple)) or not source_ids:
            raise ValueError(f"candidate_menu[{index}].source_ids must be nonempty")
        normalized_sources = []
        for source in source_ids:
            source_id = _text(source, f"candidate_menu[{index}].source_ids[]")
            if source_id not in normalized_sources:
                normalized_sources.append(source_id)
        route_source = _text(route.get("source_id"), f"routes[{candidate_id}].source_id")
        if route_source not in normalized_sources:
            raise ValueError(f"route source is absent from candidate sources: {candidate_id}")
        action_kind = item.get("action_kind", "query")
        if action_kind not in {"query", "refresh"}:
            raise ValueError(f"candidate_menu[{index}].action_kind must be query or refresh")
        utility_cost = _mapping(item.get("utility_cost"), f"candidate_menu[{index}].utility_cost")
        utility = {"amount": _finite(utility_cost.get("amount"), f"candidate_menu[{index}].utility_cost.amount"),
                   "unit": _text(utility_cost.get("unit"), f"candidate_menu[{index}].utility_cost.unit"),
                   "source": _text(utility_cost.get("source"), f"candidate_menu[{index}].utility_cost.source")}
        resource_cost = _resources(item.get("declared_cost"), f"candidate_menu[{index}].declared_cost")
        availability = item.get("availability", "unknown")
        action_availability = _action_availability(availability, f"candidate_menu[{index}].availability")
        source_states = {source: visibility.get(source, "unknown") for source in normalized_sources}
        entry = {
            "candidate_id": candidate_id, "action_id": action_id, "action_kind": action_kind,
            "source_ids": list(normalized_sources), "route": deepcopy(dict(route)),
            "source_states": source_states, "availability": availability,
            "context_affinity": deepcopy(item.get("context_affinity", {})),
            "declared_priority": item.get("declared_priority", 0),
            "resource_cost": resource_cost, "utility_cost": utility,
        }
        acquisition = {
            "candidate_id": candidate_id, "action_kind": action_kind,
            "source_ids": list(normalized_sources),
            "requires_source_ids": list(item.get("requires_source_ids", [])),
            "aggregation_mode": item.get("aggregation_mode", "all_required"),
            "action_availability": action_availability,
            "resource_cost": resource_cost, "utility_cost": utility,
        }
        entries.append(entry)
        acquisition_candidates.append(acquisition)
        by_candidate[candidate_id] = entry
        by_action[action_id] = candidate_id
    body = {
        "schema_version": VERSION, "prediction_cutoff": cutoff.isoformat().replace("+00:00", "Z"),
        "match_id": match_id, "context_digest": context_digest,
        "candidate_order": [entry["candidate_id"] for entry in entries],
        "entries": entries, "acquisition_candidates": acquisition_candidates,
        "contract": {
            "selection_delegated_to_state_bound_voi": True,
            "proposal_only_input": True, "numeric_voi_estimated": False,
            "real_source_requests": 0, "real_model_calls": 0,
            "evaluator_fields_used": False,
        },
    }
    body["by_candidate"] = deepcopy(by_candidate)
    body["by_action_id"] = deepcopy(by_action)
    body["registry_digest"] = _digest(body)
    return body


def build_state_from_menu(
    actor_context: Mapping[str, Any],
    menu_report: Mapping[str, Any],
    *,
    candidate_generation_complete: bool | None = None,
) -> dict[str, Any]:
    """Build a registry and acquisition state without selecting an action."""
    registry = build_action_registry(menu_report, actor_context=actor_context)
    complete = menu_report.get("candidate_generation_complete", True) if candidate_generation_complete is None else candidate_generation_complete
    if type(complete) is not bool:
        raise ValueError("candidate_generation_complete must be boolean")
    state = prepare_acquisition_state(actor_context, registry["acquisition_candidates"], candidate_generation_complete=complete)
    return {"registry": registry, "state": state, "contract": deepcopy(registry["contract"])}


def build_execution_request(
    trace: Mapping[str, Any],
    registry: Mapping[str, Any],
    *,
    request_id: str | None = None,
    parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Route a selected candidate to a concrete provider action."""
    _verify_registry(registry)
    decision = _text(trace.get("decision"), "trace.decision")
    if decision != "acquire":
        raise ValueError("only acquire traces can create execution requests")
    candidate_id = _text(trace.get("selected_candidate_id"), "trace.selected_candidate_id")
    entry = _mapping(registry.get("by_candidate", {}).get(candidate_id), f"registry.by_candidate[{candidate_id}]")
    if trace.get("context_digest") not in {None, registry.get("context_digest")} and registry.get("context_digest") is not None:
        raise ValueError("trace context digest differs from registry")
    action_kind = trace.get("selected_action_kind")
    if action_kind is not None and action_kind != entry["action_kind"]:
        raise ValueError("trace action kind differs from registry")
    safe_parameters = _safe({} if parameters is None else parameters)
    if request_id is None:
        request_id = "request-" + _digest([trace.get("trace_digest"), candidate_id, entry["action_id"]])[:24]
    else:
        request_id = _text(request_id, "request_id")
    return {
        "schema_version": VERSION, "request_id": request_id,
        "acquisition_id": _text(trace.get("acquisition_id"), "trace.acquisition_id") if trace.get("acquisition_id") else "acq-" + _digest({
            "trace_digest": trace.get("trace_digest"),
            "candidate_id": candidate_id,
            "action_kind": entry["action_kind"],
        })[:24],
        "candidate_id": candidate_id, "action_id": entry["action_id"],
        "action_kind": entry["action_kind"], "source_id": entry["route"]["source_id"],
        "source_ids": deepcopy(entry["source_ids"]), "route": deepcopy(entry["route"]),
        "parameters": safe_parameters, "state_digest": _text(trace.get("state_digest"), "trace.state_digest"),
        "context_digest": trace.get("context_digest"),
        "trace_digest": _text(trace.get("trace_digest"), "trace.trace_digest"),
        "prediction_cutoff": _text(trace.get("prediction_cutoff"), "trace.prediction_cutoff"),
        "contract": {
            "registry_validated": True, "selection_delegated_to_state_bound_voi": True,
            "numeric_voi_estimated": False, "real_source_requests": 0, "real_model_calls": 0,
            "evaluator_fields_used": False,
        },
    }


def replan_after_execution_receipt(
    actor_context: Mapping[str, Any],
    receipt: Mapping[str, Any],
    match: Mapping[str, Any],
    *,
    menu_report: Mapping[str, Any],
    catalog: Iterable[Mapping[str, Any]],
    existing_evidence_rows: Iterable[Mapping[str, Any]] = (),
    candidate_generation_complete: bool | None = None,
) -> dict[str, Any]:
    """Apply a receipt, rebuild public context, and prepare the next menu/state."""
    current_registry = build_action_registry(menu_report, actor_context=actor_context)
    candidate_id = _text(receipt.get("candidate_id"), "receipt.candidate_id")
    if candidate_id not in current_registry["by_candidate"]:
        raise ValueError("receipt candidate is absent from registry")
    if receipt.get("request_binding_status") == "bound":
        bound = receipt.get("request_binding", {})
        selected_route = current_registry["by_candidate"][candidate_id]
        for field in ("action_id", "action_kind", "source_ids", "route"):
            if bound.get(field) != selected_route.get(field):
                raise ValueError(f"receipt selected route {field} differs from current menu")
        # Validate the menu used to select the action, before applying the
        # optional new catalog. Unselected routes may change in that catalog.
        complete_before = (menu_report.get("candidate_generation_complete", True)
                           if candidate_generation_complete is None else candidate_generation_complete)
        before_state = prepare_acquisition_state(
            actor_context, current_registry["acquisition_candidates"],
            candidate_generation_complete=complete_before)
        if receipt.get("state_digest") != before_state["state_digest"]:
            raise ValueError("receipt state differs from current menu")
    # The actor projection is the durable public memory. Materialize additional
    # rows once: callers may supply a generator, and rebuilding must not consume
    # it a second time when compiling the next menu.
    if actor_context.get("information_mode") == "historical_reconstruction":
        raise ValueError("historical reconstruction requires the replay adapter, not operational replanning")
    previous_rows = list(actor_context.get("snapshot", {}).get("source_observations", []))
    previous_rows.extend(list(existing_evidence_rows or []))
    unique_rows = []
    for row in previous_rows:
        if row not in unique_rows:
            unique_rows.append(row)
    previous_rows = unique_rows
    rebuilt = rebuild_context_after_receipt(actor_context, receipt, match, previous_rows)
    result = {
        "schema_version": VERSION, "receipt_id": receipt.get("receipt_id"),
        "executed_route": deepcopy(current_registry["by_candidate"][candidate_id]),
        "context_changed": rebuilt["context_changed"],
        "replanning_required": rebuilt["replanning_required"],
        "actor_context": deepcopy(rebuilt["actor_context"]),
        "current_registry": current_registry,
        "contract": deepcopy(current_registry["contract"]),
    }
    if receipt.get("status") != COMPLETED:
        result["replan_trace"] = {
            "status": receipt.get("status"), "previous_context_digest": actor_context.get("context_digest"),
            "next_context_digest": actor_context.get("context_digest"), "next_state_digest": None,
            "reason": "receipt_not_completed",
        }
        return result
    # Use precisely the cutoff-visible actor projection. Raw provider rows may
    # have early available_at but late ingestion and were already excluded by
    # the platform clock gate; they must not re-enter the estimator context.
    rows = list(rebuilt["actor_context"]["snapshot"]["source_observations"])
    context_report = build_source_backed_context(
        rows, target_key=actor_context["match_id"], prediction_cutoff=receipt["prediction_cutoff"]
    )
    complete = menu_report.get("candidate_generation_complete", False) if candidate_generation_complete is None else candidate_generation_complete
    if type(complete) is not bool:
        raise ValueError("candidate_generation_complete must be boolean")
    next_catalog = []
    for raw in catalog:
        item = deepcopy(dict(_mapping(raw, "catalog item")))
        previous = current_registry["by_candidate"].get(item.get("candidate_id"))
        # A fixed-cutoff round does not lose tool reachability merely because
        # its target evidence has not yet been acquired. Explicit new catalog
        # availability still takes precedence over the earlier declaration.
        route = _mapping(item.get("route", item), "catalog item route")
        source_ids = item.get("source_ids", [])
        same_route = previous is not None and (
            isinstance(source_ids, (list, tuple)) and bool(source_ids)
            and source_ids == previous["source_ids"]
            and route.get("source_id", source_ids[0]) == previous["route"]["source_id"]
            and route.get("action_id") == previous["action_id"]
            and route.get("source_version") == previous["route"].get("source_version")
        )
        if same_route and item.get("availability") is None:
            item["availability"] = previous["availability"]
        next_catalog.append(item)
    next_menu = compile_source_backed_menu(next_catalog, context_report, source_rows=rows,
                                          candidate_generation_complete=complete)
    next_registry = build_action_registry(next_menu, actor_context=rebuilt["actor_context"])
    next_state = prepare_acquisition_state(rebuilt["actor_context"], next_registry["acquisition_candidates"], candidate_generation_complete=complete)
    result.update({
        "context_report": context_report, "menu_report": next_menu,
        "registry": next_registry, "state": next_state,
        "replan_trace": {
            "status": receipt.get("status"), "previous_context_digest": actor_context.get("context_digest"),
            "next_context_digest": rebuilt.get("new_context_digest"), "next_state_digest": next_state["state_digest"],
            "candidate_order": list(next_menu["candidate_order"]), "reason": "completed_receipt_rebuilt_public_context",
        },
    })
    return result


__all__ = ["VERSION", "build_action_registry", "build_state_from_menu",
           "build_execution_request", "replan_after_execution_receipt"]
