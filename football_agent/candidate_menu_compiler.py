"""Compile explicit query routes into an actor-visible candidate menu."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
import math
from typing import Any

from .contextual_candidate_generation import generate_contextual_candidates
from .scenario_context_binding import CONTEXT_FIELDS

VERSION = "candidate-menu-compiler-v1"


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
        raise ValueError(f"{name} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _finite(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _declared_cost(value: Any, name: str) -> dict[str, int]:
    raw = _mapping(value, name)
    result: dict[str, int] = {}
    for key, amount in raw.items():
        field = _text(str(key), f"{name}.key")
        if isinstance(amount, bool) or type(amount) is not int or amount < 0:
            raise ValueError(f"{name}.{field} must be a nonnegative integer")
        result[field] = amount
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _context_affinity(value: Any, index: int) -> dict[str, list[str]]:
    if value is None:
        return {}
    raw = _mapping(value, f"catalog[{index}].context_affinity")
    result: dict[str, list[str]] = {}
    for key, values in raw.items():
        field = _text(str(key), f"catalog[{index}].context_affinity.key")
        if field not in CONTEXT_FIELDS:
            raise ValueError(f"unsupported context affinity field: {field}")
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"catalog[{index}].context_affinity.{field} must be a nonempty list")
        result[field] = [_text(str(item), f"catalog[{index}].context_affinity.{field}[]") for item in values]
    return result


def _source_visibility(source_rows: Iterable[Mapping[str, Any]] | None, cutoff: datetime) -> dict[str, str]:
    if source_rows is None:
        return {}
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for index, raw in enumerate(source_rows):
        row = _mapping(raw, f"source_rows[{index}]")
        source_id = _text(row.get("source_id"), f"source_rows[{index}].source_id")
        grouped.setdefault(source_id, []).append(row)
    result: dict[str, str] = {}
    for source_id, rows in grouped.items():
        if len(rows) != 1:
            result[source_id] = "unknown"
            continue
        row = rows[0]
        available = row.get("available_at", row.get("operational_available_at"))
        if available is None:
            result[source_id] = "unknown"
            continue
        available_at = _time(available, f"source_rows.{source_id}.available_at")
        result[source_id] = "available" if available_at <= cutoff else "unknown"
    return result


def _binding_context(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(value.get("binding"), Mapping):
        value = value["binding"]
    context = _mapping(value.get("context"), "context_report.context")
    if value.get("evaluator_fields_used") is True:
        raise ValueError("context report uses evaluator fields")
    return context


def compile_candidate_menu(
    catalog: Iterable[Mapping[str, Any]],
    *,
    context_report: Mapping[str, Any],
    prediction_cutoff: str,
    source_rows: Iterable[Mapping[str, Any]] | None = None,
    candidate_generation_complete: bool = True,
) -> dict[str, Any]:
    """Compile explicit routes and a public-context proposal order."""
    cutoff = _time(prediction_cutoff, "prediction_cutoff")
    if type(candidate_generation_complete) is not bool:
        raise ValueError("candidate_generation_complete must be boolean")
    context = _binding_context(_mapping(context_report, "context_report"))
    visibility = _source_visibility(source_rows, cutoff)
    menu: list[dict[str, Any]] = []
    routes: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for index, raw in enumerate(catalog):
        item = _mapping(raw, f"catalog[{index}]")
        candidate_id = _text(item.get("candidate_id"), f"catalog[{index}].candidate_id")
        if candidate_id in seen:
            raise ValueError(f"duplicate candidate_id: {candidate_id}")
        seen.add(candidate_id)
        source_ids = item.get("source_ids")
        if not isinstance(source_ids, (list, tuple)) or not source_ids or any(not isinstance(source, str) or not source.strip() for source in source_ids):
            raise ValueError(f"catalog[{index}].source_ids must be a nonempty list")
        source_ids = list(dict.fromkeys(source.strip() for source in source_ids))
        # Topic names (injury/weather/etc.) do not determine the acquisition
        # operation. Preserve its independently declared execution contract.
        action_kind = item.get("action_kind", "query")
        if action_kind not in {"query", "refresh"}:
            raise ValueError(f"catalog[{index}].action_kind must be query or refresh")
        prerequisites = item.get("requires_source_ids", [])
        if not isinstance(prerequisites, (list, tuple)):
            raise ValueError(f"catalog[{index}].requires_source_ids must be a sequence")
        prerequisites = [_text(value, f"catalog[{index}].requires_source_ids[]") for value in prerequisites]
        if len(set(prerequisites)) != len(prerequisites):
            raise ValueError(f"catalog[{index}].requires_source_ids must be unique")
        aggregation_mode = item.get("aggregation_mode", "all_required")
        if aggregation_mode not in {"all_required", "any_usable"}:
            raise ValueError(f"catalog[{index}].aggregation_mode is invalid")
        route = _mapping(item.get("route", item), f"catalog[{index}].route")
        route_source = _text(route.get("source_id", source_ids[0]), f"catalog[{index}].route.source_id")
        action_id = _text(route.get("action_id"), f"catalog[{index}].route.action_id")
        cost = _finite(route.get("cost"), f"catalog[{index}].route.cost", minimum=0.0)
        utility_raw = _mapping(item.get("utility_cost"), f"catalog[{index}].utility_cost")
        unit = _text(route.get("unit", utility_raw.get("unit")), f"catalog[{index}].route.unit")
        declared_cost = _declared_cost(item.get("declared_cost"), f"catalog[{index}].declared_cost")
        utility_amount = _finite(utility_raw.get("amount"), f"catalog[{index}].utility_cost.amount", minimum=0.0)
        utility_unit = _text(utility_raw.get("unit"), f"catalog[{index}].utility_cost.unit")
        if utility_unit != unit:
            raise ValueError(f"catalog[{index}] route and utility cost units differ")
        source_states = [visibility.get(source_id, "unknown") for source_id in source_ids]
        availability = item.get("availability")
        if availability is None:
            availability = "available" if source_states and all(state == "available" for state in source_states) else "unknown"
        if availability not in {"available", "unavailable", "unknown", "stale", "duplicate"}:
            raise ValueError(f"catalog[{index}].availability is invalid")
        candidate = {
            "candidate_id": candidate_id,
            "kind": _text(item.get("kind", "query"), f"catalog[{index}].kind"),
            "action_kind": action_kind,
            "source_ids": source_ids,
            "requires_source_ids": prerequisites,
            "aggregation_mode": aggregation_mode,
            "action_id": action_id,
            "declared_cost": declared_cost,
            "utility_cost": {"amount": utility_amount, "unit": utility_unit, "source": VERSION},
            "availability": availability,
            "context_affinity": _context_affinity(item.get("context_affinity"), index),
            "declared_priority": _finite(item.get("declared_priority", 0), f"catalog[{index}].declared_priority"),
        }
        menu.append(candidate)
        routes[candidate_id] = {
            "source_id": route_source,
            "source_version": None if route.get("source_version") is None else _text(route.get("source_version"), f"catalog[{index}].route.source_version"),
            "action_id": action_id,
            "cost": cost,
            "unit": unit,
        }
    if not menu:
        raise ValueError("catalog must contain at least one candidate")
    proposal = generate_contextual_candidates({"episode_id": context_report.get("target_key"), "public": {"context": context, "candidate_menu": menu}})
    by_id = {item["candidate_id"]: item for item in menu}
    ordered_menu = [by_id[candidate_id] for candidate_id in proposal["candidate_order"]]
    return {
        "schema_version": VERSION,
        "prediction_cutoff": cutoff.isoformat().replace("+00:00", "Z"),
        "context": deepcopy(dict(context)),
        "candidate_menu": ordered_menu,
        "candidate_order": list(proposal["candidate_order"]),
        "candidate_generation_complete": candidate_generation_complete,
        "routes": routes,
        "source_visibility": visibility,
        "proposal_trace": proposal,
        "contract": {
            "proposal_only": True,
            "numeric_voi_estimated": False,
            "state_bound_selection_required": True,
            "real_source_requests": 0,
            "real_model_calls": 0,
            "evaluator_fields_used": False,
        },
    }


def compile_source_backed_menu(catalog: Iterable[Mapping[str, Any]], context_report: Mapping[str, Any], *, source_rows: Iterable[Mapping[str, Any]] | None = None, candidate_generation_complete: bool = True) -> dict[str, Any]:
    """Convenience wrapper for a source_context_adapter report."""
    report = _mapping(context_report, "context_report")
    cutoff = _text(report.get("prediction_cutoff"), "context_report.prediction_cutoff")
    return compile_candidate_menu(catalog, context_report=report, prediction_cutoff=cutoff, source_rows=source_rows,
                                  candidate_generation_complete=candidate_generation_complete)


__all__ = ["VERSION", "compile_candidate_menu", "compile_source_backed_menu"]
