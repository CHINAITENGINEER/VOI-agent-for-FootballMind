"""Normalize collector observations into the public scenario context contract.

Collectors often use short or source-specific field names.  This adapter only
performs explicit schema mapping; it never infers one football fact from
another.  The resulting rows can be passed to scenario_context_binding, which
applies the cutoff, conflict, and evaluator-isolation rules.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from typing import Any

from .contextual_candidate_generation import CONTEXT_FIELDS
from .scenario_context_binding import (
    DEFAULT_CONTEXT_FIELDS,
    apply_scenario_context,
    bind_scenario_context,
)

VERSION = "source-context-adapter-v1"
_ALIASES = {
    field.casefold(): field for field in CONTEXT_FIELDS
}
_ALIASES.update({
    "qualification": "qualification_state",
    "qualification_status": "qualification_state",
    "tactical_style": "tactical_layout",
    "tactics": "tactical_layout",
    "rotation": "rotation_risk",
    "rotation_status": "rotation_risk",
    "weather": "weather_state",
    "lineup": "lineup_status",
    "importance": "match_importance",
    "stage": "competition_stage",
    "motivation": "motivation_state",
    "schedule": "schedule_congestion",
    "uncertainty": "dominant_uncertainty",
})
_FORBIDDEN = frozenset({
    "result", "outcome", "score", "scores", "goals",
    "home_score", "away_score", "home_goals", "away_goals",
    "final_score", "label", "labels", "target",
    "prediction", "probability", "loss", "decision_loss",
    "realized_loss", "realized_gain", "baseline_loss",
    "evaluator", "sidecar", "post_state", "after_state",
    "future_result", "actual_result", "actual_outcome",
    "settlement_status", "sequence_losses", "action_returns",
})


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _private_paths(value: Any, path: str) -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            if key.casefold() in _FORBIDDEN:
                found.append(f"{path}.{key}")
            found.extend(_private_paths(child, f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(_private_paths(child, f"{path}[{index}]"))
    return found


def _source_id(row: Mapping[str, Any], index: int) -> str | None:
    raw = row.get("source_id", row.get("evidence_id", row.get("source")))
    if not isinstance(raw, str) or not raw.strip():
        return None
    return raw.strip()


def _raw_payload(row: Mapping[str, Any], index: int) -> Mapping[str, Any]:
    direct = row.get("context")
    if direct is not None:
        return _mapping(direct, f"rows[{index}].context")
    raw_values = row.get("values", row.get("payload", {}))
    raw_values = _mapping(raw_values, f"rows[{index}].values")
    nested = raw_values.get("context")
    if nested is not None:
        return _mapping(nested, f"rows[{index}].values.context")
    return raw_values


def _canonical_values(payload: Mapping[str, Any], index: int) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    values: dict[str, Any] = {}
    issues: list[dict[str, Any]] = []
    trace: list[str] = []
    aliases_seen: dict[str, list[tuple[str, Any]]] = {}
    for raw_key, raw_value in payload.items():
        key = str(raw_key).strip()
        canonical = _ALIASES.get(key.casefold())
        if canonical is None:
            issues.append({"field": key, "reason": "unsupported_context_field"})
            continue
        aliases_seen.setdefault(canonical, []).append((key, raw_value))
        trace.append(f"{key}->{canonical}")
    for canonical, entries in aliases_seen.items():
        distinct = {repr(value) for _, value in entries}
        if len(distinct) > 1:
            issues.append({"field": canonical, "reason": "field_alias_conflict", "aliases": [key for key, _ in entries]})
            continue
        values[canonical] = entries[0][1]
    return values, issues, sorted(trace)


def normalize_source_context_observations(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Map collector rows to the binding input shape without inference."""
    normalized: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            issues.append({"index": index, "reason": "row_not_mapping"})
            continue
        private = _private_paths(raw, f"rows[{index}]")
        if private:
            raise ValueError("collector row contains evaluator fields: " + ",".join(private))
        source_id = _source_id(raw, index)
        if source_id is None:
            issues.append({"index": index, "reason": "missing_source_id"})
            continue
        payload = _raw_payload(raw, index)
        values, row_issues, trace = _canonical_values(payload, index)
        for issue in row_issues:
            issues.append({"index": index, "source_id": source_id, **issue})
        projected: dict[str, Any] = {"source_id": source_id, "values": values}
        for key in ("source_version", "observed_at", "available_at", "operational_available_at", "quality", "content_digest", "content_hash", "digest"):
            if raw.get(key) is not None:
                projected[key] = deepcopy(raw[key])
        if trace:
            projected["normalization_trace"] = trace
        normalized.append(projected)
    return normalized, issues


def build_source_backed_context(
    rows: Iterable[Mapping[str, Any]],
    *,
    target_key: str,
    prediction_cutoff: str,
    required_fields: Sequence[str] = DEFAULT_CONTEXT_FIELDS,
) -> dict[str, Any]:
    """Normalize source rows and bind them into actor-visible context."""
    normalized, issues = normalize_source_context_observations(rows)
    binding = bind_scenario_context(
        normalized,
        target_key=target_key,
        prediction_cutoff=prediction_cutoff,
        required_fields=required_fields,
    )
    return {
        "schema_version": VERSION,
        "target_key": binding["target_key"],
        "prediction_cutoff": binding["prediction_cutoff"],
        "normalized_observations": normalized,
        "normalization_issues": issues,
        "binding": binding,
        "contract": {
            "explicit_alias_mapping_only": True,
            "cross_field_inference": False,
            "evaluator_fields_used": False,
            "numeric_voi_estimated": False,
            "real_source_requests": 0,
            "real_model_calls": 0,
        },
    }


def apply_source_backed_context(episode: Mapping[str, Any], report: Mapping[str, Any]) -> dict[str, Any]:
    """Attach a source-backed binding to a replay episode."""
    if not isinstance(report, Mapping) or report.get("schema_version") != VERSION:
        raise ValueError("source context report schema_version mismatch")
    return apply_scenario_context(episode, _mapping(report.get("binding"), "report.binding"))


__all__ = ["VERSION", "normalize_source_context_observations",
           "build_source_backed_context", "apply_source_backed_context"]
