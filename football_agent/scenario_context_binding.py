"""Bind public football scenario evidence without leaking future outcomes.

This module turns heterogeneous, pre-match observations into a small public
context object that the acquisition policy can use to rank which source to
query next. Missing and conflicting evidence stays unknown, and no result is
derived from another field. Every accepted value remains tied to provenance
and the prediction cutoff.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
import math
from typing import Any

from .contextual_candidate_generation import CONTEXT_FIELDS

VERSION = "scenario-context-binding-v1"
DEFAULT_CONTEXT_FIELDS = tuple(sorted(CONTEXT_FIELDS))
UNKNOWN_TOKENS = frozenset({
    "", "unknown", "missing", "unavailable", "not_available",
    "n/a", "na", "null", "none", "?", "未提供", "未知",
})
_FORBIDDEN = frozenset({
    "result", "outcome", "score", "scores", "goals",
    "home_score", "away_score", "home_goals", "away_goals",
    "final_score", "label", "labels", "target",
    "prediction", "probability", "loss", "decision_loss",
    "decision_loss_after", "realized_loss", "realized_gain",
    "baseline_loss", "evaluator", "sidecar", "post_state",
    "after_state", "future_result", "actual_result",
    "actual_outcome", "result_payload", "settlement_status",
    "sequence_losses", "action_returns",
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


def _scalar(value: Any, name: str) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, str):
        text = value.strip()
        if text.casefold() in UNKNOWN_TOKENS:
            return "unknown"
        return text
    if isinstance(value, bool):
        return "true" if value else "false"
    if type(value) in (int, float):
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
        return str(value)
    raise ValueError(f"{name} must be scalar")


def _field_name(value: Any, name: str) -> str:
    field = _text(str(value), name)
    if field not in CONTEXT_FIELDS:
        raise ValueError(f"unsupported context field: {field}")
    return field


def _context_payload(row: Mapping[str, Any], index: int) -> Mapping[str, Any] | None:
    direct = row.get("context")
    if direct is not None:
        return _mapping(direct, f"observations[{index}].context")
    values = row.get("values")
    if values is None:
        return None
    values = _mapping(values, f"observations[{index}].values")
    nested = values.get("context")
    if nested is not None:
        return _mapping(nested, f"observations[{index}].values.context")
    selected = {str(key): value for key, value in values.items() if str(key) in CONTEXT_FIELDS}
    return selected or None


def _normalise_required(required_fields: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    for index, raw in enumerate(required_fields):
        field = _field_name(raw, f"required_fields[{index}]")
        if field not in result:
            result.append(field)
    return tuple(result)


def bind_scenario_context(
    observations: Iterable[Mapping[str, Any]],
    *,
    target_key: str,
    prediction_cutoff: str,
    required_fields: Sequence[str] = DEFAULT_CONTEXT_FIELDS,
) -> dict[str, Any]:
    """Bind actor-visible scenario evidence at an as-of cutoff."""
    target = _text(target_key, "target_key")
    cutoff = _time(prediction_cutoff, "prediction_cutoff")
    fields = _normalise_required(required_fields)
    context = {field: "unknown" for field in fields}
    accepted: dict[str, list[dict[str, str]]] = {}
    evidence: list[dict[str, Any]] = []
    ignored: list[dict[str, Any]] = []

    for index, raw in enumerate(observations):
        row = _mapping(raw, f"observations[{index}]")
        private = _private_paths(row, f"observations[{index}]")
        if private:
            raise ValueError("scenario observation contains evaluator fields: " + ",".join(private))
        source_id = _text(row.get("source_id"), f"observations[{index}].source_id")
        available_raw = row.get("available_at", row.get("operational_available_at"))
        if available_raw is None:
            ignored.append({"source_id": source_id, "reason": "missing_availability"})
            evidence.append({"source_id": source_id, "accepted": False, "reason": "missing_availability"})
            continue
        available_at = _time(available_raw, f"observations[{index}].available_at")
        observed_raw = row.get("observed_at")
        observed_at = _time(observed_raw, f"observations[{index}].observed_at") if observed_raw is not None else None
        if available_at > cutoff or (observed_at is not None and observed_at > cutoff):
            ignored.append({"source_id": source_id, "reason": "late_for_cutoff"})
            evidence.append({"source_id": source_id, "available_at": available_at.isoformat().replace("+00:00", "Z"), "accepted": False, "reason": "late_for_cutoff"})
            continue
        payload = _context_payload(row, index)
        if payload is None:
            ignored.append({"source_id": source_id, "reason": "no_context_fields"})
            evidence.append({"source_id": source_id, "accepted": False, "reason": "no_context_fields"})
            continue
        quality = row.get("quality", "unspecified")
        quality_text = _text(str(quality), f"observations[{index}].quality") if quality is not None else "unspecified"
        source_version = row.get("source_version")
        source_version_text = str(source_version).strip() if source_version is not None else None
        content_digest = row.get("content_digest", row.get("content_hash"))
        content_digest_text = str(content_digest).strip() if content_digest is not None else None
        row_fields: list[str] = []
        for raw_field, raw_value in payload.items():
            field = _field_name(raw_field, f"observations[{index}].context field")
            value = _scalar(raw_value, f"observations[{index}].{field}")
            row_fields.append(field)
            accepted.setdefault(field, []).append({"source_id": source_id, "value": value})
        evidence.append({
            "source_id": source_id,
            "source_version": source_version_text,
            "available_at": available_at.isoformat().replace("+00:00", "Z"),
            "observed_at": observed_at.isoformat().replace("+00:00", "Z") if observed_at else None,
            "quality": quality_text,
            "content_digest": content_digest_text,
            "fields": sorted(row_fields),
            "accepted": True,
        })

    supporting: dict[str, list[str]] = {}
    conflicting: list[str] = []
    bound_fields: list[str] = []
    for field, rows in sorted(accepted.items()):
        values = {row["value"] for row in rows}
        supporting[field] = sorted({row["source_id"] for row in rows})
        bound_fields.append(field)
        if len(values) == 1:
            context[field] = next(iter(values))
        else:
            context[field] = "unknown"
            conflicting.append(field)

    unknown_fields = sorted(field for field, value in context.items() if value == "unknown")
    return {
        "schema_version": VERSION,
        "target_key": target,
        "prediction_cutoff": cutoff.isoformat().replace("+00:00", "Z"),
        "context": context,
        "bound_fields": sorted(bound_fields),
        "unknown_fields": unknown_fields,
        "supporting_source_ids": supporting,
        "conflicting_fields": sorted(conflicting),
        "ignored_observations": ignored,
        "evidence": evidence,
        "actor_visible": True,
        "evaluator_fields_used": False,
        "numeric_voi_estimated": False,
        "context_semantics": "observed_only; no cross-field inference",
    }


def apply_scenario_context(episode: Mapping[str, Any], binding: Mapping[str, Any]) -> dict[str, Any]:
    """Attach a validated public context binding to a replay episode."""
    source = _mapping(episode, "episode")
    public = _mapping(source.get("public"), "episode.public")
    private = _private_paths(public, "episode.public")
    if private:
        raise ValueError("episode.public contains evaluator fields: " + ",".join(private))
    bind = _mapping(binding, "binding")
    if bind.get("schema_version") != VERSION:
        raise ValueError("binding schema_version mismatch")
    context = _mapping(bind.get("context"), "binding.context")
    private = _private_paths(context, "binding.context")
    if private:
        raise ValueError("binding.context contains evaluator fields: " + ",".join(private))
    bound_fields = bind.get("bound_fields", [])
    if not isinstance(bound_fields, list):
        raise ValueError("binding.bound_fields must be a list")
    cloned = deepcopy(dict(source))
    cloned_public = dict(public)
    existing = cloned_public.get("context", {})
    existing = {} if existing is None else dict(_mapping(existing, "episode.public.context"))
    for field in bound_fields:
        name = _field_name(field, "binding.bound_fields[]")
        existing[name] = _scalar(context.get(name), f"binding.context.{name}")
    cloned_public["context"] = existing
    cloned_public["context_evidence"] = deepcopy(list(bind.get("evidence", [])))
    cloned_public["context_binding"] = {
        "schema_version": VERSION,
        "target_key": _text(bind.get("target_key"), "binding.target_key"),
        "prediction_cutoff": _time(bind.get("prediction_cutoff"), "binding.prediction_cutoff").isoformat().replace("+00:00", "Z"),
        "bound_fields": sorted(bound_fields),
        "unknown_fields": list(bind.get("unknown_fields", [])),
        "supporting_source_ids": deepcopy(dict(bind.get("supporting_source_ids", {}))),
        "conflicting_fields": list(bind.get("conflicting_fields", [])),
        "evaluator_fields_used": False,
    }
    cloned["public"] = cloned_public
    return cloned


__all__ = ["VERSION", "CONTEXT_FIELDS", "DEFAULT_CONTEXT_FIELDS",
           "bind_scenario_context", "apply_scenario_context"]
