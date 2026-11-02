"""Adapt platform AcquisitionResult records into candidate evidence rows.

The adapter consumes the public platform contract and emits metadata rows for
evidence_binding. It preserves quality, timing, source identity, references
and dependency keys while keeping platform values outside the policy audit.
"""
from __future__ import annotations

from collections import Counter
import hashlib
from typing import Any, Iterable, Mapping


VERSION = "platform-acquisition-adapter-v1"
FAILURE_STATUSES = frozenset({"FAILED", "TIMEOUT", "BAD_DATA", "UNAVAILABLE", "NO_DATA", "BUDGET_EXCEEDED"})
QUALITY_ORDER = {
    "PROVIDER_ERROR": 6, "INVALID": 5, "SOURCE_MISSING": 5, "STALE": 4,
    "CONFLICT": 3, "PARTIAL": 2, "DUPLICATE": 2, "NO_CHANGE": 1,
    "UNVERIFIED": 1, "VALIDATED": 0,
}


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return dict(value)
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _quality(group: Mapping[str, Any], status: str) -> str:
    quality = str(group.get("quality", group.get("quality_status", ""))).upper()
    if status in FAILURE_STATUSES:
        return "PROVIDER_ERROR"
    if quality in QUALITY_ORDER:
        return quality
    return "UNVERIFIED"


def acquisition_result_to_evidence_rows(
    result: Mapping[str, Any] | Any,
    *,
    action_to_candidate: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Convert one public AcquisitionResult into one candidate metadata row."""
    payload = _dump(result)
    if not isinstance(payload, Mapping):
        raise ValueError("AcquisitionResult must be a mapping or model with model_dump")
    action_id = _text(payload.get("action_id"), "action_id")
    if action_id not in action_to_candidate:
        raise ValueError(f"Action is absent from candidate map: {action_id}")
    candidate_id = _text(action_to_candidate[action_id], "candidate_id")
    status = str(payload.get("status", "FAILED")).upper()
    groups_raw = payload.get("revealed_groups", {}) or {}
    if not isinstance(groups_raw, Mapping):
        raise ValueError("revealed_groups must be a mapping")
    groups = [_dump(group) for group in groups_raw.values()]
    groups = [group for group in groups if isinstance(group, Mapping)]
    if groups:
        qualities = [_quality(group, status) for group in groups]
        primary = dict(groups[0])
        refs = list(payload.get("evidence_refs", []) or [])
        for group in groups:
            refs.extend(group.get("evidence_refs", []) or [])
        refs = list(dict.fromkeys(str(ref) for ref in refs))
        dependencies = [str(group.get("dependency_key", "")) for group in groups if group.get("dependency_key")]
        sources = [str(group.get("source_id") or group.get("source") or "") for group in groups]
        dependency = dependencies[0] if len(set(dependencies)) == 1 and dependencies else f"action:{action_id}"
        source = sources[0] if len(set(sources)) == 1 and sources else f"platform:{action_id}"
        content = [group.get("content_hash", "") for group in groups if group.get("content_hash")]
        primary.update({
            "candidate_id": candidate_id,
            "source_id": source,
            "dependency_key": dependency,
            "quality": max(qualities, key=lambda item: QUALITY_ORDER.get(item, 1)),
            "evidence_refs": refs,
            "content_hash": hashlib.sha256("|".join(sorted(content)).encode("utf-8")).hexdigest() if len(content) > 1
                            else (content[0] if content else ""),
            "status": status,
            "group_ids": [str(group.get("group_id", "")) for group in groups],
        })
        if status in FAILURE_STATUSES:
            primary["error_class"] = payload.get("reason") or status.lower()
        return [primary]
    quality = "PROVIDER_ERROR" if status in FAILURE_STATUSES else "SOURCE_MISSING"
    return [{
        "candidate_id": candidate_id,
        "source_id": f"platform:{action_id}",
        "dependency_key": f"action:{action_id}",
        "quality": quality,
        "status": status,
        "error_class": payload.get("reason") or status.lower(),
        "evidence_refs": list(payload.get("evidence_refs", []) or []),
        "content_hash": "",
    }]


def acquisition_results_to_evidence_rows(
    results: Iterable[Mapping[str, Any] | Any],
    *,
    action_to_candidate: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Convert a snapshot of distinct acquisition results with one row per candidate."""
    rows: list[dict[str, Any]] = []
    seen_candidates: set[str] = set()
    for result in results:
        converted = acquisition_result_to_evidence_rows(result, action_to_candidate=action_to_candidate)
        for row in converted:
            candidate_id = row["candidate_id"]
            if candidate_id in seen_candidates:
                raise ValueError(f"Multiple acquisition results target candidate: {candidate_id}")
            seen_candidates.add(candidate_id)
            rows.append(row)
    return rows


__all__ = [
    "VERSION", "acquisition_result_to_evidence_rows", "acquisition_results_to_evidence_rows",
]
