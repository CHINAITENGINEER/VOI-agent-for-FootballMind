"""Bind platform evidence metadata to semantic policy audits.

The platform already records operational timestamps, quality, evidence refs and
dependency keys. This module translates that public contract into the small
availability vocabulary consumed by policy_audit. It keeps availability
separate from semantic resolution: a valid response can still leave the state
unresolved, while a late or failed response remains visible as an explicit gap.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Mapping

from .types import utc
from .state_query_contract import StateQueryBoard


VERSION = "platform-evidence-binding-v1"
QUALITIES = frozenset({
    "VALIDATED", "PARTIAL", "SOURCE_MISSING", "STALE", "INVALID", "UNVERIFIED",
    "DUPLICATE", "CONFLICT", "PROVIDER_ERROR", "NO_CHANGE",
})
AVAILABILITY = frozenset({"available", "late", "failed", "duplicate"})
FAILED_QUALITIES = frozenset({"SOURCE_MISSING", "STALE", "INVALID", "PROVIDER_ERROR", "CONFLICT"})


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _quality(value: Any) -> str:
    value = _text(value, "quality").upper()
    if value not in QUALITIES:
        raise ValueError(f"Unsupported evidence quality: {value}")
    return value


@dataclass(frozen=True)
class EvidenceBinding:
    candidate_id: str
    source_id: str
    dependency_key: str
    quality: str
    observed_at: str | None
    operational_available_at: str | None
    expires_at: str | None
    evidence_refs: tuple[str, ...]
    content_hash: str
    availability: str
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _text(self.candidate_id, "candidate_id"))
        object.__setattr__(self, "source_id", _text(self.source_id, "source_id"))
        object.__setattr__(self, "dependency_key", _text(self.dependency_key, "dependency_key"))
        object.__setattr__(self, "quality", _quality(self.quality))
        if self.availability not in AVAILABILITY:
            raise ValueError(f"Unsupported evidence availability: {self.availability}")
        object.__setattr__(self, "evidence_refs", tuple(_text(ref, "evidence_ref") for ref in self.evidence_refs))
        if not isinstance(self.content_hash, str):
            raise ValueError("content_hash must be a string")
        object.__setattr__(self, "reason", _text(self.reason, "reason"))
        for field in ("observed_at", "operational_available_at", "expires_at"):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, utc(value).isoformat())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": VERSION,
            "candidate_id": self.candidate_id,
            "source_id": self.source_id,
            "dependency_key": self.dependency_key,
            "quality": self.quality,
            "observed_at": self.observed_at,
            "operational_available_at": self.operational_available_at,
            "expires_at": self.expires_at,
            "evidence_refs": list(self.evidence_refs),
            "content_hash": self.content_hash,
            "availability": self.availability,
            "reason": self.reason,
        }


def _unwrap(row: Mapping[str, Any]) -> Mapping[str, Any]:
    if "group" in row:
        group = row["group"]
        if not isinstance(group, Mapping):
            raise ValueError("group must be a mapping")
        row = {**group, **{key: value for key, value in row.items() if key != "group"}}
    times = row.get("times")
    if isinstance(times, Mapping):
        return {**row, **times}
    return row


def _raw_time(row: Mapping[str, Any], *names: str) -> str | None:
    for name in names:
        if row.get(name) is not None:
            return utc(row[name]).isoformat()
    return None


def _base_binding(candidate_id: str, raw: Mapping[str, Any], cutoff: datetime) -> EvidenceBinding:
    row = _unwrap(raw)
    source_id = row.get("source_id") or row.get("source") or "platform-unknown"
    dependency_key = row.get("dependency_key") or row.get("group_id") or "dependency-unknown"
    quality = _quality(row.get("quality", row.get("quality_status", "UNVERIFIED")))
    observed = _raw_time(row, "observed_at", "times_observed_at")
    available = _raw_time(row, "operational_available_at", "available_at", "feature_available_at")
    expires = _raw_time(row, "expires_at")
    error = row.get("error_class") or row.get("error") or row.get("failure_class")
    declared_availability = str(row.get("availability", "")).lower()
    if declared_availability == "duplicate":
        availability, reason = "duplicate", "declared_duplicate_evidence"
    elif declared_availability == "failed":
        availability, reason = "failed", "declared_failed_evidence"
    elif declared_availability == "late":
        availability, reason = "late", "declared_late_evidence"
    elif error or row.get("status") in {"FAILED", "TIMEOUT", "BAD_DATA", "UNAVAILABLE", "NO_DATA"}:
        availability, reason = "failed", "transport_or_provider_failure"
    elif quality == "DUPLICATE":
        availability, reason = "duplicate", "declared_duplicate_evidence"
    elif quality in FAILED_QUALITIES:
        availability, reason = "failed", quality.lower()
    elif observed is None or available is None:
        availability, reason = "failed", "missing_operational_timestamps"
    elif utc(available) > cutoff or utc(observed) > cutoff:
        availability, reason = "late", "not_available_at_prediction_cutoff"
    elif expires is not None and utc(expires) < cutoff:
        quality = "STALE"
        availability, reason = "failed", "evidence_expired_before_prediction_cutoff"
    else:
        availability, reason = "available", "available_before_prediction_cutoff"
    refs = row.get("evidence_refs", row.get("evidence_ids", ())) or ()
    if isinstance(refs, (str, bytes)):
        raise ValueError("evidence_refs must be a sequence")
    return EvidenceBinding(
        candidate_id=candidate_id, source_id=source_id, dependency_key=dependency_key,
        quality=quality, observed_at=observed, operational_available_at=available,
        expires_at=expires, evidence_refs=tuple(refs), content_hash=str(row.get("content_hash", "")),
        availability=availability, reason=reason,
    )


@dataclass(frozen=True)
class EvidenceBindingResult:
    cutoff: str
    bindings: tuple[EvidenceBinding, ...]
    availability_by_candidate: Mapping[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": VERSION,
            "cutoff": self.cutoff,
            "bindings": [binding.as_dict() for binding in self.bindings],
            "availability_by_candidate": dict(self.availability_by_candidate),
            "dependency_groups": {
                dependency: [binding.candidate_id for binding in self.bindings if binding.dependency_key == dependency]
                for dependency in sorted({binding.dependency_key for binding in self.bindings})
            },
        }


def bind_candidate_evidence(
    board: StateQueryBoard,
    rows: Iterable[Mapping[str, Any]],
    *,
    cutoff: str | datetime,
) -> EvidenceBindingResult:
    """Translate platform evidence rows into policy-audit availability."""
    if not isinstance(board, StateQueryBoard):
        raise ValueError("board must be a StateQueryBoard")
    cutoff_dt = utc(cutoff)
    candidate_ids = [candidate.candidate_id for candidate in board.candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("State query candidate IDs must be unique")
    rows = tuple(rows)
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Evidence rows must be mappings")
        candidate_id = _text(row.get("candidate_id"), "candidate_id")
        if candidate_id not in candidate_ids:
            raise ValueError(f"Evidence candidate is outside the board: {candidate_id}")
        if candidate_id in by_id:
            raise ValueError(f"Duplicate evidence row for candidate: {candidate_id}")
        by_id[candidate_id] = row
    bindings: list[EvidenceBinding] = []
    for candidate_id in candidate_ids:
        raw = by_id.get(candidate_id)
        if raw is None:
            raw = {"candidate_id": candidate_id, "source_id": "platform-missing",
                   "dependency_key": f"missing:{candidate_id}", "quality": "SOURCE_MISSING"}
        bindings.append(_base_binding(candidate_id, raw, cutoff_dt))

    seen_dependency: set[str] = set()
    normalized: list[EvidenceBinding] = []
    for binding in bindings:
        if binding.availability == "available" and binding.dependency_key in seen_dependency:
            binding = EvidenceBinding(**{**binding.__dict__, "availability": "duplicate",
                                         "reason": "dependency_key_already_bound"})
        if binding.availability == "available":
            seen_dependency.add(binding.dependency_key)
        normalized.append(binding)
    return EvidenceBindingResult(
        cutoff=cutoff_dt.isoformat(), bindings=tuple(normalized),
        availability_by_candidate={binding.candidate_id: binding.availability for binding in normalized},
    )


__all__ = ["VERSION", "EvidenceBinding", "EvidenceBindingResult", "bind_candidate_evidence"]
