"""Explicit action-level aggregation for multi-request information actions."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable

from .audit import json_bytes
from .evidence_ledger import EvidenceLedgerEntry
from .types import utc
from .voi_evidence_bridge import VoIEvidenceLink
from .voi_trace import VoITrace


VERSION = "action-evidence-aggregate-v1"
MODES = frozenset({"all_required", "any_usable"})
GOOD_QUALITIES = frozenset({"VALIDATED", "PARTIAL"})


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


@dataclass(frozen=True)
class ActionEvidenceAggregate:
    aggregate_id: str
    episode_id: str
    target_key: str
    action_id: str
    request_id: str
    candidate_id: str
    mode: str
    entry_ids: tuple[str, ...]
    dependency_keys: tuple[str, ...]
    source_ids: tuple[str, ...]
    quality: str
    availability: str
    operational_available_at: str | None
    prediction_cutoff: str
    captured_at: str
    missing_dependency_keys: tuple[str, ...] = ()
    dependency_statuses: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        for field in ("aggregate_id", "episode_id", "target_key", "action_id", "request_id", "candidate_id", "quality", "availability"):
            object.__setattr__(self, field, _text(getattr(self, field), field))
        if self.mode not in MODES:
            raise ValueError("Unsupported action aggregation mode")
        if not self.entry_ids or len(set(self.entry_ids)) != len(self.entry_ids):
            raise ValueError("Aggregate requires unique evidence entry IDs")
        if not self.dependency_keys:
            raise ValueError("Aggregate requires at least one dependency key")
        object.__setattr__(self, "entry_ids", tuple(_text(value, "entry_id") for value in self.entry_ids))
        dependencies = tuple(_text(value, "dependency_key") for value in self.dependency_keys)
        object.__setattr__(self, "dependency_keys", tuple(dict.fromkeys(dependencies)))
        object.__setattr__(self, "source_ids", tuple(_text(value, "source_id") for value in self.source_ids))
        missing = tuple(_text(value, "missing_dependency_key") for value in self.missing_dependency_keys)
        object.__setattr__(self, "missing_dependency_keys", tuple(dict.fromkeys(missing)))
        statuses = tuple((_text(key, "dependency_key"), _text(status, "dependency_status"))
                         for key, status in self.dependency_statuses)
        object.__setattr__(self, "dependency_statuses", tuple(sorted(dict(statuses).items())))
        object.__setattr__(self, "prediction_cutoff", utc(self.prediction_cutoff).isoformat())
        object.__setattr__(self, "captured_at", utc(self.captured_at).isoformat())
        if self.operational_available_at is not None:
            object.__setattr__(self, "operational_available_at", utc(self.operational_available_at).isoformat())

    def as_dict(self):
        return {
            "schema_version": VERSION, "aggregate_id": self.aggregate_id,
            "episode_id": self.episode_id, "target_key": self.target_key,
            "action_id": self.action_id, "request_id": self.request_id,
            "candidate_id": self.candidate_id, "mode": self.mode,
            "entry_ids": list(self.entry_ids), "dependency_keys": list(self.dependency_keys),
            "source_ids": list(self.source_ids), "quality": self.quality,
            "availability": self.availability,
            "operational_available_at": self.operational_available_at,
            "prediction_cutoff": self.prediction_cutoff, "captured_at": self.captured_at,
            "missing_dependency_keys": list(self.missing_dependency_keys),
            "dependency_statuses": [{"dependency_key": key, "status": status}
                                    for key, status in self.dependency_statuses],
        }


def _usable(entry: EvidenceLedgerEntry, cutoff):
    return (entry.availability == "available" and entry.quality in GOOD_QUALITIES
            and utc(entry.captured_at) <= cutoff
            and entry.operational_available_at is not None
            and utc(entry.operational_available_at) <= cutoff)


def aggregate_entries(
    entries: Iterable[EvidenceLedgerEntry], *, mode: str, cutoff: str,
    required_dependency_keys: Iterable[str] | None = None,
) -> ActionEvidenceAggregate:
    materialized = tuple(entries)
    if not materialized:
        raise ValueError("At least one evidence entry is required")
    if mode not in MODES:
        raise ValueError("Unsupported action aggregation mode")
    first = materialized[0]
    if any((entry.episode_id, entry.target_key, entry.action_id, entry.request_id, entry.candidate_id) !=
           (first.episode_id, first.target_key, first.action_id, first.request_id, first.candidate_id)
           for entry in materialized):
        raise ValueError("Action aggregate entries must share episode/action/request/candidate identity")
    cutoff_value = utc(cutoff)
    declared = None if required_dependency_keys is None else tuple(dict.fromkeys(_text(key, "required_dependency_key") for key in required_dependency_keys))
    if declared is not None and not declared:
        raise ValueError("required_dependency_keys must not be empty")
    groups: dict[str, list[tuple[int, EvidenceLedgerEntry]]] = {}
    for index, entry in enumerate(materialized):
        groups.setdefault(entry.dependency_key, []).append((index, entry))

    # A dependency can have several immutable attempts. Resolve retries before
    # evaluating the action: one timely usable attempt satisfies that dependency.
    dependency_states: dict[str, tuple[str, int, EvidenceLedgerEntry]] = {}
    for dependency, candidates in groups.items():
        usable_candidates = [(index, entry) for index, entry in candidates if _usable(entry, cutoff_value)]
        late_candidates = [(index, entry) for index, entry in candidates
                           if entry.availability == "late" or (entry.operational_available_at and utc(entry.operational_available_at) > cutoff_value)]
        if usable_candidates:
            index, winner = max(usable_candidates, key=lambda pair: utc(pair[1].captured_at))
            dependency_states[dependency] = ("available", index, winner)
        elif late_candidates:
            index, winner = min(late_candidates, key=lambda pair: utc(pair[1].operational_available_at))
            dependency_states[dependency] = ("late", index, winner)
        else:
            index, winner = max(candidates, key=lambda pair: utc(pair[1].captured_at))
            dependency_states[dependency] = ("failed", index, winner)

    declared_set = set(declared) if declared is not None else set(groups)
    missing_dependency_keys = tuple(sorted(declared_set - set(groups))) if mode == "all_required" and declared is not None else ()
    required_states = [dependency_states[key][0] for key in sorted(declared_set & set(dependency_states))]
    selected_indexes: list[int]
    if mode == "all_required":
        if missing_dependency_keys:
            availability, quality, operational = "failed", "SOURCE_MISSING", None
            selected_indexes = [state[1] for state in dependency_states.values()]
        elif required_states and all(state == "available" for state in required_states):
            availability = "available"
            quality = "VALIDATED" if all(dependency_states[key][2].quality == "VALIDATED" for key in declared_set) else "PARTIAL"
            operational = max(utc(dependency_states[key][2].operational_available_at) for key in declared_set).isoformat()
            selected_indexes = [dependency_states[key][1] for key in sorted(declared_set)]
        elif any(state == "failed" for state in required_states):
            availability, quality, operational = "failed", "PROVIDER_ERROR", None
            selected_indexes = [dependency_states[key][1] for key in sorted(declared_set)]
        else:
            availability, quality = "late", "PARTIAL"
            operational = max(utc(dependency_states[key][2].operational_available_at) for key in declared_set).isoformat()
            selected_indexes = [dependency_states[key][1] for key in sorted(declared_set)]
    else:
        available_keys = [key for key, state in dependency_states.items() if state[0] == "available"]
        late_keys = [key for key, state in dependency_states.items() if state[0] == "late"]
        if available_keys:
            availability = "available"
            quality = "VALIDATED" if any(dependency_states[key][2].quality == "VALIDATED" for key in available_keys) else "PARTIAL"
            operational = min(utc(dependency_states[key][2].operational_available_at) for key in available_keys).isoformat()
            selected_indexes = [dependency_states[key][1] for key in sorted(available_keys)]
        elif late_keys:
            availability, quality = "late", "PARTIAL"
            operational = min(utc(dependency_states[key][2].operational_available_at) for key in late_keys).isoformat()
            selected_indexes = [dependency_states[key][1] for key in sorted(late_keys)]
        else:
            availability, quality, operational = "failed", "PROVIDER_ERROR", None
            selected_indexes = [state[1] for state in dependency_states.values()]
    body = {"episode_id": first.episode_id, "target_key": first.target_key, "action_id": first.action_id,
            "request_id": first.request_id, "candidate_id": first.candidate_id, "mode": mode,
            "entry_ids": sorted(entry.entry_id for entry in materialized),
            "required_dependency_keys": sorted(declared_set), "cutoff": utc(cutoff).isoformat()}
    aggregate_id = hashlib.sha256(json_bytes(body)).hexdigest()
    return ActionEvidenceAggregate(
        aggregate_id=aggregate_id, episode_id=first.episode_id, target_key=first.target_key,
        action_id=first.action_id, request_id=first.request_id, candidate_id=first.candidate_id,
        mode=mode, entry_ids=tuple(sorted(entry.entry_id for entry in materialized)),
        dependency_keys=tuple(sorted({entry.dependency_key for entry in materialized})),
        source_ids=tuple(sorted(set(entry.source_id for entry in materialized))),
        quality=quality, availability=availability, operational_available_at=operational,
        prediction_cutoff=cutoff,
        captured_at=max(materialized[index].captured_at for index in selected_indexes),
        missing_dependency_keys=missing_dependency_keys,
        dependency_statuses=tuple(sorted((key, state[0]) for key, state in dependency_states.items())) +
            tuple((key, "missing") for key in missing_dependency_keys),
    )


def link_trace_to_aggregate(trace: VoITrace, aggregate: ActionEvidenceAggregate, *, cutoff: str) -> VoIEvidenceLink:
    if (trace.episode_id, trace.target_key, trace.action_id, trace.request_id, trace.candidate_id) != (
        aggregate.episode_id, aggregate.target_key, aggregate.action_id, aggregate.request_id, aggregate.candidate_id):
        raise ValueError("Trace and aggregate identity mismatch")
    if utc(aggregate.prediction_cutoff) != utc(cutoff):
        raise ValueError("Aggregate prediction cutoff does not match link cutoff")
    cutoff_value = utc(cutoff)
    available = aggregate.availability == "available" and aggregate.quality in GOOD_QUALITIES and \
        aggregate.operational_available_at is not None and utc(aggregate.operational_available_at) <= cutoff_value and \
        utc(aggregate.captured_at) <= cutoff_value
    status = "linked_available" if available else "linked_unusable"
    observed = None
    error = None
    if available and trace.observed_risk_after is not None:
        observed = trace.risk_before - trace.observed_risk_after
        if trace.expected_information_value is not None:
            error = observed - trace.expected_information_value
    body = {"trace_id": trace.trace_id, "aggregate_id": aggregate.aggregate_id,
            "status": status, "prediction_cutoff": cutoff_value.isoformat()}
    return VoIEvidenceLink(
        link_id=hashlib.sha256(json_bytes(body)).hexdigest(), trace_id=trace.trace_id,
        episode_id=trace.episode_id, target_key=trace.target_key, request_id=trace.request_id,
        action_id=trace.action_id, candidate_id=trace.candidate_id,
        evidence_entry_id=aggregate.aggregate_id, evidence_availability=aggregate.availability,
        evidence_quality=aggregate.quality, requested_at=None, source_version=None,
        source_id="aggregate:" + aggregate.aggregate_id, dependency_key="aggregate:" + aggregate.aggregate_id,
        prediction_cutoff=cutoff, expected_information_value=trace.expected_information_value,
        observed_risk_reduction=observed, calibration_error=error, status=status,
    )


__all__ = ["VERSION", "MODES", "ActionEvidenceAggregate", "aggregate_entries", "link_trace_to_aggregate"]
