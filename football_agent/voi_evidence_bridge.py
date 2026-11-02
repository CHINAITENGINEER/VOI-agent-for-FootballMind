"""Join pre-action VoI traces with point-in-time evidence ledger records."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable

from .audit import json_bytes
from .evidence_ledger import EvidenceLedgerEntry
from .types import utc
from .voi_trace import VoITrace


VERSION = "voi-evidence-bridge-v1"


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _optional_number(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


@dataclass(frozen=True)
class VoIEvidenceLink:
    link_id: str
    trace_id: str
    episode_id: str
    target_key: str
    request_id: str
    action_id: str
    candidate_id: str
    evidence_entry_id: str | None
    evidence_availability: str | None
    evidence_quality: str | None
    requested_at: str | None
    source_version: str | None
    prediction_cutoff: str
    expected_information_value: float | None
    observed_risk_reduction: float | None
    calibration_error: float | None
    status: str
    hidden_result_included: bool = False
    source_id: str | None = None
    dependency_key: str | None = None

    def __post_init__(self) -> None:
        for field in ("link_id", "trace_id", "episode_id", "target_key", "request_id",
                      "action_id", "candidate_id", "status"):
            object.__setattr__(self, field, _text(getattr(self, field), field))
        if self.evidence_entry_id is not None:
            object.__setattr__(self, "evidence_entry_id", _text(self.evidence_entry_id, "evidence_entry_id"))
        if self.evidence_availability is not None:
            object.__setattr__(self, "evidence_availability", _text(self.evidence_availability, "evidence_availability"))
        if self.evidence_quality is not None:
            object.__setattr__(self, "evidence_quality", _text(self.evidence_quality, "evidence_quality"))
        if self.requested_at is not None:
            object.__setattr__(self, "requested_at", utc(self.requested_at).isoformat())
        if self.source_version is not None:
            object.__setattr__(self, "source_version", _text(self.source_version, "source_version"))
        for field in ("source_id", "dependency_key"):
            if getattr(self, field) is not None:
                object.__setattr__(self, field, _text(getattr(self, field), field))
        object.__setattr__(self, "prediction_cutoff", utc(self.prediction_cutoff).isoformat())
        object.__setattr__(self, "expected_information_value",
                           _optional_number(self.expected_information_value, "expected_information_value"))
        object.__setattr__(self, "observed_risk_reduction",
                           _optional_number(self.observed_risk_reduction, "observed_risk_reduction"))
        object.__setattr__(self, "calibration_error",
                           _optional_number(self.calibration_error, "calibration_error"))
        if self.hidden_result_included is not False:
            raise ValueError("VoI evidence links must exclude hidden results")

    def as_dict(self) -> dict:
        return {
            "schema_version": VERSION,
            "link_id": self.link_id,
            "trace_id": self.trace_id,
            "episode_id": self.episode_id,
            "target_key": self.target_key,
            "request_id": self.request_id,
            "action_id": self.action_id,
            "candidate_id": self.candidate_id,
            "evidence_entry_id": self.evidence_entry_id,
            "evidence_availability": self.evidence_availability,
            "evidence_quality": self.evidence_quality,
            "requested_at": self.requested_at,
            "source_version": self.source_version,
            "source_id": self.source_id,
            "dependency_key": self.dependency_key,
            "prediction_cutoff": self.prediction_cutoff,
            "expected_information_value": self.expected_information_value,
            "observed_risk_reduction": self.observed_risk_reduction,
            "calibration_error": self.calibration_error,
            "status": self.status,
            "hidden_result_included": False,
        }


def link_trace(trace: VoITrace, entries: Iterable[EvidenceLedgerEntry], *, cutoff: str) -> VoIEvidenceLink:
    cutoff_value = utc(cutoff)
    if trace.action_id == "STOP":
        return _make_link(trace, cutoff=cutoff, status="stop_recorded", entry=None,
                          observed_risk_reduction=None, calibration_error=None)
    matches = [entry for entry in entries
               if entry.episode_id == trace.episode_id
               and entry.target_key == trace.target_key
               and entry.request_id == trace.request_id
               and entry.action_id == trace.action_id
               and entry.candidate_id == trace.candidate_id]
    if not matches:
        return _make_link(trace, cutoff=cutoff, status="missing_evidence", entry=None,
                          observed_risk_reduction=None, calibration_error=None)
    if len(matches) > 1:
        return _make_link(trace, cutoff=cutoff, status="ambiguous_evidence", entry=None,
                          observed_risk_reduction=None, calibration_error=None)
    entry = matches[0]
    observed = None
    error = None
    available = (entry.availability == "available"
                 and entry.quality in {"VALIDATED", "PARTIAL"}
                 and utc(entry.captured_at) <= cutoff_value
                 and entry.operational_available_at is not None
                 and utc(entry.operational_available_at) <= cutoff_value)
    status = "linked_available" if available else "linked_unusable"
    if available and trace.observed_risk_after is not None:
        observed = trace.risk_before - trace.observed_risk_after
        if trace.expected_information_value is not None:
            error = observed - trace.expected_information_value
    return _make_link(trace, cutoff=cutoff, status=status, entry=entry,
                      observed_risk_reduction=observed, calibration_error=error)


def link_traces(traces: Iterable[VoITrace], entries: Iterable[EvidenceLedgerEntry], *, cutoff: str) -> tuple[VoIEvidenceLink, ...]:
    materialized = tuple(entries)
    return tuple(link_trace(trace, materialized, cutoff=cutoff) for trace in traces)


def _make_link(trace: VoITrace, *, cutoff: str, status: str, entry: EvidenceLedgerEntry | None,
               observed_risk_reduction: float | None, calibration_error: float | None) -> VoIEvidenceLink:
    body = {"trace_id": trace.trace_id, "episode_id": trace.episode_id, "target_key": trace.target_key,
            "request_id": trace.request_id, "action_id": trace.action_id, "candidate_id": trace.candidate_id,
            "status": status, "entry_id": None if entry is None else entry.entry_id,
            "prediction_cutoff": utc(cutoff).isoformat()}
    link_id = hashlib.sha256(json_bytes(body)).hexdigest()
    return VoIEvidenceLink(
        link_id=link_id, trace_id=trace.trace_id, episode_id=trace.episode_id,
        target_key=trace.target_key, request_id=trace.request_id, action_id=trace.action_id,
        candidate_id=trace.candidate_id, evidence_entry_id=None if entry is None else entry.entry_id,
        evidence_availability=None if entry is None else entry.availability,
        evidence_quality=None if entry is None else entry.quality,
        requested_at=None if entry is None else entry.requested_at,
        source_version=None if entry is None or not entry.source_version else entry.source_version,
        source_id=None if entry is None else entry.source_id,
        dependency_key=None if entry is None else entry.dependency_key,
        prediction_cutoff=cutoff, expected_information_value=trace.expected_information_value,
        observed_risk_reduction=observed_risk_reduction, calibration_error=calibration_error,
        status=status,
    )


__all__ = ["VERSION", "VoIEvidenceLink", "link_trace", "link_traces"]
