"""Operational return and VoI calibration diagnostics from immutable records.

These reports describe source behavior and policy-value alignment. They do not
turn source return frequencies into football-result probabilities.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Iterable

from .evidence_ledger import EvidenceLedgerEntry
from .voi_evidence_bridge import VoIEvidenceLink


VERSION = "source-return-calibration-v1"
RETURN_STATUSES = ("available", "late", "failed", "duplicate")


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


@dataclass(frozen=True)
class SourceReturnReport:
    schema_version: str
    source_version: str
    source_id: str
    time_bucket: str
    attempts: int
    independent_dependency_count: int
    status_counts: dict[str, int]
    status_frequencies: dict[str, float]
    available_rate: float
    duplicate_dependency_count: int
    support_status: str
    limitations: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "schema_version": self.schema_version, "source_version": self.source_version, "source_id": self.source_id,
            "time_bucket": self.time_bucket, "attempts": self.attempts,
            "independent_dependency_count": self.independent_dependency_count,
            "status_counts": dict(self.status_counts), "status_frequencies": dict(self.status_frequencies),
            "available_rate": self.available_rate,
            "duplicate_dependency_count": self.duplicate_dependency_count,
            "support_status": self.support_status, "limitations": list(self.limitations),
        }


def calibrate_source_returns(
    entries: Iterable[EvidenceLedgerEntry], *, time_bucket: str, min_attempts: int = 5,
) -> tuple[SourceReturnReport, ...]:
    if not isinstance(time_bucket, str) or not time_bucket.strip():
        raise ValueError("time_bucket must be a nonempty string")
    if type(min_attempts) is not int or min_attempts < 1:
        raise ValueError("min_attempts must be a positive integer")
    groups: dict[tuple[str, str], list[EvidenceLedgerEntry]] = defaultdict(list)
    for entry in entries:
        groups[(entry.source_id or "UNKNOWN_SOURCE", entry.source_version or "UNVERSIONED")].append(entry)
    reports = []
    for (source_id, source_version), rows in sorted(groups.items()):
        counts = Counter(row.availability for row in rows)
        status_counts = {status: counts.get(status, 0) for status in RETURN_STATUSES}
        attempts = len(rows)
        dependencies = {row.dependency_key for row in rows}
        duplicate_dependency_count = sum(1 for value in Counter(row.dependency_key for row in rows).values() if value > 1)
        reports.append(SourceReturnReport(
            schema_version=VERSION, source_version=source_version, source_id=source_id, time_bucket=time_bucket,
            attempts=attempts, independent_dependency_count=len(dependencies),
            status_counts=status_counts,
            status_frequencies={status: status_counts[status] / attempts for status in RETURN_STATUSES},
            available_rate=status_counts["available"] / attempts,
            duplicate_dependency_count=duplicate_dependency_count,
            support_status="SUPPORTED_DIAGNOSTIC" if len(dependencies) >= min_attempts else "INSUFFICIENT_SUPPORT",
            limitations=(
                "This is an operational return diagnostic, not a football-result probability.",
                "Repeated dependency keys reduce independent support and are reported separately.",
                "Real-source calibration requires future, time-separated records and source-version control.",
            ),
        ))
    return tuple(reports)


@dataclass(frozen=True)
class VoICalibrationReport:
    schema_version: str
    source_version: str
    source_id: str
    link_count: int
    available_link_count: int
    mean_expected_information_value: float | None
    mean_observed_risk_reduction: float | None
    mean_calibration_error: float | None
    independent_dependency_count: int
    status_counts: dict[str, int]
    support_status: str
    limitations: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "schema_version": self.schema_version, "source_version": self.source_version, "source_id": self.source_id,
            "link_count": self.link_count, "available_link_count": self.available_link_count,
            "mean_expected_information_value": self.mean_expected_information_value,
            "mean_observed_risk_reduction": self.mean_observed_risk_reduction,
            "mean_calibration_error": self.mean_calibration_error,
            "independent_dependency_count": self.independent_dependency_count,
            "status_counts": dict(self.status_counts), "support_status": self.support_status,
            "limitations": list(self.limitations),
        }


def calibrate_voi_links(links: Iterable[VoIEvidenceLink], *, min_available_links: int = 5) -> tuple[VoICalibrationReport, ...]:
    if type(min_available_links) is not int or min_available_links < 1:
        raise ValueError("min_available_links must be a positive integer")
    groups: dict[tuple[str, str], list[VoIEvidenceLink]] = defaultdict(list)
    for link in links:
        if link.source_version is not None:
            groups[(link.source_id or "UNKNOWN_SOURCE", link.source_version)].append(link)
    reports = []
    for (source_id, source_version), rows in sorted(groups.items()):
        status_counts = dict(Counter(row.status for row in rows))
        available = [row for row in rows if row.status == "linked_available"
                     and row.expected_information_value is not None
                     and row.observed_risk_reduction is not None
                     and row.calibration_error is not None]
        independent_dependencies = {row.dependency_key or row.link_id for row in available}
        mean = lambda values: None if not values else math.fsum(values) / len(values)
        reports.append(VoICalibrationReport(
            schema_version=VERSION, source_version=source_version, source_id=source_id, link_count=len(rows),
            available_link_count=len(available),
            mean_expected_information_value=mean([row.expected_information_value for row in available]),
            mean_observed_risk_reduction=mean([row.observed_risk_reduction for row in available]),
            mean_calibration_error=mean([row.calibration_error for row in available]),
            independent_dependency_count=len(independent_dependencies),
            status_counts=status_counts,
            support_status="SUPPORTED_DIAGNOSTIC" if len(independent_dependencies) >= min_available_links else "INSUFFICIENT_SUPPORT",
            limitations=(
                "The error compares model-expected VoI with observed posterior risk reduction.",
                "This report is a finite-model diagnostic and does not measure football-result return.",
                "Adaptive selection and shared dependencies require separate causal or cluster-aware analysis.",
            ),
        ))
    return tuple(reports)


__all__ = ["VERSION", "RETURN_STATUSES", "SourceReturnReport", "VoICalibrationReport",
           "calibrate_source_returns", "calibrate_voi_links"]
