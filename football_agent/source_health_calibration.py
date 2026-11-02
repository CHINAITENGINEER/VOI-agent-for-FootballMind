"""Time-split calibration of observable source-channel behavior.

This module turns append-only ``EvidenceLedgerEntry`` records into a
future-safe operational diagnostic. It estimates only return availability and
status drift from records captured before a declared evaluation window. It
deliberately leaves semantic accuracy, match-result impact, and numeric VoI
unmeasured unless a separate evaluator supplies those labels.

The output is suitable for deciding whether a source-health model has enough
operational support. It is not a probability of a football outcome and it is
not a claim that a source is globally trustworthy.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
import math
from typing import Any, Iterable

from .evidence_ledger import EvidenceLedgerEntry
from .types import utc


VERSION = "source-health-calibration-v1"
STATUS_VALUES = ("available", "late", "failed", "duplicate")
DRIFT_VALUES = ("STABLE", "DEGRADED", "IMPROVED", "INSUFFICIENT_SUPPORT")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    value = float(value)
    if not 0 < value < 1:
        raise ValueError(f"{name} must be between zero and one")
    return value


def _lower_bound(rate: float, count: int, confidence: float) -> float:
    radius = math.sqrt(math.log(2.0 / confidence) / (2.0 * count))
    return max(0.0, rate - radius)


def _counts(rows: Iterable[EvidenceLedgerEntry]) -> dict[str, int]:
    counter = Counter(row.availability for row in rows)
    return {status: counter.get(status, 0) for status in STATUS_VALUES}


def _latest_per_dependency(rows: Iterable[EvidenceLedgerEntry]) -> tuple[EvidenceLedgerEntry, ...]:
    """Collapse retries to one deterministic support unit per dependency.

    The append-only ledger keeps every attempt.  Availability rates must not
    become more precise merely because one dependency was retried repeatedly,
    so the latest record in the declared window is used for the independent
    status distribution while attempt counts remain visible separately.
    """
    latest: dict[str, EvidenceLedgerEntry] = {}
    for row in rows:
        previous = latest.get(row.dependency_key)
        if previous is None or (utc(row.captured_at), row.sequence) > (
            utc(previous.captured_at), previous.sequence
        ):
            latest[row.dependency_key] = row
    return tuple(latest[key] for key in sorted(latest))


def _record_clock(entry: EvidenceLedgerEntry) -> datetime:
    # captured_at is the append/archive clock. It exists for failed attempts,
    # whereas operational_available_at is absent for failures.
    return utc(entry.captured_at)


@dataclass(frozen=True)
class SourceHealthCalibration:
    schema_version: str
    source_id: str
    source_version: str
    train_end: str
    eval_start: str
    eval_end: str
    train_attempts: int
    train_independent_dependency_count: int
    train_duplicate_attempts: int
    train_status_counts: dict[str, int]
    train_attempt_status_counts: dict[str, int]
    train_available_rate: float | None
    train_available_lower_bound: float | None
    eval_attempts: int
    eval_independent_dependency_count: int
    eval_duplicate_attempts: int
    eval_status_counts: dict[str, int]
    eval_attempt_status_counts: dict[str, int]
    eval_available_rate: float | None
    availability_rate_delta: float | None
    drift_status: str
    operational_support_status: str
    semantic_accuracy_status: str
    numeric_voi_status: str
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        for field in ("source_id", "source_version", "train_end", "eval_start", "eval_end",
                      "drift_status", "operational_support_status", "semantic_accuracy_status",
                      "numeric_voi_status"):
            object.__setattr__(self, field, _text(getattr(self, field), field))
        if self.drift_status not in DRIFT_VALUES:
            raise ValueError("Unsupported drift status")
        for field in ("train_attempts", "train_independent_dependency_count", "train_duplicate_attempts",
                      "eval_attempts", "eval_independent_dependency_count", "eval_duplicate_attempts"):
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field} must be a nonnegative integer")
        if self.train_independent_dependency_count > self.train_attempts or self.eval_independent_dependency_count > self.eval_attempts:
            raise ValueError("independent dependencies cannot exceed attempts")
        if self.train_duplicate_attempts != self.train_attempts - self.train_independent_dependency_count:
            raise ValueError("train_duplicate_attempts must match attempts and dependency count")
        if self.eval_duplicate_attempts != self.eval_attempts - self.eval_independent_dependency_count:
            raise ValueError("eval_duplicate_attempts must match attempts and dependency count")
        for field, expected_total in (
            ("train_status_counts", self.train_independent_dependency_count),
            ("train_attempt_status_counts", self.train_attempts),
            ("eval_status_counts", self.eval_independent_dependency_count),
            ("eval_attempt_status_counts", self.eval_attempts),
        ):
            counts = getattr(self, field)
            if set(counts) != set(STATUS_VALUES) or any(
                type(value) is not int or value < 0 for value in counts.values()
            ) or sum(counts.values()) != expected_total:
                raise ValueError(f"{field} must contain nonnegative counts for every status")
        for field in ("train_available_rate", "train_available_lower_bound", "eval_available_rate"):
            value = getattr(self, field)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value) or value < 0 or value > 1):
                raise ValueError(f"{field} must be a probability when present")
        if self.availability_rate_delta is not None and (
            isinstance(self.availability_rate_delta, bool)
            or not isinstance(self.availability_rate_delta, (int, float))
            or not math.isfinite(self.availability_rate_delta)
            or self.availability_rate_delta < -1 or self.availability_rate_delta > 1
        ):
            raise ValueError("availability_rate_delta must be between -1 and 1")
        object.__setattr__(self, "limitations", tuple(_text(item, "limitation") for item in self.limitations))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "source_version": self.source_version,
            "train_end": self.train_end,
            "eval_start": self.eval_start,
            "eval_end": self.eval_end,
            "train_attempts": self.train_attempts,
            "train_independent_dependency_count": self.train_independent_dependency_count,
            "train_duplicate_attempts": self.train_duplicate_attempts,
            "train_status_counts": dict(self.train_status_counts),
            "train_attempt_status_counts": dict(self.train_attempt_status_counts),
            "train_available_rate": self.train_available_rate,
            "train_available_lower_bound": self.train_available_lower_bound,
            "eval_attempts": self.eval_attempts,
            "eval_independent_dependency_count": self.eval_independent_dependency_count,
            "eval_duplicate_attempts": self.eval_duplicate_attempts,
            "eval_status_counts": dict(self.eval_status_counts),
            "eval_attempt_status_counts": dict(self.eval_attempt_status_counts),
            "eval_available_rate": self.eval_available_rate,
            "availability_rate_delta": self.availability_rate_delta,
            "drift_status": self.drift_status,
            "operational_support_status": self.operational_support_status,
            "semantic_accuracy_status": self.semantic_accuracy_status,
            "numeric_voi_status": self.numeric_voi_status,
            "limitations": list(self.limitations),
        }


def calibrate_source_health(
    entries: Iterable[EvidenceLedgerEntry], *, train_end: str | datetime,
    eval_start: str | datetime, eval_end: str | datetime,
    min_independent_dependencies: int = 5, confidence: float = .10,
    drift_tolerance: float = .10,
) -> tuple[SourceHealthCalibration, ...]:
    """Calibrate source-channel availability with a strict temporal split.

    Training rows satisfy ``captured_at <= train_end``. Evaluation rows satisfy
    ``eval_start <= captured_at <= eval_end``. The gap is intentionally unused.
    Rows after ``eval_end`` are ignored. Duplicate dependency keys remain in
    attempt counts but contribute one independent support unit.
    """
    train_cutoff = utc(train_end)
    eval_begin = utc(eval_start)
    eval_cutoff = utc(eval_end)
    if not train_cutoff < eval_begin <= eval_cutoff:
        raise ValueError("require train_end < eval_start <= eval_end")
    minimum = _positive_int(min_independent_dependencies, "min_independent_dependencies")
    confidence = _probability(confidence, "confidence")
    if isinstance(drift_tolerance, bool) or not isinstance(drift_tolerance, (int, float)) \
            or not math.isfinite(drift_tolerance) or not 0 <= drift_tolerance <= 1:
        raise ValueError("drift_tolerance must be between zero and one")
    by_source: dict[tuple[str, str], list[EvidenceLedgerEntry]] = defaultdict(list)
    for entry in entries:
        if not isinstance(entry, EvidenceLedgerEntry):
            raise ValueError("entries must contain EvidenceLedgerEntry values")
        clock = _record_clock(entry)
        if clock <= train_cutoff or eval_begin <= clock <= eval_cutoff:
            by_source[(entry.source_id, entry.source_version or "UNVERSIONED")].append(entry)

    reports = []
    limitations = (
        "This estimates operational return behavior, not semantic content accuracy.",
        "A source version and dependency key are required for independent support; missing metadata stays visible.",
        "The training lower bound is a bounded availability diagnostic, not a match-result probability.",
        "No realized result, loss, evaluator label, or numeric VoI is inferred by this function.",
    )
    for (source_id, source_version), rows in sorted(by_source.items()):
        train_rows = [row for row in rows if _record_clock(row) <= train_cutoff]
        eval_rows = [row for row in rows if eval_begin <= _record_clock(row) <= eval_cutoff]
        train_attempt_counts, eval_attempt_counts = _counts(train_rows), _counts(eval_rows)
        train_independent_rows = _latest_per_dependency(train_rows)
        eval_independent_rows = _latest_per_dependency(eval_rows)
        train_counts, eval_counts = _counts(train_independent_rows), _counts(eval_independent_rows)
        train_deps = {row.dependency_key for row in train_rows}
        eval_deps = {row.dependency_key for row in eval_rows}
        train_attempts, eval_attempts = len(train_rows), len(eval_rows)
        train_independent, eval_independent = len(train_deps), len(eval_deps)
        train_rate = (train_counts["available"] / train_independent) if train_independent else None
        eval_rate = (eval_counts["available"] / eval_independent) if eval_independent else None
        train_lcb = (_lower_bound(train_rate, train_independent, confidence)
                     if train_rate is not None else None)
        support = (
            "SUPPORTED_OPERATIONAL_DIAGNOSTIC"
            if train_independent >= minimum and eval_independent >= minimum
            else "INSUFFICIENT_INDEPENDENT_SUPPORT"
        )
        delta = None if train_rate is None or eval_rate is None else eval_rate - train_rate
        if support != "SUPPORTED_OPERATIONAL_DIAGNOSTIC" or delta is None:
            drift = "INSUFFICIENT_SUPPORT"
        elif delta < -float(drift_tolerance):
            drift = "DEGRADED"
        elif delta > float(drift_tolerance):
            drift = "IMPROVED"
        else:
            drift = "STABLE"
        reports.append(SourceHealthCalibration(
            schema_version=VERSION, source_id=source_id, source_version=source_version,
            train_end=train_cutoff.isoformat(), eval_start=eval_begin.isoformat(), eval_end=eval_cutoff.isoformat(),
            train_attempts=train_attempts, train_independent_dependency_count=train_independent,
            train_duplicate_attempts=train_attempts - train_independent,
            train_status_counts=train_counts, train_attempt_status_counts=train_attempt_counts,
            train_available_rate=train_rate, train_available_lower_bound=train_lcb,
            eval_attempts=eval_attempts, eval_independent_dependency_count=eval_independent,
            eval_duplicate_attempts=eval_attempts - eval_independent,
            eval_status_counts=eval_counts, eval_attempt_status_counts=eval_attempt_counts,
            eval_available_rate=eval_rate, availability_rate_delta=delta, drift_status=drift,
            operational_support_status=support,
            semantic_accuracy_status="UNMEASURED",
            numeric_voi_status="UNMEASURED",
            limitations=limitations,
        ))
    return tuple(reports)


__all__ = [
    "VERSION", "STATUS_VALUES", "DRIFT_VALUES", "SourceHealthCalibration", "calibrate_source_health",
]
