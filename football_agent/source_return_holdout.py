"""Future, dependency-disjoint holdout diagnostics for source returns."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Iterable

from .evidence_ledger import EvidenceLedgerEntry


VERSION = "source-return-holdout-v1"


@dataclass(frozen=True)
class SourceReturnHoldoutReport:
    schema_version: str
    source_id: str
    source_version: str
    training_attempts: int
    holdout_attempts: int
    training_independent_dependencies: int
    holdout_independent_dependencies: int
    predicted_available_rate: float | None
    observed_available_rate: float | None
    availability_rate_gap: float | None
    binary_log_loss: float | None
    brier_score: float | None
    status: str
    limitations: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "schema_version": self.schema_version, "source_id": self.source_id,
            "source_version": self.source_version, "training_attempts": self.training_attempts,
            "holdout_attempts": self.holdout_attempts,
            "training_independent_dependencies": self.training_independent_dependencies,
            "holdout_independent_dependencies": self.holdout_independent_dependencies,
            "predicted_available_rate": self.predicted_available_rate,
            "observed_available_rate": self.observed_available_rate,
            "availability_rate_gap": self.availability_rate_gap,
            "binary_log_loss": self.binary_log_loss, "brier_score": self.brier_score,
            "status": self.status, "limitations": list(self.limitations),
        }


def evaluate_future_holdout(
    training_entries: Iterable[EvidenceLedgerEntry],
    holdout_entries: Iterable[EvidenceLedgerEntry],
    *,
    min_training_dependencies: int = 5,
    min_holdout_dependencies: int = 5,
) -> tuple[SourceReturnHoldoutReport, ...]:
    if type(min_training_dependencies) is not int or min_training_dependencies < 1:
        raise ValueError("min_training_dependencies must be a positive integer")
    if type(min_holdout_dependencies) is not int or min_holdout_dependencies < 1:
        raise ValueError("min_holdout_dependencies must be a positive integer")
    train_groups = {}
    for entry in training_entries:
        train_groups.setdefault((entry.source_id or "UNKNOWN_SOURCE", entry.source_version or "UNVERSIONED"), []).append(entry)
    hold_groups = {}
    for entry in holdout_entries:
        hold_groups.setdefault((entry.source_id or "UNKNOWN_SOURCE", entry.source_version or "UNVERSIONED"), []).append(entry)
    reports = []
    for key, train in sorted(train_groups.items()):
        source_id, source_version = key
        hold = hold_groups.get(key, [])
        train_deps = {row.dependency_key for row in train}
        hold_deps = {row.dependency_key for row in hold}
        limitations = (
            "This score evaluates operational availability, not football-result probability.",
            "The holdout must remain future and dependency-disjoint; repeated clusters reduce effective support.",
            "A source version is treated as a fixed contract across the split.",
        )
        status = "SUPPORTED_HOLDOUT_DIAGNOSTIC"
        if not hold:
            status = "NO_MATCHING_HOLDOUT"
        elif train_deps & hold_deps:
            status = "OVERLAPPING_DEPENDENCE_CLUSTER"
        elif any(row.requested_at is None for row in train + hold):
            status = "MISSING_REQUEST_TIME"
        elif max(row.requested_at for row in train) >= min(row.requested_at for row in hold):
            status = "NON_FUTURE_HOLDOUT"
        elif len(train_deps) < min_training_dependencies or len(hold_deps) < min_holdout_dependencies:
            status = "INSUFFICIENT_INDEPENDENT_SUPPORT"
        if status != "SUPPORTED_HOLDOUT_DIAGNOSTIC":
            reports.append(SourceReturnHoldoutReport(
                VERSION, source_id, source_version, len(train), len(hold), len(train_deps), len(hold_deps),
                None, None, None, None, None, status, limitations,
            ))
            continue
        p = sum(row.availability == "available" for row in train) / len(train)
        observed = [1 if row.availability == "available" else 0 for row in hold]
        observed_rate = sum(observed) / len(observed)
        # Clamp only the boundary estimate for finite log loss; the raw rate is preserved above.
        probability = min(1.0 - 1e-12, max(1e-12, p))
        log_loss = -sum(value * math.log(probability) + (1 - value) * math.log(1 - probability)
                        for value in observed) / len(observed)
        brier = sum((probability - value) ** 2 for value in observed) / len(observed)
        reports.append(SourceReturnHoldoutReport(
            VERSION, source_id, source_version, len(train), len(hold), len(train_deps), len(hold_deps),
            p, observed_rate, observed_rate - p, log_loss, brier, status, limitations,
        ))
    return tuple(reports)


__all__ = ["VERSION", "SourceReturnHoldoutReport", "evaluate_future_holdout"]
