"""Holdout and dependence audits for source-return likelihood estimates.

Round37 supplied a point estimate for ``P(Z | theta, source, time)``.  This
module keeps the next boundary explicit: a point estimate may be scored on a
future, cluster-disjoint return set, but a holdout score is still a diagnostic
until there are enough independent tasks and a declared calibration protocol.
It also detects source rows that share a dependence cluster, so two tools are
not multiplied as if they were conditionally independent evidence.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

from .inquiry_returns import OUTCOMES
from .return_likelihood import (
    OBSERVED,
    SUPPORTED_ESTIMATE_STATUSES,
    ReturnLikelihoodEstimate,
    ReturnLikelihoodLedger,
    ReturnObservation,
    SUPPORTED_SELECTION_DESIGNS,
)


VERSION = "return-calibration-audit-v1"
SUPPORTED_HOLDOUT_STATUSES = frozenset({"HOLDOUT_EVALUATED_DIAGNOSTIC"})


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


@dataclass(frozen=True)
class CalibrationReport:
    schema_version: str
    status: str
    source_id: str
    time_bucket: str
    outcomes: tuple[str, ...]
    holdout_row_ids: tuple[str, ...]
    holdout_cluster_ids: tuple[str, ...]
    support_by_state: dict[str, int]
    observed_frequency_by_state: dict[str, dict[str, float]]
    predicted_frequency_by_state: dict[str, dict[str, float]]
    calibration_gap_by_state: dict[str, float]
    log_loss: float | None
    multiclass_brier: float | None
    independent_cluster_count: int
    calibration_status: str
    limitations: tuple[str, ...]

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "status": self.status,
            "source_id": self.source_id, "time_bucket": self.time_bucket,
            "outcomes": list(self.outcomes), "holdout_row_ids": list(self.holdout_row_ids),
            "holdout_cluster_ids": list(self.holdout_cluster_ids),
            "support_by_state": dict(self.support_by_state),
            "observed_frequency_by_state": {
                state: dict(values) for state, values in self.observed_frequency_by_state.items()
            },
            "predicted_frequency_by_state": {
                state: dict(values) for state, values in self.predicted_frequency_by_state.items()
            },
            "calibration_gap_by_state": dict(self.calibration_gap_by_state),
            "log_loss": self.log_loss, "multiclass_brier": self.multiclass_brier,
            "independent_cluster_count": self.independent_cluster_count,
            "calibration_status": self.calibration_status,
            "limitations": list(self.limitations),
        }


def _empty_report(estimate: ReturnLikelihoodEstimate, status: str, *, limitations: Iterable[str]) -> CalibrationReport:
    return CalibrationReport(
        VERSION, status, estimate.source_id, estimate.time_bucket, estimate.outcomes,
        (), (), {state: 0 for state in estimate.states},
        {state: {} for state in estimate.states}, {state: {} for state in estimate.states},
        {state: None for state in estimate.states}, None, None, 0,
        "NOT_CALIBRATED", tuple(limitations),
    )


def evaluate_holdout(
    estimate: ReturnLikelihoodEstimate,
    *,
    training_ledger: ReturnLikelihoodLedger,
    holdout_ledger: ReturnLikelihoodLedger,
    min_rows: int = 2,
) -> CalibrationReport:
    """Score a fitted matrix on a future, dependence-disjoint return set.

    The holdout has to carry the same pre-query state semantics as the training
    rows.  It is never joined to a match result.  Rows from adaptive selection,
    a reused cluster or a different source/time scope stop the report before a
    score is produced.
    """
    if not isinstance(estimate, ReturnLikelihoodEstimate):
        raise ValueError("A ReturnLikelihoodEstimate is required")
    if estimate.status not in SUPPORTED_ESTIMATE_STATUSES or estimate.likelihoods is None:
        return _empty_report(estimate, "UNSUPPORTED_ESTIMATE", limitations=(
            "A point estimate with explicit support is required before holdout scoring.",
        ))
    min_rows = _positive_int(min_rows, "min_rows")
    train_by_id = {row.row_id: row for row in training_ledger.rows}
    train_rows = [train_by_id[row_id] for row_id in estimate.selected_row_ids if row_id in train_by_id]
    train_clusters = {row.cluster_id for row in train_rows}
    if len(train_rows) != len(estimate.selected_row_ids):
        return _empty_report(estimate, "TRAINING_PROVENANCE_MISSING", limitations=(
            "Every fitted row must be available in the pinned training ledger.",
        ))

    matching = [row for row in holdout_ledger.rows
                if row.source_id == estimate.source_id and row.time_bucket == estimate.time_bucket]
    if not matching:
        return _empty_report(estimate, "NO_HOLDOUT_ROWS", limitations=(
            "No same-source and same-time holdout row was supplied.",
        ))
    if any(row.status != OBSERVED for row in matching):
        return _empty_report(estimate, "UNSUPPORTED_HOLDOUT_CENSORING", limitations=(
            "A holdout must be observed before its own cutoff; censored rows stay unlabeled.",
        ))
    if any(row.return_label not in estimate.outcomes for row in matching):
        return _empty_report(estimate, "UNSUPPORTED_HOLDOUT_OUTCOME_SCOPE", limitations=(
            "Every holdout return category must be in the fitted outcome vocabulary.",
        ))
    if any(row.latent_state not in estimate.states for row in matching):
        return _empty_report(estimate, "UNSUPPORTED_HOLDOUT_STATE_SCOPE", limitations=(
            "Holdout theta labels must use the same pre-query state vocabulary.",
        ))
    if any(row.latent_state_basis == "match_result" for row in matching):
        return _empty_report(estimate, "UNSUPPORTED_HOLDOUT_MATCH_RESULT_LEAKAGE", limitations=(
            "Post-event labels cannot define a pre-query holdout state.",
        ))
    if any(row.origin != train_rows[0].origin for row in matching):
        return _empty_report(estimate, "UNSUPPORTED_HOLDOUT_MIXED_ORIGIN", limitations=(
            "Training and holdout evidence origins must remain explicitly separated.",
        ))
    if any(row.selection_design not in SUPPORTED_SELECTION_DESIGNS for row in matching):
        return _empty_report(estimate, "UNSUPPORTED_HOLDOUT_ADAPTIVE_SELECTION", limitations=(
            "Holdout policy must be a declared source-probe design.",
        ))
    if any(row.cluster_id in train_clusters for row in matching):
        return _empty_report(estimate, "OVERLAPPING_DEPENDENCE_CLUSTER", limitations=(
            "Training and holdout rows share a dependence cluster.",
        ))
    if len({row.cluster_id for row in matching}) != len(matching):
        return _empty_report(estimate, "HOLDOUT_DEPENDENCE_CLUSTER_REUSE", limitations=(
            "Repeated source pages or reports are not independent holdout rows.",
        ))
    if len(matching) < min_rows:
        return _empty_report(estimate, "INSUFFICIENT_HOLDOUT", limitations=(
            f"At least {min_rows} cluster-disjoint holdout rows are required.",
        ))

    index = {outcome: i for i, outcome in enumerate(estimate.outcomes)}
    observed_counts = {state: Counter() for state in estimate.states}
    predicted = {state: dict(zip(estimate.outcomes, estimate.likelihoods[state])) for state in estimate.states}
    log_terms, brier_terms = [], []
    for row in matching:
        state, label = row.latent_state, row.return_label
        observed_counts[state][label] += 1
        probabilities = estimate.likelihoods[state]
        probability = probabilities[index[label]]
        log_terms.append(-math.log(probability))
        brier_terms.append(math.fsum((p - int(i == index[label])) ** 2 for i, p in enumerate(probabilities)))
    support = {state: sum(observed_counts[state].values()) for state in estimate.states}
    observed_frequency = {
        state: {outcome: observed_counts[state][outcome] / support[state] if support[state] else 0.0
                for outcome in estimate.outcomes}
        for state in estimate.states
    }
    gaps = {
        state: max((abs(predicted[state][outcome] - observed_frequency[state][outcome])
                    for outcome in estimate.outcomes), default=0.0)
        for state in estimate.states
    }
    return CalibrationReport(
        VERSION, "HOLDOUT_EVALUATED_DIAGNOSTIC", estimate.source_id, estimate.time_bucket,
        estimate.outcomes, tuple(row.row_id for row in matching), tuple(row.cluster_id for row in matching),
        support, observed_frequency, predicted, gaps,
        math.fsum(log_terms) / len(log_terms), math.fsum(brier_terms) / len(brier_terms),
        len({row.cluster_id for row in matching}),
        "OUT_OF_SAMPLE_DIAGNOSTIC_NOT_CALIBRATION_CERTIFICATE",
        (
            "Scores are evaluated on a cluster-disjoint holdout but do not establish deployment calibration.",
            "The holdout uses declared pre-query state labels and source-probe sampling.",
            "No match result or betting outcome enters the source-return score.",
        ),
    )


def audit_source_dependence(rows: Iterable[ReturnObservation | Mapping[str, Any]]) -> dict[str, Any]:
    """Report cross-source dependence clusters before joint Bayesian fusion."""
    ledger = ReturnLikelihoodLedger(rows)
    by_cluster: dict[str, dict[str, Any]] = defaultdict(lambda: {"sources": set(), "row_ids": []})
    for row in ledger.rows:
        by_cluster[row.cluster_id]["sources"].add(row.source_id)
        by_cluster[row.cluster_id]["row_ids"].append(row.row_id)
    shared = {
        cluster: {"sources": sorted(value["sources"]), "row_ids": sorted(value["row_ids"])}
        for cluster, value in by_cluster.items() if len(value["sources"]) > 1
    }
    return {
        "schema_version": VERSION, "status": "SHARED_DEPENDENCE_DETECTED" if shared else "DISJOINT_SOURCE_CLUSTERS",
        "factorization_allowed": not shared,
        "cluster_count": len(by_cluster), "shared_clusters": shared,
        "scope": "source_return_likelihoods_must_not_be_multiplied_across_shared_reports",
    }
