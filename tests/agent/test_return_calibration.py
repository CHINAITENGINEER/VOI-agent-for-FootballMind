from __future__ import annotations

import math

from football_agent.return_calibration import audit_source_dependence, evaluate_holdout
from football_agent.return_likelihood import CENSORED_AT_CUTOFF, OBSERVED, ReturnLikelihoodLedger, ReturnObservation


CUTOFF = "2026-09-27T10:00:00Z"


def row(row_id, *, state="defensive", label="source_support", cluster=None,
        design="constructed_randomized_source_probe", status=OBSERVED,
        source="tactical-source", bucket="pre_match", basis="constructed_pre_query_state",
        available="2026-09-27T09:55:00Z", observed="2026-09-27T09:56:00Z"):
    return ReturnObservation(
        row_id=row_id, source_id=source, time_bucket=bucket,
        query_at="2026-09-27T09:50:00Z", prediction_cutoff=CUTOFF,
        status=status, return_label=label if status == OBSERVED else None,
        available_at=available, observed_at=observed, origin="constructed",
        latent_state=state, latent_state_basis=basis,
        selection_design=design, cluster_id=cluster or row_id,
    )


def train_ledger():
    return ReturnLikelihoodLedger([
        row("train-d1", label="source_support"),
        row("train-d2", label="source_support"),
        row("train-a1", state="attacking", label="source_challenge"),
        row("train-a2", state="attacking", label="source_challenge"),
    ])


def fit():
    return train_ledger().fit(
        source_id="tactical-source", time_bucket="pre_match",
        states=("defensive", "attacking"), outcomes=("source_support", "source_challenge"), alpha=.5,
    )


def test_cluster_disjoint_holdout_gets_scored_as_diagnostic():
    train = train_ledger()
    holdout = ReturnLikelihoodLedger([
        row("hold-d1", label="source_support"),
        row("hold-d2", label="source_challenge"),
        row("hold-a1", state="attacking", label="source_challenge"),
        row("hold-a2", state="attacking", label="source_challenge"),
    ])
    report = evaluate_holdout(fit(), training_ledger=train, holdout_ledger=holdout)
    assert report.status == "HOLDOUT_EVALUATED_DIAGNOSTIC"
    assert report.calibration_status == "OUT_OF_SAMPLE_DIAGNOSTIC_NOT_CALIBRATION_CERTIFICATE"
    assert report.independent_cluster_count == 4
    assert math.isfinite(report.log_loss) and math.isfinite(report.multiclass_brier)
    assert report.support_by_state == {"defensive": 2, "attacking": 2}
    assert report.manifest()["holdout_row_ids"] == ["hold-d1", "hold-d2", "hold-a1", "hold-a2"]


def test_training_and_holdout_cluster_overlap_stops_scoring():
    holdout = ReturnLikelihoodLedger([row("hold", cluster="train-d1")])
    report = evaluate_holdout(fit(), training_ledger=train_ledger(), holdout_ledger=holdout, min_rows=1)
    assert report.status == "OVERLAPPING_DEPENDENCE_CLUSTER"
    assert report.log_loss is None


def test_adaptive_holdout_is_not_a_calibration_set():
    holdout = ReturnLikelihoodLedger([row("adaptive", design="agent_adaptive")])
    report = evaluate_holdout(fit(), training_ledger=train_ledger(), holdout_ledger=holdout, min_rows=1)
    assert report.status == "UNSUPPORTED_HOLDOUT_ADAPTIVE_SELECTION"


def test_censored_holdout_keeps_score_unknown():
    holdout = ReturnLikelihoodLedger([row("late", status=CENSORED_AT_CUTOFF,
                                          available="2026-09-27T10:01:00Z",
                                          observed="2026-09-27T10:02:00Z")])
    report = evaluate_holdout(fit(), training_ledger=train_ledger(), holdout_ledger=holdout, min_rows=1)
    assert report.status == "UNSUPPORTED_HOLDOUT_CENSORING"


def test_small_holdout_is_reported_separately():
    holdout = ReturnLikelihoodLedger([row("hold")])
    report = evaluate_holdout(fit(), training_ledger=train_ledger(), holdout_ledger=holdout, min_rows=2)
    assert report.status == "INSUFFICIENT_HOLDOUT"


def test_shared_source_cluster_blocks_independent_factorization():
    result = audit_source_dependence([
        row("a", source="source-a", cluster="shared"),
        row("b", source="source-b", cluster="shared"),
        row("c", source="source-a", cluster="separate"),
    ])
    assert result["status"] == "SHARED_DEPENDENCE_DETECTED"
    assert result["factorization_allowed"] is False
    assert result["shared_clusters"]["shared"]["sources"] == ["source-a", "source-b"]


def test_disjoint_source_clusters_are_explicitly_safe_for_factorization_scope():
    result = audit_source_dependence([
        row("a", source="source-a", cluster="a"),
        row("b", source="source-b", cluster="b"),
    ])
    assert result["status"] == "DISJOINT_SOURCE_CLUSTERS"
    assert result["factorization_allowed"] is True


def test_unsupported_training_estimate_keeps_holdout_score_empty():
    train = ReturnLikelihoodLedger([row("only")])
    estimate = train.fit(
        source_id="tactical-source", time_bucket="pre_match",
        states=("defensive", "attacking"), outcomes=("source_support", "source_challenge"),
    )
    report = evaluate_holdout(estimate, training_ledger=train,
                              holdout_ledger=ReturnLikelihoodLedger([row("hold")]), min_rows=1)
    assert estimate.status == "INSUFFICIENT_SUPPORT"
    assert report.status == "UNSUPPORTED_ESTIMATE"
    assert report.log_loss is None
