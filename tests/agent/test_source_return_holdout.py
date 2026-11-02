import pytest

from football_agent.evidence_ledger import EvidenceLedgerEntry
from football_agent.source_return_holdout import evaluate_future_holdout


def row(name, *, source_id="source-a", version="v1", status="available", dependency=None,
        requested="2026-09-27T08:00:00Z", captured="2026-09-27T08:10:00Z"):
    if status == "available":
        quality, observed, available = "VALIDATED", requested, requested
    else:
        quality, observed, available = "PROVIDER_ERROR", None, None
    return EvidenceLedgerEntry.from_row(
        {"candidate_id": name, "source_id": source_id, "dependency_key": dependency or f"dep-{name}",
         "quality": quality, "availability": status, "observed_at": observed,
         "operational_available_at": available, "evidence_refs": [f"archive:{name}"],
         "content_hash": f"hash:{name}"},
        episode_id="episode", target_key="match", action_id=name, request_id=f"request-{name}",
        sequence=1,
        prediction_cutoff=("2026-09-29T10:00:00Z" if requested.startswith("2026-09-28") else "2026-09-27T10:00:00Z"),
        requested_at=requested,
        captured_at=captured, source_version=version,
    )


def test_future_disjoint_holdout_scores_availability_rate_and_losses():
    train = [row(f"train-{i}", dependency=f"train-dep-{i}") for i in range(5)]
    hold = [row(f"hold-{i}", status="available" if i < 3 else "failed", dependency=f"hold-dep-{i}",
                requested="2026-09-28T08:00:00Z", captured="2026-09-28T08:10:00Z") for i in range(5)]
    report = evaluate_future_holdout(train, hold, min_training_dependencies=5, min_holdout_dependencies=5)[0]
    assert report.status == "SUPPORTED_HOLDOUT_DIAGNOSTIC"
    assert report.predicted_available_rate == 1.0
    assert report.observed_available_rate == .6
    assert report.availability_rate_gap == -.4
    assert report.binary_log_loss > 0
    assert report.brier_score == pytest.approx(.4)


def test_overlap_and_non_future_splits_are_explicit():
    train = [row(f"train-{i}", dependency=f"shared-{i}") for i in range(5)]
    overlap = [row(f"hold-{i}", dependency=f"shared-{i}", requested="2026-09-28T08:00:00Z",
                   captured="2026-09-28T08:10:00Z") for i in range(5)]
    assert evaluate_future_holdout(train, overlap, min_training_dependencies=1, min_holdout_dependencies=1)[0].status == "OVERLAPPING_DEPENDENCE_CLUSTER"
    nonfuture = [row(f"hold-{i}", dependency=f"future-{i}", requested="2026-09-27T07:00:00Z",
                     captured="2026-09-27T07:10:00Z") for i in range(5)]
    assert evaluate_future_holdout(train, nonfuture, min_training_dependencies=1, min_holdout_dependencies=1)[0].status == "NON_FUTURE_HOLDOUT"


def test_insufficient_support_and_source_version_mismatch_are_reported():
    train = [row("train", dependency="d1")]
    hold = [row("hold", version="v2", dependency="d2", requested="2026-09-28T08:00:00Z",
                captured="2026-09-28T08:10:00Z")]
    assert evaluate_future_holdout(train, hold, min_training_dependencies=2, min_holdout_dependencies=2)[0].status == "NO_MATCHING_HOLDOUT"
    hold_same = [row("hold", dependency="d2", requested="2026-09-28T08:00:00Z", captured="2026-09-28T08:10:00Z")]
    assert evaluate_future_holdout(train, hold_same, min_training_dependencies=2, min_holdout_dependencies=2)[0].status == "INSUFFICIENT_INDEPENDENT_SUPPORT"
