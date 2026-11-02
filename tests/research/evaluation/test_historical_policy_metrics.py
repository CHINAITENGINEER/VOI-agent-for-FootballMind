from datetime import datetime
import json

import pytest

from football_research.evaluation.historical_policy_metrics import compare_historical_policies
from football_research.evaluation.historical_protocol import EpisodeFold


def snapshot(match_id="m1", kickoff="2025-01-10T12:00:00Z", home=1, away=0):
    return {
        "match_id": match_id,
        "kickoff": kickoff,
        "prediction_cutoff": "2025-01-10T11:00:00Z" if match_id == "m1" else "2025-01-11T11:00:00Z",
        "home_goals": home,
        "away_goals": away,
    }


def episode(match_id, cutoff, strategy, probabilities, *, seed=0, brier=None, kickoff=None):
    row = {
        "match_id": match_id,
        "prediction_cutoff": cutoff,
        "strategy": strategy,
        "seed": seed,
        "final_probabilities": probabilities,
        "cost": {"money": 0.01 if strategy != "base-only" else 0.0, "requests": 1.0 if strategy != "base-only" else 0.0},
    }
    if brier is not None:
        row["brier_score"] = brier
    if kickoff is not None:
        row["kickoff"] = kickoff
    return row


def fold_for(*match_ids):
    return EpisodeFold(
        index=0,
        train_start=None,
        validation_start=datetime.fromisoformat("2025-01-01T00:00:00+00:00"),
        test_start=datetime.fromisoformat("2025-01-01T00:00:00+00:00"),
        test_end=datetime.fromisoformat("2025-02-01T00:00:00+00:00"),
        train_match_ids=(), validation_match_ids=(), test_match_ids=tuple(match_ids),
        label_cutoff=datetime.fromisoformat("2025-01-01T00:00:00+00:00"),
    )


def test_pairs_exact_keys_collapses_duplicates_and_is_deterministic():
    snaps = [snapshot(), snapshot("m2", "2025-01-11T12:00:00Z", home=0, away=0)]
    rows = [
        episode("m1", snaps[0]["prediction_cutoff"], "base-only", [0.6, 0.2, 0.2]),
        episode("m1", snaps[0]["prediction_cutoff"], "base-only", [0.6, 0.2, 0.2]),
        episode("m1", snaps[0]["prediction_cutoff"], "voi_gate", [0.8, 0.1, 0.1]),
        episode("m1", snaps[0]["prediction_cutoff"], "random", [0.7, 0.2, 0.1], seed=1),
        episode("m1", snaps[0]["prediction_cutoff"], "random", [0.5, 0.3, 0.2], seed=2),
        episode("m2", snaps[1]["prediction_cutoff"], "base-only", [0.3, 0.5, 0.2]),
        episode("m2", snaps[1]["prediction_cutoff"], "voi_gate", [0.2, 0.7, 0.1]),
        episode("unbound", snaps[0]["prediction_cutoff"], "base-only", [0.3, 0.4, 0.3]),
    ]
    first = compare_historical_policies(rows, snaps, folds=(fold_for("m1", "m2"),), bootstrap_samples=80)
    second = compare_historical_policies(rows, snaps, folds=(fold_for("m1", "m2"),), bootstrap_samples=80)
    assert first == second
    assert first["status"] == "PASS"
    assert first["coverage"]["duplicate_rows_collapsed"] == 1
    assert first["coverage"]["excluded_rows"]["unbound_snapshot"] == 1
    assert first["paired_vs_baseline"]["voi_gate"]["paired_matches"] == 2
    assert first["paired_vs_baseline"]["voi_gate"]["mean_brier_score"] < 0
    encoded = json.dumps(first, ensure_ascii=False)
    assert '"home_goals"' not in encoded
    assert '"outcome"' not in encoded


def test_rescored_metric_mismatch_is_an_error():
    snaps = [snapshot()]
    rows = [
        episode("m1", snaps[0]["prediction_cutoff"], "base-only", [0.6, 0.2, 0.2], brier=0.0),
    ]
    report = compare_historical_policies(rows, snaps, folds=(fold_for("m1"),), bootstrap_samples=10)
    assert report["status"] == "FAIL"
    assert report["issue_counts"]["EXPORTED_BRIER_MISMATCH"] == 1


def test_episode_kickoff_mismatch_is_rejected():
    snaps = [snapshot()]
    rows = [
        episode("m1", snaps[0]["prediction_cutoff"], "base-only", [0.6, 0.2, 0.2], kickoff="2025-01-10T13:00:00Z"),
    ]
    report = compare_historical_policies(rows, snaps, folds=(fold_for("m1"),), bootstrap_samples=10)
    assert report["status"] == "FAIL"
    assert report["issue_counts"]["EPISODE_KICKOFF_MISMATCH"] == 1


def test_multiple_budget_archive_requires_explicit_slice():
    snaps = [snapshot()]
    cutoff = snaps[0]["prediction_cutoff"]
    rows = [
        {**episode("m1", cutoff, "base-only", [0.6, 0.2, 0.2]), "budget_money": 0.0},
        {**episode("m1", cutoff, "voi_gate", [0.7, 0.2, 0.1]), "budget_money": 0.02},
    ]
    with pytest.raises(ValueError, match="multiple budget_money values"):
        compare_historical_policies(rows, snaps, folds=(fold_for("m1"),), bootstrap_samples=10)


def test_explicit_budget_slice_is_comparison_eligible():
    snaps = [snapshot()]
    cutoff = snaps[0]["prediction_cutoff"]
    rows = [
        {**episode("m1", cutoff, "base-only", [0.6, 0.2, 0.2]), "budget_money": 0.02},
        {**episode("m1", cutoff, "voi_gate", [0.7, 0.2, 0.1]), "budget_money": 0.02},
    ]
    report = compare_historical_policies(
        rows, snaps, folds=(fold_for("m1"),), budget_money=0.02, bootstrap_samples=20
    )
    assert report["status"] == "PASS"
    assert report["comparison_eligible"] is True
    assert report["coverage"]["excluded_rows"] == {}
    assert set(report["paired_vs_baseline"]) == {"voi_gate"}
