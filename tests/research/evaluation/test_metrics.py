"""Synthetic mechanism tests; none of these rows are real evaluation evidence."""

from datetime import datetime, timedelta, timezone
import math

import pytest

from football_research.evaluation.metrics import (
    paired_model_comparison, reliability_table, save_calibration_plot,
    score_by_lead_time, score_predictions,
)


def row(match_id="fixture-1", probabilities=(0.5, 0.25, 0.25), kickoff=None, lead=1, goals=(2, 0)):
    kickoff = kickoff or datetime(2024, 1, 1, 15, tzinfo=timezone.utc)
    return {
        "match_id": match_id, "kickoff": kickoff, "prediction_cutoff": kickoff - timedelta(hours=lead),
        "home_goals": goals[0], "away_goals": goals[1],
        **dict(zip(("home_win_probability", "draw_probability", "away_win_probability"), probabilities)),
    }


def test_known_scoring_values_and_unscaled_brier():
    result = score_predictions([row()])
    assert result["log_loss"] == pytest.approx(math.log(2))
    assert result["brier_score"] == pytest.approx(0.375)
    assert result["accuracy"] == 1
    assert result["n_matches"] == 1


def test_zero_probability_is_finite_and_empty_scores_are_missing():
    result = score_predictions([row(probabilities=(0, 0, 1))])
    assert result["log_loss"] == pytest.approx(-math.log(1e-15))
    assert result["brier_score"] == 2
    assert score_predictions([])["log_loss"] is None


@pytest.mark.parametrize("probabilities", [(0.4, 0.2, 0.1), (-0.1, 0.5, 0.6), (float("nan"), 0, 1), (float("inf"), 0, 0)])
def test_invalid_probabilities_fail_closed(probabilities):
    with pytest.raises(ValueError):
        score_predictions([row(probabilities=probabilities)])


@pytest.mark.parametrize("goals", [(None, 0), (-1, 0), (True, 0), ("2", 0)])
def test_missing_or_malformed_outcomes_are_not_imputed(goals):
    with pytest.raises(ValueError):
        score_predictions([row(goals=goals)])


def test_reliability_has_empty_bins_and_probability_one_is_last_bin():
    result = reliability_table([row(probabilities=(1, 0, 0))], n_bins=5)
    assert len(result) == 15
    assert sum(cell["count"] for cell in result) == 3
    home = [cell for cell in result if cell["class"] == "home_win"]
    assert home[0]["mean_probability"] is None
    assert home[4]["count"] == 1
    assert home[4]["observed_frequency"] == 1


def test_calibration_is_real_noninteractive_png(tmp_path):
    pytest.importorskip("matplotlib")
    path = save_calibration_plot([row()], tmp_path / "calibration.png")
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert path.stat().st_size > 1000


def test_lead_time_boundary_and_timezone_conversion():
    result = score_by_lead_time([row(lead=1), row("fixture-2", lead=24)])
    assert [bucket["n_predictions"] for bucket in result] == [0, 1, 0, 1, 0]
    invalid = row()
    invalid["prediction_cutoff"] = "2024-01-01T14:00:00"
    with pytest.raises(ValueError, match="timezone-aware"):
        score_by_lead_time([invalid])


def test_pairing_sign_coverage_and_small_sample_suppression():
    left = [row(probabilities=(0.8, 0.1, 0.1)), row("unpaired")]
    right = [row(probabilities=(0.5, 0.25, 0.25))]
    result = paired_model_comparison(left, right)
    assert result["paired_predictions"] == 1
    assert result["mean_log_loss_difference"] == pytest.approx(-math.log(0.8) + math.log(0.5))
    assert result["left_prediction_coverage"] == 0.5
    assert result["confidence_intervals"]["log_loss"] is None
    assert result["interval_status"] == "insufficient_matches_or_weeks"


def test_pairing_normalizes_equivalent_timezone_cutoffs():
    left, right = row(), row()
    right["prediction_cutoff"] = "2024-01-01T22:00:00+08:00"
    result = paired_model_comparison([left], [right])
    assert result["paired_predictions"] == 1
    assert result["mean_log_loss_difference"] == 0


def test_pairs_reject_duplicate_keys_and_inconsistent_labels():
    with pytest.raises(ValueError, match="duplicate"):
        paired_model_comparison([row(), row()], [row()])
    with pytest.raises(ValueError, match="outcome"):
        paired_model_comparison([row()], [row(goals=(0, 0))])


def test_multisnapshot_matches_have_equal_weight():
    left = [row("a", (0.8, 0.1, 0.1), lead=lead) for lead in (1, 6, 24)]
    left.append(row("b", (0.2, 0.4, 0.4)))
    right = [row("a", lead=lead) for lead in (1, 6, 24)] + [row("b")]
    result = paired_model_comparison(left, right)
    assert result["paired_predictions"] == 4
    assert result["paired_matches"] == 2
    assert result["mean_log_loss_difference"] == pytest.approx((math.log(0.5 / 0.8) + math.log(0.5 / 0.2)) / 2)


def test_week_cluster_bootstrap_reproducible_and_both_thresholds_required():
    start = datetime(2024, 1, 1, 15, tzinfo=timezone.utc)
    left, right = [], []
    for i in range(40):
        kickoff = start + timedelta(weeks=i // 5, hours=i % 5)
        home_probability = 0.55 + 0.02 * (i % 3)
        left.append(row(str(i), (home_probability, (1 - home_probability) / 2, (1 - home_probability) / 2), kickoff))
        right.append(row(str(i), kickoff=kickoff))
    result = paired_model_comparison(left, right, bootstrap_samples=100, seed=7)
    assert result == paired_model_comparison(left, right, bootstrap_samples=100, seed=7)
    assert result["utc_calendar_weeks"] == 8
    low, high = result["confidence_intervals"]["log_loss"]
    assert low <= result["mean_log_loss_difference"] <= high < 0
    assert paired_model_comparison(left[:30], right[:30])["interval_status"] != "estimated"
    assert paired_model_comparison(left[::2], right[::2])["interval_status"] != "estimated"


def test_ci_does_not_overweight_matches_with_more_snapshots():
    start = datetime(2024, 1, 1, 15, tzinfo=timezone.utc)
    left, right, expanded_left, expanded_right = [], [], [], []
    for index in range(40):
        kickoff = start + timedelta(weeks=index // 5, hours=index % 5)
        probabilities = (0.8, 0.1, 0.1) if index % 2 else (0.2, 0.4, 0.4)
        left.append(row(str(index), probabilities, kickoff))
        right.append(row(str(index), kickoff=kickoff))
        # Winners receive many more archived snapshots than losers. Their
        # contribution to both point estimate and clustered CI must be unchanged.
        leads = (1, 3, 6, 12, 24) if index % 2 else (1,)
        expanded_left.extend(row(str(index), probabilities, kickoff, lead) for lead in leads)
        expanded_right.extend(row(str(index), kickoff=kickoff, lead=lead) for lead in leads)
    baseline = paired_model_comparison(left, right, bootstrap_samples=100, seed=42)
    expanded = paired_model_comparison(expanded_left, expanded_right, bootstrap_samples=100, seed=42)
    assert expanded["paired_predictions"] > baseline["paired_predictions"]
    assert expanded["mean_log_loss_difference"] == baseline["mean_log_loss_difference"]
    assert expanded["confidence_intervals"] == baseline["confidence_intervals"]
