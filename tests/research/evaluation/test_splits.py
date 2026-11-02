"""Synthetic leakage tests; generated fixtures are never reported as real games."""

from datetime import datetime, timedelta, timezone

import pytest

from football_research.evaluation.splits import make_temporal_fold, rolling_origin_splits


def at(day, hour=12):
    return datetime(2024, 1, day, hour, tzinfo=timezone.utc)


def row(match_id, day, *, lead=1, available=None):
    kickoff = at(day)
    return {
        "match_id": match_id, "kickoff": kickoff,
        "prediction_cutoff": kickoff - timedelta(hours=lead),
        "result_available_at": available or kickoff + timedelta(hours=3),
        "home_goals": 1, "away_goals": 0,
    }


def split(rows):
    return make_temporal_fold(rows, validation_start=at(10, 0), test_start=at(20, 0), test_end=at(30, 0))


def test_groups_snapshots_by_match_and_keeps_simultaneous_round_together():
    rows = [row("train", 5), row("validation", 12), row("test", 20), row("test", 20, lead=24), row("same-round", 20)]
    result = split(rows)
    assert {r["match_id"] for r in result.train} == {"train"}
    assert {r["match_id"] for r in result.validation} == {"validation"}
    assert {r["match_id"] for r in result.test} == {"test", "same-round"}
    assert len(result.test) == 3
    assert result.manifest()["test_matches"] == 2


def test_long_lead_cutoff_purges_nominally_earlier_training_labels():
    rows = [row("early", 5), row("too-late", 9), row("validation", 10, lead=72), row("test", 21)]
    result = split(rows)
    assert result.train_cutoff == at(7)
    assert [r["match_id"] for r in result.train] == ["early"]
    assert result.purged_train_matches == 1


def test_training_result_exactly_prediction_cutoff_is_purged():
    rows = [row("late", 5, available=at(7)), row("validation", 10, lead=72), row("test", 21)]
    result = split(rows)
    assert len(result.train) == 0
    assert result.purged_train_matches == 1


def test_training_result_at_nominal_train_boundary_is_allowed_if_predictions_later():
    rows = [row("boundary", 5, available=at(10, 0)), row("validation", 12), row("test", 21)]
    result = split(rows)
    assert len(result.train) == 1
    assert result.train_cutoff == at(10, 0)


def test_validation_labels_purged_before_earliest_test_cutoff_for_tuning():
    rows = [row("train", 5), row("early-val", 12), row("late-val", 18), row("late-val", 18, lead=3), row("test", 20, lead=72)]
    result = split(rows)
    assert [r["match_id"] for r in result.validation] == ["early-val"]
    assert result.validation_label_cutoff == at(17)
    assert result.purged_validation_matches == 1


def test_missing_result_visibility_purges_whole_training_match():
    first, second = row("train", 5), row("train", 5, lead=24)
    second["result_available_at"] = None
    result = split([first, second, row("validation", 12), row("test", 21)])
    assert not result.train
    assert result.purged_train_matches == 1


def test_timezones_normalized_but_naive_values_rejected():
    value = row("train", 5)
    value["kickoff"] = "2024-01-05T20:00:00+08:00"
    result = split([value, row("validation", 12), row("test", 21)])
    assert len(result.train) == 1
    value["prediction_cutoff"] = "2024-01-05T11:00:00"
    with pytest.raises(ValueError, match="timezone-aware"):
        split([value])


def test_postponement_conflicting_kickoffs_require_resolution():
    with pytest.raises(ValueError, match="conflicting kickoff"):
        split([row("same-id", 5), row("same-id", 6)])


def test_duplicate_snapshot_cutoffs_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        split([row("same-id", 5), row("same-id", 5)])


def test_conflicting_final_scores_require_upstream_resolution():
    first, second = row("same-id", 5), row("same-id", 5, lead=24)
    second["home_goals"] = 2
    with pytest.raises(ValueError, match="conflicting final labels"):
        split([first, second])


def test_postmatch_cutoffs_and_impossible_result_times_rejected():
    with pytest.raises(ValueError, match="must not follow"):
        split([row("postmatch", 5, lead=-1)])
    with pytest.raises(ValueError, match="must not precede"):
        split([row("impossible", 5, available=at(4))])


def test_rolling_expands_train_and_never_repeats_test_matches():
    rows = [row(str(day), day) for day in range(1, 30)]
    folds = rolling_origin_splits(rows, initial_validation_start=at(8, 0), validation_window=timedelta(days=5), test_window=timedelta(days=5))
    assert len(folds) == 4
    observed = set()
    previous_train = set()
    for fold in folds:
        train = {r["match_id"] for r in fold.train}
        test = {r["match_id"] for r in fold.test}
        assert not observed & test
        assert previous_train <= train
        assert all(r["result_available_at"] <= fold.train_cutoff for r in fold.train)
        observed |= test
        previous_train = train


def test_reject_overlapping_test_windows_and_invalid_boundaries():
    with pytest.raises(ValueError, match="at least test_window"):
        rolling_origin_splits([row("a", 10)], initial_validation_start=at(5), validation_window=timedelta(days=2), test_window=timedelta(days=3), step=timedelta(days=1))
    with pytest.raises(ValueError, match="validation_start"):
        make_temporal_fold([], validation_start=at(10), test_start=at(5), test_end=at(20))
