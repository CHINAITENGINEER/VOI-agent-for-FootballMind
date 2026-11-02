"""Synthetic mechanism tests: outputs never enter the real-data artifact folder."""
from datetime import datetime, timedelta, timezone
import json

from football_research.evaluation.runner import evaluate, make_snapshots, quote_features


def constructed_matches(n=90):
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [dict(match_id=f"test-{i}", competition_id="test", home_team_id=f"t{i % 4}",
                 away_team_id=f"t{(i + 1) % 4}", kickoff=start + timedelta(days=i),
                 fixture_available_at=start + timedelta(days=i - 5),
                 result_available_at=start + timedelta(days=i, hours=3),
                 home_goals=i % 3, away_goals=(i + 1) % 3, status="finished", score_period="regulation",
                 information_mode="operational", input_manifest=["synthetic-unit-fixture"])
            for i in range(n)]


def test_realistic_late_archive_has_zero_retrospective_test_predictions(tmp_path):
    rows = constructed_matches()
    observed = rows[-1]["kickoff"] + timedelta(days=1)
    rows = [dict(r, fixture_available_at=observed, result_available_at=observed) for r in rows]
    report = evaluate(rows, dataset_version="test-only", information_mode="operational", output_dir=tmp_path,
                      initial_validation_start="2024-02-01T00:00:00Z", models=["frequency"])
    assert report["status"] == "NO_ELIGIBLE_TEST_PREDICTIONS"
    assert report["test_predictions"] == 0
    assert report["metrics"] == {}


def test_time_backtest_runs_with_multiple_leads_and_repeatable_output(tmp_path):
    rows = constructed_matches()
    report = evaluate(rows, dataset_version="test-only", information_mode="operational", output_dir=tmp_path,
                      initial_validation_start="2024-02-01T00:00:00Z", validation_days=10, test_days=10,
                      models=["frequency"], min_train_matches=10)
    assert report["status"] == "OK"
    assert report["test_matches"] == 49
    assert report["test_predictions"] == 98
    predictions = [json.loads(line) for line in (tmp_path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    for row in predictions:
        assert row["match_id"] not in row["input_manifest"]["training_match_ids"]
        assert datetime.fromisoformat(row["input_manifest"]["train_cutoff"]) <= datetime.fromisoformat(row["prediction_cutoff"])
    assert (tmp_path / "calibration-frequency-1h.png").is_file()


def test_quote_selection_rejects_future_and_mixed_mode():
    target = constructed_matches(1)[0]
    cutoff = target["kickoff"] - timedelta(hours=1)
    quote = dict(match_id=target["match_id"], observed_at=cutoff - timedelta(minutes=2), home_price=2., draw_price=3., away_price=4.,
                 period="regulation", information_mode="operational", provider="test", bookmaker="test", input_manifest=["test-quote"])
    future = dict(quote, observed_at=cutoff + timedelta(seconds=1), home_price=99)
    features = quote_features(target, cutoff, [quote, future])
    assert features["odds_home"] == 2
    assert quote_features(target, cutoff, [dict(quote, information_mode="historical_reconstruction")]) is None
