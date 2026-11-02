"""Synthetic mechanisms for the offline platform consumer (platform venv)."""
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("psycopg", reason="Platform consumer uses the platform venv's existing dependencies")

from football_spider.models.contracts import EntityRef, MatchRecord, Score, TeamStatistic
from football_research.data_adapters.reconstruction_bundle import feature_rows, ASSUMPTIONS
from football_research.data_adapters.batch_features import FeatureIndex


def match(ident, hours):
    return MatchRecord(source_id=ident, competition=EntityRef(source_id="EPL", name="Synthetic"),
        home=EntityRef(source_id="1", name="A"), away=EntityRef(source_id="2", name="B"),
        kickoff=datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(hours=hours),
        score=Score(home=2, away=1), status="finished", score_period="regulation",
        original_time="synthetic", source_timezone="UTC")


def stats(ident, value):
    return [TeamStatistic(source_match_id=ident, source_team_id=team, metric=metric,
        value=value, provider="synthetic", definition_version="test")
        for team in ("1", "2") for metric in ("xg", "xga")]


def rows(matches, statistics=()):
    return feature_rows(matches, statistics, source_evidence=[{"synthetic": True}], assumptions=ASSUMPTIONS, leads=(1,))


def test_result_delay_and_separate_xg_delay():
    before_xg = rows([match("1", 0), match("2", 10)], stats("1", 1.5))[-1]
    assert before_xg["features"]["home_elo"] > 1500
    assert before_xg["features"]["home_rolling_xg_for"] is None
    assert before_xg["input_manifest"]["xg_history_match_ids"] == []
    after_xg = rows([match("1", 0), match("2", 30)], stats("1", 1.5))[-1]
    assert after_xg["features"]["home_rolling_xg_for"] == 1.5
    assert after_xg["input_manifest"]["history_inputs"][0]["xg_available_at"] < after_xg["prediction_cutoff"]
    before_result = rows([match("1", 0), match("2", 2)])[-1]
    assert before_result["features"]["home_elo"] is None
    assert before_result["input_manifest"]["history_inputs"] == []


def test_target_scores_xg_and_later_games_do_not_change_earlier_features():
    base = rows([match("1", 0), match("2", 30)], stats("1", 1.5))[-1]
    poisoned = match("2", 30).model_copy(update={"score": Score(home=99, away=0)})
    expanded = rows([match("1", 0), poisoned, match("3", 60)], stats("1", 1.5) + stats("2", 99) + stats("3", 99))
    assert base == expanded[1]
    FeatureIndex.from_rows([base], strict=True)


def test_rolling_xg_uses_only_last_five_available_games():
    matches = [match(str(i), i * 48) for i in range(7)]
    statistics = sum((stats(str(i), float(i)) for i in range(7)), [])
    target = rows(matches, statistics)[-1]
    assert target["features"]["home_rolling_xg_for"] == 3.0  # prior 1,2,3,4,5
    assert "understat:match:0" not in target["input_manifest"]["xg_history_match_ids"]


def test_invalid_time_assumptions_and_duplicate_stats_rejected():
    with pytest.raises(ValueError, match="Duplicate normalized statistic"):
        rows([match("1", 0)], stats("1", 1) * 2)
    with pytest.raises(ValueError, match="delay order"):
        feature_rows([], [], source_evidence=[], assumptions={**ASSUMPTIONS, "xg_delay_hours": 1})
