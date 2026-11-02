from datetime import datetime

from football_spider.models.match import HistoryMatch, Match, MatchHistory, Odds
from football_spider.storage.database import Database


def test_upsert_deduplicates(tmp_path) -> None:
    match = Match(
        id="42",
        time=datetime(2026, 9, 20, 18, 0),
        league="测试联赛",
        home_team="主队",
        away_team="客队",
        odds=Odds(home=2.0, draw=3.0, away=4.0),
    )
    with Database(tmp_path / "matches.db") as database:
        match.history = MatchHistory(
            head_to_head=[HistoryMatch(id="old", home_team="A", away_team="B")]
        )
        database.upsert_match(match, full_detail=True)
        changed = match.model_copy(deep=True)
        changed.odds.home = 1.9
        changed.history = MatchHistory()
        database.upsert_match(changed)
        assert database.count_matches() == 1
        row = database.connection.execute(
            "SELECT odds_home, details_json FROM matches WHERE id='42'"
        ).fetchone()
        assert row["odds_home"] == 1.9
        saved = Match.model_validate_json(row["details_json"])
        assert saved.odds.home == 1.9
        assert saved.history.head_to_head[0].id == "old"
