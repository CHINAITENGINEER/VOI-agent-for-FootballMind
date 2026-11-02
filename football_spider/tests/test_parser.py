from datetime import date

from football_spider.parser.match_parser import MatchParser


def test_list_and_detail_mapping() -> None:
    list_payload = {
        "code": 0,
        "data": [{
            "dateFormat": "2026-09-20 周日",
            "list": [{
                "matchId": 2929733,
                "matchTime": "2026-09-20T18:00:00",
                "leagueChs": "韩职",
                "homeChs": "仁川联",
                "awayChs": "大田市民",
                "footMatchState": 0,
                "spfWinFoot": 2.88,
                "spfEqualFoot": 3.05,
                "spfLoseFoot": 2.2,
                "goalFoot": 1,
                "homeRank": "8",
                "awayRank": "6",
            }],
        }],
    }
    parser = MatchParser()
    matches = parser.parse_match_list(
        list_payload, date(2026, 9, 20), date(2026, 9, 27)
    )
    assert len(matches) == 1
    assert matches[0].id == "2929733"
    assert matches[0].odds.home == 2.88

    detail = {
        "code": 0,
        "data": {
            "matchId": 2929733,
            "matchTime": "2026-09-20T18:00:00",
            "home": "仁川联",
            "away": "大田市民",
            "league": "韩职",
            "homeScore": 2,
            "awayScore": 1,
            "homeHalfScore": 1,
            "awayHalfScore": 0,
            "footMatchState": -1,
            "finishedTitle": "完场",
            "battleList": [{
                "matchId": 1,
                "matchTime": "2026-05-05T15:30:00",
                "home": "大田市民",
                "away": "仁川联",
                "homeScore": 0,
                "awayScore": 0,
                "homeHalfScore": 0,
                "awayHalfScore": 0,
                "league": "韩K联",
                "footMatchState": -1,
            }],
        },
    }
    odds = {
        "code": 0,
        "data": {"footballMoreSpInfo": {
            "spf_win": 2.8,
            "spf_draw": 3.0,
            "spf_lost": 2.3,
            "rfspf_goal": "+1",
            "rfspf_win": 1.5,
            "rfspf_draw": 3.8,
            "rfspf_lost": 4.8,
            "homerank": "8",
            "awayrank": "6",
            "t0": 11,
        }},
    }
    match = parser.merge_match_detail(matches[0], detail, odds)
    assert match.score.home == 2
    assert match.half_score.away == 0
    assert match.handicap.line == "+1"
    assert match.over_under.line is None
    assert match.extras["total_goals_exact_odds"] == {"t0": 11}
    assert len(match.history.head_to_head) == 1
