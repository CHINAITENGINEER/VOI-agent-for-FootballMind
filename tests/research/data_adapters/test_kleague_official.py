from copy import deepcopy
from datetime import datetime, timezone
import json

import pytest

from football_research.data_adapters.kleague_official import (
    ALIASES, ENDPOINT, KLeagueOfficialSchedule, parameters, parse)

AT = "2026-10-25T08:50:00+00:00"
OBS = {"observed_at": AT, "id": "source-observation", "content_hash": "a"*64}


def row(**changes):
    return {"year": 2026, "leagueId": 1, "roundId": 33, "gameId": 193,
        "gameDate": "2026.10.24", "gameTime": "14:00", "endYn": "Y",
        "meetName": "하나은행 K리그1 2026", "homeTeam": "K01", "homeTeamName": "울산",
        "awayTeam": "K03", "awayTeamName": "포항", "homeGoal": 2, "awayGoal": 2,
        "gameStatus": "FE", **changes}


def payload(rr):
    return {"resultCode": "200", "data": {"scheduleList": rr, "clubList": [
        {"teamId": "K01", "teamNameFull": "울산 HD", "homepage": "https://www.uhfc.tv"}]}}


def test_actual_official_korean_clock_and_composite_source_identity():
    matches, audit = parse(payload([row()]), OBS, year=2026, month=10)
    r = matches[0]
    assert r["match_id"] == "kleague:match:2026:1:193"
    assert r["kickoff"] == "2026-10-24T05:00:00+00:00"
    assert r["source_timezone"] == "Asia/Seoul"
    assert r["home_team_source_id"] == "K01" and r["away_team_source_id"] == "K03"
    assert r["status"] == "finished" and r["home_score"] == r["away_score"] == 2
    assert r["club_metadata"]["home"]["homepage"] == "https://www.uhfc.tv"
    assert r["quote_snapshot"] is None and r["supported"] is False
    assert not audit["market_prior_available"]


def test_k1_and_k2_stay_independent_and_future_zero_scores_are_not_results():
    source = row(leagueId=2, gameDate="2026.10.31", gameTime="14:00", gameStatus="", endYn="N", homeGoal=0, awayGoal=0)
    matches, _ = parse(payload([source]), OBS, year=2026, month=10, league_id=2)
    r = matches[0]
    assert r["match_id"] == "kleague:match:2026:2:193"
    assert r["competition_key"] == "K2联赛" and r["status"] == "scheduled"
    assert r["home_score"] is r["away_score"] is None
    assert ALIASES["韩职"] == "K联赛" and ALIASES["K League 2"] == "K2联赛"


def test_live_second_half_is_not_scheduled():
    source = row(gameDate="2026.10.25", gameTime="16:30", gameStatus="2S 18", endYn="N")
    matches, _ = parse(payload([source]), OBS, year=2026, month=10)
    assert matches[0]["status"] == "live"


@pytest.mark.parametrize("change", [{"year": 2021}, {"leagueId": 2},
    {"gameDate": "2026.11.01"}, {"gameTime": "TBD"}, {"homeTeam": "K03"}, {"gameId": True}])
def test_stale_or_ambiguous_rows_are_explicitly_rejected(change):
    matches, audit = parse(payload([row(**change)]), OBS, year=2026, month=10)
    assert not matches and len(audit["rejected"]) == 1


def test_source_error_does_not_become_an_empty_success():
    with pytest.raises(ValueError):
        parse({"resultCode": "500", "data": {"scheduleList": []}}, OBS, year=2026, month=10)


def test_duplicate_conflicting_time_cannot_silently_overwrite_event():
    matches, audit = parse(payload([row(), row(gameTime="15:00")]), OBS, year=2026, month=10)
    assert not matches and audit["rejected"][0]["reason"] == "conflicting_duplicate_event"


def test_exact_duplicate_is_counted_once():
    matches, audit = parse(payload([row(), row()]), OBS, year=2026, month=10)
    assert len(matches) == 1 and audit["identical_duplicate_ids"] == [matches[0]["match_id"]]


def test_empty_official_schedule_is_observed_empty_not_missing_market_fabrication():
    matches, audit = parse(payload([]), OBS, year=2026, month=11)
    assert not matches and audit["response_rows"] == 0


def test_one_bounded_json_post_contract_is_archived(tmp_path):
    calls = []
    def transport(params):
        calls.append(params)
        return 200, json.dumps(payload([row()]), ensure_ascii=False).encode()
    adapter = KLeagueOfficialSchedule(tmp_path, transport=transport,
        clock=lambda: datetime(2026,10,25,8,50,tzinfo=timezone.utc))
    result = adapter.collect_month(2026,10)
    assert calls == [parameters(2026,10)] and result["requests"] == 1
    assert result["status"] == "ready" and result["method"] == "POST"
    observations = adapter.archive.observations()
    assert len(observations) == 1 and observations[0].endpoint == ENDPOINT
    assert adapter.archive.read_body(observations[0])


def test_forbidden_response_kept_as_failed_no_retry(tmp_path):
    calls = []
    adapter = KLeagueOfficialSchedule(tmp_path,
        transport=lambda params: (calls.append(params) or (403,b"Forbidden")))
    result = adapter.collect_month(2026,10)
    assert result["status"] == "failed" and not result["matches"] and len(calls) == 1
    assert len(adapter.archive.observations()) == 1
