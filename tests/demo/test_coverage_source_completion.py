from copy import deepcopy
import json

import pytest

from demo.daily_fixtures import merge_schedule
from demo.live_g import LiveBackend
from demo.live_coverage import availability, catalog_match, summarize
from football_research.data_adapters.espn_scoreboard import parse

AT = "2026-10-25T08:25:00Z"
KICK = "2026-10-25T14:00:00Z"
OBS = {"observed_at": AT, "id": "test-observation", "content_hash": "a"*64}


def event():
    return {"id": "401878346", "date": KICK, "uid": "s:600~l:45~e:401878346",
        "competitions": [{"date": KICK, "timeValid": True, "altGameNote": "SPFL Premiership",
            "status": {"type": {"name": "STATUS_SCHEDULED", "state": "pre"}},
            "competitors": [
                {"homeAway": "home", "team": {"id": "266", "displayName": "Motherwell"}},
                {"homeAway": "away", "team": {"id": "250", "displayName": "St Mirren"}}],
            "odds": [{"provider": {"name": "Bet 365"},
                "homeTeamOdds": {"value": 1.76, "handicap": 0, "team": {"id": "266"}},
                "drawOdds": {"value": 3.8, "handicap": 0},
                "awayTeamOdds": {"value": 3.8, "handicap": 0, "team": {"id": "250"}}}]}]}


def primary(quote=False):
    return catalog_match({"fixture_id": "spftll:match:200", "source_match_id": "200",
        "home_team": "Motherwell", "away_team": "St Mirren", "home_team_source_id": "10",
        "away_team_source_id": "11", "competition_key": "苏超", "kickoff_at": KICK,
        "status": "scheduled", "inventory_quality": {"complete_1x2_observed": quote}})


def merged():
    secondary = parse({"events": [event()]}, OBS)[0][0]
    return merge_schedule([primary()], [secondary], AT)[0][0]


def manifest(row):
    return {"market_initialization": {"fixture": {
        "match_id": row["match_id"], "home_team_id": "espn:team:"+row["home_team_source_id"],
        "away_team_id": "espn:team:"+row["away_team_source_id"], "home_team_name": row["home"],
        "away_team_name": row["away"], "competition_name": row["competition_key"],
        "kickoff": row["kickoff"], "status": "scheduled", "input_manifest": ["raw-bound"]},
        "prediction_cutoff": "2026-10-25T13:59:00Z", "probabilities": [.4,.3,.3]}}


def test_observed_decimal_three_way_odds_and_exact_league_are_usable():
    row = parse({"events": [event()]}, OBS)[0][0]
    assert row["competition_key"] == "苏超" and row["competition_group"] == "europe_other"
    assert availability(row, AT)[0]
    q = row["quote_snapshot"]
    assert q["decimal_odds"] == {"home": 1.76, "draw": 3.8, "away": 3.8}
    assert sum(q["no_vig_probabilities"].values()) == pytest.approx(1)
    assert q["value_basis"] == "decimal_homeTeamOdds_drawOdds_awayTeamOdds"
    assert q["source_timestamp"] is None and q["observed_at"] == AT


@pytest.mark.parametrize("case", ["missing_draw", "boolean", "infinite", "invalid_price", "handicap", "wrong_team"])
def test_incomplete_or_wrong_market_never_grants_a_prior(case):
    e = event();o = e["competitions"][0]["odds"][0]
    if case == "missing_draw":o.pop("drawOdds")
    if case == "boolean":o["drawOdds"]["value"] = True
    if case == "infinite":o["drawOdds"]["value"] = float("inf")
    if case == "invalid_price":o["drawOdds"]["value"] = 1
    if case == "handicap":o["homeTeamOdds"]["handicap"] = 1
    if case == "wrong_team":o["homeTeamOdds"]["team"]["id"] = "250"
    row = parse({"events": [e]}, OBS)[0][0]
    assert row["quote_snapshot"] is None and not availability(row, AT)[0]


def test_argentine_actual_lpf_label_stays_mens_top_division():
    e = event();e["competitions"][0]["altGameNote"] = "Argentine LPF"
    row = parse({"events": [e]}, OBS)[0][0]
    assert row["competition_key"] == "阿甲" and row["competition_group"] == "americas"


def test_secondary_market_routes_real_id_and_same_identity(tmp_path, monkeypatch):
    row = merged();backend = LiveBackend();calls=[]
    def initialize(secondary, folder, error):
        calls.append(secondary);return manifest(secondary)
    monkeypatch.setattr(backend, "_initialize_espn", initialize)
    result = backend.initialize(row, tmp_path)
    assert row["match_id"] == "spftll:match:200"
    assert calls[0]["match_id"] == "espn:match:401878346"
    assert result["source_switch"]["display_match_id"] == row["match_id"]
    binding = backend.identity(result, tmp_path/"identity", row)
    assert binding["canonical"]["source_event_id"] == "espn:match:401878346"
    assert binding["canonical"]["home_name"] == "Motherwell"
    assert not binding["independently_confirmed"]


@pytest.mark.parametrize("field,value", [("kickoff", "2026-10-25T14:01:00Z"),
    ("home", "Different Club"), ("competition_key", "英超"),
    ("source_match_id", "999"), ("away_team_source_id", "266")])
def test_secondary_binding_cannot_initialize_another_match(tmp_path, monkeypatch, field, value):
    row = merged();row["espn_initialization_fixture"][field] = value
    backend = LiveBackend()
    monkeypatch.setattr(backend, "_initialize_espn", lambda *args: pytest.fail("Must reject before source request"))
    with pytest.raises(ValueError, match="备用来源"):
        backend.initialize(row, tmp_path)


def test_stale_secondary_quote_cannot_promote_missing_primary_market():
    s = parse({"events": [event()]}, OBS)[0][0];s["source_stale"] = True
    row = merge_schedule([primary()], [s], AT)[0][0]
    assert not row["quote_observed"] and row["initialization_source"] == "spftll"
    assert not availability(row, AT)[0]


def test_fresh_secondary_uses_its_clock_when_primary_has_no_market():
    p = primary();p["quote_catalog_observed_at"] = "2026-10-25T06:00:00Z"
    s = parse({"events": [event()]}, OBS)[0][0]
    row = merge_schedule([p], [s], AT)[0][0]
    assert row["quote_catalog_observed_at"] == AT and availability(row, AT)[0]


def test_coverage_marks_unobserved_mainstream_without_inventing_matches():
    row = merged();row["can_start"] = True
    summary = summarize([row])
    china = next(r for r in summary["mainstream_competitions"] if r["name"] == "中超")
    assert china["discovered"] == 0 and china["catalog_status"] == "not_observed_in_current_window"


def test_old_daily_cache_is_reprojected_without_http_or_new_clock(tmp_path):
    from football_research.data_adapters.espn_scoreboard import ESPNScoreboard, ENDPOINT, utc
    from football_spider.storage.raw_archive import atomic_write, json_bytes
    feed = ESPNScoreboard(tmp_path, transport=lambda params: pytest.fail("No new HTTP needed"), clock=lambda: utc(AT))
    body = json.dumps({"events": [event()]}).encode()
    obs = feed.archive.save("espn", ENDPOINT, {"dates": "20261025", "limit": 1000},
                            body, status=200, observed_at=utc(AT)).model_dump(mode="json")
    cached = {"schema": "footballmind-espn-daily-fixtures-v1", "date_utc": "2026-10-25",
              "observed_at": AT, "matches": [], "rejected": [], "observation": obs}
    atomic_write(tmp_path/"cache/2026-10-25.json", json_bytes(cached))
    result = feed._day(utc(AT).date(), utc(AT))
    assert result["projection_replayed"] and result["http_attempts"] == 0
    assert len(result["matches"]) == 1 and result["matches"][0]["quote_observed"]
    assert result["observed_at"] == AT
    assert result["matches"][0]["raw_observation"]["id"] == obs["id"]
