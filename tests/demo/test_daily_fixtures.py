import json
from copy import deepcopy
from datetime import timedelta

import pytest

from football_research.data_adapters.espn_scoreboard import ESPNScoreboard, competition_group, parse, utc, window
from demo.daily_fixtures import merge_schedule
from demo.live_coverage import availability, catalog_match
from demo.live_g import LiveGApplication, save

AT = "2026-10-19T15:00:00Z"
KICK = "2026-10-19T19:00:00Z"
OBS = {"observed_at": AT, "id": "observed-test", "content_hash": "a"*64}


def event(eid="123", *, kickoff=KICK, status="STATUS_SCHEDULED", comp="English Premier League"):
    return {"id": eid, "date": kickoff, "uid": "s:600~l:700~e:"+eid,
        "competitions": [{"date": kickoff, "timeValid": True, "altGameNote": comp,
            "status": {"type": {"name": status, "state": "in" if status == "STATUS_IN_PROGRESS" else "pre",
                                     "completed": status == "STATUS_FULL_TIME"}},
            "competitors": [
                {"homeAway": "away", "score": "1", "team": {"id": "2", "displayName": "Coventry City"}},
                {"homeAway": "home", "score": "2", "team": {"id": "1", "displayName": "Tottenham Hotspur"}},
            ]}], "links": [{"rel": ["summary"], "href": "https://www.espn.com/soccer/match/_/gameId/"+eid}]}


def source():
    return catalog_match({"fixture_id": "spftll:match:3003918", "source_match_id": "3003918",
        "home_team": "热刺", "away_team": "考文垂", "home_team_source_id": "33", "away_team_source_id": "57",
        "competition_key": "Premier League", "kickoff_at": KICK, "status": "scheduled",
        "inventory_quality": {"complete_1x2_observed": True}})


def espn(**kwargs):
    return parse({"events": [event(**kwargs)]}, OBS)[0][0]


def test_beijing_window_covers_utc_boundary_and_seven_dates():
    start, end, dates = window(AT)
    assert start.isoformat() == "2026-10-19T00:00:00+08:00"
    assert end.isoformat() == "2026-10-26T00:00:00+08:00"
    assert len(dates) == 8 and str(dates[0]) == "2026-10-18" and str(dates[-1]) == "2026-10-25"
    assert window("2026-10-19T16:00:00Z")[0].day == 20


def test_home_away_roles_not_array_order_and_pre_match_scores_not_invented():
    m = espn()
    assert m["home"] == "Tottenham Hotspur" and m["away"] == "Coventry City"
    assert m["home_score"] is m["away_score"] is None
    assert m["match_id"] == "espn:match:123" and m["competition_key"] == "Premier League"
    assert m["supported"] is False and availability(m, AT)[0] is False


def test_espn_complete_moneyline_is_an_observed_fallback_prior():
    m = event()
    m["competitions"][0]["odds"] = [{"provider": {"name": "DraftKings"}, "moneyline": {
        "home": {"close": {"odds": "+145"}}, "draw": {"close": {"odds": "+260"}},
        "away": {"close": {"odds": "+165"}}}}]
    parsed = parse({"events": [m]}, OBS)[0][0]
    assert parsed["quote_observed"] is True and parsed["supported"] is True
    assert parsed["analysis_source"] == "espn"
    assert parsed["quote_snapshot"]["provider"] == "DraftKings"
    probabilities = parsed["quote_snapshot"]["no_vig_probabilities"]
    assert set(probabilities) == {"home", "draw", "away"} and sum(probabilities.values()) == pytest.approx(1)
    assert availability(parsed, AT)[0]


@pytest.mark.parametrize("label,canonical,group", [
    ("Chinese Super League", "中超", "asia"),
    ("Chinese League One", "中甲", "asia"),
    ("Turkish Super Lig", "土超", "europe_other"),
    ("Scottish Premiership", "苏超", "europe_other"),
    ("Belgian Pro League", "比甲", "europe_other"),
    ("Argentine Primera División", "阿甲", "americas"),
    ("AFC Champions League Elite", "亚冠精英", "continental_club"),
])
def test_mainstream_competition_aliases_are_classified_without_analysis_promotion(label, canonical, group):
    from football_research.data_adapters.espn_scoreboard import COMPETITIONS
    assert COMPETITIONS[label] == canonical
    assert competition_group(canonical) == group


@pytest.mark.parametrize("status,expected", [("STATUS_IN_PROGRESS", "live"),
    ("STATUS_FULL_TIME", "finished"), ("STATUS_POSTPONED", "postponed"),
    ("STATUS_CANCELED", "cancelled"), ("UNKNOWN", "unknown")])
def test_statuses_and_scores(status, expected):
    m = espn(status=status)
    assert m["status"] == expected
    assert m["home_score"] == (2 if expected in {"live", "finished"} else None)
    assert not availability(m, AT)[0]


def test_malformed_event_is_reported_without_erasing_other_events():
    bad = event("124");bad["competitions"][0]["competitors"][1]["homeAway"] = "away"
    matches, rejected = parse({"events": [event(), bad]}, OBS)
    assert len(matches) == len(rejected) == 1 and rejected[0]["event_id"] == "124"
    with pytest.raises(ValueError):parse({"error": "bad upstream"}, OBS)


def test_unique_exact_join_keeps_real_analysis_id_and_raw_sources():
    rows, report = merge_schedule([source()], [espn()], AT)
    assert len(rows) == 1 and report["matched_rows"] == 1
    assert rows[0]["match_id"] == "spftll:match:3003918"
    assert rows[0]["espn_match_id"] == "espn:match:123"
    assert rows[0]["sources"] == ["spftll", "espn"] and availability(rows[0], AT)[0]
    assert rows[0]["display_date"] == "2026-10-20"


def test_exact_join_can_promote_observed_espn_moneyline_without_changing_identity():
    source_row = source()
    source_row["quote_observed"] = False
    source_row["inventory_quality"]["complete_1x2_observed"] = False
    secondary = event()
    secondary["competitions"][0]["odds"] = [{"provider": {"name": "DraftKings"}, "moneyline": {
        "home": {"close": {"odds": "+145"}}, "draw": {"close": {"odds": "+260"}},
        "away": {"close": {"odds": "+165"}}}}]
    rows, report = merge_schedule([source_row], [parse({"events": [secondary]}, OBS)[0][0]], AT)
    assert report["matched_rows"] == 1 and rows[0]["match_id"] == source_row["match_id"]
    assert rows[0]["quote_observed"] is True and rows[0]["quote_snapshot"]["provider"] == "DraftKings"


@pytest.mark.parametrize("change", ["kickoff", "reverse", "league", "ambiguous"])
def test_no_fuzzy_or_ambiguous_odds_join(change):
    m = espn()
    if change == "kickoff":m["kickoff"] = "2026-10-19T19:01:00Z"
    if change == "reverse":m["home"], m["away"] = m["away"], m["home"]
    if change == "league":m["competition_key"] = "English Women's Super League"
    additional = [{**m, "match_id": "espn:match:124"}] if change == "ambiguous" else []
    rows, report = merge_schedule([source()], [m]+additional, AT)
    assert report["matched_rows"] == 0
    assert all(not r["supported"] for r in rows if r["source"] == "espn")


def test_postponed_second_source_blocks_start_without_changing_selected_identity():
    rows, _ = merge_schedule([source()], [espn(status="STATUS_POSTPONED")], AT)
    assert rows[0]["source_match_id"] == "3003918"
    assert rows[0]["status"] == "postponed" and not availability(rows[0], AT)[0]


def test_collector_cache_raw_integrity_and_failure_preserves_old_timestamp(tmp_path):
    calls = []
    stamp = [utc(AT)]
    def transport(params):
        calls.append(params)
        return 200, json.dumps({"events": [event()]}).encode()
    feed = ESPNScoreboard(tmp_path, transport=transport, clock=lambda: stamp[0])
    first = feed.collect()
    assert first["http_attempts"] == 8 and len(first["matches"]) == 1
    assert len(feed.archive.observations()) == 8
    assert all(feed.archive.read_body(o) for o in feed.archive.observations())
    second = feed.collect()
    assert second["http_attempts"] == 0 and len(calls) == 8
    stamp[0] += timedelta(seconds=61)
    feed.transport = lambda p: (503, b"service unavailable")
    third = feed.collect()
    assert third["status"] == "partial" and third["http_attempts"] == 1
    failed = next(r for r in third["dates"] if r["date_utc"] == "2026-10-19")
    assert failed["stale"] and failed["observed_at"] == utc(AT).isoformat().replace("+00:00", "Z")
    assert "503" in failed["error"]


def test_rollover_does_not_retain_yesterdays_matches(tmp_path):
    feed = ESPNScoreboard(tmp_path, transport=lambda p: (200, json.dumps({"events": [
        event("1", kickoff="2026-10-19T15:59:00Z"), event("2", kickoff="2026-10-19T16:00:00Z")]}).encode()),
        clock=lambda: utc("2026-10-19T16:01:00Z"))
    assert [m["source_match_id"] for m in feed.collect()["matches"]] == ["2"]


def test_spftll_failure_does_not_prevent_espn_board(tmp_path):
    class Backend:
        def inventory(self, folder):raise TimeoutError("source unavailable")
        def scoreboard(self, folder):return {"status": "ready", "matches": [espn()], "dates": [], "http_attempts": 1}
    app = LiveGApplication(tmp_path, backend=Backend(), clock=lambda: AT)
    app._refresh();c=app.catalog()
    assert len(c["matches"]) == 1 and c["status"] == "partial"
    assert c["sources"]["spftll"]["status"] == "failed"
    assert c["matches"][0]["can_start"] is False


def test_prior_quote_freshness_not_reset_by_scoreboard_refresh():
    m=source();m["quote_catalog_observed_at"]="2026-10-19T14:00:00Z"
    rows,_=merge_schedule([m],[espn()],AT)
    assert not availability(rows[0],AT)[0]


def test_first_catalog_read_autorefreshes_schedule_but_does_not_launch_agent(tmp_path):
    import threading
    class Backend:
        def inventory(self, folder):return {"candidate_registrations": [], "request_budget": {"actual_archived_attempts": 0}}
        def scoreboard(self, folder):return {"status": "ready", "matches": [espn()], "dates": [], "http_attempts": 1}
    app=LiveGApplication(tmp_path,backend=Backend(),clock=lambda:AT);app.auto_catalog=True
    app.catalog();worker=app.catalog_worker;worker.join(10)
    assert not worker.is_alive() and len(app.catalog()["matches"]) == 1
    assert app.workers == {} and app.catalog_worker is worker


def test_secondary_failure_does_not_erase_fresh_primary_inventory(tmp_path):
    class Backend:
        def inventory(self, folder):
            s=source()
            return {"candidate_registrations": [s], "request_budget": {"actual_archived_attempts": 4}}
        def scoreboard(self, folder):raise ValueError("corrupt provider cache")
    app=LiveGApplication(tmp_path,backend=Backend(),clock=lambda:AT)
    app._refresh();c=app.catalog()
    assert c["status"] == "partial" and len(c["matches"]) == 1
    assert c["matches"][0]["can_start"] and c["sources"]["espn"]["status"] == "failed"


def test_scoreboard_refresh_reuses_primary_quote_clock_and_does_not_call_models(tmp_path):
    counts={"inventory":0,"scoreboard":0};stamp=[AT]
    class Backend:
        def inventory(self, folder):
            counts["inventory"]+=1
            return {"candidate_registrations":[source()],"request_budget":{"actual_archived_attempts":4}}
        def scoreboard(self, folder):
            counts["scoreboard"]+=1
            return {"status":"ready","matches":[espn()],"dates":[],"http_attempts":1}
    app=LiveGApplication(tmp_path,backend=Backend(),clock=lambda:stamp[0])
    app._refresh();stamp[0]="2026-10-19T15:01:01Z";app._refresh()
    c=app.catalog()
    assert counts=={"inventory":1,"scoreboard":2} and app.workers=={}
    assert c["matches"][0]["quote_catalog_observed_at"]==AT
    assert c["sources"]["spftll"]["cache_hit"]


def test_upgrade_old_empty_catalog_fetches_daily_schedule_and_declares_loaded_capability(tmp_path):
    from demo.live_g import SCHEDULE_UI_CONTRACT
    class Backend:
        def inventory(self, folder):
            return {"candidate_registrations": [], "request_budget": {"actual_archived_attempts": 0}}
        def scoreboard(self, folder):
            return {"status": "ready", "matches": [espn()], "dates": [], "http_attempts": 1}
    save(tmp_path/"catalog.json", {"status":"no_future_premier_league", "updated_at":"2026-10-19T14:00:00Z", "matches":[]})
    app=LiveGApplication(tmp_path, backend=Backend(), clock=lambda:AT);app.auto_catalog=True
    first=app.catalog()
    assert first["schedule_ui_contract"]==SCHEDULE_UI_CONTRACT
    app.catalog_worker.join(10)
    after=app.catalog()
    assert len(after["matches"])==1 and after["status"]=="ready"
    assert after["sources"]["espn"]["status"]=="ready"
    assert app.workers=={}  # Opening the upgraded page never launches f/g.
