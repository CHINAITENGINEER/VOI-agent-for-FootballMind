from datetime import datetime, timezone
import json

import pytest

from football_research.data_adapters.qiumibao_schedule import (
    QiumibaoDailySchedule, bind_league, bind_league_detail, default_bindings, parse)
from football_research.data_adapters.combined_schedule import combine_reports
from demo.daily_fixtures import merge_schedule
from demo.live_coverage import availability, competition_key
from demo.live_g import LiveBackend

AT="2026-10-25T08:55:00+00:00"
OBS={"observed_at":AT,"id":"source-id","content_hash":"a"*64}
URL="https://www.zhibo8.com/zhibo/zuqiu/2026/match1932319vplayer.htm"


def row(**changes):
    return {"id":"1932319","leagueid":"351","sdate":"2026-10-25","type":"football",
        "time":"19:00","start":"2026-10-25 19:00","home_team":"无锡吴钩","visit_team":"陕西联合",
        "home_id":"28916","visit_id":"34707","home_score":"","visit_score":"","state":"1",
        "period_cn":"","url":"/zhibo/zuqiu/2026/match1932319vplayer.htm",**changes}


def binding():
    return bind_league({"list":[row()]},'<title>10月25日19:00 中甲第28轮 无锡吴钩vs陕西联合</title>'
        '<meta name="keywords" content="中甲,无锡吴钩,陕西联合"><div match-time="2026-10-25 19:00"></div>',
        URL,OBS,match_id="1932319",competition_name="中甲")


def test_verified_source_match_page_binds_current_league_id():
    b=binding()
    assert b["source_league_id"]=="351" and b["competition_name"]=="中甲"
    assert b["reference_match_id"]=="1932319" and not b["independently_verified"]


def test_unknown_league_id_cannot_gain_a_name_from_team_guessing():
    matches,audit=parse({"list":[row(leagueid="unknown")]},OBS,source_date="2026-10-25",league_bindings=[binding()])
    assert not matches and audit["ignored_unbound_or_other_sport"]==1


def test_wrong_match_page_or_kickoff_is_rejected():
    with pytest.raises(ValueError):
        bind_league({"list":[row()]},'<title>Other team 中甲</title><meta name="keywords" content="中甲">',
                    URL,OBS,match_id="1932319",competition_name="中甲")


def test_current_fixture_clock_ids_and_absent_prior():
    matches,audit=parse({"list":[row()]},OBS,source_date="2026-10-25",league_bindings=[binding()])
    m=matches[0]
    assert m["match_id"]=="qiumibao:match:1932319" and m["kickoff"]=="2026-10-25T11:00:00+00:00"
    assert m["home_team_source_id"]=="28916" and m["away_team_source_id"]=="34707"
    assert m["home_score"] is m["away_score"] is None
    assert m["competition_key"]=="中甲" and m["status"]=="scheduled"
    assert not m["quote_observed"] and not availability(m,AT)[0]


def test_old_season_or_opponent_id_does_not_become_a_current_fixture():
    for r in [row(start="2021-10-25 19:00"),row(visit_id="28916")]:
        matches,audit=parse({"list":[r]},OBS,source_date="2026-10-25",league_bindings=[binding()])
        assert not matches and audit["rejected"]


def test_live_finished_postponed_and_unknown_states_do_not_start():
    for state,status in [("2","live"),("3","finished"),("4","postponed"),("6","cancelled"),("7","unknown")]:
        m=parse({"list":[row(state=state)]},OBS,source_date="2026-10-25",league_bindings=[binding()])[0][0]
        assert m["status"]==status and not availability(m,AT)[0]


def test_pinned_real_source_binding_loads_from_verified_raw_without_http():
    bb=default_bindings()
    assert len(bb)==1 and bb[0]["source_league_id"]=="351"
    evidence=bb[0].get("detail_observation") or bb[0].get("page_observation")
    assert evidence["http_status"]==200


def test_explicit_current_league_detail_checks_season_and_both_team_ids():
    d={"match_id":1932319,"type":"zuqiu","league":{"id":"351","ziliao_name":"中甲"},
       "season":"2026","match_time":"2026-10-25 19:00","home_team":"无锡吴钩","visit_team":"陕西联合",
       "home_id":"28916","visit_id":"34707"}
    b=bind_league_detail({"list":[row()]},d,OBS,match_id="1932319",competition_name="中甲")
    assert b["source_league_id"]=="351"
    d["season"]="2021"
    with pytest.raises(ValueError,match="Structured source detail"):
        bind_league_detail({"list":[row()]},d,OBS,match_id="1932319",competition_name="中甲")


def test_daily_cache_fifteen_minutes_does_not_repeat_requests(tmp_path):
    calls=[]
    def transport(endpoint):
        calls.append(endpoint);return 200,json.dumps({"list":[row()]},ensure_ascii=False).encode()
    feed=QiumibaoDailySchedule(tmp_path,transport=transport,league_bindings=[binding()],
        clock=lambda:datetime(2026,10,25,8,55,tzinfo=timezone.utc))
    first=feed.collect();second=feed.collect()
    assert first["http_attempts"]==1 and second["http_attempts"]==0 and len(calls)==1
    assert second["observed_at"]==first["observed_at"] and second["matches"]==first["matches"]


def test_supplement_keeps_visible_real_schedule_without_routing_to_espn():
    m=parse({"list":[row()]},OBS,source_date="2026-10-25",league_bindings=[binding()])[0][0]
    board=combine_reports({"status":"ready","matches":[],"http_attempts":0,"dates":[]},
        {"status":"ready","matches":[m],"http_attempts":1},{"status":"ready","matches":[],"requests":1},at=AT)
    result,audit=merge_schedule([],board["matches"],AT)
    assert len(result)==1 and result[0]["source"]=="qiumibao"
    assert audit["espn_rows"]==0 and audit["supplemental_schedule_rows"]==1
    assert board["http_attempts"]==2 and not availability(result[0],AT)[0]


def test_espan_failure_does_not_erase_supplement_or_require_dates_key():
    m=parse({"list":[row()]},OBS,source_date="2026-10-25",league_bindings=[binding()])[0][0]
    report=combine_reports({"status":"failed","matches":[]},
        {"status":"ready","matches":[m],"http_attempts":1},{"status":"ready","matches":[],"requests":0},at=AT)
    assert report["status"]=="partial" and report["dates"]==[] and len(report["matches"])==1


def test_missing_market_fixture_cannot_accidentally_enter_spftll_initializer(tmp_path):
    with pytest.raises(ValueError,match="只有赛程"):
        LiveBackend().initialize({"source":"qiumibao","match_id":"qiumibao:match:1932319"},tmp_path)
    assert competition_key("韩职")=="K联赛" and competition_key("K League 2")=="K2联赛"
