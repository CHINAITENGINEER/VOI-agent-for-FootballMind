from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json

import pytest
from football_research.data_adapters.espn_scoreboard import ESPNScoreboard, parse, ENDPOINT
from demo.live_g import LiveBackend, _directory_candidate_reason, _team_tokens, _review_pending_host_links
from demo.live_sources import discovery_queries, search_team_binding, LiveNewsTools
from football_spider.storage.raw_archive import RawArchive

AT = datetime(2026, 10, 25, 2, 0, tzinfo=timezone.utc)


def raw_event(odds=True):
    event = {"id": "123", "date": "2026-10-25T13:00:00Z", "uid": "s:600~l:3945~e:123",
        "competitions": [{"date": "2026-10-25T13:00:00Z", "timeValid": True,
            "altGameNote": "Swedish Allsvenskan", "status": {"type": {"name": "STATUS_SCHEDULED", "state": "pre"}},
            "competitors": [{"homeAway": "home", "team": {"id": "1", "displayName": "GAIS"}},
                            {"homeAway": "away", "team": {"id": "2", "displayName": "IFK Göteborg"}}]}]}
    if odds:
        event["competitions"][0]["odds"] = [{"provider": {"name": "DraftKings"}, "moneyline": {
            "home": {"close": {"odds": "+140"}}, "draw": {"close": {"odds": "+240"}},
            "away": {"close": {"odds": "+185"}}}}]
    return json.dumps({"events": [event]}).encode()


def fixture(tmp_path, age=30):
    root = tmp_path / "catalog"
    obs = RawArchive(root / "archive").save("espn", ENDPOINT, {"dates": "20261025", "limit": 1000},
        raw_event(), status=200, observed_at=AT-timedelta(seconds=age))
    row = parse(json.loads(raw_event()), obs.model_dump(mode="json"))[0][0]
    return root, row


def offline(params):
    raise ConnectionError("test disconnect")


def test_target_initialization_requests_only_target_day(tmp_path):
    root, row = fixture(tmp_path)
    calls = []
    def transport(params):
        calls.append(params)
        return 200, raw_event()
    report = ESPNScoreboard(tmp_path/"run", transport=transport, clock=lambda: AT).collect_fixture(row)
    assert len(calls) == report["http_attempts"] == 1
    assert calls[0]["dates"] == "20261025" and report["matches"][0]["match_id"] == row["match_id"]


def test_transport_failure_uses_verified_recent_raw_without_new_timestamp(tmp_path):
    root, row = fixture(tmp_path)
    report = ESPNScoreboard(tmp_path/"run", transport=offline, clock=lambda: AT).collect_fixture(row, catalog_root=root)
    assert report["status"] == "ready"
    assert report["acquisition_mode"] == "recent_catalogue_raw_replay_after_transport_failure"
    restored = report["matches"][0]
    assert restored["observed_at"] == row["observed_at"]
    assert restored["quote_snapshot"] == row["quote_snapshot"]
    assert restored["raw_observation"]["content_hash"] == row["raw_observation"]["content_hash"]


@pytest.mark.parametrize("case", ["expired", "future", "kickoff", "forged_hash", "forged_time"])
def test_bad_cached_evidence_never_initializes(tmp_path, case):
    root, row = fixture(tmp_path, age=301 if case == "expired" else -1 if case == "future" else 30)
    at = AT
    if case == "kickoff":
        at = datetime(2026, 10, 25, 13, tzinfo=timezone.utc)
    if case == "forged_hash":
        row["raw_observation"]["content_hash"] = "0"*64
    if case == "forged_time":
        row["raw_observation"]["observed_at"] = AT.isoformat()
    report = ESPNScoreboard(tmp_path/"run", transport=offline, clock=lambda: at).collect_fixture(row, catalog_root=root)
    assert report["matches"] == [] and report["status"] == "failed"


@pytest.mark.parametrize("payload", [b'{"events": []}', raw_event(False), b'{"unexpected":true}'])
def test_successful_withdrawn_or_invalid_response_never_uses_old_odds(tmp_path, payload):
    root, row = fixture(tmp_path)
    report = ESPNScoreboard(tmp_path/"run", transport=lambda p: (200, payload), clock=lambda: AT).collect_fixture(row, catalog_root=root)
    assert report["acquisition_mode"] == "fresh_target_day"
    assert not report["matches"] or not report["matches"][0]["quote_snapshot"]


def test_espn_id_does_not_call_other_source(tmp_path, monkeypatch):
    b = LiveBackend()
    monkeypatch.setattr(b, "_initialize_espn", lambda *a: {"direct": True})
    assert b.initialize({"source": "espn", "match_id": "espn:match:123"}, tmp_path) == {"direct": True}


def test_network_error_and_missing_odds_have_distinct_messages(tmp_path, monkeypatch):
    b = LiveBackend()
    monkeypatch.setattr(b, "fixture_scoreboard", lambda *a: {"status": "failed", "matches": []})
    with pytest.raises(ValueError, match="来源请求失败"):
        b._initialize_espn({"match_id": "espn:match:123"}, tmp_path, None)


def test_queries_and_unicode_binding_are_competition_specific():
    canonical = {"home_name": "AV Alta FC", "away_name": "Corpus Christi FC", "competition_name": "USL League One", "kickoff": AT.isoformat()}
    queries = discovery_queries(canonical)
    assert not any("kicker.de" in q or "bundesliga.com" in q for q in queries)
    assert any("avaltafc.com" in q for q in queries)
    assert _team_tokens("GAIS IFK Göteborg") == {"gais", "goteborg"}
    assert not search_team_binding("AV movies latest videos", "https://example.org/adult-videos-news", canonical, [])
    assert search_team_binding("Corpus Christi match preview", "https://sports.example/news/preview", canonical, [])
    assert _directory_candidate_reason("https://www.gais.se/nyheter/senaste-nytt-om-spelare", "Senaste nytt om spelare", canonical, ["https://www.gais.se/"], set()) is None


def test_multiple_articles_from_one_approved_host_survive(tmp_path):
    tools = LiveNewsTools(RawArchive(tmp_path), [], projection_version=13)
    run = {"fixture": {"home_name": "GAIS", "away_name": "IFK Göteborg"}}
    doc = {"links": [], "structured_news": {"pending_host_links": [
        {"url": "https://sports.example/news/gais-team-news", "title": "GAIS team news"},
        {"url": "https://sports.example/news/gais-match-preview", "title": "GAIS match preview"}]}}
    _review_pending_host_links(run, tools, doc)
    assert len(doc["links"]) == 2 and run["dynamic_allowed_hosts"] == ["sports.example"]


def test_gais_template_uses_only_registered_body_and_keeps_time_unknown(tmp_path):
    url = "https://www.gais.se/nyheter/players"
    raw = ('<html><head><link rel="canonical" href="'+url+'"><meta property="og:title" content="GAIS | Player update">'
        '<script type="application/ld+json">{"@type":["Organization","SportsTeam"]}</script></head><body>'
        '<div class="news-p-header"><h1>Player update</h1><p class="news-p-header-bottom-text">2026-10-23 14:00</p></div>'
        '<div class="news-p-content"><div class="rich-text-article"><p>First real paragraph.</p><p>Second real paragraph.</p></div></div>'
        '<footer><p>Footer is not evidence.</p></footer></body></html>').encode()
    archive = RawArchive(tmp_path)
    obs = archive.save("open_page", url, {}, raw, status=200, observed_at=AT)
    tools = LiveNewsTools(archive, ["www.gais.se"], projection_version=13, clock=lambda: AT)
    prepared = {"tool": "read_url", "endpoint": url, "params": {}, "source": "open_page"}
    doc = tools.project(obs, prepared)
    assert doc["structured_news"]["role"] == "article"
    assert doc["text"] == "First real paragraph.\nSecond real paragraph."
    assert doc["source_published_at"] is None and doc["source_publication_local_time"] == "2026-10-23 14:00"
    for segment in doc["structured_news"]["segments"]:
        assert doc["text"][segment["start"]:segment["end"]] == segment["text"]
