from copy import deepcopy
import pytest
from demo.live_coverage import catalog_match, availability, summarize, bind_source_identity
from demo.live_sources import profile, article_link_reason
from demo.live_g import LiveGApplication, LiveBackend

AT = "2026-10-19T14:00:00+00:00"
KICK = "2026-10-19T18:45:00+00:00"


def row(**changes):
    return {"fixture_id": "spftll:match:2993821", "source_match_id": "2993821",
        "home_team": "帕尔马", "away_team": "都灵", "home_team_source_id": "189",
        "away_team_source_id": "558", "competition_key": "意甲", "kickoff_at": KICK,
        "status": "scheduled", "inventory_quality": {"complete_1x2_observed": True}, **changes}


def manifest():
    return {"market_initialization": {"fixture": {"match_id": "spftll:match:2993821",
        "home_team_name": "帕尔马", "away_team_name": "都灵", "home_team_id": "spftll:team:189",
        "away_team_id": "spftll:team:558", "competition_name": "意甲", "kickoff": KICK,
        "status": "scheduled", "input_manifest": ["raw-bound"]}}}


def test_catalog_retains_cross_competition_and_unknown_names(tmp_path):
    class Backend:
        def inventory(self, folder):
            return {"candidate_registrations": [row(), row(fixture_id="spftll:match:2", source_match_id="2",
                home_team="未译球队", competition_key="日本联赛")],
                "request_budget": {"actual_archived_attempts": 4}}
    app = LiveGApplication(tmp_path, backend=Backend(), clock=lambda: AT)
    app._refresh()
    catalog = app.catalog()
    assert len(catalog["matches"]) == 2
    assert catalog["matches"][1]["home"] == "未译球队"
    assert catalog["coverage"]["can_start"] == catalog["coverage"]["competition_count"] == 2
    assert all(m["model_scope"] == "cross_competition_experimental" for m in catalog["matches"])
    assert catalog["matches"][0]["competition_group"] == "europe_top5"
    assert catalog["matches"][1]["competition_group"] == "other"


def test_missing_quote_visible_but_not_startable():
    match = catalog_match(row(inventory_quality={"complete_1x2_observed": False}))
    match["can_start"], reason = availability(match, AT)
    assert not match["can_start"] and "赔率" in reason
    assert summarize([match])["waiting_quote"] == 1


def test_conflicting_schedule_or_team_identity_blocked():
    for changes in ({"inventory_quality": {"schedule_conflict_observed": True}},
                    {"away_team_source_id": "189"}, {"status": "finished"}):
        assert not availability(catalog_match(row(**changes)), AT)[0]


def test_source_binding_is_not_independent_official_identity():
    binding = bind_source_identity(manifest(), catalog_match(row()), AT)
    assert binding["status"] == "SOURCE_CONFIRMED"
    assert not binding["independently_confirmed"]
    assert binding["canonical"]["source_event_id"] == "spftll:match:2993821"
    assert binding["canonical"]["season"] is None
    assert binding["aliases"]["home"] == ["帕尔马", "Parma"]
    assert len(profile(binding)["seed_urls"]) == 2


@pytest.mark.parametrize("key,value", [("match_id", "spftll:match:other"),
    ("home_team_id", "spftll:team:558"), ("home_team_name", "其他球队"),
    ("competition_name", "英超"), ("kickoff", "2026-10-19T19:00:00+00:00"),
    ("status", "finished")])
def test_fresh_identity_must_match_selected_row(key, value):
    data = manifest()
    data["market_initialization"]["fixture"][key] = value
    with pytest.raises(ValueError):
        bind_source_identity(data, catalog_match(row()), AT)


def test_unknown_club_directory_uses_public_search():
    binding = {"canonical": {"home_name": "未注册队一", "away_name": "未注册队二"}}
    config = profile(binding)
    assert config["seed_urls"] == [] and config["discovery_mode"] == "public_match_search"
    assert "www.bbc.com" in config["allowed_hosts"]


def test_german_team_aliases_feed_search_in_english():
    match = catalog_match(row(home_team="奥格斯堡", away_team="柏林联合",
                               home_team_source_id="901", away_team_source_id="902",
                               fixture_id="spftll:match:3018840", source_match_id="3018840",
                               competition_key="德甲"))
    binding = bind_source_identity({"market_initialization": {"fixture": {
        "match_id": "spftll:match:3018840", "home_team_name": "奥格斯堡",
        "away_team_name": "柏林联合", "home_team_id": "spftll:team:901",
        "away_team_id": "spftll:team:902", "competition_name": "德甲",
        "kickoff": KICK, "status": "scheduled", "input_manifest": ["raw-bound"]}}}, match, AT)
    assert binding["canonical"]["home_name"] == "Augsburg"
    assert binding["canonical"]["away_name"] == "Union Berlin"


def test_verified_german_official_directories_are_seeded():
    binding = {"canonical": {"home_name": "Augsburg", "away_name": "Union Berlin"}}
    config = profile(binding)
    assert config["seed_urls"] == ["https://www.fcaugsburg.de/", "https://www.fc-union-berlin.de/de"]
    assert config["unknown_teams"] == []


def test_search_articles_not_restricted_to_news_path():
    url = "https://football-italia.net/parma-torino-team-news-preview/"
    assert article_link_reason(url, "Parma Torino team news preview", [], set(), search=True) is None
    assert article_link_reason(url, "Parma Torino team news preview", [], set()) is not None
    assert article_link_reason("https://www.bbc.com/sport/football", "Latest football news", [], set(), search=True)
    assert article_link_reason("https://www.torinofc.it/en/news/biglietteria", "News Biglietteria", [], set()) == "observed_news_category_not_article"
    assert article_link_reason("https://www.fc-union-berlin.de/de/meldungen/auswaerts-in-augsburg-union-will-punkte-nachlegen-id", "Union will Punkte nachlegen", [], set()) is None


def test_non_pl_identity_needs_no_pl_network(tmp_path, monkeypatch):
    backend = LiveBackend()
    monkeypatch.setattr(backend, "tools", lambda *a: pytest.fail("Must not fetch PL for an Italian match"))
    monkeypatch.setattr("demo.live_g.now", lambda: AT)
    assert backend.identity(manifest(), tmp_path, catalog_match(row()))["status"] == "SOURCE_CONFIRMED"


def test_espn_identity_accepts_exact_event_and_observed_moneyline(tmp_path):
    backend = LiveBackend()
    expected = {"source": "espn", "match_id": "espn:match:123", "home": "Tottenham Hotspur",
                "away": "Coventry City", "home_team_source_id": "1", "away_team_source_id": "2",
                "competition_key": "Premier League", "kickoff": "2026-10-19T19:00:00+00:00"}
    manifest = {"market_initialization": {"fixture": {
        "match_id": "espn:match:123", "home_team_id": "espn:team:1", "away_team_id": "espn:team:2",
        "home_team_name": "Tottenham Hotspur", "away_team_name": "Coventry City",
        "competition_name": "Premier League", "kickoff": expected["kickoff"], "status": "scheduled",
        "input_manifest": ["espn:observation:test"]}}}
    binding = backend.identity(manifest, tmp_path, expected)
    assert binding["status"] == "SOURCE_CONFIRMED"
    assert binding["canonical"]["source_event_id"] == "espn:match:123"


def test_espn_initialization_uses_only_complete_observed_moneyline(tmp_path, monkeypatch):
    backend = LiveBackend()
    match = {"match_id": "espn:match:123", "source_match_id": "123", "source": "espn",
             "status": "scheduled", "home": "Tottenham Hotspur", "away": "Coventry City",
             "home_team_source_id": "1", "away_team_source_id": "2", "competition_key": "Premier League",
             "kickoff": "2026-10-19T19:00:00+00:00", "observed_at": AT,
             "quote_snapshot": {"provider": "DraftKings", "no_vig_probabilities":
                 {"home": .4, "draw": .3, "away": .3}, "observed_at": AT},
             "raw_observation": {"id": "raw-test", "content_hash": "a" * 64}}
    monkeypatch.setattr(backend, "fixture_scoreboard", lambda fixture, folder: {"matches": [match]})
    result = backend._initialize_espn({"source": "espn", "match_id": "espn:match:123", "fixture_id": "espn:match:123"}, tmp_path, ValueError("spftll unavailable"))
    market = result["market_initialization"]
    assert market["probabilities"] == [.4, .3, .3]
    assert market["fixture"]["home_team_id"] == "espn:team:1"
    assert (tmp_path / "espn-initialization.json").is_file()
