"""Explicit competition/squad binding prevents same-name club substitution."""
import pytest

from demo.live_sources import (CLUB_SOURCE_SCOPE, DIRECTORIES, club_source_admission,
    competition_source_scope, discovery_queries, profile)


def binding(competition,home="Liverpool",away="Chelsea"):
    return {"canonical":{"home_name":home,"away_name":away,"competition_name":competition,
                         "kickoff":"2026-10-26T12:00:00Z"}}


def test_every_registered_club_has_explicit_association_and_squad():
    assert set(DIRECTORIES)==set(CLUB_SOURCE_SCOPE)
    assert all(x["squad"]=="men_senior" for x in CLUB_SOURCE_SCOPE.values())


def test_uruguayan_liverpool_cannot_use_english_liverpool_source():
    p=profile(binding("Liga AUF Uruguaya",home="Nacional",away="Liverpool"))
    assert not p["seed_urls"]
    assert p["unknown_teams"]==["Nacional","Liverpool"]
    assert p["club_scope_audit"][1]["reason"]=="club_competition_association_mismatch_or_unknown"
    assert "www.liverpoolfc.com" not in p["allowed_hosts"]
    assert not any("liverpoolfc.com" in q for q in discovery_queries(binding("Liga AUF Uruguaya",home="Nacional",away="Liverpool")["canonical"]))


@pytest.mark.parametrize("competition",["Women's Super League","WSL Players Cup, League Phase",
    "Dutch Vrouwen Eredivisie","UEFA Women's Champions League, League Phase","Liga MX Femenil",
    "Première Ligue","NWSL","Liga F","女足联赛"])
def test_womens_competition_never_borrows_mens_club_seeds(competition):
    p=profile(binding(competition))
    assert p["seed_urls"]==[]
    assert all(r["reason"]=="club_squad_scope_mismatch" for r in p["club_scope_audit"])


@pytest.mark.parametrize("competition",["Carabao Cup, Fourth Round","FA Cup, Third Round",
    "Premier League","English Premier League","EFL Trophy, Northern Group F"])
def test_english_club_sources_survive_english_league_and_cups(competition):
    p=profile(binding(competition))
    assert p["seed_urls"]==[DIRECTORIES["Liverpool"],DIRECTORIES["Chelsea"]]
    assert not p["unknown_teams"]


@pytest.mark.parametrize("competition",["UEFA Champions League, League Phase","UEFA Europa League","欧冠","欧协联"])
def test_legitimate_european_club_competitions_keep_club_sources(competition):
    p=profile(binding(competition,home="Barcelona",away="Liverpool"))
    assert p["seed_urls"]==[DIRECTORIES["Barcelona"],DIRECTORIES["Liverpool"]]


@pytest.mark.parametrize("competition,home,away",[
    ("Coppa Italia, Round of 16","Internazionale","Fiorentina"),
    ("German Cup, Second Round","Augsburg","Union Berlin"),
    ("Copa del Rey, First Round","Barcelona","Real Madrid"),
    ("KNVB Beker, First Round","Fortuna Sittard","Excelsior"),
    ("Coupe de France, Round of 32","Monaco","Lille"),
    ("日职","Cerezo Osaka","JEF United Ichihara-Chiba")])
def test_own_association_cups_and_jleague_sources_remain_valid(competition,home,away):
    p=profile(binding(competition,home,away))
    assert not p["unknown_teams"]
    assert len(p["seed_urls"])==2


def test_unknown_or_other_country_does_not_use_name_only_mapping():
    for league in ["Unregistered Competition","阿甲","中超","Austrian Bundesliga","巴甲",
                   "Russian Premier League","South African Premier League"]:
        p=profile(binding(league))
        assert not p["seed_urls"]


def test_missing_competition_legacy_call_is_marked_not_identity_proof():
    p=profile({"canonical":{"home_name":"Liverpool","away_name":"Chelsea"}})
    assert len(p["seed_urls"])==2
    assert all(r["reason"]=="legacy_name_only_compatibility_not_runtime_identity_proof" for r in p["club_scope_audit"])
@pytest.mark.parametrize("competition,expected",[("苏超","https://spfl.co.uk/news"),("日职","https://www.jleague.jp/j1/news/")])
def test_verified_league_news_covers_unregistered_clubs_without_using_other_club_directory(competition,expected):
    from demo.live_sources import profile
    value=profile({"canonical":{"home_name":"Unknown One","away_name":"Unknown Two","competition_name":competition}})
    assert value["seed_urls"]==[expected] and value["league_wide_seed_urls"]==[expected]

