from urllib.parse import urlsplit

import pytest

from football_agent.match_source_profiles import resolve_profile


def test_fixture_alone_resolves_official_capabilities_without_news_conclusion():
    profile = resolve_profile("Arsenal", "Leeds")
    assert profile["status"] == "registered"
    assert "www.premierleague.com" in profile["allowed_hosts"]
    assert profile["teams"][1]["canonical_name"] == "Leeds United"
    assert all(urlsplit(url).hostname in profile["allowed_hosts"] for url in profile["seed_urls"])
    assert all(entry["kind"] in {"club_home", "fixture_index", "news_index"} for entry in profile["entries"])
    assert not any("television-picks" in url or "4727187" in url for url in profile["seed_urls"])
    assert profile["current_transport_success_guaranteed"] is False


def test_registered_aliases_handle_case_spacing_and_chinese():
    assert resolve_profile("  ARSENAL  FC ", "利兹联")["status"] == "registered"
    assert resolve_profile("阿森纳", "Leeds United F.C.")["status"] == "registered"


def test_unknown_team_does_not_acquire_another_club_or_league_identity():
    partial = resolve_profile("Arsenal", "Leeds Women")
    assert partial["status"] == "partial"
    assert "www.leedsunited.com" not in partial["allowed_hosts"]
    assert "www.premierleague.com" not in partial["allowed_hosts"]
    assert partial["unknown_teams"] == ["Leeds Women"]
    absent = resolve_profile("Unknown home", "Unknown away")
    assert absent["status"] == "unsupported"
    assert absent["seed_urls"] == []


def test_profile_result_cannot_mutate_later_capabilities():
    profile = resolve_profile("Arsenal", "Leeds")
    profile["entries"][0]["url"] = "https://attacker.invalid/"
    profile["allowed_hosts"].append("attacker.invalid")
    fresh = resolve_profile("Arsenal", "Leeds")
    assert "attacker.invalid" not in fresh["allowed_hosts"]
    assert all("attacker.invalid" not in row["url"] for row in fresh["entries"])


@pytest.mark.parametrize("home,away", [("Arsenal", "Arsenal FC"), ("", "Leeds"), (None, "Leeds")])
def test_invalid_fixture_identity_rejected(home, away):
    with pytest.raises(ValueError):
        resolve_profile(home, away)


@pytest.mark.parametrize("home,away,expected_hosts,unknown", [
    ("Aston Villa", "Brentford", {"www.avfc.co.uk", "www.brentfordfc.com"}, []),
    ("Ipswich Town", "Fulham", {"www.itfc.co.uk", "www.fulhamfc.com"}, []),
    ("Chelsea", "AFC Bournemouth", {"www.chelseafc.com"}, ["AFC Bournemouth"]),
])
def test_three_match_entries_require_opt_in_and_keep_unregistered_club_unknown(
        home, away, expected_hosts, unknown):
    from football_agent.match_source_profiles import resolve_verified_profile
    binding = {"schema": "premier-league-fixture-identity-v1", "status": "CONFIRMED",
               "binding_sha256": "a" * 64, "canonical": {
                   "home_name": home, "away_name": away,
                   "source_url": "https://www.premierleague.com/en/news/4675097"}}
    old = resolve_verified_profile(binding, version=7)
    new = resolve_verified_profile(binding, version=8)
    assert not expected_hosts.intersection(old["allowed_hosts"])
    assert expected_hosts <= set(new["allowed_hosts"])
    assert new["unknown_teams"] == unknown
    assert len(new["seed_urls"]) == 1 + len(expected_hosts)
    assert "www.afcb.co.uk" not in new["allowed_hosts"]
    assert new["selection_basis"] == "verified_identity_to_capabilities_not_information_value"
