"""Directory responses become query capabilities, never unopened article facts."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import socket

import pytest

from football_agent.candidate_context import _labels
from football_agent.evidence_candidates import EvidenceDocumentCandidates
from football_agent.match_evidence_articles import _body_segment_v5
from football_agent.match_source_profiles import resolve_verified_profile
from football_agent.news_directory_metadata import DIRECTORIES, VERSION, project_directory_metadata
from football_agent.news_discovery import NewsDiscoveryTools
from football_spider.storage.raw_archive import RawArchive

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "var/work/fm-g-semantic-repair-20261011/source-discovery"
NOW = datetime(2026, 10, 11, 10, tzinfo=timezone.utc)
MATCH = {"match_id": "directory-test-only", "home_team": "Crystal Palace",
         "away_team": "Nottingham Forest", "kickoff_at": "2026-10-11T13:00:00Z",
         "prediction_cutoff": "2026-10-11T12:59:00Z"}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Directory projection must use archived bytes only")
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)


def binding():
    return {"schema": "premier-league-fixture-identity-v1", "status": "CONFIRMED",
            "binding_sha256": "a" * 64, "canonical": {"home_name": MATCH["home_team"],
            "away_name": MATCH["away_team"], "source_url": "https://www.premierleague.com/en/news/4675097"}}


def archived(host, version=13):
    archive = RawArchive(SOURCE / "archive")
    obs = next(o for o in archive.observations("official_news_directory") if host in o.endpoint)
    tools = NewsDiscoveryTools(archive, [host], projection_version=version,
                              index_urls=[obs.endpoint], clock=lambda: NOW)
    prepared = {"tool": "read_url", "source": obs.source, "endpoint": obs.endpoint, "params": {}}
    return tools, obs, prepared


@pytest.mark.parametrize("host,count,old_count", [("www.cpfc.co.uk", 10, 0),
                                               ("www.nottinghamforest.co.uk", 4, 5)])
def test_real_archives_exact_metadata_order_and_old_projection_unchanged(host, count, old_count):
    tools, obs, prepared = archived(host)
    doc = tools.project(obs, prepared)
    cards = [row for row in doc["links"] if row.get("directory_metadata_adapter") == VERSION]
    expected = next(team for team in json.loads((SOURCE / "source-entrypoints.json").read_text("utf-8"))["teams"]
                    if host in team["directory_url"])["visible_candidates"]
    assert len(cards) == count
    assert [(x["title"], x["url"], x["source_published_at"], x["source_category"]) for x in cards] == [
        (x["title"], x["url"], x["source_published_at"], x["source_category"]) for x in expected]
    assert doc["kind"] == "index" and doc["structured_news"]["segments"] == []
    assert doc["serialized_directory_projection"]["site_coverage_complete"] is False
    assert doc["raw_sha256"] == obs.content_hash
    for card in cards:
        assert card["snippet"] is None and card["source_modified_at"] is None
        assert card["source_available_at"] is None
        assert card["target_article_fetched_for_context"] is False
        assert _labels(card)[1] == "observed_CMS_card_metadata"
    old, old_obs, old_prepared = archived(host, 12)
    assert len(old.project(old_obs, old_prepared)["links"]) == old_count
    assert NewsDiscoveryTools.from_configuration(tools.archive, tools.configuration(), clock=lambda: NOW).configuration() == tools.configuration()
    changed = deepcopy(tools.configuration())
    changed["news_discovery"]["serialized_directory_adapter"] = "unregistered"
    with pytest.raises(ValueError, match="reconstruct"):
        NewsDiscoveryTools.from_configuration(tools.archive, changed, clock=lambda: NOW)


def test_official_profile_requires_v9_and_has_only_reviewed_generic_entries():
    old, new = [resolve_verified_profile(binding(), version=v) for v in (8, 9)]
    assert old["unknown_teams"] == ["Crystal Palace", "Nottingham Forest"]
    assert new["unknown_teams"] == []
    assert set(new["allowed_hosts"]) == {*DIRECTORIES, "www.premierleague.com"}
    assert {entry["url"] for entry in new["entries"] if entry["kind"] == "news_index"} == set(DIRECTORIES.values())
    wrong = binding()
    wrong["canonical"]["away_name"] = "Nottingham Forest Women"
    assert "www.nottinghamforest.co.uk" not in resolve_verified_profile(wrong, version=9)["allowed_hosts"]


def generator(cap=32):
    profile = resolve_verified_profile(binding(), version=9)
    return EvidenceDocumentCandidates(match_terms=[MATCH["home_team"], MATCH["away_team"]],
        seed_urls=profile["seed_urls"], allowed_hosts=profile["allowed_hosts"],
        numeric_actions=[], web_cost={"money": .01, "requests": 1.},
        match=MATCH, profile=profile, discovery_version=14, max_discovered_links=cap)


def test_real_metadata_reaches_actual_action_ids_and_context_without_opening_targets():
    docs = [tools.project(obs, prepared) for tools, obs, prepared in [archived(host) for host in DIRECTORIES]]
    public = {"state": {"base": {"kickoff": MATCH["kickoff_at"]},
                        "prediction_cutoff": MATCH["prediction_cutoff"]},
              "decision_at": NOW.isoformat(), "documents": docs, "attempts": []}
    gen = generator()
    actions = gen(public)
    by_url = {a["url"]: a for a in actions if a["tool"] == "read_url"}
    expected = "https://www.cpfc.co.uk/news/first-team/team-news-sage-updates-on-mateta-and-khalaili-ahead-of-forest/"
    assert expected in by_url
    assert any("third-kit" in url for url in by_url)  # No usefulness curation.
    assert any("dominant-b-team" in url for url in by_url)
    contexts = gen.candidate_contexts(public, actions)
    aid = by_url[expected]["action_id"]
    labels = contexts["by_action_id"][aid]["observations"]
    assert labels[0]["label_source"] == "observed_CMS_card_metadata"
    assert labels[0]["observed_anchors"][0]["text"].startswith("Team News:")
    assert labels[0]["parent"]["raw_sha256"] == docs[0]["raw_sha256"]
    # The bounded menu still reports its existing cap, without silently
    # treating response metadata as an unlimited set of selected requests.
    capped = generator(cap=3)
    audit = capped.candidate_contexts(public, capped(public))["ordering"]["caps"]
    assert audit["article_cap_shortage"] and audit["omitted_by_link_cap"] > 0
    future = deepcopy(public)
    for doc in future["documents"]:
        doc["available_at"] = "2026-10-11T12:59:01Z"
    assert expected not in {a.get("url") for a in gen(future)}


def palace_html(rows):
    return '<script type="application/ld+json">' + json.dumps({"@type": "ItemList", "itemListElement": rows}) + '</script>'


def item(title="Update", url="https://www.cpfc.co.uk/news/first-team/update/", **kw):
    return {"@type": "ListItem", "name": title, "url": url, **kw}


def project(html, host="www.cpfc.co.uk", **kw):
    return project_directory_metadata({"url": DIRECTORIES[host], "links": []}, html, [host], **kw)


@pytest.mark.parametrize("url", ["https://outside.example/news/article", "https://www.cpfc.co.uk.evil.example/news/a",
    "javascript:alert(1)", "http://www.cpfc.co.uk/news/article", "https://www.cpfc.co.uk/news/",
    "https://www.cpfc.co.uk/shop/kit", "https://user@www.cpfc.co.uk/news/article"])
def test_invalid_targets_never_become_read_capabilities(url):
    result = project(palace_html([item(url=url)]))
    assert result["links"] == []
    assert len(result["serialized_directory_projection"]["rejected_records"]) == 1


def test_truncation_missing_dates_raw_text_and_duplicates_remain_explicit():
    rows = [item("X" * 520), item("Duplicate"), item("U18 kit travel", url="/news/club/another/", datePublished="bad date")]
    result = project(palace_html(rows), max_links=1)
    assert result["links_truncated"] is True
    assert result["serialized_directory_projection"]["cards_omitted_by_cap"] == 1
    assert result["serialized_directory_projection"]["duplicate_urls"] == 1
    assert result["links"][0]["source_published_at"] is None
    assert _labels(result["links"][0])[2] is True
    full = project(palace_html(rows))
    assert full["links"][1]["publication_raw"] == "bad date"
    assert full["links"][1]["publication_status"] == "unparsed_directory_source_assertion"


@pytest.mark.parametrize("ref", [-1, 99, True, "4", None])
def test_malformed_nuxt_does_not_leak_partial_candidates(ref):
    values = [{"type": 1, "attributes": 2}, "news-article", {"postTitle": ref, "postSlug": 4}, "Update", "/news/article/"]
    result = project('<script id="__NUXT_DATA__">' + json.dumps(values) + '</script>', "www.nottinghamforest.co.uk")
    assert result["links"] == []
    assert result["serialized_directory_projection"]["status"] == "rejected"


def test_explicit_new_pointer_cannot_be_relabelled_as_legacy_cms():
    row = project(palace_html([item()]))["links"][0]
    assert _labels(row)[0][0]["anchor_index"] is None
    del row["directory_metadata_adapter"]
    with pytest.raises(ValueError, match="literal response pointer"):
        _labels(row)


def test_only_registered_index_url_can_project_and_does_not_execute_javascript():
    html = palace_html([item()])
    doc = {"url": "https://www.cpfc.co.uk/news/first-team/other/", "links": []}
    assert project_directory_metadata(deepcopy(doc), html, ["www.cpfc.co.uk"]) == doc
    result = project('<script>fetch("/news/body")</script><script type="application/ld+json">function bad(){}</script>')
    assert result["links"] == [] and result["serialized_directory_projection"]["status"] == "rejected"


def test_projection_v13_preserves_existing_fulham_body_contract():
    from tests.agent.test_news_cms_projection import archived as old_archive
    archive, obs = old_archive("www.fulhamfc.com")
    prepared = {"tool": "read_url", "source": "open_page", "endpoint": obs.endpoint, "params": {}}
    docs = [NewsDiscoveryTools(archive, ["www.fulhamfc.com"], projection_version=v, clock=lambda: NOW).project(obs, prepared)
            for v in (12, 13)]
    assert docs[0]["structured_news"] == docs[1]["structured_news"]
    assert docs[0]["text"] == docs[1]["text"]
    for segment in docs[1]["structured_news"]["segments"]:
        assert _body_segment_v5(segment, docs[1], docs[1]["structured_news"])[1] is None


def test_collect_cli_accepts_explicit_new_versions_without_changing_defaults(tmp_path, monkeypatch):
    from football_research.data_adapters import prospective_text
    calls = []
    monkeypatch.setattr(prospective_text, "collect_prepared_batch", lambda *a, **kw: calls.append(kw) or {})
    plan = tmp_path / "plan.json"
    plan.write_text('{"selected":[]}', encoding="utf-8")
    prospective_text.main(["collect", "--plan", str(plan), "--archive", str(tmp_path),
        "--output", str(tmp_path / "out"), "--allowed-host", "www.cpfc.co.uk",
        "--source-profile-version", "9", "--news-projection-version", "13"])
    assert calls[0]["source_profile_version"] == 9
    assert calls[0]["news_projection_version"] == 13
