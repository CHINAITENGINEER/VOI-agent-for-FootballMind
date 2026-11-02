"""Only source body spans can propose v4 evidence follow-ups; zero network."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import socket

import pytest

from football_agent.match_evidence import (
    build_match_evidence, LEGACY_VERSION, VERSION, ARTICLES_VERSION, CONTENT_VERSION,
)
from football_agent.news_discovery import NewsDiscoveryTools
from football_spider.storage.raw_archive import RawArchive
from test_match_evidence import MATCH, AS_OF, document
from test_match_evidence_articles import article


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("Evidence projection is offline"))
    monkeypatch.setattr(NewsDiscoveryTools, "execute", lambda *a, **k: pytest.fail("Do not acquire new pages"))


def project(source):
    return build_match_evidence(MATCH, [source], as_of=AS_OF, version=CONTENT_VERSION)


def frozen_digest(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def html_projection(tmp_path, html):
    archive = RawArchive(tmp_path/"raw-fixture")
    observed = datetime(2026, 10, 6, 9, tzinfo=timezone.utc)
    prepared = {"tool": "read_url", "source": "open_page", "endpoint": "https://club.example/news/preview", "params": {}}
    receipt = archive.save("open_page", prepared["endpoint"], {}, html.encode(), status=200, observed_at=observed)
    tools = NewsDiscoveryTools(archive, ["club.example"], projection_version=2,
                               clock=lambda: observed+timedelta(seconds=1))
    return tools.project(receipt, prepared)


@pytest.mark.parametrize("role", ["unknown", "index", "navigation", "identity_metadata", "", None, [], {}])
def test_non_body_role_never_falls_back_to_full_page_even_with_valid_looking_segments(role):
    source = article(["Arsenal may change the lineup because Kai Havertz has an injury."], role=role)
    source["kind"] = "page"
    source["links"] = [{"title": "Actual team update", "url": "https://club.example/news/actual-update"}]
    before = deepcopy(source)
    report = project(source)
    assert report["document_reviews"][0]["reason"] == "structured_role_not_source_body"
    assert report["evidence"] == report["unscoped_source_quotes"] == report["state"]["followup_candidates"] == []
    assert "Havertz" not in json.dumps(report)
    assert source == before  # Real links remain available to the separate discovery layer.


@pytest.mark.parametrize("metadata", [None, {}, {"version": []}, {"version": "future-version", "role": "article"}])
def test_missing_or_invalid_projection_does_not_authorize_full_text(metadata):
    source = document("Arsenal may rotate the lineup and need a fitness update.")
    if metadata is not None:
        source["structured_news"] = metadata
    report = project(source)
    assert not report["evidence"] and not report["state"]["followup_candidates"]
    assert report["document_reviews"][0]["reason"] == "registered_structured_body_projection_required"


def test_html_navigation_and_title_only_cannot_be_body_evidence(tmp_path):
    source = html_projection(tmp_path, '<html><title>Arsenal may rotate the lineup</title>'
        '<nav>Arsenal Kai Havertz may have an injury. <a href="/news/team-update">Team update</a></nav>'
        '<main><h1>Arsenal may rotate the lineup</h1></main></html>')
    assert source["structured_news"]["role"] == "unknown"
    assert source["links"]
    assert not project(source)["state"]["followup_candidates"]


def test_jsonld_independent_headline_is_excluded_but_exact_body_can_propose_query(tmp_path):
    headline = "Arsenal SECRETSTRIKER may have an injury."
    body = "Arsenal Kai Havertz may miss the starting lineup with an injury."
    raw = {"@type": "NewsArticle", "headline": headline, "articleBody": body,
           "datePublished": "2026-10-06T07:00:00Z"}
    source = html_projection(tmp_path, '<html><script type="application/ld+json">'+json.dumps(raw)+'</script></html>')
    before = deepcopy(source)
    assert source["text"] == headline+"\n"+body
    report = project(source)
    assert report["evidence"] and report["state"]["followup_candidates"]
    assert "SECRETSTRIKER" not in json.dumps(report)
    assert report["evidence"][0]["statement"] == body
    anchor = report["evidence"][0]["anchor"]
    assert anchor["start"] == len(headline)+1 and source["text"][anchor["start"]:anchor["end"]] == body
    assert report["document_reviews"][0]["excluded_segments"][0]["reason"] == "projector_headline_prefix_not_body_assertion"
    assert all(row["expected_gain"] is None for row in report["state"]["followup_candidates"])
    assert source == before and report["semantics"]["numeric_voi"] is None


def test_target_team_only_in_jsonld_headline_does_not_grant_body_applicability(tmp_path):
    raw = {"@type": "NewsArticle", "headline": "Arsenal may rotate the injured lineup.",
           "articleBody": "Manchester City may rotate their starting lineup after a busy week."}
    source = html_projection(tmp_path, '<script type="application/ld+json">'+json.dumps(raw)+'</script>')
    report = project(source)
    assert not report["evidence"] and not report["state"]["followup_candidates"]
    assert report["unscoped_source_quotes"]
    assert "Arsenal may rotate" not in json.dumps(report)


def test_jsonld_headline_without_body_and_truncated_unknown_boundary_are_excluded(tmp_path):
    raw = {"@type": "NewsArticle", "headline": "Arsenal may rotate their injured starting lineup."}
    source = html_projection(tmp_path, '<script type="application/ld+json">'+json.dumps(raw)+'</script>')
    assert not project(source)["evidence"]
    source = article(["Arsenal may rotate their injured starting lineup."])
    source["structured_news"]["segments"][0]["truncated"] = True
    result = project(source)
    assert not result["evidence"]
    assert result["document_reviews"][0]["excluded_segments"][0]["reason"] == "truncated_jsonld_body_boundary_unknown"


def test_flattened_html_article_cannot_distinguish_navigation_from_body(tmp_path):
    raw = '<article><nav>Arsenal NAVSTRIKER may need an injury update.</nav>'
    raw += '<h1>Arsenal may rotate the lineup.</h1><p>'+('Generic federation information. '*8)+'</p>'
    raw += '<p>'+('Community programme information. '*8)+'</p></article>'
    source = html_projection(tmp_path, raw)
    assert source["structured_news"]["role"] == "article"
    report = project(source)
    assert not report["evidence"] and not report["unscoped_source_quotes"]
    assert "NAVSTRIKER" not in json.dumps(report)
    assert report["document_reviews"][0]["excluded_segments"][0]["reason"] == "flattened_html_body_boundaries_unavailable"


def test_known_html_paragraph_can_be_evidence_but_headline_alone_cannot():
    source = article([
        {"text": "Arsenal SECRETSTRIKER may be injured.", "content_source": "jalgpall_news_detail_html",
         "content_role": "headline", "css_selector": "div.news-detail > div.head > h1"},
        {"text": "Arsenal Kai Havertz may miss the starting lineup.", "content_source": "jalgpall_news_detail_html",
         "content_role": "paragraph", "css_selector": "div.news-detail > div.body.def p"}])
    report = project(source)
    assert len(report["evidence"]) == 1 and report["state"]["followup_candidates"]
    assert "SECRETSTRIKER" not in json.dumps(report)


@pytest.mark.parametrize("field,value", [("content_role", "navigation"), ("content_source", "arbitrary_page"),
    ("json_pointer", None), ("json_pointer", "script[0].headline"), ("source_modified_at", "2026-10-06T10:00:00Z")])
def test_invalid_body_provenance_or_future_segment_never_contributes_followup(field, value):
    source = article(["Arsenal SECRETSTRIKER may miss the lineup with an injury."])
    source["structured_news"]["segments"][0][field] = value
    report = project(source)
    assert not report["evidence"] and not report["state"]["followup_candidates"]
    assert "SECRETSTRIKER" not in json.dumps(report)


@pytest.mark.parametrize("version,expected", [
    (LEGACY_VERSION, "d0bb8e8c5b391ecf53780344cb1646444f2897b3bfbb19a4a8322a2cd9bc5028"),
    (VERSION, "dd4ecc4a5039b68886189906da81da2a7803907f6c7f9fb666f83f07d5f60220"),
    (ARTICLES_VERSION, "ff93fb9021ea6f83ec622ede0f00e297e594ef7691bceb007f1c81b19355747a"),
])
def test_v1_v2_v3_fixture_outputs_match_pre_change_snapshot(version, expected):
    text = "Arsenal may change the lineup because Kai Havertz has an injury."
    unknown = article([text], role="unknown")
    unknown["kind"] = "page"
    sources = [document(text), article([text]), unknown]
    reports = [build_match_evidence(MATCH, [source], as_of=AS_OF, version=version) for source in sources]
    assert frozen_digest(reports) == expected
    for source in sources:
        assert build_match_evidence(MATCH, [source], as_of=AS_OF) == build_match_evidence(MATCH, [source], as_of=AS_OF, version=VERSION)


def test_real_canada_archive_no_longer_promotes_navigation_but_legacy_replays_exactly():
    from football_research.data_adapters.open_text_episodes import _verify_document
    folder = Path(__file__).resolve().parents[2]/"var/parallel/agent/round190/live-canada/match-3097883"
    if not folder.exists():
        pytest.skip("Real Round190 archive not available")
    paths = sorted(path for path in folder.rglob("*") if path.is_file())
    before = {path: sha256(path.read_bytes()).hexdigest() for path in paths}
    config = json.loads((folder/"configuration.json").read_text(encoding="utf-8"))
    receipt = json.loads((folder/"receipt-001.json").read_text(encoding="utf-8"))
    source = _verify_document(folder, config, receipt)
    assert source["structured_news"]["role"] == "unknown" and source["links"]
    hashes = {LEGACY_VERSION: "b3cf731f782da92b653a580c3cdea03ec83ad74ac1deb302c6c773b4fbc412d3",
        VERSION: "26ae3f679a36f03f0c00dff99588bea50d0350f36a6d72ee1fc6b2192f7274bd",
        ARTICLES_VERSION: "1c84e29eb80ded61cf7dffd74ab002a3023f2589ff6f031fb38932a7f568a190"}
    for version, expected in hashes.items():
        old = build_match_evidence(config["evidence_projection"]["match"], [source],
                                    as_of=source["available_at"], version=version)
        assert frozen_digest(old) == expected
        assert len(old["state"]["followup_candidates"]) == 1  # Archived old behavior is preserved.
    current = build_match_evidence(config["evidence_projection"]["match"], [source],
                                    as_of=source["available_at"], version=CONTENT_VERSION)
    assert not current["evidence"] and not current["state"]["followup_candidates"]
    assert "Coach Education Program" not in json.dumps(current)
    assert current["schema_version"] == CONTENT_VERSION
    assert {path: sha256(path.read_bytes()).hexdigest() for path in paths} == before
