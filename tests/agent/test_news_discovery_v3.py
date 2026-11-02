"""Explicit metadata-bound DOM paragraphs; no network or match-value claims."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from html import escape
import json
from pathlib import Path
import socket

import pytest

from football_agent.news_discovery import NewsDiscoveryTools
from football_spider.models.contracts import Observation
from football_spider.storage.raw_archive import RawArchive


CAPTURED = datetime(2026, 10, 6, 19, tzinfo=timezone.utc)
PARSED = CAPTURED+timedelta(hours=1)
URL = "https://club.example/observed-source-story"
TITLE = "An acquired source article"
P1 = "The team discussed an unfamiliar communication protocol before its next appearance."
P2 = "Its staff described an unresolved practical question, without establishing the answer."


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("News version 3 tests cannot access the network")
    for name in ("create_connection", "getaddrinfo"):
        monkeypatch.setattr(socket, name, blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)


def node(**changes):
    return {"@type": "NewsArticle", "headline": TITLE, "url": URL,
        "mainEntityOfPage": {"@id": URL}, "datePublished": "2026-10-06T17:00:00Z",
        "dateModified": "2026-10-06T18:00:00Z", **changes}


def html(metadata=None, *, content=None, tag="article", headline=TITLE):
    metadata = node() if metadata is None else metadata
    content = "<p>"+P1+"</p><p>"+P2+"</p>" if content is None else content
    return ('<html lang="en"><head><link rel="canonical" href="'+URL+'">'
        '<script type="application/ld+json">'+json.dumps(metadata)+'</script></head>'
        '<body><header><h1>Unrelated source masthead</h1></header><'+tag+'>'
        '<header><h1>'+escape(headline)+'</h1><p>HEADER_SENTINEL</p></header>'
        +content+'</'+tag+'></body></html>')


def project(tmp_path, content=None, *, version=3, parsed=PARSED, status=200, indexes=()):
    archive = RawArchive(tmp_path/"archive")
    prepared = {"tool": "read_url", "endpoint": URL, "params": {}, "source": "open_page"}
    obs = archive.save("open_page", URL, {}, (html() if content is None else content).encode(),
        status=status, observed_at=CAPTURED)
    tools = NewsDiscoveryTools(archive, ["club.example"], projection_version=version,
                              clock=lambda: parsed, index_urls=indexes)
    return tools.project(obs, prepared), tools, obs, prepared


@pytest.mark.parametrize("tag", ["article", "main"])
def test_bound_paragraphs_have_original_dom_indices_offsets_and_source_clocks(tmp_path, tag):
    doc, tools, obs, prepared = project(tmp_path, html(tag=tag))
    news = doc["structured_news"]
    assert news["role"] == "article" and doc["kind"] == "article"
    assert doc["text"] == P1+"\n"+P2 and TITLE not in doc["text"] and "HEADER_SENTINEL" not in doc["text"]
    assert [s["original_dom_paragraph_index"] for s in news["segments"]] == [1, 2]
    assert news["body_binding"]["metadata_headline"] == news["body_binding"]["heading_text"] == TITLE
    assert news["body_binding"]["source_url"] == URL
    for segment in news["segments"]:
        assert segment["content_source"] == "semantic_html_article_paragraph"
        assert segment["content_role"] == "paragraph" and segment["source_url"] == URL
        assert segment["css_selector"].endswith("p:nth-of-type("+str(segment["original_dom_paragraph_index"])+")")
        assert doc["text"][segment["start"]:segment["end"]] == segment["text"]
        assert segment["source_published_at"] == "2026-10-06T17:00:00+00:00"
        assert segment["source_modified_at"] == "2026-10-06T18:00:00+00:00"
    assert doc["observed_at"] == CAPTURED.isoformat() and doc["available_at"] == PARSED.isoformat()
    assert NewsDiscoveryTools.from_configuration(tools.archive, tools.configuration(), clock=lambda: PARSED).project(obs, prepared) == doc


@pytest.mark.parametrize("case", ["no_metadata", "wrong_url", "conflicting_page", "wrong_title", "hidden_title",
    "no_title", "two_metadata", "two_articles", "single_pretty_paragraph", "unknown_html", "liveblog",
    "future_published", "future_modified", "modified_before_published", "timezone_unknown", "invalid_time"])
def test_unknown_conflicting_and_temporally_invalid_sources_cannot_enter_through_dom(tmp_path, case):
    metadata, content = node(), None
    if case == "wrong_url": metadata["url"] = "https://club.example/unrelated"
    elif case == "conflicting_page": metadata["mainEntityOfPage"] = {"@id": "https://club.example/unrelated"}
    elif case == "two_metadata": metadata = [node(), node(headline="Another story")]
    elif case == "liveblog": metadata["@type"] = "LiveBlogPosting"
    elif case == "future_published": metadata["datePublished"] = "2026-10-06T19:30:00Z"
    elif case == "future_modified": metadata["dateModified"] = "2026-10-06T19:30:00Z"
    elif case == "modified_before_published": metadata["dateModified"] = "2026-10-06T16:00:00Z"
    elif case == "timezone_unknown": metadata["datePublished"] = "2026-10-06T17:00:00"
    elif case == "invalid_time": metadata["datePublished"] = "not-a-clock"
    elif case == "no_metadata": metadata = {"@type": "WebPage", "name": TITLE}
    elif case == "single_pretty_paragraph": content = "<p>"+P1+"</p>"
    text = html(metadata, content=content)
    if case == "wrong_title": text = html(metadata, headline="Unrelated heading")
    elif case == "hidden_title": text = text.replace('<h1>'+TITLE, '<h1 style="display:none">'+TITLE)
    elif case == "no_title": text = text.replace("<h1>"+TITLE+"</h1>", "")
    elif case == "two_articles": text += '<article><h1>'+TITLE+'</h1><p>'+P1+'</p><p>'+P2+'</p></article>'
    elif case == "unknown_html": text = '<div><h1>'+TITLE+'</h1><p>'+P1+'</p><p>'+P2+'</p></div>'
    doc, _, _, _ = project(tmp_path, text)
    assert doc["structured_news"]["role"] == "unknown" and doc["structured_news"]["segments"] == []
    if case in {"no_metadata", "unknown_html"}:
        assert doc["status"] == "OK" and doc["kind"] == "page" and P1 in doc["text"]
    else:
        assert doc["status"] == "NO_DATA" and doc["text"] == ""


@pytest.mark.parametrize("metadata", ["", '<script type="application/ld+json">broken JSON</script>',
    '<script type="application/ld+json">{"@type":"WebPage","name":"Unknown page"}</script>'])
def test_unknown_without_article_metadata_is_not_imputed_as_a_neutral_no_data(tmp_path, metadata):
    from football_agent.inquiry_tools import OpenWebTools
    content = '<html><body>'+metadata+'<p>'+P1+'</p><p>'+P2+'</p></body></html>'
    doc, tools, obs, prepared = project(tmp_path, content)
    base = OpenWebTools(tools.archive, ["club.example"], clock=lambda: PARSED, extraction_version=3).project(obs, prepared)
    assert {key: doc[key] for key in ("status", "kind", "text", "links")} == {
        key: base[key] for key in ("status", "kind", "text", "links")}
    assert doc["status"] == "OK" and doc["structured_news"]["role"] == "unknown"
    assert not doc["structured_news"]["segments"]


def test_navigation_share_hidden_link_cards_removed_without_removing_body_subjects(tmp_path):
    content = ('<nav><p>NAV_SENTINEL</p></nav><p>'+P1+'</p><aside><p>ASIDE_SENTINEL</p></aside>'
        '<div class="social-share"><p>SHARE_SENTINEL</p></div><div class="link-card"><p>CARD_SENTINEL</p></div>'
        '<p hidden>HIDDEN_SENTINEL</p><p><a href="/other">LINK_SENTINEL</a></p>'
        '<p>06/10/2026</p><p>Tickets, injuries, technology and weather are all permitted body subjects.</p>'
        '<footer><p>FOOTER_SENTINEL</p></footer><p>'+P2+'</p>')
    doc, _, _, _ = project(tmp_path, html(content=content))
    assert doc["status"] == "OK" and "SENTINEL" not in doc["text"]
    assert "Tickets, injuries, technology and weather" in doc["text"]
    assert len(doc["structured_news"]["segments"]) == 3
    assert [s["original_dom_paragraph_index"] for s in doc["structured_news"]["segments"]] == [2, 9, 11]


def test_missing_source_date_remains_unknown_and_new_projection_cannot_be_backdated(tmp_path):
    doc, _, _, _ = project(tmp_path, html(node(datePublished=None, dateModified=None)))
    assert doc["status"] == "OK" and doc["source_published_at"] is None
    assert all(s["source_published_at"] is None for s in doc["structured_news"]["segments"])
    with pytest.raises(ValueError, match="precede"):
        project(tmp_path/"late", parsed=CAPTURED-timedelta(seconds=1))


def test_index_failure_and_jsonld_exclusion_are_never_overridden(tmp_path):
    index, _, _, _ = project(tmp_path/"index", indexes=[URL])
    assert index["kind"] == "index" and index["structured_news"]["segments"] == []
    failure, _, _, _ = project(tmp_path/"failed", status=503)
    assert failure["status"] == "FAILED" and failure["kind"] != "article"
    rejected, _, _, _ = project(tmp_path/"excluded", html(node(articleBody=P1,
        datePublished="2026-10-06T23:00:00Z")))
    assert rejected["status"] == "NO_DATA" and rejected["text"] == ""


def test_v3_existing_jsonld_body_keeps_old_single_newline_format(tmp_path):
    payload = html(node(articleBody=P1+"\n"+P2))
    old, _, _, _ = project(tmp_path/"old", payload, version=2)
    new, _, _, _ = project(tmp_path/"new", payload)
    assert new["text"] == old["text"] == TITLE+"\n"+P1+" "+P2
    assert new["structured_news"]["segments"] == old["structured_news"]["segments"]


def test_actual_canada_receipt_recovers_body_but_preserves_original_no_data_and_wrong_fixture(tmp_path):
    root = Path(__file__).resolve().parents[2]
    folder = root/"var/parallel/agent/round191/live-canada/match-3097883"
    if not folder.exists():
        pytest.skip("Original optional Round191 archive is not included in this checkout")
    receipt_path, config_path = folder/"receipt-003.json", folder/"configuration.json"
    before = {p: p.read_bytes() for p in (receipt_path, config_path, folder/"run.json")}
    result, config = json.loads(before[receipt_path]), json.loads(before[config_path])
    archive = RawArchive(folder/"source-archive")
    observation = Observation.model_validate(result["receipt"]["observation"])
    raw_hash = sha256(archive.read_body(observation)).hexdigest()
    old_doc = result["document"]
    old = NewsDiscoveryTools.from_configuration(archive, config["supervisor_metadata"]["transport"],
        clock=lambda: datetime.fromisoformat(old_doc["available_at"]))
    assert old.project(observation, result["receipt"]["prepared"]) == old_doc
    assert old_doc["status"] == "NO_DATA" and old_doc["text"] == ""
    now = datetime.now(timezone.utc)
    tools = NewsDiscoveryTools(archive, config["supervisor_metadata"]["transport"]["allowed_hosts"],
        projection_version=3, clock=lambda: now)
    current = tools.project(observation, result["receipt"]["prepared"])
    assert current["kind"] == "article" and len(current["structured_news"]["segments"]) == 7
    assert [s["original_dom_paragraph_index"] for s in current["structured_news"]["segments"]] == list(range(2, 9))
    assert current["raw_sha256"] == raw_hash and current["observed_at"] == old_doc["observed_at"]
    assert current["available_at"] == now.isoformat() and current["available_at"] != old_doc["available_at"]
    assert "CANWNT" in current["structured_news"]["body_binding"]["metadata_headline"]
    assert "Denmark" in current["text"] and "Women" in current["text"]
    assert current["structured_news"]["segments"][0]["source_published_at"] == "2026-09-30T16:37:00+00:00"
    assert current["structured_news"]["segments"][0]["source_modified_at"] == "2026-09-30T19:09:23+00:00"
    assert sha256(archive.read_body(observation)).hexdigest() == raw_hash
    assert all(p.read_bytes() == content for p, content in before.items())
    # Scope is source text only; the evidence adapter separately rejects the
    # women's article for this men's fixture. Extraction never translates it
    # into a player absence, fixture claim, label or information-value number.
    assert not any(key in current for key in ("expected_gain", "outcome", "numeric_voi"))


def test_old_configuration_and_new_configuration_cannot_silently_change_versions(tmp_path):
    for version in (1, 2, 3):
        _, tools, _, _ = project(tmp_path/str(version), version=version)
        config = tools.configuration()
        assert NewsDiscoveryTools.from_configuration(tools.archive, config).configuration() == config
        changed = deepcopy(config)
        changed["news_discovery"]["version"] = "structured-news-discovery-v3"
        if version != 3:
            with pytest.raises(ValueError, match="exactly"):
                NewsDiscoveryTools.from_configuration(tools.archive, changed)
