"""Archived real-template and negative boundary tests; no live HTTP."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import json
import socket

import pytest

from football_agent.candidate_context import _labels
from football_agent.match_evidence_articles import _body_segment_v5
from football_agent.news_discovery import NewsDiscoveryTools
from football_agent.news_cms_projection import project_index_cards
from football_spider.storage.raw_archive import RawArchive

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "var/three-match-20261010-v1"
NOW = datetime(2026, 10, 10, 13, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("CMS projection cannot make HTTP or DNS requests")
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)


def archived(host):
    folder = ("batch-3003905/match-3003905/source-archive" if host == "www.fulhamfc.com"
              else "source-recovery/archive")
    archive = RawArchive(OLD / folder)
    observation = next(o for o in archive.observations("open_page")
        if host in o.endpoint and (host != "www.fulhamfc.com" or "matchday-preview-ipswich" in o.endpoint))
    return archive, observation


@pytest.mark.parametrize("host,minimum", [("www.fulhamfc.com", 10), ("www.brentfordfc.com", 8),
                                        ("www.chelseafc.com", 10)])
def test_real_archived_article_body_is_bound_and_accepted_by_f_contract(host, minimum):
    archive, obs = archived(host)
    tools = NewsDiscoveryTools(archive, [host], projection_version=12, clock=lambda: NOW)
    doc = tools.project(obs, {"tool": "read_url", "source": "open_page", "endpoint": obs.endpoint, "params": {}})
    assert doc["status"] == "OK", doc["structured_news"]["excluded"]
    assert doc["kind"] == "article" and doc["source_published_at"]
    news = doc["structured_news"]
    assert len(news["segments"]) >= minimum
    for segment in news["segments"]:
        assert doc["text"][segment["start"]:segment["end"]] == segment["text"]
        assert _body_segment_v5(segment, doc, news)[1] is None
    assert NewsDiscoveryTools.from_configuration(archive, tools.configuration(), clock=lambda: NOW).configuration() == tools.configuration()


@pytest.mark.parametrize("host", ["www.fulhamfc.com", "www.brentfordfc.com", "www.chelseafc.com"])
def test_future_publication_never_enters_cms_body(host):
    archive, obs = archived(host)
    old_clock = datetime(2026, 9, 1, tzinfo=timezone.utc)
    early = obs.model_copy(update={"observed_at": old_clock})
    tools = NewsDiscoveryTools(archive, [host], projection_version=12, clock=lambda: NOW)
    doc = tools.project(early, {"tool": "read_url", "source": "open_page", "endpoint": obs.endpoint, "params": {}})
    assert doc["status"] == "NO_DATA"
    assert doc["structured_news"]["segments"] == [] and doc["text"] == ""


def test_cards_expose_only_same_response_metadata_and_no_target_body():
    from html import escape
    data = {"news": [{"title": "Chelsea team news for Bournemouth", "url": "/en/news/article/team-news"},
                     {"title": "Outside", "url": "https://other.example/en/news/article/outside"}]}
    html = '<div data-props="' + escape(json.dumps(data), quote=True) + '"></div>'
    doc = {"url": "https://www.chelseafc.com/en", "links": [], "text": ""}
    result = project_index_cards(doc, html, ["www.chelseafc.com"])
    assert len(result["links"]) == 1
    label = result["links"][0]
    labels, basis, _ = _labels(label)
    assert basis == "observed_CMS_card_metadata" and labels[0]["anchor_index"] is None
    assert label["snippet"] is None and label["target_article_fetched_for_context"] is False
    malformed = deepcopy(label)
    malformed["observed_metadata_labels"][0]["pointer"] = "another-response"
    with pytest.raises(ValueError, match="literal response pointer"):
        _labels(malformed)


def test_nuxt_body_cannot_masquerade_as_old_projection():
    archive, obs = archived("www.fulhamfc.com")
    tools = NewsDiscoveryTools(archive, ["www.fulhamfc.com"], projection_version=12, clock=lambda: NOW)
    doc = tools.project(obs, {"tool": "read_url", "source": "open_page", "endpoint": obs.endpoint, "params": {}})
    doc["news_projection_version"] = "structured-news-discovery-v11"
    assert _body_segment_v5(doc["structured_news"]["segments"][0], doc, doc["structured_news"])[1] is not None
