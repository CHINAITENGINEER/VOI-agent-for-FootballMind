"""Opt-in CMS body binding, with original archived projections unchanged."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import socket
import sys

import pytest

from football_agent.news_discovery import NewsDiscoveryTools
from football_research.acquisition.visible_state import encode_visible_state
from football_spider.models.contracts import Observation
from football_spider.storage.raw_archive import RawArchive
from test_news_discovery_v3 import CAPTURED, PARSED, URL, TITLE, P1, P2, node, project


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        pytest.fail("Only saved HTTP bytes or constructed fixtures are permitted")
    for obj, name in ((socket, "create_connection"), (socket, "getaddrinfo"),
                      (socket.socket, "connect"), (socket.socket, "connect_ex")):
        monkeypatch.setattr(obj, name, denied)


def cms(*, marker='class="elementor-widget-theme-post-content"', metadata=None, extra="", content=None):
    metadata = node() if metadata is None else metadata
    body = ('<p>'+P1+'</p><p>'+P2+'</p>') if content is None else content
    return ('<html><head><link rel="canonical" href="'+URL+'">'
        '<script type="application/ld+json">'+json.dumps(metadata)+'</script></head>'
        '<body><h1>'+TITLE+'</h1><div '+marker+'>'+body+'</div>'
        '<article><h2>RELATED_CARD_ONE</h2></article><article><h2>RELATED_CARD_TWO</h2></article>'
        '<article><h2>RELATED_CARD_THREE</h2></article>'+extra+'</body></html>')


@pytest.mark.parametrize("marker", ['class="elementor-widget-theme-post-content"',
                                   'itemprop="articleBody"', 'class="entry-content"'])
def test_unique_source_structural_body_and_page_h1_bind_without_host_or_topic_rules(tmp_path, marker):
    doc, tools, obs, prepared = project(tmp_path, cms(marker=marker), version=5)
    assert doc["status"] == "OK" and doc["text"] == P1+"\n"+P2
    assert doc["news_projection_version"] == "structured-news-discovery-v5"
    news = doc["structured_news"]
    assert news["version"] == "structured-news-discovery-v3" and news["role"] == "article"
    assert news["body_binding"]["container_selection"] == "unique_registered_structural_body"
    assert news["body_binding"]["heading_text"] == TITLE
    assert "RELATED_CARD" not in doc["text"]
    assert NewsDiscoveryTools.from_configuration(tools.archive, tools.configuration(), clock=lambda: PARSED).project(obs, prepared) == doc
    for i, segment in enumerate(news["segments"]):
        assert segment["content_source"] == "semantic_html_article_paragraph"
        assert segment["original_dom_paragraph_index"] == i
        assert doc["text"][segment["start"]:segment["end"]] == segment["text"]
        assert segment["source_url"] == URL
    assert doc["observed_at"] == CAPTURED.isoformat() and doc["available_at"] == PARSED.isoformat()


@pytest.mark.parametrize("case", ["two_bodies", "hidden_body", "hidden_title", "two_titles", "wrong_title",
    "wrong_page", "future_metadata", "missing_metadata", "competing_article", "unregistered_div",
    "template_body", "template_title"])
def test_ambiguous_hidden_unbound_and_future_metadata_do_not_gain_a_body(tmp_path, case):
    text = cms()
    if case == "two_bodies": text = cms(extra='<div class="entry-content"><p>'+P1+'</p><p>'+P2+'</p></div>')
    elif case == "hidden_body": text = cms(marker='class="entry-content" style="display:none"')
    elif case == "hidden_title": text = text.replace('<h1>', '<h1 hidden>')
    elif case == "two_titles": text = cms(extra='<h1>'+TITLE+'</h1>')
    elif case == "wrong_title": text = text.replace('<h1>'+TITLE, '<h1>Another title')
    elif case == "wrong_page": text = cms(metadata=node(mainEntityOfPage={"@id": URL+"-other"}))
    elif case == "future_metadata": text = cms(metadata=node(dateModified=(CAPTURED+timedelta(hours=1)).isoformat()))
    elif case == "missing_metadata": text = cms(metadata={"@type": "WebPage", "name": TITLE})
    elif case == "competing_article": text = cms(extra='<article><p>'+P1+'</p><p>'+P2+'</p></article>')
    elif case == "unregistered_div": text = cms(marker='class="content large-body-text"')
    elif case == "template_body": text = text.replace('<div class="elementor', '<template><div class="elementor').replace('</div>', '</div></template>')
    elif case == "template_title": text = text.replace('<h1>', '<template><h1>').replace('</h1>', '</h1></template>')
    doc, _, _, _ = project(tmp_path, text, version=5)
    assert doc["structured_news"]["role"] == "unknown" and doc["structured_news"]["segments"] == []
    if case != "missing_metadata":
        assert doc["status"] == "NO_DATA"
    else:
        assert doc["status"] == "OK"  # Unknown page is not imputed as neutral.


def test_nested_navigation_hidden_text_and_duplicate_registered_roots_are_not_guessed(tmp_path):
    text = cms(content='<p>'+P1+'</p><aside><p>ASIDE_SENTINEL</p></aside>'
        '<p>'+P2+'<span aria-hidden="true">HIDDEN_SENTINEL</span></p>'
        '<p><a href="/elsewhere">LINK_CARD_SENTINEL</a></p><template><p>TEMPLATE_SENTINEL</p></template>')
    doc, _, _, _ = project(tmp_path/"good", text, version=5)
    assert doc["status"] == "OK" and "SENTINEL" not in doc["text"]
    assert [s["original_dom_paragraph_index"] for s in doc["structured_news"]["segments"]] == [0, 2]
    nested = cms(content='<div class="entry-content"><p>'+P1+'</p><p>'+P2+'</p></div>')
    rejected, _, _, _ = project(tmp_path/"nested", nested, version=5)
    assert rejected["status"] == "NO_DATA"


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_versions_one_to_four_keep_original_configuration_and_projected_bytes(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    backup = root/"var/parallel/agent/round195/source-before/football_agent"
    if not (backup/"news_discovery.py").is_file():
        pytest.skip("Optional original source archive is absent")
    old_news = _module("football_agent._r195_old_news", backup/"news_discovery.py")
    old_body = _module("football_agent._r195_old_body", backup/"news_html_body.py")
    archive = RawArchive(tmp_path/"archive")
    obs = archive.save("open_page", URL, {}, cms().encode(), status=200, observed_at=CAPTURED)
    prepared = {"tool": "read_url", "endpoint": URL, "params": {}, "source": "open_page"}
    for version in (1, 2, 3, 4):
        current = NewsDiscoveryTools(archive, ["club.example"], projection_version=version, clock=lambda: PARSED)
        previous = old_news.NewsDiscoveryTools(archive, ["club.example"], projection_version=version, clock=lambda: PARSED)
        assert current.configuration() == previous.configuration()
        new_doc = current.project(obs, prepared)
        with monkeypatch.context() as mp:
            mp.setitem(sys.modules, "football_agent.news_html_body", old_body)
            assert new_doc == previous.project(obs, prepared)


def test_two_actual_archived_pages_recover_body_and_encode_without_changing_old_no_data(tmp_path):
    root = Path(__file__).resolve().parents[2]
    folder = root/"var/parallel/agent/round195/panel"
    if not (folder/"panel.json").is_file():
        pytest.skip("Optional original research panel is absent")
    panel = json.loads((folder/"panel.json").read_text(encoding="utf-8"))
    archive = RawArchive(folder/"source-archive")
    # Reprojection must stay within this archived panel's declared horizon;
    # the wall clock eventually passes its cutoff and correctly hides the body.
    at = datetime.fromisoformat(panel["design"]["public_before"]["state"]["prediction_cutoff"])
    transport = panel["design"]["transport"]
    results = []
    frozen = {p: p.read_bytes() for p in folder.rglob("*") if p.is_file()}
    old = NewsDiscoveryTools.from_configuration(archive, transport)
    for path in sorted(folder.glob("capture-*.json")):
        capture = json.loads(path.read_text(encoding="utf-8"))
        observation = Observation.model_validate(capture["result"]["observation"])
        original = capture["result"]["document"]
        replay = NewsDiscoveryTools.from_configuration(archive, transport,
            clock=lambda d=original: datetime.fromisoformat(d["available_at"]))
        assert replay.project(observation, capture["prepared"]) == original
        assert original["status"] == "NO_DATA"
        tools = NewsDiscoveryTools(archive, old.allowed_hosts, provider=old.provider, projection_version=5,
                                   clock=lambda: at, index_urls=old.index_urls)
        doc = tools.project(observation, capture["prepared"])
        assert doc["status"] == "OK" and len(doc["structured_news"]["segments"]) >= 2
        assert doc["observed_at"] == original["observed_at"] and doc["available_at"] == at.isoformat()
        public = deepcopy(panel["design"]["public_before"])
        public.pop("candidate_contexts", None)
        public["decision_at"] = at.isoformat()
        public["documents"].append(doc)
        encoded = encode_visible_state(public)
        assert any(d["content_refs"] for d in encoded["before"]["documents"])
        assert any(segment["text"] in [v["text"] for v in encoded["before"]["contents"]]
                   for segment in doc["structured_news"]["segments"])
        results.append(len(doc["structured_news"]["segments"]))
    assert len(results) == 2
    assert all(p.read_bytes() == value for p, value in frozen.items())
