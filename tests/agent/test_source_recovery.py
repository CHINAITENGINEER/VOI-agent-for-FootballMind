"""Source-specific routes retain original archives and attempt accounting."""
from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
import requests

from football_agent.inquiry_tools import OpenWebTools
from football_agent.source_recovery import SourceRoutedOpenWebTools
from football_agent.transport_probe import DiagnosticSession, exception_facts
from football_spider.storage.raw_archive import RawArchive


NOW = datetime(2026, 10, 6, 10, 45, tzinfo=timezone.utc)


def configured(tmp_path, **kwargs):
    return SourceRoutedOpenWebTools(RawArchive(tmp_path), ["www.leedsunited.com"],
        route_overrides={"www.bing.com": "direct"}, clock=lambda: NOW, **kwargs)


def test_frozen_configuration_reconstructs_without_io(tmp_path):
    tools = configured(tmp_path)
    rebuilt = SourceRoutedOpenWebTools.from_configuration(tools.archive, tools.configuration(), clock=lambda: NOW)
    assert rebuilt.configuration() == tools.configuration()
    config = deepcopy(tools.configuration())
    config["source_route_policy"]["automatic_failover"] = True
    with pytest.raises(ValueError, match="reconstruct"):
        SourceRoutedOpenWebTools.from_configuration(tools.archive, config)


def test_search_override_does_not_change_page_route_and_both_replay(tmp_path, monkeypatch):
    calls = []

    def get(session, url, **kwargs):
        calls.append((url, session.trust_env))
        body = (b'<rss><channel><item><title>Observed article</title><link>https://www.leedsunited.com/en/news/observed</link></item></channel></rss>'
                if "bing.com" in url else b'<html><main>Actual observed article text.</main></html>')
        return SimpleNamespace(status_code=200, content=body,
            headers={"content-type": "text/html"}, close=lambda: None)

    monkeypatch.setattr(DiagnosticSession, "get", get)
    tools = configured(tmp_path)
    search = tools.prepare({"tool": "search_web", "query": "fixture match preview"}, {})
    returned_search = tools.execute(search, run_id="source-test", request_id="search")
    url = returned_search["document"]["links"][0]["url"]
    article = tools.prepare({"tool": "read_url", "url": url},
        {"task": {"seed_urls": []}, "documents": [returned_search["document"]]})
    returned_article = tools.execute(article, run_id="source-test", request_id="article")
    assert calls == [("https://www.bing.com/search", False), (url, True)]
    assert returned_search["http_attempts"] == returned_article["http_attempts"] == 1
    assert returned_search["observation"]["context"]["transport"]["route"] == "direct"
    assert returned_article["observation"]["context"]["transport"]["route"] == "environment"
    base = OpenWebTools(tools.archive, tools.allowed_hosts, extraction_version=3, clock=lambda: NOW)
    by_id = {obs.id: obs for obs in tools.archive.observations()}
    for result, prepared in ((returned_search, search), (returned_article, article)):
        assert base.project(by_id[result["observation"]["id"]], prepared) == result["document"]


def test_failed_override_remains_one_archived_failure_without_fallback(tmp_path, monkeypatch):
    calls = []

    def fail(session, url, **kwargs):
        calls.append(session.trust_env)
        error = requests.ConnectionError(ConnectionResetError(10054, "test failure"))
        session.failure = exception_facts(error)
        raise error

    monkeypatch.setattr(DiagnosticSession, "get", fail)
    tools = configured(tmp_path)
    prepared = tools.prepare({"tool": "search_web", "query": "unchanged"}, {})
    returned = tools.execute(prepared, run_id="source-test", request_id="failed")
    assert calls == [False]
    assert returned["http_attempts"] == 1 and returned["document"]["status"] == "FAILED"
    assert len(tools.archive.observations()) == 1
    assert returned["observation"]["context"]["transport"]["route"] == "direct"
    assert returned["money"] is None


def test_candidate_cannot_choose_route_and_unregistered_override_is_rejected(tmp_path):
    tools = configured(tmp_path)
    prepared = tools.prepare({"tool": "search_web", "query": "query"}, {})
    with pytest.raises(ValueError, match="override"):
        tools.execute({**prepared, "route": "direct"}, run_id="test", request_id="test")
    with pytest.raises(ValueError, match="registered"):
        SourceRoutedOpenWebTools(tools.archive, tools.allowed_hosts,
            route_overrides={"other.example": "direct"})
