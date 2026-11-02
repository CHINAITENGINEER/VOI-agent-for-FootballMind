"""Large official indexes keep observed article links for autonomous choice."""
from datetime import datetime, timezone
import json
from pathlib import Path
import socket

import pytest

from football_agent.news_discovery import NewsDiscoveryTools
from football_spider.models.contracts import Observation
from football_spider.storage.raw_archive import RawArchive


NOW = datetime(2026, 10, 9, 8, tzinfo=timezone.utc)
INDEX = "https://club.example/en"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("Index projection must use archived response bytes only")
    for obj, name in ((socket, "getaddrinfo"), (socket, "create_connection"),
                      (socket.socket, "connect"), (socket.socket, "connect_ex")):
        monkeypatch.setattr(obj, name, blocked)


def _project(tmp_path, version, *, count=100, raw=None):
    links = "".join(
        f'<a href="/en/news/article-{i}">Article {i}</a>'
        for i in range(count)
    )
    raw = raw if raw is not None else f"<html><body><main>{links}</main></body></html>"
    archive = RawArchive(tmp_path / "archive")
    observation = archive.save("open_page", INDEX, {}, raw.encode(), status=200, observed_at=NOW)
    tools = NewsDiscoveryTools(archive, ["club.example"], index_urls=[INDEX],
                               projection_version=version, clock=lambda: NOW)
    prepared = {"tool": "read_url", "source": "open_page", "endpoint": INDEX, "params": {}}
    return tools.project(observation, prepared), tools


def test_v8_does_not_hide_observed_news_links_after_generic_80_link_cap(tmp_path):
    old, old_tools = _project(tmp_path / "old", 7)
    new, new_tools = _project(tmp_path / "new", 8)

    # Historical v7 remains exactly bounded at the generic projection's 80
    # links; the new version is opt-in and preserves all 100 observed entries.
    assert len(old["links"]) == 80
    assert old["eligible_link_count"] == 100
    assert len(new["links"]) == 100
    assert new["eligible_link_count"] == 100
    assert new["links_truncated"] is False
    assert new["news_projection_version"] == "structured-news-discovery-v8"
    assert new_tools.configuration()["news_discovery"]["index_max_links"] == 320

    # No target is read, ranked, or assigned a value: only observed labels and
    # URLs are expanded in the archived index projection.
    assert new["links"][-1]["url"].endswith("/article-99")
    assert new["links"][-1]["title"] == "Article 99"
    restored = NewsDiscoveryTools.from_configuration(new_tools.archive, new_tools.configuration(), clock=lambda: NOW)
    assert restored.configuration() == new_tools.configuration()
    assert old_tools.configuration()["news_discovery"]["version"] == "structured-news-discovery-v7"


@pytest.mark.parametrize("count,truncated", [(320, False), (321, True), (350, True)])
def test_v8_discloses_actual_count_above_bound_without_reordering(tmp_path, count, truncated):
    doc, _ = _project(tmp_path, 8, count=count)
    assert len(doc["links"]) == min(count, 320)
    assert doc["eligible_link_count"] == count
    assert doc["links_truncated"] is truncated
    assert [link["url"] for link in doc["links"]] == [
        f"https://club.example/en/news/article-{i}" for i in range(min(count, 320))]


def test_v8_rejects_outside_hosts_schemes_credentials_and_private_literals(tmp_path):
    raw = '<main>' + ''.join(f'<a href="{url}">An observed label</a>' for url in [
        'https://other.example/en/news/outside', 'http://club.example/en/news/http',
        'javascript:alert(1)', 'https://user:password@club.example/en/news/credentials',
        'https://club.example:444/en/news/port', 'https://127.0.0.1/en/news/private',
        'https://club.example/en/news/query?token=secret',
        'https://club.example/en/news/allowed',
    ]) + '</main>'
    doc, _ = _project(tmp_path, 8, raw=raw)
    assert [x["url"] for x in doc["links"]] == ['https://club.example/en/news/allowed']
    assert doc['eligible_link_count'] == 1
    assert doc['links_truncated'] is False


def test_v8_hidden_anchors_and_descendants_do_not_enter_observed_menu(tmp_path):
    raw = '''<main>
        <div hidden><a href="/en/news/hidden-parent">Hidden parent</a></div>
        <a hidden href="/en/news/hidden-self">Hidden self</a>
        <a aria-hidden="true" href="/en/news/aria">ARIA hidden</a>
        <a style="display: none !important" href="/en/news/style">Style hidden</a>
        <div style="visibility: hidden"><a href="/en/news/visibility">Hidden ancestor</a></div>
        <template><a href="/en/news/template">Template</a></template>
        <a hidden href="/en/news/visible">Wrong label first</a>
        <a href="/en/news/visible">Actual label <span hidden>HIDDEN SENTINEL</span></a>
    </main>'''
    doc, _ = _project(tmp_path, 8, raw=raw)
    assert len(doc['links']) == doc['eligible_link_count'] == 1
    assert doc['links'][0]['title'] == 'Actual label'
    assert doc['links'][0]['observed_anchors'][0]['text'] == 'Actual label'
    assert 'HIDDEN SENTINEL' not in str(doc['links'])


def test_v8_resolves_observed_base_but_rechecks_resulting_host(tmp_path):
    raw = '''<head><base href="https://other.example/en/"></head><main>
        <a href="news/outside">Outside base</a>
        <a href="https://club.example/en/news/allowed">Allowed absolute</a>
    </main>'''
    doc, _ = _project(tmp_path, 8, raw=raw)
    assert [x['url'] for x in doc['links']] == ['https://club.example/en/news/allowed']


def test_real_v7_archive_replays_exactly_and_v8_expands_bounded_observed_links():
    root = Path(__file__).resolve().parents[2] / 'var/submission/prospective-shared-h2-20261009'
    if not (root / 'http/request-000.json').is_file():
        pytest.skip('Separately captured source archive is absent')
    receipt = json.loads((root / 'http/request-000.json').read_text(encoding='utf-8'))
    run = json.loads((root / 'run.json').read_text(encoding='utf-8'))
    original = receipt['returned']['document']
    archive = RawArchive(root / 'source-archive')
    observation = Observation.model_validate(receipt['returned']['observation'])
    if not (archive.root / observation.payload_path).is_file():
        pytest.skip('Separately captured raw response is absent')
    clock = lambda: datetime.fromisoformat(original['available_at'])
    old = NewsDiscoveryTools.from_configuration(archive, run['configuration']['transport'], clock=clock)
    assert old.project(observation, receipt['prepared']) == original
    current = NewsDiscoveryTools(archive, old.allowed_hosts, provider=old.provider,
        default_route=old.default_route, route_overrides=old.route_overrides,
        index_urls=old.index_urls, projection_version=8, clock=clock)
    doc = current.project(observation, receipt['prepared'])
    assert len(original['links']) == 80
    assert len(doc['links']) > 80
    assert len(doc['links']) <= 320
    assert doc['links_truncated'] == (doc['eligible_link_count'] > 320)
    assert doc['structured_news'] == original['structured_news']

