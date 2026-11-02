"""Explicit provider/filter layers and source-bound visible directory cards."""
from datetime import datetime, timezone
from urllib.parse import urlencode
from xml.sax.saxutils import escape

import pytest

from football_agent.news_discovery import NewsDiscoveryTools
from football_spider.storage.raw_archive import RawArchive


NOW = datetime(2026, 10, 9, 12, tzinfo=timezone.utc)
INDEX = "https://www.leedsunited.com/en/news"


def project(tmp_path, body, *, version=9, search=True, host="www.leedsunited.com"):
    archive = RawArchive(tmp_path / "archive")
    tools = NewsDiscoveryTools(archive, [host], projection_version=version,
                               index_urls=[INDEX] if host == "www.leedsunited.com" else [], clock=lambda: NOW)
    prepared = {"tool": "search_web" if search else "read_url", "source": "open_search" if search else "open_page",
                "endpoint": "https://www.bing.com/search" if search else INDEX,
                "params": {"q": "Leeds United & squad update", "format": "rss"} if search else {}}
    obs = archive.save(prepared["source"], prepared["endpoint"], prepared["params"], body.encode(), status=200, observed_at=NOW)
    return tools.project(obs, prepared), tools


@pytest.mark.parametrize("body,expected,provider_count", [
    ("<html>bad</html>", "parse_error", None),
    ("<rss><channel/></rss>", "provider_empty", 0),
    ('<rss><channel><item><link>https://outside.example/a</link><title>Leeds</title></item></channel></rss>', "filtered_empty", 1),
    ('<rss><channel><item><link>https://www.leedsunited.com/en/news/a</link><title>Unrelated retail announcement</title></item></channel></rss>', "returned", 1),
])
def test_v9_distinguishes_parse_provider_and_filter_layers_without_semantic_imputation(tmp_path, body, expected, provider_count):
    doc, tools = project(tmp_path, body)
    assert doc["filter_status"] == expected
    assert doc["structured_news"]["retrieval_diagnostics"]["provider_result_count"] == provider_count
    assert doc["relevance_status"] == "not_assessed"
    assert doc["structured_news"]["segments"] == []
    assert "captured_at" not in doc["projection_provenance"]
    assert doc["projection_provenance"]["response_observed_at"] == NOW.isoformat()
    assert NewsDiscoveryTools.from_configuration(tools.archive, tools.configuration(), clock=lambda: NOW).configuration() == tools.configuration()


def test_v9_query_echo_checks_encoding_without_inferring_result_relevance(tmp_path):
    query = "Leeds United & squad update"
    url = "https://www.bing.com/search?" + urlencode({"q": query})
    doc, _ = project(tmp_path, '<rss><channel><link>'+escape(url)+'</link></channel></rss>')
    assert doc["structured_news"]["retrieval_diagnostics"]["provider_query_echo_matches"] is True


def test_v9_directory_retains_visible_card_date_missing_summary_and_no_article_body(tmp_path):
    raw = '''<main><article class="articleCard_articleCard__abc">
    <a href="/en/news/a">Squad update</a><h5>Squad update</h5>
    <div data-testid="publish-date">08 Oct 26</div>
    <p hidden>SECRET UNREQUESTED BODY</p></article>
    <script>{"articleBody":"SECRET SERIALIZED BODY"}</script></main>'''
    doc, _ = project(tmp_path, raw, search=False)
    link = doc["links"][0]
    assert link["snippet"] is None
    assert link["source_publication_date"] == "2026-10-08"
    assert link["source_published_at"] is None
    assert link["target_article_fetched_for_context"] is False
    assert "SECRET" not in str(link)
    assert doc["kind"] == "index" and doc["structured_news"]["segments"] == []


def test_v9_visible_summary_is_not_filled_from_title_or_other_card(tmp_path):
    raw = '''<main><article class="articleCard_articleCard__abc">
    <a href="/en/news/a">A</a><p>Actual parent card summary</p></article>
    <article class="articleCard_articleCard__abc"><a href="/en/news/b">B</a></article></main>'''
    doc, _ = project(tmp_path, raw, search=False)
    assert doc["links"][0]["snippet"] == "Actual parent card summary"
    assert doc["links"][1]["snippet"] is None


def test_v8_projection_contract_keeps_legacy_provenance_and_no_new_filter_metadata(tmp_path):
    doc, tools = project(tmp_path, '<rss><channel/></rss>', version=8)
    assert "filter_status" not in doc
    assert "retrieval_diagnostics" not in doc["structured_news"]
    assert "captured_at" in doc["projection_provenance"]
    assert "rss_diagnostics" not in tools.configuration()["news_discovery"]
