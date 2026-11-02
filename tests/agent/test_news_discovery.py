"""Opt-in archived news extraction and real-time visibility boundaries."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from urllib.parse import quote

import pytest

from football_agent.inquiry_tools import OpenWebTools
from football_agent.news_discovery import NewsDiscoveryTools
from football_spider.storage.raw_archive import RawArchive


NOW = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)
INDEX = "https://club.example/news"
ARTICLE = "https://club.example/news/match-preview"


def tools(tmp_path, **kwargs):
    return NewsDiscoveryTools(RawArchive(tmp_path/"raw"), ["club.example"], clock=lambda: NOW, **kwargs)


def project(t, html, *, search=False, endpoint=ARTICLE):
    prepared = ({"tool": "search_web", "endpoint": "https://www.bing.com/news/search",
                 "params": {"q": "Club Opponent", "format": "rss"}, "source": "open_search"} if search else
                {"tool": "read_url", "endpoint": endpoint, "params": {}, "source": "open_page"})
    observation = t.archive.save(prepared["source"], prepared["endpoint"], prepared["params"], html.encode(),
                                 status=200, observed_at=NOW)
    return t.project(observation, prepared), observation, prepared


def node(text="Club forward is available &amp; ready.", **kwargs):
    return {"@type": "NewsArticle", "articleBody": text, "url": ARTICLE,
            "datePublished": "2026-10-06T10:00:00Z", "dateModified": "2026-10-06T11:00:00Z", **kwargs}


def markup(value, visible="Old visible shell"):
    return '<html><body><p>'+visible+'</p><script type="application/ld+json">'+json.dumps(value)+'</script></body></html>'


def test_configuration_is_explicit_exact_and_old_default_unchanged(tmp_path):
    current = tools(tmp_path, provider="bing_news_rss", default_route="direct", index_urls=[INDEX])
    config = current.configuration()
    restored = NewsDiscoveryTools.from_configuration(current.archive, config, clock=lambda: NOW)
    assert restored.configuration() == config
    old = OpenWebTools(current.archive, ["club.example"], clock=lambda: NOW, extraction_version=3)
    assert old.configuration()["search_provider"] == "bing_rss"
    assert "news_discovery" not in old.configuration()
    changed = deepcopy(config)
    changed["news_discovery"]["max_segments"] = 65
    with pytest.raises(ValueError, match="exactly"):
        NewsDiscoveryTools.from_configuration(current.archive, changed)


def test_liverpool_v10_directory_title_is_visible_card_not_target_fetch(tmp_path):
    index = "https://www.liverpoolfc.com/news/category/men"
    html = '''<article data-testid="article-card"><a data-testid="article-card__link"
      href="/news/preview" aria-label="Liverpool v Manchester City preview"></a>
      <h3>Liverpool v Manchester City preview</h3><time datetime="2026-10-06">Today</time></article>
      <article data-testid="article-card" hidden><a data-testid="article-card__link"
      href="/news/hidden" aria-label="Hidden"><h3>Hidden</h3></a></article>'''
    t = NewsDiscoveryTools(RawArchive(tmp_path), ["www.liverpoolfc.com"],
                           clock=lambda: NOW, projection_version=10, index_urls=[index])
    doc, _, _ = project(t, html, endpoint=index)
    card = next(x for x in doc["links"] if x["url"].endswith("/preview"))
    assert card["title"] == "Liverpool v Manchester City preview"
    assert card["source_publication_date"] == "2026-10-06"
    assert card["source_published_at"] is None
    assert card["snippet"] is None
    assert card["target_article_fetched_for_context"] is False
    assert doc["structured_news"]["segments"] == []
    assert not any(x["url"].endswith("/hidden") for x in doc["links"])
    assert len(t.archive.observations()) == 1
    assert NewsDiscoveryTools.from_configuration(t.archive, t.configuration()).configuration() == t.configuration()


def test_liverpool_locale_alias_needs_canonical_alternate_and_exact_slug(tmp_path):
    url = "https://www.liverpoolfc.com/news/preview"
    meta = {"@type": "NewsArticle", "headline": "Match preview", "datePublished": "2026-10-06T10:00:00Z",
            "url": "https://www.liverpoolfc.com/en/news/preview",
            "mainEntityOfPage": {"@id": "https://www.liverpoolfc.com/en/news/preview"}}
    html = ('<html lang="en"><head><link rel="canonical" href="'+url+'">'
            '<link rel="alternate" hreflang="en" href="'+url+'"></head><body>'
            '<article><h1>Match preview</h1><p>First team report.</p><p>Second paragraph.</p></article>'
            '<script type="application/ld+json">'+json.dumps(meta)+'</script></body></html>')
    t = NewsDiscoveryTools(RawArchive(tmp_path), ["www.liverpoolfc.com"], clock=lambda: NOW, projection_version=11)
    doc, _, _ = project(t, html, endpoint=url)
    assert doc["structured_news"]["role"] == "article"
    assert "page_identity_alias" in doc["structured_news"]
    mismatched = html.replace('/en/news/preview', '/en/news/different')
    bad, _, _ = project(t, mismatched, endpoint=url)
    assert bad["status"] == "NO_DATA"
    old = NewsDiscoveryTools(t.archive, ["www.liverpoolfc.com"], clock=lambda: NOW, projection_version=10)
    old_doc, _, _ = project(old, html, endpoint=url)
    assert old_doc["status"] == "NO_DATA"


def test_only_real_acquired_links_or_source_seeds_can_be_prepared(tmp_path):
    t = tools(tmp_path, provider="bing_news_rss")
    public = {"task": {"seed_urls": [INDEX]}, "documents": []}
    assert t.prepare({"tool": "search_web", "query": "Club Opponent"}, public)["endpoint"] == "https://www.bing.com/news/search"
    with pytest.raises(ValueError, match="acquired links"):
        t.prepare({"tool": "read_url", "url": ARTICLE}, public)
    public["documents"] = [{"links": [{"url": ARTICLE}]}]
    assert t.prepare({"tool": "read_url", "url": ARTICLE}, public)["endpoint"] == ARTICLE


def test_news_rss_decodes_explicit_target_without_following_redirect_and_retains_time(tmp_path):
    t = tools(tmp_path, provider="bing_news_rss")
    wrapped = "https://www.bing.com/news/apiclick.aspx?ref=FexRss&amp;url="+quote(ARTICLE, safe="")
    body = f'<rss><channel><item><title>Club vs Opponent preview</title><link>{wrapped}</link><pubDate>Tue, 06 Oct 2026 10:30:00 GMT</pubDate><description>Team &amp;amp; news</description></item></channel></rss>'
    doc, _, _ = project(t, body, search=True)
    assert doc["status"] == "OK" and doc["kind"] == "discovery"
    assert doc["links"][0]["url"] == ARTICLE
    assert doc["links"][0]["source_published_at"] == "2026-10-06T10:30:00+00:00"
    assert not doc["structured_news"]["segments"]


def test_future_rss_item_and_ambiguous_or_unregistered_target_are_not_candidates(tmp_path):
    t = tools(tmp_path)
    body = '<rss><channel>'
    for url, time in [(ARTICLE, "Wed, 07 Oct 2026 10:00:00 GMT"),
                      ("https://evil.example/page", "Tue, 06 Oct 2026 10:00:00 GMT"),
                      ("https://www.bing.com/news/apiclick.aspx?url="+quote(ARTICLE)+"&amp;url="+quote(INDEX), "")]:
        body += f'<item><title>Club news</title><link>{url}</link><pubDate>{time}</pubDate></item>'
    doc, _, _ = project(t, body+'</channel></rss>', search=True)
    assert doc["status"] == "NO_DATA" and doc["links"] == []
    assert len(doc["structured_news"]["excluded"]) == 3
    assert len(doc["structured_news"]["pending_host_links"]) == 1


def test_unknown_search_host_is_retained_for_bounded_exact_host_review(tmp_path):
    t = tools(tmp_path, projection_version=13)
    target = "https://newclub.example/news/club-opponent-preview"
    body = ('<rss><channel><item><title>Club vs Opponent preview</title><link>'+target+
            '</link><pubDate>Tue, 06 Oct 2026 10:00:00 GMT</pubDate></item></channel></rss>')
    doc, _, _ = project(t, body, search=True)
    assert doc["status"] == "NO_DATA"
    assert doc["links"] == []
    assert doc["structured_news"]["pending_host_links"] == [{
        "url": target, "title": "Club vs Opponent preview", "source_published_at": "2026-10-06T10:00:00+00:00",
        "publication_raw": "Tue, 06 Oct 2026 10:00:00 GMT", "review_status": "pending_host_review",
        "review_reason": "host_not_allowlisted"}]
    assert doc["structured_news"]["retrieval_diagnostics"]["pending_host_review_count"] == 1


@pytest.mark.parametrize("raw_time", [
    "周一, 5 10月 2026 05:03:00 GMT",
    "2026年10月5日 05:03:00 GMT",
])
def test_localized_rss_time_keeps_unknown_host_in_review_queue(tmp_path, raw_time):
    t = tools(tmp_path, projection_version=13)
    target = "https://newclub.example/news/club-opponent-preview"
    body = ('<rss><channel><item><title>Club vs Opponent preview</title><link>'+target+
            '</link><pubDate>'+raw_time+'</pubDate></item></channel></rss>')
    doc, _, _ = project(t, body, search=True)
    assert doc["status"] == "NO_DATA"
    pending = doc["structured_news"]["pending_host_links"]
    assert len(pending) == 1
    assert pending[0]["source_published_at"] == "2026-10-05T05:03:00+00:00"
    assert not any(item["reason"] == "invalid_publication_time" for item in doc["structured_news"]["excluded"])


def test_verified_union_article_container_projects_visible_body(tmp_path):
    url = "https://www.fc-union-berlin.de/de/meldungen/auswaerts-in-augsburg-union-will-punkte-nachlegen-id"
    html = ('<main><article><header><h3>Auswärts in Augsburg:</h3><h2>Union will Punkte nachlegen</h2>'
            '<p>Die Profimannschaft reist zum FC Augsburg.</p><time datetime="2026-10-05T07:00:00.000Z">5. Oktober 2026</time>'
            '</header><p>Die Partie findet am Sonnabend statt.</p></article>'
            '<article><h2>Related card</h2><p>Do not borrow this.</p></article></main>')
    t = tools(tmp_path, projection_version=13)
    doc, _, _ = project(t, html, endpoint=url)
    assert doc["status"] == "OK" and doc["structured_news"]["role"] == "article"
    assert "FC Augsburg" in doc["text"] and "Related card" not in doc["text"]
    assert doc["source_published_at"] == "2026-10-05T07:00:00+00:00"


def test_verified_augsburg_article_uses_jsonld_date_and_registered_body(tmp_path):
    url = "https://www.fcaugsburg.de/article/matchday-heimspiel-gegen-union-berlin-23596"
    meta = {"@type": "NewsArticle", "headline": "Matchday", "datePublished": "2026-10-05T10:00:00+02:00"}
    html = ('<article class="b-article"><h1 class="article-title">Matchday</h1>'
            '<div class="b-richtext"><p>Heimspiel gegen Union Berlin.</p><p>Die Mannschaft ist vorbereitet.</p></div></article>'
            '<script type="application/ld+json">'+json.dumps(meta)+'</script>')
    t = tools(tmp_path, projection_version=13)
    doc, _, _ = project(t, html, endpoint=url)
    assert doc["status"] == "OK" and doc["structured_news"]["role"] == "article"
    assert "Heimspiel gegen Union Berlin." in doc["text"]
    assert doc["source_published_at"] == "2026-10-05T08:00:00+00:00"


def test_jsonld_live_updates_exclude_future_and_keep_per_segment_time_offsets(tmp_path):
    t = tools(tmp_path)
    current = node(**{"@type": "BlogPosting"})
    future_publication = node("Future injury unavailable", datePublished="2026-10-06T13:00:00Z", dateModified="2026-10-06T13:00:00Z")
    future_modification = node("Future modified lineup", dateModified="2026-10-06T14:00:00Z")
    unknown = node("Undated live item", datePublished=None)
    live = {"@type": "LiveBlogPosting", "liveBlogUpdate": [current, future_publication, future_modification, unknown],
            "mainEntity": {"@type": "SportsEvent", "name": "Club vs Opponent", "eventStatus": "EventScheduled",
                           "startDate": "2026-10-06T18:45:00Z"}}
    doc, observation, prepared = project(t, markup(live, "Future modified lineup also visible"))
    assert doc["kind"] == "page" and doc["structured_news"]["role"] == "liveblog"
    assert "Club forward is available & ready." == doc["text"]
    assert "Future" not in doc["text"] and "Undated" not in doc["text"]
    assert doc["structured_news"]["excluded_count"] == 3
    assert doc["structured_news"]["identity_metadata"][0]["eventStatus"] == "EventScheduled"
    segment = doc["structured_news"]["segments"][0]
    assert doc["text"][segment["start"]:segment["end"]] == segment["text"]
    assert segment["source_published_at"] == "2026-10-06T10:00:00+00:00"
    assert t.project(observation, prepared) == doc


def test_excluded_future_article_cannot_reenter_through_visible_html(tmp_path):
    t = tools(tmp_path)
    doc, _, _ = project(t, markup(node("Future result 3-0", datePublished="2026-10-07T01:00:00Z"), "Future result 3-0"))
    assert doc["status"] == "NO_DATA" and doc["text"] == ""
    assert doc["structured_news"]["role"] == "unknown"


def test_event_identity_never_becomes_article_or_invented_result(tmp_path):
    t = tools(tmp_path)
    doc, _, _ = project(t, markup({"@type": "SportsEvent", "name": "Club vs Opponent",
        "eventStatus": "EventCompleted", "startDate": "2026-10-05T18:45:00Z"}, "Fixture page"))
    assert doc["structured_news"]["role"] == "unknown"
    assert doc["structured_news"]["segments"] == []
    assert doc["structured_news"]["identity_metadata"][0]["eventStatus"] == "EventCompleted"
    assert "EventCompleted" not in doc["text"]


def test_semantic_article_excludes_related_cards_and_does_not_invent_date_timezone(tmp_path):
    t = tools(tmp_path)
    paragraph = "Club are preparing to face Opponent and the coach discussed the next match. "*4
    body = '<div data-testid="publish-date"><span>05 Oct 26</span></div><article><p>'+paragraph+'</p><p>The injured player will not start.</p></article><article><h3>Youth old match</h3></article>'
    doc, _, _ = project(t, body)
    assert doc["structured_news"]["role"] == "article"
    assert "Youth" not in doc["text"]
    segment = doc["structured_news"]["segments"][0]
    assert segment["source_publication_date"] == "2026-10-05"
    assert segment["publication_raw"] == "05 Oct 26"
    assert doc["source_published_at"] is None


def test_explicit_directory_keeps_links_but_is_neutral_even_if_cards_have_article_markup(tmp_path):
    t = tools(tmp_path, index_urls=[INDEX])
    doc, _, _ = project(t, '<a href="'+ARTICLE+'">Club match preview</a>'+markup(node()), endpoint=INDEX)
    assert doc["kind"] == "index" and doc["structured_news"]["role"] == "index"
    assert doc["structured_news"]["segments"] == []
    assert doc["links"][0]["url"] == ARTICLE


def test_malformed_script_is_not_executed_and_wrong_endpoint_is_rejected(tmp_path):
    t = tools(tmp_path)
    doc, _, _ = project(t, '<script type="application/ld+json">__import__("os").system("echo unsafe")</script><p>Nothing else</p>')
    assert doc["structured_news"]["jsonld_parse_errors"] == 1
    assert doc["structured_news"]["segments"] == []
    with pytest.raises(ValueError, match="configured provider"):
        t.execute({"tool": "search_web", "endpoint": "https://evil.example", "params": {"q": "x", "format": "rss"},
                   "source": "open_search"}, run_id="test", request_id="1")
