"""Same-URL visible anchor labels must survive image-first URL deduplication."""
from datetime import datetime, timezone

from demo.live_sources import LiveNewsTools, article_link_reason
from football_spider.storage.raw_archive import RawArchive


def test_image_anchor_then_text_anchor_uses_observed_label(tmp_path):
    url="https://spfl.co.uk/news"
    html='''<html><body><main>
    <a href="/news/weekend-preview"><img alt="Unverified picture caption"></a>
    <a href="/news/weekend-preview">The official weekend match preview</a>
    </main></body></html>'''
    archive=RawArchive(tmp_path);at=datetime(2026,10,25,9,0,tzinfo=timezone.utc)
    observation=archive.save("open_page",url,{},html.encode(),status=200,observed_at=at)
    tools=LiveNewsTools(archive,["spfl.co.uk"],index_urls=[url],projection_version=13,clock=lambda:at)
    doc=tools.project(observation,{"tool":"read_url","endpoint":url,"params":{},"source":"open_page"})
    assert len(doc["links"])==1
    link=doc["links"][0]
    assert link["title"]=="The official weekend match preview"
    assert link["title_source"]=="same_url_observed_visible_anchor"
    assert link["title_source_anchor_index"]==1
    assert article_link_reason(link["url"],link["title"],[url],set()) is None


def test_image_alt_or_hidden_duplicate_never_becomes_title(tmp_path):
    url="https://spfl.co.uk/news"
    html='''<html><body><main>
    <a href="/news/unseen-title"><img alt="Do not invent a headline from an image"></a>
    <a href="/news/unseen-title" hidden>Do not use hidden duplicated headline</a>
    </main></body></html>'''
    archive=RawArchive(tmp_path);at=datetime(2026,10,25,9,0,tzinfo=timezone.utc)
    observation=archive.save("open_page",url,{},html.encode(),status=200,observed_at=at)
    tools=LiveNewsTools(archive,["spfl.co.uk"],index_urls=[url],projection_version=13,clock=lambda:at)
    doc=tools.project(observation,{"tool":"read_url","endpoint":url,"params":{},"source":"open_page"})
    assert doc["links"][0]["title"]==""
    assert not doc["links"][0].get("title_source")
