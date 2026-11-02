"""The observed Contentful/Flight template: only the bound body reaches f/g."""
from datetime import datetime, timezone
from html import escape
import gzip
import json
from pathlib import Path
import socket

import pytest

from football_agent.news_discovery import NewsDiscoveryTools
from football_spider.storage.raw_archive import RawArchive


ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "var/submission/p1-single-match/source-probe"
URL = "https://source.example/en/news/an-observed-article"
CAPTURED = datetime(2026, 10, 7, 19, tzinfo=timezone.utc)
TITLE = "An observed source heading"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("Body projection must use saved bytes without a network request")
    for obj, name in ((socket, "getaddrinfo"), (socket, "create_connection"),
                      (socket.socket, "connect"), (socket.socket, "connect_ex")):
        monkeypatch.setattr(obj, name, blocked)


def sample(*, published="2026-10-06T19:00:00.000Z", title=TITLE, extra_record=None):
    record = {"format": "News", "title": title, "summary": "A source summary.",
              "publishDate": {"date": published, "format": "DD MMM YY", "locale": "en"}}
    payload = "10:" + json.dumps([record] + ([extra_record] if extra_record else [])) + "\n"
    flight = "self.__next_f.push(" + json.dumps([1, payload]) + ")"
    return ('<html lang="en"><head><link rel="canonical" href="' + URL + '">'
        '<meta property="og:title" content="' + escape(TITLE) + ' - Source">'
        '<meta property="og:description" content="A source summary."></head><body>'
        '<div data-component="Article"><div><h3 data-testid="heading">' + TITLE + '</h3>'
        '<div data-testid="publish-date">06 Oct 26</div></div>'
        '<div data-component="ContentfulRichTextRender"><article>'
        '<p>First source paragraph.</p><p>Second source paragraph with details.</p>'
        '<aside><p>UNRELATED_HIDDEN_BODY</p></aside></article></div></div>'
        '<div data-component="RelatedContent"><article><p>OTHER_MATCH_SENTINEL</p></article></div>'
        '<script>' + flight + '</script></body></html>')


def project(tmp_path, raw, *, url=URL, captured=CAPTURED, index_urls=(), projection_version=6):
    archive = RawArchive(tmp_path / "archive")
    observation = archive.save("open_page", url, {}, raw.encode("utf-8") if isinstance(raw, str) else raw,
                               status=200, observed_at=captured)
    hosts = ["source.example", "www.leedsunited.com"]
    tools = NewsDiscoveryTools(archive, hosts, projection_version=projection_version, index_urls=index_urls,
                              clock=lambda: CAPTURED)
    prepared = {"tool": "read_url", "source": "open_page", "endpoint": url, "params": {}}
    return tools.project(observation, prepared)


def gallery_sample(*, records=None):
    record = {"format": "Gallery", "title": TITLE, "summary": "A source summary.",
              "publishDate": {"date": "2026-10-06T19:00:00Z"}}
    payload = "10:" + json.dumps([record] if records is None else records) + "\n"
    flight = "self.__next_f.push(" + json.dumps([1, payload]) + ")"
    return ('<html lang="en"><head><link rel="canonical" href="' + URL + '">'
        '<meta property="og:title" content="' + escape(TITLE) + ' - Source">'
        '<meta property="og:description" content="A source summary."></head><body>'
        '<div data-component="Article"><div><h3 data-testid="heading">' + TITLE + '</h3>'
        '<div data-testid="publish-date">06 Oct 26</div><p data-testid="paragraph">A source summary.</p></div>'
        '<div data-component="GalleryImages"><p>IMAGE_CAPTION_NOT_BODY</p></div>'
        '<div data-component="ContentfulRichTextRender"><article></article></div></div>'
        '<script>' + flight + '</script></body></html>')


def test_gallery_introduction_is_opt_in_and_never_promotes_image_captions(tmp_path):
    raw = gallery_sample()
    old = project(tmp_path / "old", raw)
    assert old["structured_news"]["role"] == "unknown"
    assert not old["structured_news"]["segments"]
    new = project(tmp_path / "new", raw, projection_version=7)
    assert new["text"] == "A source summary."
    assert new["source_published_at"] == "2026-10-06T19:00:00+00:00"
    news = new["structured_news"]
    assert news["source_format"] == "Gallery"
    assert news["source_text_scope"] == "gallery_introduction_only_captions_excluded"
    assert news["segments"][0]["paragraph_kind"] == "gallery_introduction"
    assert "IMAGE_CAPTION_NOT_BODY" not in new["text"]


@pytest.mark.parametrize("change", ["missing_intro", "hidden_intro", "duplicate_intro", "no_gallery",
    "wrong_summary", "wrong_title", "duplicate_record", "wrong_day", "future", "missing_current_record"])
def test_gallery_never_borrows_recommended_metadata_or_fabricates_body(tmp_path, change):
    record = {"format": "Gallery", "title": TITLE, "summary": "A source summary.",
              "publishDate": {"date": "2026-10-06T19:00:00Z"}}
    other = {**record, "title": "Another recommended article",
             "publishDate": {"date": "2026-10-07T10:00:00Z"}}
    raw = gallery_sample(records=[other, record])
    if change == "missing_intro": raw = raw.replace('<p data-testid="paragraph">A source summary.</p>', '')
    elif change == "hidden_intro": raw = raw.replace('data-testid="paragraph"', 'data-testid="paragraph" hidden')
    elif change == "duplicate_intro": raw = raw.replace('<p data-testid="paragraph">', '<p>A source summary.</p><p data-testid="paragraph">')
    elif change == "no_gallery": raw = raw.replace('data-component="GalleryImages"', 'data-component="Other"')
    elif change == "wrong_summary": raw = gallery_sample(records=[other, {**record, "summary": "Unbound summary"}])
    elif change == "wrong_title": raw = gallery_sample(records=[other, {**record, "title": "Unbound title"}])
    elif change == "duplicate_record": raw = gallery_sample(records=[record, record])
    elif change == "wrong_day": raw = raw.replace('06 Oct 26', '05 Oct 26')
    elif change == "future":
        raw = gallery_sample(records=[{**record, "publishDate": {"date": "2026-10-08T19:00:00Z"}}]).replace('06 Oct 26', '08 Oct 26')
    elif change == "missing_current_record": raw = gallery_sample(records=[other])
    doc = project(tmp_path, raw, projection_version=7)
    assert doc["structured_news"]["role"] == "unknown"
    assert not doc["structured_news"]["segments"]


def test_gallery_binds_matching_current_record_not_first_or_last_record(tmp_path):
    record = {"format": "Gallery", "title": TITLE, "summary": "A source summary.",
              "publishDate": {"date": "2026-10-06T19:00:00Z"}}
    other = {**record, "title": "Another recommended gallery",
             "publishDate": {"date": "2026-10-07T10:00:00Z"}}
    doc = project(tmp_path, gallery_sample(records=[other, record, {**other, "title": "Third gallery"}]),
                  projection_version=7)
    assert doc["structured_news"]["role"] == "article"
    assert doc["source_published_at"] == "2026-10-06T19:00:00+00:00"


def test_actual_live02_gallery_v7_only_consumes_real_header_introduction(tmp_path):
    folder = ROOT / "var/submission/p3-single-match/live-02/source-archive/raw/open_page/2026-10-07"
    candidates = list(folder.glob("*b4c3d99b*.body.gz"))
    if not candidates:
        pytest.skip("The separately captured live02 gallery archive is absent")
    raw = gzip.decompress(candidates[0].read_bytes())
    run = json.loads((folder.parents[3] / "run.json").read_text(encoding="utf-8"))
    saved = run["documents"][-1]
    old = project(tmp_path / "old", raw, url=saved["url"], captured=datetime.fromisoformat(saved["observed_at"]))
    assert old["structured_news"] == saved["structured_news"]
    doc = project(tmp_path / "new", raw, url=saved["url"], captured=datetime.fromisoformat(saved["observed_at"]),
                  projection_version=7)
    assert doc["source_published_at"] == "2026-09-25T14:00:00+00:00"
    assert len(doc["structured_news"]["segments"]) == 1
    assert doc["text"] == ("Take a look through our gallery, featuring photos of Leeds United players "
                           "representing their countries throughout the September/October international window.")
    assert "Anton Stach" not in doc["text"]
    assert "Related Content" not in doc["text"]


def test_v7_configuration_roundtrip_preserves_v6_contract(tmp_path):
    archive = RawArchive(tmp_path)
    old = NewsDiscoveryTools(archive, ["source.example"], projection_version=6)
    new = NewsDiscoveryTools(archive, ["source.example"], projection_version=7)
    assert "gallery_introduction_adapter" not in old.configuration()["news_discovery"]
    assert new.configuration()["news_discovery"]["gallery_text_scope"].startswith("visible_header_introduction_only")
    assert NewsDiscoveryTools.from_configuration(archive, old.configuration()).configuration() == old.configuration()
    assert NewsDiscoveryTools.from_configuration(archive, new.configuration()).configuration() == new.configuration()


def test_body_publication_and_title_are_bound_without_related_card_text(tmp_path):
    doc = project(tmp_path, sample())
    assert doc["kind"] == doc["structured_news"]["role"] == "article"
    assert doc["title"] == TITLE
    assert doc["source_published_at"] == "2026-10-06T19:00:00+00:00"
    assert doc["text"] == "First source paragraph.\nSecond source paragraph with details."
    for segment in doc["structured_news"]["segments"]:
        assert doc["text"][segment["start"]:segment["end"]] == segment["text"]
        assert segment["body_binding"]["metadata_json_pointer"].startswith("script[")
        assert segment["source_url"] == URL


@pytest.mark.parametrize("change", ["future", "no_timezone", "visible_day", "title", "canonical",
    "duplicate_article", "duplicate_record", "hidden_body", "untrusted_script", "jsonld"])
def test_unbound_ambiguous_or_future_source_remains_unknown(tmp_path, change):
    raw = sample()
    if change == "future":
        raw = sample(published="2026-10-08T19:00:00Z").replace("06 Oct 26", "08 Oct 26")
    elif change == "no_timezone": raw = sample(published="2026-10-06T19:00:00")
    elif change == "visible_day": raw = raw.replace("06 Oct 26", "05 Oct 26")
    elif change == "title": raw = sample(title="A different page")
    elif change == "canonical": raw = raw.replace('href="'+URL+'"', 'href="'+URL+'-other"')
    elif change == "duplicate_article": raw = raw.replace("</body>", '<div data-component="Article"></div></body>')
    elif change == "duplicate_record":
        raw = sample(extra_record={"format": "News", "title": TITLE, "summary": "A source summary.",
                                   "publishDate": {"date": "2026-10-06T19:00:00Z"}})
    elif change == "hidden_body": raw = raw.replace('<article>', '<article hidden>', 1)
    elif change == "untrusted_script": raw = raw.replace("self.__next_f.push(", "some_arbitrary_function(")
    elif change == "jsonld": raw = raw.replace("</head>", '<script type="application/ld+json">{bad}</script></head>')
    doc = project(tmp_path, raw)
    assert doc["structured_news"]["role"] == "unknown"
    assert doc["structured_news"]["segments"] == []


@pytest.mark.parametrize("name,paragraphs,published", [
    ("leeds-article-one", 8, "2026-10-06T19:00:00+00:00"),
    ("leeds-article-two", 7, "2026-10-06T20:00:00+00:00")])
def test_actual_saved_leeds_responses_use_the_executor_projection(tmp_path, name, paragraphs, published):
    folder = PROBE / name
    if not (folder / "body.html").is_file():
        pytest.skip("The separately captured P1 source archive is absent")
    source = json.loads((folder / "observation.json").read_text(encoding="utf-8"))
    doc = project(tmp_path, (folder / "body.html").read_bytes(), url=source["url"],
                  captured=datetime.fromisoformat(source["captured_at"]))
    assert doc["kind"] == "article" and doc["status"] == "OK"
    assert len(doc["structured_news"]["segments"]) == paragraphs
    assert doc["source_published_at"] == published
    assert "Related Content" not in doc["text"]
    assert "Leeds United" in doc["text"]


def test_actual_saved_directory_exposes_articles_without_typing_cards_as_body(tmp_path):
    folder = PROBE / "leeds-directory"
    if not (folder / "body.html").is_file():
        pytest.skip("The separately captured P1 source archive is absent")
    source = json.loads((folder / "observation.json").read_text(encoding="utf-8"))
    doc = project(tmp_path, (folder / "body.html").read_bytes(), url=source["url"],
        captured=datetime.fromisoformat(source["captured_at"]), index_urls=[source["url"]])
    assert doc["kind"] == "index" and doc["structured_news"]["segments"] == []
    urls = {link["url"] for link in doc["links"]}
    for name in ("leeds-article-one", "leeds-article-two"):
        expected = json.loads((PROBE / name / "observation.json").read_text(encoding="utf-8"))["url"]
        assert expected in urls


def test_actual_historical_preview_uses_same_parser_without_recommended_articles(tmp_path):
    raw = PROBE.parent / "corpus-probe/raw/8ef559b2b640e5c83225.html"
    if not raw.is_file():
        pytest.skip("The separately captured historical preview archive is absent")
    doc = project(tmp_path, raw.read_bytes(),
        url="https://www.leedsunited.com/en/news/preview-leeds-united-vs-arsenal-2026",
        captured=datetime.fromisoformat("2026-10-07T16:54:46.193034+00:00"))
    assert doc["title"] == "Preview: Leeds United vs Arsenal"
    assert doc["source_published_at"] == "2026-01-30T13:00:00+00:00"
    assert len(doc["structured_news"]["segments"]) == 46
    assert doc["text"].startswith("It is back to West Yorkshire for Leeds United")
    assert "Related Content" not in doc["text"]
