"""Optional source-specific parsing never rewrites the actor's old projection."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path

import pytest

from football_agent.news_discovery import NewsDiscoveryTools, VERSION
from football_research.data_adapters.open_text_episodes import document_role
from football_spider.models.contracts import Observation
from football_spider.storage.raw_archive import RawArchive


CAPTURED = datetime(2026, 10, 6, 15, 28, tzinfo=timezone.utc)
PARSED = datetime(2026, 10, 7, 8, tzinfo=timezone.utc)
URL = "https://jalgpall.ee/koondis/1/uudised/actual-source-news-n26469"


def html(*, day="06", month="10.2026", time="12:05", lang="et", headline="Eesti – Island uudised",
         body="<div class='intro'><p>Actual introduction.</p></div><p>Actual paragraph &amp; details.</p>"):
    return (f'<html lang="{lang}"><head><meta name="medium" content="news"></head><body>'
        '<header>Navigation noise</header><div class="news-detail"><div class="head">'
        f'<div class="date"><p class="day">{day}</p><p class="m-y">{month}</p><p class="m-y">{time}</p></div>'
        f'<h1>{headline}</h1></div><div class="body def">{body}</div>'
        '<div class="social-share"><p>Share noise</p></div></div><aside>Related card noise</aside></body></html>')


def project(tmp_path, content=None, *, projection_version=2, url=URL, status=200, parsed=PARSED, captured=CAPTURED, **options):
    archive = RawArchive(tmp_path/"archive")
    obs = archive.save("open_page", url, {}, (html() if content is None else content).encode(),
                       status=status, observed_at=captured)
    prepared = {"tool": "read_url", "source": "open_page", "endpoint": url, "params": {}}
    tools = NewsDiscoveryTools(archive, ["jalgpall.ee", "other.example"], clock=lambda: parsed,
                               projection_version=projection_version, **options)
    return tools.project(obs, prepared), tools, obs, prepared


def test_explicit_v2_configuration_reconstructs_and_default_v1_has_no_new_keys(tmp_path):
    _, tools, _, _ = project(tmp_path)
    config = tools.configuration()
    assert config["news_discovery"]["version"] == "structured-news-discovery-v2"
    assert config["news_discovery"]["projection_version"] == 2
    assert NewsDiscoveryTools.from_configuration(tools.archive, config).configuration() == config
    old = NewsDiscoveryTools(tools.archive, tools.allowed_hosts).configuration()
    assert old["news_discovery"]["version"] == VERSION
    assert "projection_version" not in old["news_discovery"]
    assert "html_source_adapter" not in old["news_discovery"]
    assert NewsDiscoveryTools.from_configuration(tools.archive, old).configuration() == old
    for field, value in (("projection_version", 1), ("html_source_adapter", "changed")):
        tampered = deepcopy(config)
        tampered["news_discovery"][field] = value
        with pytest.raises(ValueError, match="exactly"):
            NewsDiscoveryTools.from_configuration(tools.archive, tampered)


def test_real_paragraphs_and_offsets_exclude_layout_and_keep_local_clock_unknown(tmp_path):
    doc, tools, obs, prepared = project(tmp_path)
    assert doc["kind"] == "article" and document_role(doc) == "article_candidate_not_semantically_verified"
    assert doc["text"] == "Eesti – Island uudised\nActual introduction.\nActual paragraph & details."
    metadata = doc["structured_news"]
    assert metadata["source_language"] == "et" and metadata["language_source"] == "html@lang"
    assert len(metadata["segments"]) == 3
    for segment in metadata["segments"]:
        assert doc["text"][segment["start"]:segment["end"]] == segment["text"]
        assert segment["publication_raw"] == "06 10.2026 12:05"
        assert segment["source_publication_date"] == "2026-10-06"
        assert segment["source_publication_local_time"] == "12:05"
        assert segment["source_published_at"] is None
        assert segment["publication_timezone"] == "unknown"
    assert doc["source_published_at"] is None
    assert doc["observed_at"] == CAPTURED.isoformat()
    assert doc["available_at"] == PARSED.isoformat()
    assert doc["projection_provenance"] == {"captured_at": CAPTURED.isoformat(), "projected_at": PARSED.isoformat(),
        "scope": "actual_time_versioned_projection_of_archived_response_bytes", "network_requests_during_projection": 0}
    assert tools.project(obs, prepared) == doc


@pytest.mark.parametrize("case,reason", [
    ("wrong_host", None), ("directory", "not_registered_jalgpall_news_permalink"),
    ("missing_medium", "unique_source_news_marker_and_detail_required"),
    ("different_medium", "unique_source_news_marker_and_detail_required"),
    ("duplicate_detail", "unique_source_news_marker_and_detail_required"),
    ("duplicate_body", "unique_article_head_and_body_required"),
    ("no_title", "article_title_and_publication_block_required"),
    ("no_date", "article_title_and_publication_block_required"),
    ("empty_body", "article_body_has_no_nonempty_paragraphs"),
    ("invalid_date", "invalid_local_publication_date"),
])
def test_source_structure_counterexamples_are_not_articles(tmp_path, case, reason):
    content, url = html(), URL
    if case == "wrong_host": url = URL.replace("jalgpall.ee", "other.example")
    elif case == "directory": url = "https://jalgpall.ee/koondis/1/uudised"
    elif case == "missing_medium": content = content.replace('<meta name="medium" content="news">', "")
    elif case == "different_medium": content = content.replace('content="news"', 'content="directory"')
    elif case == "duplicate_detail": content += '<div class="news-detail"></div>'
    elif case == "duplicate_body": content = content.replace('<div class="body def">', '<div class="body def"></div><div class="body def">')
    elif case == "no_title": content = content.replace("h1", "h2")
    elif case == "no_date": content = content.replace('class="date"', 'class="unrelated-date"')
    elif case == "empty_body": content = html(body="<div>Shell without a source paragraph.</div>")
    elif case == "invalid_date": content = html(day="31", month="02.2026")
    doc, _, _, _ = project(tmp_path, content, url=url)
    assert doc["kind"] != "article" and doc["structured_news"]["role"] == "unknown"
    assert doc["structured_news"]["segments"] == []
    if reason:
        assert doc["structured_news"]["excluded"][0]["reason"] == reason


def test_certain_future_date_is_excluded_against_capture_not_new_parse_date(tmp_path):
    # Parsed on Oct 10 cannot make an Oct 8 source declaration eligible for
    # content bytes captured on Oct 6. No particular timezone is assigned.
    doc, _, _, _ = project(tmp_path, html(day="08"), parsed=PARSED+timedelta(days=3))
    assert doc["status"] == "NO_DATA" and doc["text"] == ""
    assert doc["structured_news"]["excluded"][0]["reason"] == "local_publication_certainly_after_actual_capture"
    assert doc["structured_news"]["publication"]["source_published_at"] is None


def test_ambiguous_next_calendar_day_is_not_assigned_a_timezone(tmp_path):
    doc, _, _, _ = project(tmp_path, html(day="07", time="00:05"))
    assert doc["kind"] == "article"
    assert doc["source_published_at"] is None
    assert doc["structured_news"]["publication"]["source_publication_date"] == "2026-10-07"


def test_failed_and_explicit_index_returns_stay_non_articles(tmp_path):
    failure, _, _, _ = project(tmp_path/"failure", status=503)
    assert failure["status"] == "FAILED" and failure["kind"] == "page"
    index, _, _, _ = project(tmp_path/"index", index_urls=[URL])
    assert index["kind"] == "index" and index["structured_news"]["segments"] == []


def test_scripts_images_related_modal_content_are_not_article_paragraphs(tmp_path):
    body = ('<p>Actual article.</p><script>unsafe()</script><style>.p{}</style>'
        '<div class="modal"><p>Overlay unrelated</p></div><img alt="Image not body">'
        '<aside><p>Related story</p></aside><div class="social-share"><p>Share</p></div>')
    doc, _, _, _ = project(tmp_path, html(body=body))
    assert doc["text"] == "Eesti – Island uudised\nActual article."


def test_article_requires_a_body_segment_within_projection_budget(tmp_path):
    doc, _, _, _ = project(tmp_path, html(headline="Long heading "*2000))
    assert doc["status"] == "NO_DATA" and doc["structured_news"]["segments"] == []


@pytest.mark.parametrize("role", ["article", "liveblog", "semantic_html"])
def test_v2_generic_article_contract_remains_accepted_without_changing_old_role_adapter(tmp_path, role):
    payload = {"@type": "NewsArticle", "articleBody": "An acquired source article.", "datePublished": "2026-10-06T10:00:00Z"}
    if role == "liveblog":
        payload = {"@type": "LiveBlogPosting", "liveBlogUpdate": [payload]}
    body = '<script type="application/ld+json">'+json.dumps(payload)+'</script>'
    if role == "semantic_html":
        body = '<article><p>'+('A sufficiently substantial acquired source paragraph. '*8)+'</p><p>Other source paragraph.</p></article>'
    content = '<html lang="en"><body>'+body+'</body></html>'
    doc, _, _, _ = project(tmp_path, content, url="https://other.example/match/42")
    assert doc["kind"] == "article"
    assert doc["structured_news"]["version"] == "structured-news-discovery-v2"
    assert doc["structured_news"]["role"] == ("liveblog" if role == "liveblog" else "article")
    assert doc["structured_news"]["source_language"] == "en"
    assert document_role(doc) == "article_candidate_not_semantically_verified"


def test_missing_html_language_does_not_fall_back_to_estonian(tmp_path):
    content = html().replace(' lang="et"', "")
    doc, _, _, _ = project(tmp_path, content)
    assert doc["kind"] == "article" and doc["structured_news"]["source_language"] is None


def test_cannot_backdate_new_projection_before_raw_capture(tmp_path):
    with pytest.raises(ValueError, match="precede the archived observation"):
        project(tmp_path, parsed=CAPTURED-timedelta(seconds=1))


def test_two_archived_jalgpall_bodies_reproject_without_changing_original_panel(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    path = root/"var/parallel/agent/round184/panel-repair/panel-2981554/panel.json"
    if not path.exists():
        pytest.skip("Actual Round184 archive is not included in this checkout")
    original_bytes = path.read_bytes()
    panel = json.loads(original_bytes)
    archive = RawArchive(Path(panel["archive_folder"]))
    old_config = panel["design"]["transport"]
    new = NewsDiscoveryTools(archive, old_config["allowed_hosts"],
        provider=old_config["search_provider"], default_route=old_config["source_route_policy"]["default_route"],
        route_overrides=old_config["source_route_policy"]["route_overrides"],
        index_urls=old_config["news_discovery"]["index_urls"], projection_version=2, clock=lambda: PARSED)
    monkeypatch.setattr(NewsDiscoveryTools, "execute", lambda *a, **k: pytest.fail("Archive projection made a network request"))
    paragraphs, dates = [], []
    for capture in panel["captures"]:
        old_doc, obs = capture["result"]["document"], Observation.model_validate(capture["result"]["observation"])
        payload_hash = sha256(archive.read_body(obs)).hexdigest()
        legacy = NewsDiscoveryTools.from_configuration(archive, old_config,
            clock=lambda: datetime.fromisoformat(old_doc["available_at"]))
        assert legacy.project(obs, capture["prepared"]) == old_doc
        derived = new.project(obs, capture["prepared"])
        assert derived["kind"] == "article" and derived["raw_sha256"] == old_doc["raw_sha256"] == payload_hash
        assert derived["observed_at"] == old_doc["observed_at"]
        assert derived["available_at"] == PARSED.isoformat() and derived["available_at"] != old_doc["available_at"]
        assert derived["structured_news"]["source_language"] == "et"
        segments = derived["structured_news"]["segments"]
        assert all(derived["text"][s["start"]:s["end"]] == s["text"] for s in segments)
        assert all(s["source_publication_date"] == "2026-10-06" and s["source_published_at"] is None for s in segments)
        dates.append(segments[0]["source_publication_local_time"])
        paragraphs.append(len([s for s in segments if s["content_role"] == "paragraph"]))
        assert sha256(archive.read_body(obs)).hexdigest() == payload_hash
    assert dates == ["12:05", "15:36"]
    assert paragraphs == [11, 19]
    assert path.read_bytes() == original_bytes
