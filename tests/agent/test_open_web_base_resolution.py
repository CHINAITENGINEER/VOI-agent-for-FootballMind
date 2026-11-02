from datetime import datetime, timezone

from football_agent.inquiry_tools import OpenWebTools
from football_spider.storage.raw_archive import RawArchive


def project(tmp_path, body, *, version=3, params=None):
    archive = RawArchive(tmp_path)
    now = datetime(2026, 10, 6, tzinfo=timezone.utc)
    endpoint = "https://understat.com/league/EPL/2026"
    params = {} if params is None else params
    observation = archive.save("open_page", endpoint, params, body.encode(), status=200, observed_at=now)
    tools = OpenWebTools(archive, ["understat.com"], clock=lambda: now, extraction_version=version)
    return tools.project(observation, {"endpoint": endpoint, "params": params, "tool": "read_url"})


def test_first_base_href_resolves_links_and_legacy_projection_is_unchanged(tmp_path):
    body = '<html><base href="https://understat.com/"><base href="https://elsewhere.example/"><a href="team/Arsenal/2026">Arsenal</a></html>'
    assert project(tmp_path/"new", body)["links"] == [{"url": "https://understat.com/team/Arsenal/2026", "title": "Arsenal"}]
    assert project(tmp_path/"old", body, version=2)["links"][0]["url"] == "https://understat.com/league/EPL/team/Arsenal/2026"


def test_disallowed_base_does_not_turn_relative_links_into_allowed_source_urls(tmp_path):
    body = '<base href="https://other.example/"><a href="team/Arsenal">Arsenal</a><a href="https://understat.com/league/EPL">EPL</a>'
    assert project(tmp_path, body)["links"] == [{"url": "https://understat.com/league/EPL", "title": "EPL"}]


def test_relative_base_and_no_base_are_resolved_by_document_location(tmp_path):
    relative = '<base href="../../"><a href="team/Leeds/2026">Leeds</a>'
    plain = '<a href="/team/Leeds/2026">Leeds</a>'
    expected = "https://understat.com/team/Leeds/2026"
    assert project(tmp_path/"relative", relative)["links"][0]["url"] == expected
    assert project(tmp_path/"plain", plain)["links"][0]["url"] == expected


def test_v3_canonical_query_spelling_survives_json_key_reordering(tmp_path):
    a = project(tmp_path/"a", "<p>text</p>", params={"q": "query", "format": "rss"})
    b = project(tmp_path/"b", "<p>text</p>", params={"format": "rss", "q": "query"})
    assert a["url"] == b["url"]
