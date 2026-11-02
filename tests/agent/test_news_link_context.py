"""Pre-read labels must remain grounded in the captured parent page."""
from copy import deepcopy
from datetime import datetime, timezone
import json

import pytest

from football_agent.news_discovery import NewsDiscoveryTools
from football_agent.match_evidence import build_match_evidence
from football_spider.storage.raw_archive import RawArchive


NOW = datetime(2026, 10, 6, 20, tzinfo=timezone.utc)
INDEX = "https://club.example/"
ARTICLE = "https://club.example/news/preview"


def project(tmp_path, body, version=4):
    tools = NewsDiscoveryTools(RawArchive(tmp_path / "raw"), ["club.example"],
        clock=lambda: NOW, index_urls=[INDEX], projection_version=version)
    prepared = {"tool": "read_url", "endpoint": INDEX, "params": {}, "source": "open_page"}
    observation = tools.archive.save("open_page", INDEX, {}, body.encode(), status=200, observed_at=NOW)
    return tools, tools.project(observation, prepared), observation, prepared


def test_date_first_anchor_preserves_later_headline_without_changing_menu(tmp_path):
    html = '<main><a href="/news/preview">October 6</a><a href="/news/preview">Team announces squad</a></main>'
    tools, new, observation, prepared = project(tmp_path, html)
    old = NewsDiscoveryTools(tools.archive, ["club.example"], clock=lambda: NOW,
        index_urls=[INDEX], projection_version=3).project(observation, prepared)
    assert [{k: link[k] for k in ("url", "title")} for link in new["links"]] == old["links"]
    assert new["links"][0]["title"] == "October 6"
    assert new["links"][0]["observed_anchors"] == [
        {"anchor_index": 0, "text": "October 6", "text_truncated": False},
        {"anchor_index": 1, "text": "Team announces squad", "text_truncated": False}]
    assert new["text"] == old["text"]
    assert "observed_anchors" not in old["links"][0]
    assert new["observed_at"] == old["observed_at"] == NOW.isoformat()


def test_base_url_hidden_template_and_external_labels_cannot_supply_context(tmp_path):
    html = '''<base href="https://club.example/news/"><main>
      <a href="preview">Observed headline</a>
      <div hidden><a href="preview">Hidden lineup</a></div>
      <template><a href="preview">Template result</a></template>
      <a href="preview" style="display: none">Hidden result</a>
      <div style="visibility:hidden"><a href="preview">Hidden lineup</a></div>
      <a href="preview">Observed headline<span hidden> Secret result</span></a>
      <a href="https://other.example/news/preview">Other host</a></main>'''
    _, document, _, _ = project(tmp_path, html)
    assert len(document["links"]) == 1
    assert document["links"][0]["url"] == ARTICLE
    assert [a["text"] for a in document["links"][0]["observed_anchors"]] == ["Observed headline"]


def test_long_and_repeated_labels_are_bounded_and_reported(tmp_path):
    labels = ["x" * 600, "same", "same", *[f"headline {i}" for i in range(9)]]
    html = "<main>" + "".join(f'<a href="/news/preview">{s}</a>' for s in labels) + "</main>"
    _, document, _, _ = project(tmp_path, html)
    link = document["links"][0]
    assert len(link["observed_anchors"]) == 8
    assert len(link["observed_anchors"][0]["text"]) == 512
    assert link["observed_anchors"][0]["text_truncated"] is True
    assert link["observed_anchors_truncated"] is True
    assert sum(a["text"] == "same" for a in link["observed_anchors"]) == 1


def test_projection_config_exactly_binds_new_behavior(tmp_path):
    tools, _, _, _ = project(tmp_path, '<a href="/news/preview">Headline</a>')
    config = tools.configuration()
    assert NewsDiscoveryTools.from_configuration(tools.archive, config, clock=lambda: NOW).configuration() == config
    altered = deepcopy(config)
    altered["news_discovery"]["link_context"]["max_distinct_labels_per_url"] = 20
    with pytest.raises(ValueError, match="exactly"):
        NewsDiscoveryTools.from_configuration(tools.archive, altered, clock=lambda: NOW)


def test_link_projection_preserves_typed_article_body_and_evidence_contract(tmp_path):
    metadata = {"@type": "NewsArticle", "headline": "Training update", "url": ARTICLE,
                "datePublished": "2026-10-06T18:00:00+00:00"}
    html = ('<script type="application/ld+json">' + json.dumps(metadata) + '</script>'
            '<article><h1>Training update</h1>'
            '<p>Arsenal may change the starting lineup because a striker has an injury.</p>'
            '<p>The Arsenal manager will check the squad again after training tomorrow.</p>'
            '</article><a href="/news/next">Next source update</a>')
    archive = RawArchive(tmp_path / "body-raw")
    observation = archive.save("open_page", ARTICLE, {}, html.encode(), status=200, observed_at=NOW)
    prepared = {"tool": "read_url", "endpoint": ARTICLE, "params": {}, "source": "open_page"}
    documents = {version: NewsDiscoveryTools(archive, ["club.example"], clock=lambda: NOW,
        projection_version=version).project(observation, prepared) for version in (3, 4)}
    assert documents[4]["news_projection_version"] == "structured-news-discovery-v4"
    assert documents[4]["structured_news"] == documents[3]["structured_news"]
    assert documents[4]["structured_news"]["version"] == "structured-news-discovery-v3"
    assert len(documents[4]["structured_news"]["segments"]) == 2
    match = {"match_id": "body-contract", "home_team": "Arsenal", "away_team": "Chelsea",
             "kickoff_at": "2026-10-08T20:00:00+00:00", "prediction_cutoff": "2026-10-08T19:00:00+00:00"}
    projections = {version: build_match_evidence(match, [document], as_of=NOW,
        version="match-evidence-extraction-v5") for version, document in documents.items()}
    assert projections[4]["evidence"] == projections[3]["evidence"]
    assert projections[4]["evidence"]
    assert projections[4]["state"]["followup_candidates"] == projections[3]["state"]["followup_candidates"]
    assert projections[4]["state"]["followup_candidates"]
