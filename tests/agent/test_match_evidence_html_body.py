"""Opt-in typed HTML body evidence; previous versions stay frozen."""
from copy import deepcopy
from datetime import timedelta
from hashlib import sha256
import json
from pathlib import Path
import socket

import pytest

from football_agent.match_evidence import build_match_evidence, ARTICLES_VERSION, CONTENT_VERSION, HTML_BODY_VERSION
from football_agent.news_discovery import NewsDiscoveryTools
from football_agent.inquiry_contract import utc
from football_spider.models.contracts import Observation
from football_spider.storage.raw_archive import RawArchive
from test_match_evidence import MATCH, AS_OF, document
from test_match_evidence_articles import article


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("No HTTP for archived body projection"))
    monkeypatch.setattr(NewsDiscoveryTools, "execute", lambda *a, **k: pytest.fail("Do not acquire another page"))


def typed(parts):
    source = article(parts)
    source["structured_news"]["version"] = "structured-news-discovery-v3"
    for i, segment in enumerate(source["structured_news"]["segments"]):
        segment.update(content_source="semantic_html_article_paragraph", content_role="paragraph",
            css_selector="article div.body p", original_dom_paragraph_index=i+2, source_url=source["url"])
    return source


def project(source, match=MATCH):
    return build_match_evidence(match, [source], as_of=AS_OF, version=HTML_BODY_VERSION)


def test_new_typed_body_is_explicit_v5_only_and_keeps_offsets():
    source = typed(["Arsenal Kai Havertz may miss the starting lineup because of an injury."])
    before = deepcopy(source)
    for old in (ARTICLES_VERSION, CONTENT_VERSION):
        previous = build_match_evidence(MATCH, [source], as_of=AS_OF, version=old)
        assert previous["evidence"] == previous["state"]["followup_candidates"] == []
    result = project(source)
    assert result["schema_version"] == HTML_BODY_VERSION
    assert result["evidence"] and result["state"]["followup_candidates"]
    row = result["evidence"][0]
    assert source["text"][row["anchor"]["start"]:row["anchor"]["end"]] == row["statement"]
    assert row["source_segment"]["original_dom_paragraph_index"] == 2
    assert row["source_segment"]["content_source"] == "semantic_html_article_paragraph"
    assert source == before and row["expected_gain"] is None
    assert result["semantics"]["numeric_voi"] is None


@pytest.mark.parametrize("field,value", [("original_dom_paragraph_index", -1), ("original_dom_paragraph_index", 1024),
    ("original_dom_paragraph_index", True), ("original_dom_paragraph_index", "2"), ("css_selector", ""),
    ("css_selector", "p"*513), ("content_role", "headline"), ("source_url", "https://other.example/fixture")])
def test_typed_paragraph_requires_same_page_and_bounded_original_dom_provenance(field, value):
    source = typed(["Arsenal SECRETSTRIKER may miss the starting lineup with an injury."])
    source["structured_news"]["segments"][0][field] = value
    result = project(source)
    assert not result["evidence"] and not result["state"]["followup_candidates"]
    assert "SECRETSTRIKER" not in json.dumps(result)


def test_new_marker_cannot_be_smuggled_into_old_news_schema():
    source = typed(["Arsenal may update their starting lineup after training."])
    source["structured_news"]["version"] = "structured-news-discovery-v2"
    result = project(source)
    assert not result["evidence"]
    assert result["document_reviews"][0]["excluded_segments"][0]["reason"] == "typed_html_paragraph_requires_news_v3"


def test_v5_keeps_unknown_role_and_flattened_article_unsupported():
    source = typed(["Arsenal may update their starting lineup after training."])
    source["structured_news"]["role"] = "unknown"
    assert not project(source)["state"]["followup_candidates"]
    source["structured_news"]["role"] = "article"
    source["structured_news"]["segments"][0]["content_source"] = "semantic_html_article"
    assert not project(source)["evidence"]


def test_explicit_other_squad_lead_keeps_later_generic_team_text_unscoped():
    source = typed(["Arsenal Women's team will play Chelsea Women in a separate competition.",
                    "Arsenal may change the starting lineup after training with Kai Havertz."])
    result = project(source)
    assert not result["evidence"] and not result["state"]["followup_candidates"]
    assert len(result["unscoped_source_quotes"]) == 2
    scope = result["document_reviews"][0]["article_subject_scope"]
    assert scope["article_wide_exclusion"] and scope["source_squad_categories"] == ["women"]
    assert scope["anchor"]["quote"] == source["structured_news"]["segments"][0]["text"]


@pytest.mark.parametrize("lead", [
    "Arsenal men's and women's teams will both train at the same facility.",
    "Arsenal v Leeds on 2026-10-10 and a separate women's fixture are both scheduled.",
])
def test_mixed_or_explicit_target_fixture_lead_does_not_exclude_entire_article(lead):
    source = typed([lead, "Arsenal Kai Havertz may miss the starting lineup with an injury."])
    result = project(source)
    assert result["evidence"] and result["state"]["followup_candidates"]
    assert result["document_reviews"][0]["article_subject_scope"]["article_wide_exclusion"] is False
    assert "Havertz" in result["state"]["followup_candidates"][0]["query"]


def test_explicitly_targeted_womens_fixture_remains_eligible():
    match = {**MATCH, "home_team": "Arsenal Women", "away_team": "Leeds Women"}
    source = typed(["Arsenal Women may rotate their starting lineup before the next match."])
    assert project(source, match)["evidence"]


def test_future_lead_cannot_set_scope_or_leak_into_public_evidence():
    source = typed([{"text": "Arsenal SECRETSTRIKER Women's team may rotate.",
                     "source_modified_at": "2026-10-06T10:00:00Z"},
                    "Arsenal Kai Havertz may miss the starting lineup with an injury."])
    result = project(source)
    assert result["evidence"] and result["state"]["followup_candidates"]
    assert "SECRETSTRIKER" not in json.dumps(result)
    assert "article_subject_scope" not in result["document_reviews"][0]


def test_existing_v4_output_snapshot_is_unchanged_including_rejection_of_news_v3():
    text = "Arsenal may change the lineup because Kai Havertz has an injury."
    unknown = article([text], role="unknown")
    unknown["kind"] = "page"
    new = article([text])
    new["structured_news"]["version"] = "structured-news-discovery-v3"
    sources = [document(text), article([text]), unknown, new]
    outputs = [build_match_evidence(MATCH, [source], as_of=AS_OF, version=CONTENT_VERSION) for source in sources]
    value = json.dumps(outputs, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    assert sha256(value).hexdigest() == "f0b230c4b62abdaf98128bcef8bb62312f57cedcc4cbf25889e413b7ee8888ab"


def test_real_saved_article_restores_body_without_binding_womens_biography_to_mens_match():
    root = Path(__file__).resolve().parents[2]
    folder = root/"var/parallel/agent/round191/live-canada/match-3097883"
    if not folder.exists():
        pytest.skip("Real Round191 article archive not available")
    paths = sorted(path for path in folder.rglob("*") if path.is_file())
    before = {path: sha256(path.read_bytes()).hexdigest() for path in paths}
    config = json.loads((folder/"configuration.json").read_text(encoding="utf-8"))
    receipt = json.loads((folder/"receipt-003.json").read_text(encoding="utf-8"))
    observation = Observation.model_validate(receipt["receipt"]["observation"])
    at = utc(receipt["document"]["available_at"])+timedelta(seconds=1)
    archive = RawArchive(folder/"source-archive")
    tools = NewsDiscoveryTools(archive, ["news.canadasoccer.com"], projection_version=3, clock=lambda: at)
    derived = tools.project(observation, receipt["receipt"]["prepared"])
    assert derived["kind"] == "article" and derived["structured_news"]["role"] == "article"
    assert len(derived["structured_news"]["segments"]) == 7
    result = build_match_evidence(config["evidence_projection"]["match"], [derived], as_of=at, version=HTML_BODY_VERSION)
    assert result["unscoped_source_quotes"]
    assert not result["evidence"] and not result["state"]["followup_candidates"]
    assert result["document_reviews"][0]["article_subject_scope"]["source_squad_categories"] == ["women"]
    for quote in result["unscoped_source_quotes"]:
        anchor = quote["anchor"]
        assert derived["text"][anchor["start"]:anchor["end"]] == quote["statement"]
    assert result["semantics"]["numeric_voi"] is None
    assert {path: sha256(path.read_bytes()).hexdigest() for path in paths} == before
