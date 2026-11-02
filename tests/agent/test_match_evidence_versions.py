"""Source-scope regression for the observed mixed Leeds homepage.

The archived fixture is the already-acquired text projection from round177
live-match/receipt-000.json. It is not fetched during testing and does not
represent a current statement about any squad's availability.
"""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from football_agent.match_evidence import LEGACY_VERSION, VERSION, build_match_evidence


MATCH = {"match_id": "bc1e406d-5997-5c62-a2a7-a5d4d8c5612e", "home_team": "Arsenal",
         "away_team": "Leeds", "home_aliases": ["Arsenal"], "away_aliases": ["Leeds United"],
         "kickoff_at": "2026-10-10T11:30:00+00:00", "prediction_cutoff": "2026-10-09T11:30:00+00:00"}
AS_OF = "2026-10-06T09:55:46.732857+00:00"


def document(text, url="https://club.example/news/team-update"):
    return {"document_id": "page-1", "url": url, "kind": "page", "status": "OK",
            "text": text, "links": [{"url": "https://club.example/news/team-update", "text": "Team update"}],
            "raw_sha256": sha256(text.encode()).hexdigest(),
            "observed_at": "2026-10-06T09:00:00Z", "available_at": "2026-10-06T09:00:01Z",
            "source_published_at": None, "publication_status": "unknown"}


def test_real_homepage_v1_reproduces_original_trace_while_v2_only_discovers():
    source = json.loads((Path(__file__).parent / "fixtures/match_evidence/leeds-home-20261006.json").read_text(encoding="utf-8"))
    original = deepcopy(source)
    legacy = build_match_evidence(MATCH, [source], as_of=AS_OF, version=LEGACY_VERSION)
    assert legacy["schema_version"] == "match-evidence-extraction-v1"
    assert len(legacy["evidence"]) == 1
    row = legacy["evidence"][0]
    # This intentionally pins the old, known-bad interpretation for historical
    # replay; new runs must never acquire its Women's/Bradford follow-up.
    assert row["evidence_id"] == "evidence-81ea363caa62f3423d8d5efa"
    assert (row["anchor"]["start"], row["anchor"]["end"]) == (13913, 14311)
    assert legacy["state"]["followup_candidates"][0]["query"] == (
        "Arsenal Leeds 2026-10-10 Women's XI Bradford Womens One Sep Alayna Millard We've Community Previous Next")
    projected = build_match_evidence(MATCH, [source], as_of=AS_OF)
    assert projected["schema_version"] == VERSION == "match-evidence-extraction-v2"
    assert projected == build_match_evidence(MATCH, [source], as_of=AS_OF, version=VERSION)
    assert not projected["evidence"] and not projected["state"]["followup_candidates"]
    assert projected["document_reviews"][0]["status"] == "discovery_only"
    assert source == original and len(source["links"]) > 0


@pytest.mark.parametrize("url", ["https://club.example/", "https://club.example/en?article=12345",
                                "https://club.example/en/news", "https://club.example/news/first-team"])
def test_index_routes_cannot_supply_facts_even_with_target_team_and_fixture_words(url):
    source = document("Women have announced the starting lineup. Academy Previous Next. "
                      "Arsenal v Leeds on 2026-10-10. Leeds have injury doubts. Community Previous Next.", url)
    report = build_match_evidence(MATCH, [source], as_of=AS_OF)
    assert not report["evidence"] and not report["state"]["followup_candidates"]
    assert report["document_reviews"][0]["reason"] == "index_page_for_link_discovery_only"
    # Page-role decisions cannot relax the actual operational clock gate.
    source["available_at"] = "2026-10-06T11:00:00Z"
    report = build_match_evidence(MATCH, [source], as_of=AS_OF)
    assert report["document_reviews"][0]["reason"] == "unavailable_at_decision_time"


def test_article_keeps_target_passage_and_excludes_cross_card_squad_fragment():
    source = document("Arsenal have given no official update on Kai Havertz's fitness.\n"
                      "Women's starting XI is announced for Bradford while Leeds support the Community event.\n"
                      "Academy U21 have changed their starting lineup for the Leeds fixture.")
    report = build_match_evidence(MATCH, [source], as_of=AS_OF)
    assert len(report["evidence"]) == 1
    assert "Havertz" in report["evidence"][0]["statement"]
    assert len(report["state"]["followup_candidates"]) == 1
    query = report["state"]["followup_candidates"][0]["query"]
    assert "Havertz" in query and "Bradford" not in query and "U21" not in query
    skipped = report["document_reviews"][0]["excluded_passages"]
    assert len(skipped) == 2
    assert {c for item in skipped for c in item["source_squad_categories"]} == {"women", "youth", "u21"}
    for item in skipped:
        anchor = item["anchor"]
        assert source["text"][anchor["start"]:anchor["end"]] == anchor["quote"]


def test_other_squad_prefix_cannot_disappear_from_short_fixture_anchor():
    source = document("Women's fixture: Arsenal v Leeds on 2026-10-10. "
                      "Arsenal have fitness concerns ahead of another match.")
    report = build_match_evidence(MATCH, [source], as_of=AS_OF)
    assert "fixture_confirmation" not in report["state"]["present_topics"]
    assert not report["document_reviews"][0]["fixture_bindings"]
    assert report["document_reviews"][0]["excluded_passages"]
    assert all(row["applicability"] == "team_context_requires_match_binding" for row in report["evidence"])


def test_named_womens_fixture_is_supported_but_foreign_squad_route_is_not():
    women = {**MATCH, "home_team": "Arsenal Women", "home_aliases": [],
             "away_team": "Leeds Women", "away_aliases": []}
    source = document("Arsenal Women have announced their starting lineup before Leeds Women arrive.",
                      "https://club.example/women/news/lineup-update")
    report = build_match_evidence(women, [source], as_of=AS_OF)
    assert report["evidence"]
    other = document("Arsenal have announced their starting lineup before Leeds arrive.",
                     "https://club.example/women/news/lineup-update")
    report = build_match_evidence(MATCH, [other], as_of=AS_OF)
    assert not report["evidence"]
    assert report["document_reviews"][0]["reason"] == "non_target_squad_source_section"


def test_unknown_extraction_version_cannot_silently_use_new_rules():
    with pytest.raises(ValueError, match="Unsupported match evidence version"):
        build_match_evidence(MATCH, [], as_of=AS_OF, version="future-unregistered-version")
