"""Local-language scoping against immutable archived bytes, with negative controls."""
from copy import deepcopy
from pathlib import Path
import json

import pytest

from football_agent.news_discovery import NewsDiscoveryTools
from football_research.acquisition.state import parse_time
from football_research.data_adapters.local_text_scope import (
    _digest, build_local_scope_proof, associate_local_text,
)
from football_spider.storage.raw_archive import RawArchive

ROOT = Path(__file__).resolve().parents[3]
ACTOR = ROOT / "var/parallel/agent/round184/prospective-panels/actor-2981554/match-2981554"
PANEL = ROOT / "var/parallel/agent/round184/panel-repair/panel-2981554/source-archive"


@pytest.fixture(scope="module")
def real_inputs():
    binding = json.loads((ACTOR / "configuration.json").read_text(encoding="utf-8"))["supervisor_metadata"]["verified_fixture_identity"]
    archive = RawArchive(ACTOR / "source-archive")
    obs = next(o for o in archive.observations() if o.endpoint == "https://jalgpall.ee/koondis/1/uudised")
    proof = build_local_scope_proof(binding, identity_archive=ACTOR / "identity-archive",
        index_observation=obs, index_archive=archive.root)
    pages = RawArchive(PANEL)
    tools = NewsDiscoveryTools(pages, ["jalgpall.ee"], projection_version=2,
        clock=lambda: parse_time("2026-10-07T08:00:00+00:00"))
    documents = [tools.project(o, {"source": o.source, "endpoint": o.endpoint, "params": o.params, "tool": "read_url"})
                 for o in pages.observations()]
    return proof, documents, binding, obs, archive


def derived_document(real_inputs, headline, body, *, publication="2026-10-06", raw="06 10.2026 12:05"):
    # Controlled text mutations are negative parser tests, not real observations.
    document = deepcopy(real_inputs[1][0])
    text = headline + "\n" + body
    document["text"] = text
    document["structured_news"].update(headline=headline, segments=[
        {"text": headline, "start": 0, "end": len(headline), "content_role": "headline"},
        {"text": body, "start": len(headline)+1, "end": len(text),
         "source_publication_date": publication, "publication_raw": raw}])
    return document


def test_real_index_and_uefa_identity_do_not_prove_translation_or_local_kickoff(real_inputs):
    proof, _, binding, obs, archive = real_inputs
    assert proof["source_scope_supported"] and proof["language"] == "et"
    assert proof["source_category_evidence"]["text"] == "Mehed"
    assert proof["index_fixture_kickoff_proof"] is None
    assert proof["alias_status"] == "unverified_source_assertion"
    raw = archive.read_body(obs).decode("utf-8")
    for entry in proof["index_entries"]:
        evidence = entry["title_evidence"]
        assert raw[evidence["start"]:evidence["end"]] == evidence["quote"] == entry["title"]
    assert proof == build_local_scope_proof(binding, identity_archive=ACTOR / "identity-archive",
        index_observation=obs, index_archive=archive.root)


def test_two_archived_articles_are_match_assertion_and_person_background(real_inputs):
    proof, documents, *_ = real_inputs
    before = deepcopy(documents)
    results = [associate_local_text(d, proof=proof) for d in documents]
    assert [r["scope"] for r in results] == ["match_specific", "team_context"]
    assert all(r["alias_status"] == "unverified_source_assertion" and not r["independent_identity_established"] for r in results)
    assert results[0]["date_evidence"][0]["event_date"] == "2026-10-06"
    assert results[0]["date_evidence"][0]["today_evidence"]["quote"].startswith("täna")
    assert results[1]["pair_evidence"] and results[1]["date_evidence"] == []
    for document, result in zip(documents, results):
        for evidence in result["mention_evidence"] + result["pair_evidence"] + result["date_evidence"]:
            assert document["text"][evidence["start"]:evidence["end"]] == evidence["quote"]
    assert documents == before


@pytest.mark.parametrize("field,value", [("url", "https://example.com/koondis/1/uudised/story"),
    ("url", "https://jalgpall.ee/koondis/2/uudised/story"), ("url", "https://jalgpall.ee.evil.test/koondis/1/uudised/story"),
    ("source_language", "en"), ("source_language", None), ("language_source", "caller_claim")])
def test_aliases_never_apply_outside_exact_source_language_and_observed_menu(real_inputs, field, value):
    document = deepcopy(real_inputs[1][0])
    (document if field == "url" else document["structured_news"])[field] = value
    result = associate_local_text(document, proof=real_inputs[0])
    assert result["scope"] == "unscoped" and result["mention_evidence"] == []


def test_ordinary_island_word_is_not_a_country_team(real_inputs):
    doc = derived_document(real_inputs, "Island hotel", "Island is a common English word on a travel poster.")
    result = associate_local_text(doc, proof=real_inputs[0])
    assert result["scope"] == "unscoped" and result["mention_evidence"] == []


@pytest.mark.parametrize("headline", ["Eesti U21 – Island U21 mäng", "Eesti – Island U19 mäng", "Eesti naiste koondis – Island mäng"])
def test_other_age_or_sex_categories_are_not_bound_to_senior_men(real_inputs, headline):
    doc = derived_document(real_inputs, headline, "Eesti – Island mäng toimub täna, 6. oktoobril.")
    result = associate_local_text(doc, proof=real_inputs[0])
    assert result["scope"] == "unscoped"


@pytest.mark.parametrize("body,publication,expected", [
    ("Mäng toimub 6. oktoobril.", "2026-10-06", "team_context"),
    ("Mäng toimub 6. oktoobril 2025.", "2026-10-06", "team_context"),
    ("Mäng toimub täna, 7. oktoobril.", "2026-10-07", "team_context"),
    ("Mäng toimub täna, 6. oktoobril.", None, "team_context"),
    ("Mäng toimub täna, 6. oktoobril.", "2025-10-06", "team_context"),
    ("Mäng toimub 6. oktoobril 2026.", None, "match_specific"),
])
def test_missing_year_is_not_filled_from_fixture_or_publication_alone(real_inputs, body, publication, expected):
    doc = derived_document(real_inputs, "Eesti – Island Rahvuste liiga mäng", body, publication=publication)
    assert associate_local_text(doc, proof=real_inputs[0])["scope"] == expected


def test_names_and_publication_date_do_not_make_person_profile_match_specific(real_inputs):
    doc = derived_document(real_inputs, "Islandi koondises mänginud Emil", "Eesti jalgpall ja Islandi koondis on tema eluloo osad.")
    result = associate_local_text(doc, proof=real_inputs[0])
    assert result["scope"] == "team_context" and len(result["mentioned_sides"]) == 2
    assert result["date_evidence"] == []


def test_changed_proof_and_segment_offsets_fail_closed(real_inputs):
    proof = deepcopy(real_inputs[0])
    proof["aliases"]["away"]["canonical_name"] = "Other team"
    with pytest.raises(ValueError, match="Changed"):
        associate_local_text(real_inputs[1][0], proof=proof)
    doc = deepcopy(real_inputs[1][0])
    doc["structured_news"]["segments"][0]["start"] += 1
    with pytest.raises(ValueError, match="offsets"):
        associate_local_text(doc, proof=real_inputs[0])


def test_unsupported_index_scope_does_not_authorize_any_alias(real_inputs):
    proof = deepcopy(real_inputs[0])
    proof["source_scope_supported"] = False
    proof["proof_sha256"] = _digest({k: v for k, v in proof.items() if k != "proof_sha256"})
    assert associate_local_text(real_inputs[1][0], proof=proof)["scope"] == "unscoped"
