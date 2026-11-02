from copy import deepcopy
from hashlib import sha256

import pytest

from football_agent.match_evidence import build_match_evidence


MATCH = {"match_id": "fixture-1", "home_team": "Arsenal", "away_team": "Leeds",
         "away_aliases": ["Leeds United"], "kickoff_at": "2026-10-10T11:30:00Z",
         "prediction_cutoff": "2026-10-09T11:30:00Z"}
AS_OF = "2026-10-06T12:00:00Z"


def document(text, identity="page-1", **updates):
    return {"document_id": identity, "url": "https://club.example/news/" + identity,
            "kind": "page", "status": "OK", "text": text, "links": [],
            "raw_sha256": sha256(text.encode()).hexdigest(),
            "observed_at": "2026-10-06T09:00:00Z", "available_at": "2026-10-06T09:00:01Z",
            "source_published_at": None, "publication_status": "unknown", **updates}


def test_only_match_input_produces_grounded_evidence_and_source_derived_followup():
    source = document("Arsenal vs Leeds United Saturday 10th October 2026. "
                      "Arsenal have given no official update on Kai Havertz's fitness ahead of the game.")
    original = deepcopy(source)
    report = build_match_evidence(MATCH, [source], as_of=AS_OF)
    assert source == original
    assert report["state"]["present_topics"] == ["fixture_confirmation"]
    assert report["state"]["context_topics"] == ["availability"]
    assert len(report["state"]["followup_candidates"]) == 1
    query = report["state"]["followup_candidates"][0]
    assert "Havertz" in query["query"] and "2026-10-10" in query["query"]
    assert query["expected_gain"] is None
    for row in report["evidence"]:
        anchor = row["anchor"]
        assert source["text"][anchor["start"]:anchor["end"]] == row["statement"]
        assert anchor["raw_sha256"] == source["raw_sha256"]
        assert row["source_published_at"] is None
        assert row["calibrated_confidence"] is None
    assert report["semantics"]["numeric_voi"] is None
    assert report["semantics"]["probability_update"] is None
    assert report["semantics"]["llm_calls"] == 0


@pytest.mark.parametrize("text", [
    "Home Arsenal Leeds United News Team Lineup Injuries Table xG Stats Contact Login",
    "Manchester City could rotate their starting lineup due to several muscle injuries.",
    "Arsenal Women have several injury doubts before playing Leeds Women this Saturday.",
    "Arsenal U21 have several injury doubts before playing Leeds U21 this Saturday.",
])
def test_navigation_unrelated_team_and_different_squad_do_not_create_evidence(text):
    report = build_match_evidence(MATCH, [document(text)], as_of=AS_OF)
    assert not report["evidence"]
    assert not report["state"]["followup_candidates"]
    assert report["state"]["unresolved"][0]["reasons"] == ["no_substantive_match_evidence_observed"]


@pytest.mark.parametrize("updates,reason", [
    ({"kind": "discovery"}, "discovery_or_unsuccessful_response_not_source_evidence"),
    ({"status": "NO_DATA"}, "discovery_or_unsuccessful_response_not_source_evidence"),
    ({"available_at": "2026-10-06T13:00:00Z"}, "unavailable_at_decision_time"),
    ({"observed_at": "2026-10-06T10:00:00Z"}, "inconsistent_source_clock"),
    ({"source_published_at": "2026-10-11T00:00:00Z"}, "inconsistent_source_clock"),
    ({"observed_at": None}, "missing_or_invalid_operational_time"),
])
def test_discovery_unavailable_and_clock_contamination_stays_out(updates, reason):
    report = build_match_evidence(MATCH, [document("Arsenal vs Leeds on 2026-10-10 have injury doubts.", **updates)], as_of=AS_OF)
    assert not report["evidence"]
    assert report["document_reviews"][0]["reason"] == reason


@pytest.mark.parametrize("text,reason", [
    ("Match report: Arsenal vs Leeds on 2026-10-10. Arsenal had injury concerns but won.", "post_match_report_text"),
    ("Arsenal 3-0 Leeds. Arsenal had fitness concerns during the game.", "target_score_in_report_lead"),
    ("Arsenal vs Leeds on 2025-10-10. Arsenal will rotate the squad.", "different_fixture_date"),
])
def test_retrospective_or_other_year_text_not_current_match_facts(text, reason):
    report = build_match_evidence(MATCH, [document(text)], as_of=AS_OF)
    assert not report["evidence"]
    assert report["document_reviews"][0]["reason"] == reason


def test_fixture_date_cannot_be_borrowed_from_different_match_on_schedule():
    text = ("Arsenal vs Leeds on 2025-10-10. Arsenal vs Chelsea on 2026-10-10. "
            "Arsenal will rotate some players because of injuries.")
    report = build_match_evidence(MATCH, [document(text)], as_of=AS_OF)
    assert not report["evidence"]


def test_missing_fixture_date_cannot_be_borrowed_from_next_fixture():
    source = document("Arsenal vs Leeds date pending. Arsenal vs Chelsea on 2026-10-10.")
    report = build_match_evidence(MATCH, [source], as_of=AS_OF)
    assert "fixture_confirmation" not in report["state"]["present_topics"]
    assert report["document_reviews"][0]["fixture_bindings"][0]["status"] == "pair_present_date_unresolved"


def test_unknown_year_and_general_team_news_stay_context_and_dont_assume_absence():
    report = build_match_evidence(MATCH, [document("Arsenal vs Leeds on 10 October. Arsenal might rotate the starting lineup.")], as_of=AS_OF)
    assert report["evidence"]
    assert all(e["applicability"] == "team_context_requires_match_binding" for e in report["evidence"])
    assert report["state"]["present_topics"] == []
    assert report["state"]["followup_candidates"]


def test_opposed_source_statements_are_visible_but_not_verified_contradictions():
    positive = document("Arsenal v Leeds on 2026-10-10: Arsenal will rotate the starting lineup.", "yes")
    negative = document("Arsenal v Leeds on 2026-10-10: Arsenal will not rotate the starting lineup.", "no")
    report = build_match_evidence(MATCH, [positive, negative], as_of=AS_OF)
    assert len(report["state"]["conflict_signals"]) == 1
    conflict = report["state"]["conflict_signals"][0]
    assert conflict["verified_contradiction"] is False
    assert len(conflict["evidence_ids"]) == 2
    assert set(report["state"]["present_topics"]) == {"fixture_confirmation", "lineup", "rotation"}
    assert report["state"]["followup_candidates"][0]["based_on_evidence_ids"] == conflict["evidence_ids"]


def test_reused_document_identity_cannot_change_and_duplicate_quote_not_new_source():
    one = document("Arsenal v Leeds on 2026-10-10. Arsenal have injury doubts.")
    same = document(one["text"], "mirror")
    report = build_match_evidence(MATCH, [one, same], as_of=AS_OF)
    single = build_match_evidence(MATCH, [one, one], as_of=AS_OF)
    assert len(report["evidence"]) == len(single["evidence"])
    assert len(single["document_reviews"]) == 1
    with pytest.raises(ValueError, match="different acquired content"):
        build_match_evidence(MATCH, [one, {**one, "text": one["text"] + " extra"}], as_of=AS_OF)


def test_bound_is_reported_and_no_generic_checklist_is_generated():
    report = build_match_evidence(MATCH, [document("Arsenal v Leeds on 2026-10-10. Arsenal have injury doubts.")], as_of=AS_OF, max_evidence=1)
    assert len(report["evidence"]) == 1
    assert report["state"]["extraction_truncated"]
    assert not report["state"]["followup_candidates"]
    assert not report["state"]["coverage_complete"]


def test_match_specific_new_topic_is_not_lost_to_a_closed_feature_checklist():
    source = document("Arsenal v Leeds on 2026-10-10 may be moved because of a transport strike.")
    report = build_match_evidence(MATCH, [source], as_of=AS_OF)
    assert "other_match_context" in report["state"]["present_topics"]
    assert any("strike" in row["query"] for row in report["state"]["followup_candidates"])


def test_aliases_and_chinese_fixture_quotes_preserve_original_offsets():
    match = {**MATCH, "home_team": "阿森纳", "away_team": "利兹联", "away_aliases": []}
    source = document("阿森纳 对阵 利兹联 2026年10月10日。 阿森纳可能因为伤病而进行轮换，目前首发待定。")
    report = build_match_evidence(match, [source], as_of=AS_OF)
    assert "fixture_confirmation" in report["state"]["present_topics"]
    assert "rotation" in report["state"]["context_topics"]
    assert any("阿森纳" in row["query"] for row in report["state"]["followup_candidates"])
