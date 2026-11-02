from __future__ import annotations

import pytest

from football_agent.prequery_state import PreQueryStateAnnotation, PreQueryStateRegistry
from football_agent.state_query_contract import (
    ACTIONABLE_UNKNOWN, EXCLUDED_CUTOFF, EXCLUDED_NO_HYPOTHESIS, KNOWN,
    STATE_ALREADY_KNOWN, StateHypothesis, StateQueryBoard, StateQueryCandidate,
)


TARGET = "match-1"
CUTOFF = "2026-09-27T10:00:00Z"


def annotations():
    return PreQueryStateRegistry([
        PreQueryStateAnnotation(
            annotation_id="qualification-1", target_key=TARGET,
            state_variable="qualification_status", value="already_qualified",
            basis="pre_query_external_annotation", provenance_kind="official_pre_match_rule",
            source_ids=("official-standings",), available_at="2026-09-27T09:00:00Z",
            observed_at="2026-09-27T09:01:00Z", prediction_cutoff=CUTOFF, scope="group_stage",
        ),
        PreQueryStateAnnotation(
            annotation_id="tactical-1", target_key=TARGET,
            state_variable="tactical_intent", value="defensive",
            basis="pre_query_external_annotation", provenance_kind="pre_match_statement",
            source_ids=("coach-statement",), available_at="2026-09-27T09:05:00Z",
            observed_at="2026-09-27T09:06:00Z", prediction_cutoff=CUTOFF, scope="match_plan",
        ),
    ])


def board():
    return StateQueryBoard(
        target_key=TARGET, prediction_cutoff=CUTOFF, registry=annotations(),
        hypotheses=[
            StateHypothesis("qualification", TARGET, "qualification_status",
                            ("already_qualified", "must_win"), status=KNOWN,
                            known_value="already_qualified", annotation_id="qualification-1"),
            StateHypothesis("rotation", TARGET, "rotation_status",
                            ("strong_xi", "rotation_unknown", "reserve_xi")),
            StateHypothesis("tactics", TARGET, "tactical_intent",
                            ("defensive", "attacking")),
        ],
        candidates=[
            StateQueryCandidate("lineup", TARGET, CUTOFF, "rotation_status", "lineup_feed", "read", .01,
                                endpoint="https://lineup.example/match-1", action_id="lineup"),
            StateQueryCandidate("coach", TARGET, CUTOFF, "tactical_intent", "coach_statement", "read", .02,
                                endpoint="https://coach.example/match-1", action_id="coach"),
            StateQueryCandidate("qualification-refresh", TARGET, CUTOFF, "qualification_status",
                                "standings", "read", .01),
            StateQueryCandidate("weather", TARGET, CUTOFF, "weather", "weather_feed", "read", .01),
            StateQueryCandidate("late-lineup", TARGET, "2026-09-27T10:30:00Z", "rotation_status",
                                "lineup_feed", "read", .01),
        ],
    )


def test_board_marks_actionable_unknowns_without_assigning_unmeasured_value():
    result = {row["candidate_id"]: row for row in board().assess_candidates()}
    assert result["lineup"]["status"] == ACTIONABLE_UNKNOWN
    assert result["coach"]["status"] == ACTIONABLE_UNKNOWN
    assert result["qualification-refresh"]["status"] == STATE_ALREADY_KNOWN
    assert result["weather"]["status"] == EXCLUDED_NO_HYPOTHESIS
    assert result["late-lineup"]["status"] == EXCLUDED_CUTOFF
    assert result["lineup"]["expected_value"] is None
    assert result["lineup"]["value_status"] == "UNMEASURED_UNTIL_RETURN_MODEL_CALIBRATED"


def test_actionable_candidates_are_scoped_to_unresolved_state_variables():
    assert [item.candidate_id for item in board().actionable_candidates()] == ["lineup", "coach"]


def test_applying_tactical_annotation_does_not_infer_rotation_or_qualification():
    updated = board().with_annotation(annotations().require("tactical-1"))
    hypotheses = {item.state_variable: item for item in updated.hypotheses}
    assert hypotheses["tactical_intent"].status == KNOWN
    assert hypotheses["rotation_status"].status == "UNKNOWN"
    assert hypotheses["qualification_status"].known_value == "already_qualified"
    statuses = {row["candidate_id"]: row["status"] for row in updated.assess_candidates()}
    assert statuses["coach"] == STATE_ALREADY_KNOWN
    assert statuses["lineup"] == ACTIONABLE_UNKNOWN


def test_manifest_explicitly_preserves_no_automatic_derivation():
    manifest = board().manifest()
    assert manifest["automatic_derivation"] is False
    assert manifest["agent_selection_remains_policy_decision"] is True
    assert manifest["actionable_candidate_ids"] == ["lineup", "coach"]


def test_agent_view_redacts_private_known_value_and_does_not_change_candidate_menu():
    evaluator = board().manifest(view="evaluator")
    agent = board().manifest(view="agent")
    qualification_evaluator = next(item for item in evaluator["hypotheses"]
                                   if item["state_variable"] == "qualification_status")
    qualification_agent = next(item for item in agent["hypotheses"]
                               if item["state_variable"] == "qualification_status")
    assert qualification_evaluator["known_value"] == "already_qualified"
    assert qualification_evaluator["annotation_id"] == "qualification-1"
    assert qualification_agent["known_value"] is None
    assert qualification_agent["annotation_id"] is None
    assert qualification_agent["status"] == "UNKNOWN"
    assert "qualification-refresh" not in evaluator["actionable_candidate_ids"]
    assert "qualification-refresh" in agent["actionable_candidate_ids"]
    assert next(item for item in agent["candidate_assessments"]
                if item["candidate_id"] == "qualification-refresh")["reason"] == "state_not_visible_to_agent"


def test_unknown_board_view_is_rejected():
    with pytest.raises(ValueError, match="Unknown state query board view"):
        board().manifest(view="public")


def test_known_hypothesis_must_match_registry_annotation():
    with pytest.raises(ValueError, match="differs from its pre-query annotation"):
        StateQueryBoard(target_key=TARGET, prediction_cutoff=CUTOFF, registry=annotations(), hypotheses=[
            StateHypothesis("bad", TARGET, "qualification_status", ("already_qualified", "must_win"),
                            status=KNOWN, known_value="must_win", annotation_id="qualification-1")
        ], candidates=[])


def test_annotation_value_outside_declared_support_is_rejected():
    unsupported = PreQueryStateAnnotation(
        annotation_id="tactical-bad", target_key=TARGET, state_variable="tactical_intent",
        value="counterpressing", basis="pre_query_external_annotation", provenance_kind="pre_match_statement",
        source_ids=("coach-statement",), available_at="2026-09-27T09:05:00Z",
        observed_at="2026-09-27T09:06:00Z", prediction_cutoff=CUTOFF, scope="match_plan",
    )
    with pytest.raises(ValueError, match="outside the hypothesis support"):
        board().with_annotation(unsupported)


def test_annotation_for_unlisted_variable_is_rejected():
    unknown = PreQueryStateAnnotation(
        annotation_id="weather-1", target_key=TARGET, state_variable="weather",
        value="rain", basis="pre_query_external_annotation", provenance_kind="external_pre_match_annotation",
        source_ids=("weather-feed",), available_at="2026-09-27T09:05:00Z",
        observed_at="2026-09-27T09:06:00Z", prediction_cutoff=CUTOFF, scope="stadium",
    )
    with pytest.raises(ValueError, match="not declared"):
        board().with_annotation(unknown)


def test_duplicate_state_variable_hypotheses_are_rejected():
    with pytest.raises(ValueError, match="one explicit hypothesis"):
        StateQueryBoard(target_key=TARGET, prediction_cutoff=CUTOFF, hypotheses=[
            StateHypothesis("r1", TARGET, "rotation_status", ("strong_xi",)),
            StateHypothesis("r2", TARGET, "rotation_status", ("reserve_xi",)),
        ], candidates=[])


def test_mapping_round_trip_keeps_candidate_scope_and_cost():
    candidate = StateQueryCandidate.from_mapping({
        "candidate_id": "lineup", "target_key": TARGET, "prediction_cutoff": CUTOFF,
        "state_variable": "rotation_status", "source_role": "lineup_feed",
        "query_kind": "read", "cost": .01, "endpoint": "https://lineup.example/match-1",
        "action_id": "lineup",
    })
    assert StateQueryCandidate.from_mapping(candidate.as_dict()) == candidate
