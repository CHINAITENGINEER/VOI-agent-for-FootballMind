from __future__ import annotations

import pytest

from football_agent.query_value_contract import (
    BUDGET_EXHAUSTED, CALIBRATION_DIAGNOSTIC_ONLY, DEFER,
    MODELED_DECLARED_ASSUMPTIONS, MODEL_SCOPED_STOP, NO_FEASIBLE_REGISTERED_ACTION,
    QUERY, STOP, VALUE_UNMEASURED, WITHHELD_OPEN_WORLD_VALUE_UNKNOWN,
    QueryValueAssessment, QueryValueGate, StopCertificate,
)
from football_agent.action_readiness import ActionReadiness, prepare_query_board
from football_agent.state_query_contract import StateHypothesis, StateQueryBoard, StateQueryCandidate


TARGET = "match-1"
CUTOFF = "2026-09-27T10:00:00Z"


def board(*, include_candidates=True):
    candidates = []
    if include_candidates:
        candidates = [
            StateQueryCandidate("lineup", TARGET, CUTOFF, "rotation_status", "lineup", "read", .01,
                                action_id="lineup"),
            StateQueryCandidate("coach", TARGET, CUTOFF, "tactical_intent", "coach", "read", .02,
                                action_id="coach"),
        ]
    return StateQueryBoard(
        target_key=TARGET, prediction_cutoff=CUTOFF,
        hypotheses=[
            StateHypothesis("rotation", TARGET, "rotation_status", ("strong_xi", "reserve_xi")),
            StateHypothesis("tactics", TARGET, "tactical_intent", ("defensive", "attacking")),
        ], candidates=candidates,
    )


def test_unmeasured_value_keeps_numeric_value_unknown():
    assessment = QueryValueAssessment.unmeasured("lineup", cost=.01)
    assert assessment.value_status == VALUE_UNMEASURED
    assert assessment.expected_value is None and assessment.net_value is None
    with pytest.raises(ValueError, match="expected value"):
        QueryValueAssessment("lineup", VALUE_UNMEASURED, 0.0, .01, None,
                             "open_world", "NOT_CALIBRATED", "bad")


def test_modeled_finite_value_computes_net_value_with_explicit_scope():
    assessment = QueryValueAssessment.modeled("lineup", expected_value=.04, cost=.01)
    assert assessment.value_status == MODELED_DECLARED_ASSUMPTIONS
    assert assessment.net_value == pytest.approx(.03)
    assert assessment.calibration_status == "ASSUMPTION_ONLY"


def test_finite_model_row_without_support_stays_unmeasured():
    assert QueryValueAssessment.from_finite_model_row("lineup", {"feasible": False}).value_status == VALUE_UNMEASURED
    with pytest.raises(ValueError, match="immediate_evsi"):
        QueryValueAssessment.from_finite_model_row("lineup", {"feasible": True, "cost": .01})


def test_gate_queries_highest_positive_modeled_net_value():
    decision = QueryValueGate.decide(board(), [
        QueryValueAssessment.modeled("lineup", expected_value=.04, cost=.01),
        QueryValueAssessment.modeled("coach", expected_value=.03, cost=.02),
    ], remaining_budget=.05)
    assert decision.action == QUERY and decision.candidate_id == "lineup"
    assert decision.stop_certificate is None


def test_gate_withholds_open_world_value_when_no_supported_positive_action_exists():
    decision = QueryValueGate.decide(board(), [
        QueryValueAssessment.modeled("lineup", expected_value=.01, cost=.01),
        QueryValueAssessment.unmeasured("coach", cost=.02),
    ], remaining_budget=.05)
    assert decision.action == DEFER
    assert decision.stop_certificate.kind == WITHHELD_OPEN_WORLD_VALUE_UNKNOWN
    assert decision.stop_certificate.global_claim is False


def test_gate_stops_only_within_declared_model_when_all_candidates_are_nonpositive():
    decision = QueryValueGate.decide(board(), [
        QueryValueAssessment.modeled("lineup", expected_value=.01, cost=.01),
        QueryValueAssessment.modeled("coach", expected_value=.01, cost=.02),
    ], remaining_budget=.05)
    assert decision.action == STOP
    assert decision.stop_certificate.kind == MODEL_SCOPED_STOP
    assert decision.stop_certificate.scope == "declared_finite_model"
    assert decision.stop_certificate.maximum_modeled_net_value <= 0


def test_budget_exhaustion_is_distinct_from_zero_value():
    decision = QueryValueGate.decide(board(), [
        QueryValueAssessment.modeled("lineup", expected_value=.2, cost=.01),
        QueryValueAssessment.modeled("coach", expected_value=.2, cost=.02),
    ], remaining_budget=0)
    assert decision.action == DEFER
    assert decision.stop_certificate.kind == BUDGET_EXHAUSTED


def test_candidates_over_remaining_budget_receive_budget_certificate():
    decision = QueryValueGate.decide(board(), [
        QueryValueAssessment.modeled("lineup", expected_value=.2, cost=.1),
        QueryValueAssessment.modeled("coach", expected_value=.2, cost=.2),
    ], remaining_budget=.05)
    assert decision.stop_certificate.kind == BUDGET_EXHAUSTED


def test_no_actionable_candidates_use_registered_scope_certificate():
    decision = QueryValueGate.decide(board(include_candidates=False), [], remaining_budget=.05)
    assert decision.action == STOP
    assert decision.stop_certificate.kind == NO_FEASIBLE_REGISTERED_ACTION


def test_diagnostic_only_value_cannot_become_net_value():
    with pytest.raises(ValueError, match="expected value"):
        QueryValueAssessment("lineup", CALIBRATION_DIAGNOSTIC_ONLY, .01, .01, None,
                             "out_of_sample_diagnostic", "NOT_CALIBRATED", "diagnostic")


def test_global_stop_claim_is_rejected():
    with pytest.raises(ValueError, match="global optimum"):
        StopCertificate(MODEL_SCOPED_STOP, "declared_finite_model", True,
                        "finite model stop", 1, ("lineup",), .05, 0.0)


def test_assessment_ids_must_match_actionable_candidates():
    with pytest.raises(ValueError, match="currently actionable"):
        QueryValueGate.decide(board(), [QueryValueAssessment.unmeasured("weather", cost=.01)], remaining_budget=.05)


def test_query_decision_serializes_assessments_and_certificates():
    decision = QueryValueGate.decide(board(), [
        QueryValueAssessment.modeled("lineup", expected_value=.04, cost=.01),
        QueryValueAssessment.unmeasured("coach", cost=.02),
    ], remaining_budget=.05)
    manifest = decision.as_dict()
    assert manifest["action"] == QUERY
    assert len(manifest["assessments"]) == 2


def test_readiness_query_plan_filters_board_before_numeric_gate():
    readiness = ActionReadiness(
        aggregate_id="aggregate-1", mode="any_usable", availability="failed",
        quality="PROVIDER_ERROR", next_step="QUERY_ALTERNATIVE_SOURCE",
        actionable_dependency_keys=("lineup:r1",), late_dependency_keys=(),
        failed_dependency_keys=("lineup:r1",), missing_dependency_keys=(),
    )
    plan = prepare_query_board(readiness=readiness, board=board(), candidates=[
        {"candidate_id": "lineup", "dependency_key": "lineup:r1", "route_role": "alternative"},
        {"candidate_id": "coach", "dependency_key": "coach:r2", "route_role": "alternative"},
    ])
    assert plan.eligible_candidate_ids == ("lineup",)
    assert [item.candidate_id for item in plan.board.actionable_candidates()] == ["lineup"]
    decision = QueryValueGate.decide(plan.board, [
        QueryValueAssessment.modeled("lineup", expected_value=.04, cost=.01),
    ], remaining_budget=.05)
    assert decision.action == QUERY and decision.candidate_id == "lineup"
