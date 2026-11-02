import json

from football_agent.football_context_routing import (
    VERSION,
    build_football_context_cases,
    replay_cases,
)


def case_map():
    return {case.case_id: case for case in build_football_context_cases()}


def test_qualified_context_keeps_rotation_and_tactics_as_separate_unknowns():
    case = case_map()["qualified-but-rotation-unknown"]
    plan, decision = case.plan()
    assert decision.candidate_id == "reserve-strength"
    assert "reserve_strength" not in case.context
    assert "tactical_intent" not in case.context
    lineup = next(row for row in plan.route_audit if row["candidate_id"] == "lineup")
    assert lineup["eligible"] is False
    assert lineup["reason"] == "context_prerequisite_missing"


def test_draw_sufficient_context_prioritizes_tactical_intent_when_its_value_is_higher():
    case = case_map()["draw-sufficient-but-tactics-unknown"]
    plan, decision = case.plan()
    assert decision.candidate_id == "tactical-intent"
    assert "qualification" not in plan.context.missing(("qualification",))
    assert plan.context.known("tactical_intent") is False


def test_unknown_qualification_blocks_downstream_routes():
    case = case_map()["qualification-unknown"]
    plan, decision = case.plan()
    assert decision.candidate_id == "qualification-refresh"
    assert plan.eligible_candidate_ids == ("qualification-refresh",)
    assert all("score" not in key and "result" not in key for key in case.context)


def test_replay_contains_all_context_routing_checks():
    replay = replay_cases()
    assert replay["schema_version"] == VERSION
    assert len(replay["cases"]) == 3
    assert all(replay["checks"].values())
    json.dumps(replay, ensure_ascii=False)


def test_context_routing_manifest_has_no_hidden_result_flag():
    for case in build_football_context_cases():
        payload = case.as_dict()
        assert payload["hidden_result_included"] is False
        assert payload["plan"]["context"]["schema_version"] == "context-query-readiness-v1"


def test_route_values_are_explicit_policy_inputs():
    for case in build_football_context_cases():
        _, decision = case.plan()
        assert decision.selection_basis == "declared_finite_model_positive_net_value"
