from football_agent.context_query_readiness import (
    ContextState,
    prepare_context_query_board,
)
from football_agent.query_value_contract import QueryValueAssessment, QueryValueGate, QUERY
from football_agent.state_query_contract import StateHypothesis, StateQueryBoard, StateQueryCandidate


TARGET = "context-query-match"
CUTOFF = "2026-09-29T12:00:00Z"


def board():
    return StateQueryBoard(
        target_key=TARGET, prediction_cutoff=CUTOFF,
        hypotheses=[
            StateHypothesis("qualification-h", TARGET, "qualification", ("already_qualified", "must_win")),
            StateHypothesis("reserve-h", TARGET, "reserve_strength", ("known_strong", "unknown")),
            StateHypothesis("tactics-h", TARGET, "tactical_intent", ("defensive", "attacking")),
            StateHypothesis("lineup-h", TARGET, "lineup", ("known", "unknown")),
        ],
        candidates=[
            StateQueryCandidate("qualification-refresh", TARGET, CUTOFF, "qualification", "standings", "read", .01),
            StateQueryCandidate("reserve-strength", TARGET, CUTOFF, "reserve_strength", "squad", "read", .01),
            StateQueryCandidate("tactical-intent", TARGET, CUTOFF, "tactical_intent", "tactics", "read", .01),
            StateQueryCandidate("lineup", TARGET, CUTOFF, "lineup", "lineup", "read", .01),
        ],
    )


def routes():
    return [
        {"candidate_id": "qualification-refresh", "route_kind": "context",
         "provides_context_keys": ["qualification"]},
        {"candidate_id": "reserve-strength", "route_kind": "context",
         "required_context_keys": ["qualification"], "provides_context_keys": ["reserve_strength"]},
        {"candidate_id": "tactical-intent", "route_kind": "domain",
         "required_context_keys": ["qualification"]},
        {"candidate_id": "lineup", "route_kind": "domain",
         "required_context_keys": ["qualification", "reserve_strength"]},
    ]


def test_initial_context_only_exposes_context_resolution_route():
    plan = prepare_context_query_board(board(), ContextState({"qualification": None, "reserve_strength": None}), routes())
    assert plan.eligible_candidate_ids == ("qualification-refresh",)
    lineup = next(row for row in plan.route_audit if row["candidate_id"] == "lineup")
    assert lineup["eligible"] is False
    assert lineup["missing_context_keys"] == ["qualification", "reserve_strength"]
    assert all(row["expected_value"] is None for row in plan.route_audit)


def test_context_resolution_replans_after_each_public_context_update():
    after_qualification = prepare_context_query_board(
        board(), ContextState({"qualification": "already_qualified", "reserve_strength": None}), routes(),
    )
    assert after_qualification.eligible_candidate_ids == ("reserve-strength", "tactical-intent")
    after_reserve = prepare_context_query_board(
        board(), ContextState({"qualification": "already_qualified", "reserve_strength": "known_strong"}), routes(),
    )
    assert after_reserve.eligible_candidate_ids == ("tactical-intent", "lineup")


def test_unknown_marker_and_context_state_updates_are_explicit():
    state = ContextState({"qualification": "UNKNOWN"})
    assert state.missing(("qualification",)) == ("qualification",)
    updated = state.with_updates({"qualification": "must_win"})
    assert updated.known("qualification") is True


def test_numeric_gate_runs_only_after_context_filtering():
    initial = prepare_context_query_board(board(), ContextState({"qualification": None}), routes())
    assessment = QueryValueAssessment.modeled("qualification-refresh", expected_value=.05, cost=.01)
    decision = QueryValueGate.decide(initial.board, [assessment], remaining_budget=.05)
    assert decision.action == QUERY
    assert decision.candidate_id == "qualification-refresh"

    updated = prepare_context_query_board(
        board(), ContextState({"qualification": "already_qualified", "reserve_strength": None}), routes(),
    )
    assert [candidate.candidate_id for candidate in updated.board.candidates] == ["reserve-strength", "tactical-intent"]

