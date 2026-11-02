"""Football-specific context routing for the autonomous-information demo.

The cases are synthetic analogues of tournament situations.  A known
qualification state is treated as one public fact only: it does not infer
rotation, lineup strength, or tactical intent.  Those variables remain
separate unknowns and are offered as candidate information routes subject to
the current context and the VoI gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

from .context_query_readiness import ContextState, ContextQueryPlan, prepare_context_query_board
from .query_value_contract import QueryValueAssessment, QueryValueDecision, QueryValueGate
from .state_query_contract import StateHypothesis, StateQueryBoard, StateQueryCandidate


VERSION = "football-context-routing-v1"
CUTOFF = "2026-09-29T12:00:00Z"


@dataclass(frozen=True)
class FootballContextCase:
    case_id: str
    context: Mapping[str, Any]
    value_by_candidate: Mapping[str, float]

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError("case_id must be a nonempty string")
        if not isinstance(self.context, Mapping) or not self.context:
            raise ValueError("context must be a nonempty mapping")
        if not isinstance(self.value_by_candidate, Mapping) or not self.value_by_candidate:
            raise ValueError("value_by_candidate must be a nonempty mapping")
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0 for value in self.value_by_candidate.values()):
            raise ValueError("candidate values must be nonnegative numbers")

    def board(self) -> StateQueryBoard:
        hypotheses = [
            StateHypothesis("qualification-h", "match:context-demo", "qualification",
                            ("already_qualified", "draw_sufficient", "must_win")),
            StateHypothesis("reserve-h", "match:context-demo", "reserve_strength",
                            ("rotation_likely", "strong_lineup", "unknown")),
            StateHypothesis("tactical-h", "match:context-demo", "tactical_intent",
                            ("defensive", "attacking", "unknown")),
            StateHypothesis("lineup-h", "match:context-demo", "lineup", ("known", "unknown")),
        ]
        candidates = [
            StateQueryCandidate("qualification-refresh", "match:context-demo", CUTOFF,
                                "qualification", "standings", "read", .01),
            StateQueryCandidate("reserve-strength", "match:context-demo", CUTOFF,
                                "reserve_strength", "lineup", "read", .01),
            StateQueryCandidate("tactical-intent", "match:context-demo", CUTOFF,
                                "tactical_intent", "news", "read", .01),
            StateQueryCandidate("lineup", "match:context-demo", CUTOFF,
                                "lineup", "lineup", "read", .02),
        ]
        return StateQueryBoard(target_key="match:context-demo", prediction_cutoff=CUTOFF,
                               hypotheses=hypotheses, candidates=candidates)

    @staticmethod
    def routes() -> tuple[dict[str, Any], ...]:
        return (
            {"candidate_id": "qualification-refresh", "route_kind": "context",
             "required_context_keys": (), "provides_context_keys": ("qualification",)},
            {"candidate_id": "reserve-strength", "route_kind": "context",
             "required_context_keys": ("qualification",), "provides_context_keys": ("reserve_strength",)},
            {"candidate_id": "tactical-intent", "route_kind": "context",
             "required_context_keys": ("qualification",), "provides_context_keys": ("tactical_intent",)},
            {"candidate_id": "lineup", "route_kind": "domain",
             "required_context_keys": ("fixture_id", "competition_id", "qualification",
                                         "reserve_strength", "tactical_intent"), "provides_context_keys": ()},
        )

    def plan(self) -> tuple[ContextQueryPlan, QueryValueDecision]:
        board = self.board()
        plan = prepare_context_query_board(board, ContextState(self.context), self.routes())
        assessments = [QueryValueAssessment.modeled(
            candidate_id, expected_value=self.value_by_candidate[candidate_id],
            cost=next(item.cost for item in plan.board.candidates if item.candidate_id == candidate_id),
            reason="synthetic_football_context_return_model",
        ) for candidate_id in plan.eligible_candidate_ids if candidate_id in self.value_by_candidate]
        decision = QueryValueGate.decide(plan.board, assessments, remaining_budget=.10)
        return plan, decision

    def as_dict(self) -> dict[str, Any]:
        plan, decision = self.plan()
        return {
            "schema_version": VERSION, "case_id": self.case_id,
            "context": dict(self.context), "plan": plan.as_dict(),
            "decision": decision.as_dict(), "hidden_result_included": False,
            "interpretation": "Qualification is a context fact; rotation and tactical intent remain separate unknowns.",
        }


def build_football_context_cases() -> tuple[FootballContextCase, ...]:
    base = {"fixture_id": "fixture-context-demo", "competition_id": "competition-context-demo"}
    return (
        FootballContextCase(
            "qualified-but-rotation-unknown",
            {**base, "qualification": "already_qualified"},
            {"reserve-strength": .07, "tactical-intent": .03},
        ),
        FootballContextCase(
            "draw-sufficient-but-tactics-unknown",
            {**base, "qualification": "draw_sufficient"},
            {"reserve-strength": .02, "tactical-intent": .08},
        ),
        FootballContextCase(
            "qualification-unknown",
            base,
            {"qualification-refresh": .05},
        ),
    )


def replay_cases() -> dict[str, Any]:
    rows = [case.as_dict() for case in build_football_context_cases()]
    by_id = {row["case_id"]: row for row in rows}
    checks = {
        "qualified_case_queries_reserve_context": by_id["qualified-but-rotation-unknown"]["decision"]["candidate_id"] == "reserve-strength",
        "qualified_case_lineup_waits_for_unknowns": any(
            row["candidate_id"] == "lineup" and not row["eligible"]
            and row["reason"] == "context_prerequisite_missing"
            for row in by_id["qualified-but-rotation-unknown"]["plan"]["route_audit"]
        ),
        "draw_case_queries_tactical_context": by_id["draw-sufficient-but-tactics-unknown"]["decision"]["candidate_id"] == "tactical-intent",
        "unknown_case_queries_qualification": by_id["qualification-unknown"]["decision"]["candidate_id"] == "qualification-refresh",
        "qualification_does_not_infer_rotation": "reserve_strength" not in by_id["qualified-but-rotation-unknown"]["context"],
        "qualification_does_not_infer_tactics": "tactical_intent" not in by_id["draw-sufficient-but-tactics-unknown"]["context"],
        "all_contexts_are_pre_match": all("score" not in json_row["context"] and "result" not in json_row["context"] for json_row in rows),
        "hidden_result_flag_false": all(row["hidden_result_included"] is False for row in rows),
    }
    return {"schema_version": VERSION, "status": "ROUND80_FOOTBALL_CONTEXT_ROUTING", "cases": rows, "checks": checks}


__all__ = ["VERSION", "CUTOFF", "FootballContextCase", "build_football_context_cases", "replay_cases"]
