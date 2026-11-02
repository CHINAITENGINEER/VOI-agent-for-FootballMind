"""Connect exact-context empirical VoI calibration to the query gate.

The helper keeps a global calibration estimate as a diagnostic record while
passing only the exact-context estimate into ``QueryValueGate``.  A context
with insufficient future-safe support remains explicitly unmeasured, so a
large source-wide average does not silently authorize a query in a new match
state.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Iterable

from .contextual_adaptive_voi import ContextualVoIAssessment, calibrate_contextual_candidate_values
from .query_value_contract import QueryValueAssessment, QueryValueDecision, QueryValueGate
from .state_query_contract import StateQueryBoard
from .voi_evidence_bridge import VoIEvidenceLink


VERSION = "contextual-query-gate-v1"


@dataclass(frozen=True)
class ContextualGateResult:
    context_key: str
    estimates: tuple[ContextualVoIAssessment, ...]
    assessments: tuple[QueryValueAssessment, ...]
    decision: QueryValueDecision

    def __post_init__(self) -> None:
        if not isinstance(self.context_key, str) or not self.context_key.strip():
            raise ValueError("context_key must be a nonempty string")
        if any(not isinstance(item, ContextualVoIAssessment) for item in self.estimates):
            raise ValueError("estimates must contain ContextualVoIAssessment values")
        if any(not isinstance(item, QueryValueAssessment) for item in self.assessments):
            raise ValueError("assessments must contain QueryValueAssessment values")
        if not isinstance(self.decision, QueryValueDecision):
            raise ValueError("decision must be a QueryValueDecision")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": VERSION,
            "context_key": self.context_key,
            "estimates": [item.as_dict() for item in self.estimates],
            "assessments": [item.as_dict() for item in self.assessments],
            "decision": self.decision.as_dict(),
            "hidden_result_included": False,
        }


def decide_contextual_query(
    board: StateQueryBoard,
    routes: Mapping[str, Mapping[str, Any]],
    *,
    context: Mapping[str, Any],
    links: Iterable[VoIEvidenceLink],
    context_by_link_id: Mapping[str, Mapping[str, Any]],
    as_of: str | datetime,
    remaining_budget: float,
    min_independent_dependencies: int = 5,
    confidence: float = 0.10,
    view: str = "evaluator",
) -> ContextualGateResult:
    """Calibrate current routes and make one context-scoped gate decision."""
    if not isinstance(board, StateQueryBoard):
        raise ValueError("board must be a StateQueryBoard")
    if not isinstance(routes, Mapping) or not routes:
        raise ValueError("routes must be a nonempty mapping")
    candidate_ids = {candidate.candidate_id for candidate in board.actionable_candidates(view=view)}
    if set(routes) - candidate_ids:
        raise ValueError("routes must correspond to currently actionable candidates")
    for candidate_id, route in routes.items():
        if not isinstance(route, Mapping):
            raise ValueError("each route must be a mapping")
        if route.get("source_id") is None or route.get("source_version") is None or route.get("cost") is None:
            raise ValueError(f"route {candidate_id} needs source_id, source_version and cost")
    estimates = calibrate_contextual_candidate_values(
        routes, context=context, links=links, context_by_link_id=context_by_link_id,
        as_of=as_of, min_independent_dependencies=min_independent_dependencies,
        confidence=confidence,
    )
    assessments = tuple(item.as_query_assessment() for item in estimates)
    decision = QueryValueGate.decide(board, assessments, remaining_budget=remaining_budget, view=view)
    context_key = estimates[0].context_key if estimates else ""
    return ContextualGateResult(context_key, estimates, assessments, decision)


__all__ = ["VERSION", "ContextualGateResult", "decide_contextual_query"]
