"""A small executable loop joining context readiness, VoI and query execution.

The loop is intentionally domain-neutral at its core: callers provide a
public context, a query board, route prerequisites and local tool returns.  It
replans after every accepted or unusable return and records the complete
episode in the existing context-transition ledger.  The returned report
contains hashes and context updates only; raw result bodies stay in the local
tool boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from datetime import datetime, timezone
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
import tempfile
from typing import Any, Callable

from .context_query_readiness import ContextState, prepare_context_query_board
from .context_transition import ContextTransition, ContextTransitionLedger
from .query_execution_envelope import FAILED, LATE, QueryExecutionEnvelope, QueryExecutionLedger
from .query_value_contract import DEFER, QUERY, STOP, QueryValueAssessment, QueryValueDecision, QueryValueGate
from .state_query_contract import StateHypothesis, StateQueryBoard
from .types import utc


VERSION = "autonomous-context-loop-v2"


@dataclass(frozen=True)
class LoopReturn:
    candidate_id: str
    context_updates: Mapping[str, Any]
    result_payload: Mapping[str, Any]
    evidence_refs: tuple[str, ...]
    observed_at: str
    operational_available_at: str
    captured_at: str

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("candidate_id must be a nonempty string")
        if not isinstance(self.context_updates, Mapping) or not isinstance(self.result_payload, Mapping):
            raise ValueError("loop returns need mapping payloads")
        refs = tuple(str(item) for item in self.evidence_refs)
        if not refs or any(not item.strip() for item in refs):
            raise ValueError("loop returns need evidence references")
        object.__setattr__(self, "evidence_refs", refs)


@dataclass(frozen=True)
class LoopResult:
    target_key: str
    episode_id: str
    selected_candidates: tuple[str, ...]
    statuses: tuple[str, ...]
    decisions: tuple[dict[str, Any], ...]
    final_context: Mapping[str, Any]
    transition_events: tuple[str, ...]
    execution_entries: int
    stop_reason: str
    final_action: str = STOP
    initial_budget: float = 1.0
    remaining_budget: float = 1.0
    declared_cost_spent: float = 0.0
    value_mode: str = "declared_fixture_values"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": VERSION,
            "target_key": self.target_key,
            "episode_id": self.episode_id,
            "selected_candidates": list(self.selected_candidates),
            "statuses": list(self.statuses),
            "decisions": [dict(item) for item in self.decisions],
            "final_context": dict(self.final_context),
            "transition_events": list(self.transition_events),
            "execution_entries": self.execution_entries,
            "stop_reason": self.stop_reason,
            "final_action": self.final_action,
            "initial_budget": self.initial_budget,
            "remaining_budget": self.remaining_budget,
            "declared_cost_spent": self.declared_cost_spent,
            "cost_semantics": "declared acquisition cost reserved per attempted tool call; not a billing receipt",
            "value_mode": self.value_mode,
            "hidden_result_included": False,
        }


def _remaining_board(board: StateQueryBoard, remaining: set[str]) -> StateQueryBoard:
    return StateQueryBoard(
        target_key=board.target_key, prediction_cutoff=board.prediction_cutoff,
        hypotheses=board.hypotheses,
        candidates=[candidate for candidate in board.candidates if candidate.candidate_id in remaining],
        registry=board.registry,
    )


def run_autonomous_context_loop(
    *,
    board: StateQueryBoard,
    initial_context: Mapping[str, Any],
    routes: Iterable[Mapping[str, Any]],
    values: Mapping[str, float] | None = None,
    returns: Mapping[str, LoopReturn] | None = None,
    episode_id: str,
    root: str | Path | None = None,
    max_steps: int = 8,
    initial_evidence_refs: Iterable[str] = (),
    initial_trigger: str = "agent_loop",
    initial_reason: str = "public context before autonomous acquisition",
    initial_captured_at: str | None = None,
    budget: float = 1.0,
    decision_provider: Callable[[StateQueryBoard, Mapping[str, Any], float], QueryValueDecision] | None = None,
    execute: Callable[[QueryExecutionEnvelope, Mapping[str, Any]], LoopReturn] | None = None,
    tool_version: str = "autonomous-loop-fixture-v1",
) -> LoopResult:
    """Run a bounded public-context acquisition episode and replay its ledger."""
    if not isinstance(board, StateQueryBoard):
        raise ValueError("board must be a StateQueryBoard")
    # Never expose evaluator-only annotations through callbacks or plan logs.
    board = StateQueryBoard(
        target_key=board.target_key, prediction_cutoff=board.prediction_cutoff,
        hypotheses=[StateHypothesis.from_mapping(item.as_dict(view="agent")) for item in board.hypotheses],
        candidates=board.candidates,
    )
    if decision_provider is None:
        if not isinstance(values, Mapping) or not values:
            raise ValueError("values must be a nonempty mapping when no decision provider is supplied")
    elif not callable(decision_provider) or values is not None:
        raise ValueError("provide one dynamic decision provider or declared fixture values")
    if execute is None:
        if not isinstance(returns, Mapping):
            raise ValueError("returns must be a mapping when no executor is supplied")
    elif not callable(execute) or returns is not None:
        raise ValueError("provide one executor or a fixture return mapping")
    if isinstance(budget, bool) or not isinstance(budget, (float, int)) or not math.isfinite(budget) or budget < 0:
        raise ValueError("budget must be finite and nonnegative")
    state = ContextState(deepcopy(initial_context))
    route_rows = tuple(deepcopy(dict(route)) for route in routes)
    remaining = {candidate.candidate_id for candidate in board.candidates}
    missing_returns = remaining - set(returns) if execute is None else set()
    if missing_returns:
        raise ValueError("returns must cover every candidate")
    if decision_provider is None and any(candidate_id not in values for candidate_id in remaining):
        raise ValueError("values must cover every candidate")
    episode_id = str(episode_id).strip()
    if not episode_id:
        raise ValueError("episode_id must be nonempty")
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")

    owns_root = root is None
    temporary = tempfile.TemporaryDirectory(prefix="footballmind-loop-") if owns_root else None
    ledger_root = Path(temporary.name if temporary else root)
    try:
        transitions = ContextTransitionLedger(ledger_root / "transitions")
        executions = QueryExecutionLedger(ledger_root / "executions")
        if any(entry.episode_id == episode_id for entry in transitions.entries()):
            raise ValueError("append-only episode already exists; inspect its ledger before starting a new episode")
        initial_refs = tuple(initial_evidence_refs)
        records: list[ContextTransition] = [ContextTransition.from_transition(
            episode_id=episode_id, target_key=board.target_key, sequence=1,
            event_type="INITIALIZED", previous_context={}, next_context=state.values,
            trigger=initial_trigger, reason=initial_reason,
            prediction_cutoff=board.prediction_cutoff,
            captured_at=initial_captured_at or board.prediction_cutoff,
            source_id="collector-context-bridge" if initial_refs else None,
            evidence_refs=initial_refs,
        )]
        transitions.append(records)
        selected: list[str] = []
        statuses: list[str] = []
        decisions: list[dict[str, Any]] = []
        sequence = 2
        stop_reason = "candidate_set_exhausted"
        final_action = STOP
        remaining_budget = float(budget)
        last_capture = records[0].captured_at

        for _step in range(max_steps):
            if not remaining:
                break
            current_board = _remaining_board(board, remaining)
            plan = prepare_context_query_board(
                current_board,
                state,
                tuple(route for route in route_rows if route.get("candidate_id") in remaining),
            )
            if decision_provider is None:
                assessments = [QueryValueAssessment.modeled(
                    candidate.candidate_id, expected_value=float(values[candidate.candidate_id]),
                    cost=candidate.cost, reason="loop_declared_context_return_model",
                ) for candidate in plan.board.actionable_candidates(view="agent")]
                decision = QueryValueGate.decide(plan.board, assessments, remaining_budget=remaining_budget, view="agent")
            else:
                decision = decision_provider(deepcopy(plan.board), deepcopy(state.values), remaining_budget)
                if not isinstance(decision, QueryValueDecision):
                    raise ValueError("decision provider must return QueryValueDecision")
                # Enforce the same executable gate even for an external policy.
                checked = QueryValueGate.decide(plan.board, decision.assessments, remaining_budget=remaining_budget, view="agent")
                if checked != decision:
                    raise ValueError("decision provider output must match the current board and budget gate")
            decisions.append({
                "context": state.as_dict(), "plan": plan.as_dict(), "decision": decision.as_dict(),
                "remaining_budget_before": remaining_budget,
            })
            if decision.action != QUERY:
                stop_reason = decision.stop_certificate.reason if decision.stop_certificate else "gate_stop"
                final_action = decision.action
                break
            candidate_id = str(decision.candidate_id)
            candidate = next(item for item in plan.board.candidates if item.candidate_id == candidate_id)
            route = next(item for item in route_rows if item.get("candidate_id") == candidate_id)
            expected_keys = tuple(route.get("provides_context_keys") or (candidate.state_variable,))
            if execute is None:
                expected_keys = tuple(returns[candidate_id].context_updates)
            planned = QueryExecutionEnvelope.from_decision(
                decision, plan.board,
                {"target_key": board.target_key, "candidate_id": candidate_id,
                 "request_nonce": f"{episode_id}:{sequence}"},
                tool_version=tool_version,
                expected_context_keys=expected_keys,
                requested_at=board.prediction_cutoff,
                evidence_refs=(f"request:{episode_id}:{sequence}",),
            )
            if planned.cost > remaining_budget + 1e-12 or not math.isclose(planned.cost, candidate.cost, abs_tol=1e-12, rel_tol=0):
                raise ValueError("selected cost must match the route and fit remaining budget")
            executions.append((planned,))
            remaining_budget = max(0.0, remaining_budget - planned.cost)
            selected_event = ContextTransition.from_transition(
                episode_id=episode_id, target_key=board.target_key, sequence=sequence,
                event_type="QUERY_SELECTED", previous_context=state.values,
                next_context=state.values, trigger="agent_loop", reason=decision.selection_basis,
                prediction_cutoff=board.prediction_cutoff, captured_at=planned.requested_at,
                selected_candidate_id=candidate_id, action=QUERY, source_id=planned.tool_version,
                evidence_refs=planned.evidence_refs,
            )
            transitions.append((selected_event,))
            route_return = None
            capture_unknown = False
            try:
                route_return = execute(planned, deepcopy(state.values)) if execute is not None else returns[candidate_id]
                if not isinstance(route_return, LoopReturn) or route_return.candidate_id != candidate_id:
                    raise ValueError("executor returned a mismatched candidate")
                completed = planned.complete(
                    route_return.result_payload, context_updates=route_return.context_updates,
                    evidence_refs=route_return.evidence_refs, observed_at=route_return.observed_at,
                    operational_available_at=route_return.operational_available_at,
                    captured_at=route_return.captured_at,
                )
            except Exception:
                # Failure can consume acquisition cost too. Never leak provider
                # exception text or invalid result bodies into public context.
                # An invalid return has no trusted execution receipt. Record
                # when this local failure is actually captured, without using
                # any timestamps from the rejected return as source evidence.
                failed_capture = datetime.now(timezone.utc).isoformat()
                capture_unknown = True
                completed = planned.complete(
                    {}, status=FAILED, error_code="EXECUTION_OR_VALIDATION_FAILED",
                    evidence_refs=(f"failure:{planned.query_id}",), captured_at=failed_capture,
                )
            executions.append((completed,))
            result_event = completed.to_context_transition(
                episode_id=episode_id, sequence=sequence + 1, previous_context=state.values,
                captured_at=completed.captured_at,
            )
            transitions.append((result_event,))
            selected.append(candidate_id)
            statuses.append(completed.status)
            last_capture = completed.captured_at
            state = ContextState(result_event.next_context)
            remaining.remove(candidate_id)
            sequence += 2
            if utc(completed.captured_at) > utc(board.prediction_cutoff):
                stop_reason, final_action = "prediction_cutoff_reached_after_return", DEFER
                break
            if capture_unknown:
                stop_reason, final_action = "execution_capture_unavailable", DEFER
                break
        else:
            if remaining:
                stop_reason, final_action = "maximum_steps_reached", DEFER
        stop = ContextTransition.from_transition(
            episode_id=episode_id, target_key=board.target_key, sequence=sequence,
            event_type="STOPPED", previous_context=state.values, next_context=state.values,
            trigger="agent_loop", reason=stop_reason, prediction_cutoff=board.prediction_cutoff,
            captured_at=last_capture, action=final_action,
        )
        transitions.append((stop,))
        replay = transitions.replay_episode(episode_id, target_key=board.target_key)
        return LoopResult(
            target_key=board.target_key, episode_id=episode_id,
            selected_candidates=tuple(selected), statuses=tuple(statuses), decisions=tuple(decisions),
            final_context=replay["final_context"],
            transition_events=tuple(item["event_type"] for item in replay["events"]),
            execution_entries=2 * len(selected), stop_reason=stop_reason,
            final_action=final_action, initial_budget=float(budget), remaining_budget=remaining_budget,
            declared_cost_spent=float(budget) - remaining_budget,
            value_mode="dynamic_context_policy" if decision_provider is not None else "declared_fixture_values",
        )
    finally:
        if temporary is not None:
            temporary.cleanup()


__all__ = ["VERSION", "LoopReturn", "LoopResult", "run_autonomous_context_loop"]
