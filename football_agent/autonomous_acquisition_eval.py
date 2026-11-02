"""Evaluate autonomous information acquisition under explicit synthetic assumptions.

This module is a mechanism evaluator, not a football forecasting model.  A policy
sees only a frozen public context, candidate query metadata, and the public return
of a query.  The realized outcome and the response tape live in the evaluator
layer and are never put into a query request, ``ContextState`` or a transition
ledger.  The experiment compares a fixed all-information policy, a first-query
heuristic, and the value-gated policy used by FootballMind.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import tempfile
from typing import Any

from .context_transition import ContextTransition, ContextTransitionLedger
from .query_execution_envelope import (
    AMBIGUOUS,
    FAILED,
    LATE,
    QueryExecutionEnvelope,
    QueryExecutionLedger,
)
from .query_value_contract import (
    QUERY,
    STOP,
    QueryValueAssessment,
    QueryValueDecision,
    QueryValueGate,
)
from .state_query_contract import StateHypothesis, StateQueryBoard, StateQueryCandidate


VERSION = "autonomous-acquisition-eval-v1"
POLICIES = ("fixed_all", "first_query", "voi_gate")
OUTCOMES = ("home_win", "draw", "away_win")
CUTOFF = "2026-09-29T12:00:00Z"
_FORBIDDEN = frozenset({
    "result", "outcome", "score", "home_score", "away_score", "half_score",
    "final_score", "probability", "sampled_result", "realized_loss",
})


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _time(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a timestamp")
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()


def _probabilities(value: Iterable[float], name: str = "prediction") -> tuple[float, ...]:
    values = tuple(value)
    if len(values) != len(OUTCOMES) or any(
        isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) or item < 0 or item > 1
        for item in values
    ) or abs(math.fsum(values) - 1.0) > 1e-9:
        raise ValueError(f"{name} must be a probability vector over {OUTCOMES}")
    return tuple(float(item) for item in values)


def _public(value: Any, path: str = "payload") -> Any:
    if isinstance(value, Mapping):
        output = {}
        for raw_key, raw_value in value.items():
            key = _text(str(raw_key), f"{path}.key")
            if key.lower() in _FORBIDDEN:
                raise ValueError(f"{path} contains an evaluator-only field")
            output[key] = _public(raw_value, f"{path}.{key}")
        return output
    if isinstance(value, (list, tuple)):
        return [_public(item, f"{path}[]") for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(f"{path} contains a non-serializable value")


@dataclass(frozen=True)
class ReturnSpec:
    """One public tool return selected by the evaluator's private response tape."""

    label: str
    result_payload: Mapping[str, Any]
    context_updates: Mapping[str, Any]
    prediction: tuple[float, ...] | None
    evidence_refs: tuple[str, ...]
    observed_at: str | None = None
    operational_available_at: str | None = None
    captured_at: str = "2026-09-29T10:01:00Z"
    status: str | None = None
    error_code: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", _text(self.label, "label"))
        payload = _public(self.result_payload, "result_payload")
        updates = _public(self.context_updates, "context_updates")
        if not isinstance(payload, Mapping) or not isinstance(updates, Mapping):
            raise ValueError("result_payload and context_updates must be mappings")
        object.__setattr__(self, "result_payload", dict(payload))
        object.__setattr__(self, "context_updates", dict(updates))
        if self.prediction is not None:
            object.__setattr__(self, "prediction", _probabilities(self.prediction))
        refs = tuple(_text(item, "evidence_ref") for item in self.evidence_refs)
        if not refs or len(refs) != len(set(refs)):
            raise ValueError("evidence_refs must contain unique values")
        object.__setattr__(self, "evidence_refs", refs)
        for field in ("observed_at", "operational_available_at", "captured_at"):
            raw_time = getattr(self, field)
            object.__setattr__(self, field, None if raw_time is None and field != "captured_at" else _time(raw_time, field))
        if self.status is not None:
            object.__setattr__(self, "status", _text(self.status, "status"))
        if self.error_code is not None:
            object.__setattr__(self, "error_code", _text(self.error_code, "error_code"))


@dataclass(frozen=True)
class QuerySpec:
    candidate_id: str
    state_variable: str
    possible_values: tuple[str, ...]
    cost: float
    expected_value: float
    responses: Mapping[str, ReturnSpec]

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _text(self.candidate_id, "candidate_id"))
        object.__setattr__(self, "state_variable", _text(self.state_variable, "state_variable"))
        values = tuple(_text(item, "possible_value") for item in self.possible_values)
        if not values or len(values) != len(set(values)):
            raise ValueError("possible_values must be unique and nonempty")
        object.__setattr__(self, "possible_values", values)
        if isinstance(self.cost, bool) or not isinstance(self.cost, (int, float)) or not math.isfinite(self.cost) or self.cost < 0:
            raise ValueError("cost must be finite and nonnegative")
        if isinstance(self.expected_value, bool) or not isinstance(self.expected_value, (int, float)) or not math.isfinite(self.expected_value) or self.expected_value < 0:
            raise ValueError("expected_value must be finite and nonnegative")
        responses = dict(self.responses)
        if set(responses) != set(values) or any(not isinstance(item, ReturnSpec) for item in responses.values()):
            raise ValueError("responses must cover every possible value")
        object.__setattr__(self, "responses", responses)


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    target_key: str
    prediction_cutoff: str
    initial_context: Mapping[str, Any]
    initial_prediction: tuple[float, ...]
    queries: tuple[QuerySpec, ...]
    private_response_labels: Mapping[str, str]
    private_outcome: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenario_id", _text(self.scenario_id, "scenario_id"))
        object.__setattr__(self, "target_key", _text(self.target_key, "target_key"))
        object.__setattr__(self, "prediction_cutoff", _time(self.prediction_cutoff, "prediction_cutoff"))
        context = _public(self.initial_context, "initial_context")
        if not isinstance(context, Mapping) or not context:
            raise ValueError("initial_context must be a nonempty public mapping")
        object.__setattr__(self, "initial_context", dict(context))
        object.__setattr__(self, "initial_prediction", _probabilities(self.initial_prediction, "initial_prediction"))
        queries = tuple(self.queries)
        if not queries or len({item.candidate_id for item in queries}) != len(queries):
            raise ValueError("queries must have unique candidate IDs")
        if len({item.state_variable for item in queries}) != len(queries):
            raise ValueError("queries must have independent state variables")
        object.__setattr__(self, "queries", queries)
        labels = dict(self.private_response_labels)
        if set(labels) != {item.candidate_id for item in queries}:
            raise ValueError("private response labels must cover every query")
        for query in queries:
            if labels[query.candidate_id] not in query.responses:
                raise ValueError("private response label is outside the query response support")
        object.__setattr__(self, "private_response_labels", labels)
        if self.private_outcome not in OUTCOMES:
            raise ValueError("private_outcome is outside the evaluator outcome support")

    def board(self, remaining_ids: Iterable[str]) -> StateQueryBoard:
        remaining = set(remaining_ids)
        hypotheses = [StateHypothesis(
            hypothesis_id=f"{query.candidate_id}-hypothesis", target_key=self.target_key,
            state_variable=query.state_variable, possible_values=query.possible_values,
        ) for query in self.queries if query.candidate_id in remaining]
        candidates = [StateQueryCandidate(
            candidate_id=query.candidate_id, target_key=self.target_key,
            prediction_cutoff=self.prediction_cutoff, state_variable=query.state_variable,
            source_role="synthetic_fixture", query_kind="read", cost=query.cost,
            action_id=f"{query.candidate_id}-read",
        ) for query in self.queries if query.candidate_id in remaining]
        return StateQueryBoard(target_key=self.target_key, prediction_cutoff=self.prediction_cutoff,
                               hypotheses=hypotheses, candidates=candidates)

    def public_manifest(self) -> dict[str, Any]:
        """Return only what a policy may see before execution."""
        return {
            "schema_version": VERSION, "scenario_id": self.scenario_id,
            "target_key": self.target_key, "prediction_cutoff": self.prediction_cutoff,
            "initial_context": dict(self.initial_context),
            "initial_prediction": list(self.initial_prediction),
            "queries": [{
                "candidate_id": query.candidate_id, "state_variable": query.state_variable,
                "possible_values": list(query.possible_values), "cost": query.cost,
                "expected_value": query.expected_value,
            } for query in self.queries],
            "evaluator_only": False,
        }


@dataclass(frozen=True)
class PolicyRun:
    scenario_id: str
    policy: str
    initial_loss: float
    final_loss: float
    final_prediction: tuple[float, ...]
    acquisition_cost: float
    query_count: int
    net_utility: float
    selected_candidates: tuple[str, ...]
    statuses: tuple[str, ...]
    stop_reason: str
    replan_reasons: tuple[str, ...]
    final_context: Mapping[str, Any]
    transition_events: tuple[str, ...]
    execution_entries: int
    evaluator_only: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": VERSION, "scenario_id": self.scenario_id, "policy": self.policy,
            "initial_loss": self.initial_loss, "final_loss": self.final_loss,
            "final_prediction": list(self.final_prediction),
            "acquisition_cost": self.acquisition_cost, "query_count": self.query_count,
            "net_utility": self.net_utility, "selected_candidates": list(self.selected_candidates),
            "statuses": list(self.statuses), "stop_reason": self.stop_reason,
            "replan_reasons": list(self.replan_reasons), "final_context": dict(self.final_context),
            "transition_events": list(self.transition_events), "execution_entries": self.execution_entries,
            "hidden_result_included": False,
            "evaluator_only": dict(self.evaluator_only),
        }


def _brier(prediction: tuple[float, ...], outcome: str) -> float:
    index = OUTCOMES.index(outcome)
    return math.fsum((probability - float(i == index)) ** 2 for i, probability in enumerate(prediction))


def _decision(candidate_id: str, board: StateQueryBoard, *, basis: str) -> QueryValueDecision:
    candidate = next(item for item in board.candidates if item.candidate_id == candidate_id)
    assessment = QueryValueAssessment.modeled(
        candidate_id, expected_value=candidate.cost + 0.001, cost=candidate.cost,
        reason=f"synthetic_policy:{basis}",
    )
    return QueryValueDecision(QUERY, candidate_id, basis, (assessment,), None)


def _query_envelope(
    scenario: Scenario, query: QuerySpec, decision: QueryValueDecision, *,
    previous_context: Mapping[str, Any], sequence: int, episode_id: str,
    execution_root: Path,
) -> tuple[QueryExecutionEnvelope, QueryExecutionEnvelope, ContextTransition]:
    # The synthetic tool request is issued before every return timestamp; the
    # nonce still distinguishes repeated calls in the execution ledger.
    requested_at = "2026-09-29T10:00:00Z"
    planned = QueryExecutionEnvelope.from_decision(
        decision, scenario.board((item.candidate_id for item in scenario.queries)),
        {"target_key": scenario.target_key, "state_variable": query.state_variable,
         "request_nonce": f"{scenario.scenario_id}-{sequence}"},
        tool_version="synthetic-context-tool-v1", expected_context_keys=(query.state_variable,),
        requested_at=requested_at, evidence_refs=(f"request:{scenario.scenario_id}:{sequence}",),
    )
    response = query.responses[scenario.private_response_labels[query.candidate_id]]
    completed = planned.complete(
        response.result_payload, context_updates=response.context_updates,
        evidence_refs=response.evidence_refs, observed_at=response.observed_at,
        operational_available_at=response.operational_available_at,
        captured_at=response.captured_at, status=response.status, error_code=response.error_code,
    )
    execution_ledger = QueryExecutionLedger(execution_root / "executions")
    execution_ledger.append((planned, completed))
    transition = completed.to_context_transition(
        episode_id=episode_id, sequence=sequence, previous_context=previous_context,
        captured_at=response.captured_at,
    )
    return planned, completed, transition


def run_policy(scenario: Scenario, policy: str) -> PolicyRun:
    if policy not in POLICIES:
        raise ValueError("Unsupported acquisition policy")
    with tempfile.TemporaryDirectory(prefix=f"footballmind-{scenario.scenario_id}-{policy}-") as folder:
        root = Path(folder)
        episode_id = f"{scenario.scenario_id}-{policy}"
        initial = ContextTransition.from_transition(
            episode_id=episode_id, target_key=scenario.target_key, sequence=1,
            event_type="INITIALIZED", previous_context={}, next_context=scenario.initial_context,
            trigger="scenario", reason="public context before information acquisition",
            prediction_cutoff=scenario.prediction_cutoff, captured_at="2026-09-29T09:59:00Z",
        )
        current_context = dict(scenario.initial_context)
        prediction = scenario.initial_prediction
        remaining_ids = [query.candidate_id for query in scenario.queries]
        selected: list[str] = []
        statuses: list[str] = []
        replan_reasons: list[str] = []
        transition_records: list[ContextTransition] = [initial]
        spent = 0.0
        sequence = 2
        stop_reason = "policy_exhausted"

        while remaining_ids:
            board = scenario.board(remaining_ids)
            if policy == "fixed_all":
                candidate_id = remaining_ids[0]
                basis = "fixed_all_information"
            elif policy == "first_query":
                candidate_id = remaining_ids[0]
                basis = "first_query_heuristic"
            else:
                assessments = [QueryValueAssessment.modeled(
                    query.candidate_id, expected_value=query.expected_value, cost=query.cost,
                    reason="synthetic_declared_return_model",
                ) for query in scenario.queries if query.candidate_id in remaining_ids]
                decision = QueryValueGate.decide(board, assessments, remaining_budget=1.0)
                if decision.action != QUERY:
                    certificate = decision.stop_certificate
                    stop_reason = certificate.reason if certificate else "voi_gate_stop"
                    break
                candidate_id = decision.candidate_id
                basis = decision.selection_basis
            decision = decision if policy == "voi_gate" else _decision(candidate_id, board, basis=basis)
            query = next(item for item in scenario.queries if item.candidate_id == candidate_id)
            planned, completed, _unused_transition = _query_envelope(
                scenario, query, decision, previous_context=current_context, sequence=sequence,
                episode_id=episode_id, execution_root=root,
            )
            selected_event = ContextTransition.from_transition(
                episode_id=episode_id, target_key=scenario.target_key, sequence=sequence,
                event_type="QUERY_SELECTED", previous_context=current_context,
                next_context=current_context, trigger="policy", reason=decision.selection_basis,
                prediction_cutoff=scenario.prediction_cutoff, captured_at=planned.requested_at,
                selected_candidate_id=candidate_id, action=QUERY, source_id=planned.tool_version,
                evidence_refs=planned.evidence_refs,
            )
            result_event = completed.to_context_transition(
                episode_id=episode_id, sequence=sequence + 1, previous_context=current_context,
                captured_at=query.responses[scenario.private_response_labels[candidate_id]].captured_at,
            )
            transition_records.extend((selected_event, result_event))
            current_context = dict(result_event.next_context)
            statuses.append(completed.status)
            selected.append(candidate_id)
            spent += query.cost
            if completed.success and query.responses[scenario.private_response_labels[candidate_id]].prediction is not None:
                prediction = query.responses[scenario.private_response_labels[candidate_id]].prediction  # type: ignore[assignment]
            if completed.replanning_required:
                replan_reasons.append(result_event.reason)
            remaining_ids.remove(candidate_id)
            sequence += 2
            if policy == "first_query":
                stop_reason = "first_query_policy_stop"
                break

        final_stop = ContextTransition.from_transition(
            episode_id=episode_id, target_key=scenario.target_key, sequence=sequence,
            event_type="STOPPED", previous_context=current_context, next_context=current_context,
            trigger="policy", reason=stop_reason, prediction_cutoff=scenario.prediction_cutoff,
            captured_at=f"2026-09-29T11:{min(sequence, 59):02d}:00Z", action=STOP,
        )
        transition_records.append(final_stop)
        transition_ledger = ContextTransitionLedger(root / "transitions")
        transition_ledger.append(tuple(transition_records))
        replay = transition_ledger.replay_episode(episode_id, target_key=scenario.target_key)
        initial_loss = _brier(scenario.initial_prediction, scenario.private_outcome)
        final_loss = _brier(prediction, scenario.private_outcome)
        return PolicyRun(
            scenario_id=scenario.scenario_id, policy=policy, initial_loss=initial_loss,
            final_loss=final_loss, final_prediction=tuple(prediction), acquisition_cost=spent,
            query_count=len(selected),
            net_utility=initial_loss - final_loss - spent, selected_candidates=tuple(selected),
            statuses=tuple(statuses), stop_reason=stop_reason, replan_reasons=tuple(replan_reasons),
            final_context=replay["final_context"], transition_events=tuple(item["event_type"] for item in replay["events"]),
            execution_entries=2 * len(selected),
            evaluator_only={"true_outcome": scenario.private_outcome,
                            "response_labels": dict(scenario.private_response_labels),
                            "interpretation": "Evaluator-only realization; excluded from policy inputs."},
        )


def _success(label: str, updates: Mapping[str, Any], prediction: tuple[float, ...], ref: str) -> ReturnSpec:
    return ReturnSpec(label, {"source_observation": ref}, updates, prediction, (f"raw:{ref}",),
                      observed_at="2026-09-29T10:01:00Z", operational_available_at="2026-09-29T10:01:01Z",
                      captured_at="2026-09-29T10:01:02Z")


def _late(label: str, updates: Mapping[str, Any], prediction: tuple[float, ...], ref: str) -> ReturnSpec:
    return ReturnSpec(label, {"source_observation": ref}, updates, prediction, (f"raw:{ref}",),
                      observed_at="2026-09-29T13:01:00Z", operational_available_at="2026-09-29T13:01:01Z",
                      captured_at="2026-09-29T13:01:02Z")


def _ambiguous(label: str, ref: str) -> ReturnSpec:
    return ReturnSpec(label, {"source_observation": ref}, {}, None, (f"raw:{ref}",),
                      captured_at="2026-09-29T10:01:02Z")


def build_scenarios() -> tuple[Scenario, ...]:
    initial = (0.45, 0.30, 0.25)
    improved = (0.65, 0.20, 0.15)
    context = {"fixture_id": "fixture-synthetic", "competition_id": "competition-synthetic",
               "kickoff_at": "2026-09-29T12:30:00+00:00"}

    def make(scenario_id: str, qualification: ReturnSpec, weather: ReturnSpec,
             *, outcome: str, qualification_value: float = .06) -> Scenario:
        # Weather appears first to make the first-query heuristic visibly different
        # from the value-gated policy; the fixed policy intentionally reads both.
        queries = (
            QuerySpec("weather-refresh", "weather_relevance", ("neutral", "severe"), .01, .005,
                      {"neutral": weather, "severe": weather}),
            QuerySpec("qualification-refresh", "qualification", ("already_qualified", "must_win"), .01, qualification_value,
                      {"already_qualified": qualification, "must_win": qualification}),
        )
        labels = {"weather-refresh": "neutral", "qualification-refresh": "already_qualified"}
        return Scenario(scenario_id, "match-synthetic", CUTOFF, context, initial, queries, labels, outcome)

    return (
        make("qualification-high-value", _success("already_qualified", {"qualification": "already_qualified"}, improved, "qualification-high"),
             _success("neutral", {"weather_relevance": "low"}, initial, "weather-low"), outcome="home_win"),
        make("weather-low-value", _success("already_qualified", {"qualification": "already_qualified"}, initial, "qualification-unused"),
             _success("neutral", {"weather_relevance": "low"}, initial, "weather-low"), outcome="draw", qualification_value=.005),
        make("qualification-late", _late("already_qualified", {"qualification": "already_qualified"}, improved, "qualification-late"),
             _success("neutral", {"weather_relevance": "low"}, initial, "weather-after-late"), outcome="away_win"),
        make("qualification-ambiguous", _ambiguous("already_qualified", "qualification-conflict"),
             _success("neutral", {"weather_relevance": "low"}, initial, "weather-after-ambiguous"), outcome="home_win"),
    )


def evaluate_all() -> dict[str, Any]:
    scenarios = build_scenarios()
    rows = []
    for scenario in scenarios:
        rows.append({
            "scenario": scenario.public_manifest(),
            "policies": {policy: run_policy(scenario, policy).as_dict() for policy in POLICIES},
        })
    return {"schema_version": VERSION, "scenarios": rows,
            "policy_order": list(POLICIES), "outcome_labels": list(OUTCOMES),
            "evaluator_only": {"scope": "synthetic mechanism evaluation", "truth_in_policy_view": False}}


__all__ = [
    "VERSION", "POLICIES", "OUTCOMES", "ReturnSpec", "QuerySpec", "Scenario", "PolicyRun",
    "build_scenarios", "run_policy", "evaluate_all",
]
