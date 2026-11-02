"""Mechanism tests use constructed history, never claimed live policy gains."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from football_agent.autonomous_context_loop import LoopReturn, run_autonomous_context_loop
from football_agent.empirical_context_policy import (
    FrozenEmpiricalContextPolicy, frozen_loop_configuration, run_frozen_empirical_context_loop,
)
from football_agent.query_execution_envelope import QueryExecutionLedger
from football_agent.query_value_contract import DEFER, QUERY, STOP, QueryValueAssessment, QueryValueDecision
from football_agent.state_query_contract import StateHypothesis, StateQueryBoard, StateQueryCandidate
from football_agent.voi_evidence_bridge import VoIEvidenceLink
from football_research.evaluation.policy_experiment_protocol import manifest_hash

CUTOFF = "2026-09-29T12:00:00Z"
AS_OF = "2026-09-28T10:00:00Z"
TARGET = "new-evaluation-match"
KEYS = ("qualification", "tactical_intent")
CONTEXTS = (
    {"qualification": None, "tactical_intent": None},
    {"qualification": "already_qualified", "tactical_intent": None},
    {"qualification": "already_qualified", "tactical_intent": "defensive"},
)
VALUES = ((.95, .65, .05, 0), (0, 0, .85, 0), (0, 0, 0, 0))
VARIABLES = ("qualification", "lineup", "tactical_intent", "weather")


def board():
    return StateQueryBoard(target_key=TARGET, prediction_cutoff=CUTOFF,
        hypotheses=[StateHypothesis(f"{v}-h", TARGET, v, ("known", "unknown")) for v in VARIABLES],
        candidates=[StateQueryCandidate(v, TARGET, CUTOFF, v, "public-source", "read", .01,
                                        action_id=f"read-{v}") for v in VARIABLES])


def routes():
    return [{"candidate_id": v, "route_kind": "context" if v == "qualification" else "domain",
             "required_context_keys": (), "provides_context_keys": (v,)} for v in VARIABLES]


def history():
    links, contexts, available = [], {}, {}
    for state_index, context in enumerate(CONTEXTS):
        for variable_index, variable in enumerate(VARIABLES):
            for i in range(30):
                identity = f"{state_index}-{variable}-{i}"
                value = VALUES[state_index][variable_index]
                item = VoIEvidenceLink(
                    link_id=f"link-{identity}", trace_id=f"trace-{identity}", episode_id=f"episode-{identity}",
                    target_key=f"historical-{state_index}-{i}", request_id=f"request-{identity}",
                    action_id=f"read-{variable}", candidate_id=variable, evidence_entry_id=f"entry-{identity}",
                    evidence_availability="available", evidence_quality="VALIDATED",
                    requested_at="2026-09-27T08:00:00Z", prediction_cutoff="2026-09-27T09:00:00Z",
                    source_id="shared-public-provider", source_version="v1", dependency_key=f"match-{state_index}-{i}",
                    expected_information_value=value, observed_risk_reduction=value,
                    calibration_error=0.0, status="linked_available",
                )
                links.append(item)
                contexts[item.link_id] = deepcopy(context)
                available[item.link_id] = "2026-09-27T15:00:00Z"
    return links, contexts, available


def policy(*, transform=None, availability=None):
    links, contexts, available = history()
    if transform:
        links = [transform(item) for item in links]
    if availability is not None:
        available = availability(available)
    return FrozenEmpiricalContextPolicy(
        routes={v: {"source_id": "shared-public-provider", "source_version": "v1",
                    "action_id": f"read-{v}", "cost": .01} for v in VARIABLES},
        context_keys=KEYS, links=links, context_by_link_id=contexts,
        available_at_by_link_id=available, calibration_as_of=AS_OF)


def executor(calls, root=None, mode="success"):
    def execute(plan, context):
        calls.append(plan.candidate_id)
        if root is not None:
            entries = QueryExecutionLedger(root / "executions").entries()
            assert entries[-1].status == "PLANNED"
            assert entries[-1].query_id == plan.query_id
        if mode == "exception":
            raise RuntimeError("private provider text")
        key = plan.candidate_id
        value = {"qualification": "already_qualified", "tactical_intent": "defensive"}.get(key, "known")
        stamp = "2026-09-29T12:01:00Z" if mode == "late" else CUTOFF
        capture = "2026-09-29T12:01:00Z" if mode == "late_capture" else stamp
        if mode == "missing_capture":
            capture = None
        payload = {"source_observation": key} if mode != "label" else {"outcome": "home"}
        return LoopReturn(key, {key: value}, payload, (f"evidence:{key}",),
                          observed_at=stamp, operational_available_at=stamp, captured_at=capture)
    return execute


def run(*, value_policy=None, budget=.10, initial=None, max_steps=8, mode="success", root=None):
    calls = []
    p = value_policy or policy()
    result = run_autonomous_context_loop(
        board=board(), initial_context=initial or {"fixture_id": "unseen-id", "competition_id": "league"},
        routes=routes(), episode_id="mechanism-only", root=root, budget=budget, max_steps=max_steps,
        decision_provider=p, execute=executor(calls, root, mode), tool_version="synthetic-adapter-v1")
    return result, p, calls


def freeze(p, budget=.10, max_steps=8):
    config = frozen_loop_configuration(board=board(), routes=routes(), policy=p,
                                       budget=budget, max_steps=max_steps, tool_version="synthetic-adapter-v1")
    payload = {"parameters": {"autonomous_context_loop": config},
               "thresholds": {"min_independent_dependencies": 5, "confidence": .1, "stop_net_value": 0.0}}
    return {**payload, "parameter_hash": manifest_hash(payload), "frozen": True,
            "frozen_at": "2026-09-28T11:00:00Z", "fit_partition": "train",
            "selection_partition": "validation", "evaluation_partition": "test"}


def test_updated_context_changes_next_query_and_then_stops(tmp_path):
    result, p, calls = run(root=tmp_path)
    assert calls == ["qualification", "tactical_intent"]
    assert result.selected_candidates == tuple(calls)
    assert result.final_action == STOP
    assert result.remaining_budget == pytest.approx(.08)
    estimates = p.assessments_history()
    initial = {row["candidate_id"]: row for row in estimates[0]["estimates"]}
    following = {row["candidate_id"]: row for row in estimates[1]["estimates"]}
    assert initial["lineup"]["expected_information_value"] > .01
    assert following["lineup"]["expected_information_value"] == 0
    assert estimates[0]["context_key"] != estimates[1]["context_key"]
    assert estimates[-1]["decision"]["stop_certificate"]["global_claim"] is False


@pytest.mark.parametrize("budget,expected", [(0, []), (.005, []), (.015, ["qualification"])])
def test_budget_limits_real_executor_calls(budget, expected):
    result, _, calls = run(budget=budget)
    assert calls == expected
    assert result.final_action == DEFER
    assert result.declared_cost_spent <= budget


def test_novel_context_defers_without_query_or_global_average_transfer():
    result, _, calls = run(initial={"qualification": "draw_sufficient"})
    assert calls == []
    assert result.final_action == DEFER
    assert all(row["value_status"] == "VALUE_UNMEASURED" for row in result.decisions[0]["decision"]["assessments"])


@pytest.mark.parametrize("transform", [
    lambda r: replace(r, target_key=TARGET),
    lambda r: replace(r, source_version="old"),
    lambda r: replace(r, action_id="different-query"),
    lambda r: replace(r, candidate_id="different-candidate"),
    lambda r: replace(r, dependency_key="one-repeated-match"),
])
def test_incompatible_or_nonindependent_history_never_authorizes_query(transform):
    result, _, calls = run(value_policy=policy(transform=transform))
    assert calls == []
    assert result.final_action == DEFER


@pytest.mark.parametrize("availability", [
    lambda rows: {},
    lambda rows: {key: "2026-09-29T10:00:00Z" for key in rows},
    lambda rows: {key: "2026-09-26T10:00:00Z" for key in rows},
])
def test_calibration_value_must_have_valid_pre_freeze_availability(availability):
    result, _, calls = run(value_policy=policy(availability=availability))
    assert calls == []
    assert result.final_action == DEFER


@pytest.mark.parametrize("mode", ["exception", "label"])
def test_failure_is_audited_charged_and_leaves_context_unchanged(tmp_path, mode):
    result, _, calls = run(budget=.01, mode=mode, root=tmp_path)
    assert calls == ["qualification"]
    assert result.statuses == ("FAILED",)
    assert result.declared_cost_spent == pytest.approx(.01)
    assert "qualification" not in result.final_context
    assert "REPLANNED" in result.transition_events
    assert result.final_action == DEFER
    assert "private provider text" not in json.dumps(result.as_dict())


@pytest.mark.parametrize("mode", ["late", "late_capture"])
def test_late_result_does_not_start_another_query_at_an_earlier_clock(tmp_path, mode):
    result, _, calls = run(mode=mode, root=tmp_path)
    assert calls == ["qualification"]
    assert result.statuses == ("LATE",)
    assert result.final_action == DEFER
    assert "qualification" not in result.final_context
    assert result.remaining_budget == pytest.approx(.09)
    planned, late = QueryExecutionLedger(tmp_path / "executions").entries()
    assert planned.status == "PLANNED"
    assert late.status == "LATE"
    assert not late.context_updates
    assert late.captured_at == "2026-09-29T12:01:00+00:00"
    if mode == "late_capture":
        assert late.observed_at == "2026-09-29T12:00:00+00:00"
        assert late.operational_available_at == "2026-09-29T12:00:00+00:00"


def test_missing_capture_stops_dynamic_policy_without_learning_returned_context(tmp_path):
    result, p, calls = run(mode="missing_capture", root=tmp_path)
    assert calls == ["qualification"]
    assert result.statuses == ("FAILED",)
    assert result.final_action == DEFER
    assert result.final_context == {"fixture_id": "unseen-id", "competition_id": "league"}
    assert result.remaining_budget == pytest.approx(.09)
    assert result.declared_cost_spent == pytest.approx(.01)
    assert len(p.assessments_history()) == 1
    assert result.transition_events == ("INITIALIZED", "QUERY_SELECTED", "REPLANNED", "STOPPED")
    planned, failed = QueryExecutionLedger(tmp_path / "executions").entries()
    assert planned.status == "PLANNED"
    assert failed.status == "FAILED"
    assert failed.observed_at is None
    assert failed.operational_available_at is None
    assert not failed.context_updates


def test_step_limit_is_defer_not_a_false_exhaustion_certificate():
    result, _, calls = run(max_steps=1)
    assert calls == ["qualification"]
    assert result.final_action == DEFER
    assert result.stop_reason == "maximum_steps_reached"


def test_frozen_wrapper_binds_used_configuration_and_thresholds(tmp_path):
    p, calls = policy(), []
    manifest = freeze(p)
    # Exercise the actual JSON serialization boundary.
    manifest = json.loads(json.dumps(manifest))
    result = run_frozen_empirical_context_loop(
        board=board(), initial_context={"fixture_id": "m"}, routes=routes(), policy=p,
        freeze_manifest=manifest, execute=executor(calls), episode_id="frozen-fixture",
        root=tmp_path, budget=.10, tool_version="synthetic-adapter-v1")
    assert result["policy_parameter_hash"] == manifest["parameter_hash"]
    assert calls == ["qualification", "tactical_intent"]
    assert result["independent_confirmation"] is False


@pytest.mark.parametrize("change", ["budget", "hash", "threshold", "before-calibration", "after-cutoff"])
def test_freeze_mismatch_is_detected_before_any_tool_call(change):
    p, calls = policy(), []
    manifest = freeze(p)
    budget = .11 if change == "budget" else .10
    if change == "hash":
        manifest["parameter_hash"] = "0" * 64
    elif change == "threshold":
        manifest["thresholds"]["confidence"] = .2
        manifest["parameter_hash"] = manifest_hash({key: manifest[key] for key in ("parameters", "thresholds")})
    elif change == "before-calibration":
        manifest["frozen_at"] = "2026-09-28T09:00:00Z"
    elif change == "after-cutoff":
        manifest["frozen_at"] = "2026-09-30T00:00:00Z"
    with pytest.raises(ValueError):
        run_frozen_empirical_context_loop(
            board=board(), initial_context={"fixture_id": "m"}, routes=routes(), policy=p,
            freeze_manifest=manifest, execute=executor(calls), episode_id="bad-freeze",
            budget=budget, tool_version="synthetic-adapter-v1")
    assert calls == []


def test_arbitrary_provider_cannot_bypass_gate_budget():
    calls = []
    def invalid(board, context, remaining):
        return QueryValueDecision(QUERY, "qualification", "forced", (), None)
    with pytest.raises(ValueError, match="must match"):
        run_autonomous_context_loop(board=board(), initial_context={}, routes=routes(),
            episode_id="bad-provider", budget=0, decision_provider=invalid, execute=executor(calls))
    assert calls == []


def test_calling_same_episode_does_not_repeat_external_acquisition(tmp_path):
    first, _, calls = run(root=tmp_path)
    assert len(calls) == 2
    second_calls = []
    with pytest.raises(ValueError, match="append-only"):
        run_autonomous_context_loop(board=board(), initial_context={"fixture_id": "unseen-id", "competition_id": "league"},
            routes=routes(), episode_id="mechanism-only", root=tmp_path, budget=.10,
            decision_provider=policy(), execute=executor(second_calls))
    assert second_calls == []


def test_history_with_partial_context_is_not_silently_treated_as_unknown():
    p = policy()
    # Fresh construction using incomplete historical context is permitted as
    # sparse evidence, but it does not manufacture exact-context support.
    links, contexts, available = history()
    for row in contexts.values():
        row.pop("tactical_intent")
    cfg = p.configuration()
    sparse = FrozenEmpiricalContextPolicy(routes=cfg["routes"], context_keys=KEYS, links=links,
        context_by_link_id=contexts, available_at_by_link_id=available, calibration_as_of=AS_OF)
    result, _, calls = run(value_policy=sparse)
    assert calls == []
    assert result.final_action == DEFER


def test_current_context_is_projected_without_unique_match_identity():
    p = policy()
    first = p(board(), {"fixture_id": "one"}, .1)
    second = p(board(), {"fixture_id": "another"}, .1)
    assert first == second
    assert set(p.assessments_history()[0]["context"]) == set(KEYS)


def test_snapshot_mutation_is_detected_before_calling_a_tool():
    p = policy()
    p._confidence = .2
    with pytest.raises(ValueError, match="configuration changed"):
        run(value_policy=p)


def test_each_frozen_episode_returns_only_its_own_value_trace():
    p = policy()
    manifest = freeze(p)
    for index in range(2):
        result = run_frozen_empirical_context_loop(
            board=board(), initial_context={"fixture_id": str(index)}, routes=routes(), policy=p,
            freeze_manifest=manifest, execute=executor([]), episode_id=f"separate-{index}",
            budget=.1, tool_version="synthetic-adapter-v1")
        assert len(result["value_assessments"]) == 3


def test_public_callback_board_does_not_retain_private_registry():
    p = policy()
    seen = []
    def inspect(public_board, context, budget):
        seen.append(public_board.registry)
        return p(public_board, context, budget)
    result, _, _ = run(value_policy=inspect)
    assert result.selected_candidates == ("qualification", "tactical_intent")
    assert seen and all(item is None for item in seen)


def test_changed_initial_context_does_not_repeat_existing_episode(tmp_path):
    run(root=tmp_path)
    calls = []
    with pytest.raises(ValueError, match="episode already exists"):
        run_autonomous_context_loop(board=board(), initial_context={"fixture_id": "changed"},
            routes=routes(), episode_id="mechanism-only", root=tmp_path, budget=.1,
            decision_provider=policy(), execute=executor(calls))
    assert calls == []


def test_interrupted_episode_keeps_planned_intent_and_prevents_blind_retry(tmp_path):
    calls = []
    def interrupted(plan, context):
        calls.append(plan.candidate_id)
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        run_autonomous_context_loop(board=board(), initial_context={}, routes=routes(),
            episode_id="interrupted", root=tmp_path, budget=.1,
            decision_provider=policy(), execute=interrupted)
    assert QueryExecutionLedger(tmp_path / "executions").entries()[-1].status == "PLANNED"
    with pytest.raises(ValueError, match="episode already exists"):
        run_autonomous_context_loop(board=board(), initial_context={}, routes=routes(),
            episode_id="interrupted", root=tmp_path, budget=.1,
            decision_provider=policy(), execute=executor(calls))
    assert calls == ["qualification"]
