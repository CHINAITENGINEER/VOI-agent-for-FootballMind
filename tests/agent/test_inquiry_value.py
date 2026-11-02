"""Risk arithmetic and executed information acquisition, not forecast scores."""
from copy import deepcopy
from datetime import datetime, timezone
import json

import pytest

from football_agent.decision_loss import FiniteDecisionLossModel
from football_agent.audit import digest
from football_agent.decision_risk_experiment import (
    build_selector, proposal, ConstructedTools, make_parent, task, H1, H2, TEXT_K,
)
from football_agent.decision_value import ObservationTool, simulate_episode
from football_agent.inquiry import InquiryWorkflow
from football_agent.inquiry_evidence_state import EvidenceStateProtocol
from football_agent.inquiry_value import DecisionRiskSelector
from football_agent.inquiry_verify import verify


NOW = datetime(2026, 9, 27, 1, tzinfo=timezone.utc)


def conflict_run(selector):
    return {"task": task(NOW, selector), "hypotheses": [H1, H2], "documents": [
        {"document_id": "old-a", "url": selector.bindings["report_a"]["url"], "status": "OK", "text": TEXT_K["yes"]},
        {"document_id": "old-b", "url": selector.bindings["report_b"]["url"], "status": "OK", "text": TEXT_K["no"]}],
        "receipts": [], "budget": {"max_requests": 3, "http_attempts": 0}}


def value(summary, action):
    return next(item for item in summary["plan"]["candidates"] if item["action_id"] == action)


@pytest.mark.parametrize("weight,choice,risk,verify_gain", [(.5, "fill_j", .3, .08), (3, "verify_k", .8, .48)])
def test_same_information_changes_action_with_terminal_loss(weight, choice, risk, verify_gain):
    selector = build_selector(weight)
    run = conflict_run(selector)
    summary = selector.public_summary(run)
    assert summary["terminal_decision"]["decision"] == "K_defer__J_defer"
    assert summary["plan"]["current_risk"] == pytest.approx(risk)
    # Independently calculated: successful exact answer removes the abstention
    # loss, failure retains it. S succeeds .8; independent J query succeeds .9.
    assert value(summary, "verify_k")["immediate_evsi"] == pytest.approx(verify_gain)
    assert value(summary, "fill_j")["immediate_evsi"] == pytest.approx(.9*.2)
    assert value(summary, "verify_k")["immediate_net_value"] == pytest.approx(verify_gain-.03)
    decision = selector.select(run, proposal(["verify_k", "fill_j"], chosen="fill_j"))
    assert decision["candidate_id"] == choice and decision["llm_proposal"]["candidate_id"] == "fill_j"
    assert decision["value_status"] == "DECLARED_ASSUMPTIONS_ONLY"
    assert not decision["risk_assessment"]["global_open_world_stop_proven"]


def test_failure_updates_shared_availability_and_switches_to_independent_input():
    selector = build_selector()
    run = conflict_run(selector)
    assert value(selector.public_summary(run), "refresh_k")["immediate_evsi"] == pytest.approx(.48)
    run["documents"].append({"document_id": "failed", "url": selector.bindings["verify_k"]["url"], "status": "FAILED", "text": ""})
    run["receipts"].append({"document_id": "failed"})
    run["budget"]["http_attempts"] = 1
    summary = selector.public_summary(run)
    assert summary["spent_loss_penalty"] == pytest.approx(.03)
    refresh = value(summary, "refresh_k")
    assert refresh["feasible"] and refresh["immediate_evsi"] == pytest.approx(0)
    assert next(b for b in refresh["branches"] if b["observation"] == "failure")["probability"] == pytest.approx(1)
    assert selector.select(run, proposal(["refresh_k", "fill_j"], chosen="refresh_k"))["candidate_id"] == "fill_j"


@pytest.mark.parametrize("token", ["yes", "no"])
def test_either_direction_reduces_risk_and_scoped_stop_keeps_unknowns(token):
    selector = build_selector()
    run = conflict_run(selector)
    for action, outcome in (("verify_k", "failure"), ("fill_j", token)):
        binding = selector.bindings[action]
        run["documents"].append({"document_id": action, "url": binding["url"],
            "status": "FAILED" if outcome == "failure" else "OK", "text": binding["markers"].get(outcome, "")})
        run["receipts"].append({"document_id": action})
    run["budget"]["http_attempts"] = 2
    summary = selector.public_summary(run)
    assert summary["terminal_decision"]["decision"] == f"K_defer__J_{token}"
    assert summary["terminal_decision"]["expected_loss"] == pytest.approx(.6)
    chosen = selector.select(run, proposal(["refresh_k"]))
    assert chosen["kind"] == "stop"
    assert chosen["risk_assessment"]["stop_certificate"] == "NO_POSITIVE_ONE_STEP_VALUE_IN_REGISTERED_MODEL"
    assert not chosen["risk_assessment"]["global_open_world_stop_proven"]


@pytest.mark.parametrize("mutation", ["text", "new_document", "wrong_task", "duplicate", "late"])
def test_unmodeled_or_invalid_observation_disables_numeric_selection(mutation):
    selector = build_selector()
    run = conflict_run(selector)
    if mutation == "text": run["documents"][0]["text"] += " An unmodeled qualifier."
    if mutation == "new_document": run["documents"][0]["url"] = "https://example.com/new"
    if mutation == "wrong_task": run["task"]["task_id"] = "other"
    if mutation == "duplicate": run["documents"].append(deepcopy(run["documents"][0]))
    if mutation == "late": run["documents"][0]["status"] = "LATE_RETURN"
    chosen = selector.select(run, proposal(["fill_j"]))
    assert chosen["selection_basis"] == "llm_bounded_exploration" and chosen["expected_gain"] is None
    assert chosen["risk_assessment"]["state"]["status"] == "UNSUPPORTED_MODEL_STATE"


def test_unmodeled_candidate_is_unknown_not_zero_and_blocks_stop_claim():
    selector = build_selector()
    selector.loss_budget = 0
    answer = proposal(["fill_j"], kind="stop")
    answer["candidates"][0]["url"] = "https://example.com/unknown"
    chosen = selector.select(conflict_run(selector), answer)
    assert chosen["kind"] == "defer" and chosen["expected_gain"] is None
    assert chosen["risk_assessment"]["stop_certificate"] == "WITHHELD_UNMODELED_CANDIDATES"
    assert chosen["risk_assessment"]["candidate_values"][0]["expected_gain"] is None


def test_missing_proposal_and_exhausted_budget_are_not_zero_value():
    selector = build_selector()
    run = conflict_run(selector)
    chosen = selector.select(run, proposal([], kind="stop"))
    assert chosen["kind"] == "defer"
    assert chosen["risk_assessment"]["stop_certificate"] == "WITHHELD_VALUABLE_REGISTERED_ACTION_NOT_PROPOSED"
    run["budget"]["http_attempts"] = 3
    chosen = selector.select(run, proposal(["verify_k"]))
    assert chosen["kind"] == "defer" and chosen["expected_gain"] is None


@pytest.mark.parametrize("field", ["scope_key", "statement", "required_evidence"])
def test_same_url_does_not_authorize_a_different_question(field):
    selector = build_selector()
    run = conflict_run(selector)
    run["hypotheses"] = deepcopy(run["hypotheses"])
    run["hypotheses"][0][field] = ["roster"] if field == "required_evidence" else "other"
    answer = proposal(["verify_k"])
    chosen = selector.select(run, answer)
    assert chosen["risk_assessment"]["candidate_values"][0]["status"] == "UNSUPPORTED"


def test_configuration_json_round_trip_preserves_full_value_state():
    selector = build_selector()
    restored = DecisionRiskSelector.from_configuration(json.loads(json.dumps(selector.configuration(), sort_keys=True)))
    assert restored.public_summary(conflict_run(selector)) == selector.public_summary(conflict_run(selector))
    with pytest.raises(ValueError, match="Prediction simulation"):
        simulate_episode(selector.model, seed=1, budget=.1)


@pytest.mark.parametrize("bad", [True, -1, float("nan"), float("inf")])
def test_terminal_loss_rejects_invalid_rows(bad):
    with pytest.raises(ValueError):
        FiniteDecisionLossModel({"t": (1, 0)}, {"t": 1}, {}, outcome_labels=("yes", "no"),
            decision_losses={"yes": (0, bad)}, cost_unit="declared_loss")


def test_information_can_reduce_entropy_without_changing_optimal_decision():
    model = FiniteDecisionLossModel({"a": (1, 0), "b": (0, 1)}, {"a": .5, "b": .5},
        {"signal": ObservationTool(("a", "b"), {"a": (.6, .4), "b": (.4, .6)}, .01)},
        outcome_labels=("a", "b"), decision_losses={"a": (0, 1), "b": (1, 0), "defer": (.2, .2)}, cost_unit="report_loss")
    state = model.initial_state()
    result = model.evaluate(state, remaining_budget=.1)
    assert model.target_mutual_information(state, "signal") > 0
    assert result["candidates"][0]["immediate_evsi"] == pytest.approx(0)
    assert result["action_id"] == "STOP" and result["terminal_decision_if_stopping"]["decision"] == "defer"


def test_unsupported_state_preserves_qualitative_stop_label(tmp_path):
    selector = build_selector()
    class Provider:
        def configuration(self): return {"provider": "scripted_unknown_scope", "configured": True}
        def decide(self, public):
            return {"response": proposal([], kind="stop"), "receipt": {"real_provider_call": False}}
    tools = ConstructedTools(tmp_path, {}, clock=lambda: NOW)
    workflow = InquiryWorkflow(tmp_path/"runs", Provider(), tools, clock=lambda: NOW,
        protocol=EvidenceStateProtocol(), selector=selector)
    other_task = task(NOW, selector)
    other_task["task_id"] = "a_task_outside_the_model"
    run = workflow.run(workflow.start(other_task)["run_id"])
    assert run["reason"] == "LLM_QUALITATIVE_STOP"
    assert run["decisions"][-1]["expected_gain"] is None
    assert not run["decisions"][-1]["risk_assessment"]["global_open_world_stop_proven"]
    assert not tools.calls
    assert verify(tmp_path, run["run_id"])["numeric_value_scope"] == "declared_assumptions_not_calibrated"


def test_full_execution_uses_computed_value_over_raw_proposal_then_replays(tmp_path):
    parent = make_parent(tmp_path/"parent", clock=lambda: NOW)
    selector = build_selector()
    class Provider:
        def configuration(self): return {"provider": "scripted_adversarial_proposal", "configured": True}
        def decide(self, public):
            actions = (["fill_j", "verify_k"] if not public["action_history"] else
                       ["refresh_k", "fill_j"] if len(public["action_history"]) == 1 else ["refresh_k"])
            return {"response": proposal(actions), "receipt": {"real_provider_call": False}}
    root = tmp_path/"child"
    tools = ConstructedTools(root, {"verify_k": "failure", "fill_j": "yes", "refresh_k": "failure"}, clock=lambda: NOW)
    workflow = InquiryWorkflow(root/"runs", Provider(), tools, clock=lambda: NOW,
        protocol=EvidenceStateProtocol(), selector=selector)
    run = workflow.start(task(NOW, selector), max_requests=3, max_llm_calls=4)
    workflow.inherit_evidence(run["run_id"], parent_root=tmp_path/"parent", parent_run_id=parent["run_id"],
        parent_head_hash=parent["events"][-1]["event_hash"], document_ids=[d["document_id"] for d in parent["documents"]])
    run = workflow.run(run["run_id"])
    assert run["state"] == "STOPPED", run["reason"]
    assert tools.calls == ["https://example.com/verify_k", "https://example.com/fill_j"]
    assert [d["llm_proposal"]["candidate_id"] for d in run["decisions"]] == ["fill_j", "refresh_k", "refresh_k"]
    assert [d["candidate_id"] for d in run["decisions"]] == ["verify_k", "fill_j", None]
    assert run["budget"]["http_attempts"] == 2 < run["budget"]["max_requests"]
    proof = verify(root, run["run_id"])
    assert proof["scoped_risk_decisions_replayed"] == 3 and proof["new_network_requests"] == 0
    assert proof["real_completed_llm_turns"] == 0
    workflow.export(run["run_id"], root/"report.md")
    assert "构造传输调用" in (root/"report.md").read_text(encoding="utf-8")
    # A freshly hashed trace still must prove that the request executed the
    # mathematical choice, rather than the different raw model proposal.
    intent = next(e for e in run["events"] if e["type"] == "inquiry_acquisition_intent")
    intent["data"]["selected_candidate"] = run["decisions"][0]["candidates"][0]
    previous = None
    for event in run["events"]:
        event["previous_hash"] = previous
        event["event_hash"] = digest({k: v for k, v in event.items() if k != "event_hash"})
        previous = event["event_hash"]
    workflow.store.save(run)
    with pytest.raises(ValueError, match="Executed candidate differs"):
        verify(root, run["run_id"])
