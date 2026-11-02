"""Terminal decision changes, evidence attribution and distribution-free limits."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from itertools import product

import pytest
from jsonschema import ValidationError

from football_agent.audit import digest
from football_agent.decision_risk_experiment import H1, H2, ConstructedTools, proposal
from football_agent.inquiry import InquiryWorkflow
from football_agent.inquiry_continuation import ContinuationPolicy
from football_agent.inquiry_hypotheses import get_protocol
from football_agent.inquiry_impact import (ACTIONS, STATES, DecisionImpactProtocol,
    decision_ceiling, terminal_state, validate_problem)
from football_agent.inquiry_verify import verify


NOW = datetime(2026, 9, 27, 4, tzinfo=timezone.utc)
LOSS = {"report_supported": {"positive": 0, "negative": 1},
        "report_challenged": {"positive": 1, "negative": 0},
        "withhold": {"positive": .2, "negative": .2}}
PROBLEM = {"task_id": "constructed-impact", "decision_id": "source-report-K", "target": H1,
    "semantics": "source_position_not_world_truth",
    "state_definitions": {"positive": "The scoped source assertion is positive.", "negative": "The scoped source assertion is negative."},
    "loss": LOSS, "loss_basis": "Declared test reporting preferences, not measured football cost."}
TASK = {"task_id": PROBLEM["task_id"], "objective": "Choose information for a source-position report.",
    "deadline": (NOW+timedelta(hours=1)).isoformat(), "initial_facts": [],
    "seed_urls": ["https://example.com/"+x for x in ("report_a", "report_b", "fill_j")]}


def response(actions=(), *, chosen=None, kind=None, claims=()):
    result = proposal(list(actions), chosen=chosen, kind=kind or ("query" if actions else "defer"))
    result["claim_updates"] = deepcopy(list(claims))
    result["rule_scope_bindings"] = []
    result["choice_review"] = {"alternatives": [{"candidate_id": c["candidate_id"],
        "information_advantage": "May change the designated source report.", "source_limitations": "Unknown semantics and availability.",
        "evidence_document_ids": [], "not_selected_reason": None if c["candidate_id"] == result["decision"]["candidate_id"] else "Leave unobserved."}
        for c in result["candidates"]], "tradeoff": "Compare decisions rather than input counts.", "missing_routes": "Not exhaustive."}
    for candidate in result["candidates"]:
        candidate["outcome_forecast"] = {"status": "abstain", "probabilities": None,
            "basis_document_ids": [], "reason": "No learned or elicited probabilities in this test."}
        candidate["terminal_effect"] = {"route": "indirect" if candidate["candidate_id"] == "fill_j" else "direct_evidence",
            "could_change_to": ["report_supported", "report_challenged"],
            "condition": "A scoped direct source statement about Player K.", "further_information_needed": "Read and inspect the actual source."}
    return result


def claim(document, *, identity="claim-K", hypothesis=H1, relation="supports", other_doc=None):
    citations = [{"document_id": document["document_id"], "quote": document["text"]}]
    if other_doc:
        citations.append({"document_id": other_doc["document_id"], "quote": other_doc["text"]})
    return {"claim_id": identity, "question_id": hypothesis["question_id"], "status": "source_claim",
        "text": "The supplied source states the scoped personnel position.", "citations": citations,
        "evidence": {k: hypothesis[k] for k in ("subject", "scope", "scope_key")} | {"kind": hypothesis["required_evidence"][0]},
        "hypothesis_links": [{"hypothesis_id": hypothesis["hypothesis_id"], "relation": relation}]}


def setup(root, responses, *, returns=None, problem=None, clock=lambda: NOW):
    inputs = []
    class Provider:
        def configuration(self): return {"provider": "constructed-decision-impact"}
        def decide(self, public):
            inputs.append(deepcopy(public))
            item = responses[len(inputs)-1]
            return {"response": item(public) if callable(item) else deepcopy(item), "receipt": {"real_provider_call": False}}
    tools = ConstructedTools(root, returns or {"report_a": "yes", "report_b": "no", "fill_j": "yes"}, clock=clock)
    protocol = DecisionImpactProtocol(problem or PROBLEM)
    workflow = InquiryWorkflow(root/"runs", Provider(), tools, clock=clock, protocol=protocol,
        continuation=ContinuationPolicy(), source_catalog=[])
    return workflow, tools, inputs


def test_bound_does_not_rank_queries_or_measure_risk():
    assert decision_ceiling(LOSS, "withhold")["all_action_improvement_ceiling"] == .2
    assert decision_ceiling(LOSS, "report_supported")["all_action_improvement_ceiling"] == 1
    # Enumerate return-dependent actions and arbitrary laws; no fitted prior is used.
    for current in ACTIONS:
        bound = decision_ceiling(LOSS, current)["all_action_improvement_ceiling"]
        for p, q0, q1, policy in product((0, .3, 1), (0, .7, 1), (0, .2, 1), product(ACTIONS, repeat=2)):
            gain = 0
            for state, mass, q in zip(STATES, (p, 1-p), (q0, q1)):
                after = q*LOSS[policy[0]][state]+(1-q)*LOSS[policy[1]][state]
                gain += mass*(LOSS[current][state]-after)
            assert gain <= bound+1e-12
    # Same upper bound permits both positive and negative actual net values.
    assert .001*.2-.1 < 0 < .9*.2-.1


@pytest.mark.parametrize("bad", [-1, float("inf"), float("nan"), True])
def test_invalid_loss_values_rejected(bad):
    problem = deepcopy(PROBLEM)
    problem["loss"]["withhold"]["positive"] = bad
    with pytest.raises((ValueError, ValidationError)):
        validate_problem(problem)


def test_real_execution_shape_freezes_after_extraction_and_failure_preserves_answer(tmp_path):
    def second(public):
        return response(["report_b"], claims=[claim(public["documents"][0])])
    workflow, tools, inputs = setup(tmp_path, [response(["report_a"]), second, response()],
        returns={"report_a": "yes", "report_b": "failure"})
    run = workflow.run(workflow.start(TASK, max_requests=2, max_llm_calls=3)["run_id"])
    assert run["terminal_state"]["action"] == "report_supported"
    first, second = run["decision_impact_reviews"]
    assert first["action_changed"] and first["attribution"] == "change_with_attributable_target_claims"
    assert first["sole_document_target_claim_ids"] == ["claim-K"]
    assert second["decision_before"]["action"] == "report_supported"
    assert not second["action_changed"] and second["sole_document_target_claim_ids"] == []
    assert run["receipts"][1]["inquiry_contract"]["terminal_contract"]["state_before"] == second["decision_before"]
    assert inputs[-1]["terminal_state"]["action"] == "report_supported"
    assert run["decision_ceiling"] == inputs[-1]["decision_ceiling"]
    assert run["decision_ceiling"]["all_action_improvement_ceiling"] == 1
    assert first["actual_decision_utility"] is None and second["causal_risk_reduction"] is None
    assert len(tools.calls) == 2 and verify(tmp_path, run["run_id"])["decisions_verified"] == 3
    workflow.export(run["run_id"], tmp_path/"report.md")
    assert "固定终端问题" in (tmp_path/"report.md").read_text(encoding="utf-8")


def test_opposing_signals_withhold_without_voting(tmp_path):
    def extract_a(public): return response(["report_b"], claims=[claim(public["documents"][0])])
    def extract_b(public): return response(claims=[claim(public["documents"][-1], identity="opposite", relation="challenges")])
    workflow, _, _ = setup(tmp_path, [response(["report_a"]), extract_a, extract_b])
    run = workflow.run(workflow.start(TASK, max_requests=2, max_llm_calls=3)["run_id"])
    assert run["terminal_state"]["action"] == "withhold"
    assert run["terminal_state"]["source_status"] == "mixed_source_signals"
    assert run["decision_impact_reviews"][-1]["action_changed"]
    assert verify(tmp_path, run["run_id"])["decisions_verified"] == 3


@pytest.mark.parametrize("mixed", [False, True])
def test_late_old_or_joint_claim_not_attributed_only_to_new_query(tmp_path, mixed):
    def delayed(public):
        docs = public["documents"]
        return response(claims=[claim(docs[0], other_doc=docs[-1] if mixed else None)])
    workflow, _, _ = setup(tmp_path, [response(["report_a"]), response(["report_b"]), delayed])
    run = workflow.run(workflow.start(TASK, max_requests=2, max_llm_calls=3)["run_id"])
    review = run["decision_impact_reviews"][-1]
    assert review["action_changed"] and review["attribution"] == "joint_or_other_evidence_change"
    assert review["sole_document_target_claim_ids"] == [] and review["other_new_target_claim_ids"] == ["claim-K"]
    assert not run["decision_impact_reviews"][0]["action_changed"]


def test_support_of_another_target_is_not_support_of_terminal_target(tmp_path):
    def extract(public): return response(claims=[claim(public["documents"][0], hypothesis=H2)])
    workflow, _, _ = setup(tmp_path, [response(["fill_j"]), extract])
    run = workflow.run(workflow.start(TASK, max_requests=1, max_llm_calls=2)["run_id"])
    assert run["return_reviews"][0]["outcome"] == "source_support"
    assert run["terminal_state"]["action"] == "withhold"
    assert run["decision_impact_reviews"][0]["sole_document_target_claim_ids"] == []
    assert run["decisions"][0]["hypothesis_assessment"]["decision_impact"]["candidate_effects"][0]["query_value"] is None


@pytest.mark.parametrize("change", ["opposite_statement", "wrong_id", "wrong_task", "search_direct", "different_target_direct"])
def test_terminal_semantic_endpoint_and_route_checks(tmp_path, change):
    workflow, _, _ = setup(tmp_path, [])
    public = workflow.public(workflow.start(TASK))
    reply = response(["fill_j"] if change == "different_target_direct" else ["report_a"])
    if change == "opposite_statement": reply["hypotheses"][0]["statement"] = "Player K does not start in fixture M."
    elif change == "wrong_id": reply["hypotheses"][0]["hypothesis_id"] = "other-hypothesis"
    elif change == "wrong_task": public["task"]["task_id"] = "another-task"
    elif change == "search_direct": reply["candidates"][0].update(tool="search_web", url=None, query="site:example.com Player K")
    else: reply["candidates"][0]["terminal_effect"]["route"] = "direct_evidence"
    with pytest.raises(ValueError): workflow.protocol.validate(reply, public)


def test_empty_illustrative_action_set_does_not_zero_or_filter_indirect_query(tmp_path):
    reply = response(["fill_j"])
    reply["candidates"][0]["terminal_effect"]["could_change_to"] = []
    workflow, tools, _ = setup(tmp_path, [reply, response()])
    run = workflow.run(workflow.start(TASK, max_requests=1, max_llm_calls=2)["run_id"])
    assert tools.calls == ["https://example.com/fill_j"]
    effect = run["decisions"][0]["hypothesis_assessment"]["decision_impact"]["candidate_effects"][0]
    assert effect["query_value"] is None and not run["decision_impact_reviews"][0]["action_changed"]


def test_registry_requires_explicit_problem_and_preserves_old_protocol():
    assert get_protocol("decision-impact-inquiry-v1", decision_problem=PROBLEM).decision_problem == PROBLEM
    with pytest.raises(ValidationError): get_protocol("decision-impact-inquiry-v1")
    with pytest.raises(ValueError): get_protocol("rule-scope-inquiry-v1", decision_problem=PROBLEM)
    assert "terminal_effect" not in get_protocol("rule-scope-inquiry-v1").schema["properties"]["candidates"]["items"]["properties"]


def test_rehashed_terminal_projection_injection_rejected(tmp_path):
    workflow, _, _ = setup(tmp_path, [response(["report_a"]), response()])
    run = workflow.run(workflow.start(TASK, max_requests=1, max_llm_calls=2)["run_id"])
    event = next(e for e in run["events"] if e["type"] == "inquiry_acquisition_returned")
    event["snapshot"]["terminal_state"]["action"] = "report_supported"
    previous = None
    for e in run["events"]:
        e["previous_hash"] = previous
        e["event_hash"] = digest({k:v for k,v in e.items() if k != "event_hash"})
        previous = e["event_hash"]
    workflow.store.save(run)
    with pytest.raises(ValueError, match="projection prefix"):
        verify(tmp_path, run["run_id"])
