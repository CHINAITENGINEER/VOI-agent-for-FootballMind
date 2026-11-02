from datetime import datetime, timedelta, timezone
from copy import deepcopy
import json
import pytest

from football_agent.ambiguity_value import FiniteAmbiguityDecisionModel
from football_agent.decision_loss import FiniteDecisionLossModel
from football_agent.decision_risk_experiment import build_selector, proposal, ConstructedTools, task, H1, H2
from football_agent.inquiry import InquiryWorkflow
from football_agent.inquiry_ambiguity import AmbiguityRiskSelector
from football_agent.inquiry_evidence_state import EvidenceStateProtocol
from football_agent.inquiry_selection import get_selector
from football_agent.inquiry_verify import verify


NOW = datetime(2026, 9, 27, 1, tzinfo=timezone.utc)


def model_identification_selector():
    base = build_selector(k_weight=1)
    models = {}
    for label, truth in (("assume_yes", 1), ("assume_no", 0)):
        config = base.model.manifest()
        config["prior"] = {k: float(k == f"K{truth}J1S1") for k in config["prior"]}
        models[label] = FiniteDecisionLossModel.from_manifest(config)
    return AmbiguityRiskSelector(FiniteAmbiguityDecisionModel(models), base.bindings,
        task_id=base.task_id, loss_budget=.2)


def empty_run(selector):
    return {"task": task(NOW, selector.binding), "documents": [], "receipts": [], "decisions": [],
            "hypotheses": [H1, H2], "budget": {"max_requests": 3, "http_attempts": 0}}


def test_shared_policy_identifies_model_while_each_point_model_would_stop():
    selector = model_identification_selector()
    for model in selector.model_set.models.values():
        assert model.evaluate(model.initial_state(), remaining_budget=.2)["action_id"] == "STOP"
    decision = selector.select(empty_run(selector), proposal(["fill_j", "verify_k"]))
    assert decision["candidate_id"] == "verify_k" and decision["llm_proposal"]["candidate_id"] == "fill_j"
    assert decision["expected_gain"] is None  # A worst-risk difference is not a single-model expected gain.
    assert decision["robust_risk_reduction"] > .16
    assert decision["terminal_policy_commitment"]["response_policy"]["no"] == "K_no__J_yes"


def test_json_configuration_round_trip_replays_the_same_public_model_set():
    selector = model_identification_selector()
    restored = get_selector(json.loads(json.dumps(selector.configuration(), sort_keys=True)))
    assert restored.public_summary(empty_run(selector)) == selector.public_summary(empty_run(selector))


@pytest.mark.parametrize("llm_limit", [1, 4])
def test_actual_query_settles_without_budget_or_second_provider_call(tmp_path, llm_limit):
    selector = model_identification_selector()
    class Provider:
        calls = 0
        def configuration(self): return {"provider": "constructed_shared_policy_test", "configured": True}
        def decide(self, public):
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("No provider is available after the committed query")
            return {"response": proposal(["fill_j", "verify_k"]), "receipt": {"real_provider_call": False}}
    provider = Provider()
    tools = ConstructedTools(tmp_path, {"verify_k": "no"}, clock=lambda: NOW)
    workflow = InquiryWorkflow(tmp_path/"runs", provider, tools, clock=lambda: NOW,
        protocol=EvidenceStateProtocol(), selector=selector)
    run = workflow.run(workflow.start(task(NOW, selector.binding), max_requests=3, max_llm_calls=llm_limit)["run_id"])
    assert tools.calls == ["https://example.com/verify_k"]
    last = run["decisions"][-1]
    assert last["kind"] == "stop" and last["llm_proposal"] is None
    assert last["decision_origin"] == "deterministic_frozen_policy"
    assert provider.calls == run["budget"]["llm_calls"] == 1
    assert last["terminal_decision"]["decision"] == "K_no__J_yes" and last["terminal_decision"]["policy_followed"]
    assert last["risk_assessment"]["state"]["plan"]["active_model_ids"] == ["assume_no"]
    assert last["risk_assessment"]["stop_certificate"] == "FROZEN_ONE_QUERY_TERMINAL_POLICY_EXECUTED"
    assert not last["risk_assessment"]["global_open_world_stop_proven"]
    assert run["budget"]["http_attempts"] == 1 and run["budget"]["max_requests"] == 3
    assert run["reason"] == "SCOPED_FROZEN_POLICY_EXECUTED"
    assert verify(tmp_path, run["run_id"])["frozen_terminal_decisions_replayed"] == 1


@pytest.mark.parametrize("interrupted_after", ["inquiry_acquisition_returned", "inquiry_frozen_terminal"])
def test_resume_persisted_return_or_terminal_does_not_call_provider_or_tool_again(tmp_path, interrupted_after):
    selector = model_identification_selector()
    class Provider:
        calls = 0
        def configuration(self): return {"provider": "constructed_recovery", "configured": True}
        def decide(self, public):
            self.calls += 1
            if self.calls > 1:
                raise AssertionError("The committed terminal policy needs no new proposal")
            return {"response": proposal(["verify_k"]), "receipt": {"real_provider_call": False}}
    provider = Provider()
    tools = ConstructedTools(tmp_path, {"verify_k": "no"}, clock=lambda: NOW)
    workflow = InquiryWorkflow(tmp_path/"runs", provider, tools, clock=lambda: NOW,
        protocol=EvidenceStateProtocol(), selector=selector)
    run = workflow.run(workflow.start(task(NOW, selector.binding), max_llm_calls=1)["run_id"])
    event = next(e for e in run["events"] if e["type"] == interrupted_after)
    prefix = {**deepcopy(event["snapshot"]), "events": deepcopy(run["events"][:event["seq"]])}
    workflow.store.save(prefix)
    resumed = workflow.run(run["run_id"])
    assert resumed["state"] == "STOPPED" and resumed["reason"] == "SCOPED_FROZEN_POLICY_EXECUTED"
    assert resumed["decisions"][-1]["terminal_decision"]["decision"] == "K_no__J_yes"
    assert provider.calls == 1 and len(tools.calls) == 1
    assert sum(e["type"] == "inquiry_frozen_terminal" for e in resumed["events"]) == 1
    assert verify(tmp_path, run["run_id"])["frozen_terminal_decisions_replayed"] == 1


def test_all_model_impossible_return_never_executes_placeholder_terminal(tmp_path):
    selector = model_identification_selector()
    class Provider:
        def configuration(self): return {"provider": "constructed_misspecification", "configured": True}
        def decide(self, public):
            answer = proposal(["verify_k"]) if not public["action_history"] else proposal([], kind="stop")
            return {"response": answer, "receipt": {"real_provider_call": False}}
    tools = ConstructedTools(tmp_path, {"verify_k": "failure"}, clock=lambda: NOW)
    workflow = InquiryWorkflow(tmp_path/"runs", Provider(), tools, clock=lambda: NOW,
        protocol=EvidenceStateProtocol(), selector=selector)
    run = workflow.run(workflow.start(task(NOW, selector.binding))["run_id"])
    final = run["decisions"][-1]
    assert final["risk_assessment"]["state"]["status"] == "MODEL_SET_MISSPECIFIED"
    assert final["terminal_decision"] is None and final["value_status"] == "UNSUPPORTED_OPEN_TEXT"
    assert run["state"] == "DEFERRED" and run["reason"] == "FROZEN_POLICY_SUPPORT_WITHHELD"
    assert run["budget"]["llm_calls"] == 1 and final["llm_proposal"] is None
    assert verify(tmp_path, run["run_id"])["frozen_terminal_decisions_replayed"] == 1


def test_late_actual_return_cannot_execute_frozen_conclusion(tmp_path):
    current = [NOW]
    selector = model_identification_selector()
    class Provider:
        def configuration(self): return {"provider": "constructed_late_return", "configured": True}
        def decide(self, public):
            return {"response": proposal(["verify_k"]), "receipt": {"real_provider_call": False}}
    class DelayedTools(ConstructedTools):
        def execute(self, action, **kwargs):
            current[0] = NOW + timedelta(hours=3)
            return super().execute(action, **kwargs)
    tools = DelayedTools(tmp_path, {"verify_k": "no"}, clock=lambda: current[0])
    workflow = InquiryWorkflow(tmp_path/"runs", Provider(), tools, clock=lambda: current[0],
        protocol=EvidenceStateProtocol(), selector=selector)
    run = workflow.run(workflow.start(task(NOW, selector.binding), max_llm_calls=1)["run_id"])
    assert run["reason"] == "LATE_ACQUISITION_RETURN"
    assert run["documents"][-1]["status"] == "LATE_RETURN"
    assert not any(d.get("terminal_decision") for d in run["decisions"])
    assert verify(tmp_path, run["run_id"])["frozen_terminal_decisions_replayed"] == 0


def test_absent_candidate_and_unknown_candidate_do_not_get_global_stop():
    selector = model_identification_selector()
    run = empty_run(selector)
    selected = selector.select(run, proposal([], kind="stop"))
    assert selected["kind"] == "defer"
    assert selected["risk_assessment"]["stop_certificate"] == "WITHHELD_VALUABLE_REGISTERED_ACTION_NOT_PROPOSED"
    raw = proposal(["verify_k"])
    raw["candidates"][0]["url"] = "https://example.com/unmodeled"
    result = selector.select(run, raw)
    assert result["risk_assessment"]["candidate_values"][0]["status"] == "UNSUPPORTED"
    assert result["expected_gain"] is None
