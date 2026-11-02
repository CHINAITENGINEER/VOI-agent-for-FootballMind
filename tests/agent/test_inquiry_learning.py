from datetime import datetime, timezone

from football_agent.decision_risk_experiment import build_selector, proposal, ConstructedTools, task
from football_agent.inquiry import InquiryWorkflow
from football_agent.inquiry_evidence_state import EvidenceStateProtocol
from football_agent.inquiry_exploration import UniformInquiryExploration
from football_agent.inquiry_selection import LearningExplorationSelector, get_selector
from football_agent.inquiry_verify import verify


NOW = datetime(2026, 9, 27, 1, tzinfo=timezone.utc)


def test_real_execution_binds_conditional_probability_and_hides_seed_from_provider(tmp_path):
    seen = []
    selector = LearningExplorationSelector(UniformInquiryExploration(seed=123456789), max_exploration_requests=2)
    class Provider:
        def configuration(self): return {"provider": "constructed_learning_exploration", "configured": True}
        def decide(self, public):
            seen.append(public)
            answer = proposal(["fill_j", "verify_k"]) if not public["action_history"] else proposal([], kind="stop")
            return {"response": answer, "receipt": {"real_provider_call": False}}
    tools = ConstructedTools(tmp_path, {"fill_j": "yes", "verify_k": "no"}, clock=lambda: NOW)
    workflow = InquiryWorkflow(tmp_path/"runs", Provider(), tools, clock=lambda: NOW,
        protocol=EvidenceStateProtocol(), selector=selector)
    run = workflow.run(workflow.start(task(NOW, build_selector()))["run_id"])
    sampled = run["decisions"][0]
    record = sampled["exploration_record"]
    assert record["probabilities"] == {"fill_j": .5, "verify_k": .5}
    assert tools.calls == ["https://example.com/"+record["chosen_id"]]
    assert record["selected_p"] == .5 and sampled["expected_gain"] is None
    assert "123456789" not in str(seen) and "seed" not in seen[0]["scoped_decision_value"]["exploration"]
    assert run["decisions"][-1]["kind"] == "stop" and run["decisions"][-1]["exploration_record"] is None
    assert verify(tmp_path, run["run_id"])["real_completed_llm_turns"] == 0
    restored = get_selector(selector.configuration())
    assert restored.configuration() == selector.configuration()


def test_exploration_cap_does_not_become_a_value_stop(tmp_path):
    selector = LearningExplorationSelector(UniformInquiryExploration(seed=0), max_exploration_requests=1)
    class Provider:
        def configuration(self): return {"provider": "constructed_budget", "configured": True}
        def decide(self, public):
            action = "verify_k" if not public["action_history"] else "fill_j"
            return {"response": proposal([action]), "receipt": {"real_provider_call": False}}
    tools = ConstructedTools(tmp_path, {"verify_k": "no"}, clock=lambda: NOW)
    workflow = InquiryWorkflow(tmp_path/"runs", Provider(), tools, clock=lambda: NOW,
        protocol=EvidenceStateProtocol(), selector=selector)
    run = workflow.run(workflow.start(task(NOW, build_selector()), max_requests=3)["run_id"])
    assert run["state"] == "DEFERRED" and run["budget"]["http_attempts"] == 1
    assert run["decisions"][-1]["expected_gain"] is None
    assert run["decisions"][-1]["risk_assessment"]["stop_certificate"] is None
    assert verify(tmp_path, run["run_id"])["new_network_requests"] == 0
