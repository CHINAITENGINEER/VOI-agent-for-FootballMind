"""The optional Agent planner shares actual execution costs and limits."""
from football_agent.engine import Workflow
from football_agent.policy import ToolPlanningPolicy
from football_agent.testing import SyntheticTools, TEST_MATCH_ID, TEST_CUTOFF
from football_agent.types import Budget, ToolResult


class PlanningFixture(SyntheticTools):
    def __init__(self, decision=None):
        super().__init__()
        self.decision = decision or {"action": "stop", "reason": "SYNTHETIC_PLANNER_STOP"}

    def estimate_cost(self, name, inputs):
        return 0.5 if name == "plan_next_action" else super().estimate_cost(name, inputs)

    def execute(self, name, inputs):
        if name == "plan_next_action":
            return ToolResult(data=self.decision, cost=0.5)
        return super().execute(name, inputs)


def test_planner_cost_and_decision_are_audited(tmp_path):
    workflow = Workflow(PlanningFixture(), tmp_path, ToolPlanningPolicy())
    run = workflow.start(TEST_MATCH_ID, TEST_CUTOFF)
    run = workflow.run_to_completion(run["run_id"])
    assert run["state"] == "STOPPED"
    assert run["reason"] == "SYNTHETIC_PLANNER_STOP"
    assert run["budget"]["spent_cost"] == 0.5
    assert run["budget"]["calls"] == 5
    assert any(e["data"].get("tool") == "plan_next_action" for e in run["events"])
    assert workflow.store.verify(run["run_id"])["status"] == "OK"


def test_planner_budget_rejected_before_execution(tmp_path):
    workflow = Workflow(PlanningFixture(), tmp_path, ToolPlanningPolicy())
    run = workflow.start(TEST_MATCH_ID, TEST_CUTOFF, Budget(max_cost=0.1))
    run = workflow.run_to_completion(run["run_id"])
    assert run["reason"] == "COST_BUDGET_EXHAUSTED"
    assert run["budget"]["calls"] == 4
    assert run["budget"]["spent_cost"] == 0


def test_invalid_agent_action_becomes_reviewable_failure(tmp_path):
    workflow = Workflow(PlanningFixture({"action": "invent_probability"}), tmp_path, ToolPlanningPolicy())
    run = workflow.start(TEST_MATCH_ID, TEST_CUTOFF)
    run = workflow.run_to_completion(run["run_id"])
    assert run["state"] == "DEFERRED"
    assert run["reason"].startswith("POLICY_ERROR_")
    assert run["budget"]["spent_cost"] == 0.5


def test_new_evidence_snapshot_does_not_present_previous_probability_as_current(tmp_path):
    workflow = Workflow(SyntheticTools(), tmp_path)
    run = workflow.start(TEST_MATCH_ID, TEST_CUTOFF)
    run = workflow.run_to_completion(run["run_id"])
    arrivals = [e for e in run["events"] if e["type"] == "evidence_arrived"]
    assert arrivals[-1]["snapshot"]["prediction"] is None
    assert arrivals[-1]["snapshot"]["last_prediction"] is not None
