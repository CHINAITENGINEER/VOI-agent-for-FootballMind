"""Constructed live transitions test lineage and budgets, never network outcomes."""
from datetime import datetime, timedelta, timezone

from football_agent.engine import Workflow
from football_agent.types import Budget, ToolResult


class LiveFixture:
    mode = "test"

    def __init__(self, time_mode="live", result_status="OK"):
        self.timestamp = datetime.now(timezone.utc)
        self.cutoff = (self.timestamp - timedelta(hours=1)).isoformat()
        self.time_mode, self.result_status = time_mode, result_status
        self.refreshes = 0

    def capabilities(self):
        return {"mode": "test", "information_mode": "synthetic_test", "time_mode": self.time_mode}

    def estimate_cost(self, tool, inputs):
        return 0.25 if tool == "refresh_information" else 0

    def estimate_requests(self, tool, inputs):
        return int(tool == "refresh_information")

    def evidence(self, fresh=False):
        observed = self.timestamp - timedelta(seconds=1) if fresh else self.timestamp - timedelta(hours=2)
        return {"evidence_id": "new" if fresh else "old", "version": "2" if fresh else "1", "source": "synthetic",
                "kind": "fixture", "match_id": "synthetic:live", "content_hash": "new" if fresh else "old",
                "content": {"value": int(fresh)}, "raw_ref": "synthetic://new" if fresh else "synthetic://old",
                "observed_at": observed.isoformat(), "available_at": observed.isoformat(), "conflict_key": "fixture"}

    def execute(self, tool, inputs):
        if tool == "query_matches":
            return ToolResult(data=[{"match_id": "synthetic:live", "kickoff": (self.timestamp + timedelta(hours=3)).isoformat(),
                                    "fixture_available_at": self.evidence()["available_at"], "status": "scheduled"}])
        if tool == "read_evidence":
            fresh = self.refreshes and datetime.fromisoformat(inputs["prediction_cutoff"]) >= self.timestamp
            return ToolResult(data=[self.evidence(bool(fresh))])
        if tool == "get_features":
            return ToolResult(data={"feature_version": "synthetic-v1", "match_id": inputs["match_id"],
                                    "prediction_cutoff": inputs["prediction_cutoff"], "input_manifest": ["synthetic://feature"]})
        if tool == "predict":
            value = inputs["evidence"][-1]["content"]["value"]
            return ToolResult(data={"match_id": inputs["match_id"], "prediction_cutoff": inputs["prediction_cutoff"],
                "home_win_probability": 0.4 + value * 0.1, "draw_probability": 0.3, "away_win_probability": 0.3 - value * 0.1,
                "model_version": "synthetic-model", "dataset_version": "synthetic", "input_manifest": {"source": "fixture"},
                "information_mode": "synthetic_test"})
        if tool == "refresh_information":
            self.refreshes += 1
            assert inputs["max_requests"] >= 1
            assert inputs["request_id"]
            return ToolResult(self.result_status, data={"evidence": [self.evidence(True)],
                "decision_cutoff": self.timestamp.isoformat(), "observation_refs": ["synthetic://new"],
                "content_changed": True}, cost=0.25, requests=1, retry_after_seconds=0)
        raise ValueError(tool)


def predicted(workflow, tools, budget=None):
    run = workflow.start("synthetic:live", tools.cutoff, budget)
    for _ in range(4):
        run = workflow.step(run["run_id"])
    assert run["state"] == "PREDICTED"
    return run


def test_live_child_keeps_parent_and_inherits_cumulative_budget(tmp_path):
    tools = LiveFixture()
    workflow = Workflow(tools, tmp_path)
    old = predicted(workflow, tools)
    parent = workflow.step(old["run_id"], action="query", topic="odds")
    assert parent["reason"] == "LIVE_CUTOFF_ADVANCED"
    assert parent["prediction"] == old["prediction"]
    assert parent["evidence"] == old["evidence"]
    child = workflow.get_run(parent["child_run_id"])
    assert child["budget"]["calls"] == 5
    assert child["budget"]["requests"] == 1
    assert child["budget"]["spent_cost"] == 0.25
    assert child["prediction_cutoff"] != parent["prediction_cutoff"]
    for _ in range(4):
        child = workflow.step(child["run_id"])
    assert child["prediction"]["home_win_probability"] == 0.5
    assert child["budget"]["calls"] == 9
    assert child["evidence"][0]["raw_ref"] == "synthetic://new"
    blocked = workflow.step(child["run_id"], action="query", topic="odds")
    assert blocked["reason"] == "REPEATED_TOPIC_WITHOUT_NEW_EVIDENCE"
    assert tools.refreshes == 1
    assert workflow.store.verify(parent["run_id"])["status"] == "OK"
    assert workflow.store.verify(child["run_id"])["status"] == "OK"


def test_live_child_does_not_reset_exhausted_call_budget(tmp_path):
    tools = LiveFixture()
    workflow = Workflow(tools, tmp_path)
    run = predicted(workflow, tools, Budget(max_calls=5))
    parent = workflow.step(run["run_id"], action="query", topic="odds")
    child = workflow.step(parent["child_run_id"])
    assert child["reason"] == "CALL_BUDGET_EXHAUSTED"
    assert child["budget"]["calls"] == 5


def test_request_budget_checked_before_source_call(tmp_path):
    tools = LiveFixture()
    workflow = Workflow(tools, tmp_path)
    run = predicted(workflow, tools, Budget(max_requests=0))
    run = workflow.step(run["run_id"], action="query", topic="odds")
    assert run["reason"] == "REQUEST_BUDGET_EXHAUSTED"
    assert tools.refreshes == 0


def test_live_failure_and_retry_remain_in_original_budget(tmp_path):
    tools = LiveFixture(result_status="RATE_LIMITED")
    workflow = Workflow(tools, tmp_path)
    run = predicted(workflow, tools, Budget(max_requests=1))
    run = workflow.step(run["run_id"], action="query", topic="odds")
    assert run["state"] == "WAITING"
    assert not run.get("child_run_id")
    workflow.step(run["run_id"], action="resume")
    run = workflow.step(run["run_id"], action="query", topic="odds")
    assert run["reason"] == "REQUEST_BUDGET_EXHAUSTED"
    assert tools.refreshes == 1


def test_fixed_cutoff_excludes_live_arrivals(tmp_path):
    tools = LiveFixture(time_mode="archive_replay")
    workflow = Workflow(tools, tmp_path)
    before = predicted(workflow, tools)
    after = workflow.step(before["run_id"], action="query", topic="odds")
    assert after["prediction"] == before["prediction"]
    assert after["evidence"] == before["evidence"]
    assert after["reason"] == "NO_NEW_VALID_EVIDENCE"
    assert not after.get("child_run_id")

