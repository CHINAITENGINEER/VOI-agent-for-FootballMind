"""Black-box regressions for deterministic, auditable Agent execution.

The tools in this module are explicitly synthetic. Their fixed probabilities
exercise workflow transitions only and are not football forecasts.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from football_agent.engine import Workflow
from football_agent.audit import RunStore
from football_agent.policy import CallbackPolicy, ToolPlanningPolicy
from football_agent.types import Budget, Decision, ToolResult


MATCH_ID = "synthetic-match-1"
CUTOFF = "2026-09-19T12:00:00Z"


def evidence(evidence_id="synthetic-evidence-1", **changes):
    item = {
        "evidence_id": evidence_id,
        "match_id": MATCH_ID,
        "version": "1",
        "source": "synthetic-source",
        "kind": "odds",
        "topic": "odds",
        "observed_at": "2026-09-19T10:00:00Z",
        "available_at": "2026-09-19T10:00:00Z",
        "content_hash": "synthetic-hash-1",
        "content": {"note": "Explicit synthetic fixture; no live football data."},
        "raw_ref": "fixture://synthetic-evidence-1",
    }
    item.update(changes)
    return item


class ScriptedTools:
    mode = "test"

    def __init__(self, initial=None, refreshed=None, failures=None, cost=0.1):
        self.initial = [evidence()] if initial is None else initial
        self.refreshed = [] if refreshed is None else refreshed
        self.failures = failures or {}
        self.cost = cost
        self.calls = []
        self.predictions = 0

    def capabilities(self):
        capabilities = {
            name: {"ready": True, "mode": "test"}
            for name in (
                "query_matches", "read_evidence", "get_features", "predict",
                "refresh_information", "get_run_records",
            )
        }
        capabilities["information_mode"] = "synthetic_test"
        return capabilities

    def estimate_cost(self, name, inputs):
        return self.cost

    def execute(self, name, inputs):
        self.calls.append((name, deepcopy(inputs)))
        if name in self.failures:
            response = self.failures[name]
            if callable(response):
                response = response(inputs)
            if isinstance(response, Exception):
                raise response
            return response
        if name == "query_matches":
            result = [{
                "match_id": MATCH_ID,
                "kickoff": "2026-09-19T18:00:00Z",
                "fixture_available_at": "2026-09-18T12:00:00Z",
                "home": "Synthetic Home", "away": "Synthetic Away",
            }]
        elif name == "read_evidence":
            result = deepcopy(self.initial)
        elif name == "get_features":
            result = {
                "match_id": MATCH_ID,
                "prediction_cutoff": CUTOFF,
                "version": "synthetic-features-v1",
                "feature_version": "synthetic-features-v1",
                "features": {"synthetic_input": len(self.calls)},
                "input_manifest": ["fixture://synthetic-feature-input"],
            }
        elif name == "predict":
            self.predictions += 1
            probability = 0.4 if self.predictions == 1 else 0.5
            result = {
                "match_id": MATCH_ID,
                "prediction_cutoff": CUTOFF,
                "home_win_probability": probability,
                "draw_probability": 0.3,
                "away_win_probability": 1 - probability - 0.3,
                "model_version": "synthetic-model-v1",
                "dataset_version": "synthetic-dataset-v1",
                "input_manifest": ["fixture://synthetic-model-input"],
                "information_mode": "synthetic_test",
            }
        elif name == "refresh_information":
            result = {"evidence": deepcopy(self.refreshed)}
        else:
            raise AssertionError(f"Unexpected tool {name}")
        return ToolResult(status="OK", data=result, cost=self.cost)


def workflow(tmp_path, tools=None, budget=None):
    tools = ScriptedTools() if tools is None else tools
    runner = Workflow(tools, store_dir=tmp_path)
    initial = runner.start(MATCH_ID, CUTOFF, budget=budget)
    return runner, initial["run_id"], tools


def advance_to_prediction(runner, run_id):
    current = runner.get_run(run_id)
    for _ in range(8):
        if current["state"] in {"PREDICTED", "WAITING", "STOPPED", "DEFERRED"}:
            break
        current = runner.step(run_id)
    return current


def call_names(tools):
    return [name for name, _ in tools.calls]


def test_initialization_does_not_call_tools(tmp_path):
    runner, run_id, tools = workflow(tmp_path)
    current = runner.get_run(run_id)
    assert current["state"] == "INITIALIZED"
    assert current["match_id"] == MATCH_ID
    assert current["test_mode"] is True
    assert current["mode"] == "test"
    assert current["prediction"] is None
    assert current["budget"]["calls"] == 0
    assert tools.calls == []


def test_happy_path_uses_model_output_then_stops_on_empty_refresh(tmp_path):
    runner, run_id, tools = workflow(tmp_path)
    current = runner.run_to_completion(run_id)
    assert call_names(tools)[:4] == [
        "query_matches", "read_evidence", "get_features", "predict",
    ]
    assert current["state"] == "STOPPED"
    assert current["prediction"]["home_win_probability"] == 0.4
    assert current["prediction"]["model_version"] == "synthetic-model-v1"
    assert current["budget"]["queries"] == 1
    assert tools.predictions == 1
    assert current["reason"]


@pytest.mark.parametrize("field", ["available_at", "observed_at"])
def test_future_evidence_is_excluded_at_both_visibility_times(tmp_path, field):
    tools = ScriptedTools(initial=[
        evidence(),
        evidence("future-evidence", content_hash="future-hash", **{
            field: "2026-09-19T12:00:01Z",
        }),
    ])
    runner, run_id, _ = workflow(tmp_path, tools)
    current = runner.run_to_completion(run_id)
    assert {item["evidence_id"] for item in current["evidence"]} == {
        "synthetic-evidence-1",
    }
    modeling_inputs = [inputs for name, inputs in tools.calls if name in {"get_features", "predict"}]
    assert "future-evidence" not in json.dumps(modeling_inputs)


def test_evidence_from_another_match_is_excluded(tmp_path):
    tools = ScriptedTools(initial=[
        evidence(),
        evidence("foreign-evidence", match_id="another-match", content_hash="foreign-hash"),
    ])
    runner, run_id, _ = workflow(tmp_path, tools)
    current = runner.run_to_completion(run_id)
    assert len(current["evidence"]) == 1
    assert current["evidence"][0]["match_id"] == MATCH_ID


def test_same_source_duplicate_does_not_repredict(tmp_path):
    tools = ScriptedTools(refreshed=[evidence("duplicate-with-different-id")])
    runner, run_id, _ = workflow(tmp_path, tools)
    current = runner.run_to_completion(run_id)
    assert current["state"] == "STOPPED"
    assert len(current["evidence"]) == 1
    assert tools.predictions == 1
    assert call_names(tools).count("refresh_information") == 1


def test_valid_new_evidence_repredicts_at_original_cutoff(tmp_path):
    tools = ScriptedTools(refreshed=[evidence(
        "new-evidence", content_hash="new-hash",
        available_at="2026-09-19T11:00:00Z",
        observed_at="2026-09-19T11:00:00Z",
    )])
    runner, run_id, _ = workflow(tmp_path, tools)
    current = runner.run_to_completion(run_id)
    assert tools.predictions == 2
    assert current["prediction"]["home_win_probability"] == 0.5
    assert len(current["evidence"]) == 2
    prediction_inputs = [inputs for name, inputs in tools.calls if name == "predict"]
    assert len(prediction_inputs) == 2
    assert prediction_inputs[0]["prediction_cutoff"] == prediction_inputs[1]["prediction_cutoff"]
    assert call_names(tools).count("refresh_information") == 2


def test_refresh_cannot_import_evidence_after_fixed_cutoff(tmp_path):
    tools = ScriptedTools(refreshed=[evidence(
        "too-late", content_hash="too-late-hash",
        available_at="2026-09-19T12:00:01Z",
        observed_at="2026-09-19T12:00:01Z",
    )])
    runner, run_id, _ = workflow(tmp_path, tools)
    current = runner.run_to_completion(run_id)
    assert tools.predictions == 1
    assert current["state"] == "STOPPED"
    assert len(current["evidence"]) == 1


def test_expired_evidence_does_not_reach_current_evidence(tmp_path):
    tools = ScriptedTools(initial=[evidence(), evidence(
        "expired-evidence", content_hash="expired-hash",
        expires_at="2026-09-19T11:00:00Z",
    )])
    runner, run_id, _ = workflow(tmp_path, tools)
    current = runner.run_to_completion(run_id)
    assert [item["evidence_id"] for item in current["evidence"]] == [
        "synthetic-evidence-1",
    ]


def test_conflicting_information_defers_without_current_probability(tmp_path):
    tools = ScriptedTools(initial=[
        evidence(conflict_key="starting-goalkeeper"),
        evidence("conflicting-evidence", source="another-synthetic-source",
                 content_hash="conflicting-hash", conflict_key="starting-goalkeeper"),
    ])
    runner, run_id, _ = workflow(tmp_path, tools)
    current = runner.run_to_completion(run_id)
    assert current["state"] == "DEFERRED"
    assert current["prediction"] is None
    assert tools.predictions == 0
    assert current["reason"]


def test_conflict_after_prediction_retains_auditable_previous_result(tmp_path):
    tools = ScriptedTools(
        initial=[evidence(conflict_key="starting-goalkeeper")],
        refreshed=[evidence("conflicting-evidence", source="another-synthetic-source",
                            content_hash="conflicting-hash", conflict_key="starting-goalkeeper")],
    )
    runner, run_id, _ = workflow(tmp_path, tools)
    current = runner.run_to_completion(run_id)
    assert current["state"] == "DEFERRED"
    assert current["prediction"] is None
    assert current["last_prediction"]["home_win_probability"] == 0.4
    assert tools.predictions == 1


def test_empty_match_query_never_produces_probability(tmp_path):
    tools = ScriptedTools(failures={"query_matches": ToolResult(status="OK", data=[])})
    runner, run_id, _ = workflow(tmp_path, tools)
    current = runner.run_to_completion(run_id)
    assert current["state"] == "DEFERRED"
    assert current["prediction"] is None
    assert tools.predictions == 0


@pytest.mark.parametrize("status", ["MISSING_CONFIG", "UNAVAILABLE", "ERROR"])
def test_model_unavailability_does_not_make_up_probabilities(tmp_path, status):
    tools = ScriptedTools(failures={
        "predict": ToolResult(status=status, error="Synthetic model unavailable", cost=0.2),
    })
    runner, run_id, _ = workflow(tmp_path, tools)
    current = runner.run_to_completion(run_id)
    assert current["state"] == "DEFERRED"
    assert current["prediction"] is None
    assert current["budget"]["spent_cost"] == pytest.approx(0.5)
    assert status in json.dumps(current["events"])


def test_call_budget_prevents_over_budget_tool_invocation(tmp_path):
    runner, run_id, tools = workflow(tmp_path, budget=Budget(max_calls=3))
    current = runner.run_to_completion(run_id)
    assert len(tools.calls) == 3
    assert tools.predictions == 0
    assert current["budget"]["remaining_calls"] == 0
    assert current["state"] == "DEFERRED"
    assert current["prediction"] is None


def test_cost_estimate_reserves_budget_before_tool_execution(tmp_path):
    runner, run_id, tools = workflow(
        tmp_path, ScriptedTools(cost=0.25), Budget(max_cost=0.3),
    )
    current = runner.run_to_completion(run_id)
    assert len(tools.calls) == 1
    assert current["budget"]["spent_cost"] == pytest.approx(0.25)
    assert current["budget"]["remaining_cost"] == pytest.approx(0.05)
    assert current["state"] == "DEFERRED"


def test_query_budget_zero_allows_initial_prediction_but_no_refresh(tmp_path):
    runner, run_id, tools = workflow(tmp_path, budget=Budget(max_queries=0))
    current = runner.run_to_completion(run_id)
    assert tools.predictions == 1
    assert "refresh_information" not in call_names(tools)
    assert current["budget"]["queries"] == 0
    assert current["state"] == "STOPPED"


def test_terminal_run_is_idempotent(tmp_path):
    runner, run_id, tools = workflow(tmp_path)
    first = runner.run_to_completion(run_id)
    count = len(tools.calls)
    event_count = len(first["events"])
    again = runner.step(run_id)
    third = runner.run_to_completion(run_id)
    assert len(tools.calls) == count
    assert len(again["events"]) == event_count
    assert len(third["events"]) == event_count
    assert third["prediction"] == first["prediction"]


def test_rate_limit_enters_wait_without_automatic_busy_retry(tmp_path):
    tools = ScriptedTools(failures={
        "refresh_information": ToolResult(
            status="RATE_LIMITED", error="Synthetic HTTP 429", cost=0.15,
            retry_after_seconds=60,
        ),
    })
    runner, run_id, _ = workflow(tmp_path, tools)
    current = runner.run_to_completion(run_id)
    assert current["state"] == "WAITING"
    assert current["budget"]["spent_cost"] == pytest.approx(0.55)
    count = len(tools.calls)
    runner.step(run_id)
    again = runner.run_to_completion(run_id)
    assert again["state"] == "WAITING"
    assert len(tools.calls) == count


def test_explicit_resume_recovers_after_transient_rate_limit(tmp_path):
    tools = ScriptedTools(failures={
        "refresh_information": ToolResult(
            status="RATE_LIMITED", error="Synthetic HTTP 429", retry_after_seconds=0,
        ),
    })
    runner, run_id, _ = workflow(tmp_path, tools)
    assert runner.run_to_completion(run_id)["state"] == "WAITING"
    del tools.failures["refresh_information"]
    runner.step(run_id, action="resume")
    current = runner.run_to_completion(run_id)
    assert current["state"] == "STOPPED"
    assert call_names(tools).count("refresh_information") == 2


def test_timeout_is_recorded_and_estimated_cost_is_charged(tmp_path):
    tools = ScriptedTools(failures={"read_evidence": TimeoutError("synthetic timeout")})
    runner, run_id, _ = workflow(tmp_path, tools)
    current = runner.run_to_completion(run_id)
    assert current["state"] in {"WAITING", "DEFERRED"}
    assert current["prediction"] is None
    assert current["budget"]["spent_cost"] == pytest.approx(0.2)
    assert "TIMEOUT" in json.dumps(current["events"])


def test_latency_budget_is_measured_for_failed_calls(tmp_path):
    def delayed_failure(inputs):
        time.sleep(0.01)
        return ToolResult(status="ERROR", error="Measured synthetic failure", cost=0.05)

    tools = ScriptedTools(failures={"read_evidence": delayed_failure})
    runner, run_id, _ = workflow(tmp_path, tools)
    current = runner.run_to_completion(run_id)
    assert current["budget"]["elapsed_ms"] >= 10
    assert current["budget"]["spent_cost"] == pytest.approx(0.15)


def test_exhausted_elapsed_budget_prevents_first_tool_call(tmp_path):
    runner, run_id, tools = workflow(tmp_path, budget=Budget(max_elapsed_seconds=0))
    current = runner.run_to_completion(run_id)
    assert tools.calls == []
    assert current["state"] == "DEFERRED"
    assert current["prediction"] is None


def test_tool_deadline_returns_timeout_and_keeps_late_result_out_of_state(tmp_path):
    def delayed_result(inputs):
        time.sleep(0.12)
        return ToolResult(status="OK", data=[evidence()], cost=0.1)

    tools = ScriptedTools(failures={"read_evidence": delayed_result})
    runner, run_id, _ = workflow(
        tmp_path, tools, Budget(max_tool_seconds=0.01),
    )
    current = runner.run_to_completion(run_id)
    assert current["state"] in {"WAITING", "DEFERRED"}
    assert current["prediction"] is None
    assert "TIMEOUT" in json.dumps(current["events"])
    evidence_before = deepcopy(current["evidence"])
    time.sleep(0.15)
    assert runner.get_run(run_id)["evidence"] == evidence_before


def test_run_archive_contains_ordered_replay_and_tool_io_references(tmp_path):
    runner, run_id, tools = workflow(tmp_path)
    current = runner.run_to_completion(run_id)
    events = current["events"]
    sequences = [event["seq"] for event in events]
    assert sequences == sorted(set(sequences))
    assert all(event["at"] and event["type"] for event in events)
    assert all("snapshot" in event for event in events)
    json_files = list(Path(tmp_path).rglob("*.json"))
    assert json_files, "A completed run must persist a checkable JSON archive."
    archived = [json.loads(path.read_text(encoding="utf-8")) for path in json_files]
    assert any(isinstance(item, dict) and item.get("run_id") == run_id for item in archived)
    serialized_events = json.dumps(events)
    assert "input_ref" in serialized_events
    assert "output_ref" in serialized_events
    assert "synthetic-model-v1" in serialized_events
    assert "synthetic-features-v1" in serialized_events
    assert current["budget"]["calls"] == len(tools.calls)


def test_audit_verification_detects_modified_event(tmp_path):
    runner, run_id, _ = workflow(tmp_path)
    runner.run_to_completion(run_id)
    store = RunStore(tmp_path)
    assert store.verify(run_id)["status"] == "OK"
    path = tmp_path / run_id / "run.json"
    archived = json.loads(path.read_text(encoding="utf-8"))
    archived["events"][0]["data"]["tampered"] = True
    path.write_text(json.dumps(archived), encoding="utf-8")
    with pytest.raises(ValueError, match="chain"):
        store.verify(run_id)


def test_audit_verification_detects_modified_tool_blob(tmp_path):
    runner, run_id, _ = workflow(tmp_path)
    current = runner.run_to_completion(run_id)
    store = RunStore(tmp_path)
    output_refs = [event["data"]["output_ref"] for event in current["events"]
                   if event["data"].get("output_ref")]
    assert output_refs
    path = tmp_path / run_id / output_refs[0]
    path.write_text(json.dumps({"tampered": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        store.verify(run_id)


def test_snapshot_is_detached_from_mutable_workflow_state(tmp_path):
    runner, run_id, _ = workflow(tmp_path)
    current = advance_to_prediction(runner, run_id)
    assert current["prediction"] is not None
    current["prediction"]["home_win_probability"] = 999
    current["evidence"].clear()
    preserved = runner.get_run(run_id)
    assert preserved["prediction"]["home_win_probability"] == 0.4
    assert len(preserved["evidence"]) == 1


def test_replacement_policy_uses_same_workflow_and_budget(tmp_path):
    seen = []

    def choose(current):
        seen.append(current)
        return Decision("stop", reason="SYNTHETIC_REPLACEMENT_POLICY")

    tools = ScriptedTools()
    runner = Workflow(tools, store_dir=tmp_path, policy=CallbackPolicy(choose))
    run_id = runner.start(MATCH_ID, CUTOFF)["run_id"]
    current = runner.run_to_completion(run_id)
    assert len(seen) == 1
    assert seen[0]["prediction"]["home_win_probability"] == 0.4
    assert seen[0]["budget"]["calls"] == 4
    assert current["state"] == "STOPPED"
    assert "refresh_information" not in call_names(tools)


@pytest.mark.parametrize("probabilities", [
    (0.8, 0.3, 0.3),
    (-0.1, 0.6, 0.5),
    (float("nan"), 0.3, 0.7),
])
def test_invalid_model_distribution_is_not_published(tmp_path, probabilities):
    prediction = {
        "match_id": MATCH_ID, "prediction_cutoff": CUTOFF,
        "home_win_probability": probabilities[0],
        "draw_probability": probabilities[1],
        "away_win_probability": probabilities[2],
        "model_version": "malformed-synthetic-model",
        "dataset_version": "synthetic-dataset-v1",
        "input_manifest": ["fixture://synthetic-model-input"],
        "information_mode": "synthetic_test",
    }
    tools = ScriptedTools(failures={"predict": ToolResult(status="OK", data=prediction)})
    runner, run_id, _ = workflow(tmp_path, tools)
    current = runner.run_to_completion(run_id)
    assert current["state"] == "DEFERRED"
    assert current["prediction"] is None


@pytest.mark.parametrize("changes", [
    {"max_calls": -1}, {"max_queries": -1}, {"max_cost": -0.01},
    {"max_cost": float("nan")}, {"max_cost": float("inf")},
    {"max_elapsed_seconds": -1}, {"max_tool_seconds": 0},
])
def test_invalid_budget_is_rejected_before_start(changes):
    with pytest.raises(ValueError):
        Budget(**changes)


@pytest.mark.parametrize("changes", [
    {"match_id": "another-synthetic-match"},
    {"prediction_cutoff": "2026-09-19T12:00:01Z"},
    {"prediction_cutoff": "2026-09-19T11:59:59Z"},
    {"information_mode": "operational"},
])
def test_prediction_identity_cutoff_and_mode_must_match_the_run(tmp_path, changes):
    prediction = {
        "match_id": MATCH_ID, "prediction_cutoff": CUTOFF,
        "home_win_probability": 0.4, "draw_probability": 0.3,
        "away_win_probability": 0.3,
        "model_version": "synthetic-model-v1",
        "dataset_version": "synthetic-dataset-v1",
        "input_manifest": ["fixture://synthetic-model-input"],
        "information_mode": "synthetic_test", **changes,
    }
    tools = ScriptedTools(failures={"predict": ToolResult(status="OK", data=prediction)})
    runner, run_id, _ = workflow(tmp_path, tools)
    current = runner.run_to_completion(run_id)
    assert current["state"] == "DEFERRED"
    assert current["prediction"] is None
    assert "INVALID_MODEL_RESULT" in current["reason"]


@pytest.mark.parametrize("interrupt_recovery", [False, True])
def test_process_crash_preserves_call_intent_and_does_not_reexecute(tmp_path, interrupt_recovery):
    script = """
import os
from pathlib import Path
import runpy
import sys

namespace = runpy.run_path(sys.argv[2])
tools = namespace['ScriptedTools'](failures={'query_matches': lambda _: os._exit(23)})
runner = namespace['Workflow'](tools, store_dir=sys.argv[1])
run_id = runner.start(namespace['MATCH_ID'], namespace['CUTOFF'])['run_id']
Path(sys.argv[1], 'run-id.txt').write_text(run_id, encoding='utf-8')
runner.step(run_id)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), str(Path(__file__).resolve())],
        cwd=Path(__file__).resolve().parents[2], capture_output=True,
        text=True, timeout=15,
    )
    assert completed.returncode == 23, completed.stderr
    run_id = (tmp_path / "run-id.txt").read_text(encoding="utf-8")
    pending = RunStore(tmp_path).load(run_id)
    assert pending["pending_call"]["tool"] == "query_matches"
    assert pending["budget"]["calls"] == 1

    fresh_tools = ScriptedTools()
    recovered_runner = Workflow(fresh_tools, store_dir=tmp_path)
    if interrupt_recovery:
        save = recovered_runner.store.save

        def interrupt_after_recovery_event(run):
            save(run)
            if run["events"][-1]["type"] == "interrupted_transition":
                raise RuntimeError("Synthetic interruption after durable recovery event")

        recovered_runner.store.save = interrupt_after_recovery_event
        try:
            recovered_runner.get_run(run_id)
        except RuntimeError as exc:
            assert "Synthetic interruption" in str(exc)
        recovered_runner = Workflow(fresh_tools, store_dir=tmp_path)
    recovered = recovered_runner.get_run(run_id)
    assert recovered["state"] == "DEFERRED"
    assert recovered["pending_call"] is None
    assert "INTERRUPTED" in recovered["reason"]
    assert recovered["budget"]["spent_cost"] == pytest.approx(0.1)
    assert recovered["budget"]["remaining_cost"] == pytest.approx(2.9)
    assert recovered["budget"]["estimated_cost_charges"] == 1
    assert len(recovered["unreconciled_calls"]) == 1
    assert recovered["unreconciled_calls"][0]["cost_status"] == "UNKNOWN_RESERVED_ESTIMATE"
    assert recovered["unreconciled_calls"][0]["reserved_cost"] == pytest.approx(0.1)
    assert recovered["report"]["unreconciled_calls"] == recovered["unreconciled_calls"]
    recovered_runner.step(run_id)
    repeated = recovered_runner.run_to_completion(run_id)
    assert repeated["budget"]["spent_cost"] == pytest.approx(0.1)
    assert fresh_tools.calls == []
    assert RunStore(tmp_path).verify(run_id)["status"] == "OK"


def test_duplicate_content_with_new_conflict_flag_defers(tmp_path):
    tools = ScriptedTools(refreshed=[evidence("new-quality-assessment", conflict=True)])
    runner, run_id, _ = workflow(tmp_path, tools)
    current = runner.run_to_completion(run_id)
    assert current["state"] == "DEFERRED"
    assert current["prediction"] is None
    assert current["last_prediction"]["home_win_probability"] == 0.4


def test_corrupted_summary_does_not_execute_another_tool(tmp_path):
    runner, run_id, tools = workflow(tmp_path)
    path = tmp_path / run_id / "run.json"
    persisted = json.loads(path.read_text(encoding="utf-8"))
    persisted["budget"]["max_cost"] = 999
    path.write_text(json.dumps(persisted), encoding="utf-8")
    with pytest.raises(ValueError):
        runner.step(run_id)
    assert tools.calls == []


def test_crash_after_tool_result_does_not_repeat_completed_tool(tmp_path):
    script = """
import os
from pathlib import Path
import runpy
import sys

namespace = runpy.run_path(sys.argv[2])
runner = namespace['Workflow'](namespace['ScriptedTools'](), store_dir=sys.argv[1])
run_id = runner.start(namespace['MATCH_ID'], namespace['CUTOFF'])['run_id']
Path(sys.argv[1], 'run-id.txt').write_text(run_id, encoding='utf-8')
save = runner.store.save
def interrupt_after_result(run):
    save(run)
    if run['events'][-1]['type'] == 'tool_returned':
        os._exit(24)
runner.store.save = interrupt_after_result
runner.step(run_id)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), str(Path(__file__).resolve())],
        cwd=Path(__file__).resolve().parents[2], capture_output=True,
        text=True, timeout=15,
    )
    assert completed.returncode == 24, completed.stderr
    run_id = (tmp_path / "run-id.txt").read_text(encoding="utf-8")
    archived = RunStore(tmp_path).load(run_id)
    assert archived["events"][-1]["type"] == "tool_returned"
    fresh_tools = ScriptedTools()
    recovered_runner = Workflow(fresh_tools, store_dir=tmp_path)
    recovered_runner.step(run_id)
    assert fresh_tools.calls == [], "A persisted tool result must not cause external reexecution."
    recovered = recovered_runner.get_run(run_id)
    assert recovered["state"] in {"MATCH_READY", "DEFERRED"}


def test_audit_checksum_detects_changes_to_redacted_stored_fields(tmp_path):
    runner, run_id, _ = workflow(tmp_path)
    store = RunStore(tmp_path)
    ref = store.put_blob(run_id, {"api_key": "synthetic-fixture-secret"})
    path = tmp_path / run_id / ref
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["api_key"] == "[REDACTED]"
    data["api_key"] = "synthetic-modified-value"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        store.get_blob(run_id, ref)


def test_tool_planner_cost_and_output_are_in_the_same_journal(tmp_path):
    tools = ScriptedTools(failures={"plan_next_action": ToolResult(
        status="OK", data={"action": "stop", "reason": "SYNTHETIC_PLANNER_STOP"}, cost=0.2,
    )})
    runner = Workflow(tools, store_dir=tmp_path, policy=ToolPlanningPolicy())
    run_id = runner.start(MATCH_ID, CUTOFF)["run_id"]
    current = runner.run_to_completion(run_id)
    assert current["state"] == "STOPPED"
    assert current["budget"]["calls"] == 5
    assert current["budget"]["queries"] == 0
    assert current["budget"]["spent_cost"] == pytest.approx(0.6)
    assert call_names(tools)[-1] == "plan_next_action"
    planner_event = [event for event in current["events"]
                     if event["type"] == "tool_returned"
                     and event["data"].get("tool") == "plan_next_action"]
    assert len(planner_event) == 1
    assert planner_event[0]["data"]["input_ref"]
    assert planner_event[0]["data"]["output_ref"]
    assert RunStore(tmp_path).verify(run_id)["status"] == "OK"


def test_tool_planner_cannot_bypass_call_budget(tmp_path):
    tools = ScriptedTools()
    runner = Workflow(tools, store_dir=tmp_path, policy=ToolPlanningPolicy())
    run_id = runner.start(MATCH_ID, CUTOFF, budget=Budget(max_calls=4))["run_id"]
    current = runner.run_to_completion(run_id)
    assert current["state"] == "STOPPED"
    assert current["prediction"]["home_win_probability"] == 0.4
    assert "plan_next_action" not in call_names(tools)
    assert current["budget"]["calls"] == 4


def test_newer_source_revision_replaces_current_version_and_repredicts(tmp_path):
    tools = ScriptedTools(refreshed=[evidence(
        version="2", content_hash="synthetic-hash-v2",
        available_at="2026-09-19T11:00:00Z", observed_at="2026-09-19T11:00:00Z",
        content={"note": "Updated synthetic source revision."},
    )])
    runner, run_id, _ = workflow(tmp_path, tools)
    current = runner.run_to_completion(run_id)
    assert tools.predictions == 2
    assert len(current["evidence"]) == 1
    assert current["evidence"][0]["version"] == "2"
    assert current["report"]["evidence_versions"][0]["version"] == "2"
    updates = [event["data"] for event in current["events"]
               if event["type"] == "prediction_updated"]
    assert updates[-1]["before"]["home_win_probability"] == 0.4
    assert updates[-1]["after"]["home_win_probability"] == 0.5
    assert updates[-1]["evidence_versions"][0]["version"] == "2"


def test_older_revision_arriving_later_does_not_replace_current_version(tmp_path):
    tools = ScriptedTools(initial=[evidence(
        version="2", content_hash="synthetic-hash-v2",
        available_at="2026-09-19T11:00:00Z", observed_at="2026-09-19T11:00:00Z",
    )], refreshed=[evidence()])
    runner, run_id, _ = workflow(tmp_path, tools)
    current = runner.run_to_completion(run_id)
    assert current["state"] == "STOPPED"
    assert tools.predictions == 1
    assert len(current["evidence"]) == 1
    assert current["evidence"][0]["version"] == "2"
    assert "SUPERSEDED_VERSION" in json.dumps(current["events"])
