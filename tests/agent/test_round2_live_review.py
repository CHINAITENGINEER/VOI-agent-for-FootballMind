"""Independent constructed audit of live lineage, budgets, and crash windows."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
import json
from pathlib import Path
import runpy
import subprocess
import sys
import time

import pytest

from football_agent.audit import RunBusyError, RunStore
from football_agent.engine import Workflow
from football_agent.types import Budget


BASE_TEST = Path(__file__).with_name("test_round2_live.py")
_base = runpy.run_path(str(BASE_TEST))
LiveFixture, predicted = _base["LiveFixture"], _base["predicted"]
WORKSPACE = Path(__file__).resolve().parents[2]


def _child(workflow, parent):
    return workflow.get_run(parent["child_run_id"])


def test_child_inherits_all_original_budget_limits_and_consumption(tmp_path):
    tools = LiveFixture()
    workflow = Workflow(tools, tmp_path)
    before = predicted(workflow, tools, Budget(max_requests=1, max_queries=1, max_cost=0.25, max_calls=11))
    parent = workflow.step(before["run_id"], action="query", topic="odds")
    child = _child(workflow, parent)
    assert child["budget"] == parent["budget"]
    assert child["inherited_budget"] == parent["budget"]
    assert child["previous_decision"]["prediction"] == before["prediction"]
    assert child["previous_decision"]["evidence_versions"] == before["report"]["evidence_versions"]
    assert child["root_run_id"] == parent["run_id"]
    for _ in range(4):
        child = workflow.step(child["run_id"])
    blocked = workflow.step(child["run_id"], action="query", topic="lineups")
    assert blocked["state"] == "STOPPED"
    assert blocked["reason"] in {"QUERY_BUDGET_EXHAUSTED", "COST_BUDGET_EXHAUSTED", "REQUEST_BUDGET_EXHAUSTED"}
    assert blocked["budget"]["requests"] == 1
    assert blocked["budget"]["spent_cost"] == 0.25
    assert tools.refreshes == 1
    assert workflow.get_run(parent["run_id"])["prediction"] == before["prediction"]
    assert workflow.store.verify(child["run_id"])["status"] == "OK"


def test_child_with_remaining_query_budget_still_obeys_shared_request_limit(tmp_path):
    tools = LiveFixture()
    workflow = Workflow(tools, tmp_path)
    before = predicted(workflow, tools, Budget(max_requests=1, max_queries=3))
    parent = workflow.step(before["run_id"], action="query", topic="odds")
    child = _child(workflow, parent)
    for _ in range(4):
        child = workflow.step(child["run_id"])
    blocked = workflow.step(child["run_id"], action="query", topic="lineups")
    assert blocked["reason"] == "REQUEST_BUDGET_EXHAUSTED"
    assert blocked["budget"]["calls"] == 9
    assert tools.refreshes == 1


def test_new_raw_observation_with_unchanged_input_keeps_equal_probabilities(tmp_path):
    class Unchanged(LiveFixture):
        def evidence(self, fresh=False):
            item = super().evidence(fresh)
            item["content"] = {"value": 0}
            item["content_hash"] = "unchanged-content"
            return item

        def execute(self, tool, inputs):
            result = super().execute(tool, inputs)
            if tool == "refresh_information":
                result = replace(result, data={**result.data, "content_changed": False})
            return result

    tools = Unchanged()
    workflow = Workflow(tools, tmp_path)
    before = predicted(workflow, tools)
    parent = workflow.step(before["run_id"], action="query", topic="odds")
    child = _child(workflow, parent)
    for _ in range(4):
        child = workflow.step(child["run_id"])
    assert parent["evidence"][0]["raw_ref"] == "synthetic://old"
    assert child["evidence"][0]["raw_ref"] == "synthetic://new"
    assert child["prediction_cutoff"] > parent["prediction_cutoff"]
    for key in ("home_win_probability", "draw_probability", "away_win_probability"):
        assert child["prediction"][key] == parent["prediction"][key]
    refresh = next(e for e in parent["events"] if e["type"] == "live_refresh_archived")
    assert refresh["data"]["content_changed"] is False


def test_live_child_does_not_silently_adopt_configuration_changed_during_refresh(tmp_path):
    class ChangedModel(LiveFixture):
        def capabilities(self):
            return {**super().capabilities(), "model_id": "configured-model",
                    "model_version": "version-2" if self.refreshes else "version-1"}

        def execute(self, tool, inputs):
            result = super().execute(tool, inputs)
            if tool == "predict":
                return replace(result, data={**result.data, "model_version": self.capabilities()["model_version"]})
            return result

    tools = ChangedModel()
    workflow = Workflow(tools, tmp_path)
    before = predicted(workflow, tools)
    parent = workflow.step(before["run_id"], action="query", topic="odds")
    if parent.get("child_run_id"):
        child = _child(workflow, parent)
        assert child["tool_configuration"]["model_version"] == before["tool_configuration"]["model_version"]
    else:
        assert parent["state"] == "DEFERRED"
        assert "CONFIGURATION" in parent["reason"]


def test_configuration_hash_drift_is_checked_for_existing_workflow_instances(tmp_path):
    class Configured(LiveFixture):
        configuration_hash = "snapshot-config-1"

        def capabilities(self):
            return {**super().capabilities(), "configuration_hash": self.configuration_hash}

    tools = Configured()
    workflow = Workflow(tools, tmp_path)
    run = workflow.start("synthetic:live", tools.cutoff)
    tools.configuration_hash = "snapshot-config-2"
    stopped = workflow.step(run["run_id"])
    assert stopped["reason"] == "RUN_CONFIGURATION_CHANGED"
    assert stopped["budget"]["calls"] == 0


@pytest.mark.parametrize("references", ["synthetic://string-is-not-a-list", [""], [None], True])
def test_live_refresh_requires_usable_observation_reference_list(tmp_path, references):
    class Malformed(LiveFixture):
        def execute(self, tool, inputs):
            result = super().execute(tool, inputs)
            if tool == "refresh_information":
                return replace(result, data={**result.data, "observation_refs": references})
            return result

    tools = Malformed()
    workflow = Workflow(tools, tmp_path)
    run = predicted(workflow, tools)
    parent = workflow.step(run["run_id"], action="query", topic="odds")
    assert parent["reason"] == "INVALID_LIVE_REFRESH_RESULT"
    assert parent["budget"]["requests"] == 1
    assert not parent.get("child_run_id")


CRASH_WORKER = """
import json, os, runpy, sys, time
from pathlib import Path
root, base_path, point = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
ns = runpy.run_path(base_path)
class Counted(ns['LiveFixture']):
    def execute(self, tool, inputs):
        if tool == 'refresh_information':
            with (root / 'refreshes.jsonl').open('a') as stream:
                stream.write(json.dumps({'request_id': inputs['request_id']}) + '\\n')
                stream.flush()
                os.fsync(stream.fileno())
        return super().execute(tool, inputs)
tools = Counted()
workflow = ns['Workflow'](tools, root)
parent = ns['predicted'](workflow, tools)
(root / 'parent-id.txt').write_text(parent['run_id'])
(root / 'fixture-clock.json').write_text(json.dumps({'timestamp': tools.timestamp.isoformat(), 'cutoff': tools.cutoff}))
save = workflow.store.save
def interrupt(record):
    save(record)
    event = record['events'][-1]
    matched = ((point == 'archived' and event['type'] == 'live_refresh_archived')
        or (point in {'child', 'pause-child'} and event['type'] == 'initialized' and record.get('parent_run_id'))
        or (point == 'committed' and event['type'] == 'stopped' and record['reason'] == 'LIVE_CUTOFF_ADVANCED'))
    if matched:
        (root / 'crash-point.json').write_text(json.dumps({'run_id': record['run_id'], 'event': event['type']}))
        if point == 'pause-child':
            while not (root / 'release-parent').exists():
                time.sleep(0.01)
            return
        os._exit(83)
workflow.store.save = interrupt
workflow.step(parent['run_id'], action='query', topic='odds')
"""


def _restored_fixture(root):
    saved = json.loads((root / "fixture-clock.json").read_text())
    tools = LiveFixture()
    tools.timestamp, tools.cutoff = datetime.fromisoformat(saved["timestamp"]), saved["cutoff"]
    tools.refreshes = 1
    return tools


@pytest.mark.parametrize("point", ["archived", "child", "committed"])
def test_crash_at_live_handoff_keeps_links_coherent_and_never_repeats_refresh(tmp_path, point):
    completed = subprocess.run([sys.executable, "-c", CRASH_WORKER, str(tmp_path), str(BASE_TEST), point],
                               cwd=WORKSPACE, capture_output=True, text=True, timeout=10)
    assert completed.returncode == 83, completed.stdout + completed.stderr
    parent_id = (tmp_path / "parent-id.txt").read_text()
    workflow = Workflow(_restored_fixture(tmp_path), tmp_path)
    parent = workflow.get_run(parent_id)
    assert parent["budget"]["requests"] == 1
    assert parent["budget"]["spent_cost"] == 0.25
    assert parent["prediction"]["home_win_probability"] == 0.4
    assert len((tmp_path / "refreshes.jsonl").read_text().splitlines()) == 1
    assert workflow.store.verify(parent_id)["status"] == "OK"
    if parent.get("child_run_id"):
        child = workflow.get_run(parent["child_run_id"])
        # A child that can still run requires a committed budget handoff.
        # Explicitly deferred children are also a coherent conservative result.
        assert child["state"] == "DEFERRED" or parent["reason"] == "LIVE_CUTOFF_ADVANCED"
        assert child["budget"]["requests"] == 1
        assert child["budget"]["spent_cost"] == 0.25
        assert workflow.store.verify(child["run_id"])["status"] == "OK"
    assert workflow.step(parent_id)["budget"]["requests"] == 1


def test_child_step_during_parent_commit_is_busy_without_marking_child_failed(tmp_path):
    process = subprocess.Popen([sys.executable, "-c", CRASH_WORKER, str(tmp_path), str(BASE_TEST), "pause-child"],
                               cwd=WORKSPACE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / "crash-point.json").exists():
            if process.poll() is not None or time.monotonic() >= deadline:
                process.kill()
                stdout, stderr = process.communicate()
                pytest.fail(stdout + stderr)
            time.sleep(0.01)
        child_id = json.loads((tmp_path / "crash-point.json").read_text())["run_id"]
        workflow = Workflow(_restored_fixture(tmp_path), tmp_path)
        original = RunStore(tmp_path).load(child_id)
        with pytest.raises(RunBusyError):
            workflow.step(child_id)
        assert RunStore(tmp_path).load(child_id) == original
        (tmp_path / "release-parent").write_text("commit")
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, stdout + stderr
        child = workflow.step(child_id)
        assert child["state"] == "MATCH_READY"
        assert child["budget"]["requests"] == 1
        assert len((tmp_path / "refreshes.jsonl").read_text().splitlines()) == 1
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)


def test_child_first_recovery_completes_parent_and_preserves_the_child_genesis(tmp_path):
    completed = subprocess.run([sys.executable, "-c", CRASH_WORKER, str(tmp_path), str(BASE_TEST), "child"],
                               cwd=WORKSPACE, capture_output=True, text=True, timeout=10)
    assert completed.returncode == 83, completed.stderr
    parent_id = (tmp_path / "parent-id.txt").read_text()
    child_id = json.loads((tmp_path / "crash-point.json").read_text())["run_id"]
    initial = RunStore(tmp_path).load(child_id)
    workflow = Workflow(_restored_fixture(tmp_path), tmp_path)
    child = workflow.get_run(child_id)
    parent = workflow.get_run(parent_id)
    assert parent["reason"] == "LIVE_CUTOFF_ADVANCED"
    assert child["state"] == "INITIALIZED"
    assert child["events"] == initial["events"]
    assert child["budget"] == initial["budget"]
    assert len((tmp_path / "refreshes.jsonl").read_text().splitlines()) == 1


def test_second_crash_during_handoff_recovery_keeps_the_same_child_and_budget(tmp_path):
    completed = subprocess.run([sys.executable, "-c", CRASH_WORKER, str(tmp_path), str(BASE_TEST), "archived"],
                               cwd=WORKSPACE, capture_output=True, text=True, timeout=10)
    assert completed.returncode == 83, completed.stderr
    parent_id = (tmp_path / "parent-id.txt").read_text()
    planned_child = RunStore(tmp_path).load(parent_id)["child_run_id"]
    script = """
import os, runpy, sys
ns = runpy.run_path(sys.argv[2])
workflow = ns['Workflow'](ns['LiveFixture'](), sys.argv[1])
save = workflow.store.save
def crash(record):
    save(record)
    if record['events'][-1]['type'] == 'live_handoff_recovered':
        os._exit(84)
workflow.store.save = crash
workflow.get_run(sys.argv[3])
"""
    recovered_once = subprocess.run([sys.executable, "-c", script, str(tmp_path), str(BASE_TEST), parent_id],
                                    cwd=WORKSPACE, capture_output=True, text=True, timeout=10)
    assert recovered_once.returncode == 84, recovered_once.stderr
    original_child = RunStore(tmp_path).load(planned_child)
    workflow = Workflow(_restored_fixture(tmp_path), tmp_path)
    parent = workflow.get_run(parent_id)
    assert parent["reason"] == "LIVE_CUTOFF_ADVANCED"
    assert parent["child_run_id"] == planned_child
    child = workflow.get_run(planned_child)
    assert child["events"] == original_child["events"]
    assert child["budget"] == original_child["budget"]
    assert child["budget"]["spent_cost"] == 0.25
    assert child["budget"]["requests"] == 1
    assert len((tmp_path / "refreshes.jsonl").read_text().splitlines()) == 1
    assert len(list(tmp_path.glob("*/run.json"))) == 2
    assert workflow.store.verify(parent_id)["status"] == "OK"
    assert workflow.store.verify(planned_child)["status"] == "OK"


def test_explicit_existing_run_id_never_replaces_its_audit_journal(tmp_path):
    tools = LiveFixture()
    workflow = Workflow(tools, tmp_path)
    initial = predicted(workflow, tools)
    with pytest.raises(FileExistsError):
        workflow.start("synthetic:live", tools.cutoff, _run_id=initial["run_id"])
    assert workflow.get_run(initial["run_id"])["events"] == initial["events"]


@pytest.mark.parametrize("time_mode,status", [("archive_replay", "OK"), ("live", "OK"),
                                              ("live", "RATE_LIMITED"), ("live", "FAILED")])
def test_refresh_summary_preserves_raw_references_for_replay_and_failed_results(tmp_path, time_mode, status):
    tools = LiveFixture(time_mode=time_mode, result_status=status)
    workflow = Workflow(tools, tmp_path)
    before = predicted(workflow, tools)
    after = workflow.step(before["run_id"], action="query", topic="odds")
    assert len(after["refresh_results"]) == 1
    result = after["report"]["refresh_results"][0]
    assert result["status"] == status
    assert result["observation_refs"] == ["synthetic://new"]
    assert result["decision_cutoff"] == tools.timestamp.isoformat()
    assert result["content_changed"] is True
    assert result["cost"] == 0.25
    assert result["requests"] == 1
    output = workflow.store.get_blob(after["run_id"], result["output_ref"])
    assert output["data"]["observation_refs"] == result["observation_refs"]
    persisted = RunStore(tmp_path).load(after["run_id"])
    assert persisted["refresh_results"] == after["refresh_results"]
    if time_mode == "archive_replay":
        assert after["prediction"] == before["prediction"]
        assert all(e["raw_ref"] == "synthetic://old" for e in after["evidence"])


@pytest.mark.parametrize("field,reason", [("source_published_at", "OUTSIDE_CUTOFF"),
                                         ("expires_at", "STALE_EVIDENCE"), (None, None)])
def test_historical_reconstruction_still_checks_publication_and_expiry(tmp_path, field, reason):
    class Reconstructed(LiveFixture):
        def capabilities(self):
            return {**super().capabilities(), "information_mode": "historical_reconstruction"}

        def evidence(self, fresh=False):
            item = super().evidence(fresh)
            item["observed_at"] = self.timestamp.isoformat()
            item["information_mode"] = "historical_reconstruction"
            item["reconstruction"] = {"basis": "Explicit synthetic publication assumption", "revision_risk": "Fixture"}
            if field == "source_published_at":
                item[field] = (datetime.fromisoformat(self.cutoff) + timedelta(seconds=1)).isoformat()
            elif field == "expires_at":
                item[field] = self.cutoff
            return item

    tools = Reconstructed()
    workflow = Workflow(tools, tmp_path)
    run = workflow.start("synthetic:live", tools.cutoff)
    workflow.step(run["run_id"])
    run = workflow.step(run["run_id"])
    if reason:
        assert run["state"] == "DEFERRED"
        assert run["evidence"] == []
        assert any(item["reason"] == reason for item in run["quality"])
    else:
        assert run["state"] == "EVIDENCE_READY"
        assert run["evidence"][0]["reconstruction"]["basis"]
