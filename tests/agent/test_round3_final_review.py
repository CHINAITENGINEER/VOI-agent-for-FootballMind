"""Targeted independent constructed checks for recovery and cost boundaries."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

import football_agent.acquisition as execution
from football_agent.acquisition import AcquisitionWorkflow
from football_agent.audit import RunStore
from football_agent.locks import RunBusyError
from football_agent.research_bridge import SyntheticBackend


MATCH = "synthetic:alpha-beta"
CUTOFF = "2025-01-01T12:00:00Z"


def start(tmp_path, backend=None, **kwargs):
    workflow = AcquisitionWorkflow(backend or SyntheticBackend(), tmp_path)
    run = workflow.start(MATCH, CUTOFF, **kwargs)
    return workflow, run


def test_one_expired_backoff_is_not_blocked_by_another_action(tmp_path, monkeypatch):
    clock = [datetime(2025, 1, 1, 12, tzinfo=timezone.utc)]
    monkeypatch.setattr(execution, "now", lambda: clock[0].isoformat())

    class IndependentBackoff(SyntheticBackend):
        def actions(self, visible):
            actions = super().actions(visible)[:2]
            for action in actions:
                attempts = sum(h["action_id"] == action["action_id"] for h in visible["action_history"])
                action.update(status="AVAILABLE", version=f"revision-{attempts}")
            return actions

        def acquire(self, visible, action_id, parameters, request_id):
            state = deepcopy(visible)
            state["action_history"].append({"action_id": action_id, "request_id": request_id})
            return {"status": "RATE_LIMITED", "visible_state": state,
                    "cost": {"amount": 0, "unit": "simulation_credit", "status": "simulated"},
                    "retry_after_seconds": 10 if action_id.endswith("strength") else 20}

    workflow, run = start(tmp_path, IndependentBackoff(), strategy="fixed")
    run = workflow.run_to_completion(run["run_id"])
    assert run["state"] == "WAITING" and run["budget"]["actions"] == 2
    # Advance only past the earliest retry.  At least one other action remains
    # in its retry window, which exposes a global `any(cooldown)` gate.
    earliest = min(datetime.fromisoformat(value) for value in run["cooldowns"].values())
    latest = max(datetime.fromisoformat(value) for value in run["cooldowns"].values())
    assert earliest < latest
    clock[0] = earliest + timedelta(seconds=.1)
    run = workflow.step(run["run_id"])
    assert run["budget"]["actions"] == 3, "The first action is ready even though the second is still cooling down"
    assert run["history"][-1]["action_id"].endswith("strength")


@pytest.mark.parametrize("bad_values", [[None], ["not-an-estimate"], [17]])
def test_malformed_value_item_becomes_explicit_state(tmp_path, bad_values):
    class Malformed(SyntheticBackend):
        def estimate(self, *_args):
            return bad_values

    workflow, run = start(tmp_path, Malformed())
    run = workflow.run_to_completion(run["run_id"])
    assert run["state"] in {"STOPPED", "DEFERRED"}
    assert run["reason"] in {"ESTIMATE_UNAVAILABLE", "INVALID_VALUE_ESTIMATE", "INVALID_TOOL_PUBLIC_OUTPUT"}
    assert run["budget"]["actions"] == 0
    assert workflow.store.verify(run["run_id"])["status"] == "OK"


def test_actual_simulated_cost_in_another_unit_is_not_silently_settled(tmp_path):
    class WrongUnit(SyntheticBackend):
        def acquire(self, *args):
            result = super().acquire(*args)
            result["cost"] = {"amount": 0, "unit": "another_simulation_currency", "status": "simulated"}
            return result

    workflow, run = start(tmp_path, WrongUnit(), budget={"max_actions": 1})
    run = workflow.run_to_completion(run["run_id"])
    assert run["budget"]["simulated_cost"] == 1, "A different unit cannot release the original reserved credits"
    entry = run["budget"]["cost_ledger"][0]
    assert entry["actual"]["unit"] == "another_simulation_currency"
    assert entry.get("settled_amount") is None


def test_source_unknown_money_stays_unknown_and_separate_from_reservation(tmp_path):
    class UnknownFee(SyntheticBackend):
        def actions(self, state):
            result = super().actions(state)
            for item in result:
                item["cost"] = {"amount": None, "unit": "USD", "status": "unknown", "reserve": 0.25}
            return result

    workflow, run = start(tmp_path, UnknownFee(), budget={"max_actions": 1})
    run = workflow.run_to_completion(run["run_id"])
    assert run["budget"]["online_reserve"] == 0.25
    assert run["budget"]["simulated_cost"] == 0
    actual = run["budget"]["cost_ledger"][0]["actual"]
    assert actual["amount"] is None and actual["status"] == "unknown" and actual["unit"] == "USD"


def test_actual_workflow_cross_process_execution_has_one_writer(tmp_path):
    workflow, run = start(tmp_path / "runs", budget={"max_actions": 1})
    workflow.step(run["run_id"])
    marker = tmp_path / "entered.txt"
    code = """
from pathlib import Path
import sys,time
from football_agent.acquisition import AcquisitionWorkflow
from football_agent.research_bridge import SyntheticBackend
class Slow(SyntheticBackend):
    def acquire(self,*args):
        Path(sys.argv[3]).write_text('entered',encoding='utf-8')
        time.sleep(1)
        return super().acquire(*args)
workflow=AcquisitionWorkflow(Slow(),sys.argv[1])
workflow.step(sys.argv[2])
"""
    child = subprocess.Popen([sys.executable, "-c", code, str(tmp_path / "runs"), run["run_id"], str(marker)],
                             cwd=Path(__file__).resolve().parents[2], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        until = time.monotonic() + 15
        while not marker.exists() and child.poll() is None and time.monotonic() < until:
            time.sleep(0.02)
        assert marker.exists(), child.communicate(timeout=2)
        with pytest.raises(RunBusyError):
            workflow.step(run["run_id"])
        stdout, stderr = child.communicate(timeout=15)
        assert child.returncode == 0, stdout + stderr
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate()
    restored = AcquisitionWorkflow(SyntheticBackend(), tmp_path / "runs").get_run(run["run_id"])
    assert restored["budget"]["actions"] == 1
    assert len(restored["history"]) == 1
    assert restored["budget"]["simulated_cost"] == 1
    assert RunStore(tmp_path / "runs").verify(run["run_id"])["status"] == "OK"
