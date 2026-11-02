"""Constructed execution tests; research values remain backend outputs."""
from copy import deepcopy
import json
from pathlib import Path
import time

import pytest

from football_agent.acquisition import AcquisitionWorkflow, AcquisitionBudget
from football_agent.research_bridge import SyntheticBackend
from football_agent.proposer import LLMProposer


def started(tmp_path, backend=None, **kwargs):
    w = AcquisitionWorkflow(backend or SyntheticBackend(), tmp_path)
    r = w.start("synthetic:alpha-beta", "2025-01-01T12:00:00Z", **kwargs)
    return w, r


@pytest.mark.parametrize("scenario,reason,actions", [
    ("immediate_stop", "VALUE_STOP_NONPOSITIVE_NET_GAIN", []),
    ("cost_exceeds_gain", "VALUE_STOP_NONPOSITIVE_NET_GAIN", []),
    ("normal", "VALUE_STOP_NONPOSITIVE_NET_GAIN", ["fixture.reveal.strength", "fixture.reveal.xg"]),
    ("no_data", "NO_LEGAL_ACTIONS", ["fixture.reveal.strength", "fixture.reveal.form", "fixture.reveal.xg"]),
])
def test_stop_and_dynamic_choice(tmp_path, scenario, reason, actions):
    w, r = started(tmp_path, SyntheticBackend(scenario))
    r = w.run_to_completion(r["run_id"])
    assert r["reason"] == reason
    assert [h["action_id"] for h in r["history"]] == actions
    assert w.store.verify(r["run_id"])["status"] == "OK"


def test_budget_stops_even_with_positive_estimates(tmp_path):
    w, r = started(tmp_path, budget={"max_actions": 1})
    r = w.run_to_completion(r["run_id"])
    assert r["reason"] == "BUDGET_EXHAUSTED"
    assert r["budget"]["actions"] == 1 and r["budget"]["simulated_cost"] == 1


def test_hidden_changes_do_not_affect_initial_prediction_or_action(tmp_path):
    class Hidden(SyntheticBackend):
        def __init__(self, hidden, label):
            super().__init__(); self.hidden, self.label = hidden, label
        def acquire(self, *args):
            result = super().acquire(*args)
            for group in result["visible_state"]["groups"].values(): group["fixture_value"] = self.hidden
            return result
        def predict(self, state):
            result = super().predict(state)
            if state["groups"]:
                p = next(iter(state["groups"].values()))["fixture_value"]
                result["probabilities"] = {"home": p, "draw": (1-p)/2, "away": (1-p)/2}
            return result
    before, chosen, after = [], [], []
    for i, (hidden, label) in enumerate([(0.2, "home"), (0.8, "away")]):
        w, r = started(tmp_path / str(i), Hidden(hidden, label))
        r = w.step(r["run_id"]); before.append(r["prediction"]["probabilities"])
        r = w.step(r["run_id"]); chosen.append(r["history"][0]["action_id"])
        r = w.step(r["run_id"]); after.append(r["prediction"]["probabilities"])
    assert before[0] == before[1] and chosen[0] == chosen[1]
    assert after[0] != after[1]


def test_hidden_fields_rejected_before_model_and_persistence(tmp_path):
    class Leaky(SyntheticBackend):
        def start(self, *args):
            r = super().start(*args); r["target_label"] = "DO_NOT_SAVE_LABEL"; return r
        def predict(self, *_): pytest.fail("Predictor got leaked state")
    w, r = started(tmp_path, Leaky())
    assert r["state"] == "DEFERRED"
    assert "DO_NOT_SAVE_LABEL" not in json.dumps(r)
    for path in tmp_path.rglob("*.json"):
        assert "DO_NOT_SAVE_LABEL" not in path.read_text(encoding="utf-8")


def test_crash_after_acquire_return_never_repeats_paid_call(tmp_path, monkeypatch):
    backend = SyntheticBackend()
    calls = []
    acquire = backend.acquire
    def counted(*args): calls.append(args); return acquire(*args)
    backend.acquire = counted
    w, r = started(tmp_path, backend)
    w.step(r["run_id"])
    event = w._event
    def crash(run, kind, data):
        event(run, kind, data)
        if kind == "tool_returned" and data["tool"] == "acquire": raise RuntimeError("process crash window")
    monkeypatch.setattr(w, "_event", crash)
    with pytest.raises(RuntimeError): w.step(r["run_id"])
    restored = AcquisitionWorkflow(backend, tmp_path).get_run(r["run_id"])
    assert restored["reason"] == "INTERRUPTED_CALL_REQUIRES_RECONCILIATION"
    assert restored["budget"]["actions"] == 1 and len(calls) == 1
    assert restored["budget"]["simulated_cost"] == 1
    assert w.store.verify(r["run_id"])["status"] == "OK"


def test_restore_configuration_change_stops(tmp_path):
    w, r = started(tmp_path)
    changed = AcquisitionWorkflow(SyntheticBackend("immediate_stop"), tmp_path).step(r["run_id"])
    assert changed["reason"] == "RUN_CONFIGURATION_CHANGED"


def test_llm_missing_configuration_is_explicit_and_free(tmp_path):
    w = AcquisitionWorkflow(SyntheticBackend(), tmp_path, proposer=LLMProposer(model="", api_key=""))
    r = w.start("synthetic:alpha-beta", "2025-01-01T12:00:00Z", strategy="llm_gate")
    r = w.run_to_completion(r["run_id"])
    assert r["reason"] == "LLM_NEEDS_CONFIGURATION"
    assert r["budget"]["llm_calls"] == 0 and r["budget"]["online_reserve"] == 0


def test_low_support_or_absent_estimate_is_not_value_stop(tmp_path):
    class NoEstimate(SyntheticBackend):
        def estimate(self, *_): return []
    w, r = started(tmp_path, NoEstimate())
    assert w.run_to_completion(r["run_id"])["reason"] == "ESTIMATE_UNAVAILABLE"


def test_new_action_version_permits_retry_after_empty_result(tmp_path):
    class Revision(SyntheticBackend):
        def __init__(self): super().__init__("no_data")
        def actions(self, state):
            actions = super().actions(state)
            # Only an explicit changed upstream version makes a failed action new.
            for a in actions:
                if a["group_ids"] == ["strength"] and state["mask"].get("form"):
                    a.update(status="AVAILABLE", version="fixture-actions-v2")
            return actions
    w, r = started(tmp_path, Revision(), strategy="fixed", budget={"max_actions": 4, "max_simulated_cost": 4})
    r = w.run_to_completion(r["run_id"])
    ids = [h["action_id"] for h in r["history"]]
    assert ids[:3] == ["fixture.reveal.strength", "fixture.reveal.form", "fixture.reveal.strength"]
    assert len(set(h["execution_key"] for h in r["history"])) == len(ids)


def test_llm_max_five_is_enforced():
    with pytest.raises(ValueError): AcquisitionBudget(max_llm_calls=6)


def test_retry_backoff_blocks_new_revision_but_allows_other_actions(tmp_path):
    class Backoff(SyntheticBackend):
        def __init__(self): super().__init__("no_data")
        def acquire(self, *args):
            result = super().acquire(*args)
            if result["status"] == "NO_DATA": result["retry_after_seconds"] = 3600
            return result
        def actions(self, state):
            actions = super().actions(state)
            if state["action_history"]:
                actions[0].update(status="AVAILABLE", version="fixture-actions-v2")
            return actions
    workflow, run = started(tmp_path, Backoff(), strategy="fixed",
                            budget={"max_actions": 4, "max_simulated_cost": 4})
    run = workflow.run_to_completion(run["run_id"])
    assert run["state"] == "WAITING" and run["reason"] == "ACTION_BACKOFF"
    assert [h["action_id"] for h in run["history"]] == ["fixture.reveal.strength", "fixture.reveal.form", "fixture.reveal.xg"]
    assert workflow.step(run["run_id"])["budget"]["actions"] == 3


def test_unsupported_estimate_reason_remains_visible(tmp_path):
    class Unsupported(SyntheticBackend):
        def estimate(self, state, actions):
            return [{"action_id": a["action_id"], "status": "UNAVAILABLE", "reason": "insufficient_support",
                     "support": 0, "estimator_version": "fixture-empty-v1"} for a in actions]
    workflow, run = started(tmp_path, Unsupported())
    run = workflow.run_to_completion(run["run_id"])
    assert run["reason"] == "ESTIMATE_UNAVAILABLE"
    assert run["candidates"][0]["value"]["reason"] == "insufficient_support"
    assert any(e["type"] == "value_estimated" for e in run["events"])


def test_malformed_estimate_item_is_audited_as_unavailable(tmp_path):
    class Malformed(SyntheticBackend):
        def estimate(self, state, actions): return [None, {"action_id": actions[1]["action_id"]}]
    workflow, run = started(tmp_path, Malformed())
    result = workflow.run_to_completion(run["run_id"])
    assert result["state"] == "STOPPED" and result["reason"] == "ESTIMATE_UNAVAILABLE"

