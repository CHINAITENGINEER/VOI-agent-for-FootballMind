"""Independent synthetic regressions for public boundaries and fair accounting."""
import json

from football_agent.acquisition import AcquisitionWorkflow
from football_agent.research_bridge import SyntheticBackend


class StopProposer:
    def configuration(self):
        return {"configured": True, "model": "synthetic-stop", "version": "test-v1", "reserve_per_call": 0}

    def propose(self, *_args, **_kwargs):
        return {"status": "OK", "stop": True, "proposals": [], "reason": "Synthetic stop proposal", "usage": {"unit": "tokens", "total_count": 1}}


def start(tmp_path, backend=None, **kwargs):
    workflow = AcquisitionWorkflow(backend or SyntheticBackend(), tmp_path, kwargs.pop("proposer", None))
    run = workflow.start("synthetic:alpha-beta", "2025-01-01T12:00:00Z", **kwargs)
    return workflow, run


def test_llm_stop_still_measures_omitted_positive_actions(tmp_path):
    workflow, run = start(tmp_path, proposer=StopProposer(), strategy="llm_gate")
    run = workflow.run_to_completion(run["run_id"])
    events = [event for event in run["events"] if event["type"] == "value_estimated"]
    assert events, "Hybrid LLM stop must still enumerate candidate values for omission measurement"
    assert "fixture.reveal.strength" in events[-1]["data"]["positive_actions_omitted_by_proposer"]
    assert run["budget"]["actions"] == 0


def test_pure_llm_stop_remains_an_explicit_comparator(tmp_path):
    workflow, run = start(tmp_path, proposer=StopProposer(), strategy="pure_llm")
    run = workflow.run_to_completion(run["run_id"])
    assert run["reason"] == "LLM_STOP"
    assert not [event for event in run["events"] if event["type"] == "value_estimated"]


def test_nested_catalog_label_rejected_before_estimation_and_persistence(tmp_path):
    class Leaky(SyntheticBackend):
        estimates = 0
        def actions(self, state):
            result = super().actions(state)
            result[0]["metadata"] = {"target_label": "PRIVATE_CATALOG_SENTINEL"}
            return result
        def estimate(self, *args):
            self.estimates += 1
            return super().estimate(*args)
    backend = Leaky()
    workflow, run = start(tmp_path, backend)
    run = workflow.run_to_completion(run["run_id"])
    assert run["state"] == "DEFERRED"
    assert backend.estimates == 0
    assert all("PRIVATE_CATALOG_SENTINEL" not in path.read_text(encoding="utf-8") for path in tmp_path.rglob("*.json"))


def test_acquisition_extra_hidden_payload_rejected_before_persistence(tmp_path):
    class Leaky(SyntheticBackend):
        def acquire(self, *args):
            result = super().acquire(*args)
            result["hidden_groups"] = {"rolling_xg": "PRIVATE_RETURN_SENTINEL"}
            return result
    workflow, run = start(tmp_path, Leaky())
    run = workflow.run_to_completion(run["run_id"])
    assert run["state"] == "DEFERRED"
    assert all("PRIVATE_RETURN_SENTINEL" not in path.read_text(encoding="utf-8") for path in tmp_path.rglob("*.json"))


def test_actual_simulated_fee_above_reserve_is_reconciled_and_stops(tmp_path):
    class Overrun(SyntheticBackend):
        calls = 0
        def acquire(self, *args):
            self.calls += 1
            result = super().acquire(*args)
            result["cost"] = {"status": "simulated", "unit": "simulation_credit", "amount": 2.5}
            return result
    backend = Overrun()
    workflow, run = start(tmp_path, backend, budget={"max_simulated_cost": 2.0})
    run = workflow.run_to_completion(run["run_id"])
    assert backend.calls == 1, "An actual cost overrun must prevent the second charged action"
    assert run["budget"]["simulated_cost"] == 2.5
    assert run["budget"]["online_reserve"] == 0, "Simulated fee must stay separate from online reservations"
