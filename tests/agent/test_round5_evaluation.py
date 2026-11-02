from copy import deepcopy

import pytest

from football_agent.acquisition import AcquisitionWorkflow
from football_agent.acquisition_evaluation import summarize_runs, offline_stop_regret
from football_agent.research_bridge import SyntheticBackend


class Proposer:
    def __init__(self, items=None, stop=False):
        self.items, self.stop = items, stop

    def configuration(self):
        return {"configured": True, "provider": "constructed_fixture", "reserve_per_call": 0}

    def propose(self, state, actions, **kwargs):
        items = self.items if self.items is not None else [
            {"action_id": actions[-1]["action_id"], "parameters": {}},
            {"action_id": actions[-1]["action_id"], "parameters": {}}]
        return {"status": "OK", "stop": self.stop, "proposals": deepcopy(items)}


def finish(tmp_path, proposer=None, **kwargs):
    workflow = AcquisitionWorkflow(SyntheticBackend(), tmp_path, proposer=proposer)
    run = workflow.start("synthetic:alpha-beta", "2025-01-01T12:00:00Z", **kwargs)
    return workflow.run_to_completion(run["run_id"])


def test_duplicate_proposals_are_counted_without_duplicate_acquisition(tmp_path):
    run = finish(tmp_path, Proposer(), strategy="llm_gate", budget={"max_actions": 1})
    assert run["proposal_audit"]["duplicate_or_invalid_call_rate"] == 1
    assert run["proposal_audit"]["legal_parameter_rate"] == 1
    assert len(run["history"]) == 1
    audit = next(e["data"]["proposal_audit"] for e in run["events"] if e["type"] == "value_estimated")
    assert audit["proposal_recall_at_k"]["1"] == 1/3
    assert "not_realized_oracle" in audit["recall_reference"]


@pytest.mark.parametrize("items", [[3], {"action_id": "bad"}, [{"action_id": "bad", "parameters": {}}]])
def test_malformed_proposal_including_stop_is_validated(tmp_path, items):
    run = finish(tmp_path, Proposer(items, stop=True), strategy="pure_llm")
    assert run["reason"] == "ILLEGAL_LLM_PROPOSAL"
    assert run["budget"]["actions"] == 0
    assert run["proposal_audit"]["invalid_calls"] == 1


def test_numeric_and_unconfigured_llm_have_no_fabricated_proposal_metrics(tmp_path):
    numeric = finish(tmp_path / "numeric")
    missing = finish(tmp_path / "missing", strategy="pure_llm")
    assert missing["evidence"]["class"] == "pure_llm_unconfigured"
    assert missing["budget"]["llm_calls"] == 0
    summary = summarize_runs([numeric, missing])
    for group in summary["groups"]:
        assert group["actual_provider_money"] is None
        assert group["stop_regret"] is None
        assert all(v is None for v in group["proposal_recall_at_k"].values())
    assert numeric["stop_audit"]["stop_regret"] is None


def test_hindsight_regret_uses_incremental_cost_and_stop_option():
    assert offline_stop_regret(1, [{"loss": .8, "incremental_money": .1}], .5) == pytest.approx(.15)
    assert offline_stop_regret(1, [{"loss": .8, "incremental_money": 1}], .5) == 0
    with pytest.raises(ValueError):
        offline_stop_regret(1, [{"loss": .8, "incremental_money": -1}], .5)
