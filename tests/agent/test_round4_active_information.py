"""Round-four regressions for proposal recall, risk gating and recovery."""
from copy import deepcopy
import json

from football_agent.acquisition import AcquisitionWorkflow
from football_agent.proposer import LLMProposer
from football_agent.research_bridge import SyntheticBackend
from football_agent.audit import stored_digest


class OneCandidateProposer:
    """Protocol fixture: proposes one legal candidate and never supplies value."""

    def __init__(self, preferred="fixture.reveal.xg"):
        self.preferred = preferred
        self.states = []

    def configuration(self):
        return {"configured": True, "provider": "fixture", "model": "proposal-fixture-v1",
                "prompt_version": "fixture-prompt-v1", "reserve_per_call": 0, "version": "fixture-v1"}

    def propose(self, state, actions, **_kwargs):
        self.states.append(deepcopy(state))
        chosen = next((a for a in actions if a["action_id"] == self.preferred), actions[0])
        return {"status": "OK", "stop": False, "proposals": [{"action_id": chosen["action_id"],
                "parameters": deepcopy(chosen.get("parameters", {})), "reason": "The catalog exposes this candidate."}],
                "reason": "Candidate proposal only; numeric gate decides.", "reason_is_value_estimate": False,
                "usage": {"status": "simulated", "unit": "tokens", "total_count": 0}}


class LowerBoundBackend(SyntheticBackend):
    def __init__(self, lower_offset=0.001):
        super().__init__()
        self.lower_offset = lower_offset

    def estimate(self, visible, actions):
        values = super().estimate(visible, actions)
        for value in values:
            value["lower_gain"] = value["expected_loss_improvement"] - self.lower_offset
        return values


def start(tmp_path, backend=None, proposer=None, **kwargs):
    workflow = AcquisitionWorkflow(backend or SyntheticBackend(), tmp_path, proposer=proposer)
    run = workflow.start("synthetic:alpha-beta", "2025-01-01T12:00:00Z", **kwargs)
    return workflow, run


def test_llm_proposal_recall_and_numeric_gate_are_audited(tmp_path):
    proposer = OneCandidateProposer()
    workflow, run = start(tmp_path, proposer=proposer, strategy="llm-proposal+numeric-gate")
    final = workflow.run_to_completion(run["run_id"])
    estimates = [event["data"] for event in final["events"] if event["type"] == "value_estimated"]
    assert estimates
    audit = estimates[0]["proposal_audit"]
    assert audit["risk_mode"] == "mean"
    assert audit["positive_count"] == 3
    assert audit["positive_recall"] == 1 / 3
    assert set(audit["positive_omitted"]) == {"fixture.reveal.strength", "fixture.reveal.form"}
    assert all(event["data"].get("reason_is_value_estimate") is not True
               for event in final["events"] if event["type"] == "llm_proposed")
    assert final["history"][0]["action_id"] == "fixture.reveal.xg"


def test_conservative_lower_bound_can_stop_an_mean_positive_action(tmp_path):
    workflow, run = start(tmp_path, backend=LowerBoundBackend(lower_offset=0.04),
                          strategy="numeric-all-candidates", risk_mode="conservative")
    final = workflow.run_to_completion(run["run_id"])
    assert final["state"] == "STOPPED"
    assert final["reason"] == "VALUE_STOP_NONPOSITIVE_CONSERVATIVE_GAIN"
    value_event = next(event for event in final["events"] if event["type"] == "value_estimated")
    values = value_event["data"]["candidates"]
    strength = next(item["value"] for item in values if item["action_id"] == "fixture.reveal.strength")
    assert strength["net_gain"] > 0 and strength["conservative_net_gain"] <= 0
    assert final["budget"]["actions"] == 0


def test_no_data_reselects_another_legal_action(tmp_path):
    backend = SyntheticBackend("no_data")
    workflow, run = start(tmp_path, backend=backend, strategy="numeric-all-candidates")
    final = workflow.run_to_completion(run["run_id"])
    action_ids = [entry["action_id"] for entry in final["history"]]
    assert action_ids[0] == "fixture.reveal.strength"
    assert len(action_ids) >= 2 and action_ids[1] != action_ids[0]
    assert final["budget"]["actions"] == len(action_ids)


def test_hidden_value_and_label_do_not_reach_first_llm_proposal(tmp_path):
    class HiddenAfterAcquire(SyntheticBackend):
        def __init__(self, value, label):
            super().__init__()
            self.value, self.label = value, label

        def acquire(self, *args):
            result = super().acquire(*args)
            for group in result["visible_state"]["groups"].values():
                group["fixture_value"] = self.value
            return result

        def predict(self, state):
            result = super().predict(state)
            if state["groups"]:
                value = next(iter(state["groups"].values()))["fixture_value"]
                result["probabilities"] = {"home": value, "draw": (1 - value) / 2, "away": (1 - value) / 2}
            return result

    first_states, first_actions, after_predictions = [], [], []
    for index, (value, label) in enumerate(((0.2, "home"), (0.8, "away"))):
        proposer = OneCandidateProposer()
        workflow, run = start(tmp_path / str(index), backend=HiddenAfterAcquire(value, label), proposer=proposer,
                              strategy="llm-proposal+numeric-gate")
        run = workflow.step(run["run_id"])  # initial partial prediction
        run = workflow.step(run["run_id"])  # first proposal and value gate
        first_states.append(proposer.states[0])
        first_actions.append(run["history"][0]["action_id"])
        run = workflow.step(run["run_id"])
        after_predictions.append(run["prediction"]["probabilities"])
    def stable_public_input(state):
        value = deepcopy(state)
        value["budget"].pop("elapsed_seconds", None)
        value["budget"].pop("compute_calls", None)
        value.get("current_prediction", {}).pop("compute_cost", None)
        value.get("current_prediction", {}).pop("state_hash", None)
        return value

    assert stable_public_input(first_states[0]) == stable_public_input(first_states[1])
    assert first_actions[0] == first_actions[1]
    assert after_predictions[0] != after_predictions[1]


def test_provider_response_hash_is_persisted_without_raw_response(tmp_path):
    state = {"episode_id": "e1", "match_id": "m1", "prediction_cutoff": "2026-01-01T00:00:00Z",
             "information_mode": "historical_reconstruction", "base_facts": {"home_team": "A"},
             "groups": {}, "mask": {"strength": False}, "action_history": [], "budget": {"requests": 5}}
    actions = [{"action_id": "reveal:strength", "group_ids": ["strength"], "status": "available",
                "parameters_schema": {"type": "object", "properties": {}}}]
    answer = {"proposals": [{"action_id": "reveal:strength", "parameters": {}, "reason": "unrevealed"}],
              "stop": False}
    raw = {"id": "provider-request", "model": "fixture-model",
           "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
           "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(answer)}}]}
    proposer = LLMProposer(model="fixture-model", api_key="fixture-key", ledger_path=tmp_path / "ledger.json",
                            transport=lambda _payload: raw)
    result = proposer.propose(state, actions)
    assert result["status"] == "OK"
    assert result["provider_response_sha256"] == stored_digest(raw)
    ledger = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    serialized = json.dumps(ledger, ensure_ascii=False)
    assert "choices" not in serialized and "provider-key" not in serialized


def test_alias_names_are_normalized_and_recorded(tmp_path):
    workflow, run = start(tmp_path, strategy="fixed-order")
    assert run["strategy"] == "fixed"
    assert run["risk_mode"] == "mean"
