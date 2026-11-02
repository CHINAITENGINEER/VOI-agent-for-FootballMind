from copy import deepcopy
import math

import numpy as np
import pytest

from football_research.acquisition.joint_planner import JointRiskPlanner


class TableJointModel:
    """Constructed joint law used solely to test the numerical planner."""
    model_version = "constructed-joint-table"
    metadata = {"training_as_of": "2025-01-01T00:00:00Z", "dataset_version": "constructed"}

    def __init__(self, probabilities, *, missing=False):
        self.table = np.asarray(probabilities, dtype=float)
        assert self.table.shape == (3, 2, 2)
        assert self.table.sum() == pytest.approx(1)
        self.categories = ("0", "missing" if missing else "1")

    def observed_actions(self, state, *, hypothetical=False):
        return list(self._observations(state, hypothetical))

    def _observations(self, state, hypothetical):
        if not hypothetical and "joint_return_hypothesis" in state:
            raise ValueError("Hypothetical tokens rejected")
        return state.get("joint_return_hypothesis", {}).get("observations", {}) if hypothetical else state.get("actual_observations", {})

    def conditional(self, state, hypothetical=False):
        table = self.table.copy()
        for aid, token in self._observations(state, hypothetical).items():
            axis = 1 if aid == "a" else 2
            index = self.categories.index(token)
            slices = [slice(None)] * 3
            slices[axis] = 1 - index
            table[tuple(slices)] = 0
        return table / table.sum()

    def predict_partial(self, state):
        return {"probabilities": self.conditional(state).sum(axis=(1, 2)).tolist()}

    def predict_hypothetical(self, state):
        return {"probabilities": self.conditional(state, True).sum(axis=(1, 2)).tolist()}

    def enumerate_action(self, state, action, *, hypothetical=False):
        aid, table = action["action_id"], self.conditional(state, hypothetical)
        branches = []
        for index, token in enumerate(self.categories):
            joint = table[:, index, :] if aid == "a" else table[:, :, index]
            probability = float(joint.sum())
            if probability <= 0:
                continue
            posterior = joint.sum(axis=1) / probability
            observations = {**self._observations(state, hypothetical), aid: token}
            branches.append({"observation": token, "probability": probability,
                             "posterior_probabilities": posterior.tolist(),
                             "next_state": {"joint_return_hypothesis": {"schema": "constructed", "hypothetical": True, "observations": observations}},
                             "return_status": "summary_missing" if token == "missing" else "summary_observed"})
        return {"action_id": aid, "current_probabilities": table.sum(axis=(1, 2)).tolist(), "branches": branches}


def xor_model():
    table = np.zeros((3, 2, 2))
    for a in range(2):
        for b in range(2):
            table[0 if a == b else 2, a, b] = .25
    return TableJointModel(table)


def informative_model(*, correlated=False, missing=False):
    table = np.zeros((3, 2, 2))
    if correlated:
        table[0, 0, 0], table[2, 1, 1] = .5, .5
    else:
        table[0, 0, :], table[2, 1, :] = .25, .25
    return TableJointModel(table, missing=missing)


def actions():
    return [{"action_id": aid, "group_ids": [aid], "available": True,
             "cost": {"money": .02, "requests": 1, "latency_ms": 0, "tokens": 0}, "cost_status": "simulated"}
            for aid in ("a", "b")]


def budget(money=.04, requests=2):
    return {"limits": {"money": money, "requests": requests}, "spent": {"money": 0, "requests": 0}}


def test_one_step_evsi_equals_mutual_information_and_posterior_martingale():
    model = informative_model()
    decision = JointRiskPlanner(model, lambda_cost={"money": 1}).plan({}, actions(), budget())
    first = decision["candidates"][0]
    assert decision["action_id"] == "a"
    assert first["immediate_evsi"] == pytest.approx(math.log(2))
    assert first["posterior_martingale_max_error"] < 1e-12
    assert first["planning_net_gain"] == pytest.approx(math.log(2) - .02)
    tensor = model.table
    yz = tensor.sum(axis=2)
    py, pz = yz.sum(axis=1), yz.sum(axis=0)
    information = sum(yz[y, z] * math.log(yz[y, z] / (py[y] * pz[z])) for y in range(3) for z in range(2) if yz[y, z])
    assert first["immediate_evsi"] == pytest.approx(information)


def test_hypothesis_entropy_without_target_effect_has_no_value():
    table = np.empty((3, 2, 2))
    for y, py in enumerate((.45, .3, .25)):
        table[y, :, :] = py / 4
    decision = JointRiskPlanner(TableJointModel(table), horizon=2, lambda_cost={"money": 1}).plan({}, actions(), budget())
    assert decision["decision"] == "stop"
    assert all(c["immediate_evsi"] == pytest.approx(0) for c in decision["candidates"])


def test_two_step_complementarity_integrates_branch_specific_continuation():
    model = xor_model()
    one = JointRiskPlanner(model, horizon=1, lambda_cost={"money": 1}).plan({}, actions(), budget())
    two = JointRiskPlanner(model, horizon=2, lambda_cost={"money": 1}).plan({}, actions(), budget())
    assert one["decision"] == "stop"
    assert two["action_id"] == "a"
    candidate = two["candidates"][0]
    assert candidate["immediate_evsi"] == pytest.approx(0)
    assert candidate["expected_plan_cost_vector"]["money"] == pytest.approx(.04)
    assert candidate["expected_plan_cost_vector"]["requests"] == pytest.approx(2)
    assert candidate["planning_net_gain"] == pytest.approx(math.log(2) - .04)
    assert all(b["continuation_action"] == "b" and b["hypothetical"] for b in candidate["branches"])


@pytest.mark.parametrize("physical_budget", [budget(.02, 2), budget(.04, 1), budget(0, 2), budget(.04, 0)])
def test_zero_lambda_never_relaxes_money_or_request_constraints(physical_budget):
    decision = JointRiskPlanner(xor_model(), horizon=2, lambda_cost={"money": 0}).plan({}, actions(), physical_budget)
    assert decision["decision"] == "stop"


def test_zero_lambda_can_use_affordable_information_without_rescaling_budget():
    decision = JointRiskPlanner(informative_model(), horizon=2, lambda_cost={"money": 0}).plan({}, actions(), budget(.02, 1))
    assert decision["action_id"] == "a"
    assert decision["expected_plan_cost_vector"]["money"] == .02
    assert decision["expected_plan_penalty"] == 0


def test_joint_dependence_and_duplicate_identity_prevent_double_counting():
    model = informative_model(correlated=True)
    after = {"actual_observations": {"a": "0"}}
    decision = JointRiskPlanner(model, horizon=2, lambda_cost={"money": 1}).plan(after, actions(), budget())
    assert decision["decision"] == "stop"
    assert decision["candidates"][0]["reason"] == "already_attempted"
    assert decision["candidates"][1]["immediate_evsi"] == pytest.approx(0)
    duplicate_actions = actions()
    for action in duplicate_actions:
        action["evidence_id"] = "same-observation"
    assert JointRiskPlanner(xor_model(), horizon=2, lambda_cost={"money": 1}).plan({}, duplicate_actions, budget())["decision"] == "stop"


def test_missing_is_a_modeled_response_but_provider_failure_is_not_relabelled():
    model = informative_model(missing=True)
    decision = JointRiskPlanner(model, lambda_cost={"money": 1}).plan({}, actions(), budget())
    assert decision["candidates"][0]["branches"][1]["return_status"] == "summary_missing"
    original = model.enumerate_action
    def failed(*args, **kwargs):
        value = original(*args, **kwargs)
        value["branches"][0]["return_status"] = "provider_error"
        return value
    model.enumerate_action = failed
    with pytest.raises(ValueError, match="Provider failures"):
        JointRiskPlanner(model).plan({}, actions(), budget())


def test_failed_attempt_is_not_retried_in_one_shot_planning():
    state = {"action_history": [{"action_id": "a", "status": "FAILED"}]}
    decision = JointRiskPlanner(informative_model()).plan(state, actions(), budget())
    assert decision["candidates"][0]["reason"] == "already_attempted"


def test_real_entry_rejects_user_supplied_hypothetical_tokens():
    with pytest.raises(ValueError, match="rejects hypothetical"):
        JointRiskPlanner(xor_model()).plan({"joint_return_hypothesis": {"hypothetical": True}}, actions(), budget())


def test_enumeration_must_agree_with_current_prediction_and_posteriors():
    model = informative_model()
    original = model.enumerate_action
    def inconsistent(*args, **kwargs):
        value = original(*args, **kwargs)
        value["current_probabilities"] = [1, 0, 0]
        return value
    model.enumerate_action = inconsistent
    with pytest.raises(ValueError, match="different current prediction"):
        JointRiskPlanner(model).plan({}, actions(), budget())


@pytest.mark.parametrize("bad", [budget(-1, 2), {"limits": {"money": .05}, "spent": {}},
                                  {"limits": {"money": .01, "requests": 2}, "spent": {"money": .02}}])
def test_invalid_physical_budgets_rejected(bad):
    with pytest.raises(ValueError):
        JointRiskPlanner(xor_model()).plan({}, actions(), bad)


def test_unknown_cost_and_duplicate_actions_rejected():
    raw = actions()
    raw[0]["cost_status"] = "unknown"
    with pytest.raises(ValueError, match="declared"):
        JointRiskPlanner(xor_model()).plan({}, raw, budget())
    with pytest.raises(ValueError, match="unique"):
        JointRiskPlanner(xor_model()).plan({}, actions() + [actions()[0]], budget())
