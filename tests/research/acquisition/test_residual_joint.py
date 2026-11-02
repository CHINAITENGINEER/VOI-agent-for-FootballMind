from copy import deepcopy
import json

import numpy as np
import pytest

from football_research.acquisition.conservative_joint import ConservativeJointReturnPredictor
from football_research.acquisition.contextual_joint import ContextualJointReturnPredictor
from football_research.acquisition.joint_voi import LearnedJointVOIPolicy
from football_research.acquisition.residual_joint import ResidualJointReturnPredictor, minimize_residual_strength
from football_research.evaluation.learned_voi_replay import run_learned_voi_episode
from test_conservative_joint import calibration_rows
from test_joint_returns import ACTIONS, fitted, state


@pytest.mark.parametrize("eta", [0., .4, 1.])
def test_initial_marginals_and_first_results_prediction_are_preserved(fitted, eta):
    reference, _, _ = fitted
    model = ResidualJointReturnPredictor(reference, eta=eta)
    original = np.asarray(reference.joint_distribution(state())["probabilities"])
    adapted = np.asarray(model.joint_distribution(state())["probabilities"])
    np.testing.assert_allclose(adapted.sum(axis=0), original.sum(axis=0), atol=1e-12)
    np.testing.assert_allclose(adapted.sum(axis=2), original.sum(axis=2), atol=1e-12)
    assert model.predict_partial(state())["probabilities"] == reference.predict_partial(state())["probabilities"]
    for branch in model.enumerate_action(state(), ACTIONS[0])["branches"]:
        np.testing.assert_allclose(branch["posterior_probabilities"],
                                   reference.predict_hypothetical(branch["next_state"])["probabilities"], atol=1e-12)
    if eta == 1.:
        np.testing.assert_allclose(adapted, original, atol=1e-12)


@pytest.mark.parametrize("eta", [0., .4, 1.])
def test_both_orders_condition_same_joint_and_satisfy_tower(fitted, eta):
    model = ResidualJointReturnPredictor(fitted[0], eta=eta)
    posteriors = {}
    for order in ((0, 1), (1, 0)):
        for first in model.enumerate_action(state(), ACTIONS[order[0]])["branches"]:
            second = model.enumerate_action(first["next_state"], ACTIONS[order[1]], hypothetical=True)
            mean = sum(b["probability"]*np.asarray(b["posterior_probabilities"]) for b in second["branches"])
            np.testing.assert_allclose(mean, first["posterior_probabilities"], atol=1e-12)
            assert second["evsi"] >= -1e-12
            for branch in second["branches"]:
                categories = model.observed_categories(branch["next_state"], hypothetical=True)
                key = (categories["results_view"], categories["xg_view"])
                if key in posteriors:
                    np.testing.assert_allclose(branch["posterior_probabilities"], posteriors[key], atol=1e-12)
                posteriors[key] = branch["posterior_probabilities"]


def test_zero_residual_removes_only_conditional_value_not_xg_tool(fitted):
    model = ResidualJointReturnPredictor(fitted[0], eta=0.)
    # Direct X can still reveal the latent R that is informative about Y.
    assert model.enumerate_action(state(), ACTIONS[1])["evsi"] > 1e-4
    for result in model.enumerate_action(state(), ACTIONS[0])["branches"]:
        after = model.enumerate_action(result["next_state"], ACTIONS[1], hypothetical=True)
        assert abs(after["evsi"]) < 1e-12
        assert after["branches"]


def test_convex_objective_has_endpoints_and_known_interior_optimum():
    assert minimize_residual_strength([.5, .5], [.5, .5])["eta"] == 0.
    assert minimize_residual_strength([.5, .5], [.8, .9])["eta"] == 1.
    assert minimize_residual_strength([.5, .5], [.2, .3])["eta"] == 0.
    result = minimize_residual_strength([.5, .5], [.9, .25])
    assert result["eta"] == pytest.approx(.375, abs=1e-12)
    assert result["selected_objective"] < min(result["objective_at_zero"], result["objective_at_one"])


@pytest.mark.parametrize("kind", ["global", "contextual"])
def test_fit_uses_only_bound_past_validation_and_roundtrips(fitted, tmp_path, kind):
    reference, rows, _ = fitted
    heldout = calibration_rows(rows)
    cutoff = "2025-04-25T00:00:00Z"
    if kind == "global":
        reference = ConservativeJointReturnPredictor.fit_strength(reference, heldout, ACTIONS, as_of=cutoff)
    else:
        reference = ContextualJointReturnPredictor.fit_context(reference, rows, heldout, ACTIONS, as_of=cutoff)
    original = deepcopy(reference._payload())
    model = ResidualJointReturnPredictor.fit_residual(reference, as_of=cutoff)
    assert reference._payload() == original
    selection = model.selection_report
    assert selection["optimization"]["selected_objective"] <= selection["optimization"]["objective_at_zero"]+1e-12
    assert selection["source_validation_reused_for_multiple_parameters"] is True
    assert selection["costs_or_query_thresholds_changed"] is False
    assert set(selection["match_ids"]) == set(reference.selection_report["match_ids"])
    forbidden = state()
    forbidden["match_id"] = heldout[0]["match_id"]
    with pytest.raises(ValueError, match="adaptation match"):
        model.predict_partial(forbidden)
    with pytest.raises(ValueError, match="cutoff"):
        ResidualJointReturnPredictor.fit_residual(reference, as_of="2025-06-01T00:00:00Z")
    path = tmp_path/"residual.json"
    model.save(path)
    loaded = ResidualJointReturnPredictor.load(path)
    assert loaded.joint_distribution(state()) == model.joint_distribution(state())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["eta"] = .127591
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="bound|hash"):
        ResidualJointReturnPredictor.load(path)


def test_runtime_hidden_values_labels_and_actual_update_boundary(fitted):
    class Poison:
        def __deepcopy__(self, memo):
            raise AssertionError("Hidden payload copied")

        def __str__(self):
            raise AssertionError("Hidden payload inspected")

    reference, rows, _ = fitted
    model = ResidualJointReturnPredictor(reference, eta=.5)
    dirty = state()
    dirty["outcome"] = Poison()
    for group in dirty["groups"].values():
        group.update(values=Poison(), reason=Poison())
    assert model.joint_distribution(dirty) == model.joint_distribution(state())
    assert model.observation_support(state())["kernel_effective_matches"] == reference.observation_support(state())["kernel_effective_matches"]
    policy = LearnedJointVOIPolicy(model, lambda_cost={"money": 0.})
    row = calibration_rows(rows)[2]
    trace = run_learned_voi_episode(row, policy, ACTIONS, limits={"requests": 2, "money": .05})
    changed = {**row, "home_goals": 999, "away_goals": 0}
    assert trace == run_learned_voi_episode(changed, policy, ACTIONS, limits={"requests": 2, "money": .05})
    assert trace["actions"]
    for cycle in trace["cycles"]:
        if cycle["update"]:
            assert cycle["update"]["prediction_after"] == model.predict_partial(cycle["state_after"])["probabilities"]


def test_fit_refuses_declared_mechanism_without_supervision(fitted):
    reference = ConservativeJointReturnPredictor(fitted[0], strength=.5)
    with pytest.raises(ValueError, match="cutoff"):
        ResidualJointReturnPredictor.fit_residual(reference, as_of=reference.metadata["training_as_of"])


@pytest.mark.parametrize("eta", [-.1, 1.1, float("nan"), True])
def test_invalid_eta_rejected(fitted, eta):
    with pytest.raises(ValueError, match="eta"):
        ResidualJointReturnPredictor(fitted[0], eta=eta)
