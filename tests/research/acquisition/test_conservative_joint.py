from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pytest

from football_research.acquisition.conservative_joint import (
    ConservativeJointReturnPredictor, minimize_strength,
)
from football_research.acquisition.data import ControlledEpisode
from football_research.acquisition.joint_returns import HYPOTHESIS_KEY
from football_research.acquisition.joint_voi import LearnedJointVOIPolicy
from football_research.evaluation.learned_voi_replay import run_learned_voi_episode
from test_joint_returns import ACTIONS, fitted, state


def calibration_rows(rows):
    copied = deepcopy(rows[:18])
    for i, row in enumerate(copied):
        kickoff = datetime(2025, 4, 5, tzinfo=timezone.utc) + timedelta(days=i)
        prediction = (kickoff-timedelta(days=1)).isoformat()
        row.update(match_id=f"cal-{i}", kickoff=kickoff.isoformat(), prediction_cutoff=prediction,
                   features_available_at=prediction, fixture_available_at=prediction,
                   result_available_at=(kickoff+timedelta(hours=3)).isoformat(),
                   dataset_version="synthetic-heldout", feature_version="synthetic-features")
    return copied


@pytest.mark.parametrize("strength", [0., .3, 1.])
def test_initial_prior_and_joint_return_marginal_are_preserved(fitted, strength):
    reference, _, _ = fitted
    original = reference.joint_distribution(state())
    model = ConservativeJointReturnPredictor(reference, strength=strength)
    adapted = model.joint_distribution(state())
    assert adapted["current_probabilities"] == original["current_probabilities"]
    np.testing.assert_allclose(np.asarray(adapted["probabilities"]).sum(axis=0),
                               np.asarray(original["probabilities"]).sum(axis=0), atol=1e-12)
    assert model.model_version != reference.model_version


@pytest.mark.parametrize("strength", [0., .3, 1.])
@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_sequential_conditioning_is_coherent_and_never_resets_prior(fitted, strength, order):
    reference, _, _ = fitted
    model = ConservativeJointReturnPredictor(reference, strength=strength)
    p0 = np.asarray(reference.predict_partial(state())["probabilities"])
    for first in model.enumerate_action(state(), ACTIONS[order[0]])["branches"]:
        branch_state = first["next_state"]
        original = np.asarray(reference.predict_hypothetical(branch_state)["probabilities"])
        np.testing.assert_allclose(first["posterior_probabilities"], (1-strength)*p0+strength*original, atol=1e-12)
        next_query = model.enumerate_action(branch_state, ACTIONS[order[1]], hypothetical=True)
        mean = sum(b["probability"]*np.asarray(b["posterior_probabilities"]) for b in next_query["branches"])
        np.testing.assert_allclose(mean, first["posterior_probabilities"], atol=1e-12)
        for second in next_query["branches"]:
            original = np.asarray(reference.predict_hypothetical(second["next_state"])["probabilities"])
            np.testing.assert_allclose(second["posterior_probabilities"], (1-strength)*p0+strength*original, atol=1e-12)
        assert next_query["evsi"] >= -1e-12
        if strength == 0.:
            assert abs(next_query["evsi"]) < 1e-12


def test_zero_strength_means_no_positive_value_but_no_open_world_certificate(fitted):
    reference, _, _ = fitted
    model = ConservativeJointReturnPredictor(reference, strength=0.)
    policy = LearnedJointVOIPolicy(model, lambda_cost={"money": 1.})
    decision = policy.plan(state(), ACTIONS, {"limits": {"money": .05, "requests": 2}, "spent": {}})
    assert decision["selection"]["decision"] == "defer"
    assert decision["selection"]["stop_certificate"] is None
    assert all(abs(c["value"]["gross_evsi"]) < 1e-12 for c in decision["candidates"])


@pytest.mark.parametrize("strength", [-.01, 1.01, float("nan"), float("inf"), True, "0.5"])
def test_invalid_strength_rejected(fitted, strength):
    with pytest.raises(ValueError, match="strength"):
        ConservativeJointReturnPredictor(fitted[0], strength=strength)


def test_strength_minimization_has_correct_endpoints_and_interior():
    p = [[.5]*3]*2
    assert minimize_strength(p, p)["strength"] == 0.
    assert minimize_strength(p, [[.8]*3]*2)["strength"] == 1.
    assert minimize_strength(p, [[.3]*3]*2)["strength"] == 0.
    result = minimize_strength(p, [[.9]*3, [.25]*3])
    assert result["strength"] == pytest.approx(.375, abs=1e-12)
    assert result["selected_objective"] < min(result["objective_at_zero"], result["objective_at_one"])
    assert result["derivative_at_selected"] == pytest.approx(0., abs=1e-12)


@pytest.mark.parametrize("probabilities", [[], [[.4]], [[0., .3, .3]], [[float("nan"), .3, .3]]])
def test_optimizer_rejects_invalid_sample_arrays(probabilities):
    with pytest.raises(ValueError, match="probabilities"):
        minimize_strength(probabilities, probabilities)


def test_heldout_fit_is_bound_to_same_reference_and_time(fitted):
    reference, rows, _ = fitted
    reference_payload = deepcopy(reference._payload())
    heldout = calibration_rows(rows)
    model = ConservativeJointReturnPredictor.fit_strength(reference, heldout, ACTIONS,
                                                         as_of="2025-04-25T00:00:00Z")
    selection = model.selection_report
    assert selection["optimization"]["unique_matches"] == len(heldout)
    assert selection["reference_model_version"] == reference.model_version
    assert model.metadata["training_as_of"] == "2025-04-25T00:00:00+00:00"
    assert selection["optimization"]["selected_objective"] <= selection["optimization"]["objective_at_zero"]+1e-12
    assert reference._payload() == reference_payload
    assert model.predict_partial(state())["probabilities"] == reference.predict_partial(state())["probabilities"]
    changed = model.selection_report
    changed["optimization"]["strength"] = -1
    assert model.evidence_strength >= 0.
    old = ControlledEpisode(heldout[0], ACTIONS).visible_state()
    with pytest.raises(ValueError, match="time boundary"):
        model.predict_partial(old)
    old.update(prediction_cutoff=state()["prediction_cutoff"], base=state()["base"])
    with pytest.raises(ValueError, match="Adaptation match"):
        model.predict_partial(old)


@pytest.mark.parametrize("fault", ["train_overlap", "duplicate", "future_label", "early_prediction", "late_features", "mode", "score", "unknown_identity"])
def test_fit_rejects_invalid_supervision(fitted, fault):
    reference, rows, _ = fitted
    heldout = calibration_rows(rows)
    if fault == "train_overlap":
        heldout[0]["match_id"] = rows[0]["match_id"]
    elif fault == "duplicate":
        heldout[0]["match_id"] = heldout[1]["match_id"]
    elif fault == "future_label":
        heldout[0]["result_available_at"] = "2025-06-01T00:00:00Z"
    elif fault == "early_prediction":
        heldout[0]["prediction_cutoff"] = "2025-03-31T00:00:00Z"
    elif fault == "late_features":
        heldout[0]["features_available_at"] = heldout[0]["kickoff"]
    elif fault == "mode":
        heldout[0]["information_mode"] = "operational"
    elif fault == "score":
        heldout[0]["home_goals"] = True
    elif fault == "unknown_identity":
        heldout[0]["home_team_id"] = "unmapped"
    with pytest.raises(ValueError):
        ConservativeJointReturnPredictor.fit_strength(reference, heldout, ACTIONS, as_of="2025-04-25T00:00:00Z")


def test_round_trip_and_tamper_detection(fitted, tmp_path):
    reference, rows, _ = fitted
    model = ConservativeJointReturnPredictor.fit_strength(reference, calibration_rows(rows), ACTIONS,
                                                         as_of="2025-04-25T00:00:00Z")
    path = tmp_path/"adapted.json"
    model.save(path)
    restored = ConservativeJointReturnPredictor.load(path)
    assert restored.enumerate_action(state(), ACTIONS[0]) == model.enumerate_action(state(), ACTIONS[0])
    value = json.loads(path.read_text(encoding="utf-8"))
    value["evidence_strength"] = .333
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="bound|hash"):
        ConservativeJointReturnPredictor.load(path)


def test_reference_information_cutoff_equality_matches_existing_fold_contract(fitted):
    reference, rows, _ = fitted
    heldout = calibration_rows(rows)
    heldout[0].update(prediction_cutoff="2025-04-01T00:00:00Z",
                      fixture_available_at="2025-04-01T00:00:00Z",
                      features_available_at="2025-04-01T00:00:00Z")
    model = ConservativeJointReturnPredictor.fit_strength(reference, heldout, ACTIONS,
                                                         as_of="2025-04-25T00:00:00Z")
    assert model.selection_report["optimization"]["unique_matches"] == 18


def test_adapted_policy_real_updates_and_outcome_isolation(fitted):
    reference, rows, _ = fitted
    model = ConservativeJointReturnPredictor(reference, strength=.5)
    policy = LearnedJointVOIPolicy(model, lambda_cost={"money": 0.})
    row = calibration_rows(rows)[2]
    trajectory = run_learned_voi_episode(row, policy, ACTIONS, limits={"requests": 2, "money": .05})
    assert trajectory["actions"]
    changed = {**row, "home_goals": 999, "away_goals": 0}
    assert run_learned_voi_episode(changed, policy, ACTIONS, limits={"requests": 2, "money": .05}) == trajectory
    for cycle in trajectory["cycles"]:
        if cycle["update"]:
            assert cycle["update"]["realized_loss_reduction"] is None
            assert cycle["update"]["prediction_after"] == model.predict_partial(cycle["state_after"])["probabilities"]
    branch = model.enumerate_action(state(), ACTIONS[0])["branches"][0]["next_state"]
    assert HYPOTHESIS_KEY in branch
    with pytest.raises(ValueError, match="Hypothetical"):
        model.predict_partial(branch)
