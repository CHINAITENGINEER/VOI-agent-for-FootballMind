from copy import deepcopy
import json

import numpy as np
import pytest

from football_research.acquisition.contextual_joint import ContextualJointReturnPredictor
from football_research.acquisition.conservative_joint import ConservativeJointReturnPredictor
from football_research.acquisition.joint_voi import LearnedJointVOIPolicy
from football_research.evaluation.learned_voi_replay import run_learned_voi_episode
from test_conservative_joint import calibration_rows
from test_joint_returns import ACTIONS, fitted, state


@pytest.mark.parametrize("strength", [0., .5, 1.])
def test_zero_context_reproduces_existing_coherent_strength(fitted, strength):
    reference, rows, _ = fitted
    previous = ConservativeJointReturnPredictor(reference, strength=strength)
    model = ContextualJointReturnPredictor(reference, rows, context_weight=0., strength=strength)
    before = state()
    np.testing.assert_allclose(model.joint_distribution(before)["probabilities"],
                               previous.joint_distribution(before)["probabilities"], atol=1e-14)
    for branch in model.enumerate_action(before, ACTIONS[0])["branches"]:
        np.testing.assert_allclose(model.predict_hypothetical(branch["next_state"])["probabilities"],
                                   previous.predict_hypothetical(branch["next_state"])["probabilities"], atol=1e-14)


def test_equal_prior_different_public_fixture_context_changes_likelihood(fitted):
    reference, rows, _ = fitted
    model = ContextualJointReturnPredictor(reference, rows)
    p0 = [.45, .3, .25]
    familiar = model._context(state()["base"])
    reversed_teams = model._context({**state()["base"], "home_team_id": "t1", "away_team_id": "t0"})
    first, first_support = model._context_q(p0, familiar)
    second, second_support = model._context_q(p0, reversed_teams)
    assert np.max(np.abs(first-second)) > .001
    assert first_support["local_matching_training_matches"] == len(rows)
    assert second_support["local_matching_training_matches"] == 0
    assert second_support["local_backoff"] == "population"
    np.testing.assert_allclose(second, reference._q(p0)[0], atol=1e-14)


@pytest.mark.parametrize("strength", [0., .7, 1.])
def test_one_joint_preserves_prior_sequential_martingale_and_query_order(fitted, strength):
    reference, rows, _ = fitted
    model = ContextualJointReturnPredictor(reference, rows, strength=strength)
    before = model.joint_distribution(state())
    assert before["current_probabilities"] == reference.predict_partial(state())["probabilities"]
    for first in model.enumerate_action(state(), ACTIONS[0])["branches"]:
        observed = model.joint_distribution(first["next_state"], hypothetical=True)
        assert observed["q_support"] == before["q_support"]
        assert observed["prior_probabilities"] == before["prior_probabilities"]
        second = model.enumerate_action(first["next_state"], ACTIONS[1], hypothetical=True)
        expected = sum(b["probability"]*np.asarray(b["posterior_probabilities"]) for b in second["branches"])
        np.testing.assert_allclose(expected, first["posterior_probabilities"], atol=1e-12)
        assert second["evsi"] >= -1e-12
        assert second["evsi"] == pytest.approx(second["conditional_mutual_information"], abs=1e-12)
        for branch in second["branches"]:
            np.testing.assert_allclose(model.predict_hypothetical(branch["next_state"])["probabilities"],
                                       branch["posterior_probabilities"], atol=1e-12)
            if strength == 0.:
                np.testing.assert_allclose(branch["posterior_probabilities"], before["prior_probabilities"], atol=1e-12)


def test_hidden_runtime_and_training_payload_never_inspected(fitted):
    class Poison:
        def __deepcopy__(self, memo):
            raise AssertionError("Hidden payload copied")

        def __str__(self):
            raise AssertionError("Hidden payload inspected")

    reference, rows, _ = fitted
    projected_rows = [{**r, "home_goals": Poison(), "away_goals": Poison(),
                       "home_elo": Poison(), "raw": Poison()} for r in rows]
    model = ContextualJointReturnPredictor(reference, projected_rows)
    clean, dirty = state(), state()
    dirty.update(outcome=Poison(), raw=Poison())
    for group in dirty["groups"].values():
        group.update(values=Poison(), reason=Poison())
    assert model.joint_distribution(clean) == model.joint_distribution(dirty)
    assert model.enumerate_action(clean, ACTIONS[0]) == model.enumerate_action(dirty, ACTIONS[0])


def test_unseen_context_keeps_population_coverage_and_reports_local_backoff(fitted):
    reference, rows, _ = fitted
    model = ContextualJointReturnPredictor(reference, rows)
    query = state()
    query["base"].update(home_team_id="t1", away_team_id="t0")
    coverage = model.observation_support(query)
    baseline = reference.observation_support(query)
    assert coverage["kernel_effective_matches"] == baseline["kernel_effective_matches"]
    assert coverage["class_kernel_weight"] == baseline["class_kernel_weight"]
    assert coverage["contextual_support"]["local_matching_training_matches"] == 0
    assert coverage["contextual_support"]["local_backoff"] == "population"


def test_heldout_selection_is_past_only_and_has_population_fallback(fitted):
    reference, rows, _ = fitted
    source = deepcopy(reference._payload())
    heldout = calibration_rows(rows)
    model = ContextualJointReturnPredictor.fit_context(reference, rows, heldout, ACTIONS,
                                                       as_of="2025-04-25T00:00:00Z")
    assert reference._payload() == source
    report = model.selection_report
    assert report["context_weight_candidates"] == [0., .5, 1.]
    assert report["selected"]["selected_objective"] <= report["candidates"][0]["selected_objective"] + 1e-12
    assert report["costs_or_query_thresholds_changed"] is False
    assert model.metadata["training_as_of"] == "2025-04-25T00:00:00+00:00"
    poison = state()
    poison["match_id"] = heldout[0]["match_id"]
    with pytest.raises(ValueError, match="adaptation match"):
        model.predict_partial(poison)
    report["selected"]["strength"] = -1.
    assert model.selection_report["selected"]["strength"] >= 0.


@pytest.mark.parametrize("fault", ["future_label", "training_id"])
def test_selection_rejects_future_or_training_targets(fitted, fault):
    reference, rows, _ = fitted
    heldout = calibration_rows(rows)
    if fault == "future_label":
        heldout[0]["result_available_at"] = "2025-06-01T00:00:00Z"
    else:
        heldout[0]["match_id"] = rows[0]["match_id"]
    with pytest.raises(ValueError):
        ContextualJointReturnPredictor.fit_context(reference, rows, heldout, ACTIONS,
                                                   as_of="2025-04-25T00:00:00Z")


@pytest.mark.parametrize("fault", ["missing", "duplicate", "late_fixture", "wrong_match_time", "unknown_identity"])
def test_fixture_context_binding_rejects_invalid_training_context(fitted, fault):
    reference, rows, _ = fitted
    fixtures = deepcopy(rows)
    if fault == "missing":
        fixtures.pop()
    elif fault == "duplicate":
        fixtures.append(fixtures[0])
    elif fault == "late_fixture":
        fixtures[0]["fixture_available_at"] = fixtures[0]["kickoff"]
    elif fault == "wrong_match_time":
        fixtures[0]["prediction_cutoff"] = fixtures[1]["prediction_cutoff"]
    else:
        fixtures[0]["home_team_id"] = "unmapped"
    with pytest.raises(ValueError):
        ContextualJointReturnPredictor(reference, fixtures)


@pytest.mark.parametrize("value", [-.1, 1.1, float("nan"), True])
def test_invalid_context_or_strength_rejected(fitted, value):
    reference, rows, _ = fitted
    with pytest.raises(ValueError):
        ContextualJointReturnPredictor(reference, rows, context_weight=value)
    with pytest.raises(ValueError):
        ContextualJointReturnPredictor(reference, rows, strength=value)


def test_strict_policy_accepts_actual_update_and_never_uses_outcome(fitted):
    reference, rows, _ = fitted
    model = ContextualJointReturnPredictor(reference, rows, strength=.7)
    policy = LearnedJointVOIPolicy(model, lambda_cost={"money": 0.})
    row = calibration_rows(rows)[2]
    trajectory = run_learned_voi_episode(row, policy, ACTIONS, limits={"requests": 2, "money": .05})
    assert trajectory["actions"]
    assert trajectory == run_learned_voi_episode({**row, "home_goals": 999, "away_goals": 0},
                                                 policy, ACTIONS, limits={"requests": 2, "money": .05})
    for cycle in trajectory["cycles"]:
        if cycle["update"]:
            assert cycle["update"]["realized_loss_reduction"] is None
            assert cycle["update"]["prediction_after"] == model.predict_partial(cycle["state_after"])["probabilities"]


def test_roundtrip_and_fixture_tampering_detected(fitted, tmp_path):
    reference, rows, _ = fitted
    model = ContextualJointReturnPredictor.fit_context(reference, rows, calibration_rows(rows), ACTIONS,
                                                       as_of="2025-04-25T00:00:00Z")
    path = tmp_path/"contextual.json"
    model.save(path)
    restored = ContextualJointReturnPredictor.load(path)
    assert model.joint_distribution(state()) == restored.joint_distribution(state())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["fixture_contexts"][0]["home_team_id"] = "t1"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="bound|hash"):
        ContextualJointReturnPredictor.load(path)
