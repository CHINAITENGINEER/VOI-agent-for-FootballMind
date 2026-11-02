from copy import deepcopy
from hashlib import sha256
import json

import numpy as np
import pytest

from football_research.acquisition import synthetic_g_risk_v4 as g
from football_research.acquisition import synthetic_g_observable_v3 as v3


def features(action="lineup"):
    return v3.make_observable_features({
        "probabilities": [.5, .25, .25], "action": action,
        "source_reliability": .8, "freshness": .9, "redundancy": .1,
        "response_probability": .8, "updater_trust": .75,
        "time_to_kickoff_hours": 12.,
    })


def model_payload():
    return {"schema": g.SCHEMA, "feature_schema": v3.FEATURE_SCHEMA,
        "mean_feature_names": list(v3.OBSERVABLE_NAMES),
        "risk_feature_names": [*v3.OBSERVABLE_NAMES, "cost_units"],
        "risk_target": g.RISK_TARGET, "provenance": deepcopy(g.PROVENANCE),
        "mean_trees": [{"left": [-1], "right": [-1], "feature": [-2],
                        "threshold": [-2.], "value": [.05]}],
        "risk_trees": [{"left": [1, -1, -1], "right": [2, -1, -1],
                        "feature": [14, -2, -2], "threshold": [.01, -2., -2.],
                        "value": [.5, .1, .9]}],
        "calibration": {"x": [0., 1.], "y": [0., 1.]},
        "mean_rule": {"mode": "mean_only", "stop_margin": 0., "risk_limit": None},
        "risk_rule": {"mode": "risk_filtered", "stop_margin": 0., "risk_limit": .3}}


def test_mean_head_is_cost_independent_but_net_and_risk_are_not():
    model = g.RiskAwareSyntheticGModel(model_payload())
    low = model.estimate(features(), cost_units=.005)
    high = model.estimate(features(), cost_units=.02)
    assert low["expected_signed_gain"] == high["expected_signed_gain"] == .05
    assert low["expected_net_voi"] == pytest.approx(.045)
    assert high["expected_net_voi"] == pytest.approx(.03)
    assert low["probability_teacher_expected_net_negative"] == pytest.approx(.1)
    assert high["probability_teacher_expected_net_negative"] == pytest.approx(.9)
    assert high["production_ready"] is False
    assert high["empirical_selection_allowed"] is False


def test_risk_filter_checks_remaining_candidates_before_stop():
    candidates = [{"action_id": "first", "predicted_net": .06, "predicted_risk": .8},
                  {"action_id": "second", "predicted_net": .02, "predicted_risk": .2}]
    rule = {"mode": "risk_filtered", "stop_margin": 0., "risk_limit": .3}
    assert g.choose_action(candidates, rule)["action_id"] == "second"
    assert g.choose_action(candidates, {**rule, "risk_limit": .1}) is None
    assert g.choose_action(candidates, {**rule, "stop_margin": .02}) is None


def test_missing_information_does_not_produce_a_fake_stop():
    model = g.RiskAwareSyntheticGModel(model_payload())
    with pytest.raises(ValueError):
        model.rank_shadow([{"action_id": "query", "cost_units": .01}])
    with pytest.raises(ValueError):
        model.estimate({**features(), "hidden_teacher": 1.}, cost_units=.01)
    ranked = model.rank_shadow([{"action_id": "query", "features": features(), "cost_units": .02}])
    assert ranked["shadow_proposal"] == "STOP"
    assert ranked["selection_allowed"] is False
    assert ranked["request_executed"] is False


@pytest.mark.parametrize("cost", [True, -1., float("nan"), float("inf"), "0.01"])
def test_invalid_cost_rejected(cost):
    with pytest.raises(ValueError):
        g.RiskAwareSyntheticGModel(model_payload()).estimate(features(), cost_units=cost)


def test_all_stop_conditional_risk_is_unknown_not_zero():
    rows = [{"physical_match_id": "m", "state_id": "s", "action_id": "a",
             "cost_units": .01, "teacher_net_value": .02}]
    evaluated = g.evaluate(rows, [.03], [.5], {"mode": "stop_only"})
    assert evaluated["summary"]["negative_expected_net_given_query"] is None
    assert evaluated["summary"]["selected_risk_brier"] is None
    assert evaluated["summary"]["oracle_query_recall"] == 0.
    with pytest.raises(ValueError):
        g.evaluate(rows, [.03], [1.1], {"mode": "stop_only"})


def test_feature_matrix_does_not_use_teacher_targets_or_hidden_fields():
    row = {"observable_features": features(), "cost_units": .02,
           "teacher_net_value": 100, "negative_expected_net": 1,
           "latent_probability": .99, "effect_logits": [1, 1, 1]}
    matrix = g._matrix([row])
    poisoned = {**row, "teacher_net_value": -100, "negative_expected_net": 0,
                "latent_probability": .01, "effect_logits": [-1, -1, -1]}
    np.testing.assert_array_equal(matrix, g._matrix([poisoned]))
    np.testing.assert_array_equal(g._matrix([row], risk=True)[:, :-1], matrix)
    assert matrix.shape == (1, 14)
    assert g._matrix([row], risk=True)[0, -1] == .02


def test_loader_rejects_invalid_calibration_or_cyclic_forest():
    payload = model_payload()
    payload["calibration"] = {"x": [0., 1.], "y": [.9, .1]}
    with pytest.raises(ValueError):
        g.RiskAwareSyntheticGModel(payload)
    payload = model_payload()
    payload["risk_trees"][0]["left"][0] = 0
    with pytest.raises(ValueError):
        g.RiskAwareSyntheticGModel(payload)


def test_four_way_artifact_and_portable_prediction(tmp_path):
    output = tmp_path / "risk"
    report = g.build_artifact(output, seed=173, matches=80, states_per_match=1)
    model = g.RiskAwareSyntheticGModel.load(output / "model.json")
    splits = model.payload["split"]
    assert [len(splits[k]) for k in ("train", "calibration", "validation", "test")] == [40, 16, 12, 12]
    assert sum(map(len, splits.values())) == len(set().union(*map(set, splits.values())))
    assert report["portable_inference_max_error"]["gain"] < 1e-10
    assert report["portable_inference_max_error"]["risk"] < 1e-10
    assert report["empirical_training_rows"] == 0
    assert report["network_requests"] == 0
    assert report["provenance"]["production_ready"] is False
    assert report["validation_status"] == "no_feasible_nontrivial_rule"
    learner = json.loads((output / "learner-dataset.json").read_text(encoding="utf-8"))
    assert all("teacher" not in row and "features" not in row for row in learner["rows"])
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert all(sha256((output / name).read_bytes()).hexdigest() == value["sha256"]
               for name, value in manifest["files"].items())
    with pytest.raises(ValueError):
        g.build_artifact(output, seed=173, matches=80, states_per_match=1)
