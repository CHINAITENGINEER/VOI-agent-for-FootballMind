import json

import numpy as np
import pytest

import football_research.acquisition.synthetic_g_shared_world as g


def test_shared_world_rows_are_grouped_and_include_harmful_information():
    dataset = g.generate_dataset(seed=20261010, matches=30, states_per_match=1)
    assert dataset["schema"] == g.SCHEMA
    assert len(dataset["teacher_only_worlds"]) == 30
    assert any(row["teacher"]["expected_signed_gain"] < 0 for row in dataset["rows"])

    by_state = {}
    for row in dataset["rows"]:
        by_state.setdefault(row["state_id"], []).append(row)
    assert all(len(rows) == len(g.ACTIONS) for rows in by_state.values())
    assert all(len({row["world_sha256"] for row in rows}) == 1
               for rows in by_state.values())

    world_by_id = {world["world_sha256"]: world for world in dataset["teacher_only_worlds"]}
    for rows in by_state.values():
        world = world_by_id[rows[0]["world_sha256"]]
        assert np.allclose(np.asarray(world["weights"]).sum(), 1.)
        assert np.allclose(np.asarray(world["outcome_by_world"]).sum(axis=1), 1.)
        assert max(row["teacher"]["marginal_coherence_error"] for row in rows) < 1e-12


def test_shared_world_can_make_mistrusted_signal_negative():
    weights = np.full(16, 1. / 16.)
    effect = np.asarray([1., -.5, -.5])
    outcomes = g._softmax(
        np.log(np.asarray([.4, .3, .3]))[None, :]
        + ((2 * g.BITS[:, 0:1] - 1.) * effect)
    )
    scored = g.score_channel(weights, outcomes, 0, actual_reliability=.6,
                             updater_trust=.99, response_probability=1.)
    assert scored["expected_signed_gain"] < 0
    assert scored["probability_of_negative_realized_gain"] > 0


def test_decision_evaluation_has_consistent_stop_semantics():
    dataset = g.generate_dataset(seed=9, matches=30, states_per_match=1)
    rows = dataset["rows"]
    estimates = np.asarray([row["teacher"]["expected_signed_gain"] for row in rows])
    evaluated = g.evaluate_decisions(rows, estimates, threshold=.002)
    assert all("STOP" in (trace["selected"], trace["oracle"]) or trace["queried"]
               for trace in evaluated["traces"])
    assert evaluated["summary"]["oracle_stop_count"] >= 0
    assert evaluated["summary"]["mean_regret"] >= -1e-12


def test_feature_contract_rejects_implicit_or_extra_fields():
    dataset = g.generate_dataset(seed=10, matches=30, states_per_match=1)
    features = dict(dataset["rows"][0]["features"])
    g.feature_vector(features)
    with pytest.raises(ValueError):
        g.feature_vector({**features, "title": "must not be consumed"})
    with pytest.raises(ValueError):
        g.feature_vector({**features, "probabilities": [.5, .3, .3]})


def test_artifact_has_grouped_split_and_portable_model(tmp_path):
    output = tmp_path / "artifact"
    report = g.build_artifact(output, seed=11, matches=30, states_per_match=1)
    assert report["counts"]["train_matches"] == 18
    assert report["counts"]["validation_matches"] == 6
    assert report["counts"]["test_matches"] == 6
    assert report["portable_inference_max_error"] <= 1e-10
    assert report["max_marginal_coherence_error"] <= 1e-12
    assert report["provenance"]["production_ready"] is False
    model = json.loads((output / "model.json").read_text(encoding="utf-8"))
    split_ids = [set(model[name]) for name in ("train_match_ids", "validation_match_ids")]
    assert not split_ids[0] & split_ids[1]
    assert (output / "manifest.json").exists()
