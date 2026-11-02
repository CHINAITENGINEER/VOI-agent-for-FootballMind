import json

import numpy as np
import pytest

import football_research.acquisition.synthetic_g_observable_v3 as g
import football_research.acquisition.synthetic_g_shared_world as shared


def test_projection_excludes_teacher_only_fields():
    dataset = shared.generate_dataset(seed=20261010, matches=30, states_per_match=1)
    source = dataset["rows"][0]["features"]
    projected = g.make_observable_features(source)
    assert projected["schema"] == g.FEATURE_SCHEMA
    assert len(projected["values"]) == len(g.OBSERVABLE_NAMES)
    assert "latent_probability" not in projected
    assert "effect_logits" not in projected
    assert all(name not in projected for name in ("latent_probability", "effect_logits"))


def test_observable_contract_rejects_hidden_or_extra_fields():
    dataset = shared.generate_dataset(seed=7, matches=30, states_per_match=1)
    projected = g.make_observable_features(dataset["rows"][0]["features"])
    g.feature_vector(projected)
    with pytest.raises(ValueError):
        g.feature_vector({**projected, "effect_logits": [1, 2, 3]})
    with pytest.raises(ValueError):
        g.feature_vector({**projected, "values": projected["values"][:-1]})
    with pytest.raises(ValueError):
        g.feature_vector({**projected, "probabilities": [0.5, 0.3, 0.3]})
    with pytest.raises(ValueError):
        g.feature_vector({**projected, "values": [*projected["values"][:3],
                                                     0.0, *projected["values"][4:]]})
    wrong_action = dict(projected)
    wrong_action["action"] = "weather"
    with pytest.raises(ValueError):
        g.feature_vector(wrong_action)


def test_artifact_is_grouped_portable_and_diagnostic_only(tmp_path):
    output = tmp_path / "artifact"
    report = g.build_artifact(output, seed=11, matches=30, states_per_match=1)
    assert report["counts"]["train_matches"] == 18
    assert report["counts"]["validation_matches"] == 6
    assert report["counts"]["test_matches"] == 6
    assert report["portable_inference_max_error"] <= 1e-10
    assert "harmful_query_given_query" in report["test"]
    assert "positive_net_precision" in report["test"]
    assert len(report["test_risk_frontier"]) == 4
    assert report["provenance"]["production_ready"] is False
    assert report["provenance"]["empirical_selection_allowed"] is False
    model = json.loads((output / "model.json").read_text(encoding="utf-8"))
    assert model["feature_schema"] == g.FEATURE_SCHEMA
    assert set(model["train_match_ids"]).isdisjoint(model["validation_match_ids"])
    assert set(model["train_match_ids"]).isdisjoint(model["test_match_ids"])
    loaded = g.ObservableOnlyGModel.load(output / "model.json")
    row = shared.generate_dataset(seed=22, matches=30, states_per_match=1)["rows"][0]
    observable = g.make_observable_features(row["features"])
    estimate = loaded.estimate(observable, cost_units=0.01)
    assert estimate["production_ready"] is False
    assert estimate["empirical_selection_allowed"] is False
    learner_rows = json.loads((output / "learner-dataset.json").read_text(encoding="utf-8"))["rows"]
    assert learner_rows
    assert all("features" not in row and "teacher" not in row for row in learner_rows)
    assert all("observable_features" in row for row in learner_rows)
    malformed = dict(model)
    malformed["stop_margin"] = -1
    with pytest.raises(ValueError):
        g.ObservableOnlyGModel(malformed)
    assert (output / "manifest.json").exists()


def test_observable_projection_is_finite_and_public_only():
    dataset = shared.generate_dataset(seed=19, matches=30, states_per_match=1)
    for row in dataset["rows"][:12]:
        projected = g.make_observable_features(row["features"])
        values = g.feature_vector(projected)
        assert np.isfinite(values).all()
        assert np.isclose(values[:3].sum(), 1.0)
