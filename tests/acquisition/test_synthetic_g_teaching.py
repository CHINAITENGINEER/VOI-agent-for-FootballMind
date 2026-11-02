import numpy as np

from football_research.acquisition.synthetic_g_teaching import (
    VERSION, SyntheticGModel, build_artifact, fit_teaching_model, generate_teaching_rows,
)


def test_synthetic_teaching_is_deterministic_and_explicitly_nonempirical(tmp_path):
    rows_a = generate_teaching_rows(seed=7, matches=12, outcome_draws=16)
    rows_b = generate_teaching_rows(seed=7, matches=12, outcome_draws=16)
    assert rows_a == rows_b
    assert len(rows_a) == 48
    assert all(r["provenance"]["evidence_origin"] == "constructed_mechanism" for r in rows_a)
    model = fit_teaching_model(rows_a, seed=8)
    assert model["schema"] == VERSION
    assert model["synthetic"] is True
    assert model["production_ready"] is False
    assert np.isfinite(model["holdout_mse"])


def test_build_artifact_writes_dataset_model_and_report(tmp_path):
    result = build_artifact(tmp_path, seed=7, matches=12, outcome_draws=16)
    assert result["report"]["status"] == "complete"
    assert result["report"]["synthetic_rows"] == 48
    for name in ("teaching-dataset.json", "model.json", "report.json"):
        assert (tmp_path / name).exists()


def test_synthetic_model_adapter_emits_provenance(tmp_path):
    result = build_artifact(tmp_path, seed=7, matches=12, outcome_draws=16)
    model = SyntheticGModel.load(tmp_path / "model.json")
    value = model.estimate(action="lineup", scenario="rotation_sensitive", entropy=.7,
                            max_probability=.6, time_to_kickoff_hours=4,
                            source_reliability=.94, event_relevance=.95,
                            tactical_sensitivity=.7, cost_units=.02)
    assert value["support_status"] == "synthetic_teaching_only"
    assert value["evidence_origin"] == "constructed_mechanism"
    assert value["synthetic"] is True
    assert value["calibration_status"] == "not_established"
    assert value["production_ready"] is False
    assert "status" not in value or value["status"] != "supported"
    assert value["expected_net_voi"] == value["expected_signed_gain"] - .02

