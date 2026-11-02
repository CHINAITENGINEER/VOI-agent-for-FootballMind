"""Regression tests for the canonical prediction/report data-chain fixes."""

from football_research.data_adapters.prospective_stream import _prediction_mapping, _predictor_context_fields


def test_prediction_mapping_accepts_named_and_legacy_vector_forms():
    expected = {"home": 0.5, "draw": 0.3, "away": 0.2}
    assert _prediction_mapping(expected) == expected
    assert _prediction_mapping([0.5, 0.3, 0.2]) == expected
    assert _prediction_mapping((0.5, 0.3, 0.2)) == expected


def test_prediction_mapping_rejects_unverifiable_values_without_repair():
    assert _prediction_mapping({"home": 0.5, "draw": 0.3}) is None
    assert _prediction_mapping({"home": 0.6, "draw": 0.3, "away": 0.2}) is None
    assert _prediction_mapping({"home": 1.1, "draw": -0.1, "away": 0.0}) is None
    assert _prediction_mapping([0.5, 0.3]) is None


def test_predictor_projection_records_explicit_f_consumption_ledger():
    payload = {
        "schema": "runtime-predictor-context-v4",
        "prediction_before": {"home": 0.5, "draw": 0.3, "away": 0.2},
        "prediction_after": {"home": 0.6, "draw": 0.2, "away": 0.2},
        "prediction_pipeline_version": "f-pipeline:test",
        "prediction_processing": {"status": "ready"},
        "predictor_model_versions": {"f": "f:test"},
    }
    fields = _predictor_context_fields(
        {"policy_manifest": {"predictor_version": "f:test", "predictor_sha256": "a" * 64}},
        payload,
        {"raw_context_timing_proven": True},
        {"used_semantic_input_ids": ["input-1"], "modeled_read_documents": 1},
    )
    assert fields["f_consumption_status"] == "consumed"
    assert fields["f_consumed_input_ids"] == ["input-1"]
    assert fields["f_consumption_required"] is True
