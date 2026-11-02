"""Processing completion is not source receipt time; constructed clocks only."""
from copy import deepcopy
from datetime import timedelta

import pytest

from tests.platform.information.test_acquisition_journal_predictor_only import create, P


PIPELINE = "entity-market-f:fixed-extractor-prior-and-f"


def context_args(now, *, ready=True):
    clock = now[0]
    observed = clock + timedelta(seconds=8)
    forecast = clock + timedelta(seconds=6 if ready else 1)
    args = dict(snapshot_kind="final", observed_at=observed, available_at=observed,
        availability_basis="runtime_state_handoff", forecast_as_of=forecast, prediction_cutoff=forecast,
        acquisition_deadline=clock + timedelta(hours=1), prediction_after=P, model_versions=["f-frozen-v1"],
        prediction_pipeline_version=PIPELINE,
        prediction_processing={"schema": "entity-semantic-prediction-processing-v1",
            "status": "ready" if ready else "not_updated", "reason": None if ready else "semantic_extraction_failed",
            "input_at": (clock + timedelta(seconds=3)).isoformat(),
            "available_at": (clock + timedelta(seconds=6)).isoformat(),
            "effective_at": forecast.isoformat() if ready else None})
    now[0] += timedelta(seconds=10)
    return args


@pytest.mark.parametrize("ready", [True, False])
def test_processing_not_updated_stays_audit_only_without_clock_forgery(tmp_path, ready):
    journal, run, now = create(tmp_path)
    args = context_args(now, ready=ready)
    snapshot = journal.record_context(run["run_id"], **args)
    assert snapshot["forecast_eligible"] is ready
    assert snapshot["recording_scope"] == ("pre_match_context" if ready else "audit_only")
    assert snapshot["prediction_after"] == P
    assert snapshot["prediction_processing"] == args["prediction_processing"]
    assert snapshot["prediction_pipeline_version"] == PIPELINE
    assert snapshot["prediction_cutoff"] == args["forecast_as_of"].isoformat()
    assert journal.read(run["run_id"])["prediction_pipeline_version"] == PIPELINE


@pytest.mark.parametrize("case", ["future_input", "future_available", "early_effective", "ready_reason",
    "failed_effective", "missing_field", "missing_pipeline", "bad_schema"])
def test_malformed_processing_cannot_be_committed(tmp_path, case):
    journal, run, now = create(tmp_path)
    args = context_args(now, ready=case != "failed_effective")
    processing = args["prediction_processing"]
    if case == "future_input": processing["input_at"] = (now[0] + timedelta(seconds=1)).isoformat()
    elif case == "future_available": processing["available_at"] = (now[0] + timedelta(seconds=1)).isoformat()
    elif case == "early_effective": processing["effective_at"] = processing["input_at"]
    elif case == "ready_reason": processing["reason"] = "failed"
    elif case == "failed_effective": processing["effective_at"] = processing["available_at"]
    elif case == "missing_field": processing.pop("input_at")
    elif case == "missing_pipeline": args["prediction_pipeline_version"] = None
    elif case == "bad_schema": processing["schema"] = "other-v1"
    with pytest.raises(ValueError):
        journal.record_context(run["run_id"], **args)
    assert journal.read(run["run_id"])["context_snapshots"] == []


def test_cached_processing_can_be_older_but_pipeline_cannot_change(tmp_path):
    journal, run, now = create(tmp_path)
    args = context_args(now)
    args["prediction_processing"]["available_at"] = args["prediction_processing"]["input_at"]
    args["prediction_processing"]["input_at"] = args["prediction_processing"]["effective_at"]
    assert journal.record_context(run["run_id"], **args)["forecast_eligible"]
    args["prediction_pipeline_version"] = "entity-market-f:changed-extractor"
    with pytest.raises(ValueError, match="change its frozen prediction pipeline"):
        journal.record_context(run["run_id"], **args)


def test_null_processing_keeps_old_v3_snapshot_fields_unchanged(tmp_path):
    journal, run, now = create(tmp_path)
    args = context_args(now)
    args.pop("prediction_processing")
    args.pop("prediction_pipeline_version")
    snapshot = journal.record_context(run["run_id"], **args)
    assert snapshot["forecast_eligible"]
    assert "prediction_processing" not in snapshot and "prediction_pipeline_version" not in snapshot
