"""V4 journal wrapper is opt-in and retains actual processing evidence."""
from copy import deepcopy

import pytest

from tests.research.data_adapters.test_prospective_journal import (
    FrozenPredictorPolicy, context_payload, setup_executor,
)


class PipelinePolicy(FrozenPredictorPolicy):
    prediction_pipeline_version = "entity-market-f:constructed-pipeline"


def payload(*, ready=True):
    p = context_payload("final", after={"home": .5, "draw": .3, "away": .2})
    p.update(schema="runtime-predictor-context-v4", forecast_as_of=p["observed_at"],
        prediction_pipeline_version=PipelinePolicy.prediction_pipeline_version,
        prediction_processing={"schema": "entity-semantic-prediction-processing-v1",
            "status": "ready" if ready else "not_updated", "reason": None if ready else "semantic_extraction_failed",
            "input_at": p["observed_at"], "available_at": p["observed_at"],
            "effective_at": p["observed_at"] if ready else None})
    return p


@pytest.mark.parametrize("ready", [True, False])
def test_v4_processing_and_pipeline_survive_wrapper(ready, tmp_path):
    wrapped, executor, *_ = setup_executor(tmp_path, predictor_policy=PipelinePolicy())
    p = payload(ready=ready)
    snapshot = wrapped.observe_context(p)
    assert snapshot["forecast_eligible"] is ready
    assert snapshot["prediction_processing"] == p["prediction_processing"]
    assert snapshot["context"] == p
    assert executor.calls == 0


@pytest.mark.parametrize("case", ["legacy_payload", "changed_pipeline", "missing_processing", "changed_policy_pipeline"])
def test_wrapper_cannot_change_pipeline_or_drop_v4_processing(tmp_path, case):
    policy = PipelinePolicy()
    wrapped, executor, *_ = setup_executor(tmp_path, predictor_policy=policy)
    p = payload()
    if case == "legacy_payload": p["schema"] = "runtime-predictor-context-v3"
    elif case == "changed_pipeline": p["prediction_pipeline_version"] = "other"
    elif case == "missing_processing": p.pop("prediction_processing")
    else: policy.prediction_pipeline_version = "other"
    with pytest.raises(ValueError):
        wrapped.observe_context(p)
    assert executor.calls == 0


def test_menu_requires_same_pipeline_before_request(tmp_path):
    wrapped, executor, _, _, _, public, menu = setup_executor(tmp_path, predictor_policy=PipelinePolicy())
    public["prediction"] = {"probabilities": [.5, .3, .2], "model_version": "f-test-v1"}
    with pytest.raises(ValueError, match="frozen prediction pipeline"):
        wrapped.observe_menu(menu)
    public["prediction"]["prediction_pipeline_version"] = PipelinePolicy.prediction_pipeline_version
    wrapped.observe_menu(menu)
    assert executor.calls == 0
