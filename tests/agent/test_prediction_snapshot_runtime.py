from __future__ import annotations

from copy import deepcopy

import pytest

from football_agent.autonomous_runtime_loop import run_runtime_cycle
from football_agent.episode_persistence import EpisodePersistence
from football_agent.prediction_snapshot import PredictionSnapshotError, normalise_prediction_snapshot

from test_autonomous_runtime_loop import bundle, catalog, estimator, match, provider


def _projector(actor_context, state_bundle):
    state = state_bundle["state"]
    return {
        "match_id": actor_context["match_id"],
        "prediction_cutoff": actor_context["prediction_cutoff"],
        "state_digest": state["state_digest"],
        "visible_state": deepcopy(state),
    }


def _adapter_factory():
    calls = []

    def adapter(visible):
        calls.append(visible)
        probabilities = [0.5, 0.3, 0.2] if len(calls) == 1 else [0.4, 0.3, 0.3]
        return {
            "match_id": visible["match_id"],
            "prediction_cutoff": visible["prediction_cutoff"],
            "state_digest": visible["state_digest"],
            "probabilities": probabilities,
            "model_version": "fixture-predictor-v1",
            "dataset_version": "fixture-dataset-v1",
        }

    return adapter, calls


def test_runtime_binds_before_and_after_predictions_to_episode(tmp_path):
    actor_context, report, prepared = bundle()
    adapter, calls = _adapter_factory()
    store = EpisodePersistence(tmp_path)

    result = run_runtime_cycle(
        actor_context=actor_context, state_bundle=prepared, menu_report=report,
        catalog=catalog(), match=match(), estimator=estimator(),
        context={"qualification_state": "must_win"}, loss_unit="decision_loss",
        remaining_budget={"http_calls": 1}, provider_id="fixture-provider",
        accepted_at="2026-10-01T10:00:00Z", execute_fn=provider,
        episode_store=store, episode_id="prediction-episode",
        prediction_adapter=adapter, prediction_state_projector=_projector,
    )

    assert result["prediction_before"] == [0.5, 0.3, 0.2]
    assert result["prediction_after"] == [0.4, 0.3, 0.3]
    assert result["prediction_meta_before"]["model_version"] == "fixture-predictor-v1"
    assert result["prediction_meta_after"]["dataset_version"] == "fixture-dataset-v1"
    assert len(calls) == 2
    rows = store.records()
    assert rows[0]["prediction_before"] == [0.5, 0.3, 0.2]
    assert rows[0]["prediction_after"] == [0.4, 0.3, 0.3]
    assert rows[0]["prediction_cutoff"] == "2026-10-01T12:00:00Z"


def test_late_receipt_does_not_create_after_prediction():
    actor_context, report, prepared = bundle()
    adapter, calls = _adapter_factory()

    def late_provider(request):
        result = provider(request)
        result["status"] = "late"
        result["evidence_rows"] = []
        return result

    result = run_runtime_cycle(
        actor_context=actor_context, state_bundle=prepared, menu_report=report,
        catalog=catalog(), match=match(), estimator=estimator(),
        context={"qualification_state": "must_win"}, loss_unit="decision_loss",
        remaining_budget={"http_calls": 1}, provider_id="fixture-provider",
        accepted_at="2026-10-01T10:00:00Z", execute_fn=late_provider,
        prediction_adapter=adapter, prediction_state_projector=_projector,
    )

    assert result["prediction_before"] == [0.5, 0.3, 0.2]
    assert result["prediction_after"] is None
    assert len(calls) == 1


def test_prediction_adapter_requires_explicit_state_projection():
    actor_context, report, prepared = bundle()
    adapter, _ = _adapter_factory()
    with pytest.raises(ValueError, match="prediction_state_projector"):
        run_runtime_cycle(
            actor_context=actor_context, state_bundle=prepared, menu_report=report,
            catalog=catalog(), match=match(), estimator=estimator(),
            context={"qualification_state": "must_win"}, loss_unit="decision_loss",
            remaining_budget={"http_calls": 1}, provider_id="fixture-provider",
            accepted_at="2026-10-01T10:00:00Z", execute_fn=provider,
            prediction_adapter=adapter,
        )


def test_prediction_cutoff_mismatch_fails_before_provider_call():
    actor_context, report, prepared = bundle()
    calls = []

    def adapter(visible):
        return {
            "match_id": visible["match_id"],
            "prediction_cutoff": "2026-10-01T12:01:00Z",
            "probabilities": [0.5, 0.3, 0.2],
        }

    def forbidden(request):
        calls.append(request)
        raise AssertionError("provider must not run after invalid before prediction")

    with pytest.raises(PredictionSnapshotError, match="prediction_cutoff"):
        run_runtime_cycle(
            actor_context=actor_context, state_bundle=prepared, menu_report=report,
            catalog=catalog(), match=match(), estimator=estimator(),
            context={"qualification_state": "must_win"}, loss_unit="decision_loss",
            remaining_budget={"http_calls": 1}, provider_id="fixture-provider",
            accepted_at="2026-10-01T10:00:00Z", execute_fn=forbidden,
            prediction_adapter=adapter, prediction_state_projector=_projector,
        )
    assert calls == []


def test_prediction_snapshot_normalises_research_bridge_mapping():
    snapshot = normalise_prediction_snapshot(
        {
            "match_id": "fixture-151",
            "prediction_cutoff": "2026-10-01T12:00:00+00:00",
            "probabilities": {"home": 0.5, "draw": 0.3, "away": 0.2},
        },
        match_id="fixture-151",
        prediction_cutoff="2026-10-01T20:00:00+08:00",
    )
    assert snapshot["probabilities"] == [0.5, 0.3, 0.2]
    assert snapshot["prediction_cutoff"] == "2026-10-01T12:00:00Z"


def test_prediction_snapshot_preserves_explicit_semantic_consumption_ledger():
    snapshot = normalise_prediction_snapshot(
        {
            "match_id": "fixture-151",
            "prediction_cutoff": "2026-10-01T12:00:00Z",
            "probabilities": [0.5, 0.3, 0.2],
            "used_semantic_input_ids": ["article-a", "article-b"],
            "modeled_read_documents": 2,
        },
        match_id="fixture-151",
        prediction_cutoff="2026-10-01T12:00:00Z",
    )
    assert snapshot["used_semantic_input_ids"] == ["article-a", "article-b"]
    assert snapshot["modeled_read_documents"] == 2


def test_runtime_carries_after_consumption_ledger_to_action_handoff():
    actor_context, report, prepared = bundle()
    calls = []

    def adapter(visible):
        calls.append(visible)
        return {
            "match_id": visible["match_id"],
            "prediction_cutoff": visible["prediction_cutoff"],
            "state_digest": visible["state_digest"],
            "probabilities": [0.5, 0.3, 0.2] if len(calls) == 1 else [0.4, 0.3, 0.3],
            "model_version": "fixture-f-v1",
            "used_semantic_input_ids": [] if len(calls) == 1 else ["article-b"],
            "modeled_read_documents": 0 if len(calls) == 1 else 1,
        }

    result = run_runtime_cycle(
        actor_context=actor_context, state_bundle=prepared, menu_report=report,
        catalog=catalog(), match=match(), estimator=estimator(),
        context={"qualification_state": "must_win"}, loss_unit="decision_loss",
        remaining_budget={"http_calls": 1}, provider_id="fixture-provider",
        accepted_at="2026-10-01T10:00:00Z", execute_fn=provider,
        prediction_adapter=adapter, prediction_state_projector=_projector,
    )

    assert result["f_consumption_status"] == "consumed"
    assert result["f_consumed_input_ids"] == ["article-b"]
    assert result["f_modeled_read_documents"] == 1
    assert result["f_consumption_basis"] == "prediction.used_semantic_input_ids"


def test_runtime_persists_after_consumption_ledger_with_episode_row(tmp_path):
    actor_context, report, prepared = bundle()
    calls = []

    def adapter(visible):
        calls.append(visible)
        return {
            "match_id": visible["match_id"],
            "prediction_cutoff": visible["prediction_cutoff"],
            "state_digest": visible["state_digest"],
            "probabilities": [0.5, 0.3, 0.2] if len(calls) == 1 else [0.4, 0.3, 0.3],
            "model_version": "fixture-f-v1",
            "used_semantic_input_ids": [] if len(calls) == 1 else ["article-b"],
            "modeled_read_documents": 0 if len(calls) == 1 else 1,
        }

    store = EpisodePersistence(tmp_path)
    result = run_runtime_cycle(
        actor_context=actor_context, state_bundle=prepared, menu_report=report,
        catalog=catalog(), match=match(), estimator=estimator(),
        context={"qualification_state": "must_win"}, loss_unit="decision_loss",
        remaining_budget={"http_calls": 1}, provider_id="fixture-provider",
        accepted_at="2026-10-01T10:00:00Z", execute_fn=provider,
        episode_store=store, episode_id="prediction-episode",
        prediction_adapter=adapter, prediction_state_projector=_projector,
    )

    row = store.records()[0]
    for key in ("f_consumed_input_ids", "f_consumption_status",
                "f_consumption_basis", "f_modeled_read_documents"):
        assert row[key] == result[key]
    assert row["f_consumed_input_ids"] == ["article-b"]
    assert row["f_consumption_status"] == "consumed"
    assert row["f_consumption_required"] is True
