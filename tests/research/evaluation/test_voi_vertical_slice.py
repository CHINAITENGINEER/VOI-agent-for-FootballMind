"""Tests for the first real-episode VoI research vertical slice."""
from __future__ import annotations

import pytest

from football_research.evaluation.voi_vertical_slice import run_vertical_slice


PUBLIC = {
    "target_key": "spftll:2981481",
    "kickoff": "2026-09-26T13:00:00Z",
    "prediction_cutoff": "2026-09-26T12:00:00Z",
    "candidate_menu": [
        {"candidate_id": "lineup_rotation"},
        {"candidate_id": "weather"},
    ],
    "public_snapshot_digest": "a" * 64,
    "historical_provenance": {"episode_id": "round172:slovenia-scotland"},
}


def _models():
    prior = (0.43, 0.31, 0.26)
    return prior, {
        "lineup_rotation": {
            "cost": 0.04,
            "delay": 0.1,
            "delay_weight": 0.03,
            "reliability": 0.95,
            "reliability_penalty_weight": 0.02,
            "response_model_version": "development-fixture-rotation-v1",
            "branches": [
                {"branch_id": "no_rotation", "probability": 0.50,
                 "posterior": (0.65, 0.25, 0.10)},
                {"branch_id": "rotation", "probability": 0.50,
                 "posterior": (0.21, 0.37, 0.42)},
            ],
        },
        "weather": {
            "cost": 0.02,
            "response_model_version": "development-fixture-weather-v1",
            "branches": [
                {"branch_id": "normal", "probability": 0.5, "posterior": prior},
                {"branch_id": "normal_again", "probability": 0.5, "posterior": prior},
            ],
        },
    }


def test_vertical_slice_selects_informative_candidate_and_hides_result():
    prior, models = _models()
    trace = run_vertical_slice(
        PUBLIC,
        prior=prior,
        candidate_models=models,
        decision_context={"qualification_state": "unknown"},
    )
    assert trace["selection"]["decision"] == "acquire"
    assert trace["selection"]["candidate_id"] == "lineup_rotation"
    assert trace["hidden_result_included"] is False
    assert trace["numeric_voi_training_eligible"] is False
    assert trace["candidate_values"][1]["gross_evsi"] == pytest.approx(0.0)


def test_vertical_slice_refuses_evaluator_fields_in_public_state():
    prior, models = _models()
    leaked = dict(PUBLIC, outcome=1)
    with pytest.raises(ValueError, match="evaluator fields"):
        run_vertical_slice(leaked, prior=prior, candidate_models=models)


def test_vertical_slice_requires_a_model_for_every_public_candidate():
    prior, models = _models()
    with pytest.raises(ValueError, match="do not match public menu"):
        run_vertical_slice(PUBLIC, prior=prior, candidate_models={"weather": models["weather"]})


def test_vertical_slice_rejects_candidate_source_already_visible_to_actor():
    prior, models = _models()
    public = dict(PUBLIC)
    public["match_snapshot"] = {
        "source_observations": [{"source_id": "source:visible"}],
    }
    public["historical_provenance"] = {
        "episode_id": "round172:visible-source",
        "actor_source_ids": ["source:visible"],
    }
    public["candidate_menu"] = [
        {"candidate_id": "lineup_rotation", "source_id": "source:visible"},
        {"candidate_id": "weather"},
    ]
    with pytest.raises(ValueError, match="actor-visible sources"):
        run_vertical_slice(public, prior=prior, candidate_models=models)


def test_explicit_unknown_models_keep_value_null_and_defer():
    prior, _ = _models()
    models = {item["candidate_id"]: {"status": "UNKNOWN", "reason": "response_model_not_calibrated"}
              for item in PUBLIC["candidate_menu"]}
    trace = run_vertical_slice(PUBLIC, prior=prior, candidate_models=models)
    assert trace["selection"]["decision"] == "defer"
    assert trace["selection"]["stop_certificate"] is None
    assert trace["response_model_scope"] == "unavailable"
    assert all(row["net_voi"] is None and row["gross_evsi"] is None
               and row["prior_risk"] > 0 for row in trace["candidate_values"])


def test_zero_known_value_does_not_hide_an_unknown_candidate():
    prior, models = _models()
    models["lineup_rotation"] = {"status": "UNKNOWN", "reason": "no_model"}
    trace = run_vertical_slice(PUBLIC, prior=prior, candidate_models=models)
    assert trace["selection"]["decision"] == "defer"
    assert trace["candidate_values"][1]["gross_evsi"] == pytest.approx(0)


def test_nonpositive_values_certify_only_an_explicitly_complete_finite_menu():
    prior, _ = _models()
    models = {item["candidate_id"]: {"cost": 0.1, "branches": [
        {"branch_id": "same", "probability": 1.0, "posterior": prior}]}
        for item in PUBLIC["candidate_menu"]}
    for complete in (None, False, True):
        trace = run_vertical_slice(dict(PUBLIC, candidate_generation_complete=complete),
                                   prior=prior, candidate_models=models)
        assert trace["selection"]["decision"] == ("stop" if complete else "defer")
        assert bool(trace["selection"]["stop_certificate"]) == (complete is True)


def test_prior_generator_is_reused_for_every_candidate():
    prior, models = _models()
    trace = run_vertical_slice(PUBLIC, prior=(p for p in prior), candidate_models=models)
    assert trace["prior"] == list(prior)
    assert trace["selection"]["candidate_id"] == "lineup_rotation"


def test_model_cost_must_match_the_public_action_cost():
    prior, models = _models()
    public = dict(PUBLIC, candidate_menu=[
        {"candidate_id": "lineup_rotation", "cost": 0.4}, {"candidate_id": "weather", "cost": 0.02}])
    with pytest.raises(ValueError, match="cost differs from public menu"):
        run_vertical_slice(public, prior=prior, candidate_models=models)


def test_decision_context_is_also_an_actor_only_boundary():
    prior, models = _models()
    with pytest.raises(ValueError, match="decision_context.nested.outcome"):
        run_vertical_slice(PUBLIC, prior=prior, candidate_models=models,
                           decision_context={"nested": {"outcome": 1}})


def test_unknown_models_reject_contradictory_branches_and_invalid_prior():
    prior, models = _models()
    models["lineup_rotation"].update(status="UNKNOWN", reason="no_model")
    with pytest.raises(ValueError, match="must not supply branches"):
        run_vertical_slice(PUBLIC, prior=prior, candidate_models=models)
    models = {key: {"status": "UNKNOWN", "reason": "no_model"} for key in models}
    with pytest.raises(ValueError, match="sum to one"):
        run_vertical_slice(PUBLIC, prior=(0.9, 0.9, 0.9), candidate_models=models)


def test_plural_source_ids_cannot_reuse_known_evidence():
    prior, models = _models()
    public = dict(PUBLIC, historical_provenance={"actor_source_ids": ["seen"]},
                  candidate_menu=[{"candidate_id": "lineup_rotation", "source_ids": ["seen"]},
                                  {"candidate_id": "weather"}])
    with pytest.raises(ValueError, match="actor-visible sources"):
        run_vertical_slice(public, prior=prior, candidate_models=models)
