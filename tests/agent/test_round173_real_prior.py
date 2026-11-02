"""Real archive integration: a measured prior is not measured information gain."""
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

import pytest

from football_agent.round163_historical_episode_bridge import validate_round163_episode
from football_research.baselines.api import content_hash, utc
from football_research.baselines.episode_market import predict_public_market


@pytest.fixture(scope="module")
def bundle():
    root = Path(__file__).resolve().parents[2]
    spec = spec_from_file_location("record_round173", root / "var/parallel/agent/round173/record_round173.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_bundle()


def test_real_prior_is_computed_from_archived_prices_not_development_constants(bundle):
    public = bundle["public"]
    evidence = public["match_snapshot"]["source_observations"][0]
    assert evidence["content_digest"] == "290c49b3777183c07ee7c804575dc7afcbb8ba3282d4e2f0871b970dc51f5b9e"
    assert evidence["values"]["market_quote"]["prices"] == {"home": 2.45, "draw": 2.85, "away": 2.7}
    expected = [1 / price for price in (2.45, 2.85, 2.7)]
    assert bundle["prediction-before"]["probabilities"] == pytest.approx([p / sum(expected) for p in expected])
    assert bundle["voi-trace"]["prior"] == bundle["prediction-before"]["probabilities"]
    assert bundle["prediction-before"]["state_digest"] == content_hash(public)
    assert validate_round163_episode(bundle["episode"])["valid"] is True


def test_actual_result_retrieval_time_is_preserved_only_in_evaluator(bundle):
    sidecar = bundle["evaluation-sidecar"]
    assert utc(sidecar["result_observed_at"]).date().isoformat() == "2026-10-05"
    assert utc(sidecar["label_available_at"]) < utc(sidecar["result_observed_at"])
    assert sidecar["label_status"] == "INDEPENDENT_GOLD_RECONSTRUCTED"
    assert sidecar["prediction_before_losses"]["log_loss"] > 0
    public = bundle["public"]
    source = public["match_snapshot"]["source_observations"][0]["source_id"]
    assert predict_public_market(public, source_id=source)["prediction"] == bundle["prediction-before"]


def test_unknown_information_value_cannot_generate_an_after_or_stop_certificate(bundle):
    assert bundle["voi-trace"]["selection"]["decision"] == "defer"
    assert bundle["transition"]["prediction_after"] is None
    assert bundle["transition"]["action_executed"] is False
    assert bundle["transition"]["stop_certificate"] is None
    assert all(row["net_voi"] is None for row in bundle["voi-trace"]["candidate_values"])
    assert bundle["evaluation-sidecar"]["realized_loss_reduction"] is None
    assert bundle["report"]["counterfactual_replay_ready"] is False
    assert bundle["report"]["numeric_voi_training_eligible"] is False
    assert bundle["public"]["candidate_generation_complete"] is False


def test_one_raw_response_is_one_evidence_source_and_no_extra_requests(bundle):
    assert len(bundle["public"]["match_snapshot"]["source_observations"]) == 1
    assert bundle["historical-envelope"]["audit"]["allow_observed_at_fallback"] is True
    assert bundle["report"]["external_source_requests"] == 0
    assert bundle["report"]["baseline_inference_calls"] == 1


def test_saved_research_artifacts_are_reproducible_from_original_bytes(bundle):
    root = Path(__file__).resolve().parents[2] / "var/parallel/agent/round173"
    for name, regenerated in bundle.items():
        saved = json.loads((root / f"{name}.json").read_text(encoding="utf-8"))
        if name == "report":
            saved.pop("generated_at")
            regenerated = {key: value for key, value in regenerated.items() if key != "generated_at"}
        assert saved == regenerated, name
