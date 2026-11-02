"""Bind the existing market predictor to a frozen actor-visible episode.

No evaluator, archived body, or caller-supplied probabilities are accepted by
this adapter. All numeric inputs must already be in the hashed public state.
"""
from __future__ import annotations

import re
from typing import Any, Mapping

from football_agent.prediction_snapshot import normalise_prediction_snapshot
from football_agent.round163_historical_episode_bridge import public_evaluator_fields
from football_research.data_adapters.spftll import REGULATION_SETTLEMENT
from .api import PredictionService, content_hash, utc


VERSION = "episode-market-prediction-v1"


def predict_public_market(
    public_episode: Mapping[str, Any], *, source_id: str,
    max_age_seconds: float | None = 86400.0,
) -> dict[str, Any]:
    """Predict from one explicitly selected complete quote in the public state.

    This is a fixed proportional de-vig calculation, reconstructed now from
    cutoff-visible evidence, not a model that was trained or executed then.
    """
    if not isinstance(public_episode, Mapping) or public_evaluator_fields(public_episode):
        raise ValueError("An actor-only public episode is required")
    snapshot = public_episode.get("match_snapshot")
    if not isinstance(snapshot, Mapping):
        raise ValueError("Public match_snapshot is required")
    if content_hash(snapshot) != public_episode.get("public_snapshot_digest"):
        raise ValueError("Public snapshot digest mismatch")
    target = public_episode.get("target_key")
    if not isinstance(target, str) or not target or snapshot.get("target_key") != target:
        raise ValueError("Public target identity mismatch")
    cutoff = utc(public_episode["prediction_cutoff"])
    if (utc(snapshot["prediction_cutoff"]) != cutoff
            or utc(snapshot["kickoff"]) != utc(public_episode["kickoff"])):
        raise ValueError("Public fixture/cutoff binding mismatch")
    observations = [item for item in snapshot.get("source_observations", [])
                    if item.get("source_id") == source_id]
    if len(observations) != 1:
        raise ValueError("Exactly one visible market source must be selected")
    observation = observations[0]
    if not re.fullmatch(r"[0-9a-f]{64}", str(observation.get("content_digest", ""))):
        raise ValueError("Market observation requires a payload digest")
    available, observed = utc(observation["available_at"]), utc(observation["observed_at"])
    if observed > available or available > cutoff:
        raise ValueError("Market observation is unavailable at cutoff")
    values = observation.get("values", {})
    quote, identity = values.get("market_quote", {}), values.get("fixture_identity", {})
    if (quote.get("source_match_id") != snapshot.get("match_id")
            or identity.get("source_match_id") != snapshot.get("match_id")
            or any(identity.get(key) != snapshot.get(key) for key in
                   ("home_team_id", "away_team_id", "competition_id"))
            or utc(identity.get("kickoff")) != utc(snapshot["kickoff"])):
        raise ValueError("Quote fixture identity differs from public target")
    if (quote.get("market_type") != "1x2" or quote.get("period") != "regulation"
            or quote.get("line") is not None or quote.get("settlement_rule") != REGULATION_SETTLEMENT):
        raise ValueError("An unhandicapped regulation 1X2 quote is required")
    prices = quote.get("prices", {})
    if set(prices) != {"home", "draw", "away"} or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) for value in prices.values()
    ):
        raise ValueError("Complete numeric home/draw/away prices are required")
    lineage = {
        "source_id": source_id, "content_digest": observation["content_digest"],
        "observed_at": observation["observed_at"], "available_at": observation["available_at"],
        "availability_basis": observation.get("availability_basis"),
        "quote_digest": content_hash(quote),
        "public_snapshot_digest": public_episode["public_snapshot_digest"],
    }
    mode = snapshot.get("information_mode")
    if mode not in {"operational", "historical_reconstruction"}:
        raise ValueError("An explicit information mode is required")
    if observation.get("availability_basis") == "observed_at" and mode != "historical_reconstruction":
        raise ValueError("Legacy observed_at fallback cannot claim operational availability")
    fixture = {key: snapshot[key] for key in
               ("competition_id", "home_team_id", "away_team_id", "kickoff")}
    fixture.update(match_id=target, information_mode=mode, score_period="regulation",
                   fixture_available_at=available, input_manifest=[source_id])
    features = {
        "match_id": target, "information_mode": mode, "prediction_cutoff": cutoff,
        "input_manifest": lineage, "odds_available_at": available,
        "odds_home": prices["home"], "odds_draw": prices["draw"], "odds_away": prices["away"],
        "odds_market_type": quote["market_type"], "odds_period": quote["period"],
        "odds_line": quote["line"], "odds_source_timestamp": quote.get("source_timestamp"),
        "odds_snapshot_id": quote.get("snapshot_id"),
    }
    service = PredictionService.train(
        "market", [], as_of=cutoff, dataset_version=f"{VERSION}:{content_hash(lineage)}",
        information_mode=mode, parameters={"max_age_seconds": max_age_seconds},
    )
    # train_cutoff is a replay freeze boundary, not a historical execution claim.
    service.training_manifest.update(
        fitting_status="fixed_formula_no_fitting",
        execution_scope="retrospective_calculation_from_cutoff_visible_archive",
    )
    prediction = service.predict(fixture, prediction_cutoff=cutoff, features=features)
    artifact = service.to_artifact()
    bound = normalise_prediction_snapshot(
        {**prediction.model_dump(mode="json"), "visible_lineage": lineage,
         "artifact_sha256": artifact["artifact_sha256"]},
        match_id=target, prediction_cutoff=cutoff, state_digest=content_hash(public_episode),
    )
    return {"schema_version": VERSION, "prediction": bound,
            "prediction_record": prediction.model_dump(mode="json"), "model_artifact": artifact}


__all__ = ["VERSION", "predict_public_market"]
