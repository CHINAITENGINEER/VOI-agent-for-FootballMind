"""Evidence/time/identity tests for a public-state-only market prediction."""
from copy import deepcopy
from hashlib import sha256
import json

import pytest

from football_agent.historical_asof_episode import build_historical_asof_episode
from football_agent.round163_historical_episode_bridge import build_round163_episode
from football_research.baselines.api import content_hash
from football_research.baselines.episode_market import predict_public_market
from football_research.data_adapters.prematch_market import project_spftll_market_observation


CUTOFF = "2026-09-26T12:00:00Z"


def raw_archive():
    data = {"code": 0, "data": [{"list": [{
        "matchId": 7, "matchTime": "2026-09-26T21:00:00", "footMatchState": 0,
        "homeId": 1, "awayId": 2, "homeChs": "Home", "awayChs": "Away",
        "leagueChs": "Test Cup", "spfWinFoot": 2.45, "spfEqualFoot": 2.85, "spfLoseFoot": 2.7,
        "goalFoot": -1, "rspfWinFoot": 5.85, "rspfEqualFoot": 4.1, "rspfLoseFoot": 1.4,
    }]}]}
    observation = {
        "id": "test-observation", "source": "spftll", "http_status": 200,
        "endpoint": "https://example.test/GetSimpleMatchsAll", "payload_format": "json",
        "observed_at": "2026-09-26T11:44:50Z", "payload_path": "raw/observation.body.gz",
    }
    return observation, data


def project(*, mutate=None, **overrides):
    obs, data = raw_archive()
    if mutate:
        mutate(obs, data)
    payload = json.dumps(data, ensure_ascii=False).encode()
    obs["content_hash"] = sha256(payload).hexdigest()
    options = {"match_id": "7", "prediction_cutoff": CUTOFF, "allow_observed_at_fallback": True}
    options.update(overrides)
    return project_spftll_market_observation(obs, payload, **options)


def public_state():
    row = project()
    row["candidate_menu"] = [{"candidate_id": "lineup", "source_ids": [], "availability": "unknown"}]
    envelope = build_historical_asof_episode(
        row, allow_observed_at_fallback=True, candidate_catalog_complete=False,
    )
    public = build_round163_episode(envelope, candidate_menu=[{"candidate_id": "lineup", "cost": 0.1}],
                                   budget=0.3)["public"]
    return public, row["source_observations"][0]["source_id"]


def test_prediction_uses_only_visible_unhandicapped_quote_with_lineage():
    public, source = public_state()
    prediction = predict_public_market(public, source_id=source)
    denominator = 1 / 2.45 + 1 / 2.85 + 1 / 2.7
    assert prediction["prediction"]["probabilities"] == pytest.approx(
        [(1 / odd) / denominator for odd in (2.45, 2.85, 2.7)])
    assert prediction["prediction"]["state_digest"] == content_hash(public)
    assert prediction["model_artifact"]["training_manifest"]["training_matches"] == 0
    assert prediction["prediction"]["visible_lineage"]["availability_basis"] == "observed_at"
    assert public["match_snapshot"]["source_observations"][0]["values"]["market_quote"]["source_timestamp"] is None


def test_legacy_clock_needs_explicit_opt_in():
    with pytest.raises(ValueError, match="explicit opt-in"):
        project(allow_observed_at_fallback=False)


def test_explicit_availability_never_precedes_observation():
    with pytest.raises(ValueError, match="cutoff"):
        project(available_at="2026-09-26T11:00:00Z")
    with pytest.raises(ValueError, match="cutoff"):
        project(available_at="2026-09-26T12:01:00Z")


def test_original_body_hash_is_verified_before_projection():
    obs, data = raw_archive()
    payload = json.dumps(data).encode()
    obs["content_hash"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        project_spftll_market_observation(obs, payload, match_id="7", prediction_cutoff=CUTOFF,
                                         allow_observed_at_fallback=True)


@pytest.mark.parametrize("mutation,reason", [
    (lambda o, d: o.update(observed_at="2026-09-26T12:01:00Z"), "cutoff"),
    (lambda o, d: d["data"][0]["list"][0].pop("spfEqualFoot"), "complete unique"),
    (lambda o, d: d["data"][0]["list"].append(deepcopy(d["data"][0]["list"][0])), "exactly one"),
    (lambda o, d: d["data"][0]["list"][0].update(footMatchState=-1), "unplayed"),
    (lambda o, d: d["data"][0]["list"][0].update(homeScore=0, awayScore=0), "unplayed"),
])
def test_bad_or_late_raw_evidence_cannot_form_a_prior(mutation, reason):
    with pytest.raises(ValueError, match=reason):
        project(mutate=mutation)


def test_cutoff_must_strictly_precede_kickoff():
    with pytest.raises(ValueError, match="unplayed"):
        project(prediction_cutoff="2026-09-26T13:00:00Z")


def test_public_prices_cannot_change_without_a_new_snapshot():
    public, source = public_state()
    public["match_snapshot"]["source_observations"][0]["values"]["market_quote"]["prices"]["home"] = 9.0
    with pytest.raises(ValueError, match="snapshot digest"):
        predict_public_market(public, source_id=source)


def test_quote_must_be_bound_to_the_same_fixture_and_teams():
    public, source = public_state()
    public["match_snapshot"]["home_team_id"] = "another-team"
    public["public_snapshot_digest"] = content_hash(public["match_snapshot"])
    with pytest.raises(ValueError, match="fixture identity"):
        predict_public_market(public, source_id=source)


def test_evaluator_data_and_stale_quotes_are_rejected():
    public, source = public_state()
    with pytest.raises(ValueError, match="actor-only"):
        predict_public_market(dict(public, outcome=1), source_id=source)
    with pytest.raises(ValueError, match="maximum age"):
        predict_public_market(public, source_id=source, max_age_seconds=1)


def test_legacy_time_cannot_be_relabeled_operational():
    public, source = public_state()
    public["match_snapshot"]["information_mode"] = "operational"
    public["public_snapshot_digest"] = content_hash(public["match_snapshot"])
    with pytest.raises(ValueError, match="operational availability"):
        predict_public_market(public, source_id=source)
