"""Evaluator-only empirical VOI observation tests."""
import json

import pytest

from football_agent.autonomous_acquisition_step import prepare_acquisition_state, select_acquisition_action
from football_agent.execution_receipt_bridge import accept_action, complete_action
from football_agent.empirical_voi_observation import (
    ACTOR_PROVISIONAL, INDEPENDENT_GOLD, UNAVAILABLE,
    actor_projection, build_empirical_voi_observation, calibrate_observations,
    to_voi_link,
)
from football_agent.platform_acquisition_context import build_platform_acquisition_context

CUTOFF = "2026-10-01T12:00:00Z"


def match():
    return {"match_id": "fixture-138", "kickoff": "2026-10-01T18:00:00Z",
            "prediction_cutoff": CUTOFF, "observed_at": "2026-10-01T08:00:00Z",
            "ingested_at": "2026-10-01T08:01:00Z", "home_team_id": "home",
            "away_team_id": "away", "home_score": 2, "away_score": 1}


def evidence(sid="injury"):
    return {"source_id": sid, "kind": "injury",
            "observed_at": "2026-10-01T09:00:00Z", "ingested_at": "2026-10-01T09:01:00Z",
            "availability": "available", "freshness": "fresh", "digest": "a" * 64,
            "values": {"starter_status": "unknown"}}


def completed(provider="provider-138"):
    actor = build_platform_acquisition_context(match(), [])["actor_context"]
    state = prepare_acquisition_state(actor, [{
        "candidate_id": "injury", "action_kind": "query", "source_ids": ["injury"],
        "action_availability": True, "resource_cost": {"http_calls": 1},
        "utility_cost": {"amount": .1, "unit": "decision_loss", "source": "test"},
    }], candidate_generation_complete=True)
    estimate = {"candidate_id": "injury", "state_digest": state["state_digest"],
                "expected_value": .4, "unit": "decision_loss", "source": "model",
                "training_cutoff": "2026-09-01T00:00:00Z", "available_at": "2026-09-30T00:00:00Z"}
    trace = select_acquisition_action(state, loss_unit="decision_loss", estimates={"injury": estimate},
                                      remaining_budget={"http_calls": 2})
    accepted = accept_action(trace, provider_id=provider, accepted_at="2026-10-01T10:00:00Z")
    return complete_action(
        accepted, result_payload={"rotation_confirmed": True}, evidence_rows=[evidence()],
        actual_resource_cost={"http_calls": 1}, observed_at="2026-10-01T10:30:00Z",
        operational_available_at="2026-10-01T10:31:00Z", captured_at="2026-10-01T10:31:01Z")


def observe(receipt, *, label=INDEPENDENT_GOLD, before=.6, after=.3,
            observed_at="2026-10-01T11:00:00Z", dependency="episode-1"):
    return build_empirical_voi_observation(
        receipt, evaluator_id="gold-evaluator", label_status=label,
        loss_unit="decision_loss", decision_loss_before=before,
        decision_loss_after=after, observed_at=observed_at,
        source_id="injury-source", source_version="injury-v1",
        dependency_key=dependency)


def test_gold_observation_is_sidecar_only_and_binds_receipt():
    receipt = completed()
    observation = observe(receipt)
    assert observation["eligible_for_numeric_calibration"] is True
    assert observation["observed_risk_reduction"] == pytest.approx(.3)
    assert observation["receipt_digest"] == receipt["receipt_digest"]
    projection = actor_projection(observation)
    assert projection["eligible_for_numeric_calibration"] is False
    assert "decision_loss_before" not in json.dumps(projection)


def test_provisional_observation_never_becomes_numeric_support():
    observation = observe(completed(), label=ACTOR_PROVISIONAL, before=.6, after=.3)
    assert observation["eligible_for_numeric_calibration"] is False
    link = to_voi_link(observation)
    assert link.status == "linked_unusable"
    assessment = calibrate_observations(
        "injury", source_id="injury-source", source_version="injury-v1", cost=.1,
        observations=[observation], as_of="2026-10-10T00:00:00Z", min_independent_dependencies=1)
    assert assessment.calibrated is False


def test_five_independent_completed_receipts_produce_conservative_support():
    observations = [observe(completed(f"provider-{index}"), dependency=f"episode-{index}")
                    for index in range(5)]
    assessment = calibrate_observations(
        "injury", source_id="injury-source", source_version="injury-v1", cost=.1,
        observations=observations, as_of="2026-10-10T00:00:00Z",
        min_independent_dependencies=5, confidence=.1)
    assert assessment.calibrated is True
    assert assessment.independent_dependency_count == 5
    assert assessment.expected_information_value is not None


def test_negative_observed_reduction_is_retained_but_not_supported():
    observations = [observe(completed(f"provider-neg-{index}"), before=.2, after=.7,
                            dependency=f"negative-{index}") for index in range(5)]
    assessment = calibrate_observations(
        "injury", source_id="injury-source", source_version="injury-v1", cost=.1,
        observations=observations, as_of="2026-10-10T00:00:00Z",
        min_independent_dependencies=5)
    assert assessment.calibrated is False
    assert assessment.support_status == "OBSERVED_VALUE_OUT_OF_RANGE"


def test_gold_observation_requires_completed_receipt_and_time_order():
    receipt = completed()
    failed = dict(receipt)
    failed["status"] = "failed"
    failed.pop("receipt_digest")
    import hashlib
    failed["receipt_digest"] = hashlib.sha256(json.dumps(failed, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    observation = observe(failed)
    assert observation["eligible_for_numeric_calibration"] is False
    with pytest.raises(ValueError, match="precedes"):
        observe(receipt, observed_at="2026-10-01T09:00:00Z")


def test_tampered_observation_is_rejected_before_projection():
    observation = observe(completed())
    changed = dict(observation)
    changed["decision_loss_after"] = .99
    with pytest.raises(ValueError, match="digest"):
        to_voi_link(changed)


def test_calibration_flag_cannot_override_non_gold_label():
    observation = observe(completed(), label=ACTOR_PROVISIONAL, before=.6, after=.3)
    changed = dict(observation)
    changed["eligible_for_numeric_calibration"] = True
    changed.pop("observation_digest")
    import hashlib
    changed["observation_digest"] = hashlib.sha256(
        json.dumps(changed, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="eligibility"):
        to_voi_link(changed)


def test_unavailable_label_is_explicitly_noncalibrated():
    observation = observe(completed(), label=UNAVAILABLE, before=None, after=None)
    assert observation["eligible_for_numeric_calibration"] is False
    assert observation["reason"] == "not_independent_gold"
