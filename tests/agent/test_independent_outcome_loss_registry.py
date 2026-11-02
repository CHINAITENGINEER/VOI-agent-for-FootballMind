import copy
import pytest
from football_agent.independent_outcome_loss_registry import (
    COMPUTED, PARTIAL, UNKNOWN, IndependentOutcomeError,
    build_independent_outcome, build_loss_registry, compute_action_loss,
    public_projection,
)

CUTOFF = "2026-10-01T12:00:00Z"

def outcome(**over):
    row = {
        "source": "independent-results", "source_id": "result-1",
        "source_payload_digest": "a" * 64, "episode_id": "ep-1",
        "target_key": "match-1", "prediction_cutoff": CUTOFF,
        "kickoff": "2026-10-01T15:00:00Z", "observed_at": "2026-10-01T17:00:00Z",
        "available_at": "2026-10-01T17:01:00Z", "outcome_label": "home",
        "label_mapping": {"home": 0, "draw": 1, "away": 2}, "home_goals": 2, "away_goals": 0,
    }
    row.update(over)
    return row

def actor():
    return {"episode_id": "ep-1", "target_key": "match-1", "prediction_cutoff": CUTOFF,
            "sources": ["understat"], "source_ids": ["understat:match-1"], "source_digests": ["b" * 64]}

def action(**over):
    row = {"episode_id": "ep-1", "target_key": "match-1", "candidate_id": "lineup",
           "request_id": "req-1", "action_id": "query-lineup", "captured_at": "2026-10-01T11:59:00Z",
           "prediction_before": [.4, .3, .3], "prediction_after": [.7, .2, .1]}
    row.update(over)
    return row

def test_post_cutoff_outcome_is_sidecar_only():
    record = build_independent_outcome(outcome(), actor_binding=actor())
    assert record["available_at"] > CUTOFF
    assert record["actor_public_projection"]["outcome_visible_to_actor"] is False
    assert record["outcome_label"] == "home"
    assert public_projection(record["actor_public_projection"])["outcome_visible_to_actor"] is False

def test_same_source_or_digest_is_rejected():
    with pytest.raises(IndependentOutcomeError, match="NOT_INDEPENDENT"):
        build_independent_outcome(outcome(source="understat"), actor_binding=actor())
    with pytest.raises(IndependentOutcomeError, match="NOT_INDEPENDENT"):
        build_independent_outcome(outcome(source_payload_digest="b" * 64), actor_binding=actor())

def test_cutoff_ordering_and_clock_are_strict():
    with pytest.raises(IndependentOutcomeError, match="BEFORE_CUTOFF"):
        build_independent_outcome(outcome(available_at=CUTOFF), actor_binding=actor())
    with pytest.raises(IndependentOutcomeError, match="OBSERVED_AFTER_AVAILABLE"):
        build_independent_outcome(outcome(observed_at="2026-10-01T18:00:00Z", available_at="2026-10-01T17:01:00Z"), actor_binding=actor())

def test_complete_action_loss_requires_bound_prediction_and_result():
    record = build_independent_outcome(outcome(), actor_binding=actor())
    result = compute_action_loss(action(), record, actor_binding=actor())
    assert result["status"] == COMPUTED
    assert result["numeric_voi_training_eligible"] is True
    assert result["decision_loss_after"] < result["decision_loss_before"]
    assert result["outcome_record_digest"] == record["record_digest"]

def test_missing_capture_or_prediction_is_partial_and_not_eligible():
    record = build_independent_outcome(outcome(), actor_binding=actor())
    result = compute_action_loss(action(captured_at=None), record, actor_binding=actor())
    assert result["status"] == PARTIAL
    assert result["numeric_voi_training_eligible"] is False
    result = compute_action_loss(action(prediction_after=None), record, actor_binding=actor())
    assert result["status"] == PARTIAL

def test_mismatch_is_unknown_and_late_actions_do_not_score():
    record = build_independent_outcome(outcome(), actor_binding=actor())
    result = compute_action_loss(action(target_key="other"), record, actor_binding=actor())
    assert result["status"] == UNKNOWN
    assert "ACTION_OUTCOME_BINDING_MISMATCH" in result["reasons"]
    result = compute_action_loss(action(captured_at="2026-10-01T12:00:01Z"), record, actor_binding=actor())
    assert result["status"] == PARTIAL
    assert "ACTION_AFTER_CUTOFF" in result["reasons"]

def test_registry_mixed_completeness_stays_partial_and_public_has_no_result():
    record = build_independent_outcome(outcome(), actor_binding=actor())
    registry = build_loss_registry([action(), action(candidate_id="weather", prediction_before=None)], record, actor_binding=actor())
    assert registry["status"] == PARTIAL
    assert registry["numeric_voi_training_eligible"] is False
    text = str(registry["public_projection"])
    assert "decision_loss" not in text and "outcome_label" not in text

def test_public_projection_rejects_result_fields():
    with pytest.raises(IndependentOutcomeError, match="OUTCOME_FIELDS"):
        public_projection({"episode_id": "ep-1", "outcome_label": "home"})

def test_record_is_detached_and_tampering_detected():
    source = outcome()
    record = build_independent_outcome(source, actor_binding=actor())
    source["home_goals"] = 9
    assert record["home_goals"] == 2
    tampered = copy.deepcopy(record)
    tampered["outcome_label"] = "away"
    result = compute_action_loss(action(), tampered, actor_binding=actor())
    assert result["status"] == UNKNOWN

def test_brier_loss_supported():
    record = build_independent_outcome(outcome(), actor_binding=actor())
    result = compute_action_loss(action(), record, actor_binding=actor(), metric="brier")
    assert result["status"] == COMPUTED
    assert 0 <= result["decision_loss_after"] <= 2

def test_outcome_cannot_be_reused_with_a_different_actor_binding():
    record = build_independent_outcome(outcome(), actor_binding=actor())
    changed_actor = actor() | {"source_ids": ["fbref:match-1"]}
    result = compute_action_loss(action(), record, actor_binding=changed_actor)
    assert result["status"] == UNKNOWN
    assert result["reasons"] == ["OUTCOME_ACTOR_BINDING_MISMATCH"]

def test_equivalent_actor_cutoff_serialization_preserves_binding():
    record = build_independent_outcome(outcome(), actor_binding=actor())
    equivalent = actor() | {"prediction_cutoff": "2026-10-01T20:00:00+08:00"}
    result = compute_action_loss(action(), record, actor_binding=equivalent)
    assert result["status"] == COMPUTED
