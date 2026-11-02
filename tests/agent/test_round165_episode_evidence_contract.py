"""Round165 tests for action-level episode completeness."""
from copy import deepcopy
import hashlib
import json

from football_agent.round165_episode_evidence_contract import validate_round165_episode_evidence


CUTOFF = "2026-10-01T12:00:00Z"
STATE0 = "0" * 64
STATE1 = "1" * 64
AGGREGATE = "2" * 64


def _fixture():
    public = {
        "episode_id": "ep-165", "target_key": "match-165", "prediction_cutoff": CUTOFF,
        "candidate_menu": [{"candidate_id": "lineup", "cost": .1, "source_id": "lineup"}],
        "public_snapshot_digest": "snapshot-digest-placeholder", "match_snapshot": {"match_id": "m"},
        "initial_state_digest": STATE0,
    }
    trace = [
        {"episode_id": "ep-165", "target_key": "match-165", "step": 0,
         "action_id": "lineup", "candidate_id": "lineup", "request_id": "req-1",
         "captured_at": "2026-10-01T10:00:00Z", "expected_cost": .1,
         "state_before_digest": STATE0, "state_after_digest": STATE1,
         "prediction_before": [.5, .3, .2], "prediction_after": [.6, .25, .15],
         "observed_outcome": "lineup_known"},
        {"episode_id": "ep-165", "target_key": "match-165", "step": 1,
         "action_id": "STOP", "candidate_id": "STOP", "request_id": "req-2",
         "captured_at": "2026-10-01T10:01:00Z", "expected_cost": 0,
         "state_before_digest": STATE1, "prediction_before": [.6, .25, .15], "stop_reason": "all_upper_bounds_at_or_below_cost"},
    ]
    aggregates = [{"episode_id": "ep-165", "target_key": "match-165", "candidate_id": "lineup",
                   "request_id": "req-1", "prediction_cutoff": CUTOFF, "availability": "available",
                   "quality": "VALIDATED", "operational_available_at": "2026-10-01T10:00:30Z",
                   "captured_at": "2026-10-01T10:00:20Z", "aggregate_id": AGGREGATE,
                   "source_ids": ["source:lineup"]}]
    sidecar = {"label_status": "INDEPENDENT_GOLD", "sequence_losses": {"lineup": .4},
               "action_returns": {"lineup": {"available": True}}}
    return public, trace, aggregates, sidecar


def test_complete_trace_return_stop_and_independent_loss_are_required():
    public, trace, aggregates, sidecar = _fixture()
    public["public_snapshot_digest"] = __import__("hashlib").sha256(
        __import__("json").dumps(public["match_snapshot"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    report = validate_round165_episode_evidence(public, trace, aggregates, evaluator_sidecar=sidecar)
    assert report["status"] == "COMPLETE_ACTION_EVIDENCE"
    assert report["eligible_for_replay"] is True
    assert report["numeric_voi_training_eligible"] is False
    assert report["selected_sequence"] == "lineup"
    assert report["public_projection"]["hidden_result_included"] is False


def test_missing_stop_is_partial_and_not_promoted():
    public, trace, aggregates, sidecar = _fixture()
    report = validate_round165_episode_evidence(public, trace[:1], aggregates, evaluator_sidecar=sidecar)
    assert report["eligible_for_replay"] is False
    assert "terminal_stop_missing" in report["reasons"]


def test_late_action_return_is_partial_even_when_trace_is_present():
    public, trace, aggregates, sidecar = _fixture()
    aggregates[0]["operational_available_at"] = "2026-10-01T12:00:01Z"
    report = validate_round165_episode_evidence(public, trace, aggregates, evaluator_sidecar=sidecar)
    assert report["eligible_for_replay"] is False
    assert "trace_0_aggregate_after_cutoff" in report["reasons"]


def test_partial_loss_and_missing_action_return_stay_unknown():
    public, trace, aggregates, sidecar = _fixture()
    sidecar["sequence_losses"] = {}
    sidecar["action_returns"] = {}
    report = validate_round165_episode_evidence(public, trace, aggregates, evaluator_sidecar=sidecar)
    assert report["eligible_for_replay"] is False
    assert "executed_sequence_loss_missing" in report["reasons"]
    assert "action_return_missing:lineup" in report["reasons"]


def test_prediction_continuity_and_cost_are_checked():
    public, trace, aggregates, sidecar = _fixture()
    trace[1]["prediction_before"] = [.5, .3, .2]
    trace[0]["expected_cost"] = .2
    report = validate_round165_episode_evidence(public, trace, aggregates, evaluator_sidecar=sidecar)
    assert report["eligible_for_replay"] is False
    assert "trace_1_prediction_continuity" in report["reasons"]
    assert "trace_0_cost_mismatch" in report["reasons"]


def test_public_result_fields_are_rejected_before_any_evidence_claim():
    public, trace, aggregates, sidecar = _fixture()
    public["match_snapshot"]["home_goals"] = 2
    report = validate_round165_episode_evidence(public, trace, aggregates, evaluator_sidecar=sidecar)
    assert report["eligible_for_replay"] is False
    assert "public episode contains evaluator fields" in report["reasons"][0]
