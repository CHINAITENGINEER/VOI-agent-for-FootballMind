"""Contracts for the isolated Round161 evaluator, not calibrated VoI claims."""
from copy import deepcopy
import json

import pytest

from football_agent.round161_offline_evaluator import evaluate_episode


def trace(condition="complete", **changes):
    step = {"sequence": 1, "action_id": "read-injury", "receipt_id": "receipt-1",
            "condition": condition, "read_succeeded": True,
            "observed_at": "2026-10-01T11:50:00Z", "available_at": "2026-10-01T11:51:00Z",
            "source_refs": ["archive:injury-1"], "context_patch": {"starter_status": "doubtful"},
            "declared_cost": .02}
    step.update(changes)
    return {"episode_id": "episode-161", "target_key": "spftll:123", "prediction_cutoff": "2026-10-01T12:00:00Z",
            "max_age_seconds": 3600, "initial_context": {"starter_status": "unknown"},
            "steps": [step], "terminal_action": "defer", "terminal_reason": "unmeasured_remaining_information_value",
            "policy_origin": "scripted_fixture"}


def sidecar(*losses):
    return {"episode_id": "episode-161", "evaluation_id": "fixture-evaluator", "label_status": "synthetic_fixture",
            "label_available_at": "2026-10-01T20:00:00Z", "loss_unit": "decision_loss",
            "sequence_losses": [{"sequence": index, "loss": value} for index, value in enumerate(losses)]}


def test_complete_evidence_advances_context_preserving_original_clocks():
    public = trace()
    old = deepcopy(public)
    result = evaluate_episode(public, sidecar(.6, .4))
    assert result["public"]["final_context"] == {"starter_status": "doubtful"}
    assert result["public"]["events"][0]["observed_at"] == "2026-10-01T11:50:00Z"
    assert result["public"]["events"][0]["evidence_admitted"] is True
    assert result["summary"]["observed_loss_reduction"] == pytest.approx(.2)
    assert public == old


def test_missing_evidence_does_not_promote_returned_patch():
    result = evaluate_episode(trace("missing", read_succeeded=False, source_refs=[], observed_at=None, available_at=None))
    assert result["public"]["final_context"] == {"starter_status": "unknown"}
    assert result["summary"]["read_success_count"] == 0
    assert result["summary"]["replan_count"] == 1
    assert result["public"]["events"][0]["replan_reason"] == "evidence_missing"


def test_stale_condition_and_old_complete_return_both_require_replan():
    for public in (trace("stale"), trace(observed_at="2026-10-01T09:00:00Z")):
        result = evaluate_episode(public)
        assert result["public"]["final_context"] == public["initial_context"]
        assert result["public"]["events"][0]["effective_condition"] == "stale"
        assert result["public"]["events"][0]["replanning_required"] is True


def test_conflict_does_not_overwrite_previous_context():
    public = trace("conflict")
    public["steps"][0]["source_refs"] = ["source:a", "source:b"]
    result = evaluate_episode(public)
    assert result["public"]["final_context"]["starter_status"] == "unknown"
    assert result["public"]["events"][0]["replan_reason"] == "evidence_conflict"
    assert result["summary"]["condition_counts"]["conflict"] == 1


def test_conflict_requires_independent_source_refs():
    with pytest.raises(ValueError, match="at least two source_refs"):
        evaluate_episode(trace("conflict"))


def test_late_return_is_audit_only_even_if_declared_complete():
    for condition in ("late", "complete"):
        result = evaluate_episode(trace(condition, available_at="2026-10-01T12:00:01Z"))
        assert result["public"]["events"][0]["effective_condition"] == "late"
        assert result["summary"]["evidence_admitted_count"] == 0
        assert result["public"]["final_context"]["starter_status"] == "unknown"


def test_sidecar_changes_never_change_actor_projection_or_context_digest():
    better = evaluate_episode(trace(), sidecar(.6, .2))
    worse = evaluate_episode(trace(), sidecar(.6, .9))
    absent = evaluate_episode(trace())
    assert better["public"] == worse["public"] == absent["public"]
    assert better["sidecar"]["public_trace_digest"] == worse["sidecar"]["public_trace_digest"]
    public_text = json.dumps(better["public"])
    assert "sequence_losses" not in public_text
    assert "observed_loss_reduction" not in public_text
    assert "fixture-evaluator" not in public_text


def test_nested_evaluator_fields_are_rejected_at_public_boundary():
    for location in ("initial_context", "context_patch"):
        for key in ("gold", "sequence_losses", "private_outcome", "evaluator_label", "Decision_Loss"):
            public = trace()
            target = public if location == "initial_context" else public["steps"][0]
            target[location] = {"nested": [{key: .2}]}
            with pytest.raises(ValueError, match="evaluator field"):
                evaluate_episode(public)


def test_missing_intermediate_sequence_loss_invalidates_entire_sequence():
    public = trace()
    public["steps"].append({**public["steps"][0], "sequence": 2, "receipt_id": "receipt-2"})
    evaluation = sidecar(.6, .4, .2)
    evaluation["sequence_losses"].pop(1)
    result = evaluate_episode(public, evaluation)
    assert result["sidecar"]["missing_loss_sequences"] == [1]
    assert result["summary"]["sequence_loss_eligible"] is False
    assert result["summary"]["observed_loss_reduction"] is None
    assert all(item["observed_loss_reduction"] is None for item in result["sidecar"]["step_measurements"])


def test_successful_read_is_not_a_positive_gain_without_loss_measurements():
    result = evaluate_episode(trace())
    assert result["summary"]["read_success_count"] == 1
    assert result["summary"]["context_change_count"] == 1
    assert result["summary"]["sequence_loss_eligible"] is False
    assert result["summary"]["observed_loss_reduction"] is None
    zero = evaluate_episode(trace(), sidecar(.4, .4))
    assert zero["summary"]["observed_loss_reduction"] == 0
    assert zero["summary"]["observed_net_loss_reduction"] == -.02


def test_negative_gain_is_preserved_instead_of_clipped_or_calibrated():
    result = evaluate_episode(trace(), sidecar(.2, .7))
    assert result["summary"]["observed_loss_reduction"] == pytest.approx(-.5)
    assert result["summary"]["observed_net_loss_reduction"] == pytest.approx(-.52)
    for part in result.values():
        assert part["numeric_voi_calibrated"] is False
        assert part["training_eligible"] is False
    assert result["sidecar"]["observed_gain_is_expected_voi"] is False


def test_loss_reductions_telescope_and_costs_include_unusable_attempts():
    public = trace()
    public["steps"].append({**public["steps"][0], "sequence": 2, "receipt_id": "receipt-2",
                            "condition": "conflict", "source_refs": ["source:a", "source:b"],
                            "declared_cost": .03})
    result = evaluate_episode(public, sidecar(.7, .4, .5))
    deltas = [item["observed_loss_reduction"] for item in result["sidecar"]["step_measurements"]]
    assert sum(deltas) == pytest.approx(result["summary"]["observed_loss_reduction"])
    assert result["summary"]["declared_acquisition_cost"] == pytest.approx(.05)
    assert result["summary"]["observed_net_loss_reduction"] == pytest.approx(.15)
    assert result["summary"]["context_change_count"] == 1
    assert result["summary"]["replan_count"] == 1


def test_sequence_and_receipt_identity_mismatches_are_rejected():
    for changed in ({"sequence": True}, {"sequence": 2}):
        with pytest.raises(ValueError, match="sequences"):
            evaluate_episode(trace(**changed))
    public = trace()
    public["steps"].append({**public["steps"][0], "sequence": 2})
    with pytest.raises(ValueError, match="duplicate receipt"):
        evaluate_episode(public)
    evaluation = sidecar(.4, .3)
    evaluation["episode_id"] = "other-episode"
    with pytest.raises(ValueError, match="episode mismatch"):
        evaluate_episode(trace(), evaluation)


def test_bad_numbers_clocks_and_duplicate_loss_sequences_are_rejected():
    for invalid in (True, -.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite number"):
            evaluate_episode(trace(declared_cost=invalid))
        evaluation = sidecar(.4, invalid)
        with pytest.raises(ValueError, match="finite number"):
            evaluate_episode(trace(), evaluation)
    for observed, available in (("2026-10-01T12:00:00", "2026-10-01T12:00:00Z"),
                                ("2026-10-01T11:52:00Z", "2026-10-01T11:51:00Z")):
        with pytest.raises(ValueError):
            evaluate_episode(trace(observed_at=observed, available_at=available))
    evaluation = sidecar(.4, .3)
    evaluation["sequence_losses"].append({"sequence": 1, "loss": .2})
    with pytest.raises(ValueError, match="unique in-range"):
        evaluate_episode(trace(), evaluation)


def test_provenance_and_terminal_stop_are_never_upgraded_to_verified_voi():
    public = trace()
    public["terminal_action"] = "stop"
    evaluation = sidecar(.4, .3)
    evaluation["label_status"] = "independent_evaluator"
    result = evaluate_episode(public, evaluation)
    assert result["sidecar"]["provenance_verified"] is False
    assert result["summary"]["stop_optimality"] == "unmeasured"
    assert result["summary"]["policy_execution_measured"] is False
    assert result["sidecar"]["training_eligible"] is False
    evaluation["label_status"] = "unavailable"
    result = evaluate_episode(public, evaluation)
    assert result["summary"]["eligibility_reason"] == "evaluation_labels_unavailable"
    assert result["summary"]["observed_loss_reduction"] is None


def test_complete_requires_support_and_identical_patch_does_not_invent_change():
    for changes in ({"read_succeeded": False}, {"source_refs": []}, {"observed_at": None}, {"available_at": None}):
        with pytest.raises(ValueError, match="complete evidence requires"):
            evaluate_episode(trace(**changes))
    result = evaluate_episode(trace(context_patch={"starter_status": "unknown"}), sidecar(.4, .4))
    assert result["summary"]["evidence_admitted_count"] == 1
    assert result["summary"]["context_change_count"] == 0
