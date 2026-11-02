from copy import deepcopy
import json

import pytest

from football_agent.evaluator_sidecar import (
    DEFAULT_EVALUATOR_ID,
    FIXTURE_VERSION,
    SidecarValidationError,
    build_evaluator_record,
    digest,
    run_evaluator_sidecar,
    verify_evaluator_record,
)
from football_agent.value_evaluation_contract import (
    ACTOR_PROVISIONAL,
    INDEPENDENT_GOLD,
    ledger_entry_to_evaluation_episode,
)


def episode():
    return ledger_entry_to_evaluation_episode({
        "entry_id": "entry-1",
        "selection_operation_id": "selection-1",
        "candidate_id": "lineup",
        "action": "search_web",
        "selection_at": "2026-09-30T10:01:00Z",
        "prediction_cutoff": "2026-09-30T10:00:00Z",
        "target_key": "match:M-1",
        "candidate_generation_complete": True,
        "candidate_value_assessments": [
            {"candidate_id": "lineup", "value_status": "UNMEASURED"},
            {"candidate_id": "weather", "value_status": "UNMEASURED"},
        ],
        "return_observation": {"body": "runtime return is not copied"},
    })


def test_synthetic_record_has_independent_candidate_losses_and_outcome():
    result = build_evaluator_record(episode())
    assert result["schema_version"] == "evaluator-sidecar-v1"
    assert result["evaluator"]["evaluator_id"] == DEFAULT_EVALUATOR_ID
    assert result["evaluator"]["independent_from_actor"] is True
    assert result["evaluator"]["fixture_version"] == FIXTURE_VERSION
    assert [row["candidate_id"] for row in result["candidate_evaluations"]] == ["lineup", "weather"]
    assert all(row["label_status"] == INDEPENDENT_GOLD for row in result["candidate_evaluations"])
    assert all(len(row["return_digest"]) == 64 for row in result["candidate_evaluations"])
    assert all(row["return_observed_at"] > "2026-09-30T10:01:00Z" for row in result["candidate_evaluations"])
    assert result["evaluation_outcome"]["label_status"] == INDEPENDENT_GOLD
    assert result["numeric_voi_training_eligible"] is False
    assert result["raw_return_payload_included"] is False
    assert result["evidence_origin"] == "synthetic_fixture"
    assert result["world_truth_verified"] is False
    assert result["for_mechanism_tests_only"] is True
    assert "runtime return is not copied" not in repr(result)
    assert result["validation"]["status"] == "counterfactual_eligible"


def test_same_episode_produces_byte_stable_record():
    first = build_evaluator_record(episode())
    second = build_evaluator_record(episode())
    assert first == second
    assert digest(first) == digest(second)


def test_unknown_candidate_loss_is_blocked():
    with pytest.raises(SidecarValidationError, match="SIDECAR_CANDIDATE_OUTSIDE_MENU"):
        build_evaluator_record(episode(), candidate_losses={"weather": 0.2, "unknown": 0.1})


def test_explicit_partial_loss_manifest_keeps_missing_candidates_unobserved():
    result = build_evaluator_record(episode(), candidate_losses={"lineup": 0.2})
    assert [row["candidate_id"] for row in result["candidate_evaluations"]] == ["lineup"]
    assert result["validation"]["status"] == "partial_observation"
    empty = build_evaluator_record(episode(), candidate_losses={})
    assert empty["candidate_evaluations"] == []
    assert empty["validation"]["status"] == "partial_observation"


def test_actor_provisional_is_never_upgraded_to_gold():
    value = episode()
    value["candidate_evaluations"] = [{
        "candidate_id": "lineup",
        "label_status": ACTOR_PROVISIONAL,
        "evaluator_id": "actor-agent",
        "return_digest": "a" * 64,
        "return_observed_at": "2026-09-30T10:01:01Z",
    }]
    with pytest.raises(SidecarValidationError, match="ACTOR_PROVISIONAL_CANNOT_BE_UPGRADED"):
        build_evaluator_record(value)


def test_actor_provisional_outcome_and_existing_gold_are_protected():
    actor = episode()
    actor["evaluation_outcome"] = {"label_status": ACTOR_PROVISIONAL}
    with pytest.raises(SidecarValidationError, match="ACTOR_PROVISIONAL_OUTCOME_PROTECTED"):
        build_evaluator_record(actor)
    gold = episode()
    gold["evaluation_outcome"] = {"label_status": INDEPENDENT_GOLD}
    with pytest.raises(SidecarValidationError, match="SIDECAR_EXISTING_OUTCOME_PROTECTED"):
        build_evaluator_record(gold)


def test_tampered_candidate_and_time_are_rejected():
    result = build_evaluator_record(episode())
    tampered = deepcopy(result)
    tampered["candidate_evaluations"][0]["candidate_id"] = "weather"
    with pytest.raises(SidecarValidationError, match="SIDECAR_RECORD_(DIGEST_MISMATCH|BLOCKED)"):
        verify_evaluator_record(tampered, episode())

    reversed_time = deepcopy(result)
    reversed_time["evaluation_outcome"]["available_at"] = "2026-09-30T10:00:00Z"
    with pytest.raises(SidecarValidationError, match="SIDECAR_RECORD_(DIGEST_MISMATCH|BLOCKED)"):
        verify_evaluator_record(reversed_time, episode())

    loss_tampered = deepcopy(result)
    loss_tampered["candidate_evaluations"][0]["decision_loss_after"] = 0.001
    with pytest.raises(SidecarValidationError, match="SIDECAR_RECORD_DIGEST_MISMATCH"):
        verify_evaluator_record(loss_tampered, episode())

    label_tampered = deepcopy(result)
    label_tampered["evaluation_outcome"]["outcome_label"] = "draw"
    with pytest.raises(SidecarValidationError, match="SIDECAR_RECORD_DIGEST_MISMATCH"):
        verify_evaluator_record(label_tampered, episode())


def test_episode_digest_detects_public_snapshot_tamper():
    result = build_evaluator_record(episode())
    changed = deepcopy(episode())
    changed["pre_action"]["snapshot"]["candidate_id"] = "weather"
    with pytest.raises(SidecarValidationError, match="SIDECAR_EPISODE_DIGEST_MISMATCH"):
        verify_evaluator_record(result, changed)


def test_runner_writes_independent_manifest_and_stable_results(tmp_path):
    source = [episode()]
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = run_evaluator_sidecar(source, first_dir)
    second = run_evaluator_sidecar(source, second_dir)
    assert first["manifest"] == second["manifest"]
    assert first["results"] == second["results"]
    assert json.loads((first_dir / "manifest.json").read_text(encoding="utf-8"))["runtime_reads_sidecar"] is False
    assert (first_dir / "results.json").read_bytes() == (second_dir / "results.json").read_bytes()
    assert first["status_counts"] == {"counterfactual_eligible": 1}
    assert first["manifest"]["evidence_origin"] == "synthetic_fixture"
    assert first["manifest"]["synthetic_clock"] == "deterministic_synthetic_clock"
    assert first["manifest"]["generated_at"] >= "2026-09-30T10:01:00Z"


def test_runner_records_blocked_episode_without_emitting_gold_rows(tmp_path):
    value = episode()
    value["pre_action"]["candidate_ids"] = ["lineup", "unknown"]
    result = run_evaluator_sidecar([value], tmp_path)
    assert result["status_counts"] == {"blocked": 1}
    blocked = result["results"][0]
    assert blocked["status"] == "blocked"
    assert "candidate_evaluations" not in blocked
    assert blocked["actor_assessment_is_gold"] is False


def test_runner_explicit_episode_loss_mapping_preserves_unobserved_rows(tmp_path):
    result = run_evaluator_sidecar(
        [episode()], tmp_path, candidate_losses={"entry-1": {"lineup": 0.2}},
    )
    assert result["status_counts"] == {"partial_observation": 1}
    assert [row["candidate_id"] for row in result["results"][0]["candidate_evaluations"]] == ["lineup"]


def test_generated_at_must_be_after_selection():
    with pytest.raises(SidecarValidationError, match="SIDECAR_GENERATED_AT_BEFORE_SELECTION"):
        run_evaluator_sidecar([episode()], "var/parallel/agent/round133-a/test-output",
                              generated_at="2026-09-30T09:00:00Z")
