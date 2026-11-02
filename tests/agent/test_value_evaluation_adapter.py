from copy import deepcopy

import pytest

from football_agent.value_evaluation_contract import (
    PARTIAL_OBSERVATION,
    ledger_entry_to_evaluation_episode,
    ledger_to_evaluation_episodes,
    validate_evaluation_episode,
)


def entry(**overrides):
    value = {
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
        "return_observation": {"body": "technical source return"},
        "result_digest": "f" * 64,
    }
    value.update(overrides)
    return value


def test_adapter_emits_public_before_state_and_keeps_sidecar_empty():
    episode = ledger_entry_to_evaluation_episode(entry())
    assert episode["schema_version"] == "value-evaluation-contract-v1"
    assert episode["pre_action"]["candidate_ids"] == ["lineup", "weather"]
    assert episode["pre_action"]["selected_candidate_id"] == "lineup"
    assert episode["candidate_evaluations"] == []
    assert episode["evaluation_outcome"] is None
    assert episode["evaluator_sidecar"]["runtime_can_read"] is False
    assert episode["ledger_binding"]["entry_id"] == "entry-1"
    assert episode["ledger_binding"]["candidate_menu_digest"]
    assert "return_observation" not in episode["pre_action"]["snapshot"]
    report = validate_evaluation_episode(episode)
    assert report["status"] == PARTIAL_OBSERVATION
    assert report["eligibility"]["numeric_voi_training_eligible"] is False


def test_adapter_accepts_explicit_public_snapshot_but_does_not_sanitize_leaks():
    public = {"task_id": "M-1", "known": {"ranking_gap": 0.3}}
    episode = ledger_entry_to_evaluation_episode(entry(), public_snapshot=public)
    assert episode["pre_action"]["snapshot"] == public
    tampered = deepcopy(public)
    tampered["result"] = "later"
    blocked = ledger_entry_to_evaluation_episode(entry(), public_snapshot=tampered)
    assert validate_evaluation_episode(blocked)["status"] == "blocked"


def test_adapter_is_deterministic_and_does_not_copy_technical_return():
    ledger = {"run_id": "run-1", "entries": [entry()]}
    one = ledger_to_evaluation_episodes(ledger)
    two = ledger_to_evaluation_episodes(ledger)
    assert one == two
    assert one[0]["target_key"] == "match:M-1"
    assert "technical source return" not in repr(one)
    assert one[0]["ledger_binding"]["state_digest"] is None


def test_adapter_rejects_invalid_ledger_shape():
    with pytest.raises(ValueError, match="VALUE_ADAPTER_LEDGER_ENTRIES_REQUIRED"):
        ledger_to_evaluation_episodes({"run_id": "run-1"})


def test_ledger_binding_detects_candidate_menu_tampering():
    episode = ledger_entry_to_evaluation_episode(entry())
    episode["pre_action"]["candidate_ids"] = ["weather", "lineup"]
    assert validate_evaluation_episode(episode)["status"] == "blocked"
