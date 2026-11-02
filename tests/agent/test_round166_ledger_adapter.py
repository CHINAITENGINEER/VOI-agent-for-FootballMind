"""Round166 ledger-to-episode adapter boundaries."""
from copy import deepcopy
import hashlib
import json

from football_agent.round166_ledger_adapter import VERSION, adapt_action_value_ledger


S0 = "0" * 64
S1 = "1" * 64
A0 = "2" * 64


def _ledger(*, ask_user=False, complete=True):
    terminal_action = "ask_user" if ask_user else "stop"
    return {
        "schema_version": "action-value-ledger-v1",
        "run_id": "episode-166",
        "summary": {"cost_basis": "runtime reservations; not money or loss units"},
        "entries": [
            {
                "sequence": 1,
                "entry_id": "entry-1",
                "selection_operation_id": "selection-1",
                "target_key": "match:166",
                "candidate_id": "lineup",
                "action": "search_web",
                "selection_at": "2026-10-01T10:00:00Z",
                "prediction_cutoff": "2026-10-01T12:00:00Z",
                "pre_action_snapshot_digest": S0,
                "candidate_generation_complete": complete,
                "expected_cost": .1,
                "acquisition_id": "acq-1",
                "selection_resource_cost": {"llm_calls": 1},
                "resource_cost": {"http_calls": 1, "local_reads": 0, "llm_calls": 0},
                "candidate_value_assessments": [
                    {"candidate_id": "lineup", "value": {"cost": .1}},
                    {"candidate_id": "weather", "value": {"cost": .02}},
                ],
            },
            {
                "sequence": 2,
                "entry_id": "entry-2",
                "selection_operation_id": "selection-2",
                "target_key": "match:166",
                # The runtime historically carried an opaque candidate on a
                # terminal row.  The adapter must use action=stop, not this ID.
                "candidate_id": "opaque-terminal-id",
                "action": terminal_action,
                "selection_at": "2026-10-01T10:01:00Z",
                "prediction_cutoff": "2026-10-01T12:00:00Z",
                "pre_action_snapshot_digest": S1,
                "candidate_generation_complete": complete,
                "expected_cost": None,
                "selection_resource_cost": {"llm_calls": 1},
                "resource_cost": {"http_calls": 0, "local_reads": 0, "llm_calls": 0},
                "candidate_value_assessments": [
                    {"candidate_id": "lineup", "value": {"cost": .1}},
                    {"candidate_id": "weather", "value": {"cost": .02}},
                ],
            },
        ],
    }


def _enrichment():
    return {
        "selection-1": {
            "captured_at": "2026-10-01T10:00:10Z",
            "state_before_digest": S0,
            "state_after_digest": S1,
            "prediction_before": [.5, .3, .2],
            "prediction_after": [.6, .25, .15],
            "observed_outcome": {"kind": "lineup_return"},
            "aggregate": {
                "episode_id": "episode-166",
                "target_key": "match:166",
                "candidate_id": "lineup",
                "request_id": "acq-1",
                "prediction_cutoff": "2026-10-01T12:00:00Z",
                "availability": "available",
                "quality": "VALIDATED",
                "operational_available_at": "2026-10-01T10:00:30Z",
                "captured_at": "2026-10-01T10:00:10Z",
                "aggregate_id": A0,
                "source_ids": ["source:lineup"],
            },
        },
        "selection-2": {
            "captured_at": "2026-10-01T10:01:00Z",
            "state_before_digest": S1,
            "prediction_before": [.6, .25, .15],
            "stop_reason": "remaining_value_below_cost",
            "remaining_candidate_ids": [],
        },
    }


def _sidecar():
    return {
        "label_status": "INDEPENDENT_GOLD_RECONSTRUCTED",
        "sequence_losses": {"lineup": .4},
        "action_returns": {"lineup": {"available": True}},
    }


def test_complete_enriched_ledger_reaches_structural_replay_only():
    ledger = _ledger()
    enrichment = _enrichment()
    ledger["episode_evidence_digest"] = hashlib.sha256(
        json.dumps(enrichment, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    result = adapt_action_value_ledger(
        ledger,
        candidate_menu=[
            {"candidate_id": "lineup", "cost": .1, "source_id": "source:lineup"},
            {"candidate_id": "weather", "cost": .02, "source_id": "source:weather"},
        ],
        trace_enrichment=enrichment,
        evaluator_sidecar=_sidecar(),
    )
    assert result["schema_version"] == VERSION
    assert result["report"]["status"] == "COMPLETE_ACTION_EVIDENCE"
    assert result["report"]["eligible_for_replay"] is True
    assert result["report"]["numeric_voi_training_eligible"] is False
    assert result["traces"][1]["action_id"] == "STOP"
    assert result["traces"][1]["candidate_id"] == "STOP"
    assert result["aggregates"][0]["aggregate_id"] == A0


def test_ledger_only_does_not_fabricate_prediction_or_availability_evidence():
    result = adapt_action_value_ledger(_ledger())
    report = result["report"]
    assert report["status"] == "PARTIAL_ACTION_EVIDENCE"
    assert report["eligible_for_replay"] is False
    assert report["numeric_voi_training_eligible"] is False
    assert "traces_missing" not in report["reasons"]
    assert any("prediction" in reason or "aggregate" in reason for reason in report["reasons"])
    assert result["aggregates"] == []


def test_incomplete_menu_and_ask_user_are_not_autonomous_stop():
    ledger = _ledger(ask_user=True, complete=False)
    enrichment = _enrichment()
    ledger["episode_evidence_digest"] = hashlib.sha256(
        json.dumps(enrichment, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    result = adapt_action_value_ledger(
        ledger,
        candidate_menu=[
            {"candidate_id": "lineup", "cost": .1},
            {"candidate_id": "weather", "cost": .02},
        ],
        trace_enrichment=enrichment,
        evaluator_sidecar=_sidecar(),
    )
    reasons = set(result["report"]["reasons"])
    assert result["report"]["status"] == "PARTIAL_ACTION_EVIDENCE"
    assert result["report"]["eligible_for_replay"] is False
    assert "ask_user_terminal_is_not_autonomous_stop" in reasons
    assert "open_world_candidate_menu_not_complete" in reasons
    assert result["traces"][1]["action_id"] == "USER_INPUT_REQUIRED"


def test_cost_units_are_preserved_and_never_converted_to_loss():
    result = adapt_action_value_ledger(_ledger())
    first = result["cost_projection"][0]
    assert first["selection_resource_cost"] == {"llm_calls": 1}
    assert first["acquisition_resource_cost"] == {"http_calls": 1, "local_reads": 0, "llm_calls": 0}
    assert first["cost_basis"] == "preserved_separately; no conversion to decision loss"
    assert result["numeric_voi_training_eligible"] is False


def test_public_projection_does_not_copy_return_or_evaluator_fields():
    ledger = _ledger()
    ledger["entries"][0]["return_observation"] = {
        "body": "private body", "outcome": "later", "source_id": "private-source"
    }
    ledger["entries"][0]["return_assessment"] = {"outcome": "discovery_only"}
    result = adapt_action_value_ledger(ledger)
    public_text = repr(result["episode"])
    assert "private body" not in public_text
    assert "discovery_only" not in public_text
    assert "outcome" not in public_text.casefold()


def test_malformed_rows_and_missing_identity_cannot_be_filtered_into_complete():
    ledger = _ledger()
    ledger["entries"].append(None)
    result = adapt_action_value_ledger(ledger)
    assert result["report"]["eligible_for_replay"] is False
    assert "ledger_entry_not_mapping" in result["report"]["reasons"]
    ledger = _ledger()
    ledger["entries"][1]["target_key"] = None
    result = adapt_action_value_ledger(ledger)
    assert result["report"]["eligible_for_replay"] is False
    assert "entry_1_target_key_missing" in result["report"]["reasons"]


def test_external_evidence_digest_is_required_and_tamper_is_visible():
    ledger = _ledger()
    enrichment = _enrichment()
    ledger["episode_evidence_digest"] = hashlib.sha256(
        json.dumps(enrichment, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    changed = deepcopy(enrichment)
    changed["selection-1"]["prediction_after"] = [.55, .25, .2]
    result = adapt_action_value_ledger(ledger, candidate_menu=[
        {"candidate_id": "lineup", "cost": .1}, {"candidate_id": "weather", "cost": .02}],
        trace_enrichment=changed, evaluator_sidecar=_sidecar())
    assert result["report"]["eligible_for_replay"] is False
    assert "trace_enrichment_unbound" in result["report"]["reasons"]


def test_defer_and_stop_only_cannot_be_promoted_to_autonomous_replay():
    ledger = _ledger()
    ledger["entries"][1]["action"] = "defer"
    enrichment = _enrichment()
    ledger["episode_evidence_digest"] = hashlib.sha256(
        json.dumps(enrichment, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    result = adapt_action_value_ledger(ledger, candidate_menu=[
        {"candidate_id": "lineup", "cost": .1}, {"candidate_id": "weather", "cost": .02}],
        trace_enrichment=enrichment, evaluator_sidecar=_sidecar())
    assert result["report"]["eligible_for_replay"] is False
    assert "defer_terminal_is_not_autonomous_stop" in result["report"]["reasons"]
    stop_only = _ledger()["entries"][1:]
    only = _ledger()
    only["entries"] = stop_only
    result = adapt_action_value_ledger(only, candidate_menu=[
        {"candidate_id": "lineup", "cost": .1}], evaluator_sidecar={
            "label_status": "INDEPENDENT_GOLD_RECONSTRUCTED",
            "sequence_losses": {"": .4}, "action_returns": {},
        })
    assert result["report"]["eligible_for_replay"] is False
    assert "no_query_action_in_episode" in result["report"]["reasons"]


def test_multi_request_acquisition_requires_an_action_level_identity():
    ledger = _ledger()
    ledger["entries"][0].pop("acquisition_id")
    ledger["entries"][0]["request_ids"] = ["request-1", "request-2"]
    result = adapt_action_value_ledger(ledger)
    assert result["report"]["eligible_for_replay"] is False
    assert "entry_0_multi_request_action_id_missing" in result["report"]["reasons"]


def test_conflicting_candidate_costs_block_automatic_menu_projection():
    ledger = _ledger()
    ledger["entries"][0]["candidate_value_assessments"].append(
        {"candidate_id": "weather", "value": {"cost": .9}})
    result = adapt_action_value_ledger(ledger)
    assert result["report"]["eligible_for_replay"] is False
    assert "candidate_cost_conflict:weather" in result["report"]["reasons"]
