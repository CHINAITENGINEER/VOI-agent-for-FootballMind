from __future__ import annotations

from copy import deepcopy
from datetime import datetime

import pytest

from football_research.data_adapters.prospective_outcomes import derive_g_candidates
from football_spider.storage.raw_archive import RawArchive


@pytest.fixture
def complete_evidence(tmp_path):
    archive = RawArchive(tmp_path / "stream")
    kickoff = "2026-10-10T12:00:00+00:00"
    observed = "2026-10-09T09:00:02+00:00"

    def raw(label, stamp=observed):
        obs = archive.save("test_evidence", f"https://example.test/{label}", {}, label.encode(),
                           status=200, observed_at=datetime.fromisoformat(stamp))
        return {"raw_path": obs.payload_path, "raw_sha256": obs.content_hash}

    fixture = {"physical_match_id": "m1", "information_mode": "prospective_capture", "kickoff_at": kickoff,
               "registered_at": "2026-10-09T07:00:00+00:00", "arm_assigned_at": "2026-10-09T07:01:00+00:00",
               "pre_kickoff_frozen": True, "policy_arm": "fixed_order", "policy_version": "v1", **raw("fixture")}
    menu = {"discovery_id": "menu1", "physical_match_id": "m1", "information_mode": "prospective_capture",
            "observed_at": "2026-10-09T08:00:00+00:00", "original_agent_menu_proven": True,
            "policy_version": "v1", "visible_candidates": [{"candidate_id": "candidate1", "action_id": "news"}],
            **raw("menu")}
    attempt = {"attempt_id": "a1", "physical_match_id": "m1", "discovery_id": "menu1", "candidate_id": "candidate1",
               "action_id": "news", "selected_by_policy": True, "information_mode": "prospective_capture",
               "policy_arm": "fixed_order", "policy_version": "v1", "selection_basis": "fixed_order",
               "selection_probability": 1.0, "selection_probability_basis": "fixed_order_next_action",
               "decision_at": "2026-10-09T09:00:00+00:00", "requested_at": "2026-10-09T09:00:01+00:00",
               "request_started_at": "2026-10-09T09:00:01+00:00", "response_observed_at": observed,
               "status": "returned", "request_count": 1, "acquisition_cost": 0.1,
               "returned_document_id": "doc1", **raw("action")}
    before = {"physical_match_id": "m1", "parent_attempt_id": "a1", "snapshot_kind": "before_action",
              "observed_at": "2026-10-09T08:59:00+00:00", "captured_at": "2026-10-09T08:59:01+00:00",
              "available_at": "2026-10-09T08:59:00+00:00", "prediction_cutoff": "2026-10-09T08:59:00+00:00",
              "availability_basis": "runtime_state_handoff", "availability_status": "known",
              "information_mode": "prospective_capture", "prediction_before": {"home": .5, "draw": .3, "away": .2}}
    after = {"physical_match_id": "m1", "parent_attempt_id": "a1", "snapshot_kind": "after_action",
             "observed_at": "2026-10-09T09:00:04+00:00", "captured_at": "2026-10-09T09:00:05+00:00",
             "available_at": "2026-10-09T09:00:04+00:00", "prediction_cutoff": "2026-10-09T09:00:04+00:00",
             "availability_basis": "runtime_state_handoff", "availability_status": "known",
             "information_mode": "prospective_capture", "prediction_after": {"home": .55, "draw": .25, "away": .2}}
    result = {"physical_match_id": "m1", "kickoff_at": kickoff, "regulation_home_goals": 1,
              "regulation_away_goals": 0, "result": "home", "result_status": "final", "score_period": "regulation",
              "result_source_url": "https://example.test/result", "result_observed_at": "2026-10-10T15:00:00+00:00",
              "information_mode": "post_match_outcome", "independent_outcome_verified": True,
              **raw("result", "2026-10-10T15:00:00+00:00")}
    document = {"document_id": "doc1", "physical_match_id": "m1", "information_mode": "prospective_capture",
                "observed_at": observed, "captured_at": "2026-10-09T09:00:03+00:00", "capture_status": "known",
                "capture_timing_basis": "measured_local_utc_archive_save_return",
                "available_at": "2026-10-09T09:00:04+00:00", "availability_status": "known",
                "availability_basis": "verified_runtime_state_handoff", **raw("document")}
    return {"fixtures": [fixture], "discoveries": [menu], "attempts": [attempt],
            "contexts": [before, after], "results": [result], "documents": [document], "archive_root": archive.root}


def test_returned_document_requires_independent_measured_clocks_and_verified_bytes(complete_evidence):
    row = derive_g_candidates(**complete_evidence)[0]
    assert row["label_status"] == "eligible", row
    assert row["eligibility_checks"]["document_capture_proven"] is True
    assert row["eligibility_checks"]["document_availability_proven"] is True
    assert row["raw_digest_checks"]["document"] is True
    assert row["realized_gain"] is not None


def test_declared_f_requires_predictor_pair_consumption_proof(complete_evidence):
    # The fixture has a valid legacy before/after pair, but no immutable
    # predictor-context payload. A run that declares a fitted f must not turn
    # that missing proof into a zero-gain training row.
    complete_evidence["attempts"][0]["f_consumption_required"] = True
    row = derive_g_candidates(**complete_evidence)[0]
    assert row["label_status"] == "ineligible"
    assert "f_consumption_not_proven" in row["exclusion_reason"]
    assert row["realized_gain"] is None


@pytest.mark.parametrize("field,value,reason", [
    ("observed_at", None, "document_observed_time_unknown_or_after_kickoff"),
    ("captured_at", None, "document_capture_time_unknown_or_after_kickoff"),
    ("available_at", None, "document_availability_time_unknown_or_after_kickoff"),
    ("captured_at", "2026-10-10T12:00:00+00:00", "document_capture_time_unknown_or_after_kickoff"),
    ("available_at", "2026-10-10T12:00:00+00:00", "document_availability_time_unknown_or_after_kickoff"),
    ("capture_status", "unknown", "document_capture_not_proven"),
    ("capture_timing_basis", "response_observed_at", "document_capture_not_proven"),
    ("capture_timing_basis", "measured_local_utc_archive_save_start", "document_capture_not_proven"),
    ("document_admitted", False, "document_not_admitted"),
    ("document_admitted", None, "document_not_admitted"),
    ("availability_basis", None, "document_availability_not_proven"),
    ("availability_status", "unknown", "document_availability_not_proven"),
    ("available_at", "2026-10-09T09:00:06+00:00", "document_available_after_prediction_or_unknown"),
    ("captured_at", "2026-10-09T09:00:01+00:00", "document_capture_time_order_invalid"),
    ("physical_match_id", "other", "document_fixture_binding_not_proven"),
    ("information_mode", "historical_reconstruction", "document_not_prospective"),
    ("raw_sha256", "0" * 64, "document_raw_digest_unverified"),
])
def test_unproven_returned_document_keeps_gain_null(complete_evidence, field, value, reason):
    complete_evidence["documents"][0][field] = value
    row = derive_g_candidates(**complete_evidence)[0]
    assert row["label_status"] == "ineligible"
    assert reason in row["exclusion_reason"]
    assert row["realized_gain"] is None
    assert row["net_gain"] is None


def test_omitted_documents_and_duplicate_ids_do_not_prove_returned_document(complete_evidence):
    documents = complete_evidence.pop("documents")
    missing = derive_g_candidates(**complete_evidence)[0]
    assert missing["label_status"] == "ineligible"
    assert "document_missing" in missing["exclusion_reason"]
    assert missing["eligibility_checks"]["document_capture_proven"] is False
    assert missing["eligibility_checks"]["document_availability_proven"] is False
    complete_evidence["documents"] = [documents[0], deepcopy(documents[0])]
    ambiguous = derive_g_candidates(**complete_evidence)[0]
    assert "document_ambiguous" in ambiguous["exclusion_reason"]
    assert ambiguous["label_status"] == "ineligible"


def test_failed_action_without_returned_document_does_not_require_fabricated_document(complete_evidence):
    complete_evidence.pop("documents")
    complete_evidence["attempts"][0].update(status="failed", returned_document_id=None)
    row = derive_g_candidates(**complete_evidence)[0]
    assert row["label_status"] == "eligible", row
    assert row["eligibility_checks"]["document_evidence_required"] is False
    assert row["eligibility_checks"]["document_capture_proven"] is None
    assert row["eligibility_checks"]["document_availability_proven"] is None
    assert "document" not in row["raw_digest_checks"]


def test_document_gate_preserves_existing_outcome_and_historical_exclusions(complete_evidence):
    complete_evidence["fixtures"][0]["information_mode"] = "historical_reconstruction"
    complete_evidence["results"] = []
    row = derive_g_candidates(**complete_evidence)[0]
    assert row["label_status"] == "ineligible"
    assert {"historical_reconstruction", "outcome_missing"} <= set(row["exclusion_reason"])
    assert row["eligibility_checks"]["document_capture_proven"] is True
    assert row["realized_gain"] is None
