from __future__ import annotations

import copy

import pytest

from football_agent.round135_provenance_gate import (
    VERSION,
    ProvenanceGateError,
    build_provenance_gate,
    validate_provenance_gate,
)


CUTOFF = "2026-10-01T10:00:00Z"


def source(source_id: str = "s", *, digest: str = "d", available: str = "09:01",
           observed: str = "09:00", valid_until: str = "11:00", **extra):
    row = {
        "source_id": source_id,
        "observed_at": f"2026-10-01T{observed}:00Z" if observed else None,
        "available_at": f"2026-10-01T{available}:00Z" if available else None,
        "valid_until": f"2026-10-01T{valid_until}:00Z" if valid_until else None,
        "freshness": "fresh",
        "availability": "available",
        "digest": digest,
    }
    row.update(extra)
    return row


def episode():
    return {"prediction_cutoff": CUTOFF, "pre_action": {"candidate_ids": ["c"]}}


def run(candidates, observations):
    return build_provenance_gate(episode(), candidates=candidates,
                                 observations=observations)


def test_fresh_evidence_is_eligible_and_has_no_numeric_decision_fields():
    report = run([{"candidate_id": "c", "kind": "lineup", "source_ids": ["s"]}], [source()])
    candidate = report["candidates"][0]
    assert report["schema_version"] == VERSION
    assert candidate["eligibility"] == "eligible"
    assert candidate["source_states"] == {"s": "available"}
    assert "probability" not in str(report).casefold()
    assert "evaluator" not in str(report).casefold()
    assert validate_provenance_gate(report)["valid"] is True


def test_future_insertion_is_filtered_before_manifest_and_does_not_change_actor_state():
    candidate = [{"candidate_id": "c", "source_ids": ["s"]}]
    visible = [source()]
    future = source("s", digest="new", observed="12:00", available="12:01")
    first = run(candidate, visible)
    second = run(candidate, visible + [future])
    assert first["manifest_digest"] == second["manifest_digest"]
    assert first["gate_digest"] == second["gate_digest"]
    assert first["candidates"] == second["candidates"]
    assert second["audit"]["future_source_ids"] == ["s"]


def test_future_only_and_missing_source_are_same_actor_state_but_audit_is_separate():
    candidate = [{"candidate_id": "c", "source_ids": ["future"]}]
    missing = run(candidate, [])
    future = run(candidate, [source("future", observed="12:00", available="12:01")])
    assert missing["candidates"] == future["candidates"]
    assert missing["manifest_digest"] == future["manifest_digest"]
    assert future["audit"]["future_source_ids"] == ["future"]


def test_default_clocks_and_quality_remain_unknown():
    row = {"source_id": "s", "digest": "d", "observed_at": "09:00"}
    row["observed_at"] = "2026-10-01T09:00:00Z"
    report = run([{"candidate_id": "c", "source_ids": ["s"]}], [row])
    assert report["sources"]["s"]["state"] == "unknown"
    assert report["candidates"][0]["eligibility"] == "defer_unknown"


def test_ingested_at_after_cutoff_is_future_even_without_available_at():
    row = source("s")
    row.pop("available_at")
    row["ingested_at"] = "2026-10-01T12:00:00Z"
    report = run([{"candidate_id": "c", "source_ids": ["s"]}], [row])
    assert report["candidates"][0]["source_states"] == {"s": "unknown"}
    assert report["audit"]["future_source_ids"] == ["s"]


def test_same_source_revision_is_latest_visible_not_conflict():
    old = source("s", digest="old", available="08:01", observed="08:00", valid_until="09:00")
    new = source("s", digest="new", available="09:01", observed="09:00")
    report = run([{"candidate_id": "c", "source_ids": ["s"]}], [old, new])
    assert report["sources"]["s"]["state"] == "available"
    assert report["sources"]["s"]["conflict"] is False
    assert report["sources"]["s"]["observation_count"] == 1


def test_different_providers_without_claim_marker_are_not_conflict():
    left = source("provider-a", digest="a")
    right = source("provider-b", digest="b")
    report = run([{"candidate_id": "c", "source_ids": ["provider-a", "provider-b"],
                   "aggregation_mode": "all_required"}], [left, right])
    assert report["sources"]["provider-a"]["state"] == "available"
    assert report["sources"]["provider-b"]["state"] == "available"
    assert report["candidates"][0]["eligibility"] == "eligible"


def test_explicit_same_claim_conflict_is_deferred():
    left = source("provider-a", digest="a", claim_key="lineup-1")
    right = source("provider-b", digest="b", claim_key="lineup-1")
    report = run([{"candidate_id": "c", "source_ids": ["provider-a", "provider-b"]}], [left, right])
    assert report["sources"]["provider-a"]["state"] == "conflict"
    assert report["candidates"][0]["eligibility"] == "defer_unknown"


def test_identical_duplicate_is_warning_under_all_required():
    first = source("s", digest="same", observation_id="a")
    second = source("s", digest="same", observation_id="b")
    report = run([{"candidate_id": "c", "source_ids": ["s"]}], [first, second])
    assert report["sources"]["s"]["state"] == "duplicate"
    assert report["candidates"][0]["eligibility"] == "eligible_with_warnings"


def test_candidate_any_usable_override_and_action_are_separate():
    rows = [source("fresh"), source("stale", valid_until="10:00")]
    candidate = [{"candidate_id": "c", "source_ids": ["stale", "fresh"],
                  "aggregation_mode": "any_usable", "action_availability": "available"}]
    report = run(candidate, rows)
    state = report["candidates"][0]
    assert state["aggregation_mode"] == "any_usable"
    assert state["eligibility"] == "eligible_with_warnings"
    assert state["action_eligibility"] == "executable"
    assert "refresh_source" in state["next_actions"]


def test_missing_evidence_does_not_disable_query_action():
    candidate = [{"candidate_id": "c", "source_ids": ["injury"],
                  "action_availability": True}]
    state = run(candidate, json_rows := [])["candidates"][0]
    assert state["eligibility"] == "defer_unknown"
    assert state["action_eligibility"] == "executable"
    assert "query_source" in state["next_actions"]


def test_private_candidate_fields_are_not_projected_and_digest_is_deterministic():
    candidate = [{"candidate_id": "c", "source_ids": ["s"], "kind": "injury",
                  "description": {"sidecar": {"result": "secret"}},
                  "private_note": "secret", "declared_cost": {"requests": 1}}]
    rows = [source()]
    first = run(candidate, rows)
    shuffled = run(copy.deepcopy(candidate), list(reversed(rows)))
    assert first["gate_digest"] == shuffled["gate_digest"]
    assert "description" not in first["candidates"][0]
    assert "private_note" not in first["candidates"][0]
    assert "secret" not in str(first)


def test_validator_detects_tampering_and_invalid_mode():
    report = run([{"candidate_id": "c", "source_ids": ["s"]}], [source()])
    tampered = dict(report)
    tampered["candidates"] = [dict(report["candidates"][0], eligibility="defer_unknown")]
    with pytest.raises(ProvenanceGateError, match="DIGEST_MISMATCH"):
        validate_provenance_gate(tampered)
    with pytest.raises(ProvenanceGateError, match="aggregation_mode_UNSUPPORTED"):
        run([{"candidate_id": "c", "source_ids": ["s"], "aggregation_mode": "maybe"}], [source()])
