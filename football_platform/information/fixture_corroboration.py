"""Explicit fixture-identity comparison and offline corroboration audit.

Comparison describes already extracted inputs, not semantic source verification.
An unavailable or documentary-only response never becomes machine confirmation.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from football_spider.storage.raw_archive import RawArchive
from .acquisition_journal import AcquisitionDecisionJournal
from .competition_context import _identifier, _time
from .contracts import Fixture


def compare_assertion(fixture: Fixture | dict, *, source_event_id: str | None,
                      explicitly_mapped_match_id: str | None, fields: dict,
                      evidence_basis: str, available_at=None, cutoff=None) -> dict:
    """Compare exact canonical identities and a timezone-aware kickoff.

    A caller must establish a source-native event mapping before comparing.
    This does not infer a mapping from team strings or dates. CONFIRMED means
    all supplied *structured, acquired* identity inputs agree; upstream source
    extraction must be audited separately. Document reading remains separate.
    """
    fixture = Fixture.model_validate(fixture)
    allowed = {"home_team_id", "away_team_id", "competition_id", "kickoff"}
    if set(fields) - allowed or evidence_basis not in {"acquired_structured", "documentary_only", "unavailable"}:
        raise ValueError("Unexpected fixture fields or evidence basis")
    result = {"status": "UNCONFIRMED", "source_event_id": source_event_id,
              "evidence_basis": evidence_basis, "matched_fields": [], "disputed_fields": [],
              "unknown_fields": sorted(allowed), "reason": None,
              "semantics": "input_comparison_not_semantic_source_verification"}
    if (source_event_id is None or not _identifier(source_event_id)
            or explicitly_mapped_match_id != fixture.match_id):
        result["reason"] = "explicit_source_event_identity_mapping_required"
        return result
    if evidence_basis != "acquired_structured":
        result["reason"] = "official_raw_unavailable" if evidence_basis == "unavailable" else "documentary_support_only"
        return result
    if available_at is None or cutoff is None:
        result["reason"] = "actual_availability_and_cutoff_required"
        return result
    cutoff, available_at = _time(cutoff), _time(available_at)
    if not available_at <= cutoff < fixture.kickoff:
        result["reason"] = "outside_operational_prematch_window"
        return result
    expected = fixture.model_dump()
    result["unknown_fields"] = []
    for key in sorted(allowed):
        value = fields.get(key)
        if value is None:
            result["unknown_fields"].append(key)
            continue
        if key == "kickoff":
            value = _time(value)
        elif not isinstance(value, str) or not _identifier(value):
            raise ValueError("Canonical identities must be inert identifiers, not fuzzy names")
        result["matched_fields" if value == expected[key] else "disputed_fields"].append(key)
    if result["disputed_fields"]:
        result.update(status="DISPUTED", reason="explicitly_mapped_identity_or_time_conflict")
    elif result["unknown_fields"]:
        result["reason"] = "missing_structured_identity_fields"
    else:
        result.update(status="CONFIRMED", reason="all_explicit_identity_and_time_fields_agree")
    return result


def audit_failed_official_capture(root: Path) -> dict:
    """Reconcile the bounded round9 failed raw capture without reading claims in error pages."""
    root = Path(root)
    archive = RawArchive(root / "archive")
    journal = AcquisitionDecisionJournal(root / "journal", archive)
    run_id = (root / "run-id.txt").read_text(encoding="utf-8").strip()
    run = journal.read(run_id)
    fixture = Fixture.model_validate(run["fixture"])
    observations = archive.observations()
    by_id = {observation.id: observation for observation in observations}
    if len(observations) != len(run["receipts"]) or run["state"] != "STOPPED" or run["pending_request_id"]:
        raise ValueError("Unreconciled official capture")
    if run["current_evidence_ids"] or run["acquired_observation_ids"]:
        raise ValueError("Failed official capture unexpectedly published evidence")
    attempts = []
    for request_id, intent in run["intents"].items():
        receipt = run["receipts"][request_id]
        observation, body = journal._verify_observation(by_id[receipt["observation_id"]])
        parameters = journal.store.get_blob(run_id, intent["input_ref"])
        event = next(event for event in run["events"] if event["type"] == "acquisition_intent"
                     and event["data"]["request_id"] == request_id)
        receipt_event = next(event for event in run["events"] if event["type"] == "acquisition_receipt"
                             and event["data"]["request_id"] == request_id)
        if (observation.endpoint != parameters["endpoint"] or observation.params != parameters["params"]
                or observation.source != parameters["source"]
                or observation.context["journal_run_id"] != run_id
                or observation.context["journal_request_id"] != request_id):
            raise ValueError("Raw response differs from predeclared official request")
        if not (_time(intent["decision_at"]) <= _time(event["at"])
                <= _time(observation.context["requested_at"]) <= observation.observed_at
                <= _time(receipt["receipt_at"]) <= _time(receipt_event["at"]) < fixture.kickoff):
            raise ValueError("Official capture violates intent/request/return/kickoff order")
        if (receipt["return_sha256"] != observation.content_hash or receipt["return_bytes"] != len(body)
                or receipt["status"] != "FAILED" or (observation.http_status == 200 and not observation.error_class)):
            raise ValueError("This report requires archived failed official responses")
        if (intent["assessment"] != {"status": "NOT_CONFIGURED", "prediction": None,
                                    "candidate_expected_gains": None, "model_versions": []}
                or intent["policy_kind"] != "manual_predeclared" or intent["selection_probability"] is not None
                or intent["current_evidence_ids"] or intent["known_observation_ids"]):
            raise ValueError("Manual failed capture fabricated forecasts, propensity or acquired evidence")
        attempts.append({"request_id": request_id, "endpoint": observation.endpoint,
                         "source": observation.source, "intent_committed_at": event["at"],
                         "requested_at": observation.context["requested_at"], "observed_at": observation.observed_at.isoformat(),
                         "observation_id": observation.id, "raw_sha256": observation.content_hash,
                         "raw_bytes": len(body), "http_status": observation.http_status,
                         "error_class": observation.error_class, "charged_requests": receipt["charged_requests"],
                         "receipt_at": receipt["receipt_at"], "status": "UNCONFIRMED"})
    charged = sum(item["charged_requests"] for item in attempts)
    if charged != run["budget"]["charged_requests"] or run["budget"]["reserved_requests"] != 0:
        raise ValueError("Failed attempts lost their real acquisition cost")
    note = json.loads((root / "documentary-review.json").read_bytes())
    if note["evidence_class"] != "human_review_of_web_tool_output_not_provider_raw" or note["operational_evidence_ids"]:
        raise ValueError("Documentary note must stay outside operational source evidence")
    return {"schema": "round9-fixture-corroboration-v1", "generated_at": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id, "fixture": fixture.model_dump(mode="json"),
            "machine_corroboration_status": "UNCONFIRMED", "reason": "official_raw_unavailable",
            "documentary_review_status": note["conclusion"], "documentary_review_sources": note["sources"],
            "journal_events_verified": len(run["events"]), "raw_observations_verified": len(observations),
            "real_http_attempts": len(attempts), "charged_requests": charged, "reserved_requests": 0,
            "successful_official_raw_responses": 0, "new_network_requests_during_audit": 0,
            "structured_context_claims_added": 0, "verified_stage_facts_added": 0,
            "qualification": "unknown", "rank_objective": "unknown", "rotation": "unknown",
            "lineup": "unknown", "tactical_intent": "unknown", "model_status": "NOT_CONFIGURED",
            "prediction": None, "candidate_expected_gains": None, "selection_probability": None,
            "real_AI_policy_gain_samples": 0, "source_published_at": None,
            "attempts": sorted(attempts, key=lambda item: item["intent_committed_at"]),
            "stop_reason": run["stop_reason"],
            "next_needed_evidence": "one successful timestamped official match response with native identity, timezone and stage anchors"}
