"""Offline checks of the bounded round8 real capture, never a policy-gain test."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from football_agent.audit import digest
from football_spider.normalization.sources import stamp
from football_spider.storage.raw_archive import RawArchive, json_bytes
from .acquisition_journal import AcquisitionDecisionJournal
from .competition_context import _time
from .contracts import Fixture
from .evidence_registry import CompetitionEvidenceRegistry


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_capture(root: Path) -> dict:
    """Reconcile committed intents, raw returns and the separately anchored case.

    This deliberately validates one frozen case shape. It neither discovers new
    matches nor treats a provider's fixture description as independent official
    confirmation of competition rules or tactical state.
    """
    root = Path(root)
    run_id = (root / "run-id.txt").read_text(encoding="utf-8").strip()
    archive = RawArchive(root / "archive")
    journal = AcquisitionDecisionJournal(root / "journal", archive)
    run = journal.read(run_id)
    observations = archive.observations()
    by_id = {obs.id: obs for obs in observations}
    _require(len(observations) == len(by_id) == len(run["receipts"]), "Raw/receipt count mismatch")
    _require(run["state"] == "STOPPED" and run["pending_request_id"] is None, "Capture has unresolved requests")
    _require(run["budget"]["reserved_requests"] == 0, "Unsettled request reservation")
    _require(sum(receipt["charged_requests"] for receipt in run["receipts"].values())
             == run["budget"]["charged_requests"] == len(observations) == 4, "Actual request charge mismatch")
    _require(run["budget"]["provider_money"] is None, "Unmeasured money must stay unknown")
    fixture = Fixture.model_validate(run["fixture"])
    acquired = []
    known_context = []
    intent_events = {}
    rows = []
    for event in run["events"]:
        if event["type"] == "context_published":
            known_context = event["snapshot"]["current_evidence_ids"]
        if event["type"] == "acquisition_intent":
            intent = event["data"]
            intent_events[intent["request_id"]] = event
            _require(intent["known_observation_ids"] == acquired, "Intent saw future/unpurchased observations")
            _require(intent["current_evidence_ids"] == known_context, "Intent saw future/unpurchased context")
            _require(intent["policy_kind"] == "manual_predeclared" and intent["selection_probability"] is None,
                     "Real capture is manual, with unknown policy propensity")
            assessment = intent["assessment"]
            _require(assessment == {"status": "NOT_CONFIGURED", "prediction": None,
                                    "candidate_expected_gains": None, "model_versions": []},
                     "Real capture must not invent model outputs")
        if event["type"] != "acquisition_receipt":
            continue
        receipt = event["data"]
        request_id = receipt["request_id"]
        intent_event = intent_events[request_id]
        intent = run["intents"][request_id]
        observation = by_id[receipt["observation_id"]]
        observation, body = journal._verify_observation(observation)
        parameters = journal.store.get_blob(run_id, intent["input_ref"])
        _require(observation.endpoint == parameters["endpoint"] and observation.params == parameters.get("params", {}),
                 "Raw endpoint or parameters differ from intent")
        _require(observation.context["journal_run_id"] == run_id
                 and observation.context["journal_request_id"] == request_id, "Raw receipt identity mismatch")
        requested = _time(observation.context["requested_at"])
        _require(_time(intent["decision_at"]) <= _time(intent_event["at"]) <= requested <= observation.observed_at
                 <= _time(receipt["receipt_at"]) <= _time(event["at"]) < fixture.kickoff,
                 "Decision/request/return/commit order or kickoff gate failed")
        _require(receipt["return_sha256"] == observation.content_hash and receipt["return_bytes"] == len(body),
                 "Receipt return checksum mismatch")
        payload = json.loads(body)
        processing = run["processing"][request_id]
        semantic = "NO_DATA" if payload.get("data") is None else "DATA_AVAILABLE"
        _require(processing["status"] == semantic and processing["extraction_sha256"]
                 == digest({"provider_code": payload.get("code"), "data_is_null": payload.get("data") is None}),
                 "Semantic availability differs from archived JSON")
        if receipt["status"] == "RETURNED":
            acquired.append(observation.id)
        rows.append({"request_id": request_id, "action_id": intent["action_id"],
                     "decision_at": intent["decision_at"], "intent_committed_at": intent_event["at"],
                     "requested_at": observation.context["requested_at"], "observed_at": receipt["observed_at"],
                     "receipt_at": receipt["receipt_at"], "observation_id": observation.id,
                     "endpoint": observation.endpoint, "params": observation.params,
                     "raw_payload_path": observation.payload_path, "raw_sha256": observation.content_hash,
                     "http_status": observation.http_status, "semantic_status": semantic,
                     "charged_requests": receipt["charged_requests"], "provider_money": None,
                     "prediction": None, "candidate_expected_gains": None,
                     "policy_kind": intent["policy_kind"], "policy_version": intent["policy_version"],
                     "selection_probability": None})
    _require(acquired == run["acquired_observation_ids"], "Final acquired observations differ from replay")
    case = json.loads((root / "fixture.json").read_bytes())
    _require(Fixture.model_validate(case["fixture"]) == fixture, "Exported fixture differs from journal")
    source = run["fixture_source"]
    _require(source["observation_id"] == case["source_observation_id"] and source["json_path"] == case["source_json_path"],
             "Fixture source anchor differs from journal")
    match = re.fullmatch(r"\$\.data\[(\d+)\]\.list\[(\d+)\]", source["json_path"])
    _require(match is not None, "Unexpected source-native fixture JSON path")
    discovered = json.loads(archive.read_body(by_id[source["observation_id"]]))["data"][int(match[1])]["list"][int(match[2])]
    _require(discovered == case["source_row"], "Fixture export differs from anchored current list")
    detail_obs = next(obs for obs in observations if obs.context["journal_request_id"] == "round8-nations-league-detail-1")
    detail = json.loads(archive.read_body(detail_obs))["data"]
    for native, projected, prefix in (("matchId", fixture.match_id, "spftll:match:"),
                                      ("homeId", fixture.home_team_id, "spftll:team:"),
                                      ("awayId", fixture.away_team_id, "spftll:team:")):
        _require(discovered[native] == detail[native] and projected == prefix + str(detail[native]),
                 "Source-native match/team identity mismatch")
    _require(stamp(discovered["matchTime"], "Asia/Shanghai") == stamp(detail["matchTime"], "Asia/Shanghai")
             == fixture.kickoff, "Kickoff/timezone conversion mismatch")
    _require(detail["footMatchState"] == 0 and detail["finishedTitle"] == "未开", "Archived target is not prematch")
    _require(fixture.home_team_name == discovered["homeChs"] and fixture.away_team_name == discovered["awayChs"],
             "Source team names mismatch")
    _require(fixture.competition_id == "spftll:competition-name:" + hashlib.sha256(discovered["leagueChs"].encode()).hexdigest(),
             "Initial competition name identity mismatch")
    context_export = json.loads((root / "context-readiness.json").read_bytes())
    evidence_id = context_export["registration"]["evidence_id"]
    registry = CompetitionEvidenceRegistry(root / "context-registry", archive)
    registration = registry.read(evidence_id)
    _require(registration.raw_observation.id == detail_obs.id and registration.fixture == fixture, "Registry raw anchor mismatch")
    _require(registration.claim is None and registration.source_published_at is None, "Missing context/publication fields were invented")
    cutoff = context_export["context_projection"]["context"]["prediction_cutoff"]
    projection = registry.project(fixture=fixture, prediction_cutoff=cutoff, acquired_evidence_ids=[evidence_id])
    _require(projection.model_dump(mode="json") == context_export["context_projection"], "Context projection differs on replay")
    _require(projection.exclusions == {evidence_id: "no_structured_context_claim"}
             and projection.context.readiness == "no_context_observed", "Missing context must remain unknown")
    _require(detail_obs.observed_at <= registration.operational_available_at <= _time(cutoff) < fixture.kickoff,
             "Operational registry availability failed")
    return {"schema": "round8-prospective-verification-v1", "verified_at": datetime.now(timezone.utc).isoformat(),
            "status": "VERIFIED_PROSPECTIVE_CAPTURE_NOT_POLICY_GAIN_STUDY", "run_id": run_id,
            "journal_events_verified": len(run["events"]), "raw_observations_verified": len(observations),
            "real_http_attempts": len(rows), "new_http_attempts_from_verification": 0,
            "discovery_or_unconfirmed_fixture_actions": 3, "confirmed_target_specific_actions": 1,
            "request_budget": run["budget"], "fixture": fixture.model_dump(mode="json"),
            "competition_name": discovered["leagueChs"], "source_native_competition_id": detail["leagueId"],
            "competition_identity_note": "source-native league ID discovered later; initial name-hash fixture remains immutable",
            "source_scope": case["scope"], "timezone_basis": case["timezone_basis"],
            "independent_official_fixture_confirmation": False,
            "current_prematch_international_fixture_records": 1, "confirmed_future_Asian_Games_or_German_Cup_cases": 0,
            "structured_context_claims": 0, "missing_context_fields": ["stage", "qualification", "rank_objective", "rotation", "lineup", "tactical_intent"],
            "source_published_at": None, "outcome_label": None, "model_status": "NOT_CONFIGURED",
            "prediction": None, "candidate_expected_gains": None, "selection_probability": None,
            "real_AI_policy_gain_samples": 0, "counterfactual_expected_gain_ground_truth": None,
            "stop_reason": run["stop_reason"], "requests": rows,
            "context_evidence_id": evidence_id, "context_exclusions": projection.exclusions,
            "tests_are_constructed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify_capture(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(json_bytes(report))
    print(json.dumps({key: report[key] for key in ("status", "real_http_attempts", "journal_events_verified", "real_AI_policy_gain_samples")}))


if __name__ == "__main__":
    main()
