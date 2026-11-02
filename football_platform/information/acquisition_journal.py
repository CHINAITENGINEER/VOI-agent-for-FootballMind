"""Small prospective-context adapter over the existing Agent RunStore.

An intent and budget reservation are durable before a caller executes a tool.
Responses are reconciled against committed RawArchive metadata; a replay never
implicitly repeats a network request or turns an absent model into a forecast.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import math
import re
from pathlib import Path
from uuid import uuid4
from collections.abc import Mapping

from football_agent.audit import RunStore, digest, redact
from football_spider.models.contracts import Observation
from football_spider.storage.raw_archive import RawArchive
from .competition_context import _identifier, _time
from .contracts import Fixture
from .evidence_registry import CompetitionEvidenceRegistry


class AcquisitionDecisionJournal:
    def __init__(self, root: Path, archive: RawArchive, *, clock=lambda: datetime.now(timezone.utc)):
        self.store, self.archive, self.clock = RunStore(root), archive, clock

    def _now(self):
        return _time(self.clock())

    def _append(self, run, kind, data, *, at=None):
        timestamp = _time(at) if at is not None else self._now()
        if run["events"] and timestamp < _time(run["events"][-1]["at"]):
            raise ValueError("Journal clock moved backwards")
        snapshot = deepcopy({key: value for key, value in run.items() if key not in {"events", "event_blob_refs"}})
        event = redact({"seq": len(run["events"]) + 1, "at": timestamp.isoformat(), "type": kind,
                        "data": data, "snapshot": snapshot,
                        "previous_hash": run["events"][-1]["event_hash"] if run["events"] else None})
        event["event_hash"] = digest(event)
        # The existing mutable run.json is a materialized head; event blobs are
        # content-addressed immutable history, verified again on every load.
        run["event_blob_refs"].append(self.store.put_blob(run["run_id"], event))
        run["events"].append(event)
        self.store.save(run)

    def _load(self, run_id):
        run = self.store.load(run_id)
        if run.get("schema") not in {"prospective-competition-journal-v1", "prospective-competition-journal-v2"}:
            raise ValueError("Not a prospective competition journal")
        if len(run["event_blob_refs"]) != len(run["events"]):
            raise ValueError("Journal event reference count mismatch")
        for event, ref in zip(run["events"], run["event_blob_refs"]):
            if self.store.get_blob(run_id, ref) != event:
                raise ValueError("Journal event differs from its immutable blob")
        return run

    def start(self, *, request_limit: int = 4, fixture: Fixture | dict | None = None,
              policy_arm: str | None = None, policy_version: str | None = None,
              random_seed: int | None = None, propensity_logging: bool | None = None,
              initial_assessment: dict | None = None) -> dict:
        if type(request_limit) is not int or request_limit < 0:
            raise ValueError("Request budget must be a nonnegative integer")
        fixture = Fixture.model_validate(fixture) if fixture is not None else None
        timestamp = self._now()
        if fixture is not None and timestamp >= fixture.kickoff:
            raise ValueError("Prospective fixture must still be in the future")
        allowed_arms = {"voi", "fixed_order", "random_explore", "full_probe"}
        if policy_arm is not None and policy_arm not in allowed_arms:
            raise ValueError("Unknown policy arm")
        if policy_version is not None and not _identifier(policy_version):
            raise ValueError("Policy version must be an inert identifier")
        if random_seed is not None and type(random_seed) is not int:
            raise ValueError("Random seed must be an integer or null")
        if propensity_logging is not None and type(propensity_logging) is not bool:
            raise ValueError("propensity_logging must be a boolean or null")
        if propensity_logging is None:
            propensity_logging = True if policy_arm == "random_explore" else (False if policy_arm is not None else None)
        if policy_arm == "random_explore":
            if random_seed is None:
                raise ValueError("random_explore requires a random seed")
            if propensity_logging is not True:
                raise ValueError("random_explore requires propensity logging")
        assessment = self._assessment(initial_assessment)
        predictor_only = assessment["status"] == "PREDICTOR_ONLY"
        if predictor_only and (policy_arm not in {"random_explore", "fixed_order", "full_probe"}
                               or policy_version is None):
            raise ValueError("Predictor-only capture requires a versioned nonnumeric collection policy")
        run = {"schema": "prospective-competition-journal-v1", "run_id": str(uuid4()),
               "created_at": timestamp.isoformat(), "state": "READY", "fixture": fixture.model_dump(mode="json") if fixture else None,
               "fixture_source": None, "prediction_cutoff": timestamp.isoformat(), "current_evidence_ids": [],
               "acquired_observation_ids": [], "intents": {}, "receipts": {}, "processing": {}, "pending_request_id": None,
               "budget": {"request_limit": request_limit, "reserved_requests": 0, "charged_requests": 0,
                          "provider_money": None, "money_status": "not_measured"},
               "model_status": "NOT_CONFIGURED", "prediction": None, "candidate_expected_gains": None,
               "model_versions": [], "events": [], "event_blob_refs": [], "stop_reason": None,
               "stopped_at": None, "context_snapshots": [],
               "policy_arm": policy_arm, "arm_assigned_at": timestamp.isoformat() if policy_arm else None,
               "policy_version": policy_version, "random_seed": random_seed,
               "propensity_logging": propensity_logging,
               "scope": "prospective_acquisition_execution; not_policy_gain_evaluation"}
        if initial_assessment is not None:
            run.update(model_status=assessment["status"], prediction=assessment["prediction"],
                       candidate_expected_gains=assessment["candidate_expected_gains"],
                       model_versions=assessment["model_versions"])
        if predictor_only:
            run.update(schema="prospective-competition-journal-v2",
                       assessment_schema="prospective-predictor-only-assessment-v1",
                       frozen_predictor_versions=deepcopy(assessment["model_versions"]))
        with self.store.lock(run["run_id"]):
            self._append(run, "journal_started", {"request_limit": request_limit,
                                                   "policy_arm": policy_arm,
                                                   "arm_assigned_at": run["arm_assigned_at"],
                                                   "policy_version": policy_version,
                                                   "random_seed": random_seed,
                                                   "propensity_logging": propensity_logging}, at=timestamp)
        return deepcopy(run)

    @staticmethod
    def _assessment(assessment):
        assessment = deepcopy(assessment or {"status": "NOT_CONFIGURED", "prediction": None,
                                             "candidate_expected_gains": None, "model_versions": []})
        allowed = {"status", "prediction", "candidate_expected_gains", "model_versions"}
        if set(assessment) != allowed or assessment["status"] not in {"NOT_CONFIGURED", "CONFIGURED", "PREDICTOR_ONLY"}:
            raise ValueError("Explicit model assessment fields are required")
        if assessment["status"] == "NOT_CONFIGURED":
            if assessment["prediction"] is not None or assessment["candidate_expected_gains"] is not None or assessment["model_versions"]:
                raise ValueError("Unconfigured models cannot supply predictions or expected gains")
        else:
            prediction = assessment["prediction"]
            if not isinstance(prediction, dict) or set(prediction) != {"home", "draw", "away"}:
                raise ValueError("Configured prediction requires home/draw/away probabilities")
            if any(type(value) not in {int, float} or not math.isfinite(value) or not 0 <= value <= 1 for value in prediction.values()):
                raise ValueError("Invalid model probabilities")
            if abs(sum(prediction.values()) - 1) > 1e-8 or not assessment["model_versions"]:
                raise ValueError("Prediction normalization and model versions are required")
            if not isinstance(assessment["model_versions"], list) or any(
                    not isinstance(value, str) or not _identifier(value) for value in assessment["model_versions"]):
                raise ValueError("Model versions must be inert identifiers")
            values = assessment["candidate_expected_gains"]
            if assessment["status"] == "PREDICTOR_ONLY":
                if values is not None:
                    raise ValueError("Predictor-only assessment cannot supply expected gains")
            elif not isinstance(values, dict) or any(not _identifier(key) or type(value) not in {int, float}
                                                  or not math.isfinite(value) for key, value in values.items()):
                raise ValueError("Expected gains must be finite and keyed by inert action IDs")
        return assessment

    @staticmethod
    def _candidate_metadata(candidates, candidate_metadata):
        if candidate_metadata is None:
            return None
        if not isinstance(candidate_metadata, list) or len(candidate_metadata) != len(candidates):
            raise ValueError("candidate_metadata must be a list matching the candidate menu")
        seen = set()
        normalized = []
        for index, value in enumerate(candidate_metadata):
            if isinstance(value, str):
                value = {"candidate_id": value}
            if not isinstance(value, Mapping):
                raise ValueError("candidate_metadata entries must be objects")
            item = deepcopy(dict(value))
            candidate_id = item.get("candidate_id", item.get("action_id", item.get("id")))
            if not isinstance(candidate_id, str) or not _identifier(candidate_id):
                raise ValueError("candidate_metadata entries require an inert candidate_id")
            if candidate_id != candidates[index] or candidate_id in seen:
                raise ValueError("candidate_metadata must preserve candidate order and ids")
            seen.add(candidate_id)
            item.setdefault("candidate_id", candidate_id)
            item.setdefault("rank", index)
            normalized.append(item)
        return normalized

    def begin(self, run_id, *, request_id: str, action_id: str, input_parameters: dict,
              candidates: list[str], reserve_requests: int = 1, assessment: dict | None = None,
              selection_basis: str = "manual_predeclared", policy_version: str | None = None,
              selection_probability: float | None = None,
              candidate_metadata: list[dict] | None = None) -> dict:
        if not _identifier(request_id) or not _identifier(action_id) or any(not _identifier(item) for item in candidates):
            raise ValueError("Request/action identities must be inert")
        if action_id not in candidates or len(set(candidates)) != len(candidates):
            raise ValueError("Chosen action must belong to the unique public candidate set")
        if type(reserve_requests) is not int or reserve_requests < 1:
            raise ValueError("External acquisition must reserve at least one request")
        allowed_bases = {"manual_predeclared", "configured_numeric_policy", "uniform_exploration",
                         "unknown_value_exploration", "fixed_order", "full_probe", "not_configured"}
        if selection_basis not in allowed_bases:
            raise ValueError("Unknown selection basis")
        candidate_metadata = self._candidate_metadata(candidates, candidate_metadata)
        assessment = self._assessment(assessment)
        if selection_basis == "configured_numeric_policy" and assessment["status"] != "CONFIGURED":
            raise ValueError("A numeric policy requires an actual configured model assessment")
        if policy_version is not None and not _identifier(policy_version):
            raise ValueError("Policy version must be an inert identifier")
        if selection_probability is not None:
            if selection_basis in {"manual_predeclared", "not_configured"}:
                raise ValueError("Manual collection records no inferred selection probability")
            if type(selection_probability) not in {int, float} or not math.isfinite(selection_probability) or not 0 < selection_probability <= 1:
                raise ValueError("Actual selection probability must lie in (0, 1]")
        if selection_basis == "uniform_exploration" and selection_probability is None:
            raise ValueError("Uniform exploration requires an explicit design probability")
        if assessment["status"] == "CONFIGURED" and set(assessment["candidate_expected_gains"]) != set(candidates):
            raise ValueError("Every public candidate requires its recorded expected gain")
        signature_base = {"action_id": action_id, "input_parameters": input_parameters, "candidates": candidates,
                          "reserve_requests": reserve_requests, "assessment": assessment, "selection_basis": selection_basis,
                          "policy_version": policy_version, "selection_probability": selection_probability}
        signature = digest({**signature_base, "candidate_metadata": candidate_metadata})
        legacy_signature = digest(signature_base)
        with self.store.lock(run_id):
            run = self._load(run_id)
            effective_policy_version = policy_version if policy_version is not None else run.get("policy_version")
            if run.get("assessment_schema") == "prospective-predictor-only-assessment-v1":
                if assessment["status"] != "PREDICTOR_ONLY":
                    raise ValueError("Predictor-only runs cannot change assessment mode")
                if assessment["model_versions"] != run["frozen_predictor_versions"]:
                    raise ValueError("Predictor-only runs cannot change frozen model versions")
                if effective_policy_version != run["policy_version"]:
                    raise ValueError("Predictor-only runs cannot change frozen policy version")
                if selection_basis not in {"uniform_exploration", "unknown_value_exploration", "fixed_order", "full_probe"}:
                    raise ValueError("Predictor-only acquisition requires nonnumeric exploration")
                expected_basis = {"random_explore": "uniform_exploration", "fixed_order": "fixed_order",
                                  "full_probe": "full_probe"}[run["policy_arm"]]
                if selection_basis != expected_basis:
                    raise ValueError("Predictor-only selection basis must match the frozen collection arm")
            elif assessment["status"] == "PREDICTOR_ONLY":
                raise ValueError("Predictor-only assessment must be opted in at journal start")
            if selection_basis == "configured_numeric_policy" and effective_policy_version is None:
                raise ValueError("A configured policy requires its version")
            if effective_policy_version is not None and not _identifier(effective_policy_version):
                raise ValueError("Policy version must be an inert identifier")
            if effective_policy_version != policy_version:
                signature_base["policy_version"] = effective_policy_version
                signature = digest({**signature_base, "candidate_metadata": candidate_metadata})
                legacy_signature = digest(signature_base)
            if request_id in run["intents"]:
                intent = run["intents"][request_id]
                compatible_signatures = {signature}
                if candidate_metadata is None and "candidate_metadata" not in intent:
                    compatible_signatures.add(legacy_signature)
                if intent["signature"] not in compatible_signatures:
                    raise ValueError("Request identity reused with different intent")
                return {"execute": False, "intent": deepcopy(intent), "receipt": deepcopy(run["receipts"].get(request_id)),
                        "recovery_status": "RECEIPT_REPLAY" if request_id in run["receipts"] else "NEEDS_RECONCILIATION"}
            if run["state"] != "READY" or run["pending_request_id"]:
                raise ValueError("Journal must be ready with no unresolved request")
            now = self._now()
            if run["fixture"] and now >= _time(run["fixture"]["kickoff"]):
                raise ValueError("Acquisition decision must precede kickoff")
            budget = run["budget"]
            if budget["charged_requests"] + budget["reserved_requests"] + reserve_requests > budget["request_limit"]:
                raise ValueError("Request budget exhausted")
            input_ref = self.store.put_blob(run_id, input_parameters)
            intent = {"request_id": request_id, "action_id": action_id, "signature": signature, "input_ref": input_ref,
                      "decision_at": now.isoformat(), "current_evidence_ids": list(run["current_evidence_ids"]),
                      "known_observation_ids": list(run["acquired_observation_ids"]), "candidates": list(candidates),
                      "reserve_requests": reserve_requests, "assessment": assessment, "selection_basis": selection_basis,
                      "candidate_metadata": deepcopy(candidate_metadata),
                      "policy_arm": run.get("policy_arm")}
            intent.update(policy_kind=selection_basis, policy_version=effective_policy_version,
                          selection_probability=selection_probability,
                          selection_probability_basis=("not_recorded" if selection_probability is None
                                                       else ("declared_uniform_design_probability"
                                                             if selection_basis == "uniform_exploration"
                                                             else "caller_supplied_actual_policy_probability")))
            run["intents"][request_id] = intent
            run["pending_request_id"], run["state"] = request_id, "PENDING_RETURN"
            budget["reserved_requests"] += reserve_requests
            run["prediction_cutoff"] = now.isoformat()
            run["model_status"] = assessment["status"]
            for key in ("prediction", "candidate_expected_gains", "model_versions"):
                run[key] = assessment[key]
            self._append(run, "acquisition_intent", {**intent, "budget_after_reservation": deepcopy(budget)})
            return {"execute": True, "intent": deepcopy(intent), "receipt": None, "recovery_status": "FRESH_INTENT"}

    def _verify_observation(self, observation):
        observation = Observation.model_validate(observation)
        body = self.archive.read_body(observation)
        payload = self.archive.root / observation.payload_path
        metadata = payload.with_name(payload.name[:-len(".body.gz")] + ".observation.json")
        if Observation.model_validate_json(metadata.read_bytes()).model_dump() != observation.model_dump():
            raise ValueError("Return metadata differs from committed RawArchive evidence")
        return observation, body

    def complete(self, run_id, *, request_id: str, observation: Observation | dict, charged_requests: int = 1) -> dict:
        if type(charged_requests) is not int or charged_requests < 1:
            raise ValueError("Archived HTTP attempt requires a positive actual request count")
        with self.store.lock(run_id):
            run = self._load(run_id)
            observation = Observation.model_validate(observation)
            if request_id in run["receipts"]:
                observation, _ = self._verify_observation(observation)
                previous = run["receipts"][request_id]
                if (previous["observation_id"] != observation.id or previous["charged_requests"] != charged_requests
                        or previous["return_sha256"] != observation.content_hash):
                    raise ValueError("Receipt identity reused with a different return")
                return deepcopy(previous)
            if run["pending_request_id"] != request_id or request_id not in run["intents"]:
                raise ValueError("A durable pending intent is required before a tool return")
            intent = run["intents"][request_id]
            observation, body = self._verify_observation(observation)
            if observation.context.get("journal_request_id") != request_id or observation.context.get("journal_run_id") != run_id:
                raise ValueError("Raw response lacks the matching journal receipt identity")
            parameters = self.store.get_blob(run_id, intent["input_ref"])
            if (observation.endpoint != parameters.get("endpoint") or observation.params != parameters.get("params", {})
                    or ("source" in parameters and observation.source != parameters["source"])):
                raise ValueError("Raw response endpoint/parameters/source differ from the durable intent")
            now = self._now()
            requested = observation.context.get("requested_at")
            committed_intent = next(event for event in run["events"]
                                    if event["type"] == "acquisition_intent"
                                    and event["data"]["request_id"] == request_id)
            if requested is None or not (_time(intent["decision_at"]) <= _time(committed_intent["at"])
                                         <= _time(requested) <= observation.observed_at <= now):
                raise ValueError("Request and return must occur after intent and before receipt commit")
            success = observation.http_status is not None and 200 <= observation.http_status < 300 and not observation.error_class
            late = run["fixture"] is not None and now >= _time(run["fixture"]["kickoff"])
            status = "RETURNED" if success else "FAILED"
            if late:
                status = "LATE_RETURN"
            receipt = {"request_id": request_id, "action_id": intent["action_id"], "decision_at": intent["decision_at"],
                       "receipt_at": now.isoformat(), "observed_at": observation.observed_at.isoformat(),
                       "observation_id": observation.id, "return_sha256": observation.content_hash, "return_bytes": len(body),
                       "http_status": observation.http_status, "error_class": observation.error_class, "status": status,
                       "charged_requests": charged_requests, "provider_money": None, "money_status": "not_measured"}
            run["receipts"][request_id] = receipt
            budget = run["budget"]
            budget["reserved_requests"] -= intent["reserve_requests"]
            budget["charged_requests"] += charged_requests
            run["pending_request_id"] = None
            run["state"] = "READY"
            if status == "RETURNED":
                run["acquired_observation_ids"].append(observation.id)
            if late or charged_requests > intent["reserve_requests"] or budget["charged_requests"] > budget["request_limit"]:
                run["state"] = "STOPPED"
                run["stop_reason"] = "KICKOFF_REACHED" if late else "ACTUAL_REQUEST_BUDGET_OVERRUN"
                run["stopped_at"] = now.isoformat()
            self._append(run, "acquisition_receipt", receipt)
            return deepcopy(receipt)

    def recover_pending(self, run_id) -> dict:
        with self.store.lock(run_id):
            run = self._load(run_id)
            request_id = run["pending_request_id"]
            if request_id is None:
                return {"status": "NOTHING_PENDING"}
            matches = [obs for obs in self.archive.observations()
                       if obs.context.get("journal_request_id") == request_id and obs.context.get("journal_run_id") == run_id]
            if len(matches) == 1:
                return {"status": "RECOVERED_FROM_RAW_ARCHIVE", "receipt": self.complete(run_id, request_id=request_id, observation=matches[0])}
            return {"status": "NEEDS_RECONCILIATION", "matching_archived_attempts": len(matches),
                    "network_retry_performed": False, "reservation_retained": True}

    def record_processing(self, run_id, *, request_id: str, status: str, extraction_sha256: str,
                          missing_fields: list[str] | None = None):
        """Append semantic return status separately from the immutable HTTP receipt."""
        if status not in {"DATA_AVAILABLE", "NO_DATA", "INVALID_CONTENT", "REQUIRES_REVIEW"}:
            raise ValueError("Unknown response processing status")
        if not re.fullmatch(r"[a-f0-9]{64}", extraction_sha256) or any(not _identifier(field) for field in missing_fields or []):
            raise ValueError("Processing requires an extraction digest and inert field names")
        value = {"request_id": request_id, "status": status, "extraction_sha256": extraction_sha256,
                 "missing_fields": sorted(set(missing_fields or []))}
        with self.store.lock(run_id):
            run = self._load(run_id)
            if request_id not in run["receipts"]:
                raise ValueError("Response processing requires a committed receipt")
            processing = run.setdefault("processing", {})
            if request_id in processing:
                if processing[request_id] != value:
                    raise ValueError("Response processing is append-only; conflicting classification")
                return deepcopy(value)
            processing[request_id] = value
            self._append(run, "response_processed", value)
            return deepcopy(value)

    def bind_fixture(self, run_id, *, fixture: Fixture | dict, observation_id: str, source_json_path: str):
        fixture = Fixture.model_validate(fixture)
        with self.store.lock(run_id):
            run = self._load(run_id)
            if run["fixture"] is not None or run["pending_request_id"] or run["state"] != "READY":
                raise ValueError("Fixture can only be bound once after a completed discovery request")
            if observation_id not in run["acquired_observation_ids"] or self._now() >= fixture.kickoff:
                raise ValueError("Binding requires purchased evidence and a future fixture")
            run["fixture"] = fixture.model_dump(mode="json")
            run["fixture_source"] = {"observation_id": observation_id, "json_path": source_json_path}
            self._append(run, "fixture_bound", deepcopy(run["fixture_source"]))
            return deepcopy(run)

    def publish_evidence(self, run_id, *, request_id: str, evidence_ids: list[str], registry: CompetitionEvidenceRegistry):
        with self.store.lock(run_id):
            run = self._load(run_id)
            receipt = run["receipts"].get(request_id)
            now = self._now()
            if run["fixture"] is None or run["state"] != "READY" or now >= _time(run["fixture"]["kickoff"]):
                raise ValueError("Context publication requires a current prospective fixture")
            if receipt is None or receipt["status"] != "RETURNED":
                raise ValueError("Only a successfully purchased response may publish context")
            processing = run.get("processing", {}).get(request_id)
            if processing is not None and processing["status"] != "DATA_AVAILABLE":
                raise ValueError("Only a response with usable semantic data may publish context")
            for evidence_id in evidence_ids:
                record = registry.read(evidence_id)
                if record.raw_observation.id != receipt["observation_id"]:
                    raise ValueError("Context is not from this purchased response")
            ids = sorted(set(run["current_evidence_ids"]) | set(evidence_ids))
            projection = registry.project(fixture=run["fixture"], prediction_cutoff=now,
                                          acquired_evidence_ids=ids, information_mode="operational")
            if projection.exclusions or not set(evidence_ids) <= set(projection.context.evidence):
                raise ValueError("Context evidence did not pass operational projection gates")
            run["current_evidence_ids"] = ids
            run["prediction_cutoff"] = now.isoformat()
            self._append(run, "context_published", {"request_id": request_id, "evidence_ids": evidence_ids,
                                                    "context_sha256": digest(projection.context.model_dump(mode="json"))})
            return projection

    @staticmethod
    def _prediction(value, *, name):
        if value is None:
            return None
        if not isinstance(value, dict) or set(value) != {"home", "draw", "away"}:
            raise ValueError(f"{name} requires home/draw/away probabilities")
        if any(type(item) not in {int, float} or not math.isfinite(item) or not 0 <= item <= 1
               for item in value.values()) or abs(sum(value.values()) - 1) > 1e-8:
            raise ValueError(f"{name} contains invalid probabilities")
        return deepcopy(value)

    @staticmethod
    def _prediction_processing(value, *, observed_at, forecast_as_of):
        """Validate measured processing clocks without equating them to HTTP."""
        fields = {"schema", "status", "reason", "input_at", "available_at", "effective_at"}
        if not isinstance(value, Mapping) or set(value) != fields or value.get("schema") != "entity-semantic-prediction-processing-v1":
            raise ValueError("Explicit versioned prediction processing fields are required")
        if value["status"] not in {"ready", "not_updated"}:
            raise ValueError("Unknown prediction processing status")
        observed = _time(observed_at)
        input_time = _time(value["input_at"])
        available = _time(value["available_at"]) if value["available_at"] is not None else None
        if input_time > observed or available is not None and available > observed:
            raise ValueError("Prediction processing must precede its recorded handoff")
        if value["status"] == "ready":
            effective = _time(value["effective_at"])
            if (value["reason"] is not None or available is None
                    or effective != max(input_time, available) or effective != _time(forecast_as_of)):
                raise ValueError("Ready processing requires the actual effective forecast time")
        elif (value["effective_at"] is not None or not isinstance(value["reason"], str)
              or not value["reason"].strip()):
            raise ValueError("Unupdated prediction processing requires a reason and no effective forecast")
        return deepcopy(dict(value))

    def record_context(self, run_id, *, snapshot_kind: str, observed_at=None,
                       available_at=None, availability_basis: str | None = None, prediction_cutoff=None,
                       known_information: list[str] | None = None,
                       unknown_information: list[str] | None = None,
                       prediction_before: dict | None = None,
                       prediction_after: dict | None = None,
                       state_delta: dict | None = None,
                       context: dict | None = None,
                       parent_request_id: str | None = None,
                       parent_attempt_id: str | None = None,
                       stop_reason: str | None = None,
                       information_mode: str = "prospective_capture",
                       snapshot_id: str | None = None,
                       model_versions: list[str] | None = None,
                       forecast_as_of=None, acquisition_deadline=None,
                       prediction_processing: dict | None = None,
                       prediction_pipeline_version: str | None = None) -> dict:
        """Append a context snapshot with separate forecast and audit clocks.

        ``captured_at`` is always the journal clock at commit time.  The source
        observation and agent availability clocks remain separate; an omitted
        ``available_at`` is recorded as unknown rather than inferred from the
        capture clock.  Prediction fields are required to stay null while the
        run has no configured model. Versioned f-only callers can retain an
        earlier forecast in an audit-only snapshot after the acquisition
        deadline. This never marks late recording as pre-match availability.
        """
        aliases = {"before": "before_action", "after": "after_action"}
        snapshot_kind = aliases.get(snapshot_kind, snapshot_kind)
        if snapshot_kind not in {"initial", "before_action", "after_action", "final"}:
            raise ValueError("Unknown context snapshot kind")
        if information_mode not in {"prospective_capture", "archived_snapshot", "historical_reconstruction", "operational"}:
            raise ValueError("Unknown information mode")
        if parent_request_id is not None and not _identifier(parent_request_id):
            raise ValueError("parent_request_id must be an inert identifier")
        if parent_attempt_id is not None and not _identifier(parent_attempt_id):
            raise ValueError("parent_attempt_id must be an inert identifier")
        if stop_reason is not None and not _identifier(stop_reason):
            raise ValueError("Stop reason must be an inert identifier")
        if snapshot_id is not None and (not isinstance(snapshot_id, str) or not _identifier(snapshot_id)):
            raise ValueError("snapshot_id must be an inert identifier")
        if available_at is not None and (not isinstance(availability_basis, str) or not availability_basis.strip()):
            raise ValueError("Explicit available_at requires an availability_basis")
        for name, values in (("known_information", known_information), ("unknown_information", unknown_information)):
            if values is not None and (not isinstance(values, list) or any(not isinstance(item, str) for item in values)):
                raise ValueError(f"{name} must be a list of strings")
        observed = _time(observed_at) if observed_at is not None else None
        available = _time(available_at) if available_at is not None else None
        cutoff = _time(prediction_cutoff) if prediction_cutoff is not None else None
        forecast_time = _time(forecast_as_of) if forecast_as_of is not None else None
        deadline = _time(acquisition_deadline) if acquisition_deadline is not None else None
        if (forecast_time is None) != (deadline is None):
            raise ValueError("Forecast audit requires both forecast_as_of and acquisition_deadline")
        before = self._prediction(prediction_before, name="prediction_before")
        after = self._prediction(prediction_after, name="prediction_after")
        with self.store.lock(run_id):
            run = self._load(run_id)
            configured = run.get("model_status") in {"CONFIGURED", "PREDICTOR_ONLY"}
            if not configured and (before is not None or after is not None):
                raise ValueError("Unconfigured models cannot supply context predictions")
            predictor_only = run.get("model_status") == "PREDICTOR_ONLY"
            if predictor_only and model_versions != run["frozen_predictor_versions"]:
                raise ValueError("Predictor-only context requires the frozen model versions")
            if predictor_only and snapshot_kind == "initial" and after is not None and after != run["prediction"]:
                raise ValueError("Initial context prediction differs from the frozen initial f assessment")
            if predictor_only and snapshot_kind == "after_action" and parent_request_id is not None:
                intent = run["intents"].get(parent_request_id)
                if intent is None or parent_request_id not in run["receipts"]:
                    raise ValueError("Predictor-only after context requires its completed actual request")
                if before is not None and before != intent["assessment"]["prediction"]:
                    raise ValueError("Context before prediction differs from the actual acquisition assessment")
            now = self._now()
            if observed is None:
                observed = now
            if cutoff is None:
                cutoff = observed
            fixture = run.get("fixture")
            kickoff = _time(fixture["kickoff"]) if fixture and fixture.get("kickoff") else None
            forecast_audit = forecast_time is not None
            processing = None
            if prediction_processing is not None:
                if (not predictor_only or not forecast_audit or not isinstance(prediction_pipeline_version, str)
                        or not _identifier(prediction_pipeline_version)):
                    raise ValueError("Prediction processing requires a versioned frozen f pipeline and forecast audit")
                processing = self._prediction_processing(prediction_processing,
                    observed_at=observed, forecast_as_of=forecast_time)
                if (run.get("prediction_pipeline_version") is not None
                        and run["prediction_pipeline_version"] != prediction_pipeline_version):
                    raise ValueError("A predictor journal cannot change its frozen prediction pipeline")
            elif prediction_pipeline_version is not None:
                raise ValueError("A prediction pipeline declaration requires explicit processing")
            if forecast_audit:
                if not predictor_only or kickoff is None:
                    raise ValueError("Forecast audit requires a configured frozen f and fixture")
                if not (forecast_time == cutoff and forecast_time <= deadline <= kickoff
                        and forecast_time < kickoff and forecast_time <= observed):
                    raise ValueError("Forecast must precede kickoff and remain within its acquisition deadline and handoff")
            if kickoff is not None and not forecast_audit:
                for value, label in ((now, "captured_at"), (observed, "observed_at"),
                                     (available, "available_at"), (cutoff, "prediction_cutoff")):
                    if value is not None and value >= kickoff:
                        raise ValueError(f"Context {label} must precede kickoff")
            if available is not None and available > now:
                raise ValueError("available_at cannot be after captured_at")
            if available is not None and available < observed:
                raise ValueError("available_at cannot precede observed_at")
            if observed > now:
                raise ValueError("observed_at cannot be after captured_at")
            if cutoff > now:
                raise ValueError("prediction_cutoff cannot be after captured_at")
            snapshots = run.setdefault("context_snapshots", [])
            if snapshot_id is not None and any(item["snapshot_id"] == snapshot_id for item in snapshots):
                raise ValueError("Context snapshot IDs are append-only and must be unique")
            sequence = len(run.get("context_snapshots", []))
            snapshot = {"snapshot_id": snapshot_id or f"snapshot-{uuid4().hex}",
                        "physical_match_id": (fixture or {}).get("match_id") if fixture else None,
                        "sequence": sequence, "snapshot_kind": snapshot_kind,
                        "observed_at": observed.isoformat(),
                        "available_at": available.isoformat() if available is not None else None,
                        "availability_status": "known" if available is not None else "unknown",
                        "availability_basis": availability_basis if available is not None else None,
                        "captured_at": now.isoformat(),
                        "prediction_cutoff": cutoff.isoformat() if cutoff is not None else None,
                        "known_information": list(known_information or []),
                        "unknown_information": list(unknown_information or []),
                        "prediction_before": before, "prediction_after": after,
                        "state_delta": deepcopy(state_delta) if state_delta is not None else None,
                        "context": deepcopy(context or {}),
                        "parent_request": parent_request_id,
                        "parent_request_id": parent_request_id,
                        "parent_attempt_id": parent_attempt_id,
                        "stop_reason": stop_reason,
                        "information_mode": information_mode,
                        "model_status": run.get("model_status", "NOT_CONFIGURED")}
            if forecast_audit:
                on_time = now <= deadline and now < kickoff
                if processing is not None:
                    on_time = on_time and processing["status"] == "ready"
                snapshot.update(timing_schema="forecast-and-recording-clocks-v1",
                    forecast_as_of=forecast_time.isoformat(), acquisition_deadline=deadline.isoformat(),
                    forecast_eligible=on_time,
                    recording_scope="pre_match_context" if on_time else "audit_only")
            if processing is not None:
                run["prediction_pipeline_version"] = prediction_pipeline_version
                snapshot.update(prediction_processing=processing,
                                prediction_pipeline_version=prediction_pipeline_version)
            if predictor_only:
                snapshot.update(assessment_schema=run["assessment_schema"],
                                model_versions=deepcopy(model_versions),
                                candidate_expected_gains=None)
                # Missing runtime predictions remain missing; only actual
                # supplied forecasts update the materialized journal head.
                if after is not None or before is not None:
                    run["prediction"] = deepcopy(after if after is not None else before)
            snapshots.append(snapshot)
            run["prediction_cutoff"] = snapshot["prediction_cutoff"]
            self._append(run, "context_snapshot", deepcopy(snapshot), at=now)
            return deepcopy(snapshot)

    def stop(self, run_id, reason: str):
        if not _identifier(reason):
            raise ValueError("Stop reason must be an inert identifier")
        with self.store.lock(run_id):
            run = self._load(run_id)
            if run["pending_request_id"]:
                raise ValueError("Resolve pending receipt before stopping; reservation remains active")
            if run["state"] == "STOPPED":
                return deepcopy(run)
            stopped_at = self._now()
            run["state"], run["stop_reason"], run["stopped_at"] = "STOPPED", reason, stopped_at.isoformat()
            self._append(run, "stopped", {"reason": reason, "stopped_at": stopped_at.isoformat()}, at=stopped_at)
            return deepcopy(run)

    def read(self, run_id):
        with self.store.lock(run_id):
            return deepcopy(self._load(run_id))
