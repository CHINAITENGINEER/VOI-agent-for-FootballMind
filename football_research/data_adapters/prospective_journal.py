"""Journal adapter for the existing public-web executor, without new I/O.

The menu is supplied by LearnedVOIRuntime before execution.  The existing
AcquisitionDecisionJournal owns intent persistence, reservation, receipts,
recovery and stop events; RawArchive still owns all response bytes.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from football_agent.audit import digest
from football_platform.information.acquisition_journal import AcquisitionDecisionJournal


class _JournalArchiveView:
    """Add the durable journal identity to one executor call's archive saves."""

    def __init__(self, archive, run_id, request_id):
        self.archive, self.run_id, self.request_id = archive, run_id, request_id
        self.root = archive.root
        self.save_timings = {}

    def save(self, source, endpoint, params, body, **kwargs):
        context = deepcopy(kwargs.get("context") or {})
        context.update(journal_run_id=self.run_id, journal_request_id=self.request_id)
        # This is the real invocation boundary, not the transport's earlier
        # response observation clock or a claim of completed disk persistence.
        started_at = datetime.now(timezone.utc).isoformat()
        context.update(archive_save_started_at=started_at,
                       archive_save_timing_basis="measured_local_utc_archive_save_start")
        kwargs["context"] = context
        observation = self.archive.save(source, endpoint, params, body, **kwargs)
        returned_at = datetime.now(timezone.utc).isoformat()
        # Return is known only after the immutable archive record exists.  Keep
        # it in the existing executor receipt; do not rewrite archive metadata
        # and do not claim to have instrumented the filesystem's fsync event.
        self.save_timings[str(observation.id)] = {
            "observation_id": str(observation.id), "raw_sha256": observation.content_hash,
            "archive_save_started_at": started_at,
            "archive_save_returned_at": returned_at,
            "archive_save_timing_basis": "measured_local_utc_archive_save_start",
            "archive_return_timing_basis": "measured_local_utc_archive_save_return",
            "persistence_boundary": "return_observed_fsync_not_instrumented"}
        return observation

    def __getattr__(self, name):
        return getattr(self.archive, name)


class JournaledProspectiveExecutor:
    """Opt-in intent wrapper, optionally recording a frozen f without a g."""

    def __init__(self, executor, *, root, fixture, request_limit, policy_arm,
                 policy_version, random_seed, propensity_logging, clock=None,
                 predictor_policy=None):
        self.executor, self.tools = executor, executor.tools
        self.archive = self.tools.archive
        options = {} if clock is None else {"clock": clock}
        self.journal = AcquisitionDecisionJournal(Path(root), self.archive, **options)
        self.predictor_policy = predictor_policy
        self.predictor_versions = None
        self.prediction_pipeline_version = getattr(predictor_policy, "prediction_pipeline_version", None)
        self.policy_version = policy_version
        assessment = None
        if predictor_policy is not None:
            if getattr(predictor_policy, "predictor_only", False) is not True:
                raise ValueError("Journal predictor_policy must explicitly be predictor-only")
            if getattr(predictor_policy, "version", None) != policy_version:
                raise ValueError("Journal policy version must match the predictor-only policy")
            assessment = self.journal._assessment(predictor_policy.journal_initial_assessment())
            if assessment["status"] != "PREDICTOR_ONLY":
                raise ValueError("Frozen f initialization requires a PREDICTOR_ONLY assessment")
            self.predictor_versions = deepcopy(assessment["model_versions"])
            self._check_predictor()
        keys = ("match_id", "competition_id", "season", "home_team_id", "away_team_id",
                "kickoff", "home_team_name", "away_team_name")
        started = self.journal.start(request_limit=request_limit,
            fixture={key: fixture[key] for key in keys if key in fixture},
            policy_arm=policy_arm, policy_version=policy_version,
            random_seed=random_seed, propensity_logging=propensity_logging,
            initial_assessment=assessment)
        self.run_id, self.policy_arm, self.policy_version = started["run_id"], policy_arm, policy_version
        self.menu = None
        self.last_request_id = None

    def _check_predictor(self):
        if self.predictor_policy is None:
            return
        if (getattr(self.predictor_policy, "predictor_only", False) is not True
                or getattr(self.predictor_policy, "version", None) != self.policy_version
                or getattr(self.predictor_policy, "prediction_pipeline_version", None) != self.prediction_pipeline_version
                or [getattr(self.predictor_policy.predictor, "model_version", None)] != self.predictor_versions):
            raise ValueError("Predictor-only journal cannot change frozen model or policy versions")

    def _public_assessment(self, public):
        if self.predictor_policy is None:
            return None
        self._check_predictor()
        prediction = public.get("prediction")
        if not isinstance(prediction, dict) or prediction.get("model_version") != self.predictor_versions[0]:
            raise ValueError("Runtime prediction must identify the frozen predictor model")
        if (self.prediction_pipeline_version is not None
                and prediction.get("prediction_pipeline_version") != self.prediction_pipeline_version):
            raise ValueError("Runtime prediction must identify the frozen prediction pipeline")
        values = prediction.get("probabilities")
        if isinstance(values, (list, tuple)) and len(values) == 3:
            values = dict(zip(("home", "draw", "away"), values))
        return self.journal._assessment({"status": "PREDICTOR_ONLY", "prediction": values,
            "candidate_expected_gains": None, "model_versions": deepcopy(self.predictor_versions)})

    def observe_menu(self, payload):
        if self.predictor_policy is not None:
            self._public_assessment(payload["public_before"])
        self.menu = deepcopy(payload)

    def observe_context(self, payload):
        """Record the actual runtime forecast; never rerun f or manufacture g.

        The runtime supplies these predictions after inference. The original
        acquisition horizon is context metadata, not a future decision clock.
        Legacy collectors keep their existing raw-only context callback.
        """
        if self.predictor_policy is None:
            return None
        self._check_predictor()
        versions = payload.get("predictor_model_versions")
        if versions != self.predictor_versions:
            raise ValueError("Runtime context must identify the frozen predictor model versions")
        observed = payload.get("observed_at")
        if observed is None:
            raise ValueError("Predictor-only context requires its measured runtime handoff clock")
        from football_platform.information.competition_context import _time
        cutoff = payload.get("prediction_cutoff") or observed
        if _time(cutoff) > _time(observed):
            raise ValueError("Context cutoff cannot use a future acquisition horizon")
        context = deepcopy(payload)
        timing = {}
        if payload.get("schema") in {"runtime-predictor-context-v3", "runtime-predictor-context-v4"}:
            timing = {"forecast_as_of": payload["forecast_as_of"],
                      "acquisition_deadline": payload["acquisition_deadline"]}
        if payload.get("schema") == "runtime-predictor-context-v4":
            if (self.prediction_pipeline_version is None
                    or payload.get("prediction_pipeline_version") != self.prediction_pipeline_version):
                raise ValueError("Runtime context must identify the frozen prediction pipeline")
            timing.update(prediction_pipeline_version=self.prediction_pipeline_version,
                          prediction_processing=deepcopy(payload.get("prediction_processing")))
            if timing["prediction_processing"] is None:
                raise ValueError("Runtime v4 context requires prediction processing")
        elif self.prediction_pipeline_version is not None:
            raise ValueError("A versioned semantic prediction pipeline requires v4 runtime context")
        # Preserve even an absent forecast. No market prior or previous f
        # forecast is substituted for a missing runtime prediction.
        return self.journal.record_context(self.run_id,
            snapshot_kind=payload["snapshot_kind"], observed_at=observed,
            available_at=payload.get("available_at"),
            availability_basis=payload.get("availability_basis"),
            prediction_cutoff=cutoff,
            prediction_before=payload.get("prediction_before"),
            prediction_after=payload.get("prediction_after"),
            state_delta=payload.get("state_delta"), context=context,
            parent_request_id=(self.last_request_id if payload["snapshot_kind"] in {"after", "after_action"} else None),
            parent_attempt_id=payload.get("parent_attempt_id"),
            stop_reason=payload.get("stop_reason"),
            model_versions=deepcopy(versions), **timing)

    def __call__(self, action, public):
        if self.menu is None or self.menu.get("public_before") != public:
            raise ValueError("A matching actual runtime menu must precede journal execution")
        actions = self.menu["generated_actions"]
        menu_items = {item["action_id"]: item for item in self.menu["menu"]}
        candidate_ids = [item["action_id"] for item in actions]
        selected = menu_items.get(action["action_id"])
        if not selected or not selected.get("selected") or action not in actions:
            raise ValueError("Journal action must be the actual selected runtime candidate")
        probability = selected.get("selection_probability")
        basis = ("fixed_order" if self.policy_arm == "fixed_order" else
                 "full_probe" if self.policy_arm == "full_probe" else
                 "uniform_exploration" if probability is not None else "unknown_value_exploration")
        prepared = self.tools.prepare(action,
            {"task": {"seed_urls": self.executor.seed_urls}, "documents": public["documents"]})
        request_id = "runtime-step-" + str(self.menu["step"])
        metadata = [{**deepcopy(candidate), **deepcopy(menu_items[candidate["action_id"]]),
                     "candidate_id": candidate["action_id"], "rank": rank + 1}
                    for rank, candidate in enumerate(actions)]
        intent = self.journal.begin(self.run_id, request_id=request_id, action_id=action["action_id"],
            input_parameters={key: prepared[key] for key in ("source", "endpoint", "params")},
            candidates=candidate_ids, reserve_requests=1, candidate_metadata=metadata,
            selection_basis=basis, policy_version=self.policy_version,
            selection_probability=probability, assessment=self._public_assessment(public))
        self.last_request_id = request_id
        if not intent["execute"]:
            self.journal.recover_pending(self.run_id)
            raise RuntimeError("Journal intent already exists; replay or reconcile without repeating HTTP")
        archive_view = _JournalArchiveView(self.archive, self.run_id, request_id)
        self.tools.archive = archive_view
        try:
            returned = self.executor(action, public)
        except BaseException:
            self.journal.recover_pending(self.run_id)
            raise
        finally:
            self.tools.archive = self.archive
        observation = returned.get("receipt", {}).get("observation")
        if observation is None:
            self.journal.recover_pending(self.run_id)
            raise RuntimeError("Actual tool response lacks its committed RawArchive observation")
        receipt = self.journal.complete(self.run_id, request_id=request_id, observation=observation)
        processing_status = {"OK": "DATA_AVAILABLE", "NO_DATA": "NO_DATA", "FAILED": "REQUIRES_REVIEW"}
        self.journal.record_processing(self.run_id, request_id=request_id,
            status=processing_status.get(returned.get("status"), "INVALID_CONTENT"),
            extraction_sha256=digest(returned.get("document")), missing_fields=[])
        returned = deepcopy(returned)
        saved_timing = archive_view.save_timings.get(str(observation.get("id")))
        if saved_timing is not None:
            returned["receipt"]["archive_save_timing"] = deepcopy(saved_timing)
        returned["receipt"]["acquisition_journal"] = {
            "run_id": self.run_id, "request_id": request_id,
            "root": self.journal.store.root.name, "path_scope": "batch_relative",
            "receipt_status": receipt["status"], "receipt_at": receipt["receipt_at"]}
        return returned

    def finish(self, reason):
        recovery = self.journal.recover_pending(self.run_id)
        current = self.journal.read(self.run_id)
        if current["pending_request_id"]:
            return {"run_id": self.run_id, "state": current["state"], "recovery": recovery}
        stopped = self.journal.stop(self.run_id, reason)
        return {"run_id": self.run_id, "state": stopped["state"], "stop_reason": stopped["stop_reason"],
                "requests": stopped["budget"]["charged_requests"], "recovery": recovery}
