"""Execute learned VoI and explicitly budgeted unknown-information exploration.

The response model continues to own numerical values and posterior updates.
Unmodeled documents enter the candidate generator's memory, never a made-up
probability update. Exploration reuses the existing precommitted sampler.
Every reservation is persisted before I/O; unfinished runs are not replayed
against live providers. This is a headless research runner, not a UI service.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import re

from football_research.acquisition.expected_voi import COMPUTED
from football_research.acquisition.joint_planner import UNITS, _budget, _cost
from football_research.acquisition.joint_voi import LearnedJointVOIPolicy, MODELED_GROUPS
from football_research.acquisition.state import sanitize_state, parse_time
from football_research.acquisition.value import _canonical
from football_spider.storage.raw_archive import atomic_write
from .inquiry_exploration import UniformInquiryExploration, CONFIGURATION_KEY
from .g_prior_features import (ALIAS as G_PRIOR_ALIAS, FIELD as G_PRIOR_FIELD,
                                validate_any_prior_features, strip_prior_features)

VERSION = "learned-voi-open-execution-v1"
EVIDENCE_VERSION = "learned-voi-open-execution-v2"
TEXT_VERSION = "learned-voi-open-execution-v3"
CONTEXT_VERSION = "learned-voi-open-execution-v4"
EVIDENCE_ONLY_VERSION = "learned-voi-open-execution-v5"
OFFICIAL_CONTEXT_VERSION = "learned-voi-open-execution-v6"
POLICY_ARM_VERSION = "footballmind-policy-arm-v1"
POLICY_ARMS = ("voi", "fixed_order", "random_explore", "full_probe")
_ID = re.compile(r"[a-zA-Z0-9_.:-]{1,100}\Z")
_ACTION_FIELDS = {"action_id", "group_ids", "available", "cost_status", "cost",
                  "information_key", "tool", "url", "query", "source", "version",
                  # Optional, predeclared feature vector for the separate
                  # synthetic g teacher prior.  It is metadata only; the
                  # runtime still selects on the empirical value lane.
                  "synthetic_prior_features", "g_prior_features"}


def digest(value):
    return sha256(_canonical(value).encode()).hexdigest()


def _save(path, value):
    atomic_write(Path(path), (json.dumps(value, ensure_ascii=False, sort_keys=True,
                                       indent=2, allow_nan=False)+"\n").encode("utf-8"))


def _actions(items):
    if not isinstance(items, list) or len(items) > 64:
        raise ValueError("At most 64 generated actions are supported per decision")
    result, ids, keys = [], set(), set()
    for item in items:
        if not isinstance(item, dict) or set(item) - _ACTION_FIELDS:
            raise ValueError("Candidate metadata must not contain returns, labels or arbitrary fields")
        action = deepcopy(item)
        if G_PRIOR_FIELD in action and G_PRIOR_ALIAS in action:
            raise ValueError("Use only one of synthetic_prior_features or g_prior_features")
        if G_PRIOR_FIELD in action:
            validate_any_prior_features(action[G_PRIOR_FIELD])
        elif G_PRIOR_ALIAS in action:
            validate_any_prior_features(action[G_PRIOR_ALIAS])
        aid, key = action.get("action_id"), action.get("information_key")
        if not isinstance(aid, str) or not _ID.fullmatch(aid) or aid in ids:
            raise ValueError("Unique protocol action IDs required")
        if not isinstance(key, str) or not key or key in keys:
            raise ValueError("Unique stable information keys required")
        if aid in MODELED_GROUPS:
            if key != aid:
                raise ValueError("Modeled action information key must equal its ID")
        elif action.get("group_ids") != [] or key in MODELED_GROUPS:
            raise ValueError("Unmodeled actions cannot alias modeled feature groups")
        _cost(action)
        ids.add(aid)
        keys.add(key)
        result.append(action)
    return result


def _state_timing(state, *, observed_at, captured_at=None, parent_attempt_id=None):
    """Return a conservative timing sidecar for an exported state snapshot.

    A model state contains per-group ``available_at`` values only for observed
    groups.  We therefore never infer an overall availability timestamp from
    the local capture clock.  Consumers can still use the captured/observed
    clocks while treating availability as unknown when no evidence carries a
    usable bound.
    """
    available = []
    for group in (state or {}).get("groups", {}).values():
        if isinstance(group, dict) and group.get("status") == "observed":
            value = group.get("available_at")
            if value:
                try:
                    available.append(parse_time(value).isoformat())
                except (TypeError, ValueError):
                    pass
    result = {"observed_at": observed_at, "captured_at": captured_at,
              "available_at": max(available) if available else None,
              "availability_status": "known" if available else "unknown"}
    if parent_attempt_id is not None:
        result["parent_attempt_id"] = parent_attempt_id
    return result


class LearnedVOIRuntime:
    """Budgeted query execution with a public-only proposal callback.

    ``propose(public)`` sees only admitted evidence, not provider returns in
    advance. ``execute(action, public)`` returns {status, state_after?, document?,
    actual_cost, receipt}. ``receipt`` is supervisor-only. Unknown documents
    are not numerical VoI training examples. Candidate generation is incomplete.
    Costs used by the policy are reserved declared credits; measured provider
    resources are recorded separately and must not be presented as credit bills.
    """

    def __init__(self, policy, *, seed, exploration_limits, max_exploration_actions,
                 max_steps=16, policy_arm=None, arm_assigned_at=None,
                 policy_arm_version=POLICY_ARM_VERSION, random_seed=None,
                 propensity_logging=True, legacy_receipt_clock=False,
                 legacy_predictor_context=False):
        from .text_voi_policy import TextVOIPolicy
        from .context_voi_policy import ContextVOIPolicy
        from .evidence_only_policy import EvidenceOnlyPolicy
        from .official_context_policy import OfficialContextVOIPolicy
        self.official_context_mode = isinstance(policy, OfficialContextVOIPolicy)
        self.context_mode = isinstance(policy, ContextVOIPolicy)
        self.predictor_only_mode = getattr(policy, "predictor_only", False) is True
        self.evidence_only_mode = isinstance(policy, EvidenceOnlyPolicy)
        self.text_mode = isinstance(policy, TextVOIPolicy)
        if not self.text_mode and not isinstance(policy, LearnedJointVOIPolicy):
            raise ValueError("An existing learned VoI policy is required")
        if type(max_exploration_actions) is not int or not 0 <= max_exploration_actions <= 16:
            raise ValueError("Exploration allowance must be an integer from 0 to 16")
        if type(max_steps) is not int or not 1 <= max_steps <= 64:
            raise ValueError("A finite step bound from 1 to 64 is required")
        if policy_arm is not None and policy_arm not in POLICY_ARMS:
            raise ValueError("Policy arm must be one of: " + ", ".join(POLICY_ARMS))
        if self.predictor_only_mode and policy_arm not in {"random_explore", "fixed_order", "full_probe"}:
            raise ValueError("f-only collection requires an explicit controlled exploration arm")
        if self.predictor_only_mode and propensity_logging is not True:
            raise ValueError("f-only collection requires propensity logging")
        if not isinstance(policy_arm_version, str) or not policy_arm_version:
            raise ValueError("A policy arm version is required")
        if arm_assigned_at is not None:
            # Validate the frozen assignment now; run() also checks it against
            # the actual pre-I/O creation clock.
            arm_assigned_at = parse_time(arm_assigned_at).isoformat()
        if random_seed is None:
            random_seed = seed
        if type(random_seed) is not int or not 0 <= random_seed < 2**256:
            raise ValueError("Random seed must be an integer in [0, 2**256)")
        if type(propensity_logging) is not bool:
            raise ValueError("propensity_logging must be boolean")
        self.policy = policy
        # The precommitted assignment seed is the sole source of randomized
        # arm draws.  Keep ``seed`` as a backward-compatible alias when no
        # explicit random_seed was supplied.
        self.sampler = UniformInquiryExploration(random_seed)
        self.exploration_limits, _ = _budget({"limits": exploration_limits, "spent": {}})
        self.max_exploration_actions, self.max_steps = max_exploration_actions, max_steps
        self.trajectory_enabled = policy_arm is not None
        self.policy_arm = policy_arm or "voi"
        self.arm_assigned_at = arm_assigned_at
        self.policy_arm_version = policy_arm_version
        self.random_seed = random_seed
        self.propensity_logging = propensity_logging
        # Read compatibility for archives made before the separate receipt
        # observation clock. Replay opts in; new acquisition always uses v2.
        self.legacy_receipt_clock = legacy_receipt_clock
        # Pin the old handoff contract when replaying an existing f-only run.
        # New f-only runs distinguish the forecast clock from audit persistence.
        self.safe_predictor_context = self.predictor_only_mode and not legacy_predictor_context

    def run(self, initial_state, *, run_id, limits, propose, execute, output,
            provider_version, generator_version, clock=None, run_metadata=None,
            evidence_match=None, evidence_version=None, trajectory_recorder=None,
            context_recorder=None, training_exploration=None):
        if not isinstance(run_id, str) or not _ID.fullmatch(run_id):
            raise ValueError("A protocol run ID is required")
        if not provider_version or not generator_version:
            raise ValueError("Explicit provider and candidate-generator versions required")
        if trajectory_recorder is not None and not callable(trajectory_recorder):
            raise ValueError("trajectory_recorder must be callable when supplied")
        if context_recorder is not None and not callable(context_recorder):
            raise ValueError("context_recorder must be callable when supplied")
        if training_exploration is not None:
            if not callable(getattr(training_exploration, "select_for_collection", None)):
                raise ValueError("training_exploration must expose select_for_collection")
            if not getattr(training_exploration, "enabled", False):
                raise ValueError("training_exploration requires explicit enabled=true")
        # Validate before creating the run. Do not deserialize labels/hidden payloads.
        self.policy._state(initial_state)
        if (self.text_mode or self.context_mode or self.evidence_only_mode) and (initial_state.get("action_history") or
                              initial_state.get("open_text", {}).get("history")):
            raise ValueError("Start a text acquisition run before reading responses, not from a fabricated history")
        state = self.policy.public_state(initial_state) if self.text_mode or self.context_mode or self.evidence_only_mode else sanitize_state(initial_state)
        state["action_history"] = deepcopy(initial_state.get("action_history", []))
        limits, spent = _budget({"limits": limits, "spent": {}})
        exploration_spent = {u: 0. for u in UNITS}
        clock = clock or (lambda: datetime.now(timezone.utc))
        created_at = clock().isoformat()
        assigned_at = self.arm_assigned_at or created_at
        if parse_time(assigned_at) > parse_time(created_at):
            raise ValueError("Policy arm assignment must be frozen before acquisition starts")
        # Operational episodes are strictly prospective.  Historical replay
        # remains available to existing tests/tools, but it is never labelled
        # as a prospective arm run.
        kickoff_at = parse_time(state["base"]["kickoff"])
        operational_run = state.get("information_mode") == "operational"
        forecast_at = created_at
        prepare_prediction = getattr(self.policy, "prepare_public_prediction", None)
        semantic_prediction_mode = callable(prepare_prediction)
        last_prediction_processing = None

        def processing_receipt(prepared, input_at):
            ready = prepared["status"] == "ready"
            return {"schema": "entity-semantic-prediction-processing-v1",
                "status": "ready" if ready else "not_updated", "reason": prepared["reason"],
                "input_at": input_at, "available_at": prepared["available_at"],
                "effective_at": max(parse_time(input_at), parse_time(prepared["available_at"])).isoformat()
                    if ready else None}

        def record_context(kind, current, *, parent_attempt_id=None, documents=None,
                           prediction_before=None, prediction_after=None):
            # This is a measured handoff of the assembled runtime state.  It
            # says nothing about when an external source first published it.
            # Legacy runs without this callback keep their timing unknown.
            if context_recorder is None:
                return None
            handed_at = clock().isoformat()
            if operational_run and parse_time(handed_at) >= kickoff_at and not self.safe_predictor_context:
                raise ValueError("Prospective context handoff must precede kickoff")
            payload = {"schema": "runtime-context-v1", "run_id": run_id,
                "snapshot_kind": kind, "state": deepcopy(current),
                "documents": deepcopy(documents or []),
                "observed_at": handed_at, "available_at": handed_at,
                "availability_status": "known",
                "availability_basis": "runtime_state_handoff",
                "prediction_cutoff": min(parse_time(current["prediction_cutoff"]),
                                         parse_time(handed_at)).isoformat(),
                "acquisition_deadline": current.get("prediction_cutoff"),
                "parent_attempt_id": parent_attempt_id}
            if self.predictor_only_mode:
                payload.update(schema="runtime-predictor-context-v2",
                    prediction_before=(dict(zip(("home", "draw", "away"), prediction_before))
                                       if prediction_before is not None else None),
                    prediction_after=(dict(zip(("home", "draw", "away"), prediction_after))
                                      if prediction_after is not None else None),
                    predictor_model_versions=[self.policy.predictor.model_version])
            if self.safe_predictor_context:
                # A late audit record retains an earlier forecast; it does not
                # make the late receipt visible at the earlier forecast time.
                payload.update(schema="runtime-predictor-context-v3",
                    forecast_as_of=forecast_at, prediction_cutoff=forecast_at)
                if semantic_prediction_mode:
                    payload.update(schema="runtime-predictor-context-v4",
                        prediction_pipeline_version=self.policy.prediction_pipeline_version,
                        prediction_processing=deepcopy(last_prediction_processing))
            reference = context_recorder(deepcopy(payload))
            if not isinstance(reference, dict) or not reference.get("captured_at"):
                raise ValueError("Context recorder must return a measured archive capture time")
            captured = parse_time(reference["captured_at"])
            if captured < parse_time(handed_at) or (operational_run and captured >= kickoff_at and not self.safe_predictor_context):
                raise ValueError("Context capture must follow handoff and precede kickoff")
            timing = {key: payload[key] for key in ("observed_at", "available_at",
                "availability_status", "availability_basis", "prediction_cutoff", "parent_attempt_id")}
            timing.update({key: reference[key] for key in
                ("captured_at", "raw_path", "raw_sha256", "raw_observation_id",
                 "archive_save_started_at", "archive_save_returned_at",
                 "capture_timing_basis", "persistence_boundary") if key in reference})
            if self.safe_predictor_context:
                on_time = (captured <= parse_time(current["prediction_cutoff"])
                           and (not operational_run or captured < kickoff_at))
                if semantic_prediction_mode:
                    on_time = on_time and last_prediction_processing["status"] == "ready"
                    timing.update(prediction_pipeline_version=self.policy.prediction_pipeline_version,
                        prediction_processing=deepcopy(last_prediction_processing))
                timing.update(forecast_as_of=forecast_at,
                    forecast_eligible=on_time,
                    recording_scope="pre_match_context" if on_time else "audit_only")
            return timing

        if operational_run and parse_time(created_at) >= kickoff_at:
            raise ValueError("Prospective acquisition must start before kickoff")
        if self.context_mode or self.evidence_only_mode:
            self.policy.check_start(created_at)
        if self.text_mode and parse_time(self.policy.text_model.metadata["training_as_of"]) > parse_time(created_at):
            raise ValueError("The text response model must be frozen before acquisition starts")
        if self.text_mode and parse_time(self.policy.base_policy.model.metadata["training_as_of"]) > parse_time(created_at):
            raise ValueError("The initial fixture model must be frozen before acquisition starts")
        if (not self.text_mode and self.policy.manifest.get("schema") == "prospective-market-prior-policy-v1"
                and parse_time(self.policy.model.frozen_at) > parse_time(created_at)):
            raise ValueError("The observed market quote must be frozen before acquisition starts")
        discovery_model = self.policy.discovery_model if self.text_mode else None
        if discovery_model is not None and parse_time(discovery_model.metadata["training_as_of"]) > parse_time(created_at):
            raise ValueError("The discovery response model must be frozen before acquisition starts")
        risk_model = self.policy.risk_model if self.text_mode else None
        if risk_model is not None and parse_time(risk_model.manifest["training_as_of"]) > parse_time(created_at):
            raise ValueError("The empirical text risk model must be frozen before acquisition starts")
        projection = None
        if self.evidence_only_mode or self.official_context_mode:
            if evidence_match != self.policy.evidence_match:
                raise ValueError("Evidence-only projection must preserve the verified official fixture and aliases")
        if evidence_match is not None:
            from .match_evidence import build_match_evidence, VERSION as extractor_version
            extractor_version = evidence_version or extractor_version
            # Only match identity is part of this adapter; no hidden feature or
            # result payload is forwarded to the evidence/candidate generator.
            allowed = {"match_id", "home_team", "away_team", "home_aliases", "away_aliases",
                       "kickoff_at", "prediction_cutoff"}
            if not isinstance(evidence_match, dict) or set(evidence_match)-allowed:
                raise ValueError("Evidence projection requires a bounded match identity")
            if (evidence_match.get("match_id") != state["match_id"] or
                    parse_time(evidence_match["prediction_cutoff"]) != parse_time(state["prediction_cutoff"]) or
                    parse_time(evidence_match["kickoff_at"]) != parse_time(state["base"]["kickoff"])):
                raise ValueError("Evidence identity differs from the probability-model fixture")
            build_match_evidence(evidence_match, [], as_of=created_at, version=extractor_version)
            projection = {"version": extractor_version, "match": deepcopy(evidence_match)}
        if (self.text_mode or self.context_mode or self.evidence_only_mode) and projection is None:
            raise ValueError("Text acquisition needs the match identity and actual decision clocks")
        version = OFFICIAL_CONTEXT_VERSION if self.official_context_mode else EVIDENCE_ONLY_VERSION if self.evidence_only_mode else CONTEXT_VERSION if self.context_mode else TEXT_VERSION if self.text_mode else EVIDENCE_VERSION if projection else VERSION
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        config = {"schema": version, "run_id": run_id, "created_at": created_at,
                  "initial_state": state, "policy_manifest": deepcopy(self.policy.manifest),
                  "policy_version": self.policy.version, "limits": limits,
                  "exploration_limits": self.exploration_limits,
                  "max_exploration_actions": self.max_exploration_actions,
                  "max_steps": self.max_steps, "provider_version": provider_version,
                  "generator_version": generator_version,
                  "supervisor_metadata": deepcopy(run_metadata),
                  CONFIGURATION_KEY: self.sampler.configuration(),
                  "candidate_generation_complete": False,
                  "accounting": "declared_reservations_separate_from_actual_resources"}
        # The source mode is an explicit supervisor declaration.  Do not infer
        # prospective visibility merely because a trajectory was requested.
        if isinstance(run_metadata, dict) and run_metadata.get("information_mode") is not None:
            config["information_mode"] = run_metadata["information_mode"]
        if self.trajectory_enabled and trajectory_recorder is not None:
            config["trajectory_raw_archive"] = {
                "schema": "prospective-menu-archive-v1",
                "source": "agent_trajectory",
                "evidence_scope": "menu_and_generated_actions_before_io",
            }
        if self.trajectory_enabled:
            config.update(trajectory_schema="prospective-acquisition-trajectory-v1",
                policy_arm=self.policy_arm, policy_arm_version=self.policy_arm_version,
                arm_assigned_at=assigned_at, random_seed=self.random_seed,
                propensity_logging=self.propensity_logging,
                arm_selection_design={"order": "generated_menu_order",
                    "finite_feasible_candidate_sequence": True,
                    "unknown_value_not_estimated": True})
        if training_exploration is not None:
            config["training_exploration"] = {
                "adapter_schema": "footballmind-g-training-exploration-adapter-v1",
                "enabled": True,
                "selection_authority": "data_collection_only",
                "empirical_g_enabled": False,
                "f_invoked_by_adapter": False,
                "label_status": "pending_outcome",
            }
        initial_f_forecast = None
        if semantic_prediction_mode:
            self.policy.attach_runtime_archive(output)
        if self.predictor_only_mode:
            config["predictor_context_schema"] = (
                "runtime-predictor-context-v4" if semantic_prediction_mode else
                "runtime-predictor-context-v3" if self.safe_predictor_context else "runtime-predictor-context-v2")
            initial_public = {"state": deepcopy(state), "documents": [],
                "attempts": [], "prediction": self.policy.model.predict_partial(state),
                "decision_at": created_at, "budget": {"limits": limits, "spent": deepcopy(spent)}}
            if semantic_prediction_mode:
                prepared = prepare_prediction(initial_public)
                last_prediction_processing = processing_receipt(prepared, created_at)
                if prepared["status"] != "ready":
                    _save(output / "configuration.json", config)
                    _save(output / "initialization-failure.json", {
                        "schema": "entity-semantic-initialization-failure-v1",
                        "status": "not_started", "reason": prepared["reason"],
                        "prediction": None, "prediction_processing": last_prediction_processing,
                        "source_requests": 0, "numeric_voi_training_eligible": False})
                    raise ValueError("Empty initial semantic state could not be predicted")
                initial_f_forecast = prepared["prediction"]
                forecast_at = last_prediction_processing["effective_at"]
            else:
                initial_f_forecast = self.policy.predict_public(initial_public)
        if context_recorder is not None:
            config["initial_state_timing"] = record_context("initial", state,
                prediction_after=initial_f_forecast["probabilities"] if initial_f_forecast else None)
        if projection:
            config["evidence_projection"] = projection
        if self.evidence_only_mode:
            config.update(analysis_mode="evidence_only", prior_status="prior_unavailable")
        if self.official_context_mode:
            config.update(analysis_mode="official_optional_context", prior_status="prior_unavailable",
                prior_probabilities=None)
        if self.text_mode:
            # Persist the exact response law before any external request. This
            # artifact belongs to the supervisor and never reaches propose().
            model_path = output/"text-response-model.json"
            self.policy.text_model.save(model_path)
            config["text_response_model"] = {"file": model_path.name,
                "sha256": sha256(model_path.read_bytes()).hexdigest(),
                "model_version": self.policy.text_model.model_version,
                "evidence_origin": self.policy.text_model.metadata["evidence_origin"]}
            if risk_model is not None:
                risk_path = output/"text-risk-model.json"
                risk_model.save(risk_path)
                config["text_risk_model"] = {"file": risk_path.name,
                    "sha256": sha256(risk_path.read_bytes()).hexdigest(),
                    "model_version": risk_model.model_version,
                    "evidence_origin": risk_model.manifest["evidence_origin"]}
            if discovery_model is not None:
                discovery_path = output/"discovery-response-model.json"
                discovery_model.save(discovery_path)
                config["discovery_response_model"] = {"file": discovery_path.name,
                    "sha256": sha256(discovery_path.read_bytes()).hexdigest(),
                    "model_version": discovery_model.model_version,
                    "evidence_origin": discovery_model.metadata["evidence_origin"]}
        if self.context_mode:
            model_path = output/"context-predictor.json"
            atomic_write(model_path, self.policy.predictor_bytes)
            if model_path.read_bytes() != self.policy.predictor_bytes:
                raise ValueError("The runtime predictor differs from its pre-query bytes")
            config["context_predictor"] = {"file": model_path.name,
                "sha256": self.policy.predictor_sha256,
                "model_version": self.policy.predictor.model_version,
                "evidence_origin": self.policy.predictor.metadata["evidence_origin"]}
            if self.policy.value_model is not None:
                value_path = output/"context-value-model.json"
                if self.policy.value_bytes is None:
                    self.policy.value_model.save(value_path)
                else:
                    atomic_write(value_path, self.policy.value_bytes)
                config["context_value_model"] = {"file": value_path.name,
                    "sha256": sha256(value_path.read_bytes()).hexdigest(),
                    "model_version": self.policy.value_model.model_version,
                    "evidence_origin": self.policy.value_model.metadata["evidence_origin"]}
        synthetic_prior = getattr(self.policy, "synthetic_prior", None)
        if synthetic_prior is not None:
            prior_bytes = getattr(synthetic_prior, "artifact_bytes", None)
            if not isinstance(prior_bytes, (bytes, bytearray)):
                raise ValueError("A synthetic prior used by a runtime must provide immutable artifact bytes")
            prior_path = output/"synthetic-prior-model.json"
            atomic_write(prior_path, bytes(prior_bytes))
            if prior_path.read_bytes() != bytes(prior_bytes):
                raise ValueError("The archived synthetic prior differs from its pre-query bytes")
            config["synthetic_prior"] = {
                "file": prior_path.name,
                "sha256": sha256(prior_path.read_bytes()).hexdigest(),
                "adapter_schema": synthetic_prior.manifest["schema"],
                "model_schema": synthetic_prior.manifest["model_schema"],
                "source_path": synthetic_prior.manifest.get("source_path"),
                "selection_mode": synthetic_prior.manifest.get("selection_mode"),
            }
        _save(output/"configuration.json", config)
        documents, attempts, cycles, bindings = [], [], [], {}
        measured = {u: 0. for u in UNITS}
        unmeasured = {u: 0 for u in UNITS}
        used_keys = set()
        learning_count = 0
        arm_query_count = 0
        run = {"run_id": run_id, "configuration": config, "decisions": []}
        terminal = {"decision": "halt", "reason": "step_limit", "stop_certificate": None}
        context_clock = parse_time(created_at)
        last_forecast = initial_f_forecast or self.policy.model.predict_partial(state)
        if self.official_context_mode:
            last_forecast = self.policy.predict_public({'state': deepcopy(state), 'documents': [],
                'attempts': [], 'prediction': last_forecast, 'decision_at': created_at,
                'budget': {'limits': limits, 'spent': deepcopy(spent)}})
        for step in range(self.max_steps):
            semantic_halt_reason = None
            before_context_timing = (None if self.predictor_only_mode else
                record_context("before_action", state, documents=documents))
            # One shared observation clock makes the menu and decision
            # auditable without creating an extra local timestamp that could
            # be mistaken for source availability.
            cycle_time = clock().isoformat() if projection or self.trajectory_enabled else None
            public = {"state": deepcopy(state), "documents": deepcopy(documents),
                      "attempts": deepcopy(attempts), "prediction": self.policy.model.predict_partial(state),
                      "budget": {"limits": limits, "spent": deepcopy(spent)},
                      "exploration": self.sampler.public_summary()}
            if cycle_time is not None:
                public["decision_at"] = cycle_time
            expired = False
            if operational_run and cycle_time is not None and parse_time(cycle_time) >= kickoff_at:
                expired = True
            if projection:
                now = parse_time(cycle_time)
                if self.context_mode or self.evidence_only_mode:
                    if now < context_clock:
                        raise ValueError("Decision clock cannot precede the preceding actual return or creation")
                    context_clock = now
                cutoff = parse_time(state["prediction_cutoff"])
                expired = expired or now > cutoff
                # Future-dated provider content must not leak even through
                # its links. It remains in the audit rather than actor memory.
                public["documents"] = [d for d in public["documents"]
                                       if parse_time(d["available_at"]) <= min(now, cutoff)]
                public["evidence"] = build_match_evidence(
                    projection["match"], public["documents"], as_of=min(now, cutoff),
                    version=projection["version"])
            if self.context_mode:
                if semantic_prediction_mode and not expired:
                    public["prediction_input_at"] = cycle_time
                    prepared = prepare_prediction(public)
                    last_prediction_processing = processing_receipt(prepared, cycle_time)
                    public["semantic_prediction"] = {k: deepcopy(v) for k, v in prepared.items() if k != "prediction"}
                    public["decision_at"] = max(parse_time(cycle_time), parse_time(prepared["available_at"])).isoformat()
                    cycle_time = public["decision_at"]
                    context_clock = parse_time(cycle_time)
                    semantic_halt_reason = prepared["reason"] if prepared["status"] != "ready" else None
                    expired = semantic_halt_reason is not None
                    public["prediction"] = deepcopy(last_forecast) if expired else prepared["prediction"]
                else:
                    public["prediction"] = (deepcopy(last_forecast) if expired else self.policy.predict_public(public))
                    if semantic_prediction_mode:
                        last_prediction_processing = processing_receipt({"status": "not_updated",
                            "reason": "forecast_cutoff_elapsed", "available_at": forecast_at}, cycle_time)
                if semantic_prediction_mode:
                    public["prediction_processing"] = deepcopy(last_prediction_processing)
                last_forecast = deepcopy(public["prediction"])
                if not expired:
                    forecast_at = cycle_time
            if self.predictor_only_mode:
                before_context_timing = record_context("before_action", state, documents=documents,
                    prediction_before=public["prediction"]["probabilities"])
            actions = [] if expired else _actions(propose(deepcopy(public)))
            context_projector = getattr(propose, "candidate_contexts", None)
            if callable(context_projector):
                context = context_projector(deepcopy(public), deepcopy(actions))
                if context is not None:
                    # Only opted-in generators add this pre-query sidecar. It
                    # is recorded/bound with public_before, not passed as new
                    # features to the frozen numerical model or action payload.
                    public["candidate_contexts"] = deepcopy(context)
            for action in actions:
                aid = action["action_id"]
                # Prior vectors are explicit decision-local metadata.  They
                # may be recomputed from the current public state, while the
                # request identity and declared cost remain immutable.
                if aid in bindings and bindings[aid] != strip_prior_features(action):
                    raise ValueError("A generated action changed its bound request or cost")
                bindings[aid] = strip_prior_features(action)
            # Stable information keys prevent re-opening an attempted source with a new ID.
            available = [a for a in actions if a["information_key"] not in used_keys]
            if self.context_mode or self.evidence_only_mode:
                decision = (self.policy.expired_plan(public) if expired else self.policy.plan_public(
                    public, available, {"limits": limits, "spent": spent}, candidate_generation_complete=False))
                if semantic_halt_reason:
                    decision["selection"]["reason"] = semantic_halt_reason
                    decision["decision_digest"] = digest({k: v for k, v in decision.items() if k != "decision_digest"})
            else:
                decision = self.policy.plan(state, available, {"limits": limits, "spent": spent},
                                            candidate_generation_complete=False)
            candidate_by_id = {c["action_id"]: c for c in decision.get("candidates", [])}
            chosen = decision["selection"].get("candidate_id")
            basis, exploration_record = "computed_positive_voi", None
            training_exploration_record = None
            if risk_model is not None or self.context_mode:
                basis = "computed_positive_empirical_gain"
            excluded, eligible = {}, []
            if training_exploration is not None:
                # This is the only opt-in path that may replace the normal
                # deployment selector.  It receives the complete feasible
                # pre-I/O menu and current estimates, then samples with
                # positive propensity so future outcomes can label every
                # executed action.  It never calls a provider or f.
                training_menu, training_scores = [], {}
                if learning_count >= self.max_exploration_actions:
                    chosen, basis = None, "training_collection_allowance_exhausted"
                for action in ([] if learning_count >= self.max_exploration_actions else available):
                    candidate = candidate_by_id.get(action["action_id"])
                    if candidate is None or not candidate.get("feasible"):
                        excluded[action["action_id"]] = (candidate or {}).get("reason", "not_feasible")
                        continue
                    if any(spent[u] + candidate["cost"][u] > cap + 1e-12
                           for u, cap in limits.items() if u in candidate["cost"]):
                        excluded[action["action_id"]] = "resource_budget_exhausted"
                        continue
                    training_menu.append({"action_id": action["action_id"],
                                          "information_key": action["information_key"],
                                          "cost": deepcopy(action.get("cost", {}))})
                    value = candidate.get("value") or {}
                    estimate = value.get("expected_signed_gain", value.get("expected_gain"))
                    try:
                        estimate = float(estimate)
                    except (TypeError, ValueError):
                        estimate = 0.0
                    if not math.isfinite(estimate):
                        estimate = 0.0
                    training_scores[action["action_id"]] = estimate
                if training_menu:
                    training_exploration_record = training_exploration.select_for_collection(
                        training_menu, training_scores)
                    chosen = training_exploration_record.get("selected_action_id")
                    propensities = {
                        item["action_id"]: item["propensity"]
                        for item in training_exploration_record.get("candidate_menu", [])
                        if item.get("propensity") is not None
                    }
                    exploration_record = {
                        "probabilities": propensities,
                        "probability_basis": "training_epsilon_greedy_pre_io_menu",
                        "training_adapter": deepcopy(training_exploration_record),
                    }
                    basis = "training_collection_epsilon_greedy"
                elif learning_count < self.max_exploration_actions:
                    chosen, basis = None, "training_collection_no_feasible_candidate"
            elif self.policy_arm == "voi" and decision["selection"]["decision"] != "acquire":
                for candidate in decision["candidates"]:
                    aid = candidate["action_id"]
                    reason = None
                    if not candidate["feasible"]:
                        reason = candidate["reason"]
                    elif candidate["value"]["status"] == COMPUTED:
                        reason = "computed_nonpositive_value"
                    elif aid in MODELED_GROUPS:
                        # An invalid/unsupported numerical model is not an open-web route.
                        reason = "modeled_response_support_unavailable"
                    elif learning_count >= self.max_exploration_actions:
                        reason = "exploration_action_allowance_exhausted"
                    elif any(exploration_spent[u]+candidate["cost"][u] > cap+1e-12
                             for u, cap in self.exploration_limits.items()):
                        reason = "exploration_resource_allowance_exhausted"
                    if reason:
                        excluded[aid] = reason
                    else:
                        eligible.append(aid)
                if eligible:
                    response = {"decision": {"kind": "query", "candidate_id": sorted(eligible)[0],
                                              "revisit_at": None},
                                "candidates": [{"candidate_id": a["action_id"]} for a in available]}
                    draw = self.sampler.select(run, response, eligible)
                    chosen, exploration_record = draw["candidate_id"], draw["exploration_record"]
                    basis = "bounded_unknown_information_exploration"
                else:
                    chosen, basis = None, "no_executable_selection"
            elif self.policy_arm == "random_explore":
                # Randomization is over the actual generated menu after
                # stable-key de-duplication and feasibility checks.  It never
                # claims a numeric VoI and uses the existing precommitted
                # UniformInquiryExploration draw.
                if arm_query_count >= self.max_exploration_actions:
                    basis = "random_explore_allowance_exhausted"
                    chosen = None
                else:
                    for action in available:
                        candidate = candidate_by_id.get(action["action_id"])
                        if candidate is None or not candidate.get("feasible"):
                            excluded[action["action_id"]] = (candidate or {}).get("reason", "not_feasible")
                            continue
                        if any(spent[u] + candidate["cost"][u] > cap + 1e-12
                               for u, cap in limits.items() if u in candidate["cost"]):
                            excluded[action["action_id"]] = "resource_budget_exhausted"
                            continue
                        eligible.append(action["action_id"])
                    if eligible:
                        response = {"decision": {"kind": "query", "candidate_id": eligible[0],
                                                  "revisit_at": None},
                                    "candidates": [{"candidate_id": a["action_id"]} for a in available]}
                        draw = self.sampler.select(run, response, eligible)
                        chosen, exploration_record = draw["candidate_id"], draw["exploration_record"]
                        basis = "random_explore_predeclared_uniform"
                    else:
                        chosen, basis = None, "random_explore_no_feasible_candidate"
            elif self.policy_arm in {"fixed_order", "full_probe"}:
                if arm_query_count >= self.max_exploration_actions:
                    chosen, basis = None, self.policy_arm + "_allowance_exhausted"
                else:
                    for action in available:
                        candidate = candidate_by_id.get(action["action_id"])
                        if candidate is None or not candidate.get("feasible"):
                            excluded[action["action_id"]] = (candidate or {}).get("reason", "not_feasible")
                            continue
                        if any(spent[u] + candidate["cost"][u] > cap + 1e-12
                               for u, cap in limits.items() if u in candidate["cost"]):
                            excluded[action["action_id"]] = "resource_budget_exhausted"
                            continue
                        eligible.append(action["action_id"])
                    chosen = eligible[0] if eligible else None
                    basis = (self.policy_arm + "_predeclared_menu_order" if chosen is not None
                             else self.policy_arm + "_no_feasible_candidate")
            selected = next((a for a in available if a["action_id"] == chosen), None)
            probabilities = {}
            probability_basis = None
            if exploration_record is not None and self.propensity_logging:
                probabilities = deepcopy(exploration_record.get("probabilities", {}))
                probability_basis = exploration_record.get("probability_basis")
            elif (self.policy_arm_version == "footballmind-policy-arm-v2"
                  and self.policy_arm in {"fixed_order", "full_probe"}
                  and self.propensity_logging and selected is not None):
                # Degenerate distribution of the actually executed deterministic
                # selector, conditional on this exposed menu. This is not a
                # search engine's probability of exposing a document.
                probabilities = {a["action_id"]: float(a["action_id"] == chosen) for a in actions}
                probability_basis = "deterministic_predeclared_order_conditional_on_observed_menu"
            menu = []
            for menu_order, action in enumerate(actions):
                aid = action["action_id"]
                candidate = candidate_by_id.get(aid)
                if aid == chosen:
                    reason = None
                elif action["information_key"] in used_keys:
                    reason = "information_key_already_used"
                elif aid in excluded:
                    reason = excluded[aid]
                elif chosen is None:
                    reason = "policy_did_not_select"
                elif self.policy_arm in {"fixed_order", "full_probe"}:
                    reason = "earlier_feasible_candidate_in_predeclared_order"
                elif exploration_record is not None:
                    reason = "random_exploration_not_drawn"
                else:
                    reason = "not_selected_by_policy"
                menu.append({"menu_order": menu_order, "action_id": aid,
                             "selected": aid == chosen,
                             "selection_probability": probabilities.get(aid),
                             "selection_probability_basis": probability_basis,
                             "not_selected_reason": reason,
                             "feasible": candidate.get("feasible") if candidate else None})
            cycle = {"step": step, "public_before": public, "generated_actions": actions,
                     "strict_decision": decision, "selection_basis": basis,
                     "selected_action": selected, "exploration_record": exploration_record,
                     "exploration_exclusions": excluded, "execution_status": "not_requested",
                     "prediction_before": public["prediction"]["probabilities"],
                     "prediction_after": None, "admission": None, "update": None}
            if training_exploration_record is not None:
                cycle["training_exploration"] = deepcopy(training_exploration_record)
            if self.trajectory_enabled:
                cycle.update(policy_arm=self.policy_arm, policy_arm_version=self.policy_arm_version,
                    arm_assigned_at=assigned_at, decision_at=cycle_time,
                    menu_observed_at=cycle_time, menu=menu,
                    selection_distribution=(probabilities or None) if self.propensity_logging else None,
                    selected_selection_probability=probabilities.get(chosen),
                    selection_probability_basis=probability_basis,
                    before_state=deepcopy(state),
                    before_state_timing=(before_context_timing
                                         or _state_timing(state, observed_at=cycle_time)),
                    after_state=None, after_state_timing=None)
            if self.trajectory_enabled and trajectory_recorder is not None:
                # This callback is intentionally before any provider I/O.  The
                # callback may use the existing RawArchive, but cannot mutate
                # the action selection or model state.
                raw_ref = trajectory_recorder({
                    "schema": "prospective-menu-archive-v1",
                    "run_id": run_id,
                    "step": step,
                    "observed_at": cycle_time,
                    "policy_arm": self.policy_arm,
                    "policy_arm_version": self.policy_arm_version,
                    "public_before": deepcopy(public),
                    "generated_actions": deepcopy(actions),
                    "menu": deepcopy(menu),
                })
                if raw_ref is not None:
                    if not isinstance(raw_ref, dict):
                        raise ValueError("trajectory_recorder must return a mapping or None")
                    cycle.update({k: raw_ref[k] for k in
                                  ("menu_raw_path", "menu_raw_sha256", "menu_raw_observation_id")
                                  if k in raw_ref})
            if selected is None:
                terminal = ({"decision": "halt", "reason": semantic_halt_reason or "forecast_cutoff_elapsed", "stop_certificate": None}
                            if expired else ({"decision": "halt", "reason": "kickoff_elapsed", "stop_certificate": None}
                                             if operational_run and cycle_time is not None and parse_time(cycle_time) >= kickoff_at
                                             else ({"decision": "halt", "reason": basis, "stop_certificate": None}
                                                   if self.policy_arm != "voi" else deepcopy(decision["selection"]))))
                cycle["terminal"] = terminal
                _save(output/f"cycle-{step:03}.json", cycle)
                cycles.append(cycle)
                break
            cost = _cost(selected)
            for u in UNITS:
                spent[u] += cost[u]
                if exploration_record is not None:
                    exploration_spent[u] += cost[u]
            _budget({"limits": limits, "spent": spent})
            if exploration_record is not None:
                learning_count += 1
                _budget({"limits": self.exploration_limits, "spent": exploration_spent})
            if self.policy_arm != "voi":
                arm_query_count += 1
            used_keys.add(selected["information_key"])
            run["decisions"].append({"action_id": chosen})
            cycle.update(execution_status="reserved_before_io", reserved_cost=cost,
                         public_digest=digest(public), request_digest=digest(selected),
                         reservation_spent=deepcopy(spent),
                         exploration_spent=deepcopy(exploration_spent))
            if self.context_mode or self.evidence_only_mode:
                cycle["exploration_limits"] = deepcopy(self.exploration_limits)
            if self.trajectory_enabled:
                cycle["attempt_id"] = f"{run_id}:attempt:{step}"
            _save(output/f"cycle-{step:03}.json", cycle)
            try:
                returned = execute(deepcopy(selected), deepcopy(public))
            except Exception as exc:
                # I/O may have happened. Preserve reservation, avoid an automatic retry.
                cycle.update(execution_status="execution_outcome_unknown", error_type=type(exc).__name__)
                terminal = {"decision": "halt", "reason": "execution_reconciliation_required",
                            "stop_certificate": None}
                _save(output/f"cycle-{step:03}.json", cycle)
                cycles.append(cycle)
                break
            # Preserve the actual local response clock for every arm.  It is
            # distinct from source/document availability and is never used to
            # backfill an unknown available_at.
            if self.trajectory_enabled or self.text_mode or self.context_mode or self.evidence_only_mode:
                cycle["returned_at"] = clock().isoformat()
            if self.trajectory_enabled:
                if self.legacy_receipt_clock:
                    cycle["response_observed_at"] = cycle["returned_at"]
                else:
                    cycle["result_observed_at"] = cycle["returned_at"]
                    cycle["receipt_available_at"] = cycle["returned_at"]
                    cycle["receipt_timing_basis"] = "measured_runtime_execute_return"
                    observed_receipt = (returned.get("receipt") or {}) if isinstance(returned, dict) else {}
                    cycle["response_observed_at"] = (observed_receipt.get("observation") or {}).get("observed_at")
            _save(output/f"receipt-{step:03}.json", returned)
            try:
                if (self.evidence_only_mode or self.official_context_mode) and parse_time(cycle["returned_at"]) < parse_time(public["decision_at"]):
                    raise ValueError("The actual return clock precedes its request")
                late_boundary = None
                if self.safe_predictor_context:
                    returned_time = parse_time(cycle["returned_at"])
                    if operational_run and returned_time >= kickoff_at:
                        late_boundary = "kickoff_elapsed"
                    elif returned_time > parse_time(state["prediction_cutoff"]):
                        late_boundary = "forecast_cutoff_elapsed"
                if operational_run and self.trajectory_enabled and parse_time(cycle["returned_at"]) >= kickoff_at and not self.safe_predictor_context:
                    raise ValueError("The actual response was observed at or after kickoff")
                if not isinstance(returned, dict) or returned.get("status") not in {"OK", "FAILED", "NO_DATA"}:
                    raise ValueError("Invalid provider receipt")
                actual = returned.get("actual_cost")
                if not isinstance(actual, dict) or set(actual) != set(UNITS):
                    raise ValueError("Actual resources require explicit values or null for every unit")
                for u, amount in actual.items():
                    if amount is not None:
                        _cost({"cost_status": "known", "cost": {"money": 0., "requests": 0., u: amount}})
                cycle["actual_cost"] = deepcopy(actual)
                resource_overrun = False
                for u, amount in actual.items():
                    if amount is None:
                        unmeasured[u] += 1
                        if u == "requests":
                            raise ValueError("Actual request count must be reconciled")
                    else:
                        measured[u] += amount
                        # Money credits can be simulated; never equate them with bills.
                        if u != "money" and (u in limits or (self.context_mode or self.evidence_only_mode) and exploration_record is not None and u in self.exploration_limits) and amount > cost[u]:
                            extra = amount-cost[u]
                            spent[u] += extra
                            if exploration_record is not None:
                                exploration_spent[u] += extra
                resource_overrun = any(spent[u] > cap+1e-12 for u, cap in limits.items()) or any(
                    exploration_spent[u] > cap+1e-12 for u, cap in self.exploration_limits.items())
                old = deepcopy(state)
                after = deepcopy(state)
                after["action_history"].append({"action_id": chosen})
                admitted_document = None
                if late_boundary is not None:
                    admission = {"status": "not_admitted", "reason": (
                        "after_kickoff" if late_boundary == "kickoff_elapsed" else "after_information_cutoff")}
                elif returned["status"] != "OK":
                    admission = {"status": "not_admitted", "reason": "provider_"+returned["status"].lower()}
                elif chosen in MODELED_GROUPS:
                    after = deepcopy(returned["state_after"])
                    cycle["update"] = self.policy.verify_update(decision, old, after)
                    admission = {"status": "admitted", "reason": "modeled_actual_return_verified"}
                else:
                    if "state_after" in returned:
                        raise ValueError("An unmodeled return cannot alter the probability-model state")
                    document = returned.get("document")
                    if not isinstance(document, dict) or not document.get("document_id"):
                        raise ValueError("A document with source identity is required")
                    observed, available_at = parse_time(document["observed_at"]), parse_time(document["available_at"])
                    if observed > available_at:
                        raise ValueError("Evidence availability precedes observation")
                    if not re.fullmatch(r"[0-9a-f]{64}", str(document.get("raw_sha256", ""))):
                        raise ValueError("Archived response digest required")
                    if available_at > parse_time(state["prediction_cutoff"]) or ((self.evidence_only_mode or self.official_context_mode) and parse_time(cycle["returned_at"]) > parse_time(state["prediction_cutoff"])):
                        admission = {"status": "not_admitted", "reason": "after_information_cutoff"}
                    elif (self.text_mode or self.context_mode or self.evidence_only_mode) and available_at > parse_time(cycle["returned_at"]):
                        admission = {"status": "not_admitted", "reason": "after_actual_return_clock"}
                    elif document.get("status") != "OK":
                        admission = {"status": "not_admitted", "reason": "document_not_usable"}
                    else:
                        admitted_document = deepcopy(document)
                        admission = {"status": "admitted", "reason": "unmodeled_document_for_replanning_only"}
                if self.text_mode:
                    after, cycle["update"] = self.policy.apply_return(
                        decision, old, selected, returned, admitted_document)
                    if admitted_document is not None:
                        admission["reason"] = ("learned_text_return_verified" if
                            cycle["update"]["status"] == "updated" else "text_return_support_unknown")
                checked = self.policy.public_state(after) if self.text_mode or self.context_mode or self.evidence_only_mode else sanitize_state(after)
                checked["action_history"] = deepcopy(after["action_history"])
                forecast = self.policy.model.predict_partial(checked)["probabilities"]
                state = checked
                if admitted_document is not None:
                    documents.append(admitted_document)
                cycle.update(execution_status="returned", admission=admission,
                             public_state_after=deepcopy(state),
                             prediction_after=forecast, charged_cost=deepcopy(spent),
                             resource_overrun=resource_overrun)
                if self.trajectory_enabled and not self.predictor_only_mode:
                    cycle.update(after_state=deepcopy(state), after_state_timing=(
                        record_context("after_action", state, parent_attempt_id=cycle["attempt_id"], documents=documents)
                        or _state_timing(state, observed_at=cycle["returned_at"],
                                         parent_attempt_id=cycle["attempt_id"])))
                attempts.append({"action_id": chosen, "information_key": selected["information_key"],
                                 "status": returned["status"], "admission": admission["status"]})
                if self.trajectory_enabled:
                    # A policy decision precedes transport preparation and
                    # archival work; it is not the network request clock.
                    # Preserve only the existing transport's measured clock.
                    observation = (returned.get("receipt") or {}).get("observation") or {}
                    request_time = (observation.get("context") or {}).get("requested_at")
                    if request_time is not None:
                        parse_time(request_time)
                    cycle["requested_at"] = request_time
                    cycle["request_started_at"] = request_time
                    cycle["request_timing_basis"] = (
                        "transport_observation_context" if request_time is not None else "not_recorded")
                    attempts[-1].update(attempt_id=cycle["attempt_id"],
                        decision_at=public["decision_at"], requested_at=request_time,
                        request_started_at=request_time,
                        request_timing_basis=cycle["request_timing_basis"],
                        response_observed_at=cycle["response_observed_at"])
                if self.evidence_only_mode:
                    context_clock = parse_time(cycle["returned_at"])
                    cycle["exploration_charged_cost"] = deepcopy(exploration_spent)
                    visible_at = min(context_clock, parse_time(state["prediction_cutoff"]))
                    post_documents = [deepcopy(document) for document in documents
                        if parse_time(document["available_at"]) <= visible_at]
                    post = {"state": deepcopy(state), "documents": post_documents,
                        "attempts": deepcopy(attempts), "prediction": self.policy.predict_partial(state),
                        "budget": {"limits": limits, "spent": deepcopy(spent)},
                        "exploration": self.sampler.public_summary(), "decision_at": cycle["returned_at"]}
                    post["evidence"] = build_match_evidence(projection["match"], post_documents,
                        as_of=visible_at, version=projection["version"])
                    cycle["public_after"] = post
                    cycle["update"] = {"status": "evidence_only", "reason": "prior_unavailable",
                        "model_version": None, "probability_changed": False,
                        "evidence_admitted": admitted_document is not None, "as_of": cycle["returned_at"]}
                    if admitted_document is not None:
                        admission["reason"] = "evidence_only_observed_return"
                if self.context_mode:
                    if parse_time(cycle["returned_at"]) < parse_time(public["decision_at"]):
                        raise ValueError("The actual return clock precedes its request")
                    cycle["exploration_charged_cost"] = deepcopy(exploration_spent)
                    context_clock = parse_time(cycle["returned_at"])
                    cycle["public_after"] = None
                    if not resource_overrun and late_boundary is None and parse_time(cycle["returned_at"]) <= parse_time(state["prediction_cutoff"]):
                        post = {"state": deepcopy(state), "documents": deepcopy(documents),
                            "attempts": deepcopy(attempts), "prediction": deepcopy(last_forecast),
                            "budget": {"limits": limits, "spent": deepcopy(spent)},
                            "exploration": self.sampler.public_summary(), "decision_at": cycle["returned_at"]}
                        post["evidence"] = build_match_evidence(projection["match"], post["documents"],
                            as_of=post["decision_at"], version=projection["version"])
                        if semantic_prediction_mode:
                            post["prediction_input_at"] = cycle["returned_at"]
                            prepared = prepare_prediction(post)
                            last_prediction_processing = processing_receipt(prepared, cycle["returned_at"])
                            post["prediction_processing"] = deepcopy(last_prediction_processing)
                            cycle["semantic_prediction"] = {k: deepcopy(v) for k, v in prepared.items() if k != "prediction"}
                            post["decision_at"] = max(context_clock, parse_time(prepared["available_at"])).isoformat()
                            context_clock = parse_time(post["decision_at"])
                            if prepared["status"] == "ready":
                                post["prediction"] = prepared["prediction"]
                            else:
                                semantic_halt_reason = prepared["reason"]
                        else:
                            post["prediction"] = self.policy.predict_public(post)
                        if semantic_halt_reason is None:
                            last_forecast = deepcopy(post["prediction"])
                            forecast_at = post["decision_at"]
                            cycle["public_after"] = post
                            cycle["prediction_after"] = post["prediction"]["probabilities"]
                            cycle["update"] = {"status": "updated", "model_version": self.policy.predictor.model_version,
                                "probability_changed": cycle["prediction_before"] != cycle["prediction_after"],
                                "basis": "actual_return_visible_context_including_failed_attempts",
                                "as_of": post["decision_at"]}
                            if admitted_document is not None:
                                admission["reason"] = "frozen_context_predictor_observed_return"
                        else:
                            cycle["prediction_after"] = deepcopy(last_forecast["probabilities"])
                            cycle["update"] = {"status": "not_updated", "reason": semantic_halt_reason,
                                "model_version": self.policy.predictor.model_version, "probability_changed": False,
                                "prediction_retained": True, "forecast_as_of": forecast_at,
                                "as_of": post["decision_at"]}
                    else:
                        cycle["prediction_after"] = deepcopy(last_forecast["probabilities"])
                        cycle["update"] = {"status": "not_updated", "reason": "after_kickoff" if late_boundary == "kickoff_elapsed" else "after_information_cutoff" if parse_time(cycle["returned_at"]) > parse_time(state["prediction_cutoff"]) else "actual_resource_budget_overrun",
                            "probability_changed": False, "as_of": cycle["returned_at"]}
                        if self.safe_predictor_context:
                            cycle["update"].update(model_version=self.policy.predictor.model_version,
                                forecast_as_of=forecast_at, prediction_retained=True)
                        if semantic_prediction_mode:
                            last_prediction_processing = processing_receipt({"status": "not_updated",
                                "reason": cycle["update"]["reason"], "available_at": forecast_at}, cycle["returned_at"])
                    if semantic_prediction_mode:
                        cycle["prediction_processing"] = deepcopy(last_prediction_processing)
                if self.predictor_only_mode:
                    cycle.update(after_state=deepcopy(state), after_state_timing=(
                        record_context("after_action", state, parent_attempt_id=cycle["attempt_id"],
                            documents=documents, prediction_before=cycle["prediction_before"],
                            prediction_after=(cycle["prediction_after"]
                                if self.safe_predictor_context or cycle.get("update", {}).get("status") == "updated" else None))
                        or _state_timing(state, observed_at=cycle["returned_at"],
                                         parent_attempt_id=cycle["attempt_id"])))
            except (KeyError, TypeError, ValueError) as exc:
                cycle.update(execution_status="receipt_rejected", error_type=type(exc).__name__)
                terminal = {"decision": "halt", "reason": "receipt_reconciliation_required",
                            "stop_certificate": None}
                _save(output/f"cycle-{step:03}.json", cycle)
                cycles.append(cycle)
                break
            _save(output/f"cycle-{step:03}.json", cycle)
            cycles.append(cycle)
            if semantic_halt_reason is not None:
                terminal = {"decision": "halt", "reason": semantic_halt_reason, "stop_certificate": None}
                break
            if late_boundary is not None:
                terminal = {"decision": "halt", "reason": late_boundary, "stop_certificate": None}
                break
            if (self.context_mode or self.evidence_only_mode) and parse_time(cycle["returned_at"]) > parse_time(state["prediction_cutoff"]):
                terminal = {"decision": "halt", "reason": "forecast_cutoff_elapsed", "stop_certificate": None}
                break
            if resource_overrun:
                terminal = {"decision": "halt", "reason": "actual_resource_budget_overrun", "stop_certificate": None}
                break
        result = {"schema": version, "run_id": run_id, "configuration_digest": digest(config),
                  "cycles": cycles, "documents": documents, "attempts": attempts,
                  "final_state": state, "terminal_selection": terminal,
                  "reserved_cost": spent, "exploration_reserved_cost": exploration_spent,
                  "measured_resources": measured, "unmeasured_resource_receipts": unmeasured,
                  "exploration_actions": learning_count,
                  "initial_probabilities": cycles[0]["prediction_before"],
                  "final_probabilities": last_forecast["probabilities"] if self.context_mode else self.policy.model.predict_partial(state)["probabilities"],
                  "outcome": None, "numeric_voi_training_eligible": False,
                  "independent_outcome_verified": False,
                  "unmodeled_return_probability_update": False,
                  "open_world_policy_benefit_proven": False}
        if self.trajectory_enabled:
            result.update(trajectory_schema=config["trajectory_schema"],
                policy_arm=self.policy_arm, policy_arm_version=self.policy_arm_version,
                arm_assigned_at=assigned_at, random_seed=self.random_seed,
                propensity_logging=self.propensity_logging, created_at=created_at)
        if projection:
            result["finished_at"] = clock().isoformat()
            if (self.context_mode or self.evidence_only_mode) and parse_time(result["finished_at"]) < context_clock:
                raise ValueError("Finished clock cannot precede the last actual decision or return")
            result["evidence"] = build_match_evidence(
                projection["match"], documents,
                as_of=min(parse_time(result["finished_at"]), parse_time(state["prediction_cutoff"])),
                version=projection["version"])
        if self.evidence_only_mode:
            result.update(analysis_mode="evidence_only", prior_status="prior_unavailable",
                prediction_model=None, expected_gain=None, net_voi=None, numeric_confidence=None)
        if self.official_context_mode:
            result.update(analysis_mode="official_optional_context", prior_status="prior_unavailable",
                prior_probabilities=None, initial_prediction=deepcopy(cycles[0]["public_before"]["prediction"]),
                final_prediction=deepcopy(last_forecast),
                forecast_basis=last_forecast.get("forecast_basis"),
                base_probabilities=deepcopy(last_forecast.get("base_probabilities")))
        if self.context_mode:
            updates = [c["update"] for c in cycles if c.get("update")]
            result["context_response_learning"] = {
                "predictor_version": self.policy.predictor.model_version,
                "value_model_version": self.policy.value_model.model_version if self.policy.value_model else None,
                "evidence_origin": self.policy.predictor.metadata["evidence_origin"],
                "probability_updates": sum(u["probability_changed"] for u in updates),
                "value_scope": "conditional_empirical_signed_log_loss_gain",
                "after_clock_scope": "actual_return_before_next_decision",
                "production_ready": False, "real_world_benefit_proven": False}
        if self.text_mode:
            updates = [c["update"] for c in cycles if c.get("execution_status") == "returned" and c.get("update")]
            result["text_response_learning"] = {
                "model_version": self.policy.text_model.model_version,
                "evidence_origin": self.policy.text_model.metadata["evidence_origin"],
                "modeled_returns": sum(u["status"] == "updated" for u in updates),
                "unknown_returns": sum(u["status"] == "unknown" for u in updates),
                "probability_updates": sum(u["probability_changed"] for u in updates),
                "query_and_update_share_response_model": True,
                "calibration_status": "not_established", "production_ready": False}
            if risk_model is not None:
                result["text_response_learning"]["query_and_update_share_response_model"] = False
                result["text_risk_learning"] = {
                    "model_version": risk_model.model_version,
                    "evidence_origin": risk_model.manifest["evidence_origin"],
                    "prediction_model_version": self.policy.text_model.model_version,
                    "value_scope": "empirical_fixed_predictor_log_loss_reduction",
                    "calibration_status": "empirical_residual_fit_no_coverage_guarantee",
                    "production_ready": False, "real_world_benefit_proven": False}
            if discovery_model is not None:
                discovery_selections = []
                for cycle in cycles:
                    if cycle["selected_action"] is None or cycle["selection_basis"] != "computed_positive_voi":
                        continue
                    selected = next(c for c in cycle["strict_decision"]["candidates"]
                                    if c["action_id"] == cycle["selected_action"]["action_id"])
                    immediate = selected["response_enumeration"]["value"]["net_voi"]
                    if immediate is not None and immediate <= 1e-12 and selected["value"]["net_voi"] > 1e-12:
                        discovery_selections.append(cycle["step"])
                result["discovery_response_learning"] = {
                    "model_version": discovery_model.model_version,
                    "evidence_origin": discovery_model.metadata["evidence_origin"],
                    "planning_horizon": 2,
                    "continuation_enabled_selections": len(discovery_selections),
                    "continuation_enabled_steps": discovery_selections,
                    "execution_scope": "actual_generated_actions_only",
                    "value_scope": "model_conditional_bounded_two_step",
                    "open_world_optimality_proven": False,
                    "calibration_status": "not_established", "production_ready": False}
        if context_recorder is not None:
            result["final_state_timing"] = record_context("final", state, documents=documents,
                prediction_after=last_forecast["probabilities"] if self.predictor_only_mode else None)
        if semantic_prediction_mode:
            result["semantic_extraction_archive"] = self.policy.finalize_runtime_archive()
        _save(output/"run.json", result)
        return result
