"""Opt-in prospective shadow for frozen entity coefficients, with no g authority.

The training prior was reconstructed Poisson. This *new* policy explicitly
uses an observed market prior; it is a different research pipeline, not a
promotion of the historical predictor. Source assertions are extracted only
after visibility, frozen once, and replayed without calling the provider.
"""
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from jsonschema import ValidationError

from football_research.acquisition.entity_semantic_f import (
    EntitySemanticF, EntityEvidenceBank, REPRESENTATION_SHA256, _predict,
)
from football_research.acquisition.historical_context_training import _project
from football_research.acquisition.news_link_projection import VERSION as NEWS_PROJECTION
from football_research.acquisition.semantic_f import SourceAssertionProtocol, validate_response
from football_research.acquisition.state import parse_time
from .context_voi_policy import ContextVOIPolicy, _digest
from .historical_context_policy import HistoricalContextVOIPolicy

VERSION = "entity-semantic-market-shadow-f-only-policy-v1"
CONFIG_SCHEMA = "entity-semantic-market-shadow-config-v1"
TRACE_SCHEMA = "entity-semantic-shadow-extraction-trace-v1"
REQUEST_BOUND_VERSION = "entity-semantic-market-shadow-f-only-policy-v2"
REQUEST_BOUND_CONFIG_SCHEMA = "entity-semantic-market-shadow-config-v2"
REQUEST_BOUND_TRACE_SCHEMA = "entity-semantic-shadow-extraction-trace-v2"
TYPED_REQUEST_VERSION = "entity-semantic-market-shadow-f-only-policy-v3"
TYPED_REQUEST_CONFIG_SCHEMA = "entity-semantic-market-shadow-config-v3"
TYPED_REQUEST_TRACE_SCHEMA = "entity-semantic-shadow-extraction-trace-v3"
# v4 is an explicit, bounded repair mode for provider validation failures.
# v1-v3 remain byte/behavior compatible and never repair silently.
REPAIR_REQUEST_VERSION = "entity-semantic-market-shadow-f-only-policy-v4"
REPAIR_REQUEST_CONFIG_SCHEMA = "entity-semantic-market-shadow-config-v4"
REPAIR_REQUEST_TRACE_SCHEMA = "entity-semantic-shadow-extraction-trace-v4"
TRACE_FILE = "entity-semantic-extractions.json"
PRIOR_BASIS = "observed_market_prior_research_only"
OPERATIONAL_BODY_ADMISSION = "observed-body-availability-v1"


def _is_typed_validation_error(exc):
    """Return true only for host/schema/semantic validation failures.

    jsonschema raises its own ValidationError for an invalid kind/value pair,
    while the request binding and quote checks raise ValueError.  Both are
    validation failures eligible for the explicitly bounded v4 repair.  A
    transport timeout or provider exception remains outside this predicate and
    is never retried by the repair path.
    """
    return isinstance(exc, (ValueError, ValidationError))


def validate_config(spec, artifact):
    base_fields = {"schema", "model_sha256", "frozen_at", "prior_basis", "provider", "max_calls"}
    schemas = (CONFIG_SCHEMA, REQUEST_BOUND_CONFIG_SCHEMA, TYPED_REQUEST_CONFIG_SCHEMA,
               REPAIR_REQUEST_CONFIG_SCHEMA)
    if not isinstance(spec, dict) or spec.get("schema") not in schemas:
        raise ValueError("Explicit entity semantic shadow configuration required")
    fields = base_fields | ({"repair"} if spec["schema"] == REPAIR_REQUEST_CONFIG_SCHEMA else set())
    if set(spec) != fields:
        raise ValueError("Entity semantic shadow configuration fields are not exact")
    if spec["model_sha256"] != sha256(artifact).hexdigest() or spec["prior_basis"] != PRIOR_BASIS:
        raise ValueError("Entity shadow must bind exact f bytes and explicit market-prior shift")
    parse_time(spec["frozen_at"])
    if type(spec["max_calls"]) is not int or not 1 <= spec["max_calls"] <= 256:
        raise ValueError("A bounded semantic extraction allowance (1..256) is required")
    provider = spec["provider"]
    if (not isinstance(provider, dict) or set(provider) !=
            {"kind", "model", "thinking", "max_tokens", "timeout_seconds"}
            or provider["kind"] != "deepseek_api"
            or not isinstance(provider["model"], str) or not provider["model"]
            or provider["thinking"] not in {"enabled", "disabled"}
            or type(provider["max_tokens"]) is not int or not 1 <= provider["max_tokens"] <= 32768
            or type(provider["timeout_seconds"]) not in (int, float)
            or not 1 <= provider["timeout_seconds"] <= 180):
        raise ValueError("A complete explicit DeepSeek extraction configuration is required")
    if spec["schema"] == REPAIR_REQUEST_CONFIG_SCHEMA:
        repair = spec["repair"]
        if (not isinstance(repair, dict)
                or set(repair) != {"enabled", "max_attempts", "trigger", "prompt_version"}
                or repair["enabled"] is not True
                or type(repair["max_attempts"]) is not int or repair["max_attempts"] != 1
                or repair["trigger"] != "typed_validation_failure_only"
                or repair["prompt_version"] != "request-bound-football-source-assertions-v3-repair-v1"):
            raise ValueError("v4 requires exactly one explicit typed validation repair attempt")
    return deepcopy(spec)


class EntitySemanticFOnlyPolicy(ContextVOIPolicy):
    predictor_only = True
    text_projection = NEWS_PROJECTION
    # The live source gateway already proves available_at <= decision_at.
    # Do not discard a real body merely because the publisher omitted a date;
    # retain source_published_at=null and expose the timing basis in the
    # projected document for auditability.
    allow_unknown_publication_clock = True
    _visible = HistoricalContextVOIPolicy._visible

    def __init__(self, base_policy, model, *, fixture, artifact, semantic_config,
                 replay_trace=None, provider=None, semantic_clock=None,
                 operational_body_admission=True):
        if type(operational_body_admission) is not bool:
            raise ValueError("Explicit operational body admission boolean required")
        self.allow_unknown_publication_clock = operational_body_admission
        if not isinstance(model, EntitySemanticF) or base_policy.manifest["loss"] != "log_loss":
            raise ValueError("Frozen EntitySemanticF and log-loss market initialization required")
        if json.loads(artifact) != {**model.payload(), "model_version": model.model_version}:
            raise ValueError("Entity shadow artifact differs from the frozen predictor")
        meta = model.metadata
        if (meta.get("scope") != "A_purged_forward_validation"
                or meta.get("candidate_role") != "selection_capability_audit_only"
                or any(meta.get(k) is not False for k in
                       ("promotion_allowed", "production_ready", "gain_model_enabled"))):
            raise ValueError("Only the explicit A-forward research shadow may use this entrypoint")
        self.fixture = deepcopy(fixture)
        self.physical_match_id = fixture["physical_match_id"]
        if self.physical_match_id in meta["training_match_ids"]:
            raise ValueError("A training fixture cannot enter prospective shadow collection")
        self.base_policy, self.model, self.predictor = base_policy, base_policy.model, model
        self.predictor_bytes, self.predictor_sha256 = bytes(artifact), sha256(artifact).hexdigest()
        self.value_model = self.value_bytes = self.synthetic_prior = None
        self.lambda_cost = deepcopy(base_policy.lambda_cost)
        self.semantic_config = validate_config(semantic_config, artifact)
        self.request_bound = self.semantic_config["schema"] in (REQUEST_BOUND_CONFIG_SCHEMA, TYPED_REQUEST_CONFIG_SCHEMA)
        self.repair_enabled = self.semantic_config["schema"] == REPAIR_REQUEST_CONFIG_SCHEMA
        if self.repair_enabled:
            self.request_bound = True
        self.policy_schema = REQUEST_BOUND_VERSION if self.request_bound else VERSION
        self.trace_schema = REQUEST_BOUND_TRACE_SCHEMA if self.request_bound else TRACE_SCHEMA
        self.extraction_protocol = SourceAssertionProtocol
        if self.request_bound:
            from .semantic_request_binding import RequestBoundSourceAssertionProtocol
            self.extraction_protocol = RequestBoundSourceAssertionProtocol
        if self.semantic_config["schema"] == TYPED_REQUEST_CONFIG_SCHEMA:
            from .semantic_request_binding import TypedRequestBoundSourceAssertionProtocol
            self.policy_schema, self.trace_schema = TYPED_REQUEST_VERSION, TYPED_REQUEST_TRACE_SCHEMA
            self.extraction_protocol = TypedRequestBoundSourceAssertionProtocol
        if self.repair_enabled:
            from .semantic_request_binding import TypedRequestBoundSourceAssertionProtocol
            self.policy_schema, self.trace_schema = REPAIR_REQUEST_VERSION, REPAIR_REQUEST_TRACE_SCHEMA
            self.extraction_protocol = TypedRequestBoundSourceAssertionProtocol
            self.repair_protocol = __import__(
                "football_agent.semantic_request_binding", fromlist=["RepairTypedRequestBoundSourceAssertionProtocol"]
            ).RepairTypedRequestBoundSourceAssertionProtocol
        else:
            self.repair_protocol = None
        protocol = self.extraction_protocol
        self._frozen_prior = tuple(base_policy.manifest["market_initialization"]["probabilities"])
        if tuple(self.model.probabilities) != self._frozen_prior:
            raise ValueError("Market probabilities differ from the frozen initialization")
        # One f pipeline may be used across fixtures. The fixture and its
        # observed prior belong to the run binding, not the identity of f.
        self.prediction_pipeline_manifest = {
            "schema": "entity-semantic-market-prediction-pipeline-v1",
            "implementation_version": self.policy_schema,
            "text_projection": NEWS_PROJECTION,
            "evidence_version": "match-evidence-extraction-v5",
            "predictor_version": model.model_version,
            "predictor_sha256": self.predictor_sha256,
            "entity_representation_sha256": REPRESENTATION_SHA256,
            "extraction_protocol": {"version": protocol.version,
                "prompt_sha256": sha256(protocol.prompt.encode("utf-8")).hexdigest(),
                "schema_sha256": _digest(protocol.schema)},
            "provider": deepcopy(self.semantic_config["provider"]),
            "prior_basis": PRIOR_BASIS,
            "prior_domain_shift": "training_poisson_to_market_not_calibrated"}
        if self.repair_enabled:
            self.prediction_pipeline_manifest["semantic_repair"] = {
                "schema": REPAIR_REQUEST_CONFIG_SCHEMA,
                "trigger": self.semantic_config["repair"]["trigger"],
                "max_attempts": self.semantic_config["repair"]["max_attempts"],
                "prompt_version": self.semantic_config["repair"]["prompt_version"],
                "prompt_sha256": sha256(self.repair_protocol.prompt.encode("utf-8")).hexdigest(),
                "wire_schema_version": self.repair_protocol.version,
            }
        if self.request_bound:
            from .semantic_request_binding import PROJECTION_MANIFEST, BINDING_SCHEMA
            self.prediction_pipeline_manifest.update(
                semantic_input_projection=deepcopy(PROJECTION_MANIFEST),
                response_binding_schema=BINDING_SCHEMA)
        if self.allow_unknown_publication_clock:
            self.prediction_pipeline_manifest["body_admission"] = OPERATIONAL_BODY_ADMISSION
        self.prediction_pipeline_version = "entity-market-f:" + _digest(self.prediction_pipeline_manifest)
        self.manifest = {"schema": self.policy_schema, "base_policy": deepcopy(base_policy.manifest),
            "fixture": self.fixture, "physical_match_id": self.physical_match_id,
            "predictor_version": model.model_version, "predictor_sha256": self.predictor_sha256,
            "entity_representation_sha256": REPRESENTATION_SHA256,
            "extraction_protocol": {"version": protocol.version,
                "prompt_sha256": sha256(protocol.prompt.encode("utf-8")).hexdigest(),
                "schema_sha256": _digest(protocol.schema)},
            "semantic_config": deepcopy(self.semantic_config),
            "prediction_pipeline_manifest": deepcopy(self.prediction_pipeline_manifest),
            "prediction_pipeline_version": self.prediction_pipeline_version,
            "value_model_version": None, "lambda_cost": self.lambda_cost,
            "predictor_only": True, "bundled_g_enabled": False, "production_ready": False,
            "promotion_allowed": False, "selection_authority": "controlled_exploration_only",
            "value_scope": "unknown_not_estimated", "evidence_origin": meta["evidence_origin"],
            "probability_basis": "observed_market_prior_plus_frozen_entity_residual",
            "prior_domain_shift": "training_poisson_to_market_not_calibrated",
            "empirical_training_eligibility": "separate_pipeline_binding_and_outcome_audit_required"}
        self.version = self.policy_schema + ":" + _digest(self.manifest)[:24]
        self._provider = provider
        self._repair_provider = None
        self._clock = semantic_clock or (lambda: datetime.now(timezone.utc))
        self._archive = None
        self._records, self._events, self._forecasts = {}, [], {}
        self._calls = 0
        self._replay = deepcopy(replay_trace)
        if replay_trace is not None and (replay_trace.get("schema") != self.trace_schema
                or replay_trace.get("policy_version") != self.version):
            raise ValueError("Semantic extraction replay belongs to a different frozen policy")

    def public_state(self, state):
        if (tuple(self.model.probabilities) != self._frozen_prior
                or self.base_policy.manifest != self.manifest["base_policy"]):
            raise ValueError("Frozen market prior or its binding was modified")
        result = super().public_state(state)
        if state["match_id"] != self.fixture["match_id"]:
            raise ValueError("Entity shadow fixture identity changed")
        for key in ("home_team_id", "away_team_id", "competition_id", "kickoff"):
            a, b = state["base"][key], self.fixture[key]
            if (parse_time(a) != parse_time(b)) if key == "kickoff" else (a != b):
                raise ValueError("Entity shadow ordered fixture identity changed")
        return result

    def check_start(self, at):
        at = parse_time(at)
        if not (parse_time(self.model.frozen_at) <= at <= parse_time(self.model.cutoff)
                and parse_time(self.semantic_config["frozen_at"]) <= at < parse_time(self.fixture["kickoff"])):
            raise ValueError("Frozen entity shadow and prior must precede a valid acquisition start")

    def attach_runtime_archive(self, folder):
        self._archive = Path(folder) / TRACE_FILE
        if self._archive.exists():
            raise FileExistsError("Semantic trace exists; replay instead of repeating extraction")
        if self._replay is None and self._provider is None:
            from .deepseek_provider import DeepSeekInquiryProvider
            opts = {k: v for k, v in self.semantic_config["provider"].items() if k != "kind"}
            self._provider = DeepSeekInquiryProvider(Path(folder)/"semantic-provider-calls",
                protocol=self.extraction_protocol, **opts)
            if not self._provider.configuration()["configured"]:
                raise ValueError("Semantic provider is not configured; no source or provider HTTP executed")
        if self._replay is None and self.repair_enabled:
            from .deepseek_provider import DeepSeekInquiryProvider
            opts = {k: v for k, v in self.semantic_config["provider"].items() if k != "kind"}
            self._repair_provider = DeepSeekInquiryProvider(Path(folder)/"semantic-provider-repair-calls",
                protocol=self.repair_protocol, **opts)
            if not self._repair_provider.configuration()["configured"]:
                raise ValueError("Semantic repair provider is not configured; no repair HTTP executed")
        self._save_trace()

    def _save_trace(self):
        trace = {"schema": self.trace_schema, "policy_version": self.version,
                 "events": deepcopy(self._events)}
        if self._archive is not None:
            from .learned_voi_runtime import atomic_write
            atomic_write(self._archive, (json.dumps(trace, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode())
        return trace

    def finalize_runtime_archive(self):
        trace = self._save_trace()
        if self._replay is not None and trace != self._replay:
            raise ValueError("Semantic replay did not consume and reconstruct the exact extraction trace")
        return {"file": TRACE_FILE, "sha256": sha256(self._archive.read_bytes()).hexdigest(),
                "provider_attempts": self._calls, "extraction_events": len(self._events),
                "provider_cost_scope": "separate_usage_not_added_to_source_request_budget"}

    def _input(self, public):
        if json.loads(self.predictor_bytes) != {**self.predictor.payload(), "model_version": self.predictor.model_version}:
            raise ValueError("Frozen entity predictor was modified")
        self.public_state(public["state"])
        state, visible, skipped = self._visible(public)
        if self.request_bound:
            from .semantic_request_binding import filter_language_controls
            state["observed_links"], projection = filter_language_controls(state["observed_links"])
            visible["audit"]["language_navigation_projection"] = projection
        state.update(prior_probabilities=list(self.model.probabilities),
                     prior_basis=PRIOR_BASIS+":"+_digest(self.base_policy.manifest))
        if self.allow_unknown_publication_clock:
            state["allow_unknown_publication_clock"] = True
        checked = _project(state, text_projection=NEWS_PROJECTION)
        # Time/history are not features of this frozen f; failed queries cannot
        # create an entity assertion. Still validate every new visibility clock.
        key = _digest({k: v for k, v in state.items() if k not in {"decision_at", "history"}})
        return key, state, checked, visible, skipped

    def _forecast(self, state, visible, skipped):
        bank = EntityEvidenceBank(list(self._records.values()))
        projected = bank.project(state)
        return {"probabilities": _predict(self.predictor.coefficients, projected).tolist(),
            "initial_probabilities": self._initial_probability_vector(),
            "market_baseline_probabilities": list(self.model.probabilities),
            "model_version": self.predictor.model_version,
            "prediction_pipeline_version": self.prediction_pipeline_version,
            "probability_basis": self.manifest["probability_basis"],
            "prior_domain_shift": self.manifest["prior_domain_shift"],
            "used_semantic_input_ids": projected["used_input_ids"],
            "semantic_bank_sha256": bank.sha256, "entity_representation_sha256": REPRESENTATION_SHA256,
            "projection": projected, "modeled_read_documents": len(state["documents"]),
            "text_projection_audit": {**visible["audit"], "unmodeled_documents": skipped},
            "predictor_only": True, "gain_model_enabled": False, "production_ready": False,
            "training_evidence_origin": self.predictor.metadata["evidence_origin"],
            "evidence_origin": "prospective_source_assertions_not_independent_truth"}

    def _initial_probability_vector(self):
        from football_research.acquisition.semantic_f import FEATURES
        empty = {"prior": list(self.model.probabilities), "representation_sha256": REPRESENTATION_SHA256}
        for name in ("title", "novel_body", "title_entity_counts", "novel_body_entity_counts"):
            empty[name] = [0.] * len(FEATURES)
        return _predict(self.predictor.coefficients, empty).tolist()

    def journal_initial_assessment(self):
        p = self._initial_probability_vector()
        return {"status": "PREDICTOR_ONLY", "prediction": dict(zip(("home", "draw", "away"), p)),
            "candidate_expected_gains": None,
            "model_versions": [self.predictor.model_version]}

    def prepare_public_prediction(self, public):
        """The only side-effecting prediction step. Selection calls are pure."""
        key, state, checked, visible, skipped = self._input(public)
        if key in self._forecasts:
            return deepcopy(self._forecasts[key])
        event = {"state_key": key, "decision_at": public["decision_at"], "attempts": []}
        replay_event = None
        if self._replay is not None:
            if len(self._events) >= len(self._replay["events"]):
                raise ValueError("Semantic replay is missing an extraction event")
            replay_event = self._replay["events"][len(self._events)]
            if any(replay_event.get(k) != event[k] for k in ("state_key", "decision_at")):
                raise ValueError("Semantic extraction input state or clock changed")
        pieces = EntityEvidenceBank._items(checked)
        required = {k: v for channel in ("title", "body") for k, v in sorted(pieces[channel].items())}
        available = parse_time(public["decision_at"])
        deadline = min(parse_time(state["prediction_cutoff"]), parse_time(self.fixture["kickoff"]))
        reason = None
        for key_id, item in required.items():
            if key_id in self._records:
                continue
            if self._calls >= self.semantic_config["max_calls"]:
                reason = "semantic_extraction_allowance_exhausted"
                break
            if replay_event is not None:
                index = len(event["attempts"])
                if index >= len(replay_event["attempts"]):
                    if replay_event.get("reason") == "semantic_prediction_deadline_elapsed":
                        reason = replay_event["reason"]
                        available = parse_time(replay_event["available_at"])
                        if available <= deadline and available < parse_time(self.fixture["kickoff"]):
                            raise ValueError("Semantic deadline receipt does not cross a boundary")
                        break
                    raise ValueError("Semantic replay attempt missing")
                attempt = deepcopy(replay_event["attempts"][index])
                if attempt["input"] != item:
                    raise ValueError("Semantic replay attempted to extract different visible text")
            else:
                start = self._clock()
                if start > deadline or start >= parse_time(self.fixture["kickoff"]):
                    reason = "semantic_prediction_deadline_elapsed"
                    available = max(available, start)
                    break
                attempt = {"input": deepcopy(item), "requested_at": start.isoformat()}
                provider_calls = 1
                try:
                    request = deepcopy(item)
                    if self.request_bound:
                        from .semantic_request_binding import provider_input, bind_response
                        request = provider_input(item, protocol=self.extraction_protocol)
                        attempt["provider_input"] = deepcopy(request)
                    returned = self._provider.decide(request)
                    attempt["receipt"] = deepcopy(returned["receipt"])
                    if self.request_bound:
                        attempt["provider_response"] = deepcopy(returned["response"])
                        bound, binding = bind_response(item, attempt["provider_input"], returned["response"],
                                                       protocol=self.extraction_protocol)
                        attempt.update(response=bound, response_binding=binding)
                    else:
                        attempt["response"] = deepcopy(returned["response"])
                    validate_response(item, attempt["response"])
                    attempt["status"] = "OK"
                except Exception as exc:
                    # Provider logs own transport details; exception text may
                    # contain credentials, so archive only its class here.
                    attempt.update(status="FAILED", error_type=type(exc).__name__,
                                   failure_stage="validation" if _is_typed_validation_error(exc) else "provider")
                initial_completed = self._clock()
                attempt["initial_completed_at"] = initial_completed.isoformat()
                attempt["completed_at"] = initial_completed.isoformat()

                # A v4 run may make exactly one fresh provider call after a
                # typed validation failure.  The second call receives the
                # same canonical public input with a stronger, versioned
                # prompt.  It is never a local coercion or a relaxation of
                # validate_response; both the wire binding and semantic
                # validator still run on the repaired response.
                if (attempt["status"] != "OK" and self.repair_enabled
                        and attempt.get("failure_stage") == "validation"
                        and self._calls + provider_calls < self.semantic_config["max_calls"]):
                    from .semantic_request_binding import provider_input, bind_response
                    repair = {"requested_at": self._clock().isoformat()}
                    try:
                        repair_request = provider_input(item, protocol=self.repair_protocol)
                        repair["provider_input"] = deepcopy(repair_request)
                        returned = self._repair_provider.decide(repair_request)
                        repair["receipt"] = deepcopy(returned["receipt"])
                        repair["provider_response"] = deepcopy(returned["response"])
                        bound, binding = bind_response(item, repair_request, returned["response"],
                                                       protocol=self.extraction_protocol)
                        repair.update(response=bound, response_binding=binding)
                        validate_response(item, bound)
                        repair["status"] = "OK"
                    except Exception as repair_exc:
                        repair.update(status="FAILED", error_type=type(repair_exc).__name__,
                                      failure_stage="validation" if _is_typed_validation_error(repair_exc)
                                      else "provider")
                    repair_completed = self._clock()
                    repair["completed_at"] = repair_completed.isoformat()
                    attempt["initial_failure"] = {
                        "status": "FAILED", "error_type": attempt.get("error_type"),
                        "failure_stage": attempt.get("failure_stage"),
                        "completed_at": attempt["initial_completed_at"],
                    }
                    # Keep the first wire response and its receipt intact;
                    # promote only the separately validated repair response to
                    # the top-level response consumed by f.
                    if "provider_response" in attempt:
                        attempt["initial_provider_response"] = deepcopy(attempt["provider_response"])
                    if "response" in attempt:
                        attempt["initial_response"] = deepcopy(attempt["response"])
                    attempt["repair_attempt"] = repair
                    provider_calls += 1
                    if repair["status"] == "OK":
                        attempt["provider_response"] = deepcopy(repair["provider_response"])
                        attempt["response"] = deepcopy(repair["response"])
                        attempt["response_binding"] = deepcopy(repair["response_binding"])
                        attempt["status"] = "OK"
                        attempt["repair_status"] = "validated"
                        attempt["completed_at"] = repair["completed_at"]
                    else:
                        attempt["repair_status"] = "failed"
                        attempt["repair_error_type"] = repair.get("error_type")
                        attempt["completed_at"] = repair["completed_at"]
                attempt["provider_calls"] = provider_calls
            start, end = map(parse_time, (attempt["requested_at"], attempt["completed_at"]))
            if not available <= start <= end:
                raise ValueError("Semantic extraction clocks precede actual visible state")
            self._calls += attempt.get("provider_calls", 1)
            available = end
            event["attempts"].append(attempt)
            # Persist each attempt before using it, including failures/late returns.
            self._events.append({**deepcopy(event), "status": "PENDING"})
            self._save_trace()
            self._events.pop()
            if attempt["status"] != "OK":
                reason = "semantic_extraction_failed"
                break
            if self.request_bound:
                from .semantic_request_binding import bind_response
                bound, binding = bind_response(item, attempt["provider_input"], attempt["provider_response"],
                                               protocol=self.extraction_protocol)
                if bound != attempt["response"] or binding != attempt["response_binding"]:
                    raise ValueError("Request-bound semantic response differs from its archived wire response")
            validate_response(item, attempt["response"])
            if available > deadline or available >= parse_time(self.fixture["kickoff"]):
                reason = "semantic_prediction_deadline_elapsed"
                break
            self._records[key_id] = {"input": deepcopy(item), "response": deepcopy(attempt["response"]),
                "provenance": {"kind": "prospective_source_assertion_extraction",
                    "requested_at": attempt["requested_at"], "extracted_at": attempt["completed_at"],
                    "provider": deepcopy(self.semantic_config["provider"]), "receipt": deepcopy(attempt["receipt"])}}
        prediction = None
        if reason is None:
            # Availability includes local projection/inference, not only the
            # remote extractor response. Replay consumes recorded real clocks.
            start = (parse_time(replay_event["prediction_started_at"])
                     if replay_event is not None else self._clock())
            if start < available:
                raise ValueError("Prediction cannot precede its extracted evidence")
            event["prediction_started_at"] = start.isoformat()
            if start > deadline or start >= parse_time(self.fixture["kickoff"]):
                reason = "semantic_prediction_deadline_elapsed"
                available = start
            else:
                state["decision_at"] = start.isoformat()
                prediction = self._forecast(state, visible, skipped)
                end = (parse_time(replay_event["prediction_completed_at"])
                       if replay_event is not None else self._clock())
                if end < start:
                    raise ValueError("Prediction completion precedes inference")
                event["prediction_completed_at"] = end.isoformat()
                available = end
                if end > deadline or end >= parse_time(self.fixture["kickoff"]):
                    event["rejected_prediction_sha256"] = _digest(prediction)
                    prediction = None
                    reason = "semantic_prediction_deadline_elapsed"
        event.update(status="not_updated" if reason else "ready", reason=reason, available_at=available.isoformat())
        result = {"status": event["status"], "reason": reason, "available_at": event["available_at"]}
        if reason is None:
            result["prediction"] = prediction
            event["prediction_sha256"] = _digest(result["prediction"])
            self._forecasts[key] = deepcopy(result)
        if replay_event is not None and event != replay_event:
            raise ValueError("Semantic extraction replay differs from its frozen event")
        self._events.append(event)
        self._save_trace()
        return result

    def predict_public(self, public):
        key, *_ = self._input(public)
        result = self._forecasts.get(key)
        if result is None or parse_time(result["available_at"]) > parse_time(public["decision_at"]):
            raise ValueError("Semantic prediction is missing or not yet available")
        return deepcopy(result["prediction"])


def main(argv=None):
    """Freeze an explicit configuration locally, without any provider call."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--extractor-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-calls", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout-seconds", type=float, default=60)
    parser.add_argument("--thinking", choices=("enabled", "disabled"), default="disabled")
    parser.add_argument("--protocol-version", choices=("v1", "v2", "v3", "v4"), default="v1",
                        help="v2: host binding; v3: typed predicate/value schema; v4: v3 plus one explicit validation repair")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Semantic configuration exists; reuse it for this cohort")
    raw = args.model.read_bytes()
    EntitySemanticF.load(args.model)
    schema = {"v1": CONFIG_SCHEMA, "v2": REQUEST_BOUND_CONFIG_SCHEMA,
              "v3": TYPED_REQUEST_CONFIG_SCHEMA, "v4": REPAIR_REQUEST_CONFIG_SCHEMA}[args.protocol_version]
    values = {"schema": schema, "model_sha256": sha256(raw).hexdigest(),
        "frozen_at": datetime.now(timezone.utc).isoformat(), "prior_basis": PRIOR_BASIS,
        "provider": {"kind": "deepseek_api", "model": args.extractor_model,
            "thinking": args.thinking, "max_tokens": args.max_tokens, "timeout_seconds": args.timeout_seconds},
        "max_calls": args.max_calls}
    if args.protocol_version == "v4":
        values["repair"] = {"enabled": True, "max_attempts": 1,
            "trigger": "typed_validation_failure_only",
            "prompt_version": "request-bound-football-source-assertions-v3-repair-v1"}
    spec = validate_config(values, raw)
    from .learned_voi_runtime import _save
    _save(args.output, spec)
    print(json.dumps({"configuration": str(args.output.resolve()), "network_requests": 0,
        "model_sha256": spec["model_sha256"], "frozen_at": spec["frozen_at"]}))


if __name__ == "__main__":
    main()
