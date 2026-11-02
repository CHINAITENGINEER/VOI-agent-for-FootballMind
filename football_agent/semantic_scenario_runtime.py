"""One-match semantic proposals, exact conditional VoI, and real source returns.

The semantic model supplies assumptions, never action values. This controller
owns capabilities, measured requests, time admission, evidence and the solver.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from football_research.acquisition.news_link_projection import project_news_links
from .semantic_scenario_model import SemanticScenarioModel, PLANNING_POLICY_VERSION
from .semantic_scenario_protocol import (
    ProposalProtocol, ObservationProtocol, DirectionalObservationProtocol,
    DirectionalProposalProtocol, ActionExtensionProtocol, DirectionalActionExtensionProtocol,
    CandidateDiscoveryProtocol, validate_discovery, validate_proposal,
    validate_directional_proposal, validate_observation, validate_action_extension,
    validate_directional_action_extension, public_projection, no_answer,
    observation_public_projection, OBSERVATION_INPUT_VERSION,
)
from .directional_evidence import (
    validate_directional_observation, directional_update_for_action,
)
from .semantic_candidate_discovery import (
    CatalogDiscoveryProtocol, discovery_catalog, validate_catalog_discovery,
)


VERSION = "semantic-scenario-runtime-v1"
REDISCOVERY_VERSION = "semantic-scenario-rediscovery-v1"
POLICY_MODE = "conditional_semantic_voi_untrained_g"
INITIALIZATION_RESUME_VERSION = "semantic-initialization-prefix-v1"


def digest(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                             allow_nan=False).encode("utf-8")).hexdigest()


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    indent=2, allow_nan=False), encoding="utf-8")


def timestamp(value):
    instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if instant.tzinfo is None:
        raise ValueError("Source/decision timestamps must carry a timezone")
    return instant.astimezone(timezone.utc)


def admit_document(document, *, cutoff, now):
    """Only actual, already acquired and typed body paragraphs reach evidence."""
    doc = deepcopy(document)
    boundary = min(timestamp(cutoff), now)
    reasons = []
    for field in ("observed_at", "available_at"):
        try:
            if timestamp(doc[field]) > boundary:
                reasons.append(field + "_after_decision_boundary")
        except (KeyError, ValueError, TypeError, AttributeError):
            reasons.append(field + "_missing_or_invalid")
    for field in ("source_published_at", "source_modified_at"):
        if doc.get(field):
            try:
                if timestamp(doc[field]) > boundary:
                    reasons.append(field + "_after_decision_boundary")
            except (ValueError, TypeError, AttributeError):
                reasons.append(field + "_invalid")
    if reasons:
        doc.update(text="", links=[], status="INADMISSIBLE")
    parts = []
    excluded = []
    if not reasons and doc.get("status") == "OK" and doc.get("kind") == "article":
        for segment in doc.get("structured_news", {}).get("segments", []):
            if segment.get("content_role") not in (None, "paragraph"):
                continue
            published = segment.get("source_published_at")
            modified = segment.get("source_modified_at")
            invalid = False
            for label, instant in (("published", published), ("modified", modified)):
                if instant:
                    try:
                        if timestamp(instant) > boundary:
                            invalid = True
                            excluded.append(label + "_after_decision_boundary")
                    except (ValueError, TypeError, AttributeError):
                        invalid = True
                        excluded.append(label + "_invalid")
            text = segment.get("text", "")
            # The typed parser binds these bytes to the same archived page.
            if not invalid and isinstance(text, str) and text in doc.get("text", ""):
                parts.append(text)
    doc["text"] = "\n".join(parts)
    doc["admission"] = {"version": "semantic-source-body-admission-v1",
        "boundary": boundary.isoformat(), "reasons": reasons, "excluded_segments": excluded,
        "body_segments": len(parts), "source_date_unknown": not bool(doc.get("source_published_at")),
        "scope": "observed_source_text_not_verified_fixture_applicability"}
    return doc


class SemanticScenarioRuntime:
    def __init__(self, *, provider_factory, transport, output, case, seed_urls,
                 bootstrap_url, request_budget=6, request_loss_cost=.001,
                 semantic_call_budget=12, max_extensions=1, max_rediscoveries=0,
                 candidate_limit=8,
                 candidate_discovery_version=1,
                 planning_horizon=1,
                 require_directional_evidence=False,
                 clock=None, progress=None, resume_from=None):
        if isinstance(request_budget, bool) or not isinstance(request_budget, int) or not 1 <= request_budget <= 12:
            raise ValueError("Use a bounded positive request budget")
        if (not 1 <= candidate_limit <= 8 or not 3 <= semantic_call_budget <= 20 or
                isinstance(max_extensions, bool) or not isinstance(max_extensions, int) or
                not 0 <= max_extensions <= 2 or
                isinstance(max_rediscoveries, bool) or not isinstance(max_rediscoveries, int) or
                not 0 <= max_rediscoveries <= 1 or
                isinstance(planning_horizon, bool) or not isinstance(planning_horizon, int) or
                planning_horizon not in (1, 2)):
            raise ValueError("Invalid bounded semantic runtime limits")
        if bootstrap_url not in seed_urls:
            raise ValueError("Bootstrap must be a registered source entry")
        if type(candidate_discovery_version) is not int or candidate_discovery_version not in (1, 2):
            raise ValueError("Unsupported candidate discovery version")
        self.output = Path(output)
        if resume_from is not None and self.output.resolve().is_relative_to(Path(resume_from).resolve().parent):
            raise ValueError("Resume output must be outside the immutable source run directory")
        if self.output.exists():
            raise FileExistsError("Choose a new output directory; previous runs are immutable")
        self.output.mkdir(parents=True)
        self.provider_factory, self.transport = provider_factory, transport
        self.case, self.seed_urls = deepcopy(case), list(seed_urls)
        self.bootstrap_url = bootstrap_url
        self.request_budget, self.request_loss_cost = request_budget, request_loss_cost
        self.semantic_call_budget, self.max_extensions = semantic_call_budget, max_extensions
        self.max_rediscoveries = max_rediscoveries
        self.candidate_limit = candidate_limit
        self.candidate_discovery_version = candidate_discovery_version
        self.candidate_discovery = None
        self.planning_horizon = planning_horizon
        if not isinstance(require_directional_evidence, bool):
            raise ValueError("require_directional_evidence must be a boolean")
        self.require_directional_evidence = require_directional_evidence
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.progress = progress or (lambda message: None)
        self.documents, self.calls, self.receipts, self.cycles, self.observations = [], [], [], [], []
        self.candidates, self.extensions, self.rediscoveries, self.failures = {}, [], [], []
        self.requests = 0
        self.started_at = self.clock().isoformat()
        self.run_id = "semantic-case-" + digest({"output": str(self.output.resolve()), "time": self.started_at})[:20]
        teams = {t["side"]: t["canonical_name"] for t in case["source_profile"]["teams"]}
        self.match = {**case["match"], "home_name": teams["home"], "away_name": teams["away"]}
        self.autonomous_state = deepcopy(case.get("autonomous_initial_state"))
        if not timestamp(self.started_at) <= timestamp(self.match["prediction_cutoff"]) < timestamp(self.match["kickoff"]):
            raise ValueError("A live prematch run requires now <= prediction_cutoff < kickoff")
        self.baseline = deepcopy(case["structured_baseline"]["probabilities"])
        self.proposal, self.model = None, None
        self.resume_provenance, self._resumed_discovery = None, None
        initialization_protocols = [self._discovery_protocol(),
            DirectionalProposalProtocol() if require_directional_evidence else ProposalProtocol()]
        self.configuration = {"version": VERSION, "run_id": self.run_id, "started_at": self.started_at,
            "match": self.match, "baseline": self.baseline, "baseline_scope": case.get("baseline_scope"),
            "seed_urls": self.seed_urls, "bootstrap_url": bootstrap_url,
            "request_budget": request_budget, "request_loss_cost": request_loss_cost,
            "semantic_call_budget": semantic_call_budget, "max_extensions": max_extensions,
            "rediscovery": {"version": REDISCOVERY_VERSION, "max_rounds": max_rediscoveries,
                             "trigger": "after_first_observation_only",
                             "query_request_cost": 2,
                             "scope": "bounded_candidate_discovery_then_action_extension"},
            "candidate_limit": candidate_limit,
            "candidate_discovery_version": candidate_discovery_version,
            "observation_input_contract": OBSERVATION_INPUT_VERSION,
            "transport": transport.configuration(),
            "autonomous_match_state": deepcopy(self.autonomous_state),
            "cost_semantics": "declared_HTTP_loss_equivalent_nats_not_money_LLM_overhead_separately_measured",
            "contamination_levels": [0., .25, .5], "horizon": planning_horizon,
            "planning_horizon": planning_horizon,
            "planning_policy": PLANNING_POLICY_VERSION,
            "initialization_contract": {
                "version": INITIALIZATION_RESUME_VERSION,
                "case_sha256": digest(case),
                "protocols": {p.version: digest({"schema": p.schema, "prompt": p.prompt})
                              for p in initialization_protocols}},
            "policy_mode": POLICY_MODE,
            "observation_protocol": (DirectionalObservationProtocol.version
                                      if require_directional_evidence else ObservationProtocol.version),
            "directional_evidence_required": require_directional_evidence,
            "empirical_g": {"status": "deferred", "reason": "timestamped_information_stream_not_available",
                            "used_for_selection": False, "interface_ready": True},
            "empirically_calibrated": False, "future_outcome_loaded": False,
            "study_role": ("zero_loss_cost_mechanism_exercise_not_policy_success"
                           if request_loss_cost == 0 else "declared_cost_live_case"),
            "scope": "one_case_development_run_not_confirmatory_algorithm_evaluation"}
        if resume_from is not None:
            self._restore_initialization_prefix(Path(resume_from))
        save(self.output / "configuration.json", self.configuration)

    def _restore_initialization_prefix(self, source):
        """Carry an internally checked initialization prefix, never a decision trace.

        This is a continuation of one cumulative budget, not a new experiment
        or a semantic cache. Legacy runs without the case/protocol binding are
        deliberately rejected. Copied call/HTTP records retain their original
        receipts, inputs and clocks; new calls use the actual current clock.
        """
        raw = source.read_bytes()
        old = json.loads(raw)
        if old.get("version") != VERSION or old.get("status") != "incomplete":
            raise ValueError("Resume requires a matching incomplete runtime version")
        previous = old.get("configuration", {})
        if previous.get("initialization_contract") != self.configuration["initialization_contract"]:
            raise ValueError("Resume case/protocol binding missing or incompatible; legacy prefixes cannot resume")
        for key, value in self.configuration.items():
            if key not in {"run_id", "started_at"} and previous.get(key) != value:
                raise ValueError("Resume configuration mismatch: " + key)
        now = self.clock()
        if not (timestamp(previous["started_at"]) <= timestamp(old["completed_at"]) <= now
                <= timestamp(self.match["prediction_cutoff"])):
            raise ValueError("Resume requires monotonic actual time before prediction cutoff")
        if (old.get("cycles") != [] or old.get("observations") != [] or old.get("proposal") is not None
                or old.get("extensions") != [] or old.get("rediscoveries") != []):
            raise ValueError("Resume supports initialization only, without a decision/evidence trajectory")
        if old.get("initial_prediction") != self.baseline or old.get("final_prediction") != self.baseline:
            raise ValueError("Resume initialization prediction changed")
        receipts, calls = old.get("http_receipts", []), old.get("semantic_calls", [])
        resources = old.get("resources", {})
        if (len(receipts) != 1 or resources.get("http_accounting_complete") is not True
                or resources.get("http_requests") != 1 or resources.get("request_budget_reserved") != 1
                or receipts[0].get("returned", {}).get("http_attempts") != 1
                or receipts[0].get("status") == "reconciliation_required"):
            raise ValueError("Resume requires exactly one fully accounted bootstrap HTTP receipt")
        receipt = receipts[0]
        if (receipt.get("index") != 0 or receipt.get("purpose") != "declared_source_discovery_not_voi"
                or receipt.get("candidate") != {"tool": "read_url", "url": self.bootstrap_url}
                or old.get("documents") != [receipt.get("admitted")]):
            raise ValueError("Resume bootstrap receipt/document binding differs")
        admitted = receipt["admitted"]
        try:
            acquisition_time = timestamp(admitted["admission"]["boundary"])
            expected = admit_document(receipt["returned"]["document"],
                cutoff=self.match["prediction_cutoff"], now=acquisition_time)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Resume bootstrap admission cannot be replayed") from error
        if (expected != admitted or acquisition_time > timestamp(old["completed_at"])
                or acquisition_time < timestamp(previous["started_at"])):
            raise ValueError("Resume bootstrap admission or acquisition time differs")
        if (not calls or resources.get("semantic_calls") != len(calls)
                or len(calls) >= self.semantic_call_budget or calls[-1].get("status") != "failed"):
            raise ValueError("Resume requires a failed semantic call and remaining cumulative call budget")
        discovery_protocol = self._discovery_protocol()
        proposal_protocol = DirectionalProposalProtocol() if self.require_directional_evidence else ProposalProtocol()
        protocols = {p.version: p for p in (discovery_protocol, proposal_protocol)}
        discovery = None
        # Restore documents first to reconstruct the exact public capability
        # menu; no call, HTTP request or current-clock admission is performed.
        self.documents = deepcopy(old["documents"])
        self.requests = 1
        for index, call in enumerate(calls):
            protocol = protocols.get(call.get("protocol"))
            if protocol is None or call.get("index") != index or call.get("status") not in {"accepted", "failed", "invalid"}:
                raise ValueError("Resume semantic initialization sequence is invalid")
            is_discovery = protocol.version == discovery_protocol.version
            if (is_discovery and discovery is not None) or (not is_discovery and discovery is None):
                raise ValueError("Resume semantic initialization stage order differs")
            provider = self.provider_factory(self.output / "semantic-calls" / f"call-{index:03d}", protocol)
            if provider.configuration() != call.get("provider_configuration"):
                raise ValueError("Resume provider configuration differs")
            menu = [] if is_discovery else self._initialization_menu(discovery)["menu"]
            public = deepcopy(call.get("input", {}))
            repair_errors = public.pop("validation_errors", None)
            if repair_errors is not None and not isinstance(repair_errors, list):
                raise ValueError("Resume semantic validation history is invalid")
            try:
                decision_at = public["match"]["decision_at"]
                if not (acquisition_time <= timestamp(decision_at) <= timestamp(old["completed_at"])):
                    raise ValueError("out of bounds")
            except (KeyError, ValueError, TypeError) as error:
                raise ValueError("Resume semantic input has invalid decision time") from error
            expected_public = public_projection(match={**self.match, "decision_at": decision_at},
                baseline={"probabilities": self.baseline, "basis": "existing_structured_historical_reconstruction"},
                documents=self.documents, candidates=menu,
                budget={"remaining_requests": self.request_budget-self.requests,
                        "deadline": self.match["prediction_cutoff"]})
            if self.autonomous_state is not None:
                expected_public["autonomous_match_state"] = deepcopy(self.autonomous_state)
            if is_discovery and self.candidate_discovery_version == 2:
                expected_public.update(self._catalog())
                expected_public["candidate_limit"] = self.candidate_limit
            if public != expected_public:
                raise ValueError("Resume semantic input does not match frozen public state")
            if call["status"] == "accepted":
                if not is_discovery:
                    raise ValueError("Resume cannot reuse a completed probability proposal")
                discovery = self._validate_discovery(call["response"])
                if discovery != call.get("validated"):
                    raise ValueError("Resume discovery validated result differs from original response")
            elif not isinstance(call.get("error"), str):
                raise ValueError("Resume unsuccessful call lacks an error receipt")
        if discovery is not None and not self._initialization_menu(discovery)["menu"]:
            raise ValueError("Resume discovery has no executable candidates")
        self.calls, self.receipts = deepcopy(calls), deepcopy(receipts)
        self.failures = deepcopy(old.get("failures", []))
        self._resumed_discovery = deepcopy(discovery)
        self.resume_provenance = {
            "version": INITIALIZATION_RESUME_VERSION, "source_run_path": str(source.resolve()),
            "source_run_sha256": sha256(raw).hexdigest(), "source_run_id": old["run_id"],
            "source_completed_at": old["completed_at"], "resumed_at": self.started_at,
            "scope": "initialization_prefix_only_no_decisions_or_observations_replayed",
            "retained_http_receipt_indices": [0], "retained_semantic_call_indices": list(range(len(calls))),
            "retained_failed_semantic_calls": sum(c["status"] != "accepted" for c in calls),
            "reused_discovery_call_index": next((c["index"] for c in calls if c["status"] == "accepted"), None),
            "cumulative_budget_no_refund": True,
            "source_files_modified": False}
        save(self.output / "resume-provenance.json", self.resume_provenance)
        for call in self.calls:
            save(self.output / "semantic-calls" / f"record-{call['index']:03d}.json", call)
        save(self.output / "http" / "request-000.json", self.receipts[0])

    def _initialization_menu(self, discovery):
        queries = [{"action_id": "search-"+digest(q["query"])[:20], "tool": "search_then_read",
                    **q, "request_cost": 2,
                    "return_contract": "search_RSS_then_read_first_permitted_news_link_body_max_2_HTTP_no_snippet_evidence"}
                   for q in discovery["queries"]]
        observed = self._read_candidates()
        if self.candidate_discovery_version == 2:
            by_id = {item["action_id"]: item for item in observed}
            nominated = [{**by_id[item["action_id"]], "information_need": item["information_need"]}
                         for item in discovery["article_candidates"]]
            menu = queries + nominated
        else:
            menu = (queries + observed)[:self.candidate_limit]
        record = {"proposal": discovery, "menu": menu,
                "observed_article_candidates": len(observed), "menu_cap": self.candidate_limit,
                "omitted_visible_candidates": max(0, len(queries)+len(observed)-len(menu)),
                "ordering": "model_generated_queries_then_observed_DOM_order_not_value_screening"}
        if self.candidate_discovery_version == 2:
            record.update(ordering="semantic_nomination_from_observed_catalog_then_numerical_voi",
                          catalog_coverage=self._catalog()["catalog_coverage"],
                          unmodeled_article_candidates=len(observed)-len(nominated))
        return record

    def _discovery_protocol(self):
        return CatalogDiscoveryProtocol() if self.candidate_discovery_version == 2 else CandidateDiscoveryProtocol()

    def _catalog(self):
        catalog = discovery_catalog([item for item in self._read_candidates()
                                     if item["action_id"] not in self.candidates])
        catalog["catalog_coverage"]["source_projection_truncated"] = any(
            document.get("links_truncated", False) for document in self.documents)
        return catalog

    def _discovery_public(self):
        public = self._public([])
        if self.candidate_discovery_version == 2:
            public.update(self._catalog())
            public["candidate_limit"] = self.candidate_limit
        return public

    def _validate_discovery(self, response):
        if self.candidate_discovery_version == 1:
            return validate_discovery(response)
        shown = {item["action_id"] for item in self._catalog()["observed_articles"]}
        pool = [item for item in self._read_candidates() if item["action_id"] in shown]
        return validate_catalog_discovery(response, pool, self.candidate_limit)

    def _public(self, candidates):
        public = public_projection(match={**self.match, "decision_at": self.clock().isoformat()},
            baseline={"probabilities": self.baseline, "basis": "existing_structured_historical_reconstruction"},
            documents=self.documents, candidates=candidates,
            budget={"remaining_requests": self.request_budget-self.requests,
                    "deadline": self.match["prediction_cutoff"]})
        if self.autonomous_state is not None:
            public["autonomous_match_state"] = deepcopy(self.autonomous_state)
        return public

    def _semantic(self, protocol, public, validate):
        """Validate a semantic return with one narrowly scoped repair.

        A prose/evidence mistake can often be repaired by showing the exact
        local validation error to the provider.  A numeric or capability
        contract violation is different: retrying it burns another long model
        call while leaving the same frozen proposal assumptions in place.  We
        therefore close those failures immediately, without normalization or
        fabricated replacement values.
        """
        errors = []
        for _ in range(2):
            if len(self.calls) >= self.semantic_call_budget:
                raise RuntimeError("semantic_call_budget_exhausted")
            index = len(self.calls)
            workspace = self.output / "semantic-calls" / f"call-{index:03d}"
            provider = self.provider_factory(workspace, protocol)
            call = {"index": index, "protocol": protocol.version,
                    "input": deepcopy(public), "provider_configuration": provider.configuration()}
            if errors:
                call["input"]["validation_errors"] = errors
            self.calls.append(call)
            save(self.output / "semantic-calls" / f"record-{index:03d}.json", call)
            self.progress(f"semantic call {index+1}: {protocol.version}")
            try:
                returned = provider.decide(call["input"])
                call.update(returned)
                cleaned = validate(returned["response"])
                call.update(status="accepted", validated=deepcopy(cleaned))
                return cleaned
            except ValueError as error:
                message = str(error)
                errors.append(message)
                repair_eligible, failure_code = self._semantic_repair_policy(message)
                call.update(status="invalid", error=message,
                            repair_eligible=repair_eligible,
                            failure_code=failure_code)
                if not repair_eligible:
                    break
            except Exception as error:
                # Direct transports retain usage and elapsed time on failure.
                if isinstance(getattr(provider, "last_receipt", None), dict):
                    call["receipt"] = deepcopy(provider.last_receipt)
                call.update(status="failed", error=f"{type(error).__name__}: {error}")
                raise
            finally:
                save(self.output / "semantic-calls" / f"record-{index:03d}.json", call)
        raise ValueError("semantic_validation_failed: " + "; ".join(errors))

    @staticmethod
    def _semantic_repair_policy(message: str) -> tuple[bool, str]:
        """Classify local validation failures without inspecting provider text.

        The numerical and action-capability contracts are deterministic.  They
        must be fixed by a new proposal, not softened by a retry.  Evidence and
        schema-shape failures remain eligible for one explicit repair because
        the provider can often correct a quote or a missing field from the
        returned validation message.
        """
        numeric_or_capability = (
            "must sum to one",
            "strictly positive finite probabilities",
            "hypothesis-independent",
            "every hypothesis exactly once",
            "Duplicate hypothesis",
            "public candidate",
            "cover every supplied candidate",
            "unknown source direction",
            "at most one team direction",
            "cannot mix mixed and team direction",
        )
        if any(token in message for token in numeric_or_capability):
            if "sum to one" in message or "probabilities" in message or "hypothesis-independent" in message:
                return False, "numeric_contract"
            return False, "action_contract"
        return True, "schema_or_evidence"

    def _no_answer(self, action, reason):
        """Create an inert record under the active observation contract."""
        value = no_answer(action, reason)
        if not self.require_directional_evidence:
            return value
        value.update({
            "schema": DirectionalObservationProtocol.version,
            "direction": "unknown",
            "subject": "unknown",
            "direction_basis": "none",
            "directional_claim": "没有可用于本场方向判断的正文证据。",
        })
        # The directional validator is intentionally used even for a failed
        # transport return so the saved record has exactly the same contract
        # as a provider-produced no-answer.
        return validate_directional_observation(value, action, [])

    def _gate_directional_observation(self, observation, action):
        """Make source-semantic direction the only path to a live update."""
        if not self.require_directional_evidence:
            return observation, None
        contract = action.get("direction_contract")
        outcome = observation.get("outcome")
        direction = observation.get("direction")
        if outcome in {"supports", "refutes"}:
            allowed = set((contract or {}).get(outcome, ()))
            if direction not in allowed:
                inert = deepcopy(observation)
                inert["outcome"] = "no_answer"
                inert["applicability"] = "unresolved"
                inert["directional_gate"] = "inert_direction_contract_mismatch"
                inert["reason"] = (
                    "正文方向与该查询命题的" + outcome + "方向合同不一致；"
                    "不能把相反或无关方向重新解释为支持/反驳，因此不进入概率更新。"
                    + observation.get("reason", ""))[:2400]
                return inert, {"status": "inert", "reason": "direction_contract_mismatch",
                                "direction": direction, "allowed": sorted(allowed),
                                "outcome": outcome}
        # Keep the update bound to the exact action contract.  The earlier
        # mismatch branch is retained for an auditable inert observation, and
        # the action-bound adapter protects future callers that bypass it.
        effect = directional_update_for_action(observation, action)
        if effect is not None:
            return observation, {"status": "accepted", "effect": effect}
        if observation.get("outcome") in {"supports", "refutes"}:
            inert = deepcopy(observation)
            inert["outcome"] = "no_answer"
            inert["applicability"] = "unresolved"
            inert["directional_gate"] = "inert_without_explicit_match_specific_effect"
            inert["reason"] = ("正文回答了查询，但没有明确说明对本场主客队结果的方向性影响；"
                                "不进入概率更新。" + observation.get("reason", ""))[:2400]
            return inert, {"status": "inert", "reason": "missing_explicit_match_specific_direction"}
        return observation, {"status": "no_answer", "reason": "no_directional_update"}

    def _acquire(self, candidate, purpose):
        if self.requests >= self.request_budget:
            raise RuntimeError("request_budget_exhausted")
        if self.clock() > timestamp(self.match["prediction_cutoff"]):
            raise RuntimeError("prediction_cutoff_reached")
        prepared = self.transport.prepare(candidate, {"task": {"seed_urls": self.seed_urls},
                                                      "documents": self.documents})
        index = len(self.receipts)
        self.progress(f"HTTP {self.requests+1}/{self.request_budget}: {candidate['tool']} {candidate.get('url', candidate.get('query'))}")
        # Reserve the bounded attempt before calling the transport. A parser can
        # fail after an HTTP response was archived, so exceptions cannot refund
        # the budget or imply that zero network requests occurred.
        self.requests += 1
        try:
            returned = self.transport.execute(prepared, run_id=self.run_id, request_id=f"http-{index:03d}")
            if returned.get("http_attempts") != 1:
                raise RuntimeError("one_attempt_transport_contract_violated")
        except Exception as error:
            receipt = {"index": index, "purpose": purpose, "candidate": deepcopy(candidate),
                "prepared": prepared, "status": "reconciliation_required", "reserved_requests": 1,
                "returned": {"http_attempts": None}, "error": f"{type(error).__name__}: {error}"}
            self.receipts.append(receipt)
            save(self.output / "http" / f"request-{index:03d}.json", receipt)
            raise
        admitted = admit_document(returned["document"], cutoff=self.match["prediction_cutoff"], now=self.clock())
        receipt = {"index": index, "purpose": purpose, "candidate": deepcopy(candidate),
                   "prepared": prepared, "returned": returned, "admitted": admitted}
        self.receipts.append(receipt)
        self.documents.append(admitted)
        save(self.output / "http" / f"request-{index:03d}.json", receipt)
        return admitted

    def _links(self):
        links = []
        seen = set()
        for document in self.documents:
            if document.get("status") != "OK":
                continue
            eligible, _ = project_news_links(document.get("links", []))
            for link in eligible:
                if link["url"] in seen:
                    continue
                seen.add(link["url"])
                links.append({**link, "parent_url": document["url"], "available_at": document["available_at"]})
        return links

    def _read_candidates(self):
        acquired_urls = {d["url"] for d in self.documents}
        return [{"action_id": "read-"+digest(link["url"])[:20], "tool": "read_url", **link,
                 "request_cost": 1, "return_contract": "one_actual_article_body_or_no_answer"}
                for link in self._links() if link["url"] not in acquired_urls]

    def _new_model(self, proposal=None, candidates=None):
        return SemanticScenarioModel(proposal if proposal is not None else self.proposal, self.baseline,
            request_loss_cost=self.request_loss_cost,
            contamination_levels=tuple(self.configuration["contamination_levels"]),
            planning_horizon=self.planning_horizon,
            action_request_costs={key: value["request_cost"] for key, value in
                                 (candidates if candidates is not None else self.candidates).items()})

    def _decision(self):
        result = self.model.evaluate(self.observations, self.request_budget-self.requests)
        cycle = {"index": len(self.cycles), "model": self.model.manifest(),
                 "observations": deepcopy(self.observations), "requests_spent": self.requests,
                 "remaining_requests": self.request_budget-self.requests, "decision": result,
                 "candidate_bindings": deepcopy(self.candidates)}
        self.cycles.append(cycle)
        save(self.output / "cycles" / f"cycle-{cycle['index']:03d}.json", cycle)
        return result

    def _execute_action(self, candidate):
        if candidate["tool"] == "read_url":
            return [self._acquire(candidate, "voi_selected_body_read")]
        if candidate["tool"] != "search_then_read":
            raise ValueError("Unregistered semantic action capability")
        result = self._acquire({"tool": "search_web", "query": candidate["query"]}, "voi_selected_composite_search")
        links, _ = project_news_links(result.get("links", []))
        if result.get("status") != "OK" or not links:
            return [result]
        # Search ranking and the host/URL contract, never an unseen article's value,
        # choose the first permitted result in this predeclared composite tool.
        document = self._acquire({"tool": "read_url", "url": links[0]["url"]}, "voi_selected_composite_first_body")
        return [result, document]

    @staticmethod
    def _query_key(candidate):
        query = candidate.get("query")
        return " ".join(query.split()).casefold() if isinstance(query, str) else None

    def _query_candidates(self, discovery):
        """Convert discovery-only queries into unexecuted action capabilities."""
        existing_keys = {self._query_key(candidate) for candidate in self.candidates.values()}
        existing_keys.discard(None)
        new = []
        for item in discovery["queries"]:
            query = item["query"]
            key = " ".join(query.split()).casefold()
            action_id = "search-" + digest(query)[:20]
            if action_id in self.candidates or key in existing_keys:
                continue
            candidate = {"action_id": action_id, "tool": "search_then_read", **item,
                         "request_cost": 2,
                         "return_contract": "search_RSS_then_read_first_permitted_news_link_body_max_2_HTTP_no_snippet_evidence"}
            new.append(candidate)
            existing_keys.add(key)
        return new

    def _extend(self, extra_candidates=(), *, source="observed_links"):
        # v2 adds only semantically nominated capabilities. Do not silently
        # reintroduce a DOM-prefix menu after the first observation.
        if self.candidate_discovery_version == 2 and not extra_candidates:
            return None
        # max_extensions limits the legacy visible-link extension path only.
        # Rediscovered queries have their own explicit max_rediscoveries bound.
        if not extra_candidates and sum(item.get("source") == "observed_links"
                                        for item in self.extensions) >= self.max_extensions:
            return None
        if len(self.calls) >= self.semantic_call_budget-1:
            return None
        # Rediscovered queries are placed first so the candidate cap cannot
        # silently omit the very query that was just modeled. Legacy observed
        # links retain their original DOM order after those explicit queries.
        new = [c for c in extra_candidates if c["action_id"] not in self.candidates]
        if self.candidate_discovery_version == 1:
            new.extend(c for c in self._read_candidates() if c["action_id"] not in self.candidates and
                       all(c["action_id"] != item["action_id"] for item in new))
        new = new[:self.candidate_limit]
        if not new:
            return None
        before = self.model.evaluate(self.observations, self.request_budget-self.requests)["prediction_nominal"]
        public = self._public(new)
        public.update(frozen_proposal=deepcopy(self.proposal), actual_observations=deepcopy(self.observations))
        extension_protocol = (DirectionalActionExtensionProtocol()
                              if self.require_directional_evidence else ActionExtensionProtocol())
        if self.require_directional_evidence:
            extension_validator = lambda response: validate_directional_action_extension(
                response, new, [h["id"] for h in self.proposal["hypotheses"]],
                list(self.candidates))
        else:
            extension_validator = lambda response: validate_action_extension(
                response, new, [h["id"] for h in self.proposal["hypotheses"]],
                list(self.candidates))
        addition = self._semantic(extension_protocol, public, extension_validator)
        record = {"source": source, "new_candidates": new, "addition": addition,
                  "conditioned_on_observations": deepcopy(self.observations), "history_sha256": digest(public)}
        proposal = deepcopy(self.proposal)
        proposal["actions"].extend(addition["actions"])
        proposal["assumptions"].extend(addition["assumptions"])
        candidates = {**self.candidates, **{c["action_id"]: c for c in new}}
        model = self._new_model(proposal, candidates)
        after = model.evaluate(self.observations, self.request_budget-self.requests)["prediction_nominal"]
        if any(abs(a-b) > 1e-12 for a, b in zip(before, after)):
            raise RuntimeError("action_extension_changed_evidence_posterior")
        self.proposal, self.candidates, self.model = proposal, candidates, model
        self.extensions.append(record)
        save(self.output / "extensions" / f"extension-{len(self.extensions):03d}.json", record)
        return record

    def _rediscover(self):
        """Run at most one bounded discovery after the first real observation.

        Discovery returns query candidates and, in v2, observed article IDs.
        No target body is acquired here; ActionExtensionProtocol must model
        every nominated capability before numerical selection.
        The existing proposal and posterior-generating state are immutable.
        """
        if len(self.rediscoveries) >= self.max_rediscoveries or len(self.observations) != 1:
            return None
        # Queries need two requests; an observed v2 article needs one. Keep
        # one call slot for discovery and one for action modelling; the
        # protocol's bounded repair may consume the extra slot.
        minimum_requests = 1 if self.candidate_discovery_version == 2 else 2
        if self.request_budget - self.requests < minimum_requests or len(self.calls) >= self.semantic_call_budget - 2:
            return None
        public = self._discovery_public()
        public.update(frozen_proposal=deepcopy(self.proposal),
                      actual_observations=deepcopy(self.observations),
                      attempted_action_ids=[item["action_id"] for item in self.observations],
                      rediscovery_round=len(self.rediscoveries) + 1)
        discovery = self._semantic(self._discovery_protocol(), public, self._validate_discovery)
        new = self._query_candidates(discovery)
        if self.candidate_discovery_version == 2:
            by_id = {item["action_id"]: item for item in self._read_candidates()}
            new.extend({**by_id[item["action_id"]], "information_need": item["information_need"]}
                       for item in discovery["article_candidates"])
        record = {"version": REDISCOVERY_VERSION,
                  "round": len(self.rediscoveries) + 1,
                  "discovery": discovery,
                  "new_candidates": deepcopy(new),
                  "conditioned_on_observations": deepcopy(self.observations),
                  "history_sha256": digest(public),
                  "action_extension": None}
        if self.candidate_discovery_version == 2:
            record["catalog_coverage"] = deepcopy(public["catalog_coverage"])
        self.rediscoveries.append(record)
        save(self.output / "rediscoveries" / f"rediscovery-{len(self.rediscoveries):03d}.json", record)
        if not new:
            return record
        extension = self._extend(new, source="rediscovery")
        record["action_extension"] = deepcopy(extension)
        save(self.output / "rediscoveries" / f"rediscovery-{len(self.rediscoveries):03d}.json", record)
        return record

    def run(self):
        terminal = None
        status = "incomplete"
        try:
            if self.resume_provenance is None:
                self._acquire({"tool": "read_url", "url": self.bootstrap_url}, "declared_source_discovery_not_voi")
            discovery = (deepcopy(self._resumed_discovery) if self._resumed_discovery is not None else
                         self._semantic(self._discovery_protocol(), self._discovery_public(), self._validate_discovery))
            self.candidate_discovery = self._initialization_menu(discovery)
            menu = self.candidate_discovery["menu"]
            save(self.output / "candidate-discovery.json", self.candidate_discovery)
            if not menu:
                raise RuntimeError("no_executable_visible_candidates")
            self.candidates = {c["action_id"]: c for c in menu}
            proposal_protocol = (DirectionalProposalProtocol()
                                 if self.require_directional_evidence else ProposalProtocol())
            if self.require_directional_evidence:
                proposal_validator = lambda response: validate_directional_proposal(response, menu)
            else:
                proposal_validator = lambda response: validate_proposal(response, menu)
            self.proposal = self._semantic(proposal_protocol, self._public(menu), proposal_validator)
            save(self.output / "initial-proposal.json", self.proposal)
            self.model = self._new_model()
            while True:
                decision = self._decision()
                selected = decision["selected_action_id"]
                if selected is None:
                    terminal = decision
                    status = "completed"
                    break
                if len(self.calls) >= self.semantic_call_budget:
                    terminal = {**decision, "selected_action_id": None,
                                "stop_reason": "semantic_call_budget_exhausted"}
                    status = "completed_with_resource_stop"
                    break
                action = next(a for a in self.proposal["actions"] if a["action_id"] == selected)
                documents = self._execute_action(self.candidates[selected])
                applicable = [d for d in documents if d.get("status") == "OK" and d.get("kind") == "article" and d.get("text")]
                if applicable:
                    # The classifier sees ONLY this acquisition's bodies. Earlier
                    # text cannot be cited twice, and predicted probabilities
                    # cannot nudge the classifier toward the expected answer.
                    public = observation_public_projection(
                        match={**self.match, "decision_at": self.clock().isoformat()},
                        action=action, documents=applicable)
                    try:
                        protocol = (DirectionalObservationProtocol()
                                    if self.require_directional_evidence else ObservationProtocol())
                        validator = (lambda response: validate_directional_observation(
                            response, action, public["documents"])
                            if self.require_directional_evidence else
                            validate_observation(response, action, public["documents"]))
                        raw_observation = self._semantic(protocol, public, validator)
                        observation, directional_gate = self._gate_directional_observation(raw_observation, action)
                    except (ValueError, RuntimeError) as error:
                        observation = self._no_answer(action, "语义返回无法验证，不能更新概率："+str(error)[:1200])
                        directional_gate = {"status": "invalid", "reason": str(error)[:1200]}
                        self.failures.append({"stage": "classification", "error": str(error)})
                else:
                    observation = self._no_answer(action, "实际工具返回没有合格正文；标题、链接、失败和空返回均不构成本场证据。")
                    directional_gate = ({"status": "no_answer", "reason": "no_admissible_body"}
                                         if self.require_directional_evidence else None)
                record = {**observation, "returned_document_ids": [d["document_id"] for d in documents],
                          "requests_spent_after": self.requests, "semantic_classifier_verified_truth": False}
                if directional_gate is not None:
                    record["directional_gate"] = directional_gate
                self.observations.append(record)
                save(self.output / "observations" / f"observation-{len(self.observations):03d}.json", record)
                if self.request_budget-self.requests > 0:
                    # A first no-answer or informative observation can expose a
                    # new unknown. Discovery is bounded and returns candidate
                    # capabilities; their bodies are never prefetched here.
                    self._rediscover()
                    self._extend()
        except Exception as error:
            self.failures.append({"stage": "runtime", "error": f"{type(error).__name__}: {error}"})
            terminal = {"selected_action_id": None, "stop_reason": "runtime_incomplete",
                        "global_stop_certificate": False, "error": str(error)}
        # Even an extension/provider failure cannot erase accepted evidence.
        if self.model is not None and (not self.cycles or
                self.cycles[-1]["observations"] != self.observations or
                self.cycles[-1]["requests_spent"] != self.requests):
            self._decision()
        final = {"version": VERSION, "run_id": self.run_id, "configuration": self.configuration,
                 "status": status, "completed_at": self.clock().isoformat(),
                 "initial_prediction": self.baseline,
                 "final_prediction": self.cycles[-1]["decision"]["prediction_nominal"] if self.cycles else self.baseline,
                 "proposal": self.proposal, "observations": self.observations, "cycles": self.cycles,
                 "documents": self.documents, "extensions": self.extensions,
                 "candidate_discovery": self.candidate_discovery,
                 "resume_provenance": self.resume_provenance,
                 "rediscoveries": self.rediscoveries, "failures": self.failures,
                 "terminal": terminal, "http_receipts": self.receipts, "semantic_calls": self.calls,
                 "resources": {"http_requests": sum(r["returned"].get("http_attempts") or 0 for r in self.receipts),
                    "request_budget_reserved": self.requests,
                    "http_accounting_complete": all(r["returned"].get("http_attempts") is not None for r in self.receipts),
                    "semantic_calls": len(self.calls),
                    "semantic_usage": [c.get("receipt", {}).get("usage") for c in self.calls],
                    "http_latency_seconds": sum(r["returned"].get("latency_seconds", 0) for r in self.receipts),
                    "semantic_latency_seconds": sum(c.get("receipt", {}).get("latency_seconds", 0) for c in self.calls),
                    "actual_money": None, "LLM_overhead_in_voi_objective": False},
                 "evidence_scope": {"real_source_acquisition": True,
                    "scenario_probabilities": "uncalibrated_pretrained_model_assumptions",
                    "voi": ("exact_one_step_risk_integral_conditional_on_declared_models"
                            if self.planning_horizon == 1 else
                            "branchwise_robust_shared_two_step_policy_conditional_on_declared_models"),
                    "planning_horizon": self.planning_horizon,
                    "global_stop_certificate": False, "future_outcome_loaded": False,
                    "probability_improvement_proven": False}}
        save(self.output / "run.json", final)
        return final


def replay_math(run):
    """Recompute every saved finite decision with zero HTTP or semantic calls."""
    configuration = run.get("configuration", {})
    configured_horizon = configuration.get("planning_horizon", configuration.get("horizon", 1))
    if configured_horizon not in (1, 2):
        raise ValueError("Run configuration has an invalid planning horizon")
    for cycle in run["cycles"]:
        decision_horizon = cycle["decision"].get("horizon", 1)
        model_horizon = cycle["model"].get("planning_horizon", 1)
        if decision_horizon not in (1, 2) or model_horizon not in (1, 2):
            raise ValueError(f"Invalid planning horizon at cycle {cycle['index']}")
        if decision_horizon != configured_horizon or decision_horizon > model_horizon:
            raise ValueError(f"Planning horizon accounting differs at cycle {cycle['index']}")
        model = SemanticScenarioModel.from_manifest(cycle["model"])
        computed = model.evaluate(cycle["observations"], cycle["remaining_requests"],
                                  horizon=cycle["decision"].get("horizon", 1))
        if digest(computed) != digest(cycle["decision"]):
            raise ValueError(f"Decision replay mismatch at cycle {cycle['index']}")
        if cycle["remaining_requests"] + cycle["requests_spent"] != run["configuration"]["request_budget"]:
            raise ValueError("Cycle request accounting differs from configured budget")
    return {"status": "passed", "cycles": len(run["cycles"]), "http_requests": 0,
            "semantic_calls": 0, "scope": "exact_math_replay_of_frozen_proposals_and_saved_observations_not_reclassification"}
