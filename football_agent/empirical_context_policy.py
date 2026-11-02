"""Frozen empirical context policy for the existing demand-driven query loop.

Values are recomputed after every context change. Historical outcomes remain
inside the calibration snapshot; only aggregate assessments enter the gate.
This one-step policy does not claim optimal exploration or off-policy validity.
"""
from copy import deepcopy
from collections.abc import Mapping
import hashlib
import math
from types import MappingProxyType

from .adaptive_voi_calibration import calibrate_candidate_value
from .audit import json_bytes
from .contextual_adaptive_voi import context_fingerprint, normalize_context
from .query_value_contract import QueryValueAssessment, QueryValueGate
from .types import utc
from .voi_evidence_bridge import VoIEvidenceLink

VERSION = "frozen-empirical-context-policy-v1"


class FrozenEmpiricalContextPolicy:
    """Project a prespecified context and use only matching, available history.

    Historical contexts require every selected key explicitly (None means
    unknown). Omitted history provenance never becomes known zero value.
    Source, source version, candidate, action, context and availability must
    all agree. Current-target history is excluded even if its timestamp fits.
    """

    def __init__(self, *, routes, context_keys, links, context_by_link_id,
                 available_at_by_link_id, calibration_as_of,
                 min_independent_dependencies=5, confidence=0.10):
        if not isinstance(routes, Mapping) or not routes:
            raise ValueError("routes must be a nonempty mapping")
        if isinstance(context_keys, (str, bytes)):
            raise ValueError("context_keys must be a sequence")
        keys = tuple(context_keys)
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("context_keys must be nonempty and unique")
        normalize_context({key: None for key in keys})
        if type(min_independent_dependencies) is not int or min_independent_dependencies < 2:
            raise ValueError("minimum independent support must be at least two")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 < confidence < 1:
            raise ValueError("confidence must be between zero and one")
        cleaned_routes = {}
        for candidate_id, route in routes.items():
            if not isinstance(candidate_id, str) or not candidate_id.strip() or not isinstance(route, Mapping):
                raise ValueError("routes require named candidate mappings")
            cleaned = {}
            for name in ("source_id", "source_version", "action_id"):
                value = route.get(name)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"route needs {name}")
                cleaned[name] = value
            cost = route.get("cost")
            if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0:
                raise ValueError("route cost must be finite and nonnegative")
            cleaned["cost"] = float(cost)
            cleaned_routes[candidate_id] = MappingProxyType(cleaned)
        materialized = tuple(links)
        if any(not isinstance(item, VoIEvidenceLink) for item in materialized):
            raise ValueError("links must contain VoIEvidenceLink")
        if len({item.link_id for item in materialized}) != len(materialized):
            raise ValueError("calibration link IDs must be unique")
        if not isinstance(context_by_link_id, Mapping) or not isinstance(available_at_by_link_id, Mapping):
            raise ValueError("explicit context and availability inventories are required")
        contexts = {}
        availability = {}
        for item in materialized:
            raw = context_by_link_id.get(item.link_id)
            if isinstance(raw, Mapping) and all(key in raw for key in keys):
                contexts[item.link_id] = MappingProxyType(normalize_context({key: raw[key] for key in keys}))
            stamp = available_at_by_link_id.get(item.link_id)
            if stamp is not None:
                availability[item.link_id] = utc(stamp).isoformat()
        self._routes = MappingProxyType(cleaned_routes)
        self._context_keys = keys
        self._links = materialized
        self._contexts = MappingProxyType(contexts)
        self._availability = MappingProxyType(availability)
        self._as_of = utc(calibration_as_of).isoformat()
        self._minimum = min_independent_dependencies
        self._confidence = float(confidence)
        self._history = []
        self._configuration = self._build_configuration()

    def _build_configuration(self):
        snapshot = {
            "links": [item.as_dict() for item in self._links],
            "contexts": {key: dict(value) for key, value in self._contexts.items()},
            "availability": dict(self._availability),
        }
        return {
            "version": VERSION, "context_keys": list(self._context_keys),
            "routes": {key: dict(value) for key, value in self._routes.items()},
            "calibration_as_of": self._as_of, "min_independent_dependencies": self._minimum,
            "confidence": self._confidence,
            "calibration_snapshot_hash": hashlib.sha256(json_bytes(snapshot)).hexdigest(),
            "value_scope": "same candidate/action/source/version and exact projected context; bounded risk-unit one-step gain",
            "cost_unit": "same normalized risk units as expected information value",
        }

    def configuration(self):
        return deepcopy(self._configuration)

    def assessments_history(self):
        return deepcopy(self._history)

    def __call__(self, board, context, remaining_budget):
        if self._build_configuration() != self._configuration:
            raise ValueError("calibration configuration changed after snapshot creation")
        if utc(self._as_of) > utc(board.prediction_cutoff):
            raise ValueError("calibration snapshot follows prediction cutoff")
        projection = normalize_context({key: context.get(key) for key in self._context_keys})
        context_key = context_fingerprint(projection)
        cutoff = utc(self._as_of)
        assessments, summaries = [], []
        for candidate in board.actionable_candidates(view="agent"):
            route = self._routes.get(candidate.candidate_id)
            if route is None:
                assessments.append(QueryValueAssessment.unmeasured(candidate.candidate_id, cost=candidate.cost,
                                    reason="route_not_in_frozen_calibration"))
                continue
            if not math.isclose(candidate.cost, route["cost"], abs_tol=1e-12, rel_tol=0):
                raise ValueError("route cost differs from frozen calibration")
            selected = []
            for item in self._links:
                stamp = self._availability.get(item.link_id)
                declared = self._contexts.get(item.link_id)
                if item.target_key == board.target_key or item.candidate_id != candidate.candidate_id or item.action_id != route["action_id"]:
                    continue
                if item.source_id != route["source_id"] or item.source_version != route["source_version"]:
                    continue
                if stamp is None or utc(stamp) > cutoff or item.requested_at is None or utc(item.requested_at) > utc(stamp):
                    continue
                if declared is None or context_fingerprint(declared) != context_key:
                    continue
                selected.append(item)
            estimate = calibrate_candidate_value(
                candidate.candidate_id, source_id=route["source_id"], source_version=route["source_version"],
                cost=route["cost"], links=selected, as_of=self._as_of,
                min_independent_dependencies=self._minimum, confidence=self._confidence,
            )
            assessments.append(estimate.as_query_assessment())
            summaries.append(estimate.as_dict())
        decision = QueryValueGate.decide(board, assessments, remaining_budget=remaining_budget, view="agent")
        self._history.append({"target_key": board.target_key, "prediction_cutoff": board.prediction_cutoff,
                              "context": projection, "context_key": context_key, "remaining_budget": remaining_budget,
                              "estimates": summaries, "decision": decision.as_dict()})
        return decision


def frozen_loop_configuration(*, board, routes, policy, budget, max_steps, tool_version):
    """Materialize the exact policy and tool-menu configuration to register."""
    if not isinstance(policy, FrozenEmpiricalContextPolicy):
        raise ValueError("policy must be FrozenEmpiricalContextPolicy")
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(budget) or budget < 0:
        raise ValueError("budget must be finite and nonnegative")
    if not isinstance(tool_version, str) or not tool_version.strip():
        raise ValueError("tool_version must be explicit")
    return {
        "policy": policy.configuration(),
        "candidate_catalog": [
            {key: getattr(candidate, key) for key in (
                "candidate_id", "state_variable", "source_role", "query_kind", "cost", "endpoint", "action_id",
            )} for candidate in sorted(board.candidates, key=lambda item: item.candidate_id)
        ],
        "context_routes": [deepcopy(dict(row)) for row in routes],
        "budget": float(budget), "max_steps": max_steps, "tool_version": tool_version,
        "cost_semantics": "declared cost in normalized risk units reserved per attempt; not actual API billing",
    }


def run_frozen_empirical_context_loop(*, board, initial_context, routes, policy,
                                     freeze_manifest, execute, episode_id,
                                     budget, tool_version, max_steps=8, root=None):
    """Run only the policy actually bound by a Round86-compatible freeze.

    The caller supplies a pre-registered manifest; this function never creates
    or backdates one. Hash consistency is not timestamp attestation. This is
    the acquisition loop, not a fabricated final-probability episode exporter.
    """
    from football_research.evaluation.policy_experiment_protocol import audit_parameter_freeze, manifest_hash
    from .autonomous_context_loop import run_autonomous_context_loop

    freeze_manifest = deepcopy(freeze_manifest)
    audit = audit_parameter_freeze(freeze_manifest, earliest_test_prediction_cutoff=board.prediction_cutoff)
    if audit["status"] != "PASS":
        raise ValueError("parameter freeze requires review before tool execution")
    route_rows = tuple(deepcopy(dict(row)) for row in routes)
    expected = frozen_loop_configuration(board=board, routes=route_rows, policy=policy,
                                         budget=budget, max_steps=max_steps, tool_version=tool_version)
    bound = freeze_manifest["parameters"].get("autonomous_context_loop")
    # Canonical hashes also normalize list/tuple representations after JSON IO.
    if not isinstance(bound, Mapping) or manifest_hash(bound) != manifest_hash(expected):
        raise ValueError("actual acquisition loop differs from frozen configuration")
    configuration = policy.configuration()
    if utc(configuration["calibration_as_of"]) > utc(freeze_manifest["frozen_at"]):
        raise ValueError("calibration snapshot follows parameter freeze")
    required_thresholds = {"min_independent_dependencies": configuration["min_independent_dependencies"],
                           "confidence": configuration["confidence"], "stop_net_value": 0.0}
    if freeze_manifest["thresholds"] != required_thresholds:
        raise ValueError("frozen thresholds differ from the actual query gate configuration")
    history_start = len(policy.assessments_history())
    result = run_autonomous_context_loop(
        board=board, initial_context=initial_context, routes=route_rows,
        episode_id=episode_id, root=root, max_steps=max_steps, budget=budget,
        decision_provider=policy, execute=execute, tool_version=tool_version,
    )
    return {"schema_version": VERSION, "policy_parameter_hash": freeze_manifest["parameter_hash"],
            "policy_configuration": configuration, "loop": result.as_dict(),
            "value_assessments": policy.assessments_history()[history_start:], "independent_confirmation": False,
            "freeze_verification_scope": audit["verification_scope"]}


__all__ = ["VERSION", "FrozenEmpiricalContextPolicy", "frozen_loop_configuration", "run_frozen_empirical_context_loop"]
