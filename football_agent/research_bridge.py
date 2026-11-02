"""Private adapter between acquisition execution and platform/research services.

Only public states cross this boundary. The synthetic backend below is a test
fixture with fixed outputs, never a fitted predictor or an information estimator.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
from time import perf_counter
from threading import RLock
from typing import Any


class BackendUnavailable(RuntimeError):
    """A required server-side evidence/model configuration is missing."""


class PublicStateError(ValueError):
    """A service tried to expose information outside its public state."""


VISIBLE_KEYS = frozenset({
    "episode_id", "match_id", "prediction_cutoff", "information_mode", "base_facts",
    "groups", "mask", "evidence_versions", "quality", "action_history", "budget",
    "state_version",
})
FORBIDDEN_KEYS = frozenset({
    "target", "label", "labels", "outcome", "target_label", "full_snapshot",
    "hidden_groups", "hidden_features", "archive_path", "artifact_path", "source_path",
    "bundle_path", "home_goals", "away_goals", "full_time_score",
})
PLATFORM_GROUPS = {"strength": "elo", "recent_form": "recent_form", "rolling_xg": "xg"}
RESEARCH_GROUPS = {value: key for key, value in PLATFORM_GROUPS.items()}


def platform_state(value: Any) -> dict:
    """Project the platform-owned VisibleState, preserving mask quality separately."""
    from football_platform.information import VisibleState
    source = VisibleState.model_validate(value).model_dump(mode="json")
    groups, mask, statuses = {}, {}, {}
    for platform_id, research_id in PLATFORM_GROUPS.items():
        status = source["mask"].get(platform_id, "UNACQUIRED")
        mask[research_id] = status == "OBSERVED"
        statuses[research_id] = {"OBSERVED": "observed", "UNACQUIRED": "not_acquired",
                                 "SOURCE_MISSING": "source_missing", "STALE": "stale",
                                 "FAILED": "source_missing"}[status]
        if mask[research_id]:
            item = source["groups"][platform_id]
            groups[research_id] = {"status": "observed", "values": item["values"],
                                   "available_at": item["times"]["feature_available_at"],
                                   "version": item["version"], "evidence_refs": item["evidence_refs"],
                                   "source": item["source"], "quality": item["quality"],
                                   "platform_metadata": {key: val for key, val in item.items()
                                                         if key not in {"values", "version", "evidence_refs"}}}
    return public_state({"episode_id": source["episode_id"], "match_id": source["match_id"],
                         "prediction_cutoff": source["prediction_cutoff"], "information_mode": source["information_mode"],
                         "base_facts": {**source["fixture"], "base_prior": source["base_prior"], "prior_basis": source["prior_basis"]},
                         "groups": groups, "mask": mask,
                         "evidence_versions": {name: group["version"] for name, group in groups.items()},
                         "quality": {"group_statuses": statuses, "platform_mask": source["mask"],
                                     "protocol_version": source["protocol_version"], "catalog_version": source["catalog_version"],
                                     "cost_config_version": source["cost_config_version"], "experiment_role": source["experiment_role"],
                                     "platform_budget": source["budget"]},
                         "action_history": source["history"], "budget": source["budget"], "state_version": source["state_version"]})


def restore_platform_state(visible: dict):
    """Rehydrate only public fields; environment restore performs its own validation."""
    from football_platform.information import VisibleState
    state = public_state(visible)
    base = deepcopy(state["base_facts"])
    prior, basis = base.pop("base_prior"), base.pop("prior_basis")
    quality = state["quality"]
    groups = {}
    for name, group in state["groups"].items():
        metadata = deepcopy(group["platform_metadata"])
        metadata["times"]["feature_available_at"] = group["available_at"]
        groups[RESEARCH_GROUPS[name]] = {**metadata, "values": group["values"], "version": group["version"],
                                        "evidence_refs": group["evidence_refs"]}
    return VisibleState.model_validate({"episode_id": state["episode_id"], "match_id": state["match_id"],
        "prediction_cutoff": state["prediction_cutoff"], "information_mode": state["information_mode"],
        "fixture": base, "base_prior": prior, "prior_basis": basis,
        "groups": groups,
        "mask": quality["platform_mask"], "history": state["action_history"], "budget": quality["platform_budget"],
        "state_version": state["state_version"], "protocol_version": quality["protocol_version"],
        "catalog_version": quality["catalog_version"], "cost_config_version": quality["cost_config_version"],
        "experiment_role": quality["experiment_role"]})


def platform_actions(catalog: list, visible: dict, lambda_cost: dict) -> list[dict]:
    """Expose registered actions and component-labelled costs, never hidden coverage."""
    from football_platform.information import ActionSpec
    state = public_state(visible)
    result = []
    for item in catalog:
        source = ActionSpec.model_validate(item).model_dump(mode="json")
        groups = [PLATFORM_GROUPS.get(name, name) for name in source["group_ids"]]
        observed = groups and all(state["mask"].get(name, False) for name in groups)
        attempted = any(receipt["action_id"] == source["action_id"] for receipt in state["action_history"])
        status = source["capability_status"]
        reason = "" if status == "AVAILABLE" else status
        if observed or (attempted and not source["refresh_supported"]):
            status = "UNAVAILABLE"
            reason = "ALREADY_OBSERVED" if observed else "ATTEMPTED_IN_FROZEN_WINDOW"
        components = source["expected_cost"]
        vector = {key: component["amount"] for key, component in components.items()}
        money = components["money"]
        known = all(value is not None for value in vector.values())
        penalty = sum(lambda_cost.get(key, 0.0) * value for key, value in vector.items()) if known else None
        result.append({"action_id": source["action_id"], "parameters": {},
                       "parameters_schema": {"type": "object", "properties": {}, "additionalProperties": False},
                       "status": "AVAILABLE" if status == "AVAILABLE" else "UNAVAILABLE", "reason": reason,
                       "group_ids": groups, "cost": {"amount": money["amount"], "unit": money["unit"], "status": money["basis"]},
                       "cost_vector": vector, "cost_components": components, "cost_in_loss_units": penalty,
                       "source": source["source"], "tool": source["tool"], "version": state["quality"]["catalog_version"],
                       "evidence_version": state["quality"]["catalog_version"], "refresh_window": state["prediction_cutoff"],
                       "cost_scope": source["cost_scope"], "billing_key": source["billing_key"],
                       "dependency_key": source["dependency_key"], "cache_policy": source["cache_policy"],
                       "idempotency": source["idempotency"], "refresh_supported": source["refresh_supported"]})
    return result


def _json(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _check_public(value: Any) -> None:
    if isinstance(value, dict):
        rejected = FORBIDDEN_KEYS.intersection(value)
        if rejected:
            raise PublicStateError("Non-public fields: " + ", ".join(sorted(rejected)))
        for item in value.values():
            _check_public(item)
    elif isinstance(value, list):
        for item in value:
            _check_public(item)


def public_state(value: Any) -> dict:
    """Validate a private execution projection, including revealed-mask binding."""
    source = _json(value)
    unknown = set(source) - VISIBLE_KEYS
    if unknown:
        raise PublicStateError("Unexpected state fields: " + ", ".join(sorted(unknown)))
    _check_public(source)
    groups, mask = source.get("groups", {}), source.get("mask", {})
    if not isinstance(groups, dict) or not isinstance(mask, dict):
        raise PublicStateError("groups and mask must be maps")
    if any(not mask.get(key, False) for key in groups):
        raise PublicStateError("An unrevealed group entered the public state")
    if any(value is not True and value is not False for value in mask.values()):
        raise PublicStateError("mask values must be booleans")
    return source


def state_fingerprint(value: dict) -> str:
    return sha256(json.dumps(public_state(value), ensure_ascii=False, sort_keys=True,
                             separators=(",", ":")).encode()).hexdigest()


def validate_parameters(action: dict, parameters: dict) -> None:
    """Accept only declared primitive parameters; never pass arbitrary kwargs."""
    schema = action.get("parameters_schema", {"type": "object", "properties": {}})
    if not isinstance(parameters, dict):
        raise ValueError("Action parameters must be an object")
    properties = schema.get("properties", {})
    unknown = set(parameters) - set(properties)
    missing = set(schema.get("required", [])) - set(parameters)
    if unknown or missing:
        raise ValueError(f"Invalid action parameters: unknown={sorted(unknown)}, missing={sorted(missing)}")
    types = {"string": str, "number": (int, float), "integer": int,
             "boolean": bool, "object": dict, "array": list}
    for name, value in parameters.items():
        rule = properties[name]
        kind = rule.get("type")
        if kind in types and (not isinstance(value, types[kind]) or
                              (kind in {"number", "integer"} and isinstance(value, bool))):
            raise ValueError(f"Invalid type for parameter {name}")
        if "enum" in rule and value not in rule["enum"]:
            raise ValueError(f"Invalid enum for parameter {name}")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value < rule.get("minimum", value) or value > rule.get("maximum", value):
                raise ValueError(f"Out of bounds parameter {name}")


def research_state(visible: dict) -> dict:
    """Convert only the validated public projection to the published research API."""
    state = public_state(visible)
    if isinstance(state.get("quality"), dict) and state["quality"].get("protocol_version"):
        from football_research.acquisition.platform_bridge import to_research_state
        return to_research_state(restore_platform_state(state))
    from football_research.acquisition.state import GROUP_IDS, sanitize_state
    groups = {}
    statuses = state.get("quality", {})
    statuses = statuses.get("group_statuses", {}) if isinstance(statuses, dict) else {}
    for name in GROUP_IDS:
        if state["mask"].get(name):
            group = deepcopy(state["groups"][name])
            group["status"] = "observed"
            groups[name] = group
        else:
            status = statuses.get(name, "not_acquired")
            if status == "observed":
                raise PublicStateError("Observed status without a revealed mask")
            groups[name] = {"status": status}
    return sanitize_state({"match_id": state["match_id"], "prediction_cutoff": state["prediction_cutoff"],
                           "information_mode": state["information_mode"], "base": state["base_facts"],
                           "groups": groups})


class ResearchServices:
    """Use research-owned fitted predictors and OOF gain models, with frozen lambda."""

    def __init__(self, predictor, estimator, lambda_cost: dict):
        from football_research.acquisition.value import COST_UNITS
        if set(lambda_cost) - set(COST_UNITS):
            raise ValueError("Unknown loss conversion unit")
        self.lambda_cost = {key: float(lambda_cost.get(key, 0.0)) for key in COST_UNITS}
        if any(not math.isfinite(v) or v < 0 for v in self.lambda_cost.values()):
            raise ValueError("Loss conversion must be finite and nonnegative")
        self.predictor, self.estimator = predictor, estimator
        self._prediction_cache = {}

    @classmethod
    def load(cls, config_path):
        from football_research.acquisition.partial import PartialPredictor
        from football_research.acquisition.value import ValueEstimator
        path = Path(config_path).resolve()
        config = json.loads(path.read_text(encoding="utf-8"))
        predictor_path = (path.parent / config["partial_predictor"]).resolve()
        estimator_path = (path.parent / config["value_estimator"]).resolve()
        kind = config.get("value_estimator_kind", "empirical-v1")
        if kind == "joint-risk-planner-v1":
            from .joint_bridge import load_joint_services
            return load_joint_services(path, config)
        if kind == "calibrated-risk-v2":
            from football_research.acquisition.risk_calibrated import CalibratedRiskAwareValueEstimator
            estimator = CalibratedRiskAwareValueEstimator.load(estimator_path)
        elif kind == "contextual-value-v1":
            from football_research.acquisition.contextual_value import ContextualValueEstimator
            estimator = ContextualValueEstimator.load(estimator_path)
        elif kind == "source-identity-value-v1":
            from football_research.acquisition.contextual_identity import SourceIdentityValueEstimator
            estimator = SourceIdentityValueEstimator.load(estimator_path)
        elif kind == "empirical-v1":
            estimator = ValueEstimator.load(estimator_path)
        else:
            raise ValueError("Unrecognized value estimator kind")
        predictor_kind = config.get("partial_predictor_kind", "partial-subset-v1")
        if predictor_kind == "partial-subset-v1":
            predictor = PartialPredictor.load(predictor_path)
        elif predictor_kind == "contextual-partial-v1":
            from football_research.acquisition.contextual_partial import ContextualPartialPredictor
            predictor = ContextualPartialPredictor.load(predictor_path)
        else:
            raise ValueError("Unrecognized partial predictor kind")
        service = cls(predictor, estimator, config["lambda_cost"])
        service.artifact_hashes = {"configuration": sha256(path.read_bytes()).hexdigest(),
                                   "predictor": sha256(predictor_path.read_bytes()).hexdigest(),
                                   "estimator": sha256(estimator_path.read_bytes()).hexdigest()}
        return service

    def configuration(self):
        return {"predictor_version": self.predictor.model_version,
                "estimator_version": self.estimator.version,
                "predictor_training_cutoff": self.predictor.metadata["training_as_of"],
                "estimator_training_cutoff": self.estimator.manifest["trained_as_of"],
                "dataset_version": self.predictor.metadata["dataset_version"],
                "artifact_hashes": deepcopy(getattr(self, "artifact_hashes", {})),
                "lambda_cost": deepcopy(self.lambda_cost),
                "lambda_units": {key: "log_loss_nats_per_" + key for key in self.lambda_cost}}

    def predict(self, visible: dict):
        started = perf_counter()
        result = self.predictor.predict_partial(research_state(visible))
        self._prediction_cache[state_fingerprint(visible)] = deepcopy(result)
        probabilities = result["probabilities"]
        return {**result, "status": "OK", "probabilities": dict(zip(("home", "draw", "away"), probabilities)),
                "compute_cost": {"requests": 0, "money": 0, "tokens": 0,
                                 "latency_ms": (perf_counter() - started) * 1000}}

    def estimate(self, visible: dict, actions: list[dict]):
        research_actions = []
        for action in actions:
            vector = action.get("cost_vector")
            if not isinstance(vector, dict):
                raise ValueError("Action lacks explicit research cost vector")
            research_actions.append({"action_id": action["action_id"], "group_ids": action["group_ids"],
                                     "available": action["status"] == "AVAILABLE", "cost": deepcopy(vector),
                                     "cost_status": action["cost"]["status"],
                                     **{k: deepcopy(action[k]) for k in ("dependency_key", "cache_policy", "operation", "quality_policy", "cost_scope") if k in action}})
        key = state_fingerprint(visible)
        prediction_cost = None
        if key not in self._prediction_cache:
            prediction_cost = self.predict(visible)["compute_cost"]
        started = perf_counter()
        estimate_fn = getattr(self.estimator, "estimate_value", None) or self.estimator.estimate
        estimates = estimate_fn(research_state(visible), research_actions, prediction=self._prediction_cache[key])
        batch_ms = (perf_counter()-started)*1000
        converted = []
        action_cost_status = {action["action_id"]: action["cost"]["status"] for action in actions}
        for estimate in estimates:
            vector = estimate["cost"]
            required_cost = {"money", "requests", "latency_ms", "tokens"}
            estimate_cost_status = estimate.get("cost_status")
            if estimate_cost_status is None and isinstance(estimate.get("action"), dict):
                estimate_cost_status = estimate["action"].get("cost_status")
            if estimate_cost_status is None:
                estimate_cost_status = action_cost_status.get(estimate["action_id"])
            known = (set(vector) == required_cost
                     and all(isinstance(vector[key], (int, float)) and not isinstance(vector[key], bool)
                             and math.isfinite(vector[key]) and vector[key] >= 0 for key in required_cost)
                     and estimate_cost_status in {"known", "estimated", "simulated"})
            penalty = sum(self.lambda_cost[key] * vector[key] for key in self.lambda_cost) if known else None
            gain = estimate["gain"]
            supported = estimate.get("supported", estimate.get("support_matches", 0) >= getattr(self.estimator, "min_support", 12)
                                     and estimate.get("support_status") not in {"no_support", "parent_only"})
            converted.append({"action_id": estimate["action_id"], "expected_loss_improvement": gain,
                              "net_gain": gain - penalty if gain is not None and penalty is not None else None,
                              "cost_in_loss_units": penalty, "uncertainty": estimate["standard_error"],
                              "lower_gain": estimate["lower_gain"], "support": estimate["support_matches"],
                              "status": "OK" if supported and known else "UNAVAILABLE",
                              "reason": estimate["reason"] if known else "unknown_cost",
                              "estimator_version": estimate["version"], "research_estimate": estimate,
                              "support_status": estimate.get("support_status"), "gate_reason": estimate.get("gate_reason"),
                              "predictive_lower_gain": estimate.get("predictive_lower_gain"),
                              "predictive_upper_gain": estimate.get("predictive_upper_gain"),
                              "uncertainty_kind": estimate.get("uncertainty_kind"),
                              "compute_cost": estimate.get("estimator_cost", {"requests": 0, "money": 0, "tokens": 0,
                                  "latency_ms": batch_ms if not converted else 0})})
        if converted and prediction_cost:
            converted[0]["additional_prediction_compute_cost"] = prediction_cost
        return converted


class FrozenResearchBackend:
    """Serialize environment transitions; predictor/estimator see public dicts only."""

    def __init__(self, environment, services: ResearchServices, *, configuration: dict, fixtures: list[dict] | None = None):
        self._environment, self._services = environment, services
        self._configuration = deepcopy(configuration)
        self._fixtures = deepcopy(fixtures or [])
        self._lock = RLock()

    def configuration(self):
        return {"backend": "platform_frozen_research", "mode": "historical_controlled_reveal",
                "bridge_version": "platform-visible-research-v1", **deepcopy(self._configuration),
                **self._services.configuration()}

    def matches(self):
        # Server-provided schedule metadata is safe to enumerate; start validates
        # identity/cutoff against the actual environment before a run executes.
        return deepcopy(self._fixtures)

    def start(self, match_id: str, cutoff: str, seed: int = 0):
        with self._lock:
            source = self._environment.start(match_id, cutoff, information_mode="historical_reconstruction")
            return platform_state(source)

    def actions(self, visible: dict):
        with self._lock:
            return platform_actions(self._environment.catalog(), visible, self._services.lambda_cost)

    def predict(self, visible: dict):
        return self._services.predict(visible)

    def prediction_conditioning(self, visible: dict):
        """Joint-only public conditioning identity, including modeled missingness."""
        observer = getattr(self._services.predictor, "observed_categories", None)
        if not callable(observer):
            raise BackendUnavailable("BACKEND_HAS_NO_JOINT_CONDITIONING_CONTRACT")
        return observer(research_state(visible))

    def estimate(self, visible: dict, actions: list[dict]):
        return self._services.estimate(visible, actions)

    def acquire(self, visible: dict, action_id: str, parameters: dict, request_id: str):
        started = perf_counter()
        with self._lock:
            action = next((a for a in self.actions(visible) if a["action_id"] == action_id), None)
            if action is None or action["status"] != "AVAILABLE":
                raise ValueError("Action is not in the executable catalog")
            validate_parameters(action, parameters)
            restore = getattr(self._environment, "restore", None)
            if not callable(restore):
                raise BackendUnavailable("ENVIRONMENT_RESTORE_NOT_READY: paid actions are never replayed to reconstruct state")
            restore(restore_platform_state(visible))
            result = _json(self._environment.acquire(action_id, request_id))
            state = platform_state(self._environment.visible_state())
            money = result["charged_cost"]["money"]
            status = {"ACQUIRED": "OK", "CACHE_HIT": "OK", "UNCHANGED": "NO_CHANGE"}.get(result["status"], result["status"])
            return {"status": status, "visible_state": state,
                    "cost": {"amount": money["amount"], "unit": money["unit"], "status": money["basis"]},
                    "cost_components": result["charged_cost"], "latency_seconds": perf_counter() - started,
                    "requests": 0, "request_id": request_id, "action_id": action_id,
                    "evidence_refs": result["evidence_refs"], "reason": result.get("reason"),
                    "retry_after_seconds": result.get("retry_after_seconds"), "platform_result": result}


class SyntheticBackend:
    """Deterministic flow fixture; every probability/gain is marked synthetic."""

    def __init__(self, scenario: str = "normal"):
        if scenario not in {"normal", "immediate_stop", "no_data", "cost_exceeds_gain"}:
            raise ValueError("Unknown synthetic scenario")
        self.scenario = scenario

    def configuration(self) -> dict:
        return {"backend": "synthetic_fixture", "mode": "synthetic_test", "scenario": self.scenario,
                "data_version": "synthetic-reveal-fixture-v1", "catalog_version": "fixture-actions-v1",
                "predictor_version": "fixture-probabilities-v1", "estimator_version": "fixture-gains-v1",
                "cost_table_version": "fixture-simulated-cost-v1", "cost_unit": "simulation_credit",
                "cost_status": "simulated", "description": "构造测试：概率和增益为固定测试替身"}

    def matches(self) -> list[dict]:
        return [{"match_id": "synthetic:alpha-beta", "prediction_cutoff": "2025-01-01T12:00:00+00:00",
                 "home_team": "合成甲队", "away_team": "合成乙队", "mode": "synthetic_test"}]

    def start(self, match_id: str, cutoff: str, seed: int = 0) -> dict:
        if match_id != "synthetic:alpha-beta":
            raise ValueError("Unknown synthetic match")
        return public_state({"episode_id": f"synthetic-{seed}", "match_id": match_id,
                             "prediction_cutoff": cutoff, "information_mode": "synthetic_test",
                             "base_facts": {"home_team": "合成甲队", "away_team": "合成乙队"},
                             "groups": {}, "mask": {"strength": False, "form": False, "xg": False},
                             "evidence_versions": {}, "quality": ["SYNTHETIC_TEST"],
                             "action_history": [], "budget": {}, "state_version": "fixture-state-v1"})

    def actions(self, visible: dict) -> list[dict]:
        state = public_state(visible)
        completed = {x["action_id"] for x in state["action_history"]}
        result = []
        for group in ("strength", "form", "xg"):
            action_id = "fixture.reveal." + group
            available = action_id not in completed and not state["mask"][group]
            result.append({"action_id": action_id, "parameters": {}, "parameters_schema": {"type": "object", "properties": {}, "additionalProperties": False},
                           "status": "AVAILABLE" if available else "UNAVAILABLE",
                           "reason": "" if available else "ALREADY_ATTEMPTED", "group_ids": [group],
                           "cost": {"amount": 1.0, "unit": "simulation_credit", "status": "simulated"},
                           "cost_in_loss_units": 0.005, "version": "fixture-actions-v1",
                           "evidence_version": "fixture-v1", "refresh_window": "once"})
        for group in ("injuries", "weather"):
            result.append({"action_id": "fixture.unavailable." + group, "parameters": {}, "parameters_schema": {"type": "object", "properties": {}},
                           "status": "UNAVAILABLE", "reason": "SOURCE_NOT_CONNECTED", "group_ids": [group],
                           "cost": {"amount": None, "unit": "simulation_credit", "status": "unknown"},
                           "version": "fixture-actions-v1"})
        return result

    def predict(self, visible: dict) -> dict:
        state = public_state(visible)
        count = sum(state["mask"].values())
        table = [(0.40, 0.30, 0.30), (0.45, 0.28, 0.27), (0.46, 0.28, 0.26), (0.46, 0.28, 0.26)]
        home, draw, away = table[count]
        return {"status": "OK", "probabilities": {"home": home, "draw": draw, "away": away},
                "model_version": "fixture-probabilities-v1", "mode": "synthetic_test",
                "mask": deepcopy(state["mask"]), "state_hash": state_fingerprint(state)}

    def estimate(self, visible: dict, actions: list[dict]) -> list[dict]:
        state = public_state(visible)
        result = []
        for action in actions:
            group = action["group_ids"][0]
            gain = {"strength": 0.04, "form": 0.02, "xg": 0.01}.get(group, 0.0)
            if state["mask"].get("strength"):
                gain = {"form": 0.003, "xg": 0.025}.get(group, 0.0)
            if self.scenario == "immediate_stop":
                gain = -0.001
            elif self.scenario == "cost_exceeds_gain":
                gain = 0.001
            cost = action.get("cost_in_loss_units", 0.005)
            result.append({"action_id": action["action_id"], "expected_loss_improvement": gain,
                           "net_gain": gain - cost, "cost_in_loss_units": cost, "uncertainty": 0.0,
                           "support": 0, "estimator_version": "fixture-gains-v1", "status": "OK",
                           "mode": "synthetic_test", "explanation": "固定测试值，非经验VoI估计"})
        return result

    def acquire(self, visible: dict, action_id: str, parameters: dict, request_id: str) -> dict:
        started = perf_counter()
        state = public_state(visible)
        action = next((item for item in self.actions(state) if item["action_id"] == action_id), None)
        if action is None or action["status"] != "AVAILABLE":
            raise ValueError("Action is not in the executable catalog")
        validate_parameters(action, parameters)
        group = action["group_ids"][0]
        status = "NO_DATA" if self.scenario == "no_data" and group == "strength" else "OK"
        if status == "OK":
            state["groups"][group] = {"fixture_value": {"strength": 0.7, "form": 0.4, "xg": 0.6}[group],
                                      "evidence_refs": ["synthetic-evidence:" + group]}
            state["mask"][group] = True
            state["evidence_versions"][group] = "fixture-v1"
        state["action_history"].append({"action_id": action_id, "request_id": request_id, "status": status})
        return {"status": status, "visible_state": public_state(state), "cost": action["cost"],
                "latency_seconds": perf_counter() - started, "requests": 0,
                "action_id": action_id, "request_id": request_id,
                "evidence_refs": ["synthetic-evidence:" + group] if status == "OK" else []}


def make_backend(*, bundle_path=None, model_artifact=None, mode="real", scenario="normal"):
    if mode == "synthetic_test":
        return SyntheticBackend(scenario=scenario)
    if not bundle_path or not model_artifact:
        raise BackendUnavailable("PLATFORM_RESEARCH_INTERFACE_NOT_READY: configure the published evidence environment and fitted acquisition artifact")
    try:
        from football_platform.information import FrozenEvidenceEnvironment, ExperimentConfig
    except ImportError as exc:
        raise BackendUnavailable("PLATFORM_ENVIRONMENT_NOT_READY") from exc
    config_path = Path(model_artifact).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    experiment = ExperimentConfig.model_validate(config["experiment"])
    environment = FrozenEvidenceEnvironment.from_export(bundle_path, experiment)
    services = ResearchServices.load(config_path)
    return FrozenResearchBackend(environment, services, configuration={
        "cost_table_version": experiment.version, "experiment": experiment.model_dump(mode="json"),
        "recommended_budget": {"max_simulated_cost": experiment.budget.get("money", 0.05),
                               "max_actions": int(experiment.budget.get("requests", 2))},
        "research_release": config.get("research_release"),
        "lambda_validation_budget": config.get("lambda_validation_budget"),
        "group_mapping": PLATFORM_GROUPS, "configuration_sha256": sha256(config_path.read_bytes()).hexdigest(),
        "bundle_complete_sha256": sha256((Path(bundle_path) / "COMPLETE.json").read_bytes()).hexdigest()},
        fixtures=config.get("fixtures", []))
