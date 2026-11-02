"""Independent bridge for empirical one-step gates over a frozen joint predictor.

The predictor and calibrator are distinct artifacts. Their explicit reference
binding replaces the same-file rule used by the older raw joint planner.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from football_research.acquisition.state import parse_time

from .joint_bridge import physical_budget
from .research_bridge import FrozenResearchBackend, PLATFORM_GROUPS, ResearchServices, public_state, research_state


GATE_KIND = "calibrated-joint-one-step-v1"
CALIBRATOR_KIND = "residual-evsi-calibrator-v1"
UNCERTAINTY_KIND = "descriptive_week_block_standard_error_without_coverage_guarantee"
FAILURES = {"FAILED", "TIMEOUT", "PROVIDER_ERROR", "HTTP_ERROR", "NETWORK_ERROR", "BAD_DATA", "STALE", "UNAVAILABLE"}


def _real_state(visible):
    public = public_state(visible)

    def check(value):
        if isinstance(value, dict):
            if "joint_return_hypothesis" in value:
                raise ValueError("Hypothetical return tokens are outside the real calibrated gate")
            for item in value.values():
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)

    check(public)
    for receipt in public.get("action_history", []):
        if not isinstance(receipt, dict):
            raise ValueError("Public action history must contain structured receipts")
        nested = receipt.get("result")
        statuses = [receipt.get("status", ""), nested.get("status", "") if isinstance(nested, dict) else ""]
        if any(token in str(status).upper() for status in statuses for token in FAILURES):
            raise ValueError("Provider failure is outside the learned return model")
    return research_state(public)


class CalibratedJointServices(ResearchServices):
    def __init__(self, model, calibrator, *, lambda_cost, risk_level):
        if risk_level not in {"mean", "conservative"}:
            raise ValueError("Calibrated risk level must be mean or conservative")
        manifest = calibrator.manifest
        if manifest["joint_model_version"] != model.model_version or parse_time(manifest["joint_training_as_of"]) != parse_time(model.metadata["training_as_of"]):
            raise ValueError("Calibrator reference model version/training cutoff mismatch")
        if parse_time(manifest["trained_as_of"]) > parse_time(manifest["diagnostics_as_of"]):
            raise ValueError("Calibrator diagnostics must follow its parameter fitting")
        super().__init__(model, calibrator, lambda_cost)
        self.risk_level = risk_level

    def configuration(self):
        return {**super().configuration(), "partial_predictor_kind": "learned-joint-returns-v1",
                "value_estimator_kind": CALIBRATOR_KIND,
                "calibrated_gate": {"kind": GATE_KIND, "risk_level": self.risk_level, "horizon": 1,
                                    "gain_semantics": "empirically_calibrated_single_acquisition_log_loss_reduction",
                                    "uncertainty_kind": UNCERTAINTY_KIND,
                                    "selection_score_decimal_places": 12,
                                    "physical_budget_semantics": "simulation_credit_and_projection_calls_unscaled_by_lambda"},
                "reference_joint": {key: self.estimator.manifest[key] for key in
                                    ("joint_model_version", "joint_artifact_sha256", "trained_as_of", "diagnostics_as_of")}}

    def predict(self, visible):
        self._state(visible)
        return super().predict(visible)

    def _state(self, visible):
        state = _real_state(visible)
        cutoff = parse_time(state["prediction_cutoff"])
        if any(parse_time(value) > cutoff for value in (self.predictor.metadata["training_as_of"], self.estimator.manifest["trained_as_of"])):
            raise ValueError("Predictor or calibrator was trained after the requested runtime state")
        return state

    def estimate(self, visible, actions):
        started = perf_counter()
        state = self._state(visible)
        candidates = [{"action_id": a["action_id"], "group_ids": deepcopy(a["group_ids"]),
                       "available": a["status"] == "AVAILABLE" and a.get("available", True),
                       "cost": deepcopy(a["cost_vector"]), "cost_status": a["cost"]["status"],
                       **{k: deepcopy(a[k]) for k in ("dependency_key", "cache_policy", "cost_scope", "evidence_id") if k in a}}
                      for a in actions]
        plan = self.estimator.plan(state, candidates, physical_budget(visible),
                                   risk_level=self.risk_level, lambda_cost=self.lambda_cost)
        elapsed = (perf_counter() - started) * 1000
        values = []
        for native in plan["candidates"]:
            gain, lower = native.get("expected_gain"), native.get("lower_gain")
            supported = native.get("feasible", True) and gain is not None
            if self.risk_level == "conservative" and lower is None:
                supported = False
            penalty = native.get("cost_penalty")
            values.append({"action_id": native["action_id"], "status": "OK" if supported else "UNAVAILABLE",
                           "reason": native.get("reason") or native.get("gate_reason"),
                           "expected_loss_improvement": gain, "lower_gain": lower,
                           "cost_in_loss_units": penalty,
                           "net_gain": None if gain is None or penalty is None else round(gain - penalty, 12),
                           "support": native.get("support"), "parent_support": native.get("parent_support"),
                           "effective_support": native.get("effective_support"),
                           "support_status": native.get("support_status"), "gate_reason": native.get("gate_reason"),
                           "fallback_reason": native.get("fallback_reason"), "standard_error": native.get("standard_error"),
                           "uncertainty_kind": UNCERTAINTY_KIND, "uncertainty": None,
                           "estimator_version": self.estimator.version,
                           "research_estimate": {**deepcopy(native), "planned_action": plan["action_id"],
                                                 "plan_reason": plan["reason"], "risk_level": self.risk_level,
                                                 "physical_budget": plan["physical_budget"], "lambda_cost": plan["lambda_cost"]},
                           "compute_cost": {"money": 0, "requests": 0, "tokens": 0, "latency_ms": elapsed if not values else 0}})
        return values


def load_calibrated_joint_services(config_path):
    from football_research.acquisition.joint_returns import JointReturnPredictor
    from football_research.acquisition.round9_residual import ResidualEVSICalibrator

    path = Path(config_path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("partial_predictor_kind") != "learned-joint-returns-v1" or config.get("value_estimator_kind") != CALIBRATOR_KIND:
        raise ValueError("Explicit frozen joint predictor and residual calibrator kinds are required")
    gate = config.get("gate")
    if not isinstance(gate, dict) or set(gate) != {"kind", "risk_level"} or gate.get("kind") != GATE_KIND:
        raise ValueError("Explicit single-step calibrated gate configuration is required")
    if gate["risk_level"] not in {"mean", "conservative"}:
        raise ValueError("Calibrated risk level must be mean or conservative")
    model_path = (path.parent / config["partial_predictor"]).resolve()
    value_path = (path.parent / config["value_estimator"]).resolve()
    model_hash = sha256(model_path.read_bytes()).hexdigest()
    model = JointReturnPredictor.load(model_path)
    calibrator = ResidualEVSICalibrator.load(value_path, model=model, model_sha256=model_hash)
    service = CalibratedJointServices(model, calibrator, lambda_cost=config["lambda_cost"], risk_level=gate["risk_level"])
    service.artifact_hashes = {"configuration": sha256(path.read_bytes()).hexdigest(), "predictor": model_hash,
                               "estimator": sha256(value_path.read_bytes()).hexdigest()}
    return service


def make_calibrated_joint_backend(*, bundle_path, model_artifact):
    """Build the real frozen replay backend without changing legacy factories."""
    from football_platform.information import ExperimentConfig, FrozenEvidenceEnvironment

    path = Path(model_artifact).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    experiment = ExperimentConfig.model_validate(config["experiment"])
    services = load_calibrated_joint_services(path)
    environment = FrozenEvidenceEnvironment.from_export(bundle_path, experiment)
    return FrozenResearchBackend(environment, services, configuration={
        "cost_table_version": experiment.version, "experiment": experiment.model_dump(mode="json"),
        "recommended_budget": {"max_simulated_cost": experiment.budget.get("money", .05),
                               "max_actions": int(experiment.budget.get("requests", 2))},
        "research_release": config.get("research_release"), "lambda_validation_budget": config.get("lambda_validation_budget"),
        "group_mapping": PLATFORM_GROUPS, "configuration_sha256": sha256(path.read_bytes()).hexdigest(),
        "bundle_complete_sha256": sha256((Path(bundle_path) / "COMPLETE.json").read_bytes()).hexdigest()},
        fixtures=config.get("fixtures", []))
