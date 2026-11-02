"""Bind calibrated two-query continuation scores to the actual Agent executor."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path

from .calibrated_joint_bridge import CalibratedJointServices
from .calibrated_joint_workflow import CalibratedJointAcquisitionWorkflow
from .joint_workflow import JointAcquisitionWorkflow
from .query_decision_card import build_query_decision_cards
from .research_bridge import FrozenResearchBackend, PLATFORM_GROUPS, ResearchServices


ESTIMATOR_KIND = "calibrated-continuation-planner-v1"
GATE_KIND = "calibrated-continuation-two-step-v1"
MODES = ("adaptive", "commit")
GAIN_SEMANTICS = "expected_sum_of_calibrated_update_gains_along_conditional_plan"
COST_SEMANTICS = "expected_total_plan_cost_in_loss_units_not_immediate_charge"
MODEL_AUDIT_SCOPE = "model_only_mathematical_categories_not_tool_receipts_or_observed_evidence"


class CalibratedContinuationServices(CalibratedJointServices):
    def __init__(self, model, estimator, *, lambda_cost):
        if estimator.mode not in MODES or estimator.manifest.get("horizon") != 2:
            raise ValueError("Explicit continuation mode and horizon two required")
        super().__init__(model, estimator, lambda_cost=lambda_cost, risk_level="mean")

    def configuration(self):
        manifest = self.estimator.manifest
        return {**ResearchServices.configuration(self), "partial_predictor_kind": "learned-joint-returns-v1",
                "value_estimator_kind": ESTIMATOR_KIND,
                "continuation_gate": {"kind": GATE_KIND, "risk_level": "mean", "horizon": 2,
                    "mode": self.estimator.mode, "gain_semantics": GAIN_SEMANTICS, "cost_semantics": COST_SEMANTICS,
                    "planning_assumption": "calibrated_gain_plus_frozen_return_surrogate_not_coherent_bayes_risk",
                    "post_return_decision": "conditional_mean" if self.estimator.mode == "adaptive" else "content_marginal_mean",
                    "uncertainty_kind": "aggregate_uncertainty_not_estimated",
                    "physical_budget_semantics": "simulation_credit_and_projection_calls_unscaled_by_lambda",
                    "selection_score_decimal_places": 12},
                "reference_joint": {key: manifest[key] for key in
                    ("joint_model_version", "joint_artifact_sha256", "trained_as_of", "diagnostics_as_of")},
                "reference_source_calibrator": {"version": manifest["source_calibrator_version"],
                    "sha256": manifest["source_calibrator_sha256"], "kind": "residual-evsi-calibrator-v1"}}

    def estimate(self, visible, actions):
        values = super().estimate(visible, actions)
        for value in values:
            native = value["research_estimate"]
            if "expected_gain" in native:
                if native.get("estimator_kind") != ESTIMATOR_KIND or any(native.get(k) is not None for k in
                        ("model_evsi", "evsi_bin", "lower_gain", "upper_gain", "standard_error", "uncertainty")):
                    raise ValueError("Continuation planner identity or null uncertainty contract differs")
                if native["expected_gain"] is not None and (abs(native["expected_gain"] -
                        native["immediate_expected_gain"] - native["expected_continuation_gain"]) > 1e-10
                        or abs(native["cost_penalty"] - native["immediate_cost_penalty"] -
                               native["expected_continuation_cost_penalty"]) > 1e-10
                        or abs(native["mean_net_gain"] - native["expected_gain"] + native["cost_penalty"]) > 1e-10):
                    raise ValueError("Plan totals must equal immediate plus expected continuation quantities")
                if value["status"] == "OK":
                    # Preserve the planner's declared rounding/tie convention;
                    # adding then subtracting totals can straddle a 12-digit tie.
                    value["net_gain"] = native["selection_score"]
            value.update(estimator_kind=ESTIMATOR_KIND, continuation_mode=self.estimator.mode,
                         gain_semantics=GAIN_SEMANTICS, cost_semantics=COST_SEMANTICS,
                         model_evsi=None, uncertainty_kind="aggregate_uncertainty_not_estimated")
        return values


def load_calibrated_continuation_services(config_path):
    from football_research.acquisition.calibrated_continuation import CalibratedContinuationPlanner
    from football_research.acquisition.joint_returns import JointReturnPredictor
    from football_research.acquisition.round9_residual import ResidualEVSICalibrator

    path = Path(config_path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if (config.get("partial_predictor_kind") != "learned-joint-returns-v1"
            or config.get("value_estimator_kind") != ESTIMATOR_KIND
            or config.get("source_calibrator_kind") != "residual-evsi-calibrator-v1"):
        raise ValueError("Explicit predictor, wrapper and source calibrator kinds required")
    gate = config.get("gate")
    if (not isinstance(gate, dict) or set(gate) != {"kind", "risk_level", "mode"}
            or gate.get("kind") != GATE_KIND or gate.get("risk_level") != "mean" or gate.get("mode") not in MODES):
        raise ValueError("An explicit mean-only continuation gate and mode are required")
    paths = {key: (path.parent / config[key]).resolve() for key in ("partial_predictor", "source_calibrator", "value_estimator")}
    hashes = {key: sha256(item.read_bytes()).hexdigest() for key, item in paths.items()}
    model = JointReturnPredictor.load(paths["partial_predictor"])
    calibrator = ResidualEVSICalibrator.load(paths["source_calibrator"], model=model, model_sha256=hashes["partial_predictor"])
    estimator = CalibratedContinuationPlanner.load(paths["value_estimator"], model=model, model_sha256=hashes["partial_predictor"],
                                                 calibrator=calibrator, calibrator_sha256=hashes["source_calibrator"])
    if estimator.mode != gate["mode"]:
        raise ValueError("Continuation gate mode differs from the bound planner")
    service = CalibratedContinuationServices(model, estimator, lambda_cost=config["lambda_cost"])
    service.artifact_hashes = {"configuration": sha256(path.read_bytes()).hexdigest(), "predictor": hashes["partial_predictor"],
                               "source_calibrator": hashes["source_calibrator"], "estimator": hashes["value_estimator"]}
    return service


def make_calibrated_continuation_backend(*, bundle_path, model_artifact):
    from football_platform.information import ExperimentConfig, FrozenEvidenceEnvironment

    path = Path(model_artifact).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    experiment = ExperimentConfig.model_validate(config["experiment"])
    services = load_calibrated_continuation_services(path)
    environment = FrozenEvidenceEnvironment.from_export(bundle_path, experiment)
    return FrozenResearchBackend(environment, services, configuration={
        "cost_table_version": experiment.version, "experiment": experiment.model_dump(mode="json"),
        "recommended_budget": {"max_simulated_cost": experiment.budget.get("money", .05),
                               "max_actions": int(experiment.budget.get("requests", 2))},
        "research_release": config.get("research_release"), "group_mapping": PLATFORM_GROUPS,
        "configuration_sha256": sha256(path.read_bytes()).hexdigest(),
        "bundle_complete_sha256": sha256((Path(bundle_path) / "COMPLETE.json").read_bytes()).hexdigest()},
        fixtures=config.get("fixtures", []))


class CalibratedContinuationAcquisitionWorkflow(CalibratedJointAcquisitionWorkflow):
    version = "visible-calibrated-continuation-acquisition-executor-v1"

    def start(self, match_id, cutoff, strategy="all_candidates", budget=None, seed=0, risk_mode="mean"):
        gate = self.backend.configuration().get("continuation_gate", {})
        if (gate.get("kind") != GATE_KIND or gate.get("horizon") != 2 or gate.get("risk_level") != "mean"
                or gate.get("mode") not in MODES or risk_mode != "mean"):
            raise ValueError("A mean-only calibrated continuation gate is required")
        return JointAcquisitionWorkflow.start(self, match_id, cutoff, strategy=strategy, budget=budget, seed=seed, risk_mode="mean")


def build_calibrated_continuation_cards(run):
    """Planned expectations and model branches stay separate from actual receipts."""
    cards = build_query_decision_cards(run)
    config = run["configuration"]["backend"]
    mode = config.get("continuation_gate", {}).get("mode")
    if config.get("value_estimator_kind") != ESTIMATOR_KIND or mode not in MODES:
        raise ValueError("Continuation card requires its actual estimator kind and mode")
    cards.update(value_estimator_kind=ESTIMATOR_KIND, continuation_mode=mode,
                 gain_semantics=GAIN_SEMANTICS, cost_semantics=COST_SEMANTICS,
                 reference_source_calibrator=deepcopy(config["reference_source_calibrator"]))
    events = {e["seq"]: e for e in run["events"]}
    for card in cards["cards"]:
        card.update(value_estimator_kind=ESTIMATOR_KIND, continuation_mode=mode)
        native_by_action = {a["action_id"]: a.get("value", {}).get("research_estimate", {})
                            for a in events[card["audit"]["seq"]]["data"].get("candidates", [])}
        for candidate in card["candidates"]:
            native = native_by_action[candidate["action_id"]]
            if any(candidate[k] is not None for k in ("model_evsi", "lower_gain", "standard_error")):
                raise ValueError("Continuation card must preserve null EVSI, lower bound and standard error")
            branches = deepcopy(native.get("branches", []))
            immediate_branches = deepcopy(native.get("immediate_estimate", {}).get("branches", []))
            for group in (branches, immediate_branches):
                if group and (any(b.get("hypothetical") is not True or type(b.get("probability")) not in (int, float)
                                or not math.isfinite(b["probability"]) or not 0 < b["probability"] <= 1 for b in group)
                             or abs(math.fsum(b["probability"] for b in group) - 1.) > 1e-10):
                    raise ValueError("Continuation branch audit needs normalized explicit model-only categories")
            candidate.update(estimator_kind=ESTIMATOR_KIND, continuation_mode=mode,
                gain_semantics=GAIN_SEMANTICS, cost_semantics=COST_SEMANTICS,
                plan_value={key:deepcopy(native.get(key)) for key in (
                    "immediate_estimate", "immediate_expected_gain", "immediate_cost_penalty", "immediate_net_gain",
                    "continuation_bonus", "adaptive_bonus", "commit_bonus", "surrogate_option_gap",
                    "expected_continuation_gain", "expected_continuation_cost",
                    "expected_continuation_cost_penalty", "expected_plan_cost_vector", "expected_gain", "cost_penalty",
                    "mean_net_gain", "selection_score", "continuation_reason")},
                model_only_value_audit={"scope": MODEL_AUDIT_SCOPE, "counts_as_acquisition": False,
                    "branches": branches, "immediate_valuation_branches": immediate_branches,
                    "source_calibrator_version": config["reference_source_calibrator"]["version"],
                    "source_calibrator_sha256": config["reference_source_calibrator"]["sha256"]})
    return cards
