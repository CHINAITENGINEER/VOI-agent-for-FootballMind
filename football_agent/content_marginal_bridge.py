"""Erase acquired contents from query valuation, preserving actual prediction."""
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


ESTIMATOR_KIND = "content-marginal-gain-v1"
GATE_KIND = "content-marginal-one-step-v1"
ERASURE_MODE = "marginalize_acquired_contents_given_free_prior_and_acquired_set"
ERASURE_SCOPE = "content_erased_from_query_valuation_only_actual_prediction_uses_real_returns"
MODEL_AUDIT_SCOPE = "model_only_mathematical_categories_not_tool_receipts_or_observed_evidence"


class ContentMarginalServices(CalibratedJointServices):
    def __init__(self, model, estimator, *, lambda_cost):
        if estimator.manifest.get("erasure_mode") != ERASURE_MODE:
            raise ValueError("Content marginal estimator with explicit erasure scope required")
        super().__init__(model, estimator, lambda_cost=lambda_cost, risk_level="mean")

    def configuration(self):
        manifest = self.estimator.manifest
        return {**ResearchServices.configuration(self), "partial_predictor_kind": "learned-joint-returns-v1",
                "value_estimator_kind": ESTIMATOR_KIND,
                "content_marginal_gate": {"kind": GATE_KIND, "risk_level": "mean", "horizon": 1,
                    "erasure_mode": ERASURE_MODE, "erasure_scope": ERASURE_SCOPE,
                    "weights": "frozen_joint_given_free_prior_not_actual_posterior",
                    "aggregation": "mean_of_supported_branch_calibrated_gains_no_branch_dropping",
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
                    raise ValueError("Marginal wrapper identity or null uncertainty contract differs")
            value.update(estimator_kind=ESTIMATOR_KIND, erasure_mode=ERASURE_MODE, erasure_scope=ERASURE_SCOPE,
                         model_evsi=None, uncertainty_kind="aggregate_uncertainty_not_estimated")
        return values


def load_content_marginal_services(config_path):
    from football_research.acquisition.content_marginal import ContentMarginalGainEstimator
    from football_research.acquisition.joint_returns import JointReturnPredictor
    from football_research.acquisition.round9_residual import ResidualEVSICalibrator

    path = Path(config_path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if (config.get("partial_predictor_kind") != "learned-joint-returns-v1"
            or config.get("value_estimator_kind") != ESTIMATOR_KIND
            or config.get("source_calibrator_kind") != "residual-evsi-calibrator-v1"):
        raise ValueError("Explicit predictor, wrapper and source calibrator kinds required")
    gate = config.get("gate")
    if not isinstance(gate, dict) or set(gate) != {"kind", "risk_level"} or gate != {"kind": GATE_KIND, "risk_level": "mean"}:
        raise ValueError("An explicit mean-only content marginal one-step gate is required")
    paths = {key: (path.parent / config[key]).resolve() for key in ("partial_predictor", "source_calibrator", "value_estimator")}
    hashes = {key: sha256(item.read_bytes()).hexdigest() for key, item in paths.items()}
    model = JointReturnPredictor.load(paths["partial_predictor"])
    calibrator = ResidualEVSICalibrator.load(paths["source_calibrator"], model=model, model_sha256=hashes["partial_predictor"])
    estimator = ContentMarginalGainEstimator.load(paths["value_estimator"], model=model, model_sha256=hashes["partial_predictor"],
                                                 calibrator=calibrator, calibrator_sha256=hashes["source_calibrator"])
    service = ContentMarginalServices(model, estimator, lambda_cost=config["lambda_cost"])
    service.artifact_hashes = {"configuration": sha256(path.read_bytes()).hexdigest(), "predictor": hashes["partial_predictor"],
                               "source_calibrator": hashes["source_calibrator"], "estimator": hashes["value_estimator"]}
    return service


def make_content_marginal_backend(*, bundle_path, model_artifact):
    from football_platform.information import ExperimentConfig, FrozenEvidenceEnvironment

    path = Path(model_artifact).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    experiment = ExperimentConfig.model_validate(config["experiment"])
    services = load_content_marginal_services(path)
    environment = FrozenEvidenceEnvironment.from_export(bundle_path, experiment)
    return FrozenResearchBackend(environment, services, configuration={
        "cost_table_version": experiment.version, "experiment": experiment.model_dump(mode="json"),
        "recommended_budget": {"max_simulated_cost": experiment.budget.get("money", .05),
                               "max_actions": int(experiment.budget.get("requests", 2))},
        "research_release": config.get("research_release"), "group_mapping": PLATFORM_GROUPS,
        "configuration_sha256": sha256(path.read_bytes()).hexdigest(),
        "bundle_complete_sha256": sha256((Path(bundle_path) / "COMPLETE.json").read_bytes()).hexdigest()},
        fixtures=config.get("fixtures", []))


class ContentMarginalAcquisitionWorkflow(CalibratedJointAcquisitionWorkflow):
    version = "visible-content-marginal-acquisition-executor-v1"

    def start(self, match_id, cutoff, strategy="all_candidates", budget=None, seed=0, risk_mode="mean"):
        gate = self.backend.configuration().get("content_marginal_gate", {})
        if (gate.get("kind") != GATE_KIND or gate.get("horizon") != 1 or gate.get("risk_level") != "mean"
                or gate.get("erasure_mode") != ERASURE_MODE or risk_mode != "mean"):
            raise ValueError("A mean-only content marginal gate is required")
        return JointAcquisitionWorkflow.start(self, match_id, cutoff, strategy=strategy, budget=budget, seed=seed, risk_mode="mean")


def build_content_marginal_cards(run):
    """Keep category integration audit separate from actual tool-return receipts."""
    cards = build_query_decision_cards(run)
    config = run["configuration"]["backend"]
    if config.get("value_estimator_kind") != ESTIMATOR_KIND:
        raise ValueError("Content marginal card requires its actual estimator kind")
    cards.update(value_estimator_kind=ESTIMATOR_KIND, erasure_scope=ERASURE_SCOPE,
                 reference_source_calibrator=deepcopy(config["reference_source_calibrator"]))
    events = {e["seq"]: e for e in run["events"]}
    for card in cards["cards"]:
        card.update(value_estimator_kind=ESTIMATOR_KIND, erasure_scope=ERASURE_SCOPE)
        native_by_action = {a["action_id"]: a.get("value", {}).get("research_estimate", {})
                            for a in events[card["audit"]["seq"]]["data"].get("candidates", [])}
        for candidate in card["candidates"]:
            native = native_by_action[candidate["action_id"]]
            if any(candidate[k] is not None for k in ("model_evsi", "lower_gain", "standard_error")):
                raise ValueError("Marginal card must preserve null EVSI, lower bound and standard error")
            branches = deepcopy(native.get("branches", []))
            if branches and (any(b.get("hypothetical") is not True or type(b.get("probability")) not in (int, float)
                                or not math.isfinite(b["probability"]) or not 0 < b["probability"] <= 1 for b in branches)
                             or abs(math.fsum(b["probability"] for b in branches) - 1.) > 1e-10):
                raise ValueError("Marginal branch audit needs normalized explicit model-only categories")
            candidate.update(estimator_kind=ESTIMATOR_KIND, erasure_scope=ERASURE_SCOPE,
                model_only_value_audit={"scope": MODEL_AUDIT_SCOPE, "counts_as_acquisition": False,
                    "marginal_model_evsi": native.get("marginal_model_evsi"), "branches": branches,
                    "source_calibrator_version": config["reference_source_calibrator"]["version"],
                    "source_calibrator_sha256": config["reference_source_calibrator"]["sha256"]})
    return cards
