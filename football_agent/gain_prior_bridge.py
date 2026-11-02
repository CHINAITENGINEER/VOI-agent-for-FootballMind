"""Explicit mean-only average-gain controls over the unchanged joint predictor."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

from .calibrated_joint_bridge import CalibratedJointServices
from .calibrated_joint_workflow import CalibratedJointAcquisitionWorkflow
from .joint_workflow import JointAcquisitionWorkflow
from .query_decision_card import build_query_decision_cards
from .research_bridge import FrozenResearchBackend, PLATFORM_GROUPS, ResearchServices


ESTIMATOR_KIND = "historical-gain-prior-v1"
GATE_KIND = "historical-gain-prior-one-step-v1"
MODES = ("action_only", "acquired_set")


class GainPriorServices(CalibratedJointServices):
    """Reuse numeric conversion and runtime guards, with a distinct identity."""

    def __init__(self, model, estimator, *, lambda_cost, erasure_mode):
        if erasure_mode not in MODES or estimator.mode != erasure_mode:
            raise ValueError("Explicit erasure mode must match the fitted average-gain estimator")
        super().__init__(model, estimator, lambda_cost=lambda_cost, risk_level="mean")
        self.erasure_mode = erasure_mode

    def configuration(self):
        manifest = self.estimator.manifest
        return {**ResearchServices.configuration(self), "partial_predictor_kind": "learned-joint-returns-v1",
                "value_estimator_kind": ESTIMATOR_KIND,
                "gain_prior_gate": {"kind": GATE_KIND, "risk_level": "mean", "horizon": 1, "erasure_mode": self.erasure_mode,
                    "model_evsi": None, "gain_semantics": "historical_mean_fixed_predictor_one_step_update_gain",
                    "value_inputs": ["action_id"] + (["acquired_action_ids"] if self.erasure_mode == "acquired_set" else []),
                    "erased_value_inputs": ["current_probabilities", "model_evsi", "team_identity", "fixture_date", "return_contents"],
                    "retained_guards": ["identity", "as_of", "information_mode", "availability", "history", "money_and_request_budget"],
                    "prediction_semantics": "unchanged_joint_predictor_conditions_on_actual_acquired_returns",
                    "uncertainty_kind": "descriptive_week_cluster_se_for_audit_only_no_lcb_policy",
                    "selection_score_decimal_places": 12},
                "reference_joint": {key: manifest[key] for key in
                                    ("joint_model_version", "joint_artifact_sha256", "trained_as_of", "diagnostics_as_of")}}

    def estimate(self, visible, actions):
        values = super().estimate(visible, actions)
        for value in values:
            native = value["research_estimate"]
            if "expected_gain" in native and (native.get("model_evsi") is not None or native.get("lower_gain") is not None
                    or native.get("estimator_kind") != ESTIMATOR_KIND or native.get("erasure_mode") != self.erasure_mode):
                raise ValueError("Average-gain candidate contains incompatible EVSI, lower bound, or estimator identity")
            value.update(estimator_kind=ESTIMATOR_KIND, erasure_mode=self.erasure_mode, model_evsi=None,
                         uncertainty_kind="descriptive_week_cluster_se_for_audit_only_no_lcb_policy")
        return values


def load_gain_prior_services(config_path):
    from football_research.acquisition.gain_prior import GainPriorEstimator
    from football_research.acquisition.joint_returns import JointReturnPredictor

    path = Path(config_path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("partial_predictor_kind") != "learned-joint-returns-v1" or config.get("value_estimator_kind") != ESTIMATOR_KIND:
        raise ValueError("Explicit joint predictor and historical-gain-prior estimator kinds are required")
    gate = config.get("gate")
    if not isinstance(gate, dict) or set(gate) != {"kind", "risk_level", "erasure_mode"} or gate.get("kind") != GATE_KIND:
        raise ValueError("Explicit average-gain one-step gate configuration is required")
    if gate["risk_level"] != "mean" or gate["erasure_mode"] not in MODES:
        raise ValueError("Average-gain controls support only mean and declared erasure modes")
    model_path = (path.parent / config["partial_predictor"]).resolve()
    value_path = (path.parent / config["value_estimator"]).resolve()
    model_hash = sha256(model_path.read_bytes()).hexdigest()
    model = JointReturnPredictor.load(model_path)
    estimator = GainPriorEstimator.load(value_path, model=model, model_sha256=model_hash)
    service = GainPriorServices(model, estimator, lambda_cost=config["lambda_cost"], erasure_mode=gate["erasure_mode"])
    service.artifact_hashes = {"configuration": sha256(path.read_bytes()).hexdigest(), "predictor": model_hash,
                               "estimator": sha256(value_path.read_bytes()).hexdigest()}
    return service


def make_gain_prior_backend(*, bundle_path, model_artifact):
    from football_platform.information import ExperimentConfig, FrozenEvidenceEnvironment

    path = Path(model_artifact).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    experiment = ExperimentConfig.model_validate(config["experiment"])
    services = load_gain_prior_services(path)
    environment = FrozenEvidenceEnvironment.from_export(bundle_path, experiment)
    return FrozenResearchBackend(environment, services, configuration={
        "cost_table_version": experiment.version, "experiment": experiment.model_dump(mode="json"),
        "recommended_budget": {"max_simulated_cost": experiment.budget.get("money", .05),
                               "max_actions": int(experiment.budget.get("requests", 2))},
        "research_release": config.get("research_release"), "group_mapping": PLATFORM_GROUPS,
        "configuration_sha256": sha256(path.read_bytes()).hexdigest(),
        "bundle_complete_sha256": sha256((Path(bundle_path) / "COMPLETE.json").read_bytes()).hexdigest()},
        fixtures=config.get("fixtures", []))


class GainPriorAcquisitionWorkflow(CalibratedJointAcquisitionWorkflow):
    version = "visible-average-gain-acquisition-executor-v1"

    def start(self, match_id, cutoff, strategy="all_candidates", budget=None, seed=0, risk_mode="mean"):
        gate = self.backend.configuration().get("gain_prior_gate", {})
        if (gate.get("kind") != GATE_KIND or gate.get("horizon") != 1 or gate.get("risk_level") != "mean"
                or gate.get("erasure_mode") not in MODES or risk_mode != "mean"):
            raise ValueError("A mean-only historical-gain-prior gate is required")
        return JointAcquisitionWorkflow.start(self, match_id, cutoff, strategy=strategy, budget=budget, seed=seed, risk_mode="mean")


def build_gain_prior_cards(run):
    cards = build_query_decision_cards(run)
    config = run["configuration"]["backend"]
    gate = config.get("gain_prior_gate", {})
    if config.get("value_estimator_kind") != ESTIMATOR_KIND or gate.get("erasure_mode") not in MODES:
        raise ValueError("Average-gain card requires its actual estimator kind and erasure mode")
    cards["value_estimator_kind"] = ESTIMATOR_KIND
    cards["state_erasure"] = deepcopy(gate)
    for card in cards["cards"]:
        card["value_estimator_kind"] = ESTIMATOR_KIND
        card["erasure_mode"] = gate["erasure_mode"]
        for candidate in card["candidates"]:
            if candidate["model_evsi"] is not None or candidate["lower_gain"] is not None:
                raise ValueError("Average-gain card must preserve null model EVSI and lower bound")
            candidate["estimator_kind"] = ESTIMATOR_KIND
            candidate["erasure_mode"] = gate["erasure_mode"]
    return cards
