"""Explicit bridge for a learned joint model and finite mean-risk planning."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from time import perf_counter

from football_research.acquisition.joint_planner import JointRiskPlanner, VERSION
from .research_bridge import ResearchServices, public_state, research_state


def physical_budget(visible):
    """Executor projection-call limits, not a budget in lambda-scaled units."""
    budget = public_state(visible).get("budget", {})
    if all(k in budget for k in ("max_simulated_cost", "simulated_cost", "max_actions", "actions")):
        return {"limits": {"money": budget["max_simulated_cost"], "requests": budget["max_actions"]},
                "spent": {"money": budget["simulated_cost"], "requests": budget["actions"]}}
    if "limits" in budget and "spent" in budget:
        return {"limits": deepcopy(budget["limits"]), "spent": deepcopy(budget["spent"])}
    raise ValueError("Joint planning requires an explicit physical money/request budget")


class JointPlanningServices(ResearchServices):
    def __init__(self, model, *, horizon, lambda_cost):
        planner = JointRiskPlanner(model, horizon=horizon, lambda_cost=lambda_cost)
        super().__init__(model, planner, lambda_cost)

    def configuration(self):
        return {**super().configuration(), "partial_predictor_kind": "learned-joint-returns-v1",
                "value_estimator_kind": "joint-risk-planner-v1",
                "planner": {"kind": VERSION, "horizon": self.estimator.horizon, "supported_risk_modes": ["mean"],
                            "uncertainty": "joint_parameter_uncertainty_not_estimated",
                            "candidate_gain_semantics": "expected_terminal_loss_reduction_over_the_complete_conditional_plan",
                            "candidate_cost_semantics": "expected_cumulative_loss_converted_plan_cost",
                            "physical_budget_semantics": "simulation_credit_and_projection_calls_unscaled_by_lambda"}}

    def estimate(self, visible, actions):
        started = perf_counter()
        public = public_state(visible)
        candidates = []
        for action in actions:
            candidates.append({"action_id": action["action_id"], "group_ids": deepcopy(action["group_ids"]),
                               "available": action["status"] == "AVAILABLE" and action.get("available", True),
                               "cost": deepcopy(action["cost_vector"]), "cost_status": action["cost"]["status"],
                               **{key: deepcopy(action[key]) for key in ("dependency_key", "cache_policy", "cost_scope", "evidence_id") if key in action}})
        plan = self.estimator.plan(research_state(public), candidates, physical_budget(public))
        elapsed = (perf_counter() - started) * 1000
        values = []
        for candidate in plan["candidates"]:
            feasible = candidate["feasible"]
            values.append({"action_id": candidate["action_id"], "status": "OK" if feasible else "UNAVAILABLE",
                           "reason": candidate["reason"], "expected_loss_improvement": candidate.get("expected_loss_improvement"),
                           "cost_in_loss_units": candidate.get("expected_plan_penalty"), "net_gain": candidate.get("selection_score"),
                           "lower_gain": None, "uncertainty": None, "support_status": "fitted_joint_model" if feasible else "unavailable",
                           "uncertainty_kind": "joint_parameter_uncertainty_not_estimated",
                           "conservative_status": "UNSUPPORTED_JOINT_PARAMETER_UNCERTAINTY",
                           "estimator_version": self.estimator.version,
                           "research_estimate": {**deepcopy(candidate), "planner_version": self.estimator.version,
                                                 "model_version": self.predictor.model_version, "horizon": self.estimator.horizon,
                                                 "risk_mode": "mean_only", "physical_budget": plan["physical_budget"],
                                                 "lambda_cost": plan["lambda_cost"], "planned_action": plan["action_id"],
                                                 "selection_score_precision_decimal_places": 12,
                                                 "plan_reason": plan["reason"]},
                           "compute_cost": {"money": 0, "requests": 0, "tokens": 0, "latency_ms": elapsed if not values else 0}})
        return values


def load_joint_services(config_path, config):
    from football_research.acquisition.joint_returns import JointReturnPredictor

    if config.get("partial_predictor_kind") != "learned-joint-returns-v1":
        raise ValueError("Joint planning requires an explicit learned-joint-returns-v1 predictor")
    planner = config.get("planner")
    if not isinstance(planner, dict) or planner.get("kind") != VERSION or set(planner) != {"kind", "horizon"}:
        raise ValueError("Explicit finite-joint-bellman-v1 planner kind and horizon are required")
    path = Path(config_path).resolve()
    model_path = (path.parent / config["partial_predictor"]).resolve()
    value_path = (path.parent / config["value_estimator"]).resolve()
    if sha256(model_path.read_bytes()).digest() != sha256(value_path.read_bytes()).digest():
        raise ValueError("Prediction and acquisition must use the same frozen joint model artifact")
    model = JointReturnPredictor.load(model_path)
    service = JointPlanningServices(model, horizon=planner["horizon"], lambda_cost=config["lambda_cost"])
    service.artifact_hashes = {"configuration": sha256(path.read_bytes()).hexdigest(),
                               "predictor": sha256(model_path.read_bytes()).hexdigest(),
                               "estimator": sha256(value_path.read_bytes()).hexdigest()}
    return service
