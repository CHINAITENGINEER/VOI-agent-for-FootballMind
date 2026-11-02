"""Inquiry selection with a shared, executable policy across declared models.

Numerical acquisitions commit to one query followed by its frozen terminal
decision. This does not turn receding-horizon replanning into a minimax proof.
"""
from __future__ import annotations

from copy import deepcopy

from .ambiguity_value import FiniteAmbiguityDecisionModel
from .audit import digest
from .decision_value import VALUE_TOLERANCE
from .inquiry_value import DecisionRiskSelector


VERSION = "model-set-inquiry-risk-v1"
EXECUTION_MODE = "one_query_then_frozen_terminal_decision"


class AmbiguityRiskSelector:
    def __init__(self, model_set, bindings, *, task_id, loss_budget):
        if not isinstance(model_set, FiniteAmbiguityDecisionModel):
            raise ValueError("An explicit finite decision-model set is required")
        self.model_set = model_set
        self.binding = DecisionRiskSelector(model_set.reference_model, bindings, task_id=task_id, loss_budget=loss_budget)

    def configuration(self):
        registered = self.binding.configuration()
        return {"version": VERSION, "model_set": self.model_set.manifest(), "bindings": registered["bindings"],
                "task_id": registered["task_id"], "loss_budget": registered["loss_budget"],
                "execution_mode": EXECUTION_MODE, "stop_rule": "no_model_can_gain_against_shared_stop"}

    @classmethod
    def from_configuration(cls, config):
        if (config.get("version") != VERSION or config.get("execution_mode") != EXECUTION_MODE
                or config.get("stop_rule") != "no_model_can_gain_against_shared_stop"):
            raise ValueError("Unsupported model-set inquiry execution/stop contract")
        return cls(FiniteAmbiguityDecisionModel.from_manifest(config["model_set"]), config["bindings"],
                   task_id=config["task_id"], loss_budget=config["loss_budget"])

    def public_summary(self, run):
        result = {"version": VERSION, "model_hash": digest(self.model_set.manifest()),
            "evidence_class": "DECLARED_MODEL_SET_NOT_STATISTICAL_CONFIDENCE", "model_is_calibrated": False,
            "statistical_confidence_level": None, "cost_unit": self.model_set.cost_unit, "horizon": 1,
            "execution_mode": EXECUTION_MODE, "global_open_world_stop_proven": False,
            "registered_queries": [{"action_id": aid, **deepcopy(self.binding.bindings[aid])} for aid in sorted(self.binding.bindings)]}
        try:
            history, mapped, spent = self.binding.extract_observations(run)
            plan = self.model_set.evaluate(history, remaining_budget=max(0., self.binding.loss_budget-spent))
            result.update(status=plan["status"], observations=mapped, spent_loss_penalty=spent,
                remaining_loss_budget=max(0., self.binding.loss_budget-spent), plan=plan,
                terminal_decision=plan["terminal_decision_if_stopping"])
        except ValueError as exc:
            result.update(status="UNSUPPORTED_MODEL_STATE", reason=str(exc), plan=None, terminal_decision=None)
        return result

    def _frozen_terminal(self, run, summary):
        previous = run["decisions"][-1] if run["decisions"] else None
        if not previous or not previous.get("terminal_policy_commitment") or not run["receipts"]:
            return None
        commitment = previous["terminal_policy_commitment"]
        if commitment["model_hash"] != summary["model_hash"]:
            raise ValueError("Frozen response policy belongs to a different model set")
        receipt = run["receipts"][-1]
        observed = next((row for row in summary["observations"] if row["document_id"] == receipt["document_id"]), None)
        before_ids = {row["document_id"] for row in previous["risk_assessment"]["state"]["observations"]}
        if not observed or observed["document_id"] in before_ids or observed["action_id"] != commitment["action_id"]:
            return None
        # The caller checks model-set support first. All-model-impossible
        # branches must never produce their arbitrary enumeration placeholder.
        decision = commitment["response_policy"][observed["outcome"]]
        risks = summary["plan"]["terminal_decision_if_stopping"]["per_decision_expected_risk"][decision]
        return {"decision": decision, "worst_expected_loss": max(risks.values()),
            "per_model_expected_risk": risks, "request_id": receipt["request_id"],
            "observed_return": observed["outcome"], "committed_action_id": commitment["action_id"],
            "response_policy_hash": digest(commitment["response_policy"]),
            "policy_followed": True, "selection_time_value_scope": EXECUTION_MODE,
            "truth_verified": False}

    def settle_after_return(self, run):
        """Execute the pre-query commitment without another provider turn.

        A supported return selects the frozen terminal action, rather than a
        newly optimized action. An out-of-model return withholds that action.
        No settlement is produced until a new physical return is available.
        """
        previous = run["decisions"][-1] if run["decisions"] else None
        if (not previous or previous["kind"] != "query"
                or not previous.get("terminal_policy_commitment") or not run["receipts"]):
            return None
        before = {row["document_id"] for row in previous["risk_assessment"]["state"]["observations"]}
        if run["receipts"][-1]["document_id"] in before:
            return None
        summary = self.public_summary(run)
        assessment = {"version": VERSION, "state": summary, "candidate_values": [],
            "global_open_world_stop_proven": False, "statistical_confidence_level": None,
            "stop_scope": "completion_of_precommitted_one_query_policy",
            "stop_certificate": "FROZEN_POLICY_WITHHELD_MODEL_SUPPORT_UNKNOWN"}
        result = {"kind": "defer", "candidate_id": None, "revisit_at": None,
            "reason": "实际返回超出已声明模型或返回映射的支持范围；保留未知，不执行占位终止决策。",
            "selection_basis": "model_set_frozen_terminal_policy", "value_status": "UNSUPPORTED_OPEN_TEXT",
            "expected_gain": None, "cost_in_loss_units": None, "robust_risk_reduction": None,
            "llm_proposal": None, "risk_assessment": assessment, "terminal_policy_commitment": None,
            "terminal_decision": None}
        if summary["status"] != "SUPPORTED_DECLARED_MODEL_SET":
            return result
        frozen = self._frozen_terminal(run, summary)
        if frozen is None:
            assessment["stop_certificate"] = "FROZEN_POLICY_WITHHELD_RETURN_BINDING_UNKNOWN"
            return result
        assessment["stop_certificate"] = "FROZEN_ONE_QUERY_TERMINAL_POLICY_EXECUTED"
        return {**result, "kind": "stop", "value_status": "DECLARED_MODEL_SET_ONLY",
            "terminal_decision": frozen,
            "reason": "实际返回已到达，直接执行查询前冻结的共同终止策略；本次结束不表示其余开放信息没有价值。"}

    def select(self, run, response):
        raw = response["decision"]
        summary = self.public_summary(run)
        assessment = {"version": VERSION, "state": summary, "candidate_values": [],
            "global_open_world_stop_proven": False, "statistical_confidence_level": None,
            "stop_scope": "registered_one_query_shared_deterministic_policies"}
        result = {**deepcopy(raw), "selection_basis": "llm_bounded_exploration", "value_status": "UNSUPPORTED_OPEN_TEXT",
            "expected_gain": None, "cost_in_loss_units": None, "robust_risk_reduction": None,
            "llm_proposal": deepcopy(raw), "risk_assessment": assessment, "terminal_policy_commitment": None,
            "terminal_decision": None}
        if summary["status"] != "SUPPORTED_DECLARED_MODEL_SET":
            assessment["stop_certificate"] = "WITHHELD_MODEL_SUPPORT_UNKNOWN"
            return result
        frozen = self._frozen_terminal(run, summary)
        if frozen:
            assessment["stop_certificate"] = "FROZEN_ONE_QUERY_TERMINAL_POLICY_EXECUTED"
            return {**result, "kind": "stop", "candidate_id": None, "revisit_at": None,
                "selection_basis": "model_set_frozen_terminal_policy", "value_status": "DECLARED_MODEL_SET_ONLY",
                "terminal_decision": frozen,
                "reason": "按查询前冻结的共同返回策略给出终止决策；本次结束是已声明的一步策略完成，不表示其余开放信息没有价值。"}
        if raw["kind"] == "reflect":
            return result
        if run["budget"]["http_attempts"] >= run["budget"]["max_requests"]:
            assessment["stop_certificate"] = "REQUEST_BUDGET_EXHAUSTED_NOT_VALUE_EXHAUSTED"
            return {**result, "kind": "defer", "candidate_id": None, "revisit_at": None,
                    "reason": "请求预算已用完；保留不确定性及价值未知的范围。"}
        plan = summary["plan"]
        values = {c["action_id"]: c for c in plan["candidates"]}
        definitions = {h["hypothesis_id"]: h for h in run["hypotheses"]}
        supported, unknown = [], []
        for candidate in response["candidates"]:
            aid = self.binding.match_candidate(candidate, definitions)
            if aid is None:
                item = {"candidate_id": candidate["candidate_id"], "status": "UNSUPPORTED", "robust_risk_reduction": None}
                unknown.append(candidate["candidate_id"])
            else:
                value = values[aid]
                item = {"candidate_id": candidate["candidate_id"], "action_id": aid,
                        "status": "MODELED" if value["feasible"] else "UNAVAILABLE", "model_value": value}
                if value["feasible"]:
                    supported.append((candidate, value))
            assessment["candidate_values"].append(item)
        assessment["unsupported_candidate_ids"] = unknown
        positive = [(c, v) for c, v in supported if v["robust_net_value"] > VALUE_TOLERANCE]
        if positive:
            candidate, value = min(positive, key=lambda pair: (pair[1]["worst_expected_risk"], pair[0]["candidate_id"]))
            commitment = {"model_hash": summary["model_hash"], "action_id": value["action_id"],
                "response_policy": deepcopy(value["response_policy"]), "execution_mode": EXECUTION_MODE,
                "per_model_expected_risk": deepcopy(value["per_model_expected_risk"]),
                "worst_expected_risk": value["worst_expected_risk"]}
            return {**result, "kind": "query", "candidate_id": candidate["candidate_id"], "revisit_at": None,
                "selection_basis": "model_set_shared_policy_minimax", "value_status": "DECLARED_MODEL_SET_ONLY",
                "robust_risk_reduction": value["robust_net_value"], "cost_in_loss_units": value["cost"],
                "terminal_policy_commitment": commitment,
                "reason": "在已提出且已建模的候选中，选择共同返回策略下最坏期望损失最小的查询，并冻结取得返回后的终止决策。"}
        profitable = [v["action_id"] for v in values.values() if v["feasible"] and v["robust_net_value"] > VALUE_TOLERANCE]
        if profitable:
            assessment["stop_certificate"] = "WITHHELD_VALUABLE_REGISTERED_ACTION_NOT_PROPOSED"
            assessment["unproposed_positive_actions"] = profitable
        elif unknown or not plan["dominance_stop_certificate"]:
            assessment["stop_certificate"] = ("WITHHELD_UNMODELED_CANDIDATES" if unknown else
                                               "WITHHELD_POSSIBLE_GAIN_OR_FEASIBILITY_LIMIT")
            plausible = {c["candidate_id"] for c, v in supported if v["dominance_upper_gain"] > VALUE_TOLERANCE} | set(unknown)
            if raw["kind"] == "query" and raw["candidate_id"] in plausible:
                return result
        else:
            assessment["stop_certificate"] = plan["dominance_stop_certificate"]
            return {**result, "kind": "stop", "candidate_id": None, "revisit_at": None,
                "selection_basis": "model_set_shared_policy_minimax", "value_status": "DECLARED_MODEL_SET_ONLY",
                "terminal_decision": {**deepcopy(plan["terminal_decision_if_stopping"]), "truth_verified": False},
                "reason": "相对于同一终止决策，声明模型集合内的可行一步查询都没有正收益上界；停止范围不覆盖未建模信息或多步组合。"}
        return {**result, "kind": "defer", "candidate_id": None, "revisit_at": None,
            "reason": "当前缺少可执行的稳健改善提案，且未满足有范围的一步收益停止条件；保留不确定性。"}
