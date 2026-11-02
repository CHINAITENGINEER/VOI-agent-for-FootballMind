"""Two-step planning with frozen empirical query-gain estimates.

This is an approximate calibrated reward surrogate. It is not a coherent Bayes
risk decomposition or a guarantee that predicted continuation benefits occur.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path

from .content_marginal import ContentMarginalGainEstimator
from .joint_planner import _budget, _cost, _number, _probabilities, UNITS
from .joint_returns import ACTION_IDS
from .value import _canonical

VERSION = "calibrated-continuation-planner-v1"
MODES = ("adaptive", "commit")


def _reason(action, *, used, used_evidence, limits, spent):
    cost = _cost(action)
    return ("action_unavailable" if not action.get("available",True) else
        "already_attempted" if action["action_id"] in used else
        "evidence_already_observed" if action.get("evidence_id") in used_evidence else
        "physical_budget_exhausted" if any(spent[u]+cost[u] > cap+1e-12 for u,cap in limits.items()) else None)


class CalibratedContinuationPlanner:
    def __init__(self, model, calibrator, *, mode, model_sha256, calibrator_sha256):
        if mode not in MODES:
            raise ValueError("Declared adaptive or commit planning mode required")
        self.marginal = ContentMarginalGainEstimator(model,calibrator,model_sha256=model_sha256,calibrator_sha256=calibrator_sha256)
        self.model, self.calibrator, self.mode = model, calibrator, mode
        self.model_sha256, self.calibrator_sha256 = model_sha256, calibrator_sha256
        self.manifest = deepcopy(self.marginal.manifest)
        self.manifest.update(estimator_kind=VERSION,mode=mode,horizon=2,
            erasure_mode="retain_contents_for_future_query_stop" if mode=="adaptive" else "content_marginal_future_query_stop",
            target="approximate_expected_sum_of_frozen_calibrated_update_gains_along_conditional_plan",
            unsupported_rule="unsupported immediate disables candidate; any unsupported future branch disables all continuation bonus",
            gain_semantics="approximate_planning_reward_surrogate_not_Bayes_optimal_VoI",
            uncertainty="no aggregate plan interval; source one-step diagnostics retained",
            post_return_rule="original_conditional_mean" if mode=="adaptive" else "round13_content_marginal_mean")
        self.version = VERSION+":"+sha256(_canonical(self._payload()).encode()).hexdigest()[:20]

    def _immediate(self, clean, acquired, action):
        if acquired and self.mode=="commit":
            return self.marginal._estimate(clean,acquired,action)
        enumeration = self.model.enumerate_action(clean,action)
        return self.calibrator.estimate_numbers(action["action_id"],acquired,enumeration["evsi"],risk_level="mean")

    def _candidate(self,clean,acquired,action,actions,*,used,used_evidence,limits,spent,lam):
        aid = action["action_id"]
        immediate = self._immediate(clean,acquired,action)
        gain = immediate["expected_gain"]
        cost = _cost(action)
        penalty = math.fsum(lam[u]*cost[u] for u in UNITS)
        immediate_net = gain-penalty if gain is not None else None
        future_cost = {u:0. for u in UNITS}
        future_gain = bonus = adaptive_bonus = commit_bonus = 0.
        branches, unsupported = [], []
        continuation_reason = "already_one_action_acquired" if acquired else "no_legal_affordable_second_action"
        second_action = None
        after_spent = {u:spent[u]+cost[u] for u in UNITS}
        after_evidence = used_evidence | ({action["evidence_id"]} if action.get("evidence_id") is not None else set())
        legal_second = [a for a in actions if _reason(a,used=used|{aid},used_evidence=after_evidence,limits=limits,spent=after_spent) is None]
        if gain is None:
            continuation_reason = "immediate_gain_unsupported"
        elif not acquired and legal_second:
            if len(legal_second) != 1:
                raise ValueError("Two-view planner expects at most one remaining action")
            second_action = legal_second[0]
            second_id, second_cost = second_action["action_id"], _cost(second_action)
            second_penalty = math.fsum(lam[u]*second_cost[u] for u in UNITS)
            enumeration = self.model.enumerate_action(clean,action)
            for branch in enumeration["branches"]:
                mass = float(branch["probability"])
                if not math.isfinite(mass) or mass < 0:
                    raise ValueError("Invalid continuation return mass")
                if mass == 0:
                    continue
                future = self.model.enumerate_action(branch["next_state"],second_action,hypothetical=True)
                estimate = self.calibrator.estimate_numbers(second_id,[aid],future["evsi"],risk_level="mean")
                branch_gain = estimate["expected_gain"]
                branch_net = branch_gain-second_penalty if branch_gain is not None else None
                if branch_gain is None:
                    unsupported.append(branch["observation"])
                branches.append({"observation":branch["observation"],"probability":mass,"model_evsi":estimate["model_evsi"],
                    "expected_gain":branch_gain,"net_gain":branch_net,"cost_penalty":second_penalty,"cost_vector":second_cost,
                    "support":estimate["support"],"estimate":estimate,"hypothetical":True,
                    "scope":"model_only_continuation_branch_not_receipt_or_realized_outcome",
                    "continuation_action_id":second_id,"physical_budget":{"limits":limits,"spent_after_first":after_spent},
                    "adaptive_query":False,"commit_query":False,"continuation_decision":"stop"})
            mass = math.fsum(b["probability"] for b in branches)
            if not branches or abs(mass-1.) > 1e-10:
                raise ValueError("Complete normalized positive-mass continuation branches required")
            if unsupported:
                continuation_reason = "positive_mass_future_branch_unsupported_disable_all_continuation"
            else:
                mean_gain = math.fsum(b["probability"]*b["expected_gain"] for b in branches)
                mean_net = mean_gain-second_penalty
                commit_query = round(mean_net,12) > 0
                for branch in branches:
                    branch["adaptive_query"] = round(branch["net_gain"],12) > 0
                    branch["commit_query"] = commit_query
                    branch["continuation_decision"] = "acquire" if branch[self.mode+"_query"] else "stop"
                adaptive_bonus = math.fsum(b["probability"]*b["net_gain"] for b in branches if b["adaptive_query"])
                commit_bonus = mean_net if commit_query else 0.
                chosen = [b for b in branches if b["continuation_decision"]=="acquire"]
                future_gain = math.fsum(b["probability"]*b["expected_gain"] for b in chosen)
                future_cost = {u:math.fsum(b["probability"]*second_cost[u] for b in chosen) for u in UNITS}
                bonus = adaptive_bonus if self.mode=="adaptive" else commit_bonus
                continuation_reason = "supported_branchwise_query_or_stop" if self.mode=="adaptive" else "supported_content_independent_continuation"
        future_penalty = math.fsum(lam[u]*future_cost[u] for u in UNITS)
        total_gain = gain+future_gain if gain is not None else None
        total_cost = {u:cost[u]+future_cost[u] for u in UNITS}
        total_penalty = penalty+future_penalty
        q = immediate_net+bonus if immediate_net is not None else None
        supported = gain is not None
        reason = "stop_no_support" if not supported else "positive_total_plan_gain" if round(q,12)>0 else "stop_nonpositive_total_plan_gain"
        return {"action_id":aid,"acquired_action_ids":list(acquired),"immediate_estimate":immediate,
            "immediate_expected_gain":gain,"immediate_cost_penalty":penalty,"immediate_net_gain":immediate_net,
            "continuation_bonus":bonus,"adaptive_bonus":adaptive_bonus,"commit_bonus":commit_bonus,
            "surrogate_option_gap":adaptive_bonus-commit_bonus,"expected_continuation_gain":future_gain,
            "expected_continuation_cost":future_cost,"expected_continuation_cost_penalty":future_penalty,
            "expected_plan_cost_vector":total_cost,"expected_gain":total_gain,"g_hat":total_gain,"decision_gain":total_gain,
            "cost_penalty":total_penalty,"mean_net_gain":q,"net_gain":q,"selection_score":round(q,12) if q is not None else None,
            "lower_gain":None,"upper_gain":None,"lower_net_gain":None,"model_evsi":None,"evsi_bin":None,
            "standard_error":None,"uncertainty":None,"uncertainty_kind":"aggregate_plan_uncertainty_not_estimated",
            "support":immediate["support"],"support_matches":immediate["support_matches"],"parent_support":immediate["parent_support"],
            "effective_support":immediate["effective_support"],"time_blocks":immediate["time_blocks"],
            "support_status":"supported_immediate" if supported else "unsupported","fallback_reason":None if supported else "immediate_unsupported",
            "support_semantics":"immediate source support; each continuation branch retains separate support",
            "feasible":supported,"reason":reason,"gate_reason":reason,"continuation_reason":continuation_reason,
            "unsupported_continuation_observations":unsupported,"branches":branches,"continuation_action_id":second_action["action_id"] if second_action else None,
            "risk_level":"mean","estimator_kind":VERSION,"estimator_version":self.version,"mode":self.mode,
            "source_calibrator_version":self.calibrator.version,"source_calibrator_sha256":self.calibrator_sha256,
            "joint_model_version":self.model.model_version,"gain_semantics":"expected_sum_of_calibrated_update_gains_under_approximate_conditional_plan",
            "cost_vector":cost,"planning_cost_scope":"expected_future_cost_is_not_actual_charge"}

    def plan(self,state,actions,budget,*,risk_level="mean",lambda_cost=None):
        if risk_level != "mean":
            raise ValueError("Calibrated continuation supports only mean planning")
        if not isinstance(actions,(list,tuple)):
            raise ValueError("Actions must be a sequence")
        clean,acquired,attempted = self.marginal._visible(state)
        limits,spent = _budget(budget)
        weights = lambda_cost or {}
        if set(weights)-set(UNITS):
            raise ValueError("Unknown cost-conversion unit")
        lam = {u:_number(weights.get(u,0.),"lambda "+u) for u in UNITS}
        seen = set()
        for action in actions:
            if not isinstance(action,dict) or action.get("action_id") not in ACTION_IDS or type(action.get("available",True)) is not bool:
                raise ValueError("Modeled action identity and Boolean availability required")
            if action["action_id"] in seen:
                raise ValueError("Duplicate candidate action")
            seen.add(action["action_id"])
            if action.get("evidence_id") is not None and (not isinstance(action["evidence_id"],str) or not action["evidence_id"]):
                raise ValueError("Evidence identity must be a nonempty inert string")
            _cost(action)
        used = set(acquired)|attempted
        used_evidence = {a["evidence_id"] for a in actions if a["action_id"] in used and a.get("evidence_id") is not None}
        output = []
        for action in sorted(actions,key=lambda a:a["action_id"]):
            reason = _reason(action,used=used,used_evidence=used_evidence,limits=limits,spent=spent)
            if reason:
                output.append({"action_id":action["action_id"],"feasible":False,"reason":reason})
            else:
                output.append(self._candidate(clean,acquired,action,actions,used=used,used_evidence=used_evidence,limits=limits,spent=spent,lam=lam))
        eligible = [r for r in output if r["feasible"] and r["selection_score"] > 0]
        selected = min(eligible,key=lambda r:(-r["selection_score"],r["action_id"])) if eligible else None
        return {"schema_version":VERSION,"estimator_kind":VERSION,"estimator_version":self.version,"mode":self.mode,
            "model_version":self.model.model_version,"source_calibrator_version":self.calibrator.version,
            "source_calibrator_sha256":self.calibrator_sha256,"decision":"acquire" if selected else "stop",
            "action_id":selected["action_id"] if selected else None,
            "reason":"positive_calibrated_plan_net_gain" if selected else "no_positive_supported_calibrated_plan_gain",
            "risk_level":"mean","horizon":2,"physical_budget":{"limits":limits,"spent":spent},"lambda_cost":lam,
            "selection_score_precision_decimal_places":12,"candidates":output,
            "planning_semantics":"approximate_frozen_empirical_gain_surrogate_not_Bayes_optimal_information_value"}

    def estimate_value(self,state,candidates,*,budget,prediction=None,risk_level="mean",lambda_cost=None):
        if prediction is not None:
            supplied = _probabilities(prediction["probabilities"],"Supplied prediction")
            actual = self.model.predict_partial(state)["probabilities"]
            if max(abs(a-b) for a,b in zip(supplied,actual)) > 1e-10:
                raise ValueError("Supplied prediction differs from actual-return predictor")
        return self.plan(state,candidates,budget,risk_level=risk_level,lambda_cost=lambda_cost)["candidates"]

    def _payload(self):
        return {"schema":VERSION,"manifest":self.manifest}

    def save(self,path):
        path = Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps({**self._payload(),"version":self.version},ensure_ascii=False,allow_nan=False),encoding="utf-8")

    @classmethod
    def load(cls,path,*,model,model_sha256,calibrator,calibrator_sha256):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if raw.get("schema") != VERSION:
            raise ValueError("Calibrated continuation schema required")
        result = cls(model,calibrator,mode=raw["manifest"]["mode"],model_sha256=model_sha256,calibrator_sha256=calibrator_sha256)
        if raw.get("manifest") != result.manifest or raw.get("version") != result.version:
            raise ValueError("Calibrated continuation artifact/reference binding mismatch")
        return result
