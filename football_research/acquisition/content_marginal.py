"""Marginalize acquired contents in query valuation, preserving the free prior.

The predictor still updates from real returns. This wrapper neither fits a new
model nor substitutes imagined returns for executed evidence. Branch estimates
are calibrated before the exact finite weighted average is formed.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path

from .gain_prior import visible_context
from .joint_planner import _budget, _cost, _number, _probabilities, UNITS
from .joint_returns import ACTION_IDS, JointReturnPredictor
from .round9_residual import ResidualEVSICalibrator, context_key
from .state import GROUP_FIELDS, parse_time
from .value import _canonical

VERSION = "content-marginal-gain-v1"
ERASURE_MODE = "marginalize_acquired_contents_given_free_prior_and_acquired_set"


class ContentMarginalGainEstimator:
    def __init__(self, model, calibrator, *, model_sha256, calibrator_sha256):
        if not isinstance(model, JointReturnPredictor) or not isinstance(calibrator, ResidualEVSICalibrator):
            raise ValueError("Frozen joint predictor and residual calibrator required")
        source = calibrator.manifest
        if (calibrator.model is None or calibrator.model.model_version != model.model_version
                or source.get("joint_model_version") != model.model_version
                or source.get("joint_artifact_sha256") != model_sha256
                or source.get("joint_training_as_of") != model.metadata["training_as_of"]):
            raise ValueError("Source calibrator joint reference binding mismatch")
        if source.get("metadata", {}).get("supervision_kind") != "same_version_heldout_fixed_predictor":
            raise ValueError("Same-version held-out source calibrator required")
        if not isinstance(calibrator_sha256, str) or not calibrator_sha256:
            raise ValueError("Explicit source calibrator artifact hash required")
        self.model, self.calibrator = model, calibrator
        self.model_sha256, self.calibrator_sha256 = model_sha256, calibrator_sha256
        self.manifest = {key:deepcopy(source[key]) for key in ("dataset_version", "trained_as_of", "training_as_of",
            "diagnostics_as_of", "information_mode", "training_match_ids", "training_matches", "joint_model_version",
            "joint_artifact_sha256", "joint_training_as_of")}
        self.manifest.update(estimator_kind=VERSION, erasure_mode=ERASURE_MODE,
            source_calibrator_version=calibrator.version, source_calibrator_sha256=calibrator_sha256,
            latest_training_label_available_at=max((parse_time(r["result_available_at"]) for r in calibrator.records)).isoformat(),
            target="source_calibrated_empirical_predictor_update_gain_marginalized_over_acquired_content",
            weighting="frozen free-prior joint marginal, never actual observed category or posterior",
            branch_calibration_order="calibrate each conditional EVSI, then probability-weighted sum",
            unsupported_rule="any positive-mass unsupported branch makes entire candidate unsupported",
            uncertainty="no aggregate interval or standard error; original branch diagnostics retained",
            new_fitting=False)
        self.version = VERSION+":"+sha256(_canonical(self._payload()).encode()).hexdigest()[:20]

    def _visible(self, state):
        clean, acquired, attempted = visible_context(state, model=self.model, trained_as_of=self.manifest["trained_as_of"])
        if parse_time(self.manifest["latest_training_label_available_at"]) >= parse_time(clean["prediction_cutoff"]):
            raise ValueError("Calibration outcomes must precede runtime prediction")
        return clean, acquired, attempted

    @staticmethod
    def _free(clean):
        free = {key:deepcopy(clean[key]) for key in ("match_id", "prediction_cutoff", "information_mode", "base")}
        free["groups"] = {g:{"status":"not_acquired", "values":{}} for g in GROUP_FIELDS}
        return free

    def _estimate(self, clean, acquired, action):
        aid = action["action_id"]
        key = context_key(aid, acquired)
        free = self._free(clean)
        branches = []
        if not acquired:
            # Same model method, source correction and floating-point value as
            # original runtime estimation; no weighted arithmetic for this case.
            enumeration = self.model.enumerate_action(free, action)
            estimate = self.calibrator.estimate_numbers(aid, [], enumeration["evsi"], risk_level="mean")
            branches.append({"observation":None, "probability":1., "model_evsi":estimate["model_evsi"],
                "expected_gain":estimate["expected_gain"], "support":estimate["support"], "estimate":estimate,
                "hypothetical":True, "scope":"no_acquired_content_to_marginalize"})
        elif len(acquired) == 1:
            # These are marginals of the FREE joint cube. Actual content and its
            # posterior were removed before asking for the distribution.
            first = self.model.enumerate_action(free, {"action_id":acquired[0]})
            for branch in first["branches"]:
                mass = float(branch["probability"])
                if not math.isfinite(mass) or mass < 0:
                    raise ValueError("Invalid marginal return mass")
                if mass == 0:
                    continue
                conditional = self.model.enumerate_action(branch["next_state"], action, hypothetical=True)
                estimate = self.calibrator.estimate_numbers(aid, acquired, conditional["evsi"], risk_level="mean")
                branches.append({"observation":branch["observation"], "probability":mass,
                    "model_evsi":estimate["model_evsi"], "expected_gain":estimate["expected_gain"],
                    "support":estimate["support"], "estimate":estimate, "hypothetical":True,
                    "scope":"model_only_category_branch_not_a_tool_receipt"})
        else:
            raise ValueError("No remaining modeled action after both tools acquired")
        total_mass = math.fsum(b["probability"] for b in branches)
        if not branches or abs(total_mass-1.) > 1e-10:
            raise ValueError("Complete normalized positive-mass return branches required")
        unsupported = [b["observation"] for b in branches if b["expected_gain"] is None]
        if unsupported:
            mean = None
        elif not acquired:
            mean = branches[0]["expected_gain"]  # preserve exact empty-state value
        else:
            mean = math.fsum(b["probability"]*b["expected_gain"] for b in branches)
        marginal_evsi = branches[0]["model_evsi"] if not acquired else math.fsum(b["probability"]*b["model_evsi"] for b in branches)
        def minimum(name):
            return min(b["estimate"][name] for b in branches)
        status = "unsupported" if unsupported else "supported_all_branches"
        return {"action_id":aid, "acquired_action_ids":list(acquired), "context_key":key,
            "expected_gain":mean, "g_hat":mean, "decision_gain":mean, "model_evsi":None, "evsi_bin":None,
            "lower_gain":None, "upper_gain":None, "standard_error":None, "uncertainty":None,
            "marginal_model_evsi":marginal_evsi, "branches":branches, "branch_probability_mass":total_mass,
            "positive_mass_branches":len(branches), "unsupported_observations":unsupported,
            "support":minimum("support"), "support_matches":minimum("support_matches"),
            "parent_support":minimum("parent_support"), "effective_support":minimum("effective_support"),
            "time_blocks":minimum("time_blocks"), "support_status":status,
            "support_semantics":"minimum source branch diagnostic; not a new independent sample count",
            "fallback_reason":"positive_mass_branch_unsupported" if unsupported else None,
            "gate_reason":"stop_no_support" if mean is None else "stop_nonpositive_update_gain" if mean <= 0 else "eligible",
            "risk_level":"mean", "uncertainty_kind":"aggregate_uncertainty_not_estimated",
            "gain_semantics":"source_calibrated_one_step_update_gain_after_content_marginalization",
            "estimator_kind":VERSION, "erasure_mode":ERASURE_MODE, "estimator_version":self.version,
            "source_calibrator_version":self.calibrator.version, "source_calibrator_sha256":self.calibrator_sha256,
            "joint_model_version":self.model.model_version, "cost_vector":_cost(action)}

    def estimate_value(self, state, candidates, *, prediction=None, risk_level="mean"):
        if risk_level != "mean":
            raise ValueError("Content-marginal control supports only mean decisions")
        clean, acquired, attempted = self._visible(state)
        if prediction is not None:
            supplied = _probabilities(prediction["probabilities"], "Supplied prediction")
            actual = self.model.predict_partial(state)["probabilities"]
            if max(abs(a-b) for a,b in zip(supplied,actual)) > 1e-10:
                raise ValueError("Supplied prediction differs from frozen actual-return predictor")
        if not isinstance(candidates, (list,tuple)):
            raise ValueError("Candidates must be a sequence")
        output, seen = [], set()
        for action in candidates:
            aid = action.get("action_id") if isinstance(action,dict) else None
            if aid not in ACTION_IDS or aid in acquired or aid in attempted or aid in seen:
                raise ValueError("Unique unpurchased modeled action required")
            if type(action.get("available",True)) is not bool:
                raise ValueError("Boolean action availability required")
            seen.add(aid)
            output.append(self._estimate(clean,acquired,action))
        return output

    def plan(self, state, actions, budget, *, risk_level="mean", lambda_cost=None):
        if risk_level != "mean":
            raise ValueError("Content-marginal control supports only mean decisions")
        if not isinstance(actions,(list,tuple)):
            raise ValueError("Actions must be a sequence")
        clean, acquired, attempted = self._visible(state)
        used = set(acquired) | attempted
        limits, spent = _budget(budget)
        weights = lambda_cost or {}
        if set(weights)-set(UNITS):
            raise ValueError("Unknown cost-conversion unit")
        lam = {u:_number(weights.get(u,0.),"lambda "+u) for u in UNITS}
        for action in actions:
            if not isinstance(action,dict) or action.get("action_id") not in ACTION_IDS or type(action.get("available",True)) is not bool:
                raise ValueError("Modeled action identity and Boolean availability required")
            if action.get("evidence_id") is not None and (not isinstance(action["evidence_id"],str) or not action["evidence_id"]):
                raise ValueError("Evidence identity must be a nonempty inert string")
        used_evidence = {a["evidence_id"] for a in actions if a["action_id"] in used and a.get("evidence_id") is not None}
        output, seen = [], set()
        for action in sorted(actions,key=lambda a:a["action_id"]):
            aid = action["action_id"]
            if aid in seen:
                raise ValueError("Duplicate candidate action")
            seen.add(aid)
            cost = _cost(action)
            reason = ("action_unavailable" if not action.get("available",True) else "already_attempted" if aid in used else
                "evidence_already_observed" if action.get("evidence_id") in used_evidence else
                "physical_budget_exhausted" if any(spent[u]+cost[u] > cap+1e-12 for u,cap in limits.items()) else None)
            if reason:
                output.append({"action_id":aid,"feasible":False,"reason":reason})
                continue
            estimate = self._estimate(clean,acquired,action)
            penalty = math.fsum(lam[u]*cost[u] for u in UNITS)
            mean = estimate["expected_gain"]
            net = mean-penalty if mean is not None else None
            output.append({**estimate,"feasible":mean is not None,"reason":estimate["gate_reason"],"cost_penalty":penalty,
                "mean_net_gain":net,"lower_net_gain":None,"net_gain":net,"selection_score":round(net,12) if net is not None else None})
        eligible = [r for r in output if r["feasible"] and r["selection_score"] > 0]
        selected = min(eligible,key=lambda r:(-r["selection_score"],r["action_id"])) if eligible else None
        return {"schema_version":VERSION,"estimator_kind":VERSION,"erasure_mode":ERASURE_MODE,
            "estimator_version":self.version,"model_version":self.model.model_version,
            "source_calibrator_version":self.calibrator.version,"source_calibrator_sha256":self.calibrator_sha256,
            "decision":"acquire" if selected else "stop","action_id":selected["action_id"] if selected else None,
            "reason":"positive_content_marginal_net_gain" if selected else "no_positive_supported_content_marginal_gain",
            "risk_level":"mean","horizon":1,"physical_budget":{"limits":limits,"spent":spent},
            "lambda_cost":lam,"selection_score_precision_decimal_places":12,"candidates":output}

    def _payload(self):
        return {"schema":VERSION,"manifest":self.manifest}

    def save(self,path):
        path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps({**self._payload(),"version":self.version},ensure_ascii=False,allow_nan=False),encoding="utf-8")

    @classmethod
    def load(cls,path,*,model,model_sha256,calibrator,calibrator_sha256):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if raw.get("schema") != VERSION:
            raise ValueError("Content-marginal wrapper schema required")
        result = cls(model,calibrator,model_sha256=model_sha256,calibrator_sha256=calibrator_sha256)
        if raw.get("manifest") != result.manifest or raw.get("version") != result.version:
            raise ValueError("Content-marginal artifact/reference binding mismatch")
        return result
