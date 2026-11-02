"""Past-OOF residual calibration of a fixed predictor's one-step update gain.

This estimates empirical predictor-update gain, which may be negative. It is
not Bayes-optimal information value. Week-cluster bounds are descriptive mean
estimation summaries, not posterior uncertainty or coverage guarantees.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from .joint_returns import JointReturnPredictor, ACTION_IDS
from .joint_planner import _budget, _cost, _number, _probabilities, UNITS
from .state import parse_time, sanitize_state
from .value import _canonical

VERSION = "residual-evsi-calibrator-v1"
BIN_EDGES = (.02, .05, .10)


def evsi_bin(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < -1e-10:
        raise ValueError("Model EVSI must be finite and nonnegative within numerical tolerance")
    return int(np.searchsorted(BIN_EDGES, max(0., value), side="right"))


def context_key(action_id, acquired):
    used = tuple(sorted(acquired))
    if action_id not in ACTION_IDS or len(used) != len(set(used)) or set(used)-set(ACTION_IDS) or action_id in used:
        raise ValueError("Illegal one-step action/observed-action context")
    return _canonical([action_id, list(used)])


def residual_summary(local, parent, *, min_support=12, shrinkage=20.):
    """Deduplicated match weights; merge overlapping influence before squaring."""
    def unique(rows):
        found = {}
        for row in rows:
            mid = row["match_id"]
            item = (float(row["residual"]), parse_time(row["kickoff"]).isocalendar()[:2])
            if mid in found and found[mid] != item:
                raise ValueError("Conflicting residuals for one match/context")
            found[mid] = item
        return found
    broad, narrow = unique(parent), unique(local)
    if not set(narrow) <= set(broad):
        raise ValueError("Local records must be parent members")
    n, total = len(narrow), len(broad)
    common = {"support": n, "parent_support": total, "support_matches": n,
              "local_match_ids": sorted(narrow), "parent_match_ids": sorted(broad)}
    if total < min_support:
        return {**common, "residual_mean": None, "standard_error": None, "local_weight": 0.,
                "effective_support": 0., "time_blocks": 0, "week_influences": {},
                "support_status": "unsupported", "fallback_reason": "parent_support_below_minimum"}
    pm = sum(v[0] for v in broad.values())/total
    lm = sum(v[0] for v in narrow.values())/n if n else pm
    w = n/(n+shrinkage) if n >= min_support else 0.
    influence = defaultdict(float)
    weights = defaultdict(float)
    for mid, (value, week) in broad.items():
        influence[week] += (1-w)*(value-pm)/total
        weights[mid] += (1-w)/total
    if w:
        for mid, (value, week) in narrow.items():
            influence[week] += w*(value-lm)/n
            weights[mid] += w/n
    blocks = len(influence)
    se = math.sqrt(blocks/(blocks-1)*sum(v*v for v in influence.values())) if blocks >= 2 else None
    return {**common, "residual_mean": w*lm+(1-w)*pm, "local_mean": lm if n else None,
            "parent_mean": pm, "standard_error": se, "local_weight": w,
            "effective_support": 1/sum(v*v for v in weights.values()), "time_blocks": blocks,
            "week_influences": {f"{year}-W{week:02d}": v for (year, week), v in sorted(influence.items())},
            "support_status": "supported_local" if w else "supported_parent",
            "fallback_reason": None if w else "local_support_below_minimum_use_parent"}


class ResidualEVSICalibrator:
    def __init__(self, *, min_support=12, shrinkage=20., risk_multiplier=1.64):
        if (type(min_support) is not int or min_support < 1 or not math.isfinite(shrinkage) or shrinkage < 0
                or not math.isfinite(risk_multiplier) or risk_multiplier < 0):
            raise ValueError("Invalid residual calibration parameters")
        self.min_support, self.shrinkage, self.risk_multiplier = min_support, float(shrinkage), float(risk_multiplier)
        self.records, self.cells, self.manifest, self.calibration = [], {}, {}, {}
        self.model, self.version = None, VERSION

    def fit(self, samples, *, as_of, diagnostics_as_of, dataset_version, joint_model, joint_artifact_sha256,
            calibration_samples=(), metadata=None):
        cutoff, diagnostic_cutoff = parse_time(as_of), parse_time(diagnostics_as_of)
        if cutoff > diagnostic_cutoff:
            raise ValueError("Calibration-fit cutoff exceeds diagnostic availability cutoff")
        train = self._validate(samples, cutoff)
        tail = self._validate(calibration_samples, diagnostic_cutoff)
        if not train:
            raise ValueError("At least one eligible past OOF residual required")
        train_ids, tail_ids = {r["match_id"] for r in train}, {r["match_id"] for r in tail}
        if train_ids & tail_ids:
            raise ValueError("Training and diagnostic matches overlap")
        if tail and (max(parse_time(r["kickoff"]) for r in train) >= min(parse_time(r["kickoff"]) for r in tail)
                     or cutoff >= min(parse_time(r["prediction_cutoff"]) for r in tail)):
            raise ValueError("Diagnostic tail must strictly follow calibration fitting")
        modes = {r["information_mode"] for r in train+tail}
        if len(modes) != 1 or next(iter(modes)) != joint_model.metadata["information_mode"]:
            raise ValueError("Information mode mismatch")
        self.records = train
        parents = defaultdict(list)
        for row in train:
            parents[row["context_key"]].append(row)
        self.cells = {}
        for key, broad in parents.items():
            for b in range(4):
                self.cells[_canonical([key,b])] = residual_summary([r for r in broad if r["evsi_bin"] == b], broad,
                    min_support=self.min_support, shrinkage=self.shrinkage)
        self.manifest = {"dataset_version": dataset_version, "trained_as_of": cutoff.isoformat(),
            "training_as_of": cutoff.isoformat(), "diagnostics_as_of": diagnostic_cutoff.isoformat(),
            "training_match_ids": sorted(train_ids), "calibration_match_ids": sorted(tail_ids),
            "training_matches": len(train_ids), "calibration_matches": len(tail_ids),
            "oof_samples": len(train), "calibration_samples": len(tail), "information_mode": next(iter(modes)),
            "joint_model_version": joint_model.model_version, "joint_artifact_sha256": joint_artifact_sha256,
            "joint_training_as_of": joint_model.metadata["training_as_of"],
            "target": "empirical_predictor_update_gain_minus_fitted_model_evsi",
            "bin_edges": list(BIN_EDGES), "bin_side": "right", "support_unit": "unique_match_per_action_and_acquired_mask",
            "training_joint_versions": sorted({r["oof_predictor_version"] for r in train}),
            "diagnostic_joint_versions": sorted({r["oof_predictor_version"] for r in tail}),
            "calibration_use": "diagnostics_only_no_parameter_or_lambda_selection",
            "bound_scope": "descriptive_UTC_week_cluster_mean_estimation_lower_summary_no_coverage_guarantee",
            "limitations": ["OOF and deployed joint predictor populations differ", "complete historical projections are not adaptively sampled live tools",
                "conditional residual stationarity not established", "no post-selection or time-drift uniform coverage guarantee",
                "mean bounds exclude uncertainty in fitted joint parameters and model EVSI"], "metadata": metadata or {}}
        self.model = joint_model
        tail_records = [self.diagnose_record(r) for r in tail]
        for record in tail_records:
            record.pop("estimator_version", None)
        self.calibration = {"scope": "independent_chronological_tail_only", "version_binding": "enclosing_artifact_version",
                            "records": tail_records}
        self.version = VERSION+":"+sha256(_canonical(self._payload()).encode()).hexdigest()[:20]
        return self

    def _validate(self, samples, cutoff):
        records = {}
        for row in samples:
            clean = sanitize_state(row["state"])
            mid = str(row["match_id"])
            prediction_at, kickoff = parse_time(clean["prediction_cutoff"]), parse_time(row["kickoff"])
            if mid != clean["match_id"] or not parse_time(row["oof_train_cutoff"]) < prediction_at < kickoff <= parse_time(row["result_available_at"]) <= cutoff:
                raise ValueError("Residual training requires strict forward OOF and known outcomes")
            if mid in set(map(str,row["oof_train_ids"])):
                raise ValueError("OOF target occurs in training IDs")
            acquired = sorted(row["acquired_action_ids"])
            # Missing is an acquired response too; never infer this from Boolean mask.
            for aid, groups in ((ACTION_IDS[0],("elo","recent_form")), (ACTION_IDS[1],("xg",))):
                observed = all(clean["groups"][g]["status"] in {"observed","source_missing"} for g in groups)
                if observed != (aid in acquired):
                    raise ValueError("Acquired-action mask disagrees with public group statuses")
            aid = row["action"]["action_id"]
            key = context_key(aid, acquired)
            value, gain = float(row["model_evsi"]), float(row["delta"])
            if not math.isfinite(gain):
                raise ValueError("Finite empirical predictor-update gain required")
            b = evsi_bin(value)
            record = {"match_id": mid, "kickoff": row["kickoff"], "prediction_cutoff": clean["prediction_cutoff"],
                "information_mode": clean["information_mode"], "result_available_at": row["result_available_at"],
                "action_id": aid, "acquired_action_ids": acquired, "context_key": key, "evsi_bin": b,
                "model_evsi": max(0.,value), "gain": gain, "residual": gain-max(0.,value),
                "oof_train_cutoff": row["oof_train_cutoff"], "oof_train_ids": list(row["oof_train_ids"]),
                "oof_manifest_ref": row["oof_manifest_ref"], "oof_predictor_version": row["oof_predictor_version"]}
            identity = (mid,key)
            if identity in records and records[identity] != record:
                raise ValueError("Conflicting duplicate match/context residual")
            records[identity] = record
        return list(records.values())

    def estimate_numbers(self, action_id, acquired, model_evsi, *, risk_level="mean"):
        if risk_level not in {"mean","conservative"}:
            raise ValueError("Risk level must be mean or conservative")
        key, b = context_key(action_id, acquired), evsi_bin(model_evsi)
        summary = self.cells.get(_canonical([key,b]))
        if summary is None:
            summary = residual_summary([],[],min_support=self.min_support,shrinkage=self.shrinkage)
        # Raw membership is retained in the model, not repeated in every trace.
        result = {k:v for k,v in summary.items() if k not in {"local_match_ids","parent_match_ids","week_influences"}}
        mean = max(0.,model_evsi)+summary["residual_mean"] if summary["residual_mean"] is not None else None
        se = summary["standard_error"]
        lower = mean-self.risk_multiplier*se if mean is not None and se is not None else None
        upper = mean+self.risk_multiplier*se if mean is not None and se is not None else None
        gain = mean if risk_level == "mean" else lower
        gate = ("stop_no_support" if mean is None else "stop_uncertainty_unavailable" if gain is None
                else "stop_nonpositive_update_gain" if gain <= 0 else "eligible")
        return {**result,"action_id":action_id,"acquired_action_ids":sorted(acquired),"context_key":key,"evsi_bin":b,
            "model_evsi":max(0.,model_evsi),"expected_gain":mean,"g_hat":mean,"lower_gain":lower,"upper_gain":upper,
            "decision_gain":gain,"risk_level":risk_level,"gate_reason":gate,"uncertainty":se,
            "uncertainty_kind":"descriptive_week_cluster_mean_se_no_coverage_guarantee",
            "gain_semantics":"empirical_fixed_predictor_one_step_update_gain",
            "bound_semantics":"one_step_mean_estimation_summary_not_multistep_plan_or_joint_posterior_bound",
            "estimator_version":self.version}

    def diagnose_record(self, record):
        result = self.estimate_numbers(record["action_id"],record["acquired_action_ids"],record["model_evsi"])
        return {"match_id":record["match_id"],"kickoff":record["kickoff"],"gain":record["gain"],
                "oof_predictor_version":record["oof_predictor_version"],**result}

    def estimate_value(self, state, candidates, *, prediction=None, risk_level="mean"):
        if self.model is None or not self.manifest:
            raise ValueError("Fitted calibrator must be bound to its reference joint model")
        if parse_time(state["prediction_cutoff"]) < parse_time(self.manifest["trained_as_of"]):
            raise ValueError("Calibrator trained after requested runtime state")
        current = self.model.predict_partial(state)
        if prediction is not None:
            supplied = _probabilities(prediction["probabilities"], "Supplied prediction")
            if max(abs(a-b) for a,b in zip(supplied,current["probabilities"])) > 1e-10:
                raise ValueError("Supplied prediction differs from bound joint model")
        acquired = self.model.observed_actions(state)
        output=[]
        for action in candidates:
            enumeration = self.model.enumerate_action(state,action)
            result = self.estimate_numbers(action["action_id"],acquired,enumeration["evsi"],risk_level=risk_level)
            result.update(cost_vector=_cost(action),joint_model_version=self.model.model_version)
            output.append(result)
        return output

    def plan(self,state,actions,budget,*,risk_level="mean",lambda_cost=None):
        if risk_level not in {"mean","conservative"}:
            raise ValueError("Risk level must be mean or conservative")
        if not isinstance(actions,(list,tuple)):
            raise ValueError("Actions must be a sequence")
        limits, spent = _budget(budget)
        weights = lambda_cost or {}
        if set(weights)-set(UNITS):
            raise ValueError("Unknown cost-conversion unit")
        lam = {u:_number(weights.get(u,0.),"lambda "+u) for u in UNITS}
        self.model.predict_partial(state)  # Runtime token/failure/time boundary, even at zero budget.
        acquired = set(self.model.observed_actions(state))
        acquired.update(event["action_id"] for event in state.get("action_history",[]) if isinstance(event,dict) and isinstance(event.get("action_id"),str))
        for action in actions:
            if not isinstance(action,dict) or action.get("action_id") not in ACTION_IDS or type(action.get("available",True)) is not bool:
                raise ValueError("Unique modeled action ID and Boolean availability required")
            if action.get("evidence_id") is not None and (not isinstance(action["evidence_id"],str) or not action["evidence_id"]):
                raise ValueError("Evidence identity must be an inert nonempty string")
        used_evidence={a["evidence_id"] for a in actions if a["action_id"] in acquired and a.get("evidence_id") is not None}
        seen, output = set(),[]
        for action in sorted(actions,key=lambda a:a["action_id"]):
            aid = action["action_id"]
            if aid in seen:
                raise ValueError("Duplicate candidate action")
            seen.add(aid)
            cost = _cost(action)
            reason = ("action_unavailable" if not action.get("available",True) else "already_attempted" if aid in acquired else
                      "evidence_already_observed" if action.get("evidence_id") in used_evidence else
                      "physical_budget_exhausted" if any(spent[u]+cost[u] > cap+1e-12 for u,cap in limits.items()) else None)
            if reason:
                output.append({"action_id":aid,"feasible":False,"reason":reason})
                continue
            estimate = self.estimate_value(state,[action],risk_level=risk_level)[0]
            penalty = math.fsum(lam[u]*cost[u] for u in UNITS)
            mean,lower = estimate["expected_gain"],estimate["lower_gain"]
            decision_gain = estimate["decision_gain"]
            net = decision_gain-penalty if decision_gain is not None else None
            feasible = decision_gain is not None
            output.append({**estimate,"feasible":feasible,"reason":estimate["gate_reason"],"cost_penalty":penalty,
                "mean_net_gain":mean-penalty if mean is not None else None,"lower_net_gain":lower-penalty if lower is not None else None,
                "net_gain":net,"selection_score":round(net,12) if net is not None else None})
        eligible = [r for r in output if r["feasible"] and r["selection_score"] > 0]
        selected = min(eligible,key=lambda r:(-r["selection_score"],r["action_id"])) if eligible else None
        return {"schema_version":VERSION,"estimator_version":self.version,"model_version":self.model.model_version,
            "decision":"acquire" if selected else "stop","action_id":selected["action_id"] if selected else None,
            "reason":"positive_calibrated_one_step_net_gain" if selected else "no_positive_supported_calibrated_gain",
            "risk_level":risk_level,"horizon":1,"physical_budget":{"limits":limits,"spent":spent},
            "lambda_cost":lam,"selection_score_precision_decimal_places":12,"candidates":output}

    def _payload(self):
        return {"schema":VERSION,"parameters":{"min_support":self.min_support,"shrinkage":self.shrinkage,"risk_multiplier":self.risk_multiplier},
                "records":self.records,"cells":self.cells,"manifest":self.manifest,"calibration":self.calibration}

    def save(self,path):
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps({**self._payload(),"version":self.version},ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")

    @classmethod
    def load(cls,path,*,model,model_sha256):
        raw=json.loads(Path(path).read_text(encoding="utf-8"))
        if raw.get("schema") != VERSION or not isinstance(model,JointReturnPredictor):
            raise ValueError("Residual calibrator and learned joint model schemas required")
        result=cls(**raw["parameters"])
        for name in ("records","cells","manifest","calibration"):
            setattr(result,name,raw[name])
        expected=VERSION+":"+sha256(_canonical(result._payload()).encode()).hexdigest()[:20]
        if raw.get("version") != expected:
            raise ValueError("Residual calibrator artifact hash mismatch")
        if result.manifest["joint_model_version"] != model.model_version or result.manifest["joint_artifact_sha256"] != model_sha256:
            raise ValueError("Residual calibrator reference joint version/hash binding mismatch")
        if (result.manifest["joint_training_as_of"] != model.metadata["training_as_of"]
                or parse_time(result.manifest["diagnostics_as_of"]) < parse_time(result.manifest["trained_as_of"])):
            raise ValueError("Residual calibrator time binding mismatch")
        result.version,result.model=expected,model
        return result
