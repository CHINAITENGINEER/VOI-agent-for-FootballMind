"""Learned finite joint tool returns and coherent Bayesian conditioning.

P(Y,Zr,Zx|H) = p0(Y|H) q(Zr,Zx|Y,p0(H)). q uses strictly past
OOF free-context forecasts, Gaussian kernel weights and a fixed Dirichlet
mass. Quantile summaries lose information. Conditioning only on p0 is a
model assumption, not a sufficient-statistic theorem for football.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from football_research.baselines.models import _training, _label
from .contextual_partial import ContextualPartialPredictor
from .state import GROUP_FIELDS, parse_time, sanitize_state
from .value import _canonical

VERSION = "learned-joint-returns-v1"
ACTION_IDS = ("results_view", "xg_view")
RESULT_CATEGORIES = tuple(f"elo{i}_form{j}" for i in range(3) for j in range(3))+("missing",)
XG_CATEGORIES = ("xg0", "xg1", "xg2", "missing")
SUMMARY_FIELDS = {"elo_difference": ("home_elo", "away_elo"),
    "form_difference": ("home_rolling_points", "away_rolling_points"),
    "net_xg_difference": ("home_rolling_xg_for", "home_rolling_xg_against", "away_rolling_xg_for", "away_rolling_xg_against")}
HYPOTHESIS_KEY = "joint_return_hypothesis"


def entropy(probabilities):
    return -sum(float(p)*math.log(float(p)) for p in probabilities if p > 0)


def _probabilities(values):
    p = np.asarray(values,dtype=float)
    if p.shape != (3,) or not np.isfinite(p).all() or (p <= 0).any() or abs(p.sum()-1) > 1e-8:
        raise ValueError("Strictly positive normalized H/D/A probabilities required")
    return p/p.sum()


def _summary(values):
    result = {}
    for name,fields in SUMMARY_FIELDS.items():
        observed = [values.get(f) for f in fields]
        if any(v is None for v in observed):
            result[name] = None
            continue
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in observed):
            raise ValueError("Finite numeric training/observed summary values required")
        result[name] = float(observed[0]-observed[1]) if len(observed) == 2 else float(observed[0]-observed[1]-observed[2]+observed[3])
    return result


def extract_oof_contexts(samples):
    """Strip old delta/paid observations; only original empty-mask priors remain."""
    contexts = {}
    for row in samples:
        state = sanitize_state(row["state"])
        if any(g["status"] != "not_acquired" for g in state["groups"].values()):
            continue
        if row.get("acquired_action_ids"):
            continue
        record = {"match_id": str(row["match_id"]),"probabilities": _probabilities(row["prediction_before"]).tolist(),
            "prediction_cutoff": state["prediction_cutoff"],"kickoff": state["base"]["kickoff"],
            "oof_train_cutoff": row["oof_train_cutoff"],"oof_train_ids": list(row["oof_train_ids"]),
            "oof_manifest_ref": row["oof_manifest_ref"],"oof_predictor_version": row.get("oof_predictor_version")}
        previous = contexts.get(record["match_id"])
        if previous is not None and previous != record:
            raise ValueError("Inconsistent duplicate OOF free-context prior")
        contexts[record["match_id"]] = record
    return contexts


class JointReturnPredictor:
    version = VERSION

    def __init__(self, base_predictor=None, bandwidth=.2, dirichlet_mass=5.):
        if not math.isfinite(bandwidth) or bandwidth <= 0 or not math.isfinite(dirichlet_mass) or dirichlet_mass <= 0:
            raise ValueError("Positive fixed kernel bandwidth and smoothing mass required")
        self.base_predictor = base_predictor
        self.bandwidth,self.dirichlet_mass = float(bandwidth),float(dirichlet_mass)
        self.quantiles,self.records,self.metadata = {},[],{}
        self.model_version = ""
        self._arrays,self._prior_cache,self._joint_cache = None,{},{}

    def fit(self, rows, oof_contexts, *, as_of, dataset_version, actions):
        if not isinstance(self.base_predictor,ContextualPartialPredictor):
            raise ValueError("Frozen contextual-partial-v1 base predictor required")
        cutoff = parse_time(as_of)
        if parse_time(self.base_predictor.metadata["training_as_of"]) > cutoff:
            raise ValueError("Base predictor trained after joint-model cutoff")
        if tuple(a["action_id"] for a in actions) != ACTION_IDS:
            raise ValueError("This model has exactly results_view and xg_view")
        if set(actions[0]["group_ids"]) != {"elo","recent_form"} or set(actions[1]["group_ids"]) != {"xg"}:
            raise ValueError("Unexpected finite tool projection contract")
        selected = _training(rows,as_of)
        eligible = []
        for row in selected:
            context = oof_contexts.get(str(row["match_id"]))
            if context is None:
                continue
            prediction_at,kickoff = parse_time(row["prediction_cutoff"]),parse_time(row["kickoff"])
            if not parse_time(context["oof_train_cutoff"]) < prediction_at < kickoff:
                raise ValueError("Joint training requires forward OOF context")
            if parse_time(context["prediction_cutoff"]) != prediction_at or parse_time(context["kickoff"]) != kickoff:
                raise ValueError("OOF/row prediction-time identity mismatch")
            if str(row["match_id"]) in set(map(str,context["oof_train_ids"])):
                raise ValueError("OOF target occurs in its prior predictor training IDs")
            for field in ("features_available_at","fixture_available_at"):
                if row.get(field) is None or parse_time(row[field]) > prediction_at:
                    raise ValueError("Training feature or fixture unavailable at historical cutoff")
            if row["information_mode"] != self.base_predictor.metadata["information_mode"]:
                raise ValueError("Information mode mismatch")
            eligible.append((row,context,_summary(row)))
        if not eligible:
            raise ValueError("No eligible forward OOF joint-return training matches")
        self.quantiles = {}
        for field in SUMMARY_FIELDS:
            values = [s[field] for _,_,s in eligible if s[field] is not None]
            if not values:
                raise ValueError("Missing training support for summary: "+field)
            self.quantiles[field] = np.quantile(values,[1/3,2/3]).tolist()
        self.records = []
        for row,context,summary in eligible:
            categories = self._categories(summary)
            self.records.append({"match_id": str(row["match_id"]),"outcome": _label(row),
                "probabilities": _probabilities(context["probabilities"]).tolist(),
                "results_category": categories[0],"xg_category": categories[1],
                "prediction_cutoff": row["prediction_cutoff"],"kickoff": row["kickoff"],
                "result_available_at": row["result_available_at"],"oof_train_cutoff": context["oof_train_cutoff"],
                "oof_manifest_ref": context["oof_manifest_ref"],"oof_predictor_version": context["oof_predictor_version"]})
        self.metadata = {"dataset_version": dataset_version,"training_as_of": cutoff.isoformat(),"trained_as_of": cutoff.isoformat(),
            "information_mode": self.base_predictor.metadata["information_mode"],"training_rows": len(eligible),
            "training_match_ids": sorted(r["match_id"] for r in self.records),"input_training_matches": len(selected),
            "excluded_without_forward_oof_prior": len(selected)-len(eligible),
            "base_model_version": self.base_predictor.model_version,
            "base_training_match_ids": self.base_predictor.metadata["training_match_ids"],
            "quantile_fit_scope": "eligible training matches only, fixed tertile boundaries, duplicate boundaries retained",
            "summary_fields": {k:list(v) for k,v in SUMMARY_FIELDS.items()},
            "results_categories": list(RESULT_CATEGORIES),"xg_categories": list(XG_CATEGORIES),
            "outcome_class_counts": {str(y):sum(r["outcome"] == y for r in self.records) for y in range(3)},
            "probability_model": "p0(Y|free_fixture_context) * q(Z_results,Z_xg|Y,p0)",
            "q_context": "past forward OOF p0, Gaussian probability-vector kernel",
            "compression": "results: tertile(Elo difference) x tertile(recent points difference); xg: tertile(net xg difference); missing explicit",
            "failure_policy": "provider errors are unmodeled and rejected; missing summary is not provider failure",
            "identity_time_interpretation": self.base_predictor.metadata["identity_time_interpretation"],
            "sufficiency_assumption": "q depends on free context through p0 only; unproven and potentially misspecified",
            "guarantee_scope": "finite fitted joint model coherence only, not real-world calibration or positive realized gain"}
        self._arrays,self._prior_cache,self._joint_cache = None,{},{}
        self.model_version = VERSION+":"+sha256(_canonical(self._payload()).encode()).hexdigest()[:20]
        self.version = self.model_version
        return self

    def _categories(self, summary):
        def bucket(name):
            return int(np.searchsorted(self.quantiles[name],summary[name],side="right"))
        result = "missing" if summary["elo_difference"] is None or summary["form_difference"] is None else f"elo{bucket('elo_difference')}_form{bucket('form_difference')}"
        xg = "missing" if summary["net_xg_difference"] is None else f"xg{bucket('net_xg_difference')}"
        return result,xg

    def _visible(self,state,*,hypothetical=False):
        if HYPOTHESIS_KEY in state and not hypothetical:
            raise ValueError("Hypothetical return tokens are forbidden at runtime prediction boundary")
        failure_tokens = ("FAILED","TIMEOUT","PROVIDER_ERROR","HTTP_ERROR","NETWORK_ERROR")
        for event in state.get("action_history",[]):
            if isinstance(event,dict) and any(token in str(event.get("status","")).upper() for token in failure_tokens):
                raise ValueError("Provider failure is unmodeled, not a missing-summary observation")
        for group in state.get("groups",{}).values():
            # A never-acquired payload, including its reason/metadata, is hidden.
            # Only its public status can be considered until a tool is attempted.
            if (isinstance(group,dict) and group.get("status","not_acquired") != "not_acquired" and
                    any(token in (str(group.get("status",""))+" "+str(group.get("reason",""))).upper() for token in failure_tokens)):
                raise ValueError("Provider failure is unmodeled, not a missing-summary observation")
        clean = sanitize_state(state)
        if not self.metadata:
            raise ValueError("Fit joint return model before inference")
        if clean["information_mode"] != self.metadata["information_mode"] or parse_time(clean["prediction_cutoff"]) < parse_time(self.metadata["training_as_of"]):
            raise ValueError("Inference information mode/time boundary violation")
        values = {f:v for group in clean["groups"].values() for f,v in group["values"].items()}
        summary = _summary(values)
        categories = self._categories(summary)
        observations = {}
        for i,(action,groups) in enumerate((("results_view",("elo","recent_form")),("xg_view",("xg",)))):
            statuses = [clean["groups"][g]["status"] for g in groups]
            known = [s in {"observed","source_missing"} for s in statuses]
            if all(known):
                observations[action] = categories[i]
            elif any(known):
                raise ValueError("Joint model requires the complete bundled tool projection")
        if HYPOTHESIS_KEY in state:
            supplied = state[HYPOTHESIS_KEY]
            if not isinstance(supplied,dict) or supplied.get("schema") != VERSION or supplied.get("hypothetical") is not True:
                raise ValueError("Invalid planning-only return token envelope")
            tokens = supplied.get("observations")
            if not isinstance(tokens,dict) or set(tokens)-set(ACTION_IDS):
                raise ValueError("Invalid hypothetical action tokens")
            for action,token in tokens.items():
                vocabulary = RESULT_CATEGORIES if action == "results_view" else XG_CATEGORIES
                if token not in vocabulary or (action in observations and observations[action] != token):
                    raise ValueError("Invalid or conflicting hypothetical return token")
                observations[action] = token
        return clean,observations

    def observed_actions(self,state,*,hypothetical=False):
        return sorted(self._visible(state,hypothetical=hypothetical)[1])

    def observed_categories(self,state,*,hypothetical=False):
        return dict(self._visible(state,hypothetical=hypothetical)[1])

    def _free_prior(self,clean):
        free = {k:deepcopy(clean[k]) for k in ("match_id","prediction_cutoff","information_mode","base")}
        free["groups"] = {g:{"status":"not_acquired","values":{}} for g in GROUP_FIELDS}
        key = _canonical(free)
        if key not in self._prior_cache:
            self._prior_cache[key] = self.base_predictor.predict_partial(free)
        return self._prior_cache[key]

    def _q(self,p0,*,condition_on_p0=True):
        key = (tuple(p0),bool(condition_on_p0))
        if key in self._joint_cache:
            return self._joint_cache[key]
        if self._arrays is None:
            self._arrays = {"priors":np.asarray([r["probabilities"] for r in self.records]),
                "y":np.asarray([r["outcome"] for r in self.records]),
                "r":np.asarray([RESULT_CATEGORIES.index(r["results_category"]) for r in self.records]),
                "x":np.asarray([XG_CATEGORIES.index(r["xg_category"]) for r in self.records])}
        data = self._arrays
        weights = np.exp(-np.sum((data["priors"]-np.asarray(p0))**2,axis=1)/(2*self.bandwidth**2)) if condition_on_p0 else np.ones(len(self.records))
        q = np.full((3,len(RESULT_CATEGORIES),len(XG_CATEGORIES)),self.dirichlet_mass/(len(RESULT_CATEGORIES)*len(XG_CATEGORIES)))
        np.add.at(q,(data["y"],data["r"],data["x"]),weights)
        mass = q.sum(axis=(1,2))
        q /= mass[:,None,None]
        audit = {"condition_on_p0":bool(condition_on_p0),"kernel_effective_matches":float(weights.sum()**2/np.sum(weights**2)),
            "class_kernel_weight":{str(y):float(weights[data["y"] == y].sum()) for y in range(3)},
            "dirichlet_mass_per_outcome":self.dirichlet_mass}
        self._joint_cache[key] = q,audit
        return q,audit

    def joint_distribution(self,state,*,hypothetical=False,condition_on_p0=True):
        clean,observations = self._visible(state,hypothetical=hypothetical)
        prior = self._free_prior(clean)
        p0 = _probabilities(prior["probabilities"])
        q,audit = self._q(p0,condition_on_p0=condition_on_p0)
        joint = p0[:,None,None]*q
        if "results_view" in observations:
            keep = RESULT_CATEGORIES.index(observations["results_view"])
            joint[:,np.arange(len(RESULT_CATEGORIES)) != keep,:] = 0.
        if "xg_view" in observations:
            keep = XG_CATEGORIES.index(observations["xg_view"])
            joint[:,:,np.arange(len(XG_CATEGORIES)) != keep] = 0.
        evidence_probability = float(joint.sum())
        if evidence_probability <= 0:
            raise ValueError("Impossible observed category under fitted joint model")
        joint /= evidence_probability
        return {"probabilities":joint.tolist(),"prior_probabilities":list(prior["probabilities"]),
            "current_probabilities":joint.sum(axis=(1,2)).tolist() if observations else list(prior["probabilities"]),"observations":observations,
            "results_categories":list(RESULT_CATEGORIES),"xg_categories":list(XG_CATEGORIES),
            "evidence_probability":evidence_probability,"q_support":audit,
            "base_model_version":self.base_predictor.model_version,"model_version":self.model_version}

    def _predict(self,state,*,hypothetical=False):
        clean,observations = self._visible(state,hypothetical=hypothetical)
        joint = self.joint_distribution(state,hypothetical=hypothetical)
        prior = self._free_prior(clean)
        p = joint["current_probabilities"]
        return {**prior,"probabilities":p,"home_win_probability":p[0],"draw_probability":p[1],"away_win_probability":p[2],
            "mask":clean["mask"],"mask_key":clean["mask_key"],"group_statuses":{g:v["status"] for g,v in clean["groups"].items()},
            "visible_lineage":{g:{k:v for k,v in item.items() if k != "values"} for g,item in clean["groups"].items() if item["status"] == "observed"},
            "prior_probabilities":joint["prior_probabilities"],"model_version":self.model_version,"base_model_version":self.base_predictor.model_version,
            "observed_return_categories":observations,"q_support":joint["q_support"],
            "prediction_kind":"hypothetical_finite_return_branch" if hypothetical else "runtime_joint_bayes_posterior",
            "compression_scope":self.metadata["compression"],"fallback_reason":None}

    def predict_partial(self,state):
        return self._predict(state,hypothetical=False)

    def predict_hypothetical(self,state):
        return self._predict(state,hypothetical=True)

    def observation_support(self, state):
        """Past unsmoothed support for the *currently observed* return context.

        Dirichlet pseudocounts make rare categories mathematically possible;
        they do not supply empirical evidence for acquisition decisions.
        These diagnostics are coverage guards, not calibration guarantees.
        """
        distribution = self.joint_distribution(state)
        p0 = np.asarray(distribution["prior_probabilities"])
        observations = distribution["observations"]
        weights = np.asarray([
            math.exp(-sum((p - q) ** 2 for p, q in zip(record["probabilities"], p0))
                     / (2 * self.bandwidth ** 2))
            if all(record["results_category" if aid == "results_view" else "xg_category"] == token
                   for aid, token in observations.items()) else 0.
            for record in self.records
        ])
        mass = float(weights.sum())
        square_mass = float(np.square(weights).sum())
        return {"observed_categories": observations,
                "matching_training_matches": int(np.count_nonzero(weights)),
                "kernel_effective_matches": mass * mass / square_mass if square_mass else 0.,
                "class_kernel_weight": {
                    str(y): float(sum(w for w, r in zip(weights, self.records) if r["outcome"] == y))
                    for y in range(3)},
                "smoothing_counts_as_support": False,
                "scope": "past conditional coverage, not probability calibration"}

    def enumerate_action(self,state,action,*,hypothetical=False):
        aid = action["action_id"] if isinstance(action,dict) else str(action)
        if aid not in ACTION_IDS:
            raise ValueError("Unmodeled tool action")
        distribution = self.joint_distribution(state,hypothetical=hypothetical)
        if aid in distribution["observations"]:
            raise ValueError("Tool already observed, including missing summary")
        joint = np.asarray(distribution["probabilities"])
        vocabulary = RESULT_CATEGORIES if aid == "results_view" else XG_CATEGORIES
        public_branch_base = sanitize_state(state)
        branches = []
        for index,token in enumerate(vocabulary):
            sliced = joint[:,index,:] if aid == "results_view" else joint[:,:,index]
            mass = float(sliced.sum())
            if mass <= 0:
                continue
            posterior = (sliced.sum(axis=1)/mass).tolist()
            # Planning branches contain only the same allow-listed public
            # projection, never opaque unpurchased values or caller extras.
            next_state = deepcopy(public_branch_base)
            next_state[HYPOTHESIS_KEY] = {"schema":VERSION,"hypothetical":True,
                "observations":{**distribution["observations"],aid:token}}
            branches.append({"observation":token,"probability":mass,"posterior_probabilities":posterior,
                "next_state":next_state,"return_status":"summary_missing" if token == "missing" else "summary_observed"})
        current = distribution["current_probabilities"]
        expected_entropy = sum(b["probability"]*entropy(b["posterior_probabilities"]) for b in branches)
        cmi = sum(b["probability"]*sum(p*math.log(p/current[i]) for i,p in enumerate(b["posterior_probabilities"]) if p > 0) for b in branches)
        return {"action_id":aid,"current_probabilities":current,"branches":branches,
            "expected_posterior_entropy":expected_entropy,"evsi":entropy(current)-expected_entropy,"conditional_mutual_information":cmi,
            "scope":"expected log-score improvement under this fitted joint model, not guaranteed real loss reduction"}

    def _payload(self):
        return {"schema":VERSION,"parameters":{"bandwidth":self.bandwidth,"dirichlet_mass":self.dirichlet_mass},
            "base_predictor":self.base_predictor.to_state() if self.base_predictor is not None else None,
            "quantiles":self.quantiles,"records":self.records,"metadata":self.metadata}

    def save(self,path):
        if not self.model_version:
            raise ValueError("Fit before saving joint model")
        path = Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps({**self._payload(),"model_version":self.model_version},ensure_ascii=False,allow_nan=False),encoding="utf-8")

    @classmethod
    def load(cls,path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != VERSION:
            raise ValueError("Unsupported joint return model schema")
        model = cls(ContextualPartialPredictor.from_state(payload["base_predictor"]),**payload["parameters"])
        model.quantiles,model.records,model.metadata = payload["quantiles"],payload["records"],payload["metadata"]
        version = VERSION+":"+sha256(_canonical(model._payload()).encode()).hexdigest()[:20]
        if version != payload.get("model_version"):
            raise ValueError("Joint return artifact hash mismatch")
        model.model_version = model.version = version
        return model
