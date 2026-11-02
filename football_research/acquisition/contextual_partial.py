"""Partial outcome models with identical free fixture context in every mask.

All eight models are regularized multinomial logistic regressions. Dictionaries,
imputation and standardization use training rows only. Exact source identity is
offline reconstructed metadata; no outcome or unpurchased value enters runtime.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from football_research.baselines.models import _training, _label
from .contextual_identity import validate_crosswalk
from .state import GROUP_FIELDS, GROUP_IDS, parse_time, sanitize_state
from .value import _canonical

VERSION = "contextual-partial-v1"


def _features(state):
    kickoff = parse_time(state["base"]["kickoff"])
    day = kickoff.timetuple().tm_yday
    numeric = {"day_sin": math.sin(2*math.pi*day/365.25), "day_cos": math.cos(2*math.pi*day/365.25),
        "weekday_sin": math.sin(2*math.pi*kickoff.weekday()/7), "weekday_cos": math.cos(2*math.pi*kickoff.weekday()/7),
        "lead_hours": (kickoff-parse_time(state["prediction_cutoff"])).total_seconds()/3600}
    for group, fields in GROUP_FIELDS.items():
        if state["groups"][group]["status"] == "observed":
            numeric.update({field: state["groups"][group]["values"].get(field) for field in fields})
    categorical = {key: state["base"][key] for key in ("home_team_id", "away_team_id", "competition_id")}
    return numeric, categorical


def _encode(state, head):
    numeric, categories = _features(state)
    values, unknown = [1.0], []
    for name in head["numeric_names"]:
        value = numeric.get(name)
        mean, scale = head["numeric_fit"][name]
        values.extend((0.0 if value is None else (value-mean)/scale, float(value is None)))
    for name, vocabulary in head["categorical_fit"].items():
        current = categories[name]
        values.extend(float(current == item) for item in vocabulary)
        if current not in vocabulary:
            unknown.append(name)
    return np.asarray(values), unknown


class ContextualPartialPredictor:
    version = VERSION

    def __init__(self, crosswalk=None):
        if crosswalk is not None:
            validate_crosswalk(crosswalk)
        self.crosswalk = json.loads(json.dumps(crosswalk)) if crosswalk is not None else None
        self.metadata, self.heads, self._arrays = {}, {}, {}
        self.model_version = self.prior_version = ""

    def _project(self, visible_state):
        state = sanitize_state(visible_state)
        mapped, unmapped = [], []
        for field, kind in (("home_team_id", "team"), ("away_team_id", "team"), ("competition_id", "competition")):
            original = state["base"][field]
            token = self.crosswalk["aliases"][kind].get(original)
            if token is None:
                unmapped.append(field)
            else:
                state["base"][field] = token
                mapped.append(field)
        return state, {"mapped_fields": mapped, "unmapped_fields": unmapped, "crosswalk_version": self.crosswalk["version"]}

    def fit(self, rows, as_of, dataset_version, regularization=10.0, min_train_matches=15, crosswalk=None):
        if crosswalk is not None:
            validate_crosswalk(crosswalk)
            self.crosswalk = json.loads(json.dumps(crosswalk))
        if self.crosswalk is None:
            raise ValueError("Explicit exact source identity crosswalk required")
        if not math.isfinite(regularization) or regularization <= 0 or type(min_train_matches) is not int or min_train_matches < 1:
            raise ValueError("Invalid contextual outcome model configuration")
        if not dataset_version:
            raise ValueError("Dataset version required")
        selected = _training(rows, as_of)
        modes = {r["information_mode"] for r in selected}
        if len(modes) != 1 or not modes <= {"historical_reconstruction", "operational"}:
            raise ValueError("One valid information mode required")
        for row in selected:
            cutoff = parse_time(row["prediction_cutoff"])
            if cutoff >= parse_time(row["kickoff"]):
                raise ValueError("Outcome training features must be prematch")
            for field in ("features_available_at", "fixture_available_at"):
                if row.get(field) is not None and parse_time(row[field]) > cutoff:
                    raise ValueError("Training information unavailable at cutoff")
        labels = np.asarray([_label(row) for row in selected], dtype=int)
        counts = np.bincount(labels, minlength=3)+1.
        fallback_probabilities = (counts/counts.sum()).tolist()
        self.heads, self._arrays = {}, {}
        self.metadata = {"dataset_version": dataset_version, "training_as_of": parse_time(as_of).isoformat(),
            "information_mode": next(iter(modes)), "training_rows": len(selected), "regularization": float(regularization),
            "min_train_matches": min_train_matches, "training_match_ids": sorted(str(r["match_id"]) for r in selected),
            "training": [{"match_id": str(r["match_id"]), "prediction_cutoff": parse_time(r["prediction_cutoff"]).isoformat(),
                          "result_available_at": parse_time(r["result_available_at"]).isoformat()} for r in selected],
            "group_fields": {k: list(v) for k, v in GROUP_FIELDS.items()}, "subsets": {},
            "free_context": ["exact_source_team_ids", "exact_source_competition_id", "kickoff_day_sin_cos", "kickoff_weekday_sin_cos", "lead_hours"],
            "free_context_policy": "identical fixture identity/calendar availability for every acquisition mask and strategy",
            "identity_crosswalk_version": self.crosswalk["version"], "identity_time_interpretation": self.crosswalk["time_interpretation"],
            "unknown_identity": "zero_one_hot_effect_reported; no out_of_domain_probability_calibration_claim",
            "probability_calibration": "none_posthoc; regularized_multinomial_probability_not_calibration_guarantee"}
        for mask in range(8):
            groups = [g for i, g in enumerate(GROUP_IDS) if mask & (1 << i)]
            states = []
            for row in selected:
                public = {"match_id": str(row["match_id"]), "information_mode": row["information_mode"],
                    "prediction_cutoff": row["prediction_cutoff"], "base": {key: row[key] for key in ("competition_id", "home_team_id", "away_team_id", "kickoff")},
                    "groups": {g: {"status": "observed", "available_at": row["prediction_cutoff"],
                                "values": {field: row.get(field) for field in GROUP_FIELDS[g]}, "evidence_refs": []}
                               if g in groups else {"status": "not_acquired", "values": {}} for g in GROUP_IDS}}
                states.append(self._project(public)[0])
            features = [_features(s) for s in states]
            numeric_names = sorted(features[0][0])
            numeric_fit = {}
            for name in numeric_names:
                values = [n[name] for n, _ in features if n[name] is not None]
                mean, scale = (float(np.mean(values)), float(np.std(values))) if values else (0., 1.)
                numeric_fit[name] = [mean, scale if scale > 1e-10 else 1.]
            head = {"groups": groups, "numeric_names": numeric_names, "numeric_fit": numeric_fit,
                "categorical_fit": {name: sorted({c[name] for _, c in features}) for name in sorted(features[0][1])}}
            x = np.vstack([_encode(s, head)[0] for s in states])
            reason = "insufficient_training_matches" if len(selected) < min_train_matches else "training_class_support" if set(labels) != {0,1,2} else None
            head.update(fallback_reason=reason, fallback_probabilities=fallback_probabilities, training_rows=len(selected), design_columns=x.shape[1])
            if reason:
                head.update(coefficients=None, optimizer=None)
            else:
                y = np.eye(3)[labels]
                initial = np.zeros((x.shape[1], 3))
                initial[0] = np.log(counts/counts.sum())
                def objective(flat):
                    theta = flat.reshape(x.shape[1], 3)
                    logits = x@theta
                    loss = np.sum(logsumexp(logits, axis=1)-logits[np.arange(len(labels)), labels])
                    penalized = theta.copy()
                    penalized[0] = 0.
                    loss += .5*regularization*np.sum(penalized**2)
                    gradient = x.T@(softmax(logits, axis=1)-y)+regularization*penalized
                    return float(loss), gradient.ravel()
                result = minimize(objective, initial.ravel(), method="L-BFGS-B", jac=True,
                    options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-6})
                if not result.success or not np.isfinite(result.x).all():
                    raise ValueError(f"Contextual multinomial fitting failed for mask {mask}: {result.message}")
                head.update(coefficients=result.x.reshape(x.shape[1],3).tolist(),
                    optimizer={"method": "scipy-L-BFGS-B", "success": bool(result.success), "iterations": int(result.nit),
                               "objective": float(result.fun), "message": str(result.message)})
            self.heads[str(mask)] = head
            self.metadata["subsets"][str(mask)] = {"groups": groups, "features": numeric_names,
                "model_kind": "training_frequency_fallback" if reason else "free_context_subset_multinomial_logistic",
                "fallback_reason": reason, "training_rows": len(selected)}
        self.model_version = VERSION+":"+sha256(_canonical(self._payload()).encode()).hexdigest()[:20]
        self.prior_version = "free-context-empty-mask:"+sha256(_canonical(self.heads["0"]).encode()).hexdigest()[:20]
        return self

    def predict_partial(self, visible_state):
        if not self.metadata:
            raise ValueError("Fit contextual partial model before prediction")
        state, projection = self._project(visible_state)
        if state["information_mode"] != self.metadata["information_mode"]:
            raise ValueError("Model and visible information modes differ")
        if parse_time(state["prediction_cutoff"]) < parse_time(self.metadata["training_as_of"]):
            raise ValueError("Prediction precedes model training cutoff")
        mask = str(state["mask_key"])
        head = self.heads[mask]
        vector, unknown = _encode(state, head)
        if head["coefficients"] is None:
            probabilities = head["fallback_probabilities"]
        else:
            if mask not in self._arrays:
                self._arrays[mask] = np.asarray(head["coefficients"])
            probabilities = softmax(vector@self._arrays[mask]).tolist()
        return {"match_id": state["match_id"], "prediction_cutoff": state["prediction_cutoff"],
            "information_mode": state["information_mode"], "home_win_probability": probabilities[0],
            "draw_probability": probabilities[1], "away_win_probability": probabilities[2], "probabilities": probabilities,
            "mask": state["mask"], "mask_key": state["mask_key"], "model_version": self.model_version,
            "prior_version": self.prior_version, "dataset_version": self.metadata["dataset_version"],
            "fallback_reason": head["fallback_reason"], "free_context": self.metadata["free_context"],
            "unknown_identity_fields": unknown, "identity_projection": projection,
            "group_statuses": {g: v["status"] for g, v in state["groups"].items()},
            "visible_lineage": {g: {k: v for k, v in group.items() if k != "values"} for g, group in state["groups"].items() if group["status"] == "observed"}}

    def _payload(self):
        return {"schema": VERSION, "crosswalk": self.crosswalk, "metadata": self.metadata, "heads": self.heads}

    def to_state(self):
        if not self.model_version:
            raise ValueError("Fit before serialization")
        return json.loads(json.dumps({**self._payload(), "model_version": self.model_version, "prior_version": self.prior_version}, allow_nan=False))

    @classmethod
    def from_state(cls, payload):
        if payload.get("schema") != VERSION:
            raise ValueError("Unsupported contextual partial schema")
        model = cls(payload["crosswalk"])
        model.metadata, model.heads = payload["metadata"], payload["heads"]
        if set(model.heads) != {str(i) for i in range(8)} or model.metadata["group_fields"] != {g: list(v) for g,v in GROUP_FIELDS.items()}:
            raise ValueError("Contextual partial subset schema mismatch")
        version = VERSION+":"+sha256(_canonical(model._payload()).encode()).hexdigest()[:20]
        prior = "free-context-empty-mask:"+sha256(_canonical(model.heads["0"]).encode()).hexdigest()[:20]
        if payload.get("model_version") != version or payload.get("prior_version") != prior:
            raise ValueError("Contextual partial artifact hash mismatch")
        model.model_version, model.prior_version = version, prior
        return model

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_state(), ensure_ascii=False, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls.from_state(json.loads(Path(path).read_text(encoding="utf-8")))
