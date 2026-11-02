"""Auditable ridge estimates of forward-OOF conditional acquisition gain.

Three prespecified ablations share labels, action/mask cells and regularization.
Only sanitized public state crosses inference. Team identity is a historical
association, not a causal estimate of motivation, rotation or tactical intent.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .risk_calibrated import _validate_sample
from .state import GROUP_FIELDS, parse_time, sanitize_state
from .value import _action, _canonical, _key

VERSION = "contextual-oof-ridge-value-v1"
MODES = ("no-context", "prediction-only", "visible-context")


def _probabilities(prediction):
    values = prediction.get("probabilities") if isinstance(prediction, dict) else prediction
    if values is None or len(values) != 3:
        raise ValueError("Three current probabilities are required")
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or abs(values.sum()-1) > 1e-7:
        raise ValueError("Invalid current probability distribution")
    return values.tolist()


def _cell(state, action):
    # Unqueried and queried-but-source-missing share numeric mask zero, but
    # denote different public acquisition states and must remain separate.
    return _canonical([state["mask_key"], [state["groups"][g]["status"] for g in GROUP_FIELDS], _key(action)])


def _features(state, prediction, mode):
    """Unknown fields, unavailable values and external references never enter."""
    numeric, categorical = {}, {}
    if mode != "no-context":
        probabilities = _probabilities(prediction)
        numeric.update({f"p_{i}": p for i, p in enumerate(probabilities)})
        numeric["entropy"] = -sum(p*math.log(max(p, 1e-15)) for p in probabilities)
    if mode == "visible-context":
        base = state["base"]
        categorical = {key: base[key] for key in ("home_team_id", "away_team_id", "competition_id")}
        kickoff = parse_time(base["kickoff"])
        day = kickoff.timetuple().tm_yday
        numeric.update(day_sin=math.sin(2*math.pi*day/365.25), day_cos=math.cos(2*math.pi*day/365.25),
                       weekday_sin=math.sin(2*math.pi*kickoff.weekday()/7),
                       weekday_cos=math.cos(2*math.pi*kickoff.weekday()/7),
                       lead_hours=(kickoff-parse_time(state["prediction_cutoff"])).total_seconds()/3600)
        for group, fields in GROUP_FIELDS.items():
            # A cell already fixes acquisition mask. Null imputation is fitted
            # only for genuinely observed groups, never for unpurchased data.
            if state["groups"][group]["status"] == "observed":
                for field in fields:
                    numeric[field] = state["groups"][group]["values"].get(field)
    return numeric, categorical


class ContextualValueEstimator:
    def __init__(self, mode="visible-context", regularization=20.0, min_support=12, risk_multiplier=1.64):
        if mode not in MODES or not math.isfinite(regularization) or regularization <= 0:
            raise ValueError("Invalid contextual model configuration")
        if type(min_support) is not int or min_support < 1 or not math.isfinite(risk_multiplier) or risk_multiplier < 0:
            raise ValueError("Invalid support or risk multiplier")
        self.mode, self.regularization = mode, float(regularization)
        self.min_support, self.risk_multiplier = min_support, float(risk_multiplier)
        self.cells, self.manifest, self.calibration = {}, {}, {}
        self.version = VERSION
        self._arrays = {}

    def _encode(self, state, prediction, cell):
        numeric, categories = _features(state, prediction, self.mode)
        values = [1.0]
        for name in cell["numeric_names"]:
            value = numeric.get(name)
            mean, scale = cell["numeric_fit"][name]
            values.extend((0.0 if value is None else (value-mean)/scale, float(value is None)))
        unknown = []
        for name, vocabulary in cell["categorical_fit"].items():
            current = categories.get(name)
            values.extend(float(current == entry) for entry in vocabulary)
            if current not in vocabulary:
                unknown.append(name)
        return np.asarray(values), unknown

    def fit(self, samples, *, as_of, dataset_version, calibration_samples=None,
            calibration_target_coverage=.8, metadata=None):
        cutoff = parse_time(as_of)
        if not samples or not 0 < calibration_target_coverage < 1:
            raise ValueError("Nonempty samples and valid coverage required")
        train = [_validate_sample(row, cutoff, role="train") for row in samples]
        calibration = [_validate_sample(row, cutoff, role="calibration") for row in (calibration_samples or [])]
        train_ids, cal_ids = {r["match_id"] for r in train}, {r["match_id"] for r in calibration}
        if train_ids & cal_ids:
            raise ValueError("Training and calibration matches overlap")
        if calibration and (max(parse_time(r["kickoff"]) for r in train) >= min(parse_time(r["kickoff"]) for r in calibration)
                or max(parse_time(r["result_available_at"]) for r in train) >= min(parse_time(r["state"]["prediction_cutoff"]) for r in calibration)):
            raise ValueError("Calibration tail must follow available training labels")
        modes = {r["state"]["information_mode"] for r in train+calibration}
        if len(modes) != 1:
            raise ValueError("One information mode required")
        grouped = defaultdict(list)
        for row in train:
            row["state"] = sanitize_state(row["state"])
            if parse_time(row["state"]["base"]["kickoff"]) != parse_time(row["kickoff"]):
                raise ValueError("Sample and public kickoff mismatch")
            grouped[_cell(row["state"], _action(row["action"]))].append(row)
        self.cells, self._arrays = {}, {}
        for key, rows in grouped.items():
            # Repeated target-match transitions must not fabricate support.
            if len({r["match_id"] for r in rows}) != len(rows):
                raise ValueError("Duplicate match within action/mask training cell")
            features = [_features(r["state"], r["prediction_before"], self.mode) for r in rows]
            numeric_names = sorted({name for numeric, _ in features for name in numeric})
            numeric_fit = {}
            for name in numeric_names:
                values = [num[name] for num, _ in features if num.get(name) is not None]
                mean = float(np.mean(values)) if values else 0.0
                scale = float(np.std(values)) if values else 1.0
                numeric_fit[name] = [mean, scale if scale > 1e-10 else 1.0]
            categorical_fit = {name: sorted({cat[name] for _, cat in features})
                               for name in sorted(features[0][1])}
            cell = {"numeric_names": numeric_names, "numeric_fit": numeric_fit, "categorical_fit": categorical_fit,
                    "support": len(rows), "training_match_ids": sorted(r["match_id"] for r in rows)}
            x = np.vstack([self._encode(r["state"], r["prediction_before"], cell)[0] for r in rows])
            y = np.asarray([r["delta"] for r in rows])
            penalty = np.eye(x.shape[1])*self.regularization
            penalty[0, 0] = 0.0
            bread = np.linalg.inv(x.T@x+penalty)
            beta = bread@x.T@y
            residual = y-x@beta
            # Week clusters include simultaneous matches and all team dependence
            # within week. Longer serial dependence and ridge bias remain.
            scores = defaultdict(lambda: np.zeros(x.shape[1]))
            for i, row in enumerate(rows):
                year, week, _ = parse_time(row["kickoff"]).isocalendar()
                scores[f"{year}-{week:02}"] += x[i]*residual[i]
            g = len(scores)
            covariance = None
            if g >= 2:
                meat = sum(np.outer(v, v) for v in scores.values())*g/(g-1)
                covariance = bread@meat@bread
            cell.update(coefficients=beta.tolist(), covariance=covariance.tolist() if covariance is not None else None,
                        time_blocks=g, design_columns=x.shape[1], effective_degrees_of_freedom=float(np.trace(bread@x.T@x)))
            self.cells[key] = cell
        self.manifest = {"dataset_version": dataset_version, "trained_as_of": cutoff.isoformat(),
            "information_mode": next(iter(modes)), "mode": self.mode, "training_matches": len(train_ids),
            "training_match_ids": sorted(train_ids), "calibration_matches": len(cal_ids), "calibration_match_ids": sorted(cal_ids),
            "oof_samples": len(train), "target": "forward_oof_log_loss_before_minus_after", "support_unit": "unique_match_per_action_mask",
            "feature_boundary": "train-only vocabulary/scaling; fixture identity/calendar; current p; acquired numeric values only",
            "forbidden_features": ["result", "match_id", "unacquired_values", "unverified_motivation", "lineup_or_tactics_not_in_export"],
            "temporal_boundary": {"train_max_result_available_at": max(parse_time(r["result_available_at"]) for r in train).isoformat(),
                                  "calibration_min_prediction_cutoff": min((parse_time(r["state"]["prediction_cutoff"]) for r in calibration), default=None).isoformat() if calibration else None},
            "mean_interval_scope": "descriptive regularized linear mean week-cluster SE; ridge bias and model misspecification excluded; no coverage guarantee",
            "metadata": metadata or {}}
        residuals = []
        for row in calibration:
            state = sanitize_state(row["state"])
            key = _cell(state, _action(row["action"]))
            cell = self.cells.get(key)
            if cell is not None:
                vector, _ = self._encode(state, row["prediction_before"], cell)
                residuals.append(abs(row["delta"]-float(vector@np.asarray(cell["coefficients"]))))
        residuals.sort()
        radius = residuals[min(len(residuals)-1, math.ceil(calibration_target_coverage*len(residuals))-1)] if residuals else None
        self.calibration = {"calibration_kind": "later_tail_absolute_residual_quantile", "predictive_radius": radius,
            "target_coverage": calibration_target_coverage, "transitions": len(residuals), "matches": len(cal_ids),
            "coverage_scope": "marginal descriptive transition coverage; matches repeated across cells; no conditional/sequence guarantee",
            "mean_interval_calibrated_by_realized_residuals": False}
        self.version = VERSION+":"+hashlib.sha256(_canonical(self._payload()).encode()).hexdigest()[:20]
        return self

    def estimate_value(self, visible_state, candidates, prediction=None, risk_level="mean"):
        if risk_level not in ("mean", "conservative") or not self.manifest:
            raise ValueError("Fitted model and valid risk level required")
        state = sanitize_state(visible_state)
        if parse_time(state["prediction_cutoff"]) < parse_time(self.manifest["trained_as_of"]):
            raise ValueError("Inference precedes training as-of")
        if state["information_mode"] != self.manifest["information_mode"]:
            raise ValueError("Information mode mismatch")
        output = []
        for raw in candidates:
            action = _action(raw)
            key = _cell(state, action)
            cell = self.cells.get(key)
            gain = uncertainty = lower = upper = None
            unknown, support = [], 0
            if cell:
                vector, unknown = self._encode(state, prediction, cell)
                if key not in self._arrays:
                    self._arrays[key] = (np.asarray(cell["coefficients"]), np.asarray(cell["covariance"]) if cell["covariance"] is not None else None)
                beta, covariance = self._arrays[key]
                gain = float(vector@beta)
                if covariance is not None:
                    uncertainty = math.sqrt(max(0.0, float(vector@covariance@vector)))
                    lower, upper = gain-self.risk_multiplier*uncertainty, gain+self.risk_multiplier*uncertainty
                support = cell["support"]
            reason = "unsupported_state_action" if not cell else "unknown_identity_zero_effect_fallback" if unknown else "estimated"
            gate = ("stop_no_support" if not cell else "stop_insufficient_support" if support < self.min_support else
                    "stop_uncertainty_unavailable" if risk_level == "conservative" and lower is None else
                    "stop_lower_bound_nonpositive" if risk_level == "conservative" and lower <= 0 else
                    "stop_mean_gain_nonpositive" if risk_level == "mean" and gain <= 0 else "eligible")
            radius = self.calibration.get("predictive_radius")
            output.append({"action_id": action["action_id"], "action": action, "gain": gain, "expected_gain": gain,
                "uncertainty": uncertainty, "standard_error": uncertainty, "variance": uncertainty**2 if uncertainty is not None else None,
                "lower_gain": lower, "upper_gain": upper, "decision_gain": lower if risk_level == "conservative" else gain,
                "predictive_lower_gain": gain-radius if gain is not None and radius is not None else None,
                "predictive_upper_gain": gain+radius if gain is not None and radius is not None else None,
                "support": support, "support_matches": support, "parent_support": support, "effective_support": float(support),
                "support_status": "supported" if support >= self.min_support else "low_support" if support else "no_support",
                "support_scope": "training action/mask cell, not exact fixture or identity interaction",
                "unknown_identity_fields": unknown, "identity_scope": "zero effect for unseen identity; no calibrated out-of-domain guarantee",
                "cost": action["cost"], "cost_vector": action["cost"], "cost_status": action["cost_status"],
                "risk_level": risk_level, "reason": reason, "fallback_reason": reason if reason != "estimated" else None,
                "gate_reason": gate, "version": self.version, "estimator_version": self.version,
                "context_key": key, "mode": self.mode, "quality_context": {"scope": "not a learned feature in these ablations"},
                "uncertainty_kind": "descriptive_ridge_week_cluster_mean_se", "mean_interval_scope": self.manifest["mean_interval_scope"],
                "calibration_status": self.calibration["calibration_kind"], "net_gain": None})
        return output

    def _payload(self):
        return {"schema": VERSION, "parameters": {"mode": self.mode, "regularization": self.regularization,
                "min_support": self.min_support, "risk_multiplier": self.risk_multiplier},
                "cells": self.cells, "manifest": self.manifest, "calibration": self.calibration}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**self._payload(), "version": self.version}, ensure_ascii=False, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != VERSION:
            raise ValueError("Unsupported contextual artifact")
        model = cls(**payload["parameters"])
        model.cells, model.manifest, model.calibration = payload["cells"], payload["manifest"], payload["calibration"]
        version = VERSION+":"+hashlib.sha256(_canonical(model._payload()).encode()).hexdigest()[:20]
        if version != payload.get("version"):
            raise ValueError("Contextual artifact hash mismatch")
        model.version = version
        return model
