"""Small-sample, time-ordered calibration for active-information value.

This module is the round-five successor to :mod:`risk`.  The round-four
``RiskAwareValueEstimator`` remains frozen so that its artifacts can still be
replayed.  The calibrated estimator deliberately uses a simple empirical
model: unique-match aggregation, action/context shrinkage, and separate
mean-uncertainty and single-gain predictive intervals.  It is an estimate of forward OOF loss
improvement, not an exact Bayesian VoI or a coverage guarantee.

Runtime inference accepts only a sanitized visible state, current prediction,
and allow-listed ActionSpec metadata.  Labels, returned evidence, full
snapshots, and paths never cross the inference boundary.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .state import parse_time
from .value import COST_UNITS, _action, _canonical, _confidence, _key, _mask
from .risk import _context_key, _quality_bucket, _sd

VERSION = "risk-aware-conditional-value-v2"
RISK_LEVELS = ("mean", "conservative")


def _time(value: Any) -> datetime:
    return parse_time(value)


def _unique_rows(rows: list[dict]) -> dict[str, tuple[float, str]]:
    """Aggregate repeated observations to one mean per match and week."""
    by_match: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_match[str(row["match_id"])].append(row)
    result: dict[str, tuple[float, str]] = {}
    for match_id, values in by_match.items():
        kickoff = _time(values[0]["kickoff"])
        year, week, _ = kickoff.isocalendar()
        result[match_id] = (sum(float(v["delta"]) for v in values) / len(values), f"{year}-{week:02}")
    return result


def _summary(rows: list[dict], parent: list[dict], shrinkage: float,
            interval_multiplier: float, *, interval_floor: float = 0.0) -> dict:
    """Fit a match-weighted shrinkage mean and an honest small-sample interval."""
    local = _unique_rows(rows)
    broad = _unique_rows(parent)
    n, total = len(local), len(broad)
    if total == 0:
        return {
            "expected_gain": None, "variance": None, "uncertainty": None,
            "support": 0, "parent_support": 0, "effective_support": 0.0,
            "lower_gain": None, "upper_gain": None,
            "support_status": "no_support", "uncertainty_kind": "no_support",
        }
    parent_mean = sum(v[0] for v in broad.values()) / total
    local_mean = sum(v[0] for v in local.values()) / n if n else parent_mean
    weight = n / (n + shrinkage) if n else 0.0
    expected = weight * local_mean + (1.0 - weight) * parent_mean
    values = list(local.values()) if n else list(broad.values())
    residuals = [value - expected for value, _ in values]
    residual_sd = _sd(residuals)
    # The parent scale is a conservative fallback for a one-point local cell.
    if residual_sd is None and total >= 2:
        residual_sd = _sd([value - parent_mean for value, _ in broad.values()])
    # Uncertainty of the shrunken MEAN, clustered by week. Parent and local
    # samples overlap; sum their influence contributions before squaring.
    influences = defaultdict(float)
    for value, week in broad.values():
        influences[week] += (1.0 - weight) * (value - parent_mean) / total
    if n:
        for value, week in local.values():
            influences[week] += weight * (value - local_mean) / n
    weeks = set(influences)
    blocks = len(weeks)
    uncertainty = (math.sqrt(blocks / (blocks - 1) * sum(v*v for v in influences.values()))
                   if blocks >= 2 else None)
    variance = uncertainty * uncertainty if uncertainty is not None else None
    radius = None
    if uncertainty is not None:
        radius = max(interval_floor, interval_multiplier * uncertainty)
    return {
        "expected_gain": expected,
        "variance": variance,
        "uncertainty": uncertainty,
        "standard_error": uncertainty,
        "support": n,
        "parent_support": total,
        "effective_support": float(n + (1.0 - weight) * max(0, total - n)),
        "lower_gain": expected - radius if radius is not None else None,
        "upper_gain": expected + radius if radius is not None else None,
        "interval_radius": radius,
        "residual_sd": residual_sd,
        "time_blocks": len(weeks),
        "local_weight": weight,
        "support_status": "supported" if n else "parent_only",
        "uncertainty_kind": "descriptive_week_cluster_mean_se_no_coverage_guarantee"
        if radius is not None else "insufficient_time_blocks",
    }


def _validate_sample(sample: dict, cutoff: datetime, *, role: str) -> dict:
    state = sample["state"]
    match_id = str(sample["match_id"])
    if match_id != str(state.get("match_id")):
        raise ValueError(f"{role}: sample/state match mismatch")
    prediction_at = _time(state["prediction_cutoff"])
    kickoff = _time(sample["kickoff"])
    label_at = _time(sample["result_available_at"])
    train_at = _time(sample["oof_train_cutoff"])
    if not train_at < prediction_at < kickoff <= label_at <= cutoff:
        raise ValueError(f"{role}: forward OOF time order violation")
    if match_id in {str(value) for value in sample.get("oof_train_ids", [])}:
        raise ValueError(f"{role}: target occurs in predictor training IDs")
    delta = float(sample["delta"])
    if not math.isfinite(delta):
        raise ValueError(f"{role}: delta must be finite")
    return {"match_id": match_id, "kickoff": sample["kickoff"], "delta": delta,
            "state": state, "action": sample["action"],
            "prediction_before": sample.get("prediction_before"),
            "result_available_at": sample["result_available_at"],
            "oof_train_cutoff": sample["oof_train_cutoff"],
            "oof_train_ids": list(sample.get("oof_train_ids", []))}


class CalibratedRiskAwareValueEstimator:
    """Round-five estimator with explicit support and interval calibration.

    ``fit`` expects a chronological training part and an optional later
    calibration tail.  The two parts must have disjoint matches and a strict
    kickoff boundary.  At inference a state/action cell first uses an exact
    context, then an action/confidence parent.  Falling back to a parent never
    hides zero local support: ``support_status`` and ``fallback_reason`` remain
    explicit and the policy can reject it.
    """

    def __init__(self, min_support: int = 12, shrinkage: float = 20.0,
                 risk_multiplier: float = 1.64, interval_floor: float = 0.0):
        if (type(min_support) is not int or min_support < 1 or any(
                not math.isfinite(v) or v < 0 for v in (shrinkage, risk_multiplier, interval_floor))):
            raise ValueError("Invalid calibrated risk parameters")
        self.min_support = int(min_support)
        self.shrinkage = float(shrinkage)
        self.risk_multiplier = float(risk_multiplier)
        self.interval_floor = float(interval_floor)
        self.cells: dict[str, dict] = {}
        self.parents: dict[str, dict] = {}
        self.manifest: dict = {}
        self.calibration: dict = {}
        self.version = VERSION

    def fit(self, samples: list[dict], *, as_of: Any, dataset_version: str,
            calibration_samples: list[dict] | None = None,
            calibration_target_coverage: float = .8,
            metadata: dict | None = None) -> "CalibratedRiskAwareValueEstimator":
        cutoff = _time(as_of)
        if not samples:
            raise ValueError("At least one OOF sample is required")
        if not 0 < calibration_target_coverage < 1:
            raise ValueError("calibration_target_coverage must be between zero and one")
        train = [_validate_sample(sample, cutoff, role="train") for sample in samples]
        calibration = [_validate_sample(sample, cutoff, role="calibration")
                       for sample in (calibration_samples or [])]
        train_ids = {row["match_id"] for row in train}
        calibration_ids = {row["match_id"] for row in calibration}
        overlap = train_ids & calibration_ids
        if overlap:
            raise ValueError("training and calibration matches overlap")
        if calibration:
            max_train_kickoff = max(_time(row["kickoff"]) for row in train)
            min_calibration_kickoff = min(_time(row["kickoff"]) for row in calibration)
            if (not max_train_kickoff < min_calibration_kickoff or
                    max(_time(row["result_available_at"]) for row in train) >=
                    min(_time(row["state"]["prediction_cutoff"]) for row in calibration)):
                raise ValueError("calibration tail must start after training labels are available")

        grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
        parents: dict[str, list[dict]] = defaultdict(list)
        modes = set()
        provenance = []
        for row in train:
            state, action = row["state"], _action(row["action"])
            modes.add(state["information_mode"])
            context = _context_key(state, row["action"], row.get("prediction_before"))
            parent = _canonical([_key(action), _confidence(row.get("prediction_before")) or "prediction_unknown"])
            clean = {"match_id": row["match_id"], "delta": row["delta"], "kickoff": row["kickoff"]}
            grouped[(context, parent)].append(clean)
            parents[parent].append(clean)
            provenance.append({"match_id": row["match_id"], "prediction_cutoff": state["prediction_cutoff"],
                               "oof_train_cutoff": row["oof_train_cutoff"], "mask": _mask(state)[0],
                               "action_key": _key(action)})
        modes.update(row["state"]["information_mode"] for row in calibration)
        if len(modes) != 1:
            raise ValueError("Calibrated risk estimator requires one information mode")
        broad_summaries = {key: _summary([], rows, self.shrinkage, self.risk_multiplier,
                                         interval_floor=self.interval_floor)
                           for key, rows in parents.items()}
        self.parents = broad_summaries
        self.cells = {}
        for (context, parent), rows in grouped.items():
            summary = _summary(rows, parents[parent], self.shrinkage, self.risk_multiplier,
                               interval_floor=self.interval_floor)
            summary["parent_key"] = parent
            self.cells[context] = summary

        # Prediction intervals describe a SINGLE future realized gain. They
        # are calibrated on the later tail, separately from uncertainty of the
        # conditional mean. Never inflate mean SE by single-game residuals.
        residuals = {}
        calibration_predictions = []
        for row in calibration:
            context = _context_key(row["state"], row["action"], row.get("prediction_before"))
            action = _action(row["action"])
            parent = _canonical([_key(action), _confidence(row.get("prediction_before")) or "prediction_unknown"])
            summary = self.cells.get(context) or self.parents.get(parent)
            if not summary or summary.get("expected_gain") is None:
                continue
            key = (row["match_id"], context)
            residual = abs(row["delta"] - summary["expected_gain"])
            residuals.setdefault(key, []).append(residual)
            calibration_predictions.append((row, summary))
        # Repeated masks/lead records for one match/context get one weight.
        absolute = sorted(sum(v)/len(v) for v in residuals.values())
        radius = absolute[min(len(absolute)-1, math.ceil(calibration_target_coverage * len(absolute))-1)] if absolute else None
        for summary in [*self.cells.values(), *self.parents.values()]:
            gain = summary.get("expected_gain")
            summary["predictive_lower_gain"] = gain - radius if gain is not None and radius is not None else None
            summary["predictive_upper_gain"] = gain + radius if gain is not None and radius is not None else None
            summary["predictive_radius"] = radius
        self.calibration = {
            "samples": len(calibration), "samples_with_prediction": len(calibration_predictions),
            "unique_match_contexts": len(absolute), "standardized_residuals": len(absolute),
            "target_coverage": calibration_target_coverage, "risk_multiplier": self.risk_multiplier,
            "predictive_radius": radius,
            "predictive_interval_coverage": sum(v <= radius for v in absolute)/len(absolute) if absolute else None,
            "calibration_kind": "split_tail_predictive_absolute_residual_quantile" if absolute else "no_predictive_calibration_support",
            "scope": "inner chronological tail; single-gain predictive coverage only; no strict guarantee",
            "mean_interval_scope": "week-cluster uncertainty of a shrunken cell mean; no conditional-coverage guarantee",
            "mean_interval_calibrated_by_realized_residuals": False,
        }
        self.manifest = {
            "dataset_version": dataset_version, "trained_as_of": cutoff.isoformat(),
            "training_match_ids": sorted(train_ids), "training_matches": len(train_ids),
            "oof_samples": len(train), "calibration_match_ids": sorted(calibration_ids),
            "calibration_matches": len(calibration_ids), "calibration_samples": len(calibration),
            "calibration_target_coverage": calibration_target_coverage,
            "risk_multiplier": self.risk_multiplier, "information_mode": next(iter(modes)),
            "runtime_features": ["visible_mask", "action_metadata", "quality_status", "freshness_age",
                                 "dependency_key", "cache_policy", "prediction_confidence"],
            "target": "forward_oof_log_loss_before_minus_after", "support_unit": "unique_match",
            "cost_units": COST_UNITS,
            "uncertainty_note": "Mean interval uses cluster SE; predictive interval uses held-tail residuals. Neither has a strict coverage guarantee.",
            "temporal_boundary": {"train_matches": len(train_ids), "calibration_matches": len(calibration_ids),
                                  "train_max_kickoff": max(_time(row["kickoff"]) for row in train).isoformat(),
                                  "calibration_min_kickoff": min(_time(row["kickoff"]) for row in calibration).isoformat()
                                  if calibration else None},
            "metadata": metadata or {}, "oof_provenance": provenance,
        }
        self.version = VERSION + ":" + hashlib.sha256(_canonical(self._payload()).encode()).hexdigest()[:20]
        return self

    def estimate_value(self, visible_state: dict, candidates: list[dict], *, prediction: Any = None,
                       risk_level: str = "mean") -> list[dict]:
        if risk_level not in RISK_LEVELS:
            raise ValueError("risk_level must be mean or conservative")
        if not self.manifest:
            raise ValueError("CalibratedRiskAwareValueEstimator must be fitted")
        if visible_state["information_mode"] != self.manifest["information_mode"]:
            raise ValueError("Risk estimator information mode mismatch")
        # A model fitted at cutoff T can serve only predictions observed after
        # T; querying an earlier state would leak later OOF supervision.
        if _time(visible_state["prediction_cutoff"]) < _time(self.manifest["trained_as_of"]):
            raise ValueError("Risk estimator was fitted after requested prediction cutoff")
        output = []
        confidence = _confidence(prediction)
        for raw in candidates:
            action = _action(raw)
            context = _context_key(visible_state, raw, prediction)
            summary = self.cells.get(context)
            fallback = None
            if summary is None:
                parent_key = _canonical([_key(action), confidence or "prediction_unknown"])
                summary = self.parents.get(parent_key)
                fallback = "state_context_fallback_to_action_parent" if summary else "unsupported_action"
            if summary is None:
                summary = {"expected_gain": None, "uncertainty": None, "variance": None,
                           "lower_gain": None, "upper_gain": None, "support": 0,
                           "parent_support": 0, "effective_support": 0.0,
                           "support_status": "no_support", "uncertainty_kind": "no_support"}
            support = int(summary.get("support", 0) or 0)
            parent_support = int(summary.get("parent_support", 0) or 0)
            if support == 0 and fallback is None:
                fallback = "no_local_support"
            elif 0 < support < self.min_support:
                fallback = "insufficient_support"
            expected = summary.get("expected_gain")
            uncertainty = summary.get("uncertainty")
            lower = summary.get("lower_gain")
            decision_gain = lower if risk_level == "conservative" else expected
            if expected is None:
                gate_reason = "stop_no_support"
            elif support < self.min_support:
                gate_reason = "stop_insufficient_support"
            elif risk_level == "conservative" and lower is None:
                gate_reason = "stop_uncertainty_unavailable"
            elif risk_level == "conservative" and lower <= 0:
                gate_reason = "stop_lower_bound_nonpositive"
            elif risk_level == "mean" and expected <= 0:
                gate_reason = "stop_mean_gain_nonpositive"
            else:
                gate_reason = "eligible"
            output.append({
                "action_id": action["action_id"], "action": action,
                "expected_gain": expected, "gain": expected, "variance": summary.get("variance"),
                "uncertainty": uncertainty, "standard_error": uncertainty,
                "lower_gain": lower, "upper_gain": summary.get("upper_gain"),
                "decision_gain": decision_gain, "interval_radius": summary.get("interval_radius"),
                "predictive_lower_gain": summary.get("predictive_lower_gain"),
                "predictive_upper_gain": summary.get("predictive_upper_gain"),
                "uncertainty_kind": summary.get("uncertainty_kind"),
                "mean_interval_scope": "conditional_mean_cluster_se_no_coverage_guarantee",
                "support": support, "support_matches": support, "parent_support": parent_support,
                "effective_support": summary.get("effective_support", float(support)),
                "support_status": summary.get("support_status", "no_support"),
                "cost_vector": action["cost"], "cost": action["cost"],
                "cost_status": action["cost_status"], "risk_level": risk_level,
                "estimator_version": self.version, "version": self.version,
                "fallback_reason": fallback, "reason": fallback or "estimated",
                "gate_reason": gate_reason, "context_key": context,
                "quality_context": _quality_bucket(visible_state, raw),
                "calibration_status": self.calibration.get("calibration_kind"),
                "net_gain": None,
                "net_gain_note": "Computed by policy with frozen unit lambda.",
            })
        return output

    def _payload(self) -> dict:
        return {"schema": VERSION,
                "parameters": {"min_support": self.min_support, "shrinkage": self.shrinkage,
                               "risk_multiplier": self.risk_multiplier, "interval_floor": self.interval_floor},
                "cells": self.cells, "parents": self.parents,
                "calibration": self.calibration, "manifest": self.manifest}

    def save(self, path: str | Path) -> None:
        payload = self._payload()
        payload["version"] = self.version
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CalibratedRiskAwareValueEstimator":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != VERSION:
            raise ValueError("Unsupported calibrated risk estimator schema")
        model = cls(**payload["parameters"])
        model.cells, model.parents = payload["cells"], payload["parents"]
        model.calibration, model.manifest = payload["calibration"], payload["manifest"]
        expected = VERSION + ":" + hashlib.sha256(_canonical(model._payload()).encode()).hexdigest()[:20]
        if payload.get("version") != expected:
            raise ValueError("Calibrated risk estimator artifact hash mismatch")
        model.version = payload["version"]
        return model
