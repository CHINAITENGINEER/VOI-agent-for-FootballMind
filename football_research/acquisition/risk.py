"""Risk-aware conditional information value from forward OOF supervision.

The runtime side only sees a sanitized visible state and action metadata.  OOF
loss differences, labels, and returned evidence are retained in the fitted
artifact and never accepted by :meth:`estimate_value`.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .state import parse_time
from .value import COST_UNITS, _action, _canonical, _confidence, _key, _mask, ValueEstimator

VERSION = "risk-aware-conditional-value-v1"
RISK_LEVELS = ("mean", "conservative")


def _quality_bucket(state: dict, action: dict) -> str:
    """Derive a quality/freshness/dependency context without inventing reliability."""
    visible = state.get("groups", {})
    statuses = [visible.get(group, {}).get("status", "not_acquired") for group in action["group_ids"]]
    if any(status in {"source_missing", "stale"} for status in statuses):
        quality = "degraded"
    elif any(status == "failed" for status in statuses):
        quality = "failed"
    elif any(status == "observed" for status in statuses):
        quality = "observed_dependency_overlap"
    else:
        quality = "unknown_quality"
    ages = []
    cutoff = parse_time(state["prediction_cutoff"])
    for group in action["group_ids"]:
        item = visible.get(group, {})
        if item.get("status") == "observed" and item.get("available_at"):
            ages.append(max(0.0, (cutoff - parse_time(item["available_at"])) .total_seconds() / 3600.0))
    freshness = "age_unknown" if not ages else ("fresh_lt_24h" if max(ages) < 24 else "fresh_24_168h" if max(ages) < 168 else "stale_age")
    dependency = str(action.get("dependency_key", action.get("dependency", "unknown_dependency")))
    cache = str(action.get("cache_policy", "unknown_cache"))
    return f"quality={quality}|freshness={freshness}|dependency={dependency}|cache={cache}"


def _context_key(state: dict, action: dict, prediction: Any = None) -> str:
    mask, _ = _mask(state)
    confidence = _confidence(prediction) or "prediction_unknown"
    return _canonical([mask, _key(action), confidence, _quality_bucket(state, action)])


def _match_means(rows: list[dict]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["match_id"])].append(float(row["delta"]))
    return {match: sum(values) / len(values) for match, values in grouped.items()}


def _sd(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _summary(rows: list[dict], parent: list[dict], shrinkage: float, risk_multiplier: float) -> dict:
    local = _match_means(rows)
    broad = _match_means(parent)
    n, total = len(local), len(broad)
    if total == 0:
        return {"expected_gain": None, "uncertainty": None, "support": 0, "parent_support": 0,
                "lower_gain": None, "uncertainty_kind": "no_support"}
    local_mean = sum(local.values()) / n if n else sum(broad.values()) / total
    parent_mean = sum(broad.values()) / total
    weight = n / (n + shrinkage) if n else 0.0
    expected = weight * local_mean + (1.0 - weight) * parent_mean
    residuals = [value - expected for value in (list(local.values()) if n else list(broad.values()))]
    residual_sd = _sd(residuals)
    # This is a descriptive standard error over unique matches.  We retain the
    # residual scale so the conservative interval does not collapse in a small cell.
    uncertainty = None if residual_sd is None else residual_sd / math.sqrt(max(1, n))
    if uncertainty is None and n == 1 and total >= 2:
        parent_residuals = [value - parent_mean for value in broad.values()]
        parent_sd = _sd(parent_residuals)
        uncertainty = parent_sd / math.sqrt(total) if parent_sd is not None else None
    lower = expected - risk_multiplier * uncertainty if uncertainty is not None else None
    return {"expected_gain": expected, "uncertainty": uncertainty, "support": n,
            "parent_support": total, "lower_gain": lower, "local_weight": weight,
            "residual_sd": residual_sd, "uncertainty_kind": "oof_unique_match_residual_se_no_coverage_guarantee"}


class RiskAwareValueEstimator:
    """Conditional mean plus empirical residual uncertainty and shrinkage."""

    def __init__(self, min_support: int = 12, shrinkage: float = 20.0,
                 risk_multiplier: float = 1.64):
        if min_support < 1 or shrinkage < 0 or risk_multiplier < 0:
            raise ValueError("Invalid risk estimator parameters")
        self.min_support = int(min_support)
        self.shrinkage = float(shrinkage)
        self.risk_multiplier = float(risk_multiplier)
        self.cells: dict[str, dict] = {}
        self.parents: dict[str, dict] = {}
        self.manifest: dict = {}
        self.version = VERSION

    def fit(self, samples: list[dict], *, as_of: Any, dataset_version: str,
            calibration_samples: list[dict] | None = None, calibration_target_coverage: float = .8,
            metadata: dict | None = None) -> "RiskAwareValueEstimator":
        cutoff = parse_time(as_of)
        if not samples:
            raise ValueError("At least one OOF sample is required")
        if not 0 < calibration_target_coverage < 1:
            raise ValueError("calibration_target_coverage must be between zero and one")
        grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
        parents: dict[str, list[dict]] = defaultdict(list)
        ids = set()
        modes = set()
        for sample in samples:
            state = sample["state"]
            if sample["match_id"] != state["match_id"]:
                raise ValueError("Sample/state match mismatch")
            prediction_at = parse_time(state["prediction_cutoff"])
            kickoff = parse_time(sample["kickoff"])
            label_at = parse_time(sample["result_available_at"])
            train_at = parse_time(sample["oof_train_cutoff"])
            if not train_at < prediction_at < kickoff <= label_at <= cutoff:
                raise ValueError("OOF time order violation")
            if sample["match_id"] in sample.get("oof_train_ids", []):
                raise ValueError("OOF target in training IDs")
            raw_action = sample["action"]
            action = _action(raw_action)
            key = _context_key(state, raw_action, sample.get("prediction_before"))
            parent_key = _canonical([_key(action), _confidence(sample.get("prediction_before")) or "prediction_unknown"])
            row = {"match_id": sample["match_id"], "delta": float(sample["delta"]),
                   "kickoff": sample["kickoff"], "key": key, "parent_key": parent_key}
            if not math.isfinite(row["delta"]):
                raise ValueError("Delta must be finite")
            grouped[(key, parent_key)].append(row)
            parents[parent_key].append(row)
            modes.add(state["information_mode"])
            ids.add(sample["match_id"])
        if len(modes) != 1:
            raise ValueError("Risk estimator requires one information mode")
        parent_summaries = {key: _summary([], rows, self.shrinkage, self.risk_multiplier)
                            for key, rows in parents.items()}
        self.parents = parent_summaries
        self.cells = {}
        for (key, parent_key), rows in grouped.items():
            self.cells[key] = _summary(rows, parents[parent_key], self.shrinkage, self.risk_multiplier)
            self.cells[key]["parent_key"] = parent_key

        calibration = list(calibration_samples or [])
        ratios = []
        for sample in calibration:
            raw_action = sample["action"]
            action = _action(raw_action)
            key = _context_key(sample["state"], raw_action, sample.get("prediction_before"))
            summary = self.cells.get(key)
            if summary and summary.get("uncertainty") and summary["uncertainty"] > 0:
                ratios.append(abs(float(sample["delta"]) - summary["expected_gain"]) / summary["uncertainty"])
        if ratios:
            ratios.sort()
            index = min(len(ratios) - 1, max(0, int(math.ceil(calibration_target_coverage * len(ratios))) - 1))
            self.risk_multiplier = max(self.risk_multiplier, ratios[index])
            calibrated = True
        else:
            calibrated = False
        # Recompute lower bounds after the validation-only risk multiplier choice.
        for summary in list(self.cells.values()) + list(self.parents.values()):
            uncertainty = summary.get("uncertainty")
            expected = summary.get("expected_gain")
            summary["lower_gain"] = expected - self.risk_multiplier * uncertainty if expected is not None and uncertainty is not None else None
        self.manifest = {"dataset_version": dataset_version, "trained_as_of": cutoff.isoformat(),
                         "training_match_ids": sorted(ids), "training_matches": len(ids),
                         "oof_samples": len(samples), "calibration_samples": len(calibration),
                         "calibration_target_coverage": calibration_target_coverage,
                         "calibration_used": calibrated, "risk_multiplier": self.risk_multiplier,
                         "information_mode": next(iter(modes)),
                         "runtime_features": ["visible_mask", "action_metadata", "quality_status", "freshness_age", "dependency_key", "cache_policy", "prediction_confidence"],
                         "target": "forward_oof_log_loss_before_minus_after",
                         "support_unit": "unique_match", "cost_units": COST_UNITS,
                         "uncertainty_note": "Empirical OOF residual scale; no Bayesian posterior or strict coverage guarantee.",
                         "metadata": metadata or {}}
        self.version = VERSION + ":" + hashlib.sha256(_canonical(self._payload()).encode()).hexdigest()[:20]
        return self

    def estimate_value(self, visible_state: dict, candidates: list[dict], *, prediction: Any = None,
                       risk_level: str = "mean") -> list[dict]:
        if risk_level not in RISK_LEVELS:
            raise ValueError("risk_level must be mean or conservative")
        if not self.manifest:
            raise ValueError("RiskAwareValueEstimator must be fitted")
        if visible_state["information_mode"] != self.manifest["information_mode"]:
            raise ValueError("Information mode mismatch")
        if parse_time(visible_state["prediction_cutoff"]) < parse_time(self.manifest["trained_as_of"]):
            raise ValueError("Estimator fitted after requested cutoff")
        output = []
        for raw in candidates:
            action = _action(raw)
            key = _context_key(visible_state, raw, prediction)
            summary = self.cells.get(key)
            fallback = None
            if summary is None:
                parent_key = _canonical([_key(action), _confidence(prediction) or "prediction_unknown"])
                summary = self.parents.get(parent_key)
                fallback = "state_quality_fallback_to_action_confidence" if summary else "unsupported_action"
            if summary is None:
                expected = uncertainty = lower = None
                support = 0
            else:
                expected = summary.get("expected_gain")
                uncertainty = summary.get("uncertainty")
                lower = summary.get("lower_gain")
                support = summary.get("support", 0)
                if support < self.min_support:
                    fallback = "insufficient_support" if support else fallback
            decision_gain = lower if risk_level == "conservative" else expected
            output.append({"action_id": action["action_id"], "action": action,
                           "expected_gain": expected, "gain": expected,
                           "uncertainty": uncertainty, "standard_error": uncertainty,
                           "lower_gain": lower, "decision_gain": decision_gain,
                           "support": support, "support_matches": support,
                           "parent_support": summary.get("parent_support", 0) if summary else 0,
                           "cost_vector": action["cost"], "cost": action["cost"],
                           "cost_status": action["cost_status"], "risk_level": risk_level,
                           "estimator_version": self.version, "version": self.version,
                           "fallback_reason": fallback, "reason": fallback or "estimated",
                           "context_key": key, "quality_context": _quality_bucket(visible_state, raw),
                           "net_gain": None, "net_gain_note": "Computed by policy with frozen unit lambda."})
        return output

    def _payload(self) -> dict:
        return {"schema": VERSION, "parameters": {"min_support": self.min_support,
                "shrinkage": self.shrinkage, "risk_multiplier": self.risk_multiplier},
                "cells": self.cells, "parents": self.parents, "manifest": self.manifest}

    def save(self, path: str | Path) -> None:
        payload = self._payload()
        payload["version"] = self.version
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "RiskAwareValueEstimator":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != VERSION:
            raise ValueError("Unsupported risk estimator schema")
        model = cls(**payload["parameters"])
        model.cells, model.parents, model.manifest = payload["cells"], payload["parents"], payload["manifest"]
        expected = VERSION + ":" + hashlib.sha256(_canonical(model._payload()).encode()).hexdigest()[:20]
        if payload.get("version") != expected:
            raise ValueError("Risk estimator artifact hash mismatch")
        model.version = payload["version"]
        return model
