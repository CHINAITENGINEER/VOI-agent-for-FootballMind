"""Empirical, forward-OOF loss improvements; this is not exact Bayes VoI.

Only mask and registered action identity are used at inference. Each match first
contributes one mean per cell, so repeated leads/masks do not inflate support.
Uncertainty is a descriptive week-cluster sandwich SE, without coverage claims.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

GROUPS = ("elo", "recent_form", "xg")
COST_UNITS = {"requests": "request", "money": "currency_unit", "latency_ms": "millisecond", "tokens": "token"}
VERSION = "empirical-mask-action-shrinkage-v1"


def _time(value: Any) -> datetime:
    value = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("Timezone-aware timestamp required")
    return value.astimezone(timezone.utc)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _mask(state: dict) -> tuple[str, dict]:
    from .state import sanitize_state
    state = sanitize_state(state)
    return "|".join(state.get("groups", {}).get(group, {}).get("status", "not_acquired") for group in GROUPS), state


def _action(action: dict) -> dict:
    """Metadata allowlist: never propagate returned evidence/feature payloads."""
    if not isinstance(action.get("action_id"), str) or not action["action_id"]:
        raise ValueError("Action requires action_id")
    groups = action.get("group_ids")
    if not isinstance(groups, (list, tuple)) or not groups or any(g not in GROUPS for g in groups):
        raise ValueError("Action requires registered group_ids")
    return {"action_id": action["action_id"], "group_ids": sorted(set(groups)),
            "available": action.get("available", True), "cost": dict(action.get("cost", {})),
            "cost_status": action.get("cost_status", "unknown"),
            "retry_allowed": action.get("retry_allowed", False),
            "refresh_allowed": action.get("refresh_allowed", False),
            "max_attempts": action.get("max_attempts", 1)}


def _key(action: dict) -> str:
    return _canonical([action["action_id"], action["group_ids"]])


def _confidence(prediction) -> str | None:
    if prediction is None:
        return None
    if isinstance(prediction, dict):
        if "probabilities" in prediction:
            prediction = prediction["probabilities"]
        else:
            prediction = [prediction[k] for k in ("home_win_probability", "draw_probability", "away_win_probability")]
    if isinstance(prediction, dict):
        prediction = [prediction[k] for k in ("home", "draw", "away")]
    values = [float(v) for v in prediction]
    if len(values) != 3 or any(not math.isfinite(v) or v < 0 or v > 1 for v in values) or abs(sum(values) - 1) > 1e-6:
        raise ValueError("Expected three normalized current-state probabilities")
    return "max_probability_lt_0.45" if max(values) < 0.45 else "max_probability_ge_0.45"


def _aggregate(rows: list[dict]) -> dict[str, tuple[float, str]]:
    by_match: dict[str, list] = defaultdict(list)
    for row in rows:
        by_match[row["match_id"]].append(row)
    return {match: (sum(r["delta"] for r in values) / len(values), values[0]["week"])
            for match, values in by_match.items()}


def _summary(local: dict, parent: dict, shrinkage: float, z: float) -> dict:
    n, total = len(local), len(parent)
    weight = n / (n + shrinkage) if n else 0.0
    if n and shrinkage == 0:
        weight = 1.0
    parent_mean = sum(v[0] for v in parent.values()) / total
    local_mean = sum(v[0] for v in local.values()) / n if n else parent_mean
    gain = weight * local_mean + (1 - weight) * parent_mean
    influences: dict[str, float] = defaultdict(float)
    if weight < 1:
        for match, (value, week) in parent.items():
            influences[week] += (1 - weight) * (value - parent_mean) / total
    for match, (value, week) in local.items():
        influences[week] += weight * (value - local_mean) / n
    blocks = len(influences)
    se = math.sqrt(blocks / (blocks - 1) * sum(v * v for v in influences.values())) if blocks >= 2 else None
    return {"gain": gain, "standard_error": se,
            "lower_gain": gain - z * se if se is not None else None,
            "support_matches": n, "parent_support_matches": total,
            "time_blocks": blocks, "local_time_blocks": len({v[1] for v in local.values()}),
            "parent_time_blocks": len({v[1] for v in parent.values()}), "local_weight": weight,
            "uncertainty_kind": "descriptive_week_cluster_sandwich_no_coverage_guarantee"}


class ValueEstimator:
    """Shrink action means toward their action-wide, match-weighted average."""

    def __init__(self, min_support: int = 12, shrinkage: float = 20.0,
                 uncertainty_multiplier: float = 1.64):
        if min_support < 1 or not math.isfinite(shrinkage) or shrinkage < 0:
            raise ValueError("Invalid support/shrinkage")
        if not math.isfinite(uncertainty_multiplier) or uncertainty_multiplier < 0:
            raise ValueError("Invalid uncertainty multiplier")
        self.min_support, self.shrinkage, self.uncertainty_multiplier = min_support, shrinkage, uncertainty_multiplier
        self.cells: dict = {}
        self.parents: dict = {}
        self.conditional: dict = {}
        self.manifest: dict = {}
        self.version = VERSION

    def fit(self, samples: list[dict], *, as_of: Any, dataset_version: str, **metadata) -> "ValueEstimator":
        cutoff = _time(as_of)
        cells, parents, conditional = defaultdict(list), defaultdict(list), defaultdict(list)
        provenance, ids, match_facts, modes = [], set(), {}, set()
        for sample in samples:
            mask, state = _mask(sample["state"])
            mid = sample["match_id"]
            if mid != state["match_id"]:
                raise ValueError("Sample and visible-state ID mismatch")
            kickoff, label_at = _time(sample["kickoff"]), _time(sample["result_available_at"])
            if kickoff != _time(state["base"]["kickoff"]):
                raise ValueError("Sample and visible-state kickoff mismatch")
            modes.add(state["information_mode"])
            train_at, prediction_at = _time(sample["oof_train_cutoff"]), _time(state["prediction_cutoff"])
            if not train_at < prediction_at < kickoff or not kickoff <= label_at <= cutoff:
                raise ValueError("Forward OOF/sample availability violation")
            if mid in sample.get("oof_train_ids", []):
                raise ValueError("OOF target occurs in predictor training IDs")
            facts = (kickoff, label_at)
            if mid in match_facts and match_facts[mid] != facts:
                raise ValueError("Conflicting match timing across OOF samples")
            match_facts[mid] = facts
            delta = float(sample["delta"])
            if not math.isfinite(delta):
                raise ValueError("OOF delta must be finite")
            action = _action(sample["action"])
            key = _key(action)
            year, week, _ = kickoff.isocalendar()
            row = {"match_id": mid, "delta": delta, "week": f"{year}-{week:02}"}
            cells[(mask, key)].append(row)
            parents[key].append(row)
            confidence = _confidence(sample.get("prediction_before"))
            if confidence is not None:
                conditional[(mask, key, confidence)].append(row)
            ids.add(mid)
            provenance.append({"match_id": mid, "prediction_cutoff": prediction_at.isoformat(),
                               "result_available_at": label_at.isoformat(),
                               "oof_train_cutoff": train_at.isoformat(),
                               "oof_train_ids": sorted(sample.get("oof_train_ids", [])),
                               "oof_manifest_ref": sample.get("oof_manifest_ref"),
                               "mask": mask, "action_key": key})
        if not samples:
            raise ValueError("At least one historical OOF sample required")
        if len(modes) != 1:
            raise ValueError("Value estimator requires one information mode")
        aggregated = {key: _aggregate(rows) for key, rows in parents.items()}
        self.parents = {key: _summary({}, rows, self.shrinkage, self.uncertainty_multiplier) for key, rows in aggregated.items()}
        self.cells = {_canonical([mask, key]): _summary(_aggregate(rows), aggregated[key], self.shrinkage, self.uncertainty_multiplier)
                      for (mask, key), rows in cells.items()}
        self.conditional = {_canonical([mask, key, confidence]): _summary(_aggregate(rows), _aggregate(cells[(mask, key)]), self.shrinkage, self.uncertainty_multiplier)
                            for (mask, key, confidence), rows in conditional.items()}
        self.manifest = {"dataset_version": dataset_version, "trained_as_of": cutoff.isoformat(),
                         "training_match_ids": sorted(ids), "training_matches": len(ids),
                         "oof_samples": len(samples), "oof_provenance": provenance,
                         "information_mode": next(iter(modes)),
                         "runtime_features": ["visible_group_status_mask", "action_id", "action_group_ids", "current_prediction_max_probability_lt_0.45"],
                         "target": "out_of_fold_log_loss_before_minus_after", "metadata": metadata,
                         "support_unit": "unique_match", "cost_units": COST_UNITS,
                         "uncertainty_note": "Descriptive clustered estimate; fitted predictor uncertainty and transport bias are not covered."}
        digest = hashlib.sha256(_canonical(self._payload()).encode()).hexdigest()[:20]
        self.version = f"{VERSION}:{digest}"
        return self

    def estimate(self, state: dict, actions: list[dict], prediction=None) -> list[dict]:
        start = time.perf_counter()
        if not self.manifest:
            raise ValueError("ValueEstimator must be fitted")
        mask, visible = _mask(state)
        if visible["information_mode"] != self.manifest["information_mode"]:
            raise ValueError("Value estimator information mode mismatch")
        confidence = _confidence(prediction)
        if _time(visible["prediction_cutoff"]) < _time(self.manifest["trained_as_of"]):
            raise ValueError("Value estimator was fitted after requested prediction cutoff")
        results = []
        for raw in actions:
            action = _action(raw)
            key = _key(action)
            cell = self.cells.get(_canonical([mask, key]))
            reason = "estimated"
            conditioned = self.conditional.get(_canonical([mask, key, confidence])) if confidence is not None else None
            conditional_support = conditioned["support_matches"] if conditioned else 0
            if conditioned is not None and conditional_support >= self.min_support:
                cell = conditioned
            elif cell is not None:
                reason = "conditional_support_fallback_to_mask" if confidence is not None else "prediction_missing_fallback_to_mask"
            if cell is None:
                cell = self.parents.get(key)
                reason = "unsupported_state_action" if cell else "unsupported_action"
            if cell is None:
                cell = {"gain": None, "standard_error": None, "lower_gain": None,
                        "support_matches": 0, "parent_support_matches": 0, "time_blocks": 0,
                        "uncertainty_kind": "no_support"}
            elif cell["support_matches"] < self.min_support:
                reason = "insufficient_support" if cell["support_matches"] else reason
            results.append({**cell, "action_id": action["action_id"], "action": action,
                            "cost": action["cost"], "cost_status": action["cost_status"],
                            "mask": mask, "version": self.version, "reason": reason,
                            "supported": cell["support_matches"] >= self.min_support,
                            "confidence_bin": confidence, "conditional_support_matches": conditional_support,
                            "gain_unit": "expected_log_loss_nats",
                            "net_gain": None, "net_gain_note": "Computed by policy using frozen unit-converting lambda."})
        elapsed = (time.perf_counter() - start) * 1000
        for result in results:
            result["estimator_batch_latency_ms"] = elapsed
            result["estimator_cost"] = {"requests": 0, "money": 0, "tokens": 0, "latency_ms": elapsed}
        return results

    def _payload(self) -> dict:
        return {"schema": VERSION, "parameters": {"min_support": self.min_support,
                "shrinkage": self.shrinkage, "uncertainty_multiplier": self.uncertainty_multiplier},
                "cells": self.cells, "parents": self.parents, "conditional": self.conditional, "manifest": self.manifest}

    def save(self, path: str | Path) -> None:
        payload = self._payload()
        payload["version"] = self.version
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ValueEstimator":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.pop("schema") != VERSION:
            raise ValueError("Unsupported value estimator schema")
        model = cls(**payload["parameters"])
        model.cells, model.parents, model.manifest = payload["cells"], payload["parents"], payload["manifest"]
        model.conditional = payload["conditional"]
        expected = f"{VERSION}:" + hashlib.sha256(_canonical(model._payload()).encode()).hexdigest()[:20]
        if payload["version"] != expected:
            raise ValueError("Value estimator artifact hash mismatch")
        model.version = payload["version"]
        return model
