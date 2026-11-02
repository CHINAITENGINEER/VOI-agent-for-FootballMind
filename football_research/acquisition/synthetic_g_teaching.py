"""Transparent synthetic teaching data for the FootballMind information-value g.

This module is deliberately separate from the historical g artifact.  It does
not claim that synthetic rows are football observations.  It teaches a small
value-of-information surrogate from an explicit football decision mechanism:
uncertainty x event relevance x source reliability x time sensitivity, minus
acquisition cost.  The output is useful for wiring and demonstrations while
real timestamped gain labels are still sparse.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

VERSION = "synthetic-g-teaching-v1"
ACTIONS = ("lineup", "injury", "motivation", "weather")
SCENARIOS = ("rotation_sensitive", "tactical_sensitive", "stable", "weather_sensitive")
REGULARIZATION_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
FEATURE_NAMES = (
    "entropy", "max_probability", "time_to_kickoff_hours", "source_reliability",
    "event_relevance", "tactical_sensitivity", "cost_units",
    "action_lineup", "action_injury", "action_motivation", "action_weather",
    "scenario_rotation_sensitive", "scenario_tactical_sensitive",
    "scenario_stable", "scenario_weather_sensitive",
)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _normalise(p: np.ndarray) -> np.ndarray:
    p = np.maximum(np.asarray(p, dtype=float), 1e-8)
    return p / p.sum()


def _entropy(p: np.ndarray) -> float:
    return float(-np.sum(p * np.log(p)) / math.log(3.0))


def _scenario_for(rng: np.random.Generator) -> str:
    return str(rng.choice(SCENARIOS, p=[0.26, 0.26, 0.34, 0.14]))


def _base_probabilities(rng: np.random.Generator, scenario: str) -> np.ndarray:
    # These are stylised pre-query beliefs, not match results or bookmaker data.
    concentration = {"rotation_sensitive": 1.0, "tactical_sensitive": 1.4,
                     "stable": 2.3, "weather_sensitive": 1.2}[scenario]
    p = rng.dirichlet(np.full(3, concentration))
    return _normalise(p)


def _relevance(action: str, scenario: str, rng: np.random.Generator) -> float:
    table = {
        "rotation_sensitive": {"lineup": .96, "injury": .88, "motivation": .62, "weather": .18},
        "tactical_sensitive": {"lineup": .73, "injury": .48, "motivation": .35, "weather": .22},
        "stable": {"lineup": .28, "injury": .23, "motivation": .16, "weather": .08},
        "weather_sensitive": {"lineup": .35, "injury": .22, "motivation": .28, "weather": .91},
    }
    # Small bounded perturbation stops the teacher from being a lookup table.
    return float(np.clip(table[scenario][action] + rng.normal(0, .035), .02, .99))


def _reliability(action: str) -> float:
    return {"lineup": .94, "injury": .90, "motivation": .70, "weather": .78}[action]


def _direction(action: str, scenario: str, rng: np.random.Generator) -> np.ndarray:
    # Update directions encode football semantics only as a mechanism prior.
    if action == "lineup":
        d = np.array([.30, -.06, -.24]) if scenario == "rotation_sensitive" else np.array([.14, .02, -.16])
    elif action == "injury":
        d = np.array([-.17, .04, .13]) if scenario == "rotation_sensitive" else np.array([-.10, .03, .07])
    elif action == "motivation":
        d = np.array([.11, -.03, -.08]) if scenario == "tactical_sensitive" else np.array([.06, .01, -.07])
    else:
        d = np.array([-.05, .06, -.01]) if scenario == "weather_sensitive" else np.array([-.02, .03, -.01])
    return d + rng.normal(0, .015, 3)


def _features(row: Mapping[str, Any]) -> np.ndarray:
    values = [float(row[n]) for n in FEATURE_NAMES[:7]]
    values += [float(row["action"] == a) for a in ACTIONS]
    values += [float(row["scenario"] == s) for s in SCENARIOS]
    return np.asarray(values, dtype=float)


def generate_teaching_rows(*, seed: int = 20261010, matches: int = 48,
                            outcome_draws: int = 128) -> list[dict[str, Any]]:
    """Generate labelled candidate actions with an explicit synthetic provenance."""
    if matches < 12 or outcome_draws < 16:
        raise ValueError("teaching data needs at least 12 matches and 16 outcome draws")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for match_index in range(matches):
        scenario = _scenario_for(rng)
        p_before = _base_probabilities(rng, scenario)
        entropy = _entropy(p_before)
        max_probability = float(p_before.max())
        tactical_sensitivity = float(np.clip(rng.beta(2.0, 2.0), .02, .98))
        time_to_kickoff = float(rng.uniform(2.0, 72.0))
        for action in ACTIONS:
            relevance = _relevance(action, scenario, rng)
            reliability = _reliability(action)
            # Last-minute information can be more decision-relevant but has a
            # slightly higher request cost.  This is a declared mechanism prior.
            urgency = float(np.exp(-time_to_kickoff / 48.0))
            shift = relevance * reliability * (.20 + .80 * entropy) * (.65 + .35 * urgency)
            q_before = p_before.copy()
            q_after = _normalise(q_before + shift * _direction(action, scenario, rng))
            cost = {"lineup": .018, "injury": .014, "motivation": .010, "weather": .007}[action]
            cost *= 1.0 + .15 * urgency
            draws = rng.choice(3, size=outcome_draws, p=q_after)
            realized = np.log(np.maximum(q_after[draws], 1e-12)) - np.log(np.maximum(q_before[draws], 1e-12))
            expected_voi = float(np.sum(q_after * (np.log(q_after) - np.log(q_before))))
            rows.append({
                "sample_id": f"synthetic:{match_index:03d}:{action}",
                "physical_match_id": f"synthetic-match:{match_index:03d}",
                "action": action, "scenario": scenario,
                "entropy": entropy, "max_probability": max_probability,
                "time_to_kickoff_hours": time_to_kickoff,
                "source_reliability": reliability, "event_relevance": relevance,
                "tactical_sensitivity": tactical_sensitivity, "cost_units": cost,
                "p_before": p_before.tolist(), "q_after": q_after.tolist(),
                "outcome_draws": draws.tolist(),
                "realized_signed_gain_mean": float(realized.mean()),
                "realized_signed_gain_std": float(realized.std()),
                "teacher_expected_voi": expected_voi,
                "teacher_net_voi": expected_voi - cost,
                "provenance": {"evidence_origin": "constructed_mechanism",
                    "synthetic": True,
                    "teacher": "entropy x relevance x reliability x time sensitivity",
                    "not_real_match_observation": True},
            })
    return rows


def _fit_ridge(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    penalty = np.eye(x.shape[1], dtype=float) * lam
    penalty[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def fit_teaching_model(rows: Iterable[Mapping[str, Any]], *, seed: int = 20261010) -> dict[str, Any]:
    rows = list(rows)
    if not rows:
        raise ValueError("nonempty teaching rows required")
    mids = sorted({r["physical_match_id"] for r in rows})
    rng = np.random.default_rng(seed)
    shuffled = mids.copy(); rng.shuffle(shuffled)
    split = max(2, int(len(mids) * .8))
    train_ids, test_ids = set(shuffled[:split]), set(shuffled[split:])
    x_all = np.vstack([np.r_[1.0, _features(r)] for r in rows])
    y_all = np.asarray([float(r["realized_signed_gain_mean"]) for r in rows])
    train_mask = np.asarray([r["physical_match_id"] in train_ids for r in rows])
    test_mask = ~train_mask
    scores = []
    for lam in REGULARIZATION_GRID:
        beta = _fit_ridge(x_all[train_mask], y_all[train_mask], lam)
        pred = x_all[test_mask] @ beta
        scores.append({"regularization": lam,
                       "holdout_mse": float(np.mean((pred-y_all[test_mask])**2)),
                       "holdout_rows": int(test_mask.sum()),
                       "holdout_matches": len(test_ids)})
    winner = min(scores, key=lambda s: (s["holdout_mse"], s["regularization"]))
    beta = _fit_ridge(x_all[train_mask], y_all[train_mask], winner["regularization"])
    pred_all = x_all @ beta
    baseline = float(np.mean(y_all[train_mask]))
    baseline_mse = float(np.mean((y_all[test_mask] - baseline) ** 2))
    selected, oracle, matches_agree = [], [], []
    for mid in sorted(test_ids):
        group = [i for i, r in enumerate(rows) if r["physical_match_id"] == mid]
        chosen = max(group, key=lambda i: float(pred_all[i]))
        oracle_i = max(group, key=lambda i: float(rows[i]["teacher_net_voi"]))
        selected.append(float(rows[chosen]["teacher_net_voi"]))
        oracle.append(float(rows[oracle_i]["teacher_net_voi"]))
        matches_agree.append(chosen == oracle_i)
    return {
        "schema": VERSION, "status": "fitted_synthetic_teaching_surrogate",
        "target": "mean_realized_signed_fixed_f_log_loss_reduction",
        "target_note": "Monte-Carlo mean over synthetic outcomes; not Bayes EVSI and not real football gain",
        "evidence_origin": "constructed_mechanism", "synthetic": True,
        "seed": seed, "training_matches": len(train_ids), "holdout_matches": len(test_ids),
        "training_rows": int(train_mask.sum()), "holdout_rows": int(test_mask.sum()),
        "feature_names": ["intercept", *FEATURE_NAMES],
        "regularization_grid": list(REGULARIZATION_GRID), "selected_regularization": winner["regularization"],
        "holdout_scores": scores, "coefficients": beta.tolist(),
        "train_mse": float(np.mean((pred_all[train_mask]-y_all[train_mask])**2)),
        "holdout_mse": winner["holdout_mse"],
        "holdout_constant_baseline_mse": baseline_mse,
        "holdout_mse_improvement_vs_constant": baseline_mse - winner["holdout_mse"],
        "holdout_action_selection": {
            "teacher_net_voi_top1_agreement": float(np.mean(matches_agree)) if matches_agree else None,
            "mean_teacher_net_voi_selected_by_model": float(np.mean(selected)) if selected else None,
            "mean_teacher_net_voi_oracle": float(np.mean(oracle)) if oracle else None,
        },
        "support": {"independent_matches": len(train_ids), "minimum_for_this_teaching_demo": 12},
        "production_ready": False,
    }


class SyntheticGModel:
    """Read-only predictor for the synthetic teaching artifact.

    The adapter intentionally exposes provenance on every estimate.  It is
    suitable for mechanism demos and fallback wiring only; callers must not
    silently turn its result into an empirical or calibrated VoI estimate.
    """
    def __init__(self, payload: Mapping[str, Any]):
        if payload.get("schema") != VERSION or payload.get("synthetic") is not True:
            raise ValueError("A synthetic-g-teaching-v1 artifact is required")
        names = payload.get("feature_names")
        if names != ["intercept", *FEATURE_NAMES]:
            raise ValueError("Unexpected synthetic feature schema")
        coefficients = np.asarray(payload.get("coefficients"), dtype=float)
        if coefficients.shape != (len(names),) or not np.isfinite(coefficients).all():
            raise ValueError("Finite synthetic coefficients required")
        self.payload = dict(payload)
        self.coefficients = coefficients

    @classmethod
    def load(cls, path: str | Path) -> "SyntheticGModel":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def estimate(self, *, action: str, scenario: str, entropy: float,
                 max_probability: float, time_to_kickoff_hours: float,
                 source_reliability: float, event_relevance: float,
                 tactical_sensitivity: float, cost_units: float) -> dict[str, Any]:
        if action not in ACTIONS or scenario not in SCENARIOS:
            raise ValueError("Unknown synthetic action/scenario")
        row = {"action": action, "scenario": scenario, "entropy": entropy,
               "max_probability": max_probability,
               "time_to_kickoff_hours": time_to_kickoff_hours,
               "source_reliability": source_reliability,
               "event_relevance": event_relevance,
               "tactical_sensitivity": tactical_sensitivity,
               "cost_units": cost_units}
        x = np.r_[1.0, _features(row)]
        gain = float(x @ self.coefficients)
        return {"schema": "synthetic-g-estimate-v1", "expected_signed_gain": gain,
                "expected_net_voi": gain - float(cost_units), "action": action,
                "support_status": "synthetic_teaching_only", "evidence_origin": "constructed_mechanism",
                "synthetic": True, "calibration_status": "not_established",
                "production_ready": False, "model_schema": VERSION}


def build_artifact(output_dir: str | Path, *, seed: int = 20261010, matches: int = 48,
                   outcome_draws: int = 128) -> dict[str, Any]:
    folder = Path(output_dir); folder.mkdir(parents=True, exist_ok=True)
    rows = generate_teaching_rows(seed=seed, matches=matches, outcome_draws=outcome_draws)
    model = fit_teaching_model(rows, seed=seed + 1)
    now = datetime.now(timezone.utc).isoformat()
    dataset = {"schema": VERSION, "generated_at": now, "seed": seed,
               "matches": matches, "outcome_draws": outcome_draws,
               "evidence_origin": "constructed_mechanism", "synthetic": True,
               "rows": rows, "sha256": _digest(rows)}
    (folder / "teaching-dataset.json").write_text(json.dumps(dataset, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (folder / "model.json").write_text(json.dumps(model, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    report = {"schema": VERSION, "status": "complete", "generated_at": now,
              "dataset_sha256": dataset["sha256"], "model_sha256": _digest(model),
              "evidence_origin": "constructed_mechanism", "synthetic_rows": len(rows),
              "independent_matches": matches, "warning": "Synthetic teaching only; cannot be reported as measured real-world VoI or calibration.",
              "next_step": "replace with timestamped real before/after gains and keep this artifact only as a mechanism prior."}
    (folder / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return {"folder": str(folder.resolve()), "report": report, "model": model}


__all__ = ["VERSION", "ACTIONS", "SCENARIOS", "FEATURE_NAMES", "generate_teaching_rows", "fit_teaching_model", "SyntheticGModel", "build_artifact"]
