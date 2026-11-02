"""Build the first auditable g-training bridge.

This command deliberately keeps three things separate:

* ``empirical``: the frozen historical B transitions and their realised
  signed log gains;
* ``one_match_probe``: one real historical match, showing the probability
  before/after an information reveal;
* ``synthetic_teacher_prior``: a small, explicitly synthetic curriculum that
  teaches the direction of an information-value heuristic when the empirical
  corpus is too small.  It is a prior, not evidence of live performance.

The production adapter may consume the resulting prior later.  This file does
not mutate the frozen p9 model or silently turn synthetic rows into empirical
training rows.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "footballmind-g-training-bridge-v1"
SYNTHETIC_SCHEMA = "footballmind-g-synthetic-teacher-prior-v1"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _iso(value: str) -> str:
    return value.replace("Z", "+00:00")


def _timestamp_before(a: str, b: str) -> bool:
    from datetime import datetime
    return datetime.fromisoformat(_iso(a)) < datetime.fromisoformat(_iso(b))


def _real_case_index(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {case["fixture"]["physical_match_id"]: case for case in cases}


def _select_showcase(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Prefer the real Leeds--Manchester United fitness reveal.

    It has a timestamped official article, a draw label, and a clear positive
    realised gain.  Fall back deterministically if a future corpus removes it.
    """
    preferred = [
        row for row in rows
        if row.get("physical_match_id") == "understat:28976"
        and row.get("action_index") == 0
        and row.get("acquired_indices") == []
    ]
    if preferred:
        return preferred[0]
    return max(rows, key=lambda row: float(row.get("gain", -math.inf)))


def _teacher_rows() -> tuple[list[dict[str, Any]], list[str]]:
    """Create a transparent, deterministic synthetic g curriculum.

    The target is an *expected signed log-score gain* prior.  It encodes the
    research hypothesis that an information source is valuable when current
    uncertainty and decision sensitivity are high, the source is reliable and
    specific, and redundancy/cost/misdirection risk are low.  This is a
    teaching prior, not a claim about observed matches.
    """
    kinds = [
        ("lineup_injury", .90, .90, .82, .88, .10, .08),
        ("tactical_plan", .72, .80, .72, .78, .18, .14),
        ("motivation_rotation", .78, .72, .58, .70, .22, .22),
        ("odds_shock", .65, .67, .76, .62, .30, .18),
        ("weather", .35, .30, .70, .55, .45, .28),
        ("historical_h2h", .25, .18, .52, .45, .65, .35),
    ]
    rows: list[dict[str, Any]] = []
    # Three uncertainty/sensitivity regimes per information family create
    # enough variation to fit a small prior without pretending to have 100s of
    # independent football fixtures.
    regimes = [
        ("low", .55, .65, .00),
        ("medium", .78, .90, .06),
        ("high", .96, 1.00, .12),
    ]
    for kind, base_u, base_s, reliability, specificity, redundancy, misdirection in kinds:
        for regime, u_mult, s_mult, noise in regimes:
            uncertainty = min(1., base_u * u_mult)
            sensitivity = min(1., base_s * s_mult)
            # A bounded, signed target: expected upside after accounting for
            # the chance the article is stale, generic, or misleading.
            raw = .34 * uncertainty * sensitivity * reliability * specificity
            raw *= (1. - redundancy) * (1. - 2. * (misdirection + noise))
            raw -= .015 * (0.1 + redundancy)
            target = float(max(-.16, min(.40, raw)))
            rows.append({
                "row_id": f"synthetic:{kind}:{regime}",
                "information_kind": kind,
                "regime": regime,
                "uncertainty": round(uncertainty, 6),
                "decision_sensitivity": round(sensitivity, 6),
                "source_reliability": reliability,
                "specificity": specificity,
                "redundancy": redundancy,
                "misdirection_risk": round(misdirection + noise, 6),
                "query_cost": round(.02 + .04 * redundancy, 6),
                "teacher_expected_signed_gain": target,
                "label_origin": "synthetic_domain_teacher",
            })
    return rows, [k[0] for k in kinds]


def _synthetic_design(rows: list[dict[str, Any]], kinds: list[str]) -> tuple[np.ndarray, list[str]]:
    names = [
        "intercept", "uncertainty", "decision_sensitivity", "source_reliability",
        "specificity", "redundancy", "misdirection_risk", "query_cost",
    ] + [f"kind:{kind}" for kind in kinds]
    matrix = []
    for row in rows:
        matrix.append([
            1., row["uncertainty"], row["decision_sensitivity"], row["source_reliability"],
            row["specificity"], row["redundancy"], row["misdirection_risk"], row["query_cost"],
            *[float(row["information_kind"] == kind) for kind in kinds],
        ])
    return np.asarray(matrix, dtype=float), names


def _fit_synthetic_prior(rows: list[dict[str, Any]], kinds: list[str]) -> dict[str, Any]:
    x, feature_names = _synthetic_design(rows, kinds)
    y = np.asarray([row["teacher_expected_signed_gain"] for row in rows], dtype=float)
    # Ridge with an unpenalised intercept.  The value is fixed and disclosed;
    # no hyperparameter search is claimed for this synthetic curriculum.
    lam = .10
    gram = x.T @ x
    penalty = np.diag([0.] + [lam] * (x.shape[1] - 1))
    coefficients = np.linalg.solve(gram + penalty, x.T @ y)
    predictions = x @ coefficients
    mse = float(np.mean((predictions - y) ** 2))
    # Group holdout is a sanity check against simply memorising row IDs.
    group_mse = {}
    for kind in kinds:
        train = np.asarray([row["information_kind"] != kind for row in rows])
        test = ~train
        g = x[train]
        z = np.diag([0.] + [lam] * (x.shape[1] - 1))
        beta = np.linalg.solve(g.T @ g + z, g.T @ y[train])
        group_mse[kind] = float(np.mean((x[test] @ beta - y[test]) ** 2))
    return {
        "schema": SYNTHETIC_SCHEMA,
        "label_origin": "synthetic_domain_teacher",
        "target": "teacher_expected_signed_gain_before_query_cost",
        "feature_names": feature_names,
        "regularization": lam,
        "coefficients": coefficients.tolist(),
        "training_rows": len(rows),
        "training_mse": mse,
        "leave_one_information_kind_out_mse": group_mse,
        "calibration_status": "not_established",
        "production_ready": False,
        "usage_rule": "prior_only_until_real_independent_gains_calibrate_it",
    }


def _predict_prior(model: dict[str, Any], row: dict[str, Any], kinds: list[str]) -> float:
    values = [
        1., row["uncertainty"], row["decision_sensitivity"], row["source_reliability"],
        row["specificity"], row["redundancy"], row["misdirection_risk"], row["query_cost"],
        *[float(row["information_kind"] == kind) for kind in kinds],
    ]
    return float(np.asarray(values) @ np.asarray(model["coefficients"]))


def build(p9_dir: Path, cases_path: Path, output: Path) -> dict[str, Any]:
    rows = _load(p9_dir / "B-targets.json")
    cases = _load(cases_path)
    if not rows or not cases:
        raise ValueError("p9 B targets and cases are required")
    by_id = _real_case_index(cases)
    showcase = _select_showcase(rows)
    case = by_id.get(showcase["physical_match_id"])
    if case is None:
        raise ValueError("showcase target has no frozen case")
    docs = {doc["url"]: doc for doc in case["documents"]}
    doc = docs.get(showcase["url"])
    if doc is None:
        raise ValueError("showcase target has no source document")
    kickoff = case["fixture"]["kickoff"]
    publication = doc["source_published_at"]
    if not _timestamp_before(publication, kickoff):
        raise ValueError("showcase publication must precede kickoff")

    model_json = _load(p9_dir / "model.json")
    g_selection = model_json["metadata"].get("g_selection", {})
    g_estimate = model_json["g"]["coefficients"][0]
    real_gains = [float(row["gain"]) for row in rows]
    empirical = {
        "model_path": str(p9_dir / "model.json"),
        "model_version": model_json.get("model_version"),
        "g_selection": g_selection,
        "g_estimate_constant": g_estimate,
        "independent_matches": sorted({row["physical_match_id"] for row in rows}),
        "transition_count": len(rows),
        "positive_gain_count": sum(g > 0 for g in real_gains),
        "negative_gain_count": sum(g < 0 for g in real_gains),
        "mean_realised_gain": float(np.mean(real_gains)),
        "mse_against_constant": float(np.mean((np.asarray(real_gains) - g_estimate) ** 2)),
        "evidence_origin": "historical_reconstruction",
        "calibration_status": "not_established",
    }
    one_match = {
        "schema": "footballmind-g-one-match-probe-v1",
        "source_kind": "real_historical_reconstruction",
        "fixture": case["fixture"],
        "outcome": showcase["outcome"],
        "action": {
            "index": showcase["action_index"],
            "url": showcase["url"],
            "title": doc["title"],
            "source_published_at": publication,
            "captured_at": doc["captured_at"],
            "publication_before_kickoff": True,
        },
        "prediction_before": showcase["prediction_before"],
        "prediction_after": showcase["prediction_after"],
        "true_outcome_probability_before": showcase["prediction_before"][showcase["outcome"]],
        "true_outcome_probability_after": showcase["prediction_after"][showcase["outcome"]],
        "realised_signed_log_gain": showcase["gain"],
        "frozen_g_expected_gain": g_estimate,
        "residual_realised_minus_g": float(showcase["gain"] - g_estimate),
        "interpretation": "The fitness-update article increased the draw probability from 0.1011 to 0.1472; the match ended in a draw, so the realised signed log gain was positive. This is one target-construction example, not a calibrated generalisation test.",
    }

    synthetic_rows, kinds = _teacher_rows()
    synthetic_model = _fit_synthetic_prior(synthetic_rows, kinds)
    teaching_example = next(row for row in synthetic_rows if row["information_kind"] == "lineup_injury" and row["regime"] == "high")
    teaching_example = dict(teaching_example)
    teaching_example["prior_prediction"] = _predict_prior(synthetic_model, teaching_example, kinds)
    teaching_example["net_prior_after_cost"] = teaching_example["prior_prediction"] - teaching_example["query_cost"]

    result = {
        "schema": SCHEMA,
        "created_at": "2026-10-10T00:00:00+00:00",
        "purpose": "g_training_only",
        "empirical_refit": empirical,
        "one_match_probe": one_match,
        "synthetic_teacher_prior": {
            "model": synthetic_model,
            "rows": synthetic_rows,
            "showcase_lineup_injury_row": teaching_example,
        },
        "governance": {
            "synthetic_rows_are_not_real_matches": True,
            "synthetic_prior_may_rank_candidates_but_may_not_be_reported_as_empirical_performance": True,
            "real_g_remains_constant_until_more_independent_real_fixtures_are_added": True,
            "next_calibration_requirement": "at least 12 independent prospective or time-valid historical fixtures with f before/after and realised outcome labels",
        },
    }
    _dump(output / "g-training-bridge.json", result)
    _dump(output / "one-match-probe.json", one_match)
    _dump(output / "synthetic-g-model.json", synthetic_model)
    _dump(output / "synthetic-g-rows.json", synthetic_rows)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p9-dir", type=Path, default=Path("var/submission/p9-g-timestamped-replay-20261010"))
    parser.add_argument("--cases", type=Path, default=Path("var/submission/p8-g-stream-v3/cases.json"))
    parser.add_argument("--output", type=Path, default=Path("var/submission/p9-g-training-bridge-20261010"))
    args = parser.parse_args()
    result = build(args.p9_dir, args.cases, args.output)
    print(json.dumps({
        "output": str(args.output),
        "fixture": result["one_match_probe"]["fixture"]["physical_match_id"],
        "realised_signed_log_gain": result["one_match_probe"]["realised_signed_log_gain"],
        "frozen_g": result["one_match_probe"]["frozen_g_expected_gain"],
        "synthetic_rows": len(result["synthetic_teacher_prior"]["rows"]),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
