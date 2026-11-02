"""Fit the full diagnostic g panel with match-balanced weights.

All observed positive, negative, and zero transitions are retained.  A match
with many sequential transitions receives the same total weight as a match
with few transitions.  This is a diagnostic fit only; it cannot enable the
formal empirical-g runtime.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = ROOT / "var/reviews/fm-g-nonleeds-expansion-20261015-v1"
OLD = ROOT / "var/reviews/fm-historical-transition-g-training-20261014-v2/transitions.json"
NEW = BASE / "g-diagnostic/rows.json"
OUT = ROOT / "var/reviews/fm-g-exploration-contract-20261015-v1"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def entropy(p):
    return float(-sum(x * math.log(max(x, 1e-12)) for x in p))


def title_bits(title):
    t = (title or "").casefold()
    return [
        float(any(x in t for x in ("injury", "injuries", "availability", "fitness", "suspension"))),
        float(any(x in t for x in ("line-up", "lineup", "line up", "xi"))),
        float("training" in t or "cobham" in t),
        float("preview" in t),
    ]


def old_vec(row):
    q = row["query"]
    p = row["prediction_before"]
    bits = title_bits(q.get("title", ""))
    return [
        *p,
        entropy(p),
        float(q.get("acquired_documents", 0)),
        1.0,
        *bits,
        float(not any(bits)),
        0.0,
    ]


def class_of(y):
    if y > 1e-12:
        return "positive"
    if y < -1e-12:
        return "negative"
    return "zero"


def mse(y, p):
    return float(np.mean((np.asarray(y) - np.asarray(p)) ** 2))


def main():
    old = read_json(OLD)
    new = read_json(NEW)
    rows = []
    for r in old:
        rows.append({
            "physical_match_id": r["physical_match_id"],
            "gain": float(r["signed_gain"]),
            "x": old_vec(r),
            "source": "historical_transition",
        })
    for r in new:
        rows.append({
            "physical_match_id": r["physical_match_id"],
            "gain": float(r["gain"]),
            "x": r["x"],
            "source": "nonleeds_expansion",
        })

    by_match = defaultdict(list)
    for row in rows:
        by_match[row["physical_match_id"]].append(row)
    weights = {mid: 1.0 / len(group) for mid, group in by_match.items()}
    sample_weights = np.asarray([weights[r["physical_match_id"]] for r in rows])
    x = np.asarray([r["x"] for r in rows])
    y = np.asarray([r["gain"] for r in rows])
    model = Ridge(alpha=0.1).fit(x, y, sample_weight=sample_weights)
    pred = model.predict(x)
    global_mean = float(np.average(y, weights=sample_weights))

    group_rows = []
    for mid in sorted(by_match):
        train = [r for r in rows if r["physical_match_id"] != mid]
        test = by_match[mid]
        train_x = np.asarray([r["x"] for r in train])
        train_y = np.asarray([r["gain"] for r in train])
        train_w = np.asarray([weights[r["physical_match_id"]] for r in train])
        fitted = Ridge(alpha=0.1).fit(train_x, train_y, sample_weight=train_w)
        test_y = [r["gain"] for r in test]
        test_p = fitted.predict(np.asarray([r["x"] for r in test]))
        group_rows.append({
            "physical_match_id": mid,
            "n": len(test),
            "mse_model": mse(test_y, test_p),
            "mse_zero": mse(test_y, [0.0] * len(test_y)),
            "mse_global_mean": mse(test_y, [global_mean] * len(test_y)),
            "nonzero_direction_accuracy": (
                sum((float(p) > 0) == (float(t) > 0) for t, p in zip(test_y, test_p) if abs(t) > 1e-12)
                / sum(abs(t) > 1e-12 for t in test_y)
                if any(abs(t) > 1e-12 for t in test_y)
                else None
            ),
        })

    report = {
        "schema": "footballmind-g-balanced-diagnostic-v1",
        "rows": len(rows),
        "physical_match_groups": len(by_match),
        "gain_class_counts": dict(Counter(class_of(float(v)) for v in y)),
        "weighting": "each physical_match_id has equal total weight",
        "model": "Ridge(alpha=0.1, sample_weight=1/n_transitions_per_match)",
        "fit_mse": mse(y, pred),
        "fit_mse_zero": mse(y, [0.0] * len(y)),
        "fit_mse_global_mean": mse(y, [global_mean] * len(y)),
        "weighted_global_mean": global_mean,
        "leave_one_match_out": group_rows,
        "selection_allowed": False,
        "empirical_support_status": "UNKNOWN",
        "production_ready": False,
        "propensity_available_rows": 0,
        "prospective_training_eligible_rows": 0,
        "interpretation": "all observed labels retained for diagnostic fit; historical propensity and source independence remain unresolved",
    }
    model_payload = {
        "schema": report["schema"],
        "features": [
            "p_home", "p_draw", "p_away", "entropy", "acquired_documents",
            "visible_candidates", "title_availability", "title_selection",
            "title_training", "title_preview", "title_other", "l1_source_delta",
        ],
        "coefficients": model.coef_.tolist(),
        "intercept": float(model.intercept_),
        "alpha": 0.1,
        "weighting": report["weighting"],
        "selection_allowed": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "balanced-fit-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "balanced-model.json").write_text(json.dumps(model_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
