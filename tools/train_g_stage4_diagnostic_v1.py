"""Stage 4 diagnostic comparison for post-read g on existing local labels.

This is intentionally small-data and research-only. It uses settled fixed-f
labels after input construction, grouped leave-one-physical-match-out fits,
and never promotes a model or modifies online runtime.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

ROOT = Path(r"D:/a1-足彩")
LABELS = ROOT / r"var/work/fm-g-f-bound-labels-v1-20261024/g-f-bound-labels-v1.jsonl"
DATASET = ROOT / r"var/work/fm-postread-g-dataset-v1-20261021/postread-g-dataset-v1.jsonl"
OUT = ROOT / r"var/work/fm-g-stage4-diagnostic-20261024-v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def balanced_weights(match_ids):
    counts = Counter(match_ids)
    return np.asarray([1.0 / (len(counts) * counts[mid]) for mid in match_ids], dtype=float)


def metric(y, pred, weights):
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    mse = float(np.sum(weights * (y - pred) ** 2) / np.sum(weights))
    mae = float(np.sum(weights * np.abs(y - pred)) / np.sum(weights))
    corr = None if len(set(y)) < 2 or len(set(pred)) < 2 else float(spearmanr(y, pred).statistic)
    nonzero = np.abs(y) > 1e-12
    sign = float(np.mean((pred[nonzero] > 0) == (y[nonzero] > 0))) if np.any(nonzero) else None
    return {"mse": mse, "mae": mae, "spearman": corr, "sign_accuracy_nonzero": sign}


def fit_predict(kind, x_train, y_train, w_train, x_test):
    if kind.startswith("ridge"):
        model = Ridge(alpha=10.0)
        model.fit(x_train, y_train, sample_weight=w_train)
        return model.predict(x_test), {"learner": "Ridge", "alpha": 10.0}
    if kind == "mlp_body_state":
        # Four hidden units are deliberately fixed; there are only five groups.
        model = MLPRegressor(hidden_layer_sizes=(4,), activation="tanh", solver="lbfgs",
                             alpha=10.0, random_state=20261024, max_iter=2000)
        model.fit(x_train, y_train)
        return model.predict(x_test), {"learner": "MLPRegressor", "hidden_layer_sizes": [4],
                                      "activation": "tanh", "alpha": 10.0, "random_state": 20261024}
    raise ValueError(kind)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    label_rows = rows(LABELS)
    dataset_rows = {row["sample_id"]: row for row in rows(DATASET)}
    settled = [row for row in label_rows if row["realized_gain"] is not None]
    if not settled:
        raise ValueError("no settled labels")
    y = np.asarray([row["realized_gain"] for row in settled], dtype=float)
    mids = [row["physical_match_id"] for row in settled]
    weights = balanced_weights(mids)
    state = np.asarray([dataset_rows[row["sample_id"]]["current_state_vector"] for row in settled], dtype=float)
    body = np.asarray([dataset_rows[row["sample_id"]]["candidate_body_vector"]["values"] for row in settled], dtype=float)
    body_state = np.c_[body, state]
    assert state.shape == (len(settled), 7)
    assert body.shape == (len(settled), 256)
    assert np.isfinite(state).all() and np.isfinite(body).all() and np.isfinite(y).all()

    matrices = {"state_only": state, "body_only": body, "body_state": body_state}
    groups = sorted(set(mids))
    predictions = {"zero": np.zeros_like(y), "constant_lomo": np.full_like(y, np.nan)}
    for hold in groups:
        train = np.asarray([mid != hold for mid in mids])
        test = ~train
        predictions["constant_lomo"][test] = float(np.sum(weights[train] * y[train]) / np.sum(weights[train]))
    model_configs = {}
    for kind, x in matrices.items():
        pred = np.full(len(y), np.nan)
        cfg = None
        for hold in groups:
            train = np.asarray([mid != hold for mid in mids])
            test = ~train
            current, cfg = fit_predict("ridge_" + kind, x[train], y[train], weights[train], x[test])
            pred[test] = current
        predictions["ridge_" + kind] = pred
        model_configs["ridge_" + kind] = {**cfg, "input_dimension": int(x.shape[1]), "fit_protocol": "leave_one_physical_match_out"}
    pred = np.full(len(y), np.nan)
    cfg = None
    for hold in groups:
        train = np.asarray([mid != hold for mid in mids])
        test = ~train
        current, cfg = fit_predict("mlp_body_state", body_state[train], y[train], weights[train], body_state[test])
        pred[test] = current
    predictions["mlp_body_state"] = pred
    model_configs["mlp_body_state"] = {**cfg, "input_dimension": int(body_state.shape[1]), "fit_protocol": "leave_one_physical_match_out"}

    metrics = []
    for name, pred in predictions.items():
        metrics.append({"model": name, **metric(y, pred, weights), "rows": len(y), "independent_matches": len(groups)})
    # A title/old-g comparison is not silently reconstructed: Stage 2 did not retain title text.
    unavailable = {"model": "title_local", "status": "UNAVAILABLE", "reason": "title field absent from Stage 2 dataset; URL/body cannot substitute"}
    dump(OUT / "model.json", {
        "schema": "footballmind-g-stage4-diagnostic-models-v1",
        "status": "RESEARCH_ONLY_NOT_FORMAL_G",
        "models": model_configs,
        "input_layout": {"state_only": 7, "body_only": 256, "body_state": 263},
        "labels_used": len(y), "independent_matches": len(groups),
        "f_model_sha256": sorted({row["f_model_sha256"] for row in settled}),
        "outcome_or_gain_in_g_input": False, "selection_allowed": False, "production_ready": False,
    })
    dump(OUT / "report.json", {
        "schema": "footballmind-g-stage4-diagnostic-report-v1",
        "status": "STAGE4_DIAGNOSTIC_COMPLETE_NOT_FORMAL_G",
        "source_labels": str(LABELS), "source_labels_sha256": sha(LABELS),
        "source_dataset": str(DATASET), "source_dataset_sha256": sha(DATASET),
        "rows": len(y), "independent_matches": len(groups),
        "gain_sign_counts": dict(Counter("positive" if value > 1e-12 else "negative" if value < -1e-12 else "zero" for value in y)),
        "metrics": metrics, "unavailable_comparisons": [unavailable],
        "group_protocol": "leave_one_physical_match_out",
        "model_selection": "fixed_ridge_alpha_10_and_fixed_mlp_hidden_4; no validation selection because only five groups",
        "formal_g_training_eligible_rows": 0, "selection_allowed": False,
        "network_requests": 0, "provider_calls": 0, "f_refit": False, "gpu_used": False,
        "limitations": [
            "seven settled rows from five physical matches",
            "historical source availability is mostly UNKNOWN",
            "action_id, propensity and complete prequery menu are absent",
            "title-only baseline cannot be reconstructed from retained Stage 2 fields",
            "LOMO is a diagnostic estimate, not a production generalization result",
        ],
    })
    with (OUT / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "mse", "mae", "spearman", "sign_accuracy_nonzero", "rows", "independent_matches"])
        writer.writeheader()
        writer.writerows(metrics)
    with (OUT / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for i, row in enumerate(settled):
            handle.write(json.dumps({"sample_id": row["sample_id"], "physical_match_id": row["physical_match_id"],
                                     "realized_gain": float(y[i]), **{name: float(pred[i]) for name, pred in predictions.items()}}, ensure_ascii=False) + "\n")
    # Visuals: one model-comparison plot and one body_state predicted-vs-realized plot.
    names = [item["model"] for item in metrics]
    mse = [item["mse"] for item in metrics]
    plt.figure(figsize=(8, 4.8)); plt.bar(names, mse, color=["#7d8790", "#8ea1b8", "#2c8c89", "#c98639", "#b55d68"]); plt.ylabel("LOMO weighted MSE"); plt.title("Stage 4 diagnostic g comparison"); plt.xticks(rotation=25, ha="right"); plt.tight_layout(); plt.savefig(OUT / "model-comparison-mse.png", dpi=180); plt.close()
    plt.figure(figsize=(6.4, 5.2)); plt.scatter(y, predictions["ridge_body_state"], c=["#2c8c89" if v > 1e-12 else "#b55d68" if v < -1e-12 else "#7d8790" for v in y], s=70, edgecolor="white"); lo=float(min(y.min(), predictions["ridge_body_state"].min()))-.003; hi=float(max(y.max(), predictions["ridge_body_state"].max()))+.003; plt.plot([lo, hi], [lo, hi], "k--", lw=1); plt.xlabel("Realized signed gain"); plt.ylabel("LOMO body+state prediction"); plt.title("Fixed-f post-read g: body + state"); plt.grid(alpha=.2); plt.tight_layout(); plt.savefig(OUT / "body-state-predicted-vs-realized.png", dpi=180); plt.close()
    (OUT / "README.md").write_text("""# Stage 4 post-read g diagnostic

This package compares zero, constant leave-one-match-out, state-only Ridge,
body-only Ridge, body+state Ridge and a four-unit MLP on the existing seven
settled fixed-f labels from five physical matches. It is a grouped diagnostic,
not formal empirical-g training. The title-only baseline is explicitly
unavailable because Stage 2 did not retain title text.

All models exclude outcome, prediction_after and realized_gain from their input
vectors. No network/provider call or f refit was performed.
""", encoding="utf-8")
    files = {path.name: sha(path) for path in OUT.iterdir() if path.is_file() and path.name != "manifest.json"}
    dump(OUT / "manifest.json", {"schema": "footballmind-g-stage4-diagnostic-manifest-v1", "files": files})
    print(json.dumps({"rows": len(y), "matches": len(groups), "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
