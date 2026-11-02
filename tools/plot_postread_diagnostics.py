"""Create auditable plots for the post-read g training diagnostics.

The plots intentionally separate (1) signed-gain regression fit and
(2) end-to-end policy log loss.  No synthetic epochs or hidden labels are
created; the script only visualizes metrics already present in the reports.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_regularization(fit, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    colors = {"mlp": "#2563eb", "ridge": "#d97706"}
    labels = {"mlp": "MLP", "ridge": "Ridge"}
    for kind, ax in zip(("mlp", "ridge"), axes):
        rows = [r for r in fit["models"] if r["mode"] == "body_state" and r["kind"] == kind]
        rows.sort(key=lambda r: r["alpha"])
        alphas = np.array([r["alpha"] for r in rows], dtype=float)
        train = np.array([r["train"]["weighted_mse"] for r in rows])
        valid = np.array([r["validation"]["weighted_mse"] for r in rows])
        ax.plot(alphas, train, "o-", color="#2563eb", label="train MSE")
        ax.plot(alphas, valid, "o-", color="#dc2626", label="validation MSE")
        ax.axhline(fit["baselines"]["train_mean_mse"], color="#2563eb", ls="--", alpha=.55,
                   label="train mean baseline")
        ax.axhline(fit["baselines"]["validation_training_mean_mse"], color="#dc2626", ls="--", alpha=.55,
                   label="validation train-mean baseline")
        selected = [r for r in rows if r["selected"]]
        if selected:
            r = selected[0]
            ax.scatter([r["alpha"]], [r["validation"]["weighted_mse"]], marker="*", s=180,
                       color="#111827", zorder=5, label="selected")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("regularization alpha (log scale)")
        ax.set_ylabel("weighted signed-gain MSE (log scale)")
        ax.set_title(labels[kind] + " capacity and generalization")
        ax.grid(True, which="both", alpha=.22)
        ax.legend(fontsize=8)
    fig.suptitle("Post-read g: regularization path on body + state features\n"
                 "Training fit and held-out fit are shown separately; star marks policy-selected model")
    save(fig, out / "01_regularization_path.png")


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def plot_scatter(fit, predictions: Path, out: Path):
    rows = read_csv(predictions)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3), sharex=True, sharey=True)
    vals = np.array([float(r["actual_gain"]) for r in rows] + [float(r["predicted_gain"]) for r in rows])
    lim = max(.01, float(np.max(np.abs(vals))) * 1.05)
    for ax, split, title in zip(axes, ("train", "validation"), ("Training matches", "Held-out matches")):
        part = [r for r in rows if r["split"] == split]
        actual = np.array([float(r["actual_gain"]) for r in part])
        pred = np.array([float(r["predicted_gain"]) for r in part])
        ax.scatter(actual, pred, s=17, alpha=.55, color="#0f766e")
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="ideal equality (reference, not fitted)")
        ax.axhline(0, color="#9ca3af", lw=.8); ax.axvline(0, color="#9ca3af", lw=.8)
        ax.set_title(f"{title} (n={len(part)})")
        ax.set_xlabel("realized signed gain")
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    axes[0].set_ylabel("predicted gain by g")
    axes[0].set_xlim(-lim, lim); axes[0].set_ylim(-lim, lim)
    fig.suptitle("Selected body_state g: predicted versus realized signed gain\n"
                 "The validation panel is the key generalization check")
    save(fig, out / "02_gain_scatter.png")


def plot_binned_calibration(predictions: Path, out: Path):
    """Plot empirical bin means; this is descriptive and does not fit a line."""
    rows = read_csv(predictions)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3), sharex=True, sharey=True)
    all_pred = np.array([float(r["predicted_gain"]) for r in rows])
    lim = max(.01, float(np.max(np.abs(all_pred))) * 1.15)
    for ax, split, title in zip(axes, ("train", "validation"), ("Training matches", "Held-out matches")):
        part = [r for r in rows if r["split"] == split]
        pred = np.array([float(r["predicted_gain"]) for r in part])
        actual = np.array([float(r["actual_gain"]) for r in part])
        # Quantile bins avoid having all validation observations in the central bin.
        edges = np.unique(np.quantile(pred, np.linspace(0, 1, 7)))
        xs = []; ys = []; err = []; counts = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (pred >= lo) & (pred <= hi if hi == edges[-1] else pred < hi)
            if not np.any(mask):
                continue
            yy = actual[mask]
            xs.append(float(np.mean(pred[mask]))); ys.append(float(np.mean(yy)))
            err.append(float(np.std(yy, ddof=1) / np.sqrt(len(yy))) if len(yy) > 1 else 0.)
            counts.append(int(len(yy)))
        ax.errorbar(xs, ys, yerr=np.array(err) * 1.96, fmt="o-", capsize=3,
                    color="#0f766e", label="bin mean ± 1.96 SE")
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="ideal equality (reference, not fitted)")
        ax.axhline(0, color="#9ca3af", lw=.8); ax.axvline(0, color="#9ca3af", lw=.8)
        ax.set_title(f"{title} (6 quantile bins)")
        ax.set_xlabel("mean predicted gain in bin")
        ax.grid(alpha=.2); ax.legend(fontsize=8)
        ax.text(.02, .04, "bin counts: " + ", ".join(map(str, counts)), transform=ax.transAxes, fontsize=8)
    axes[0].set_ylabel("mean realized gain in bin")
    axes[0].set_xlim(-lim, lim); axes[0].set_ylim(-lim, lim)
    fig.suptitle("Binned calibration of g gain predictions\n"
                 "Descriptive bin averages; intervals do not treat transitions as independent matches")
    save(fig, out / "04_binned_calibration.png")


def plot_policy(report, out: Path):
    split = "validation"
    mode = "body_state"
    budget = "3"
    stats = report["comparisons"][split][mode]["budgets"][budget]["mean_log_loss"]
    names = {"no_information": "No information", "all_information": "All information",
             "fixed_order": "Fixed order", "random_exact": "Random (same cap)",
             "random_matched_g_count_descriptive": "Random (matched count)",
             "g": "Full-text + state g"}
    keys = list(stats)
    vals = [stats[k] for k in keys]
    colors = ["#64748b", "#9ca3af", "#d97706", "#a78bfa", "#059669"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar([names[k] for k in keys], vals, color=colors)
    ax.set_ylabel("mean multiclass log loss (lower is better)")
    ax.set_title("End-to-end policy comparison: validation, budget cap = 3\n"
                 "Regression MSE is deliberately not mixed into this plot")
    ax.grid(axis="y", alpha=.22)
    ax.tick_params(axis="x", rotation=18)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    save(fig, out / "03_policy_logloss.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", type=Path, required=True)
    ap.add_argument("--experiment", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fit = load(args.fit)
    report = load(args.experiment / "report.json")
    # selected-predictions.csv is produced by diagnose_postread_fit.py in this directory.
    plot_regularization(fit, args.output)
    plot_scatter(fit, args.predictions, args.output)
    plot_binned_calibration(args.predictions, args.output)
    plot_policy(report, args.output)
    summary = {
        "source_fit_report": str(args.fit),
        "source_policy_report": str(args.experiment / "report.json"),
        "figures": ["01_regularization_path.png", "02_gain_scatter.png", "03_policy_logloss.png", "04_binned_calibration.png"],
        "interpretation": [
            "Strong regularization can underfit train; weak regularization can overfit held-out matches.",
            "The validation scatter, not the training scatter, is the generalization evidence.",
            "Policy log loss is the end-to-end metric and is shown separately from gain regression MSE.",
            "No epoch curve is fabricated because historical training did not save per-iteration traces."
        ]
    }
    (args.output / "plot-manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
