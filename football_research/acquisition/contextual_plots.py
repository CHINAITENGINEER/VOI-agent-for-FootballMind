"""Render fixed round-six report figures; performs no model selection."""
import argparse
import json
from pathlib import Path


def render(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(directory)
    report = json.loads((root/"report.json").read_text(encoding="utf-8"))
    strategies = ("base-only", "all-available", "entropy", "risk-mean", "contextual-no-context",
                  "contextual-prediction-only", "contextual-visible-context")
    if "contextual-source-identity/money=0.05" in report["metrics"]:
        strategies += ("contextual-source-identity",)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for strategy in strategies:
        rows = sorted((v for k, v in report["metrics"].items() if k.startswith(strategy+"/")), key=lambda r: r["money"])
        for axis, metric in zip(axes, ("log_loss", "brier_score")):
            axis.plot([r["money"] for r in rows], [r[metric] for r in rows], "o-", label=strategy)
            axis.set(xlabel="Mean simulated acquisition cost", ylabel=metric)
            axis.grid(alpha=.2)
    axes[1].legend(fontsize=7)
    fig.suptitle("Round 6: exploratory replay of 290 previously examined matches")
    fig.tight_layout()
    fig.savefig(root/"contextual-cost-quality.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    folds = report["folds"]
    modes = ("no-context", "prediction-only", "visible-context")
    if "source-identity" in folds[0]["initial_value_diagnostics"]:
        modes += ("source-identity",)
    for mode in modes:
        axes[0].plot([f["fold"] for f in folds], [f["initial_value_diagnostics"][mode]["distinct_gain_vectors_rounded_12dp"] for f in folds], "o-", label=mode)
    axes[0].set(xlabel="Temporal fold", ylabel="Distinct initial candidate-value vectors")
    axes[0].legend(fontsize=8)
    for mode in ("visible-context", "source-identity") if "source-identity" in modes else ("visible-context",):
        axes[1].plot([f["fold"] for f in folds], [f["initial_value_diagnostics"][mode]["unknown_team_fraction"] for f in folds], "o-", label=mode)
    axes[1].legend(fontsize=8)
    axes[1].set(xlabel="Temporal fold", ylabel="Unknown team-role fraction", ylim=(0, 1))
    fig.suptitle("Value variation and identity coverage are separate from predictive benefit")
    fig.tight_layout()
    fig.savefig(root/"contextual-identifiability.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    render(parser.parse_args().run)
