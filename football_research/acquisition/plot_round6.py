"""Standalone descriptive figures for the exploratory contextual experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def plot(run):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = Path(run)
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    labels = {"base-only": "No acquisition", "all-available": "Acquire all affordable",
              "entropy": "Entropy threshold", "risk-mean": "Round 5 value model",
              "contextual-no-context": "No context",
              "contextual-prediction-only": "Current prediction only",
              "contextual-visible-context": "Visible match context",
              "contextual-source-identity": "Source-ID aligned context (sensitivity)"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for strategy, label in labels.items():
        metrics = [row for key, row in report["metrics"].items() if key.startswith(strategy + "/")]
        if not metrics:
            continue
        metrics.sort(key=lambda row: row["money"])
        axes[0].plot([m["money"] for m in metrics], [m["log_loss"] for m in metrics], "o-", label=label, alpha=.85)
    axes[0].set(xlabel="Mean simulated acquisition cost", ylabel="Mean log loss (lower is better)",
                title="Same matches and budget options; descriptive points")
    axes[0].legend(fontsize=8)
    folds = report["folds"]
    for mode, marker in (("no-context", "o"), ("prediction-only", "x"), ("visible-context", "s"), ("source-identity", "^")):
        if not all(mode in f["initial_value_diagnostics"] for f in folds):
            continue
        axes[1].plot([f["fold"] for f in folds],
                     [f["initial_value_diagnostics"][mode]["distinct_gain_vectors_rounded_12dp"] for f in folds],
                     marker + "-", label=labels["contextual-" + mode])
    axes[1].plot([f["fold"] for f in folds], [len(f["test_match_ids"]) for f in folds], "k:", label="Matches in fold")
    axes[1].set(xlabel="Chronological fold", ylabel="Distinct initial value vectors", title="Variation is capacity, not proof of benefit")
    axes[1].legend(fontsize=8)
    fig.suptitle("Round 6: 290 previously examined matches; exploratory evidence")
    fig.tight_layout()
    path = run / "contextual-cost-and-identifiability.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    print(plot(parser.parse_args().run).resolve())


if __name__ == "__main__":
    main()
