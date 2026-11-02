"""Match-weighted exploratory comparisons and a standalone scientific figure."""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .state import parse_time
from .verify_round4 import _rows


def match_means(rows, weight):
    """Average randomized repeats inside each match before statistical comparison."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["strategy"], row["budget_money"], row["match_id"])].append(row)
    output = defaultdict(dict)
    for (strategy, budget, mid), entries in grouped.items():
        if len({r["seed"] for r in entries}) != len(entries):
            raise ValueError("Duplicate random seed for a match")
        if len({r["kickoff"] for r in entries}) != 1:
            raise ValueError("Inconsistent match kickoff")
        mean = {k: float(np.mean([r[k] for r in entries])) for k in ("log_loss", "brier_score", "accuracy")}
        mean.update({k: float(np.mean([r["cost"][k] for r in entries])) for k in ("money", "requests")})
        mean.update(kickoff=entries[0]["kickoff"], repeats=len(entries), utility_loss=mean["log_loss"] + weight * mean["money"])
        output[strategy, budget][mid] = mean
    return output


def paired(current, reference, metric, samples, seed):
    if current.keys() != reference.keys() or not current:
        raise ValueError("Paired statistics require identical nonempty match sets")
    blocks = defaultdict(list)
    for mid in sorted(current):
        a, b = current[mid], reference[mid]
        if parse_time(a["kickoff"]) != parse_time(b["kickoff"]):
            raise ValueError("Paired games have different kickoff")
        year, week, _ = parse_time(a["kickoff"]).isocalendar()
        blocks[year, week].append(a[metric] - b[metric])
    values = [blocks[k] for k in sorted(blocks)]
    sums, counts = np.array([sum(v) for v in values]), np.array([len(v) for v in values])
    indices = np.random.default_rng(seed).integers(0, len(values), size=(samples, len(values)))
    draws = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    return {"mean_difference": float(sums.sum() / counts.sum()),
            "confidence_interval_95": np.quantile(draws, [.025, .975]).tolist(),
            "independent_matches": int(counts.sum()), "weeks": len(values)}


def analyze(run, output, reference=None):
    run, output = Path(run), Path(output)
    if output.exists():
        raise FileExistsError(output)
    design = json.loads((run / "design.json").read_text(encoding="utf-8"))
    grouped = match_means(list(_rows(run / "episodes.jsonl")), design["validation_utility_money_weight"])
    metrics, comparisons = {}, {}
    for (strategy, budget), records in sorted(grouped.items()):
        metrics[f"{strategy}/money={budget:g}"] = {"independent_matches": len(records),
            **{m: float(np.mean([r[m] for r in records.values()]))
               for m in ("log_loss", "brier_score", "accuracy", "money", "requests", "utility_loss")}}
        if strategy != "contextual-source-identity":
            continue
        for (other, other_budget), previous in sorted(grouped.items()):
            if budget != other_budget or strategy == other:
                continue
            comparisons[f"{strategy}-vs-{other}/money={budget:g}"] = {
                name: paired(records, previous, name, design["bootstrap_samples"], design["bootstrap_seed"])
                for name in ("log_loss", "brier_score", "money", "utility_loss")}
    decomposition = None
    reference_hash = None
    if reference is not None:
        reference = Path(reference)
        previous = match_means(list(_rows(reference / "episodes.jsonl")), design["validation_utility_money_weight"])
        budget = max(design["money_budgets"])
        decomposition = {"comparison": "new context-only base minus previous league-prior base; neither acquires evidence",
            "log_loss": paired(grouped["base-only", budget], previous["base-only", budget], "log_loss",
                               design["bootstrap_samples"], design["bootstrap_seed"]),
            "previous_base_log_loss": float(np.mean([r["log_loss"] for r in previous["base-only", budget].values()])),
            "new_base_log_loss": metrics[f"base-only/money={budget:g}"]["log_loss"],
            "scope": "outcome predictor family/context changes jointly; this is not an acquisition-policy gain or causal decomposition"}
        reference_hash = sha256((reference / "episodes.jsonl").read_bytes()).hexdigest()
    result = {"status": "ANALYZED_EXPLORATORY", "objective": "log_loss + frozen validation money weight * simulated cost",
        "money_weight": design["validation_utility_money_weight"], "metrics": metrics,
        "paired_comparisons": comparisons, "base_predictor_change": decomposition,
        "method": "average seeds inside each match; paired UTC-week percentile bootstrap",
        "limitations": ["Same already explored test season", "Equal budget limits do not imply equal realized cost",
                       "Intervals exclude model-fitting, repeated-testing and multiplicity uncertainty",
                       "No declared noninferiority margin; interval containing zero does not establish equivalence",
                       "Only two projections from a frozen provider response; costs are simulated"],
        "episode_sha256": sha256((run / "episodes.jsonl").read_bytes()).hexdigest(),
        "reference_episode_sha256": reference_hash,
        "analyzer_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def plot(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if Path(output).exists():
        raise FileExistsError(output)
    names = {"base-only": "Free context only", "all-available": "Acquire all affordable", "entropy": "Entropy threshold",
             "fixed-order": "Fixed order", "random": "Random (5 seeds/game)", "contextual-no-context": "Unconditional value",
             "contextual-prediction-only": "Value from current prediction", "contextual-source-identity": "Conditional information value"}
    colors = {"base-only": "#555555", "all-available": "#8d6e63", "entropy": "#ff9800", "fixed-order": "#9e9e9e",
              "random": "#ab47bc", "contextual-no-context": "#26a69a", "contextual-prediction-only": "#42a5f5",
              "contextual-source-identity": "#174ea6"}
    figure, axes = plt.subplots(1, 2, figsize=(14.8, 6.5), gridspec_kw={"width_ratios": [1.05, 1]})
    for strategy in names:
        values = [(float(k.rsplit("=", 1)[1]), v) for k, v in result["metrics"].items() if k.split("/")[0] == strategy]
        values.sort()
        if strategy == "fixed-order":
            continue  # Numerically identical to the all-affordable catalog order in this design.
        axes[0].plot([v[1]["money"] for v in values], [v[1]["log_loss"] for v in values],
                     "o-", label=names[strategy], color=colors[strategy], markersize=5, linewidth=1.5)
    axes[0].set(xlabel="Mean simulated acquisition cost", ylabel="Mean test log loss (lower is better)",
                title="All policies use the same free-context predictor")
    axes[0].legend(fontsize=8.3, loc="best")
    axes[0].grid(alpha=.2)
    refs = ["base-only", "all-available", "entropy", "random", "contextual-no-context", "contextual-prediction-only"]
    budget = max(float(k.rsplit("=", 1)[1]) for k in result["metrics"])
    for i, ref in enumerate(refs):
        record = result["paired_comparisons"][f"contextual-source-identity-vs-{ref}/money={budget:g}"]["utility_loss"]
        point = record["mean_difference"]
        low, high = record["confidence_interval_95"]
        axes[1].errorbar(point, i, xerr=[[point - low], [high - point]], fmt="o", color="#174ea6", capsize=4)
    axes[1].axvline(0, color="#777777", linestyle="--", linewidth=1)
    axes[1].set_yticks(range(len(refs)), ["vs " + names[r] for r in refs])
    axes[1].invert_yaxis()
    axes[1].set(xlabel="Conditional strategy minus reference: log loss + 0.5 × cost",
                title=f"Paired utility differences, budget limit {budget:g}")
    axes[1].grid(axis="x", alpha=.2)
    figure.suptitle("FootballMind • Round 7: incremental value after equal free information", fontsize=15, y=.98)
    figure.text(.02, .035, "290 reused historical test matches • 7 temporal folds • 95% paired week-bootstrap intervals\n"
                "Exploratory; simulated costs; no noninferiority or independent confirmation claim. Random seeds are averaged per match.", fontsize=9)
    figure.tight_layout(rect=[0, .105, 1, .935], w_pad=3)
    figure.savefig(output, dpi=175)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference")
    parser.add_argument("--plot")
    args = parser.parse_args()
    result = analyze(args.run, args.output, args.reference)
    if args.plot:
        plot(result, args.plot)
    print(json.dumps({"status": result["status"], "base_predictor_change": result["base_predictor_change"]}))
