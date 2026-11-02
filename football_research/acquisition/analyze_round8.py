"""Held-out utility and likelihood-ratio mismatch diagnostics for round eight."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from .analyze_round7 import match_means, paired
from .data import load_episodes
from .experiment import outcome, state_after
from .joint_returns import JointReturnPredictor
from .verify_round4 import _rows
from .verify_round8 import branches, close, marginal, probability_checks, rebuild_joint, require, source_categories


def analyze(run, bundle, earlier_bundle, output):
    run, output = Path(run), Path(output)
    require(not output.exists(), "Analysis output already exists")
    report = json.loads((run/"report.json").read_text(encoding="utf-8"))
    design = json.loads((run/"design.json").read_text(encoding="utf-8"))
    actions = json.loads((run/"catalog.json").read_text(encoding="utf-8"))["actions"]
    grouped = match_means(list(_rows(run/"episodes.jsonl")), design["validation_utility_money_weight"])
    metrics, comparisons = {}, {}
    for (strategy, budget), records in sorted(grouped.items()):
        metrics[f"{strategy}/money={budget:g}"] = {"independent_matches": len(records),
            **{name: float(np.mean([r[name] for r in records.values()])) for name in
               ("log_loss", "brier_score", "accuracy", "money", "requests", "utility_loss")}}
        if not strategy.startswith("evsi-"):
            continue
        for (reference, other_budget), previous in sorted(grouped.items()):
            if reference == strategy or other_budget != budget:
                continue
            comparisons[f"{strategy}-vs-{reference}/money={budget:g}"] = {
                name: paired(records, previous, name, design["bootstrap_samples"], design["bootstrap_seed"])
                for name in ("log_loss", "brier_score", "money", "utility_loss")}

    current, _ = load_episodes(bundle, lead_hours=design["lead_hours"])
    earlier, _ = load_episodes(earlier_bundle, lead_hours=design["lead_hours"])
    sources = {r["match_id"]: r for r in earlier+current}
    models = {f["fold"]: JointReturnPredictor.load(run/f["joint_model"]["path"]) for f in report["folds"]}
    initial = list(_rows(run/"initial-return-diagnostics.jsonl"))
    transitions = list(_rows(run/"return-diagnostics.jsonl"))
    seen, max_identity_error = set(), 0.0
    budget = max(design["money_budgets"])
    for record in initial:
        mid, number = record["match_id"], record["fold"]
        require(mid not in seen, "Duplicate initial diagnostic")
        seen.add(mid)
        model, row = models[number], sources[mid]
        cube, _ = rebuild_joint(model, state_after(row, actions, ()))
        token = source_categories(row, model.quantiles)
        require(record["actual_return_categories"] == token, "Initial observed category differs")
        r = model.metadata["results_categories"].index(token["results_view"])
        x = model.metadata["xg_categories"].index(token["xg_view"])
        pz, prior = cube[:, r, x].sum(), marginal(cube)
        y = outcome(row)
        pyz = cube[y, r, x]/prior[y]
        close(record["joint_return_probability"], pz, "Joint observed return density")
        close(record["label_conditional_return_probability"], pyz, "Label conditional density")
        close(record["joint_return_log_loss"], -math.log(pz), "Joint return score")
        close(record["label_conditional_return_log_loss"], -math.log(pyz), "Conditional return score")
        # Bayes identity holds per match, not just for a pooled mean.
        expected_delta = math.log(pz)-math.log(pyz)
        actual_delta = grouped["all-available", budget][mid]["log_loss"] - grouped["base-only", budget][mid]["log_loss"]
        close(actual_delta, expected_delta, "Per-match log likelihood ratio decomposition")
        max_identity_error = max(max_identity_error, abs(actual_delta-expected_delta))
    require(len(seen) == report["test_matches"], "Initial diagnostic coverage differs")
    seen_transitions = set()
    for record in transitions:
        mid, number, aid = record["match_id"], record["fold"], record["action_id"]
        acquired = record["acquired_action_ids"]
        key = mid, tuple(acquired), aid
        require(key not in seen_transitions and mid in seen, "Duplicate or unknown transition")
        seen_transitions.add(key)
        row, model = sources[mid], models[number]
        cube, observed = rebuild_joint(model, state_after(row, actions, acquired))
        diagnostics = probability_checks(cube, [aid])[aid]
        close(record["estimated_evsi"], diagnostics["evsi"], "Model expected gain")
        close(record["conditional_mutual_information"], diagnostics["target_mutual_information"], "Model CMI")
        token = source_categories(row, model.quantiles)[aid]
        vocabulary = model.metadata["results_categories" if aid == "results_view" else "xg_categories"]
        category = vocabulary.index(token)
        _, probability, posterior = next(b for b in branches(cube, aid) if b[0] == category)
        close(record["predicted_return_probability"], probability, "Actual conditional return probability")
        close(record["return_log_loss"], -math.log(probability), "Conditional return log loss")
        y = outcome(row)
        gain = math.log(marginal(posterior)[y])-math.log(marginal(cube)[y])
        close(record["realized_log_loss_gain"], gain, "Actual single-transition gain")
    require(len(seen_transitions) == 4*len(seen), "Missing prespecified legal transition")
    bins = []
    for block in np.array_split(sorted(transitions, key=lambda r:r["estimated_evsi"]), 5):
        values = list(block)
        bins.append({"transitions": len(values), "independent_matches": len({r["match_id"] for r in values}),
            "model_expected_gain": float(np.mean([r["estimated_evsi"] for r in values])),
            "actual_mean_gain": float(np.mean([r["realized_log_loss_gain"] for r in values]))})
    decomposition = {"formula": "LL(posterior,Y)-LL(prior,Y) = NLL(q(Z|Y,H))-NLL(q(Z|H))",
        "independent_matches": len(seen), "max_per_match_identity_error": max_identity_error,
        "marginal_return_nll": float(np.mean([r["joint_return_log_loss"] for r in initial])),
        "label_conditional_return_nll": float(np.mean([r["label_conditional_return_log_loss"] for r in initial])),
        "all_information_minus_base_log_loss": metrics[f"all-available/money={budget:g}"]["log_loss"]-metrics[f"base-only/money={budget:g}"]["log_loss"],
        "interpretation": "Poor outcome-conditioned likelihood ratios can harm predictions even when marginal return density improves."}
    result = {"status": "ANALYZED_EXPLORATORY_NEGATIVE_GAIN", "metrics": metrics,
        "paired_comparisons": comparisons, "likelihood_ratio_decomposition": decomposition,
        "gain_diagnostics": {"transitions": len(transitions), "independent_matches": len(seen),
            "mean_model_expected_gain": float(np.mean([r["estimated_evsi"] for r in transitions])),
            "mean_actual_gain": float(np.mean([r["realized_log_loss_gain"] for r in transitions])), "bins": bins},
        "method": "Seeds averaged within match; equal match weights; sorted UTC-week paired percentile bootstrap",
        "limitations": ["Previously explored 290-match historical season, no independent confirmation",
            "Intervals omit model-fit, adaptive research, and multiplicity uncertainty",
            "Two simulated projections of one cached source response, not real supplier fees",
            "Transition bins share matches; their count does not increase independent sample size",
            "Internal EVSI is model conditional; parameter uncertainty is not included"],
        "source_sha256": {name:sha256((run/name).read_bytes()).hexdigest() for name in
             ("report.json", "episodes.jsonl", "initial-return-diagnostics.jsonl", "return-diagnostics.jsonl")},
        "implementation_sha256": {str(Path(__file__).name):sha256(Path(__file__).read_bytes()).hexdigest(),
             "analyze_round7.py":sha256(Path(__file__).with_name("analyze_round7.py").read_bytes()).hexdigest(),
             "verify_round8.py":sha256(Path(__file__).with_name("verify_round8.py").read_bytes()).hexdigest()}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return result


def plot(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    require(not Path(path).exists(), "Figure already exists")
    names = {"base-only":"Free context only", "all-available":"Acquire all affordable",
        "entropy":"Entropy threshold", "random":"Random (5 seeds/game)",
        "evsi-one-step":"One-step EVSI", "evsi-two-step":"Two-step EVSI"}
    colors = ["#333333", "#997158", "#db962d", "#9272ba", "#398b8b", "#1764ab"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.4))
    for (strategy, label), color in zip(names.items(), colors):
        rows = sorted((float(k.rsplit("=",1)[1]), r) for k,r in result["metrics"].items() if k.split("/")[0] == strategy)
        axes[0].plot([r["money"] for _,r in rows], [r["log_loss"] for _,r in rows], "o-", color=color, label=label, markersize=4)
    axes[0].set(xlabel="Mean simulated acquisition cost", ylabel="Held-out log loss (lower is better)", title="Same prior and joint model for all policies")
    axes[0].legend(fontsize=8)
    refs = ["base-only", "all-available", "entropy", "evsi-one-step"]
    for i, ref in enumerate(refs):
        row = result["paired_comparisons"][f"evsi-two-step-vs-{ref}/money=0.05"]["utility_loss"]
        point = row["mean_difference"]
        low, high = row["confidence_interval_95"]
        axes[1].errorbar(point, i, xerr=[[point-low], [high-point]], fmt="o", color="#1764ab", capsize=4)
    axes[1].axvline(0, color="gray", ls="--", lw=1)
    axes[1].set_yticks(range(len(refs)), ["vs "+names[r] for r in refs])
    axes[1].invert_yaxis()
    axes[1].set(xlabel="Two-step minus reference: LL + 0.5 x cost", title="Utility difference, budget limit 0.05")
    bins = result["gain_diagnostics"]["bins"]
    index = np.arange(len(bins))
    axes[2].bar(index-.18, [r["model_expected_gain"] for r in bins], .36, label="Model expected gain", color="#1764ab")
    axes[2].bar(index+.18, [r["actual_mean_gain"] for r in bins], .36, label="Actual mean gain", color="#cc7650")
    axes[2].axhline(0, color="gray", lw=1)
    axes[2].set_xticks(index, ["1 (low)", "2", "3", "4", "5 (high)"])
    axes[2].set(xlabel="Bins ordered by predicted information value", ylabel="Log-loss reduction (higher is better)", title="Internal value versus observed gain")
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=.15)
        axis.set_axisbelow(True)
    fig.suptitle("FootballMind / Round 8: coherent probabilities do not guarantee useful acquisition", fontsize=15, y=.985)
    fig.text(.02,.028,"290 reused historical matches / 7 temporal folds / 95% paired week-bootstrap intervals (middle)\n"
        "Exploratory. Simulated costs. 1,160 transitions (right) share matches; model uncertainty and repeated research are outside the intervals.",fontsize=9)
    fig.tight_layout(rect=[0,.11,1,.94], w_pad=2.2)
    fig.savefig(path,dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "bundle", "earlier-bundle", "output"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--plot")
    args = parser.parse_args()
    result = analyze(args.run, args.bundle, args.earlier_bundle, args.output)
    if args.plot:
        plot(result, args.plot)
    print(json.dumps({k:result[k] for k in ("status", "likelihood_ratio_decomposition", "gain_diagnostics")}, indent=2))
