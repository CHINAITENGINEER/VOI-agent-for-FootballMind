"""Separate within-model return-content sensitivity from observed policy utility."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from .analyze_round7 import match_means, paired
from .analyze_round11 import behavior_summary
from .verify_round4 import _rows
from .verify_round8 import close, marginal, require, source_categories
from .verify_round13 import CONDITIONAL, STRATEGY, ContentReference, digest, select


def content_contexts(folds, reference, diagnostics, conditional_episodes):
    """All model-supported returns at a fixed prior, set, budget and estimator.

    Original acquired categories are projected retrospectively for parity only.
    Match outcomes and realized gains never enter these branch values/choices.
    """
    indexed = {(r["partition"], r["fold"], r["match_id"], tuple(r["acquired_action_ids"]), r["action_id"]): r
               for r in diagnostics}
    require(len(indexed) == len(diagnostics), "Duplicate frozen transition")
    schedules = {(r["partition"], r["fold"], r["match_id"]): r for r in conditional_episodes
                 if r["strategy"] == CONDITIONAL and r["budget_money"] == .05}
    catalog = {a["action_id"]: a for a in reference.actions}
    output = []
    for fold in folds:
        number = fold["fold"]
        for partition in ("test", "validation"):
            for mid in fold[partition+"_match_ids"]:
                original = reference.sources[mid]
                tokens = source_categories(original, reference.models[number].quantiles)
                episode = schedules[partition, number, mid]
                first_actual = episode["actions"][0] if episode["actions"] else None
                for first in sorted(catalog):
                    remaining = next(a for a in catalog if a != first)
                    remaining_action = catalog[remaining]
                    money = .05-catalog[first]["cost"]["money"]
                    expected = reference.estimate(number, mid, remaining, (first,))
                    erased = select([{**expected, "action": remaining_action}], money=money, requests=1)
                    options = []
                    for branch in expected["branches"]:
                        gain = branch["expected_gain"]
                        choice = select([{"action": remaining_action, "expected_gain": gain}], money=money, requests=1)
                        options.append({**branch, "net_gain": gain-.5*remaining_action["cost"]["money"] if gain is not None else None,
                                        "conditional_next_action": choice, "model_only": True})
                    actual = next(b for b in options if b["observation"] == tokens[first])
                    old = indexed[partition, number, mid, (first,), remaining]
                    close(actual["model_evsi"], old["model_evsi"], "Observed category projects to original EVSI")
                    close(actual["expected_gain"], old["estimates"]["same_version"]["expected_gain"], "Observed category calibrated gain")
                    for a, b in zip(actual["posterior_probabilities"], old["prediction_before"]):
                        close(a, b, "Actual category projects to original posterior")
                    net_values = [b["net_gain"] for b in options if b["net_gain"] is not None]
                    choices = {b["conditional_next_action"] for b in options}
                    disagreement = math.fsum(b["probability"] for b in options if b["conditional_next_action"] != erased)
                    output.append({"fold": number, "partition": partition, "match_id": mid, "kickoff": original["kickoff"],
                        "free_prior_probabilities": marginal(reference.base_cube(number, mid)).tolist(),
                        "acquired_action_ids": [first], "remaining_action_id": remaining,
                        "remaining_budget_money": money, "remaining_requests": 1,
                        "marginal_expected_gain": expected["expected_gain"],
                        "marginal_net_gain": expected["expected_gain"]-.5*remaining_action["cost"]["money"] if expected["supported"] else None,
                        "marginal_next_action": erased, "branches": options,
                        "conditional_net_gain_min": min(net_values) if net_values else None,
                        "conditional_net_gain_max": max(net_values) if net_values else None,
                        "conditional_net_gain_range": max(net_values)-min(net_values) if net_values else None,
                        "within_context_decision_switch": len(choices) > 1,
                        "model_probability_weighted_disagreement": disagreement,
                        "actual_category": tokens[first], "actual_category_model_probability": actual["probability"],
                        "actual_conditional_next_action": actual["conditional_next_action"],
                        "actual_category_disagreement": actual["conditional_next_action"] != erased,
                        "frozen_conditional_first_action": first_actual,
                        "reached_after_frozen_conditional_first_query": first == first_actual,
                        "scope": "Model-only categorical alternatives at one fixed prior/mask/budget; actual category is retrospective parity, no alternative outcome label"})
    return output


def context_summary(rows):
    def group(items):
        values = [r["conditional_net_gain_range"] for r in items if r["conditional_net_gain_range"] is not None]
        return {"contexts": len(items), "matches": len({(r["fold"], r["match_id"]) for r in items}),
                "model_category_branches": sum(len(r["branches"]) for r in items),
                "contexts_with_different_numeric_value": sum(v > 1e-12 for v in values),
                "contexts_with_query_stop_switch": sum(r["within_context_decision_switch"] for r in items),
                "actual_category_policy_disagreements": sum(r["actual_category_disagreement"] for r in items),
                "mean_model_probability_weighted_disagreement": float(np.mean([r["model_probability_weighted_disagreement"] for r in items])) if items else None,
                "mean_conditional_net_gain_range": float(np.mean(values)) if values else None,
                "max_conditional_net_gain_range": max(values) if values else None,
                "conditional_actual_next_actions": dict(Counter(r["actual_conditional_next_action"] or "STOP" for r in items)),
                "marginal_next_actions": dict(Counter(r["marginal_next_action"] or "STOP" for r in items))}
    output = {}
    for partition in ("test", "validation"):
        selected = [r for r in rows if r["partition"] == partition]
        output[partition] = {"all_fixed_prior_contexts": group(selected),
            "reached_after_actual_first_query": group([r for r in selected if r["reached_after_frozen_conditional_first_query"]]),
            "by_first_action": {first: group([r for r in selected if r["acquired_action_ids"] == [first]])
                                for first in ("results_view", "xg_view")}}
    return output


def observed_policy_pairs(new_rows, old_rows):
    old = {(r["fold"], r["match_id"], r["budget_money"]): r for r in old_rows if r["strategy"] == CONDITIONAL}
    result = []
    for row in new_rows:
        previous = old[row["fold"], row["match_id"], row["budget_money"]]
        before = previous["actions"][0] if previous["actions"] else None
        after = row["actions"][0] if row["actions"] else None
        require(before == after, "Content erasure changed the initial decision")
        if row["budget_money"] < .05 or before is None:
            require(previous["actions"] == row["actions"] and previous["cost"] == row["cost"]
                    and previous["final_probabilities"] == row["final_probabilities"], "Inaccessible continuation changed an outcome")
        result.append({"fold": row["fold"], "match_id": row["match_id"], "kickoff": row["kickoff"], "budget_money": row["budget_money"],
                       "initial_action": before, "conditional_actions": previous["actions"], "marginal_actions": row["actions"],
                       "path_differs": previous["actions"] != row["actions"],
                       "conditional_minus_marginal_log_loss": previous["log_loss"]-row["log_loss"],
                       "conditional_minus_marginal_money": previous["cost"]["money"]-row["cost"]["money"],
                       "conditional_minus_marginal_utility_loss": previous["log_loss"]-row["log_loss"]+.5*(previous["cost"]["money"]-row["cost"]["money"]),
                       "scope": "Two fixed policy replays with actual revealed historical returns and one observed match outcome"})
    return result


def analyze(run, bundle, earlier_bundle, output):
    from .joint_returns import JointReturnPredictor
    from .round9_residual import ResidualEVSICalibrator
    from .data import load_episodes

    run, output = Path(run), Path(output)
    require(not output.exists(), "Analysis output already exists")
    report = json.loads((run/"report.json").read_bytes())
    design = json.loads((run/"design.json").read_bytes())
    source = Path(design["source_round11"])
    source12 = Path(design["source_round12"])
    original = json.loads((source/"report.json").read_bytes())
    sources = {r["match_id"]: r for path in (bundle, earlier_bundle) for r in load_episodes(path, lead_hours=24)[0]}
    actions = json.loads((run/"catalog.json").read_bytes())["actions"]
    models, calibrators = {}, {}
    for fold in original["folds"]:
        n = fold["fold"]
        path = source/fold["joint_model"]["path"]
        require(digest(path) == fold["joint_model"]["sha256"], "Frozen joint changed")
        model = models[n] = JointReturnPredictor.load(path)
        path_cal = source/fold["calibrators"]["same_version"]["path"]
        require(digest(path_cal) == fold["calibrators"]["same_version"]["sha256"], "Frozen calibrator changed")
        calibrators[n] = ResidualEVSICalibrator.load(path_cal, model=model, model_sha256=digest(path))
    reference = ContentReference(sources, actions, models, calibrators)
    old_test = list(_rows(source/"episodes.jsonl"))
    old_validation = list(_rows(source/"validation-episodes.jsonl"))
    contexts = content_contexts(original["folds"], reference, list(_rows(source/"transition-diagnostics.jsonl")), old_test+old_validation)
    test, validation = list(_rows(run/"episodes.jsonl")), list(_rows(run/"validation-episodes.jsonl"))
    policy_pairs = {"test": observed_policy_pairs(test, old_test), "validation": observed_policy_pairs(validation, old_validation)}
    combined = old_test+list(_rows(source12/"episodes.jsonl"))+test
    means = match_means(combined, .5)
    metrics, behaviors = {}, {}
    for (strategy, budget), entries in sorted(means.items()):
        key = f"{strategy}/money={budget:g}"
        metrics[key] = {"independent_matches": len(entries), **{name: float(np.mean([r[name] for r in entries.values()]))
                       for name in ("log_loss", "brier_score", "accuracy", "money", "requests", "utility_loss")}}
        for name, value in metrics[key].items():
            close(value, report["metrics"][key][name], "Combined metric "+name)
        if strategy in (CONDITIONAL, STRATEGY, "base-only", "stage-prior-mean", "tool-prior-mean"):
            behaviors[key] = behavior_summary([r for r in combined if r["strategy"] == strategy and r["budget_money"] == budget])
    comparisons = {}
    for label, expected in report["paired_comparisons"].items():
        pair, budget = label.rsplit("/money=", 1)
        a, b = pair.split("-vs-")
        comparisons[label] = {metric: paired(means[a, float(budget)], means[b, float(budget)], metric, 2000, 1729)
                              for metric in ("log_loss", "money", "utility_loss")}
        require(comparisons[label] == expected, "Paired result differs")
    pair_summary = {}
    for partition, pairs in policy_pairs.items():
        pair_summary[partition] = {}
        for budget in design["money_budgets"]:
            subset = [r for r in pairs if r["budget_money"] == budget]
            pair_summary[partition][f"money={budget:g}"] = {"matches": len(subset), "initial_action_disagreements": 0,
                "path_disagreements": sum(r["path_differs"] for r in subset),
                "initial_stop_matches": sum(r["initial_action"] is None for r in subset),
                "conditional_queries": sum(len(r["conditional_actions"]) for r in subset),
                "marginal_queries": sum(len(r["marginal_actions"]) for r in subset),
                "mean_utility_difference": float(np.mean([r["conditional_minus_marginal_utility_loss"] for r in subset]))}
    result = {"status": "ANALYZED_EXPLORATORY_CONTENT_ABLATION", "round": 13,
        "metrics": metrics, "paired_comparisons": comparisons, "policy_behavior": behaviors,
        "policy_pair_summary": pair_summary, "policy_pairs": policy_pairs,
        "content_context_summary": context_summary(contexts), "content_contexts": contexts,
        "independently_rebuilt_free_joint_cubes": len(reference.cubes),
        "new_independent_games": 0, "new_test_policy_replays": len(test), "new_validation_policy_replays": len(validation),
        "source_report_sha256": digest(run/"report.json"), "analyzer_sha256": digest(__file__),
        "limitations": ["Finite fitted-model categories are not alternative observed football events",
            "All fixed-prior contexts include unvisited histories; report actual-path subset separately",
            "Erasure affects query valuation only; all executed forecasts update from actual returns",
            "This reused-history ablation has no new independent matches or online randomized evidence",
            "Averaging uses the possibly misspecified frozen return model; results are specific to this control",
            "Week intervals omit fitting, repeated research, multiplicity and distribution shift"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return result


def plot(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    require(not output.exists(), "Figure already exists")
    figure, axes = plt.subplots(2, 2, figsize=(15, 10.4), constrained_layout=True)
    contexts = [r for r in result["content_contexts"] if r["partition"] == "test"]
    colors = {"results_view": "#235d9f", "xg_view": "#bd631b"}
    for first, color in colors.items():
        selected = [r for r in contexts if r["acquired_action_ids"] == [first]]
        axes[0, 0].scatter([r["marginal_net_gain"] for r in selected], [r["conditional_net_gain_range"] for r in selected],
                         s=19, alpha=.60, color=color, label="First return: "+first.replace("_view", ""))
    axes[0, 0].axvline(0, color="grey", linewidth=.8, linestyle="--")
    axes[0, 0].set(xlabel="Content-erased next-query net value", ylabel="Conditional net-value range across return categories",
                   title="A. Same match prior, acquired set and budget")
    axes[0, 0].legend(loc="best", fontsize=9)
    labels, totals, switches, disagree = [], [], [], []
    summary = result["content_context_summary"]["test"]
    for label, key in (("All modeled contexts", "all_fixed_prior_contexts"), ("After actual first query", "reached_after_actual_first_query")):
        values = summary[key]
        labels.append(label+f"\n(n={values['contexts']} contexts)")
        totals.append(values["contexts"])
        switches.append(values["contexts_with_query_stop_switch"])
        disagree.append(values["actual_category_policy_disagreements"])
    x = np.arange(2)
    axes[0, 1].bar(x-.17, switches, width=.34, color="#235d9f", label="Some modeled categories switch QUERY / STOP")
    axes[0, 1].bar(x+.17, disagree, width=.34, color="#bd631b", label="Observed category disagrees with erased policy")
    for xpos, heights in ((x-.17, switches), (x+.17, disagree)):
        for place, height in zip(xpos, heights):
            axes[0, 1].text(place, height+.5, str(height), ha="center", va="bottom", fontsize=10)
    axes[0, 1].set(xticks=x, xticklabels=labels, ylabel="Context count", title="B. Numerical sensitivity versus changed decisions")
    axes[0, 1].set_ylim(0, max(switches+disagree+[1])*1.3+2)
    axes[0, 1].legend(fontsize=8, loc="upper left")
    names = [("base-only", "Free only"), ("tool-prior-mean", "Tool mean"), ("stage-prior-mean", "Stage mean"),
             (STRATEGY, "Content erased"), (CONDITIONAL, "Conditional")]
    values = [result["metrics"][name+"/money=0.05"] for name, _ in names]
    axes[1, 0].barh([label for _, label in names], [r["utility_loss"] for r in values],
                    color=["#888888", "#a5afb8", "#7796b4", "#bd631b", "#235d9f"])
    xmin = min(r["utility_loss"] for r in values)-.01
    xmax = max(r["utility_loss"] for r in values)+.016
    axes[1, 0].set(xlim=(xmin, xmax), xlabel="Mean log loss + 0.5 x simulated cost (axis truncated; lower is better)",
                   title="C. Actual historical policy replay, budget 0.05")
    for i, item in enumerate(values):
        axes[1, 0].text(item["utility_loss"]+.0004, i, f"{item['utility_loss']:.6f}", va="center", fontsize=9)
    key = CONDITIONAL+"-vs-"+STRATEGY+"/money=0.05"
    primary = result["paired_comparisons"][key]["utility_loss"]
    low, high = primary["confidence_interval_95"]
    estimate = primary["mean_difference"]
    axes[1, 1].errorbar([estimate], [0], xerr=[[estimate-low], [high-estimate]], fmt="o", capsize=6, color="#235d9f")
    axes[1, 1].axvline(0, linestyle="--", color="grey", linewidth=1)
    axes[1, 1].set(yticks=[0], yticklabels=["Conditional minus\ncontent-erased"], ylim=(-.8, .8),
                   xlabel="Paired utility-loss difference; negative favors conditional",
                   title="D. Prespecified main comparison, 200 reused games")
    axes[1, 1].text(.03, .10, f"Mean {estimate:+.6f}\n95% descriptive UTC-week interval [{low:+.6f}, {high:+.6f}]",
                   transform=axes[1, 1].transAxes, fontsize=10)
    for axis in axes.flat:
        axis.grid(axis="x", alpha=.15)
    figure.suptitle("FootballMind round 13: return contents, autonomous query selection and stopping\n"
                    "A/B: fitted-model branch diagnostic. C/D: real-return replay. Same explored history; no independent confirmation.", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("run", "bundle", "earlier-bundle", "output"):
        parser.add_argument("--"+key, required=True)
    parser.add_argument("--figure")
    args = parser.parse_args()
    result = analyze(args.run, args.bundle, args.earlier_bundle, args.output)
    if args.figure:
        plot(result, args.figure)
    print(json.dumps({"status": result["status"], "content_context_summary": result["content_context_summary"],
                      "policy_pair_summary": result["policy_pair_summary"]}, ensure_ascii=True))
