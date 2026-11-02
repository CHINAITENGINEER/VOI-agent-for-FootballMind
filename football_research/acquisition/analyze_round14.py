"""Calibrated continuation opportunity, reachable paths and realized utility."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import numpy as np

from .analyze_round7 import match_means, paired
from .analyze_round11 import behavior_summary
from .experiment import losses, outcome
from .verify_round4 import _rows
from .verify_round8 import close, require
from .verify_round14 import MODES, REFERENCES, load_reference, digest


def initial_opportunities(folds, reference, episodes):
    indexed = {(r["partition"], r["fold"], r["match_id"], r["strategy"]): r for r in episodes if r["budget_money"] == .05}
    rows = []
    for fold in folds:
        n = fold["fold"]
        for partition in ("test", "validation"):
            for mid in fold[partition+"_match_ids"]:
                plans = {mode: reference.plan(n, mid, (), .05, mode) for mode in MODES.values()}
                policies = {mode: indexed[partition, n, mid, strategy] for strategy, mode in MODES.items()}
                for mode, plan in plans.items():
                    require(plan["action_id"] == (policies[mode]["actions"][0] if policies[mode]["actions"] else None), "Initial analysis/replay choice differs")
                old = indexed[partition, n, mid, REFERENCES["adaptive"]]
                old_action = old["actions"][0] if old["actions"] else None
                candidates = {}
                for c in plans["adaptive"]["candidates"]:
                    other = next(x for x in plans["commit"]["candidates"] if x["action_id"] == c["action_id"])
                    close(c["immediate_expected_gain"], other["immediate_expected_gain"], "Modes differ in first-query gain")
                    candidates[c["action_id"]] = {"immediate_expected_gain": c["immediate_expected_gain"],
                        "immediate_net_gain": c["immediate_net_gain"], "adaptive_bonus": c["adaptive_bonus"],
                        "commit_bonus": c["commit_bonus"], "surrogate_option_gap": c["surrogate_option_gap"],
                        "adaptive_Q": c["mean_net_gain"], "commit_Q": other["mean_net_gain"],
                        "adaptive_expected_gain": c["expected_gain"], "commit_expected_gain": other["expected_gain"],
                        "adaptive_expected_cost": c["expected_plan_cost_vector"], "commit_expected_cost": other["expected_plan_cost_vector"],
                        "adaptive_continuation_probability": sum(b["probability"] for b in c["branches"] if b["continuation_action"]),
                        "commit_continuation_probability": sum(b["probability"] for b in other["branches"] if b["continuation_action"]),
                        "adaptive_selected": plans["adaptive"]["action_id"] == c["action_id"],
                        "commit_selected": plans["commit"]["action_id"] == c["action_id"],
                        "content_contingent_future": len({b["continuation_action"] for b in c["branches"]}) > 1,
                        "adaptive_branches": c["branches"], "commit_branches": other["branches"]}
                margins = {}
                for mode in MODES.values():
                    scores = {aid: c[mode+"_Q"] for aid, c in candidates.items()}
                    kept_score = scores[old_action] if old_action else 0.
                    alternatives = [v for aid, v in scores.items() if aid != old_action]
                    if old_action:
                        alternatives.append(0.)
                    margins[mode] = kept_score-max(alternatives)
                rows.append({"partition": partition, "fold": n, "match_id": mid, "kickoff": old["kickoff"],
                    "one_step_first_action": old_action, "adaptive_first_action": plans["adaptive"]["action_id"],
                    "commit_first_action": plans["commit"]["action_id"], "candidates": candidates,
                    "old_action_margin_against_best_alternative": margins,
                    "scope": "Independent fitted-score opportunity analysis; no actual match outcome used"})
    return rows


def summarize_opportunities(rows):
    result = {}
    for part in ("test", "validation"):
        selected = [r for r in rows if r["partition"] == part]
        candidates = [c for r in selected for c in r["candidates"].values()]
        actual = [c for c in candidates if c["adaptive_selected"]]
        result[part] = {"matches": len(selected), "initial_candidates": len(candidates),
            "positive_adaptive_bonus_candidates": sum(c["adaptive_bonus"] > 1e-12 for c in candidates),
            "positive_commit_bonus_candidates": sum(c["commit_bonus"] > 1e-12 for c in candidates),
            "positive_option_gap_candidates": sum(c["surrogate_option_gap"] > 1e-12 for c in candidates),
            "max_adaptive_bonus": max(c["adaptive_bonus"] for c in candidates),
            "max_surrogate_option_gap": max(c["surrogate_option_gap"] for c in candidates),
            "negative_immediate_candidates_rescued_to_positive_Q": sum(c["immediate_net_gain"] < 0 and round(c["adaptive_Q"], 12) > 0 for c in candidates),
            "selected_first_queries": len(actual), "selected_with_positive_bonus": sum(c["adaptive_bonus"] > 1e-12 for c in actual),
            "selected_negative_immediate_net": sum(c["immediate_net_gain"] < 0 for c in actual),
            "max_selected_adaptive_bonus": max((c["adaptive_bonus"] for c in actual), default=0.),
            "initial_adaptive_vs_commit_changes": sum(r["adaptive_first_action"] != r["commit_first_action"] for r in selected),
            "adaptive_vs_one_step_initial_changes": sum(r["adaptive_first_action"] != r["one_step_first_action"] for r in selected),
            "commit_vs_one_step_initial_changes": sum(r["commit_first_action"] != r["one_step_first_action"] for r in selected),
            "old_action_minimum_remaining_margin": {mode: min(r["old_action_margin_against_best_alternative"][mode] for r in selected) for mode in MODES.values()},
            "model_expected_second_queries": {mode: math.fsum(c[mode+"_continuation_probability"] for c in candidates if c[mode+"_selected"]) for mode in MODES.values()}}
    return result


def realized_accounting(run, report, reference, episodes, opportunities):
    """Use only actual executed returns; labels enter after policy paths are fixed."""
    frozen = list(_rows(Path(report["source_round11"])/"transition-diagnostics.jsonl"))
    truth = {(r["partition"], r["fold"], r["match_id"], tuple(r["acquired_action_ids"]), r["action_id"]): r for r in frozen}
    indexed = {(r["partition"], r["fold"], r["match_id"], r["strategy"], r["budget_money"]): r for r in episodes}
    groups = defaultdict(list)
    for record in _rows(Path(run)/"realized-transitions.jsonl"):
        key = record["partition"], record["fold"], record["match_id"], record["strategy"], record["budget_money"]
        episode = indexed[key]
        step = record["step"]
        require(step < len(episode["actions"]) and episode["actions"][step] == record["action_id"], "Retrospective gain is not on actual path")
        acquired = tuple(sorted(episode["actions"][:step]))
        expected = truth[record["partition"], record["fold"], record["match_id"], acquired, record["action_id"]]
        close(record["realized_gain"], expected["actual_gain"], "Frozen actual-return gain")
        for name, source_name in (("probabilities_before", "prediction_before"), ("probabilities_after", "prediction_after")):
            for a, b in zip(record[name], expected[source_name]):
                close(a, b, "Frozen retrospective probability")
        action = next(a for a in reference.actions if a["action_id"] == record["action_id"])
        require(record["actual_cost"] == action["cost"], "Projected cost substituted for actual charge")
        close(record["realized_net_gain"], expected["actual_gain"]-.5*action["cost"]["money"], "Actual query net gain")
        groups[key].append(record)
    initial = {(r["partition"], r["fold"], r["match_id"]): r for r in opportunities}
    totals = defaultdict(list)
    for key, episode in indexed.items():
        records = sorted(groups[key], key=lambda r: r["step"])
        require(len(records) == len(episode["actions"]) and [r["step"] for r in records] == list(range(len(records))), "Missing/duplicate actual transition")
        source = reference.sources[episode["match_id"]]
        base_loss = losses(episode["initial_probabilities"], outcome(source))["log_loss"]
        gain = math.fsum(r["realized_gain"] for r in records)
        net = math.fsum(r["realized_net_gain"] for r in records)
        close(gain, base_loss-episode["log_loss"], "Actual per-query gains telescope")
        close(net, base_loss-episode["log_loss"]-.5*episode["cost"]["money"], "Actual costs telescope")
        if episode["budget_money"] != .05:
            continue
        opportunity = initial[episode["partition"], episode["fold"], episode["match_id"]]
        mode = MODES[episode["strategy"]]
        first = episode["actions"][0] if episode["actions"] else None
        candidate = opportunity["candidates"][first] if first else None
        totals[episode["partition"], episode["strategy"]].append({
            "match_id": episode["match_id"], "queries": len(records), "actual_gain": gain, "actual_net_gain": net,
            "first_actual_gain": records[0]["realized_gain"] if records else 0.,
            "second_actual_gain": records[1]["realized_gain"] if len(records) > 1 else 0.,
            "predicted_plan_gain": candidate[mode+"_expected_gain"] if candidate else 0.,
            "predicted_plan_net_gain": candidate[mode+"_Q"] if candidate else 0.,
            "expected_plan_money": candidate[mode+"_expected_cost"]["money"] if candidate else 0.,
            "actual_money": episode["cost"]["money"]})
    return {"actual_transitions_verified": sum(len(v) for v in groups.values()), "all_episode_telescoping_checks": len(indexed),
        "primary_budget": {part+"/"+strategy: {"matches": len(rows), "actual_queries": sum(r["queries"] for r in rows),
            "actual_second_queries": sum(r["queries"] > 1 for r in rows),
            **{"mean_"+key: float(np.mean([r[key] for r in rows])) for key in
               ("actual_gain", "actual_net_gain", "first_actual_gain", "second_actual_gain", "predicted_plan_gain", "predicted_plan_net_gain", "expected_plan_money", "actual_money")}}
            for (part, strategy), rows in totals.items()},
        "scope": "Actual executed transitions only; chosen-plan predictions are model scores, not guarantees or alternative observed outcomes"}


def analyze(run, bundle, earlier_bundle, output):
    run, output = Path(run), Path(output)
    require(not output.exists(), "Analysis output exists")
    reference, _, report = load_reference(run, bundle, earlier_bundle)
    old_test, old_validation = [], []
    for i in (11, 12, 13):
        root = Path(report[f"source_round{i}"])
        old_test += list(_rows(root/"episodes.jsonl"))
        old_validation += list(_rows(root/"validation-episodes.jsonl"))
    test = list(_rows(run/"episodes.jsonl"))
    validation = list(_rows(run/"validation-episodes.jsonl"))
    opportunities = initial_opportunities(report["folds"], reference, old_test+old_validation+test+validation)
    accounting = realized_accounting(run, report, reference, test+validation, opportunities)
    combined, means = old_test+test, match_means(old_test+test, .5)
    metrics, behaviors = {}, {}
    for (strategy, budget), rows in sorted(means.items()):
        key = f"{strategy}/money={budget:g}"
        metrics[key] = {"independent_matches": len(rows), **{m: float(np.mean([r[m] for r in rows.values()]))
                       for m in ("log_loss", "brier_score", "accuracy", "money", "requests", "utility_loss")}}
        for name, value in metrics[key].items():
            close(value, report["metrics"][key][name], "Combined reported metric "+name)
        if strategy in (*MODES, *REFERENCES.values(), "base-only", "stage-prior-mean", "tool-prior-mean"):
            behaviors[key] = behavior_summary([r for r in combined if r["strategy"] == strategy and r["budget_money"] == budget])
    comparisons = {}
    for label, expected in report["paired_comparisons"].items():
        pair, budget = label.rsplit("/money=", 1)
        a, b = pair.split("-vs-")
        comparisons[label] = {m: paired(means[a, float(budget)], means[b, float(budget)], m, 2000, 1729)
                              for m in ("log_loss", "money", "utility_loss")}
        require(comparisons[label] == expected, "Independent paired comparison differs")
    result = {"status": "ANALYZED_EXPLORATORY_CALIBRATED_CONTINUATION", "round": 14,
        "metrics": metrics, "paired_comparisons": comparisons, "policy_behavior": behaviors,
        "initial_opportunity_summary": summarize_opportunities(opportunities), "initial_opportunities": opportunities,
        "realized_accounting": accounting, "independently_rebuilt_free_joint_cubes": len(reference.cubes),
        "new_independent_matches": 0, "source_report_sha256": digest(run/"report.json"), "analyzer_sha256": digest(__file__),
        "limitations": ["Existing explored matches; no independent effectiveness confirmation",
            "Calibrated one-step rewards plus fitted return weights form an approximate planning surrogate",
            "Positive option gap is internal to fitted scores, not real-world information benefit",
            "Only actually revealed historical returns receive observed outcome scores",
            "Week intervals omit fitting, repeated research, multiplicity and temporal shift",
            "Two views of one cached response, simulated acquisition costs, no live LLM or tool discovery"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return result


def plot(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    require(not output.exists(), "Figure already exists")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.4), constrained_layout=True)
    contexts = [r for r in result["initial_opportunities"] if r["partition"] == "test"]
    candidates = [c for r in contexts for c in r["candidates"].values()]
    for selected, color, label in ((False, "#a5afb8", "Unselected first action"), (True, "#235d9f", "Selected first action")):
        subset = [c for c in candidates if c["adaptive_selected"] == selected]
        axes[0,0].scatter([c["immediate_net_gain"] for c in subset], [c["adaptive_Q"] for c in subset], s=22, alpha=.65, color=color, label=label)
    low = min(c["immediate_net_gain"] for c in candidates)-.005
    high = max(c["adaptive_Q"] for c in candidates)+.005
    axes[0,0].plot([low,high], [low,high], linestyle="--", color="grey", linewidth=1)
    axes[0,0].axhline(0, color="grey", linewidth=.7)
    axes[0,0].axvline(0, color="grey", linewidth=.7)
    axes[0,0].set(xlabel="Immediate calibrated net value", ylabel="Total adaptive plan net value", title="A. Continuation raises some scores, but not the winning choice")
    axes[0,0].legend(fontsize=9)
    summary = result["initial_opportunity_summary"]["test"]
    labels = ["Positive adaptive\nbonus\n/ 400 candidates", "Positive commit\nbonus\n/ 400 candidates", "Selected with\npositive bonus\n/ 152 queries", "Changed first\nchoice\n/ 200 games"]
    values = [summary["positive_adaptive_bonus_candidates"], summary["positive_commit_bonus_candidates"],
              summary["selected_with_positive_bonus"], summary["adaptive_vs_one_step_initial_changes"]]
    axes[0,1].bar(range(4), values, color=["#235d9f", "#bd631b", "#568b76", "#888888"])
    for i, value in enumerate(values):
        axes[0,1].text(i, value+1, str(value), ha="center", va="bottom")
    axes[0,1].set(xticks=range(4), xticklabels=labels, ylabel="Count (denominators differ as labelled)", ylim=(0,max(values+[1])*1.25+3), title="B. Model opportunities versus actual decisions")
    names = [("base-only", "Free only"), ("frozen-version-mean", "One-step conditional"),
             ("calibrated-commit-two-step", "Two-step commit"), ("calibrated-adaptive-two-step", "Two-step adaptive")]
    metric = [result["metrics"][key+"/money=0.05"]["utility_loss"] for key,_ in names]
    axes[1,0].barh([label for _,label in names], metric, color=["#888888", "#6e8dac", "#bd631b", "#235d9f"])
    axes[1,0].set(xlim=(min(metric)-.01,max(metric)+.013), xlabel="Mean log loss + 0.5 x simulated cost (axis truncated; lower is better)", title="C. Actual policy replay at budget 0.05")
    for i,value in enumerate(metric):
        axes[1,0].text(value+.0004,i,f"{value:.6f}",va="center",fontsize=9)
    accounting = result["realized_accounting"]["primary_budget"]["test/calibrated-adaptive-two-step"]
    values = [accounting["mean_predicted_plan_net_gain"], accounting["mean_actual_net_gain"]]
    axes[1,1].bar([0,1], values, color=["#235d9f", "#bd631b"], width=.5)
    axes[1,1].axhline(0, color="grey", linewidth=.8)
    for i,value in enumerate(values):
        axes[1,1].text(i, value+(0.001 if value>=0 else -.001), f"{value:+.6f}", ha="center", va="bottom" if value>=0 else "top")
    axes[1,1].set(xticks=[0,1], xticklabels=["Predicted by planning scores", "Realized on actual paths"],
        ylabel="Mean net gain versus no query (higher is better)", ylim=(min(values)-.009,max(values)+.009), title="D. Adaptive policy: modeled value versus observed gain")
    for axis in axes.flat:
        axis.grid(axis="y", alpha=.15)
    fig.suptitle("FootballMind round 14: should a first query buy a content-dependent follow-up?\n"
                 "Frozen calibrated reward surrogate; 200 reused test games. Model branches are not new observed outcomes.",fontsize=14)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "bundle", "earlier-bundle", "output"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--figure")
    args = parser.parse_args()
    result = analyze(args.run,args.bundle,args.earlier_bundle,args.output)
    if args.figure:
        plot(result,args.figure)
    print(json.dumps({"status":result["status"],"initial_opportunity_summary":result["initial_opportunity_summary"],
        "realized_accounting":result["realized_accounting"]}))
