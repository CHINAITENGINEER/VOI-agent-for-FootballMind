"""Match-weighted outcomes and visible-state acquisition behavior, round eleven."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .analyze_round7 import match_means, paired
from .verify_round4 import _rows
from .verify_round6 import KEYS
from .verify_round8 import close, require

PRIMARY = "frozen-version-mean"
CALIBRATED = {"frozen-version-mean":"same_version", "frozen-version-lcb":"same_version",
              "matched-mixed-mean":"matched_mixed", "matched-mixed-lcb":"matched_mixed"}


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def behavior_summary(rows):
    """Seeds contribute 1/n inside a game; integer replay counts stay separate."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["match_id"]].append(row)
    first, numbers = Counter(), Counter()
    requests, pchange = 0., 0.
    for records in grouped.values():
        require(len({r["seed"] for r in records}) == len(records), "Repeated behavior seed")
        weight = 1/len(records)
        for row in records:
            first[row["actions"][0] if row["actions"] else "STOP"] += weight
            numbers[str(len(row["actions"]))] += weight
            requests += weight*len(row["actions"])
            pchange += weight*sum(abs(a-b) for a,b in zip(row["final_probabilities"],row["initial_probabilities"]))
    count = len(grouped)
    return {"matches":count, "raw_replays":len(rows),
        "first_choice_match_equivalents":dict(first),
        "query_count_match_equivalents":dict(numbers),
        "initial_stop_rate":first["STOP"]/count,
        "mean_queries":requests/count, "mean_prediction_l1_change":pchange/count,
        "seed_aggregation":"equal weight per match, random repeats averaged inside match"}


def analyze(run, output):
    run, output = Path(run), Path(output)
    require(not output.exists(), "Analysis output exists")
    report = json.loads((run/"report.json").read_bytes())
    design = json.loads((run/"design.json").read_bytes())
    rows = list(_rows(run/"episodes.jsonl"))
    means = match_means(rows, .5)
    metrics, comparisons, behaviors, by_fold = {}, {}, {}, {}
    for (strategy,budget), records in sorted(means.items()):
        label = f"{strategy}/money={budget:g}"
        metrics[label] = {"independent_matches":len(records), **{name:float(np.mean([r[name] for r in records.values()]))
            for name in ("log_loss","brier_score","accuracy","money","requests","utility_loss")}}
        for name,value in metrics[label].items():
            close(value,report["metrics"][label][name],"Reported equal-match metric "+name)
        subset = [r for r in rows if r["strategy"] == strategy and r["budget_money"] == budget]
        behaviors[label] = behavior_summary(subset)
        by_fold[label] = {}
        for fold in sorted({r["fold"] for r in subset}):
            selected = [r for r in subset if r["fold"] == fold]
            values = match_means(selected,.5)[strategy,budget]
            by_fold[label][str(fold)] = {**behavior_summary(selected),
                **{name:float(np.mean([r[name] for r in values.values()])) for name in ("log_loss","money","utility_loss")}}
    for name,wanted in report["paired_comparisons"].items():
        pair,b = name.rsplit("/money=",1)
        strategy,reference = pair.split("-vs-")
        result = {metric:paired(means[strategy,float(b)],means[reference,float(b)],metric,
            design["bootstrap_samples"],design["bootstrap_seed"]) for metric in ("log_loss","money","utility_loss")}
        require(result == wanted,"Reported paired comparison differs")
        comparisons[name] = result
    diagnostic = {(r["fold"],r["match_id"],tuple(r["acquired_action_ids"]),r["action_id"]):r
        for r in _rows(run/"transition-diagnostics.jsonl") if r["partition"] == "test"}
    actions = {r["action_id"]:r for r in json.loads((run/"catalog.json").read_bytes())["actions"]}
    episodes = {tuple(r[k] for k in KEYS):r for r in rows}
    selected_queries, reassessments, accounting = [], [], []
    for trace in _rows(run/"trajectories.jsonl"):
        strategy = trace["strategy"]
        if strategy not in CALIBRATED:
            continue
        episode = episodes[tuple(trace[k] for k in KEYS)]
        queries = []
        for event in trace["events"]:
            decision = event["decision"]
            aid = decision["action_id"]
            if aid is None:
                continue
            candidate = next(c for c in decision["candidates"] if c["action_id"] == aid)
            acquired = tuple(candidate["acquired_action_ids"])
            truth = diagnostic[trace["fold"],trace["match_id"],acquired,aid]
            close(candidate["expected_gain"],truth["estimates"][CALIBRATED[strategy]]["expected_gain"],"Selected estimate differs")
            cost = actions[aid]["cost"]["money"]
            actual = truth["actual_gain"]
            row = {"fold":trace["fold"],"match_id":trace["match_id"],"kickoff":episode["kickoff"],
                "strategy":strategy,"budget_money":trace["budget_money"],"step":event["step"],
                "action_id":aid,"acquired_action_ids":list(acquired),"actual_gain":actual,
                "expected_gain":candidate["expected_gain"],"decision_gain":candidate["decision_gain"],
                "simulated_cost":cost,"net_actual_gain":actual-.5*cost,
                "decision_margin":candidate["decision_gain"]-.5*cost,
                "gain_minus_mean_estimate":actual-candidate["expected_gain"],
                "support":candidate["support"],"parent_support":candidate["parent_support"],
                "scope":"Outcome-dependent diagnostic, never available to the selecting Agent"}
            queries.append(row)
            selected_queries.append(row)
            # Compare the same remaining tool before and after a real return.
            # Its affordability/support can also change and is recorded explicitly.
            later = trace["events"][event["step"]+1]["decision"]
            before_candidates = {c["action_id"]:c for c in decision["candidates"]}
            for following in later["candidates"]:
                other = following["action_id"]
                previous = before_candidates[other]
                feasible = previous.get("feasible",False) and following.get("feasible",False)
                reassessments.append({"fold":trace["fold"],"match_id":trace["match_id"],"strategy":strategy,
                    "budget_money":trace["budget_money"],"acquired_action_id":aid,"remaining_action_id":other,
                    "both_feasible":feasible,"before_decision_gain":previous.get("decision_gain"),
                    "after_decision_gain":following.get("decision_gain"),
                    "before_margin":previous.get("net_gain"),"after_margin":following.get("net_gain"),
                    "after_action_id":later["action_id"],"after_decision":later["decision"],
                    "positive_to_nonpositive": bool(feasible and previous["decision_gain"]-.5*actions[other]["cost"]["money"]>0
                        and following["decision_gain"]-.5*actions[other]["cost"]["money"]<=0),
                    "scope":"Public evidence-state reassessment; no match outcome used"})
        base = means["base-only",episode["budget_money"]][episode["match_id"]]
        actual = sum(r["actual_gain"] for r in queries)
        cost = sum(r["simulated_cost"] for r in queries)
        close(actual,base["log_loss"]-episode["log_loss"],"Telescoping query gains")
        close(cost,episode["cost"]["money"],"Selected query costs")
        accounting.append({"fold":episode["fold"],"match_id":episode["match_id"],"strategy":strategy,
            "budget_money":episode["budget_money"],"queries":len(queries),"actual_gain":actual,
            "actual_net_gain":actual-.5*cost,"mean_estimated_gain":sum(r["expected_gain"] for r in queries),
            "decision_estimated_gain":sum(r["decision_gain"] for r in queries)})
    summaries, re_summaries, disagreements = {}, {}, {}
    for strategy in CALIBRATED:
        for budget in design["money_budgets"]:
            label = f"{strategy}/money={budget:g}"
            subset = [r for r in accounting if r["strategy"] == strategy and r["budget_money"] == budget]
            queries = [r for r in selected_queries if r["strategy"] == strategy and r["budget_money"] == budget]
            summaries[label] = {"matches":len(subset),"selected_queries":len(queries),
                **{field:float(np.mean([r[field] for r in subset])) for field in
                   ("actual_gain","actual_net_gain","mean_estimated_gain","decision_estimated_gain")},
                "selected_gain_minus_mean_estimate":float(np.mean([r["gain_minus_mean_estimate"] for r in queries])) if queries else None}
            reassessed = [r for r in reassessments if r["strategy"] == strategy and r["budget_money"] == budget]
            re_summaries[label] = {"remaining_tool_reassessments":len(reassessed),
                "both_feasible":sum(r["both_feasible"] for r in reassessed),
                "positive_to_nonpositive":sum(r["positive_to_nonpositive"] for r in reassessed),
                "after_stop":sum(r["after_decision"] == "stop" for r in reassessed)}
    for budget in design["money_budgets"]:
        indexed = {strategy:{r["match_id"]:r for r in rows if r["strategy"] == strategy and r["budget_money"] == budget}
            for strategy in ("frozen-version-mean","matched-mixed-mean")}
        a,b = indexed.values()
        require(a.keys() == b.keys(),"Behavior comparison match identities differ")
        pairs = Counter(((a[mid]["actions"][0] if a[mid]["actions"] else "STOP"),
                         (b[mid]["actions"][0] if b[mid]["actions"] else "STOP")) for mid in a)
        disagreements[f"money={budget:g}"] = {"matches":len(a),
            "first_action_disagreements":sum(n for (x,y),n in pairs.items() if x != y),
            "full_sequence_disagreements":sum(a[mid]["actions"] != b[mid]["actions"] for mid in a),
            "first_choice_pairs":[{"same_version":x,"matched_mixed":y,"matches":n} for (x,y),n in sorted(pairs.items())]}
    result = {"status":"ANALYZED_EXPLORATORY","metrics":metrics,"paired_comparisons":comparisons,
        "behavior":behaviors,"by_fold":by_fold,"matched_policy_disagreements":disagreements,
        "selected_query_accounting":summaries,"public_reassessment":re_summaries,
        "selected_queries":selected_queries,"reassessments":reassessments,"episode_accounting":accounting,
        "all_transition_diagnostics":report["transition_diagnostics"],
        "scope":"Same deployed predictor and calendar calibration identities; EVSI/gain-label generators differ jointly",
        "limitations":["200 already explored matches, no new independent confirmation",
            "Model age differs from prior rounds; do not compare headline metrics across predictor versions as acquisition gains",
            "A single realized gain is noisy, not the true conditional value of a query",
            "Good global gain MSE does not establish good ranking or a calibrated stop boundary",
            "State-dependent choices are verified behavior, not proof of utility improvement",
            "Only two frozen Understat projections with simulated cost; no live lineup, rotation or tactical effect claim",
            "Sorted UTC-week bootstrap excludes fitting, repeated research, drift and multiplicity uncertainty"],
        "source_hashes":{str((run/name).resolve()):digest(run/name) for name in
            ("report.json","design.json","episodes.jsonl","trajectories.jsonl","transition-diagnostics.jsonl")},
        "analyzer_sha256":digest(__file__)}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return result


def plot(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    require(not Path(output).exists(),"Figure output exists")
    names = {"base-only":"Free information / STOP", "all-available":"Acquire all", "entropy":"Entropy threshold",
        "evsi-one-step":"Raw one-step EVSI", "matched-mixed-mean":"Matched mixed mean", "frozen-version-mean":"Same-version mean",
        "matched-mixed-lcb":"Matched mixed LCB", "frozen-version-lcb":"Same-version LCB"}
    colors = {k:c for k,c in zip(names,("#555555","#a18668","#d89224","#9454aa","#d45c5c","#195dba","#d99898","#77a4df"))}
    figure,axes = plt.subplots(2,2,figsize=(15.5,10.5))
    for strategy in names:
        values = [(float(k.rsplit("=",1)[1]),v) for k,v in result["metrics"].items() if k.split("/")[0] == strategy]
        values.sort()
        axes[0,0].plot([v["money"] for _,v in values],[v["log_loss"] for _,v in values],"o-",label=names[strategy],color=colors[strategy],ms=4)
    axes[0,0].set(xlabel="Mean simulated acquisition cost",ylabel="Mean log loss (lower is better)",title="Same frozen predictor for all strategies")
    axes[0,0].legend(fontsize=8,loc="best")
    refs = ["base-only","matched-mixed-mean","all-available","entropy","evsi-one-step"]
    for i,ref in enumerate(refs):
        row = result["paired_comparisons"][f"frozen-version-mean-vs-{ref}/money=0.05"]["utility_loss"]
        point = row["mean_difference"]; low,high = row["confidence_interval_95"]
        axes[0,1].plot([low,high],[i,i],color=colors[PRIMARY],lw=2)
        axes[0,1].plot(point,i,"o",color=colors[PRIMARY])
    axes[0,1].axvline(0,ls="--",color="#777777",lw=1)
    axes[0,1].set_yticks(range(len(refs)),["vs "+names[r] for r in refs]); axes[0,1].invert_yaxis()
    axes[0,1].set(xlabel="Same-version minus reference utility loss",title="Paired week bootstrap, budget 0.05")
    choices = ["matched-mixed-mean","frozen-version-mean","matched-mixed-lcb","frozen-version-lcb","entropy"]
    left = np.zeros(len(choices))
    for action,color in (("STOP","#bdbdbd"),("results_view","#25a39b"),("xg_view","#4a79c8")):
        amounts = [result["behavior"][s+"/money=0.05"]["first_choice_match_equivalents"].get(action,0) for s in choices]
        axes[1,0].barh(range(len(choices)),amounts,left=left,label=action,color=color)
        left += amounts
    axes[1,0].set_yticks(range(len(choices)),[names[s] for s in choices]);axes[1,0].invert_yaxis()
    axes[1,0].set(xlabel="Matches (out of 200)",title="Autonomous first query or initial STOP")
    axes[1,0].legend(fontsize=8,loc="lower right")
    for strategy in ("frozen-version-mean","matched-mixed-mean"):
        folds=result["by_fold"][strategy+"/money=0.05"]
        base=result["by_fold"]["base-only/money=0.05"]
        axes[1,1].plot([int(k) for k in folds],[v["utility_loss"]-base[k]["utility_loss"] for k,v in folds.items()],
            "o-",label=names[strategy],color=colors[strategy])
    axes[1,1].axhline(0,ls="--",color="#777777",lw=1)
    axes[1,1].set(xlabel="Original target fold (reference predictor = fold - 2)",ylabel="Utility loss minus free information",title="Fold variation; descriptive, not selection")
    axes[1,1].set_xticks([2,3,4,5,6]);axes[1,1].legend(fontsize=9)
    for axis in axes.ravel():
        axis.grid(alpha=.18);axis.set_axisbelow(True)
    figure.suptitle("FootballMind | Round 11: choosing information with version-bound value labels",fontsize=16,y=.986)
    figure.text(.02,.026,"200 reused historical test matches | 5 temporal folds | Utility = log loss + 0.5 x simulated cost\n"
        "Exploratory negative primary result. Descriptive 95% intervals omit fitting and repeated-research uncertainty. No live tool requests.",fontsize=9)
    figure.tight_layout(rect=[0,.075,1,.956],h_pad=3,w_pad=3)
    figure.savefig(output,dpi=170);plt.close(figure)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("run","output"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--plot")
    args=parser.parse_args()
    result=analyze(args.run,args.output)
    if args.plot:
        plot(result,args.plot)
    print(json.dumps({"status":result["status"],"selected_queries":len(result["selected_queries"]),
        "public_reassessments":len(result["reassessments"])}))
