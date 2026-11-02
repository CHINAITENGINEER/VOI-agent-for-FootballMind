"""Descriptive round-nine analysis with canonical sorted-week paired intervals.

Adds no fitted parameters and never changes the frozen experiment artifacts.
Selected-action diagnostics are explicitly post-selection descriptive summaries.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .analyze_round7 import match_means, paired
from .verify_round4 import _rows

CALIBRATED=("calibrated-mean","calibrated-lcb")
REFERENCES=("base-only","evsi-one-step","evsi-two-step","all-available","entropy")


def _key(record):
    return record["fold"],record["match_id"],tuple(record["acquired_action_ids"]),record["action_id"]


def _gain_summary(rows):
    supported=[r for r in rows if r.get("expected_gain") is not None]
    return {"transitions":len(rows),"supported_transitions":len(supported),
        "independent_matches":len({r["match_id"] for r in rows}),
        "mean_raw_evsi":float(np.mean([r["model_evsi"] for r in supported])) if supported else None,
        "mean_estimated_gain":float(np.mean([r["expected_gain"] for r in supported])) if supported else None,
        "mean_actual_gain":float(np.mean([r["gain"] for r in supported])) if supported else None,
        "raw_mse":float(np.mean([(r["model_evsi"]-r["gain"])**2 for r in supported])) if supported else None,
        "calibrated_mse":float(np.mean([(r["expected_gain"]-r["gain"])**2 for r in supported])) if supported else None,
        "mean_calibration_error":float(np.mean([r["expected_gain"]-r["gain"] for r in supported])) if supported else None}


def analyze(run,output):
    run,output=Path(run),Path(output)
    if output.exists():
        raise FileExistsError(output)
    design=json.loads((run/"design.json").read_text(encoding="utf-8"))
    report=json.loads((run/"report.json").read_text(encoding="utf-8"))
    if report.get("status") != "OK":
        raise ValueError("Finished formal experiment required")
    episodes=list(_rows(run/"episodes.jsonl"))
    means=match_means(episodes,design["validation_utility_money_weight"])
    metrics,comparisons={},{}
    for (strategy,budget),records in sorted(means.items()):
        metrics[f"{strategy}/money={budget:g}"]={"independent_matches":len(records),
            **{m:float(np.mean([r[m] for r in records.values()])) for m in ("log_loss","brier_score","accuracy","money","requests","utility_loss")}}
        if strategy not in CALIBRATED:
            continue
        for reference in REFERENCES:
            comparisons[f"{strategy}-vs-{reference}/money={budget:g}"]={
                metric:paired(records,means[reference,budget],metric,design["bootstrap_samples"],design["bootstrap_seed"])
                for metric in ("log_loss","money","utility_loss")}
    ep_lookup={(r["fold"],r["match_id"],r["strategy"],r["budget_money"],r["seed"]):r for r in episodes}
    diagnostics=list(_rows(run/"update-gain-diagnostics.jsonl"))
    diagnostic_lookup={_key(r):r for r in diagnostics}
    if len(diagnostic_lookup) != len(diagnostics):
        raise ValueError("Duplicate complete-transition diagnostic")
    selected=[]
    totals=defaultdict(float)
    for trajectory in _rows(run/"trajectories.jsonl"):
        if trajectory["strategy"] not in CALIBRATED:
            continue
        identity=(trajectory["fold"],trajectory["match_id"],trajectory["strategy"],trajectory["budget_money"],trajectory["seed"])
        for event in trajectory["events"]:
            decision=event["decision"]
            if decision["decision"] == "stop":
                continue
            if decision["decision"] != "acquire":
                raise ValueError("Unexpected calibrated action")
            chosen=next(c for c in decision["candidates"] if c["action_id"] == decision["action_id"])
            key=(trajectory["fold"],trajectory["match_id"],tuple(chosen["acquired_action_ids"]),chosen["action_id"])
            truth=diagnostic_lookup[key]
            for name in ("expected_gain","model_evsi","lower_gain","standard_error"):
                if chosen[name] != truth[name]:
                    raise ValueError("Selected query differs from complete-transition diagnostic")
            record={"fold":trajectory["fold"],"match_id":trajectory["match_id"],"strategy":trajectory["strategy"],
                "budget_money":trajectory["budget_money"],"step":event["step"],"kickoff":truth["kickoff"],
                "action_id":chosen["action_id"],"acquired_action_ids":chosen["acquired_action_ids"],
                "model_evsi":chosen["model_evsi"],"expected_gain":chosen["expected_gain"],"lower_gain":chosen["lower_gain"],
                "standard_error":chosen["standard_error"],"gain":truth["gain"],"charged_money":event["charged_cost"]["money"],
                "lambda_money":decision["lambda_cost"]["money"],"decision_gain":chosen["decision_gain"],
                "cost_penalty":chosen["cost_penalty"],"selection_score":chosen["selection_score"],
                "actual_utility_gain":truth["gain"]-design["validation_utility_money_weight"]*event["charged_cost"]["money"]}
            selected.append(record)
            totals[identity]+=truth["gain"]
        episode=ep_lookup[identity]
        base=means["base-only",trajectory["budget_money"]][trajectory["match_id"]]
        if abs(totals[identity]-(base["log_loss"]-episode["log_loss"]))>1e-10:
            raise ValueError("Selected incremental gains do not telescope to episode improvement")
    behavior,selected_summary={},{}
    episode_groups=defaultdict(lambda:defaultdict(list))
    for row in episodes:
        episode_groups[row["strategy"],row["budget_money"]][row["match_id"]].append(row)
    for (strategy,budget),matches in sorted(episode_groups.items()):
        key=f"{strategy}/money={budget:g}"
        behavior[key]={"independent_matches":len(matches),"episodes":sum(map(len,matches.values())),
            "initial_stop_rate":float(np.mean([np.mean([len(r["actions"])==0 for r in group]) for group in matches.values()])),
            "query_count_probabilities":{str(n):float(np.mean([np.mean([len(r["actions"])==n for r in group]) for group in matches.values()])) for n in (0,1,2)},
            "mean_query_count":float(np.mean([np.mean([len(r["actions"]) for r in group]) for group in matches.values()])),
            "aggregation":"each match equal-weight after averaging seeds; zero queries equals initial STOP"}
        if strategy not in CALIBRATED:
            continue
        queries=[r for r in selected if r["strategy"]==strategy and r["budget_money"]==budget]
        selected_summary[key]={**_gain_summary(queries),"test_matches":len(matches),
            "mean_selected_query_utility_gain":float(np.mean([r["actual_utility_gain"] for r in queries])) if queries else None,
            "total_selected_gain_per_test_match":sum(r["gain"] for r in queries)/len(matches),
            "total_selected_utility_gain_per_test_match":sum(r["actual_utility_gain"] for r in queries)/len(matches),
            "scope":"actual selected queries only, descriptive post-selection outcomes; budgets reuse matches and queries"}
    per_fold={str(fold):_gain_summary([r for r in diagnostics if r["fold"]==fold]) for fold in sorted({r["fold"] for r in diagnostics})}
    files=("report.json","design.json","episodes.jsonl","trajectories.jsonl","update-gain-diagnostics.jsonl")
    result={"status":"ANALYZED_EXPLORATORY","run":str(run.resolve()),"money_weight":design["validation_utility_money_weight"],
        "metrics":metrics,"paired_comparisons":comparisons,"policy_behavior":behavior,
        "complete_transition_diagnostics":{"pooled":_gain_summary(diagnostics),"by_fold":per_fold},
        "selected_query_diagnostics":selected_summary,"selected_query_records":selected,
        "selected_gain_telescoping_check":True,"raw_round8_parity":report["raw_round8_parity"],
        "method":"analyze_round7.match_means/paired: seeds within match; UTC weeks sorted before percentile bootstrap",
        "bootstrap_samples":design["bootstrap_samples"],"bootstrap_seed":design["bootstrap_seed"],
        "interval_note":"Canonical sorted-week intervals can differ slightly from runner's unsorted-week finite-bootstrap draws; same estimand and sample, no fitting/tuning",
        "limitations":["290 reused historical test matches, not independent confirmation","All acquisition costs simulated",
            "LCB stops on all tested matches: this is base-only behavior, not demonstrated useful acquisition",
            "Conditional mean estimator bounds have no drift or post-selection uniform coverage guarantee",
            "Confidence intervals omit fitted-model, repeated-research and multiplicity uncertainty",
            "Selected-query outcomes are post-selection descriptive diagnostics; repeated budgets do not create independent data"],
        "source_hashes":{name:sha256((run/name).read_bytes()).hexdigest() for name in files},
        "analyzer_sha256":sha256(Path(__file__).read_bytes()).hexdigest(),
        "bootstrap_helper_sha256":sha256(Path(__file__).with_name("analyze_round7.py").read_bytes()).hexdigest()}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return result


def plot(result,path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path=Path(path)
    if path.exists():
        raise FileExistsError(path)
    labels={"base-only":"Free context only","all-available":"Acquire all affordable","entropy":"Entropy threshold",
        "random":"Random (5 seeds/game)","evsi-one-step":"Raw EVSI: one step","evsi-two-step":"Raw EVSI: two steps",
        "calibrated-mean":"Calibrated mean","calibrated-lcb":"Calibrated LCB: all STOP"}
    colors={"base-only":"#4b5563","all-available":"#9b7653","entropy":"#e69f00","random":"#a378b3",
        "evsi-one-step":"#56b4e9","evsi-two-step":"#0072b2","calibrated-mean":"#009e73","calibrated-lcb":"#202020"}
    fig,axes=plt.subplots(1,2,figsize=(16,7),gridspec_kw={"width_ratios":[1.05,1]})
    for strategy,label in labels.items():
        values=sorted((float(k.rsplit("=",1)[1]),r) for k,r in result["metrics"].items() if k.split("/")[0]==strategy)
        if strategy=="calibrated-lcb":
            point=values[-1][1]
            axes[0].scatter([point["money"]],[point["log_loss"]],marker="s",s=90,facecolors="none",edgecolors=colors[strategy],label=label,zorder=6)
        else:
            axes[0].plot([r["money"] for _,r in values],[r["log_loss"] for _,r in values],"o-",color=colors[strategy],
                label=label,markersize=4.5,linewidth=2.1 if strategy=="calibrated-mean" else 1.3,alpha=.95)
    axes[0].set(xlabel="Mean simulated acquisition cost",ylabel="Test log loss (lower is better)",
        title="Frozen predictor; only the acquisition policy changes")
    axes[0].legend(fontsize=8.8,loc="upper left",framealpha=.95)
    axes[0].grid(alpha=.2)
    budget=max(float(k.rsplit("=",1)[1]) for k in result["metrics"])
    for index,reference in enumerate(REFERENCES):
        data=result["paired_comparisons"][f"calibrated-mean-vs-{reference}/money={budget:g}"]["utility_loss"]
        point=data["mean_difference"]
        low,high=data["confidence_interval_95"]
        axes[1].errorbar(point,index,xerr=[[point-low],[high-point]],fmt="o",color=colors["calibrated-mean"],capsize=4)
        axes[1].annotate(f"{point:+.4f}",(point,index),xytext=(0,10),textcoords="offset points",ha="center",fontsize=9)
    axes[1].set_yticks(range(len(REFERENCES)),["vs "+labels[r] for r in REFERENCES])
    axes[1].invert_yaxis()
    axes[1].set_ylim(len(REFERENCES)-.55,-.65)
    axes[1].axvline(0,color="#777777",linestyle="--",linewidth=1)
    axes[1].set(xlabel="Calibrated mean minus reference: log loss + 0.5 × cost",
        title=f"Paired utility differences; budget limit {budget:g}")
    axes[1].grid(axis="x",alpha=.2)
    fig.suptitle("FootballMind · Round 9: empirical update-gain calibration and stopping",fontsize=15,y=.97)
    fig.text(.025,.045,"290 reused historical test matches · 7 temporal folds · Simulated costs · 95% sorted-week paired bootstrap\n"
        "LCB makes no queries. Mean calibration improves on raw EVSI; its utility advantage over free-context-only remains unestablished.",fontsize=9.5)
    fig.tight_layout(rect=[0,.12,1,.92],w_pad=3.2)
    fig.savefig(path,dpi=175)
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--plot")
    args=parser.parse_args()
    result=analyze(args.run,args.output)
    if args.plot:
        plot(result,args.plot)
    print(json.dumps({"status":result["status"],"comparisons":len(result["paired_comparisons"]),"selected_query_records":len(result["selected_query_records"])}))
