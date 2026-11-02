"""Shared-state and realized-path contrasts for autonomous query selection."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from .analyze_round7 import match_means, paired
from .analyze_round11 import behavior_summary
from .verify_round4 import _rows
from .verify_round6 import KEYS
from .verify_round8 import close, require
from .verify_round12 import POLICIES, digest, reference_cells, reference_estimate

CONDITIONAL="frozen-version-mean"
FOCUS=(CONDITIONAL,"stage-prior-mean","tool-prior-mean")


def select(candidates, *, money, requests):
    scores=[]
    for aid,row in candidates.items():
        if row["gain"] is None or row["cost"]>money+1e-12 or requests<1:
            continue
        scores.append((round(row["gain"]-.5*row["cost"],12),aid))
    scores.sort(key=lambda r:(-r[0],r[1]))
    return scores[0][1] if scores and scores[0][0]>0 else None


def shared_state_contrasts(diagnostics, cells, costs):
    """Compare policies on identical public states, regardless of visited paths.

    The output excludes actual gains/outcomes. Observed history is materialized
    offline, including branches a policy would not visit, for matched diagnosis.
    """
    states={}
    variations=defaultdict(list)
    for row in diagnostics:
        n,part,mid,acq,aid=row["fold"],row["partition"],row["match_id"],tuple(row["acquired_action_ids"]),row["action_id"]
        key=n,part,mid,acq
        if key not in states:
            states[key]={"fold":n,"partition":part,"match_id":mid,"kickoff":row["kickoff"],
                "acquired_action_ids":list(acq),"probabilities":row["prediction_before"],
                "candidates":{s:{} for s in FOCUS}}
        target=states[key]
        require(target["probabilities"]==row["prediction_before"],"Shared-state probability differs")
        for strategy in FOCUS:
            if strategy==CONDITIONAL:
                value=row["estimates"]["same_version"]["expected_gain"]
            else:
                mode=POLICIES[strategy]
                value=reference_estimate(cells[n,mode],mode,aid,acq)["expected_gain"]
            require(aid not in target["candidates"][strategy],"Duplicate shared-state action")
            target["candidates"][strategy][aid]={"gain":value,"cost":costs[aid],
                "net_gain":value-.5*costs[aid] if value is not None else None}
            variations[part,n,aid,acq,strategy].append(value)
    disagreement=defaultdict(Counter)
    for state in states.values():
        acquired=state["acquired_action_ids"]
        state["choices"]={s:select(c,money=.05-sum(costs[a] for a in acquired),requests=2-len(acquired))
            for s,c in state["candidates"].items()}
        c=state["choices"]
        phase="initial" if not acquired else "after_one_return"
        counts=disagreement[state["partition"],phase]
        counts["public_states"]+=1
        counts["conditional_vs_stage_different"]+=int(c[CONDITIONAL]!=c["stage-prior-mean"])
        counts["conditional_vs_tool_different"]+=int(c[CONDITIONAL]!=c["tool-prior-mean"])
        counts["stage_vs_tool_different"]+=int(c["stage-prior-mean"]!=c["tool-prior-mean"])
    reassessments=[]
    for (n,part,mid,acq),state in states.items():
        if len(acq)!=1:
            continue
        initial=states[n,part,mid,()]
        aid=next(iter(state["candidates"][CONDITIONAL]))
        item={"fold":n,"partition":part,"match_id":mid,"observed_action_id":acq[0],"remaining_action_id":aid,
            "policies":{}}
        for strategy in FOCUS:
            before=initial["candidates"][strategy][aid]["net_gain"]
            after=state["candidates"][strategy][aid]["net_gain"]
            valid=before is not None and after is not None
            item["policies"][strategy]={"before_net_gain":before,"after_net_gain":after,
                "positive_to_nonpositive":bool(valid and round(before,12)>0 and round(after,12)<=0),
                "nonpositive_to_positive":bool(valid and round(before,12)<=0 and round(after,12)>0),
                "net_value_changed":bool(valid and abs(before-after)>1e-12)}
        reassessments.append(item)
    summary={}
    for part in ("test","validation"):
        subset=[r for r in reassessments if r["partition"]==part]
        summary[part]={s:{"matched_return_states":len(subset),**{k:sum(r["policies"][s][k] for r in subset)
            for k in ("positive_to_nonpositive","nonpositive_to_positive","net_value_changed")}} for s in FOCUS}
    variation=[]
    for (part,n,aid,acq,s),values in sorted(variations.items()):
        numeric=[v for v in values if v is not None]
        row={"partition":part,"fold":n,"action_id":aid,"acquired_action_ids":list(acq),"strategy":s,
            "states":len(values),"unsupported":len(values)-len(numeric),
            "unique_rounded_gain_values":len({round(v,12) for v in numeric}),
            "min_gain":min(numeric) if numeric else None,"max_gain":max(numeric) if numeric else None}
        if s in POLICIES:
            require(row["unique_rounded_gain_values"]<=1,"Erased prior changed value within identical allowed inputs")
        variation.append(row)
    return {"states":list(states.values()),"reassessments":reassessments,"reassessment_summary":summary,
        "action_disagreement":{f"{part}/{phase}":dict(v) for (part,phase),v in sorted(disagreement.items())},
        "within_fold_value_variation":variation,
        "scope":"Shared public-state offline contrasts include unvisited histories; no realized outcome enters any value or choice here",
        "interpretation":"Stage-prior changes with acquired-tool identity alone. A sign change after return does not by itself establish sensitivity to its contents. Conditional variation also includes match priors and is not a pure intervention on return content."}


def analyze(run,output):
    run,output=Path(run),Path(output)
    require(not output.exists(),"Analysis output exists")
    report=json.loads((run/"report.json").read_bytes())
    design=json.loads((run/"design.json").read_bytes())
    source=Path(design["source_round11"])
    rows=list(_rows(run/"episodes.jsonl"))+list(_rows(source/"episodes.jsonl"))
    means=match_means(rows,.5)
    metrics,behaviors,by_fold={},{},{}
    for (strategy,budget),records in sorted(means.items()):
        label=f"{strategy}/money={budget:g}"
        metrics[label]={"independent_matches":len(records),**{key:float(np.mean([r[key] for r in records.values()]))
            for key in ("log_loss","brier_score","accuracy","money","requests","utility_loss")}}
        for key,value in metrics[label].items():
            close(value,report["metrics"][label][key],"Combined metric "+key)
        if strategy not in (*FOCUS,"base-only"):
            continue
        subset=[r for r in rows if r["strategy"]==strategy and r["budget_money"]==budget]
        behaviors[label]=behavior_summary(subset)
        by_fold[label]={}
        for fold in sorted({r["fold"] for r in subset}):
            part=[r for r in subset if r["fold"]==fold]
            values=match_means(part,.5)[strategy,budget]
            by_fold[label][str(fold)]={**behavior_summary(part),
                **{key:float(np.mean([r[key] for r in values.values()])) for key in ("log_loss","money","utility_loss")}}
    comparisons={}
    for label,expected in report["paired_comparisons"].items():
        pair,b=label.rsplit("/money=",1);a,ref=pair.split("-vs-")
        comparisons[label]={m:paired(means[a,float(b)],means[ref,float(b)],m,2000,1729) for m in ("log_loss","money","utility_loss")}
        require(comparisons[label]==expected,"Paired summary differs")
    costs={a["action_id"]:a["cost"]["money"] for a in json.loads((run/"catalog.json").read_bytes())["actions"]}
    cells={}
    for fold in report["folds"]:
        samples=list(_rows(fold["source_same_version_supervision"]["path"]))
        for mode in POLICIES.values():
            cells[fold["fold"],mode]=reference_cells(samples,mode)
    diagnostics=list(_rows(source/"transition-diagnostics.jsonl"))
    common=shared_state_contrasts(diagnostics,cells,costs)
    truth={(r["fold"],r["match_id"],tuple(r["acquired_action_ids"]),r["action_id"]):r for r in diagnostics if r["partition"]=="test"}
    indexed={tuple(r[k] for k in KEYS):r for r in rows}
    query_records,episode_records=[],[]
    for path in (run/"trajectories.jsonl",source/"trajectories.jsonl"):
        for trace in _rows(path):
            if trace["strategy"] not in FOCUS:
                continue
            row=indexed[tuple(trace[k] for k in KEYS)]
            queries=[]
            for event in trace["events"]:
                decision=event["decision"]
                aid=decision["action_id"]
                if aid is None:
                    continue
                candidate=next(c for c in decision["candidates"] if c["action_id"]==aid)
                acq=tuple(candidate["acquired_action_ids"])
                d=truth[row["fold"],row["match_id"],acq,aid]
                item={"fold":row["fold"],"match_id":row["match_id"],"strategy":row["strategy"],"budget_money":row["budget_money"],
                    "step":event["step"],"action_id":aid,"acquired_action_ids":list(acq),
                    "expected_gain":candidate["expected_gain"],"actual_gain":d["actual_gain"],"cost":costs[aid],
                    "expected_net_gain":candidate["expected_gain"]-.5*costs[aid],"actual_net_gain":d["actual_gain"]-.5*costs[aid],
                    "scope":"Selected query hindsight accounting, not a runtime feature or conditional true-gain label"}
                queries.append(item);query_records.append(item)
            actual_gain=sum(q["actual_gain"] for q in queries)
            cost=sum(q["cost"] for q in queries)
            close(cost,row["cost"]["money"],"Query cost telescope")
            close(actual_gain,means["base-only",row["budget_money"]][row["match_id"]]["log_loss"]-row["log_loss"],"Query gain telescope")
            episode_records.append({"fold":row["fold"],"match_id":row["match_id"],"strategy":row["strategy"],"budget_money":row["budget_money"],
                "queries":len(queries),"cost":cost,"actual_gain":actual_gain,"actual_net_gain":actual_gain-.5*cost,
                "expected_gain":sum(q["expected_gain"] for q in queries),"expected_net_gain":sum(q["expected_net_gain"] for q in queries)})
    accounting={}
    for s in FOCUS:
        for b in design["money_budgets"]:
            selected=[r for r in episode_records if r["strategy"]==s and r["budget_money"]==b]
            accounting[f"{s}/money={b:g}"]={"matches":len(selected),"queries":sum(r["queries"] for r in selected),
                **{name:float(np.mean([r[name] for r in selected])) for name in
                    ("cost","actual_gain","actual_net_gain","expected_gain","expected_net_gain")}}
    result={"status":"ANALYZED_EXPLORATORY","metrics":metrics,"paired_comparisons":comparisons,"behavior":behaviors,
        "by_fold":by_fold,"shared_public_states":common,"selected_query_accounting":accounting,
        "selected_queries":query_records,"episode_accounting":episode_records,
        "primary_comparison":"frozen-version-mean-vs-stage-prior-mean/money=0.05",
        "scope":"Implemented conditional policy versus two simpler historical-mean controls, with identical deployed predictor and calibration matches",
        "limitations":["All 200 test matches were previously explored; no independent confirmation",
            "Action-only uniformly averages two acquisition states, not an estimated deployed-state distribution",
            "The conditional estimator and historical means differ in bias/variance; no isolated causal conditioning effect",
            "After-return state changes can be driven by acquired-tool identity alone; content sensitivity needs stronger evidence",
            "Query gains are noisy realized values, not known true conditional VoI",
            "Conditional values still depend on a scalar model EVSI and past residual bins, not an LLM discovering new information sources",
            "Two historical projections and simulated costs; no live lineup or tactical evidence claim",
            "Week intervals omit fitting, adaptive research, multiplicity and drift uncertainty"],
        "source_hashes":{str(path.resolve()):digest(path) for path in (run/"report.json",run/"design.json",run/"episodes.jsonl",
            run/"trajectories.jsonl",source/"episodes.jsonl",source/"trajectories.jsonl",source/"transition-diagnostics.jsonl")},
        "analyzer_sha256":digest(__file__),"reference_math_sha256":digest(Path(__file__).with_name("verify_round12.py"))}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return result


def plot(result,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    require(not Path(output).exists(),"Figure output exists")
    names={"base-only":"Free information / STOP",CONDITIONAL:"Conditional query value",
        "stage-prior-mean":"Action + acquired-set mean","tool-prior-mean":"Action-only mean"}
    colors={"base-only":"#666666",CONDITIONAL:"#2363b5","stage-prior-mean":"#db7e24","tool-prior-mean":"#218c79"}
    fig,ax=plt.subplots(2,2,figsize=(15,10.2))
    for s in names:
        values=sorted((float(k.rsplit("=",1)[1]),v) for k,v in result["metrics"].items() if k.split("/")[0]==s)
        ax[0,0].plot([v["money"] for _,v in values],[v["log_loss"] for _,v in values],"o-",label=names[s],color=colors[s],ms=5)
    ax[0,0].set(xlabel="Mean simulated acquisition cost",ylabel="Mean log loss (lower is better)",title="Same predictor, calibration matches and budgets")
    ax[0,0].legend(fontsize=9)
    pairs=[(CONDITIONAL,"stage-prior-mean"),(CONDITIONAL,"tool-prior-mean"),("stage-prior-mean","tool-prior-mean"),
        ("stage-prior-mean","base-only"),("tool-prior-mean","base-only")]
    labels=[]
    for i,(a,b) in enumerate(pairs):
        row=result["paired_comparisons"][f"{a}-vs-{b}/money=0.05"]["utility_loss"]
        lo,hi=row["confidence_interval_95"]
        ax[0,1].plot([lo,hi],[i,i],color=colors[a],lw=2)
        ax[0,1].plot(row["mean_difference"],i,"o",color=colors[a])
        labels.append(names[a]+"\nminus "+names[b])
    ax[0,1].axvline(0,color="#777777",ls="--",lw=1)
    ax[0,1].set_yticks(range(len(pairs)),labels,fontsize=8);ax[0,1].invert_yaxis()
    ax[0,1].set(xlabel="Paired utility loss difference",title="Budget 0.05: descriptive 95% week intervals")
    choices=list(FOCUS);left=np.zeros(3)
    for aid,color in (("STOP","#c0c0c0"),("results_view","#28a398"),("xg_view","#507ec9")):
        amounts=[result["behavior"][s+"/money=0.05"]["first_choice_match_equivalents"].get(aid,0) for s in choices]
        ax[1,0].barh(range(3),amounts,left=left,label=aid,color=color);left+=amounts
    ax[1,0].set_yticks(range(3),[names[s] for s in choices]);ax[1,0].invert_yaxis()
    ax[1,0].set(xlabel="Matches (out of 200)",title="First tool or STOP at identical budget")
    ax[1,0].legend(fontsize=8,loc="lower right")
    full=result["by_fold"][CONDITIONAL+"/money=0.05"]
    for s in ("stage-prior-mean","tool-prior-mean"):
        reference=result["by_fold"][s+"/money=0.05"]
        ax[1,1].plot([int(k) for k in full],[v["utility_loss"]-reference[k]["utility_loss"] for k,v in full.items()],
            "o-",color=colors[s],label="Conditional minus "+names[s])
    ax[1,1].axhline(0,color="#777777",ls="--",lw=1)
    ax[1,1].set(xlabel="Frozen target fold",ylabel="Utility loss difference",title="Fold-level variation, no policy selection")
    ax[1,1].set_xticks([2,3,4,5,6]);ax[1,1].legend(fontsize=8)
    for axis in ax.ravel():
        axis.grid(alpha=.18);axis.set_axisbelow(True)
    fig.suptitle("FootballMind | Round 12: does current evidence improve tool choice?",fontsize=16,y=.985)
    fig.text(.02,.025,"200 previously explored matches | Utility = log loss + 0.5 x simulated cost | Reference forecasts remain frozen\n"
        "State-erased controls still update forecasts from actual returns. Intervals omit model-fit and repeated-research uncertainty.",fontsize=9)
    fig.tight_layout(rect=[0,.075,1,.957],w_pad=2.5,h_pad=3)
    fig.savefig(output,dpi=170);plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("run","output"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--plot")
    args=parser.parse_args()
    result=analyze(args.run,args.output)
    if args.plot:
        plot(result,args.plot)
    print(json.dumps({"status":result["status"],"shared_states":len(result["shared_public_states"]["states"]),
        "selected_queries":len(result["selected_queries"])}))
