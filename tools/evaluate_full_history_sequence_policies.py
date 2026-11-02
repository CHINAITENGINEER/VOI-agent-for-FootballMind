"""Evaluate frozen f/g choices once on untouched roles; never refit a head."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import socket
import sys
import time

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from football_research.acquisition.sequential_evidence_f import FrozenF
from football_research.acquisition.sequential_postread_g import (
    EvidencePanel,VectorStore,FrozenEvaluator,run_policy,settle_path,exact_oracle,summarize_paths,
    generate_transitions,predict_head,metrics,save_json,save_rows,sha_file)


def figures(output,summary,paths,prediction_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5))
    for role,name in sorted({(r["split_role"],r["strategy"]) for r in summary}):
        rr=sorted((r for r in summary if r["strategy"]==name and r["split_role"]==role),key=lambda r:r["budget"])
        ax.plot([r["budget"] for r in rr],[r["mean_final_log_loss"] for r in rr],"o-",label=role+":"+name)
    ax.set_xlabel("Maximum f updates (all = separate uncapped reference)");ax.set_ylabel("Match mean final log loss");ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/"budget-performance.png",dpi=160);plt.close(fig)
    rr=[r for r in paths if r["strategy"]=="new_g"]
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].hist([r["stopping_position"] for r in rr],bins=np.arange(9)-.5);axes[0].set_xlabel("Number of adopted documents");axes[0].set_ylabel("Match-budget paths")
    axes[1].scatter([r["f_updates"] for r in rr],[r["realized_gain"] for r in rr],alpha=.4,s=12);axes[1].axhline(0,color="black",lw=.7);axes[1].set_xlabel("f updates");axes[1].set_ylabel("Realized signed gain")
    fig.tight_layout();fig.savefig(output/"stopping-analysis.png",dpi=160);plt.close(fig)
    y=np.array([r["realized_gain"] for r in prediction_rows]);p=np.array([r["predicted_gain"] for r in prediction_rows]);fig,axes=plt.subplots(1,3,figsize=(13,4))
    lo=min(y.min(),p.min());hi=max(y.max(),p.max());axes[0].scatter(y,p,s=10,alpha=.3);axes[0].plot([lo,hi],[lo,hi],"k--",label="y=x reference; not fitted");axes[0].legend(fontsize=7);axes[0].set_xlabel("Realized gain");axes[0].set_ylabel("Predicted gain")
    groups=np.array_split(np.argsort(p),min(10,len(p)));axes[1].plot([p[ii].mean() for ii in groups],[y[ii].mean() for ii in groups],"o-");axes[1].plot([lo,hi],[lo,hi],"k--");axes[1].set_xlabel("Bin mean prediction");axes[1].set_ylabel("Bin mean gain")
    axes[2].hist(p-y,bins=30);axes[2].set_xlabel("Prediction minus realized gain");axes[2].set_ylabel("Transitions")
    fig.tight_layout();fig.savefig(output/"gain-diagnostics.png",dpi=160);plt.close(fig)


def evaluate(args):
    if args.output.exists():raise FileExistsError("Evaluation output exists; do not silently repeat a held-out evaluation")
    bundle=joblib.load(args.g_model/"g-models.joblib");selection=json.loads((args.g_model/"selection-frozen.json").read_text(encoding="utf-8"))
    if str(args.data.resolve())!=bundle["dataset_dir"]:raise ValueError("Evaluation must use the same frozen dataset")
    if any(sha_file(p)!=digest for p,digest in bundle["source_sha256"].items()):raise ValueError("Frozen g dataset source changed")
    if selection["selected"]!=bundle["selected"] or selection["test_used"] or selection["validation_used"]:raise ValueError("Model selection was not frozen without evaluation roles")
    f=FrozenF.load(args.f_model)
    if f.f_model_sha256!=bundle["f_model_sha256"]:raise ValueError("g must remain bound to the same frozen f")
    panel=EvidencePanel(args.data,VectorStore(args.vectors),roles=set(args.roles),include_outcomes=True)
    eval_ids=set(panel.sequences)
    if eval_ids&(set(bundle["train_matches"])|set(bundle["dev_matches"])):raise ValueError("Evaluation overlaps g train/development")
    args.output.mkdir(parents=True);evaluator=FrozenEvaluator(f,panel);started=time.perf_counter();paths=[];oracles=[];errors=[]
    name=bundle["selected"]["model"];spec=bundle["fitted"][name];threshold=bundle["selected"]["threshold"]
    per_model_selection={model:min([r for r in selection["all_dev_policy_choices"] if r["model"]==model],
        key=lambda r:(r["mean_final_log_loss"],r["mean_updates"],r["threshold"])) for model in bundle["fitted"]}
    for mid in sorted(panel.sequences):
        if mid not in panel.outcomes:
            errors.append(dict(physical_match_id=mid,reason="independent_outcome_missing",gain=None));continue
        for budget in (1,2,3,6):
            model_strategies={"frozen_head:"+model:model for model in bundle["fitted"] if model not in ("zero","train_mean")}
            for strategy in ("none","all","fixed","random","new_g",*model_strategies):
                for repeat in range(10 if strategy=="random" else 1):
                    try:
                        actual="g" if strategy=="new_g" or strategy in model_strategies else strategy
                        chosen_name=model_strategies.get(strategy,name)
                        chosen_spec=bundle["fitted"][chosen_name]
                        chosen_threshold=per_model_selection[chosen_name]["threshold"] if strategy in model_strategies else threshold
                        path=run_policy(evaluator,mid,budget=budget,strategy=actual,spec=chosen_spec,
                            target_mean=bundle["target_mean"],target_scale=bundle["target_scale"],threshold=chosen_threshold,seed=20261025+repeat)
                        path=settle_path(path,panel.outcomes[mid]);path["strategy"]=strategy;path["random_repeat"]=repeat if strategy=="random" else None
                        path["split_role"]=panel.matches[mid]["split_role"];path["division"]=panel.matches[mid].get("division");path["home_team"]=panel.matches[mid].get("home_team");path["away_team"]=panel.matches[mid].get("away_team")
                        path["previous_development_exposure"]=panel.matches[mid].get("previous_development_exposure")
                        paths.append(path)
                    except (ValueError,RuntimeError,KeyError) as exc:errors.append(dict(physical_match_id=mid,strategy=strategy,budget=budget,reason=str(exc),gain=None))
            oracle=exact_oracle(evaluator,mid,budget,max_states=1000);oracle["budget"]=budget;oracles.append(oracle)
    summary=summarize_paths(paths)
    exposure_summary=[]
    for exposure in (True,False,None):
        subgroup=[r for r in paths if r["previous_development_exposure"] is exposure]
        exposure_summary.extend(dict(previous_development_exposure=exposure,**r) for r in summarize_paths(subgroup))
    for row in summary:
        relevant=[r for r in paths if r["split_role"]==row["split_role"] and r["strategy"]==row["strategy"] and r["budget"]==row["budget"]]
        row["split_roles"]=sorted({r["split_role"] for r in relevant})
    # Complete counterfactual gain diagnostics occur only after final choice was
    # frozen. All signed values, including zero and negative, are retained.
    x,labels,label_failures,_=generate_transitions(evaluator,budget=3,orders_per_match=3)
    predicted=predict_head(spec,x,bundle["target_mean"],bundle["target_scale"])
    pred_rows=[dict(sample_id=r["sample_id"],physical_match_id=r["physical_match_id"],split_role=r["split_role"],realized_gain=r["realized_gain"],predicted_gain=float(p)) for r,p in zip(labels,predicted)]
    gain_metrics=[]
    all_pred_rows=[]
    for model,head in bundle["fitted"].items():
        pp=predict_head(head,x,bundle["target_mean"],bundle["target_scale"])
        all_pred_rows.extend(dict(model=model,**{k:r[k] for k in ("sample_id","physical_match_id","split_role","realized_gain")},predicted_gain=float(p)) for r,p in zip(labels,pp))
        for role in args.roles:
            ii=[i for i,r in enumerate(labels) if r["split_role"]==role]
            if ii:gain_metrics.append(dict(model=model,split_role=role,**metrics([labels[i]["realized_gain"] for i in ii],pp[ii],[labels[i]["physical_match_id"] for i in ii])))
    missed=[]
    for path in paths:
        if path["strategy"]!="new_g":continue
        mid=path["physical_match_id"];selected=path["selected_indices"];y=path["outcome_class"];current=evaluator.predict(mid,selected)
        omitted=[]
        for candidate in range(len(panel.sequences[mid]["document_ids"])):
            if candidate in selected:continue
            p=evaluator.predict(mid,(*selected,candidate));gain=float(np.log(p[y])-np.log(current[y]))
            omitted.append(dict(document_id=panel.sequences[mid]["document_ids"][candidate],hindsight_one_step_gain=gain))
        missed.append(dict(physical_match_id=mid,budget=path["budget"],stop_reason=path["stop_reason"],omitted_candidates=omitted,
            hindsight_missed_positive=sum(r["hindsight_one_step_gain"]>0 for r in omitted),
            not_permanent_article_value=True,not_causal_proof_of_wrong_stop=True))
    save_rows(args.output/"per-match-paths.jsonl",paths);save_rows(args.output/"oracle.jsonl",oracles)
    save_rows(args.output/"labels.jsonl",labels);save_rows(args.output/"gain-predictions.jsonl",pred_rows)
    save_rows(args.output/"all-frozen-head-predictions.jsonl",all_pred_rows)
    save_rows(args.output/"failures.jsonl",panel.failures+errors+label_failures);save_json(args.output/"gain-metrics.json",gain_metrics)
    save_json(args.output/"per-match-error-analysis.json",missed)
    save_json(args.output/"strategy-comparison.json",dict(summary=summary,development_exposure_stratified_summary=exposure_summary,roles=args.roles,selected_model=name,threshold=threshold,
        f_model_sha256=f.f_model_sha256,g_model_sha256=sha_file(args.g_model/"g-models.joblib"),
        candidate_pool="all eligible documents from frozen sequence; no gain-based truncation",random_repeats=10,oracle_state_limit=1000,
        old_g=dict(status="not_comparable_f_binding_and_feature_contract",original_model_not_mislabelled_as_refit=True),
        source_requests=0,training_calls=0,actual_local_f_calls=evaluator.calls,actual_local_f_seconds=evaluator.seconds,
        elapsed_seconds=time.perf_counter()-started,full_system_cost_unknown=True,
        limitations=["All strategy is an uncapped reference; marked separately where it exceeds the comparison budget","Cache reuse timing is actual replay compute, not a prospective full-system savings estimate","Historical snapshot assumptions retained","Previously examined development records are not external unseen validation"] ))
    with (args.output/"strategy-comparison.csv").open("x",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    figures(args.output,summary,paths,pred_rows)
    if any(sha_file(p)!=digest for p,digest in bundle["source_sha256"].items()):raise ValueError("Frozen dataset changed during evaluation")
    save_json(args.output/"manifest.json",{p.name:sha_file(p) for p in sorted(args.output.iterdir()) if p.is_file()})
    print(json.dumps(dict(status="evaluation_complete",matches=len(panel.sequences),paths=len(paths),output=str(args.output)),ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument("--data",type=Path,required=True);p.add_argument("--vectors",type=Path,required=True)
    p.add_argument("--f-model",type=Path,required=True);p.add_argument("--g-model",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--roles",nargs="+",default=["validation","test"]);args=p.parse_args()
    import torch
    torch.set_num_threads(1)
    def offline(*a,**k):raise RuntimeError("Frozen offline evaluation")
    socket.create_connection=socket.socket.connect=socket.getaddrinfo=offline
    with threadpool_limits(limits=1):evaluate(args)


if __name__=="__main__":main()
