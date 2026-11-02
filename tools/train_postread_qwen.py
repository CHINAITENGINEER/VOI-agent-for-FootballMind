"""Paired frozen-encoder / frozen-f research experiment, original grouped split.

Only new heads are trained. Old models and labels are read-only. Test is already
inspected development evidence, never represented as a new blind evaluation.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import socket
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from football_research.acquisition.postread_semantic_embedding import (
    EncoderContract, build_semantic_design, predict_frozen_head, text_sha, validate_feature_rows)

DATA = Path("E:/FootballMind/experiments/postread-clean-expansion-20261025-v1")
OUTPUT = Path("E:/FootballMind/experiments/postread-qwen-semantic-20261025-v1")
SPLITS = ("train", "validation", "test")
SEEDS = (20261025, 20261026, 20261027)


def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def rows(path): return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").split("\n") if x.strip()]
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path, obj):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False); f.write("\n")
def save_rows(path, obj):
    with Path(path).open("x", encoding="utf-8") as f:
        for r in obj: f.write(json.dumps(r, ensure_ascii=False, allow_nan=False)+"\n")
def csv_save(path, obj):
    import csv
    with Path(path).open("x", encoding="utf-8-sig", newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(obj[0])); w.writeheader(); w.writerows(obj)


def load_helpers():
    path = DATA / "code/train_clean_postread_20261025.py"
    spec = importlib.util.spec_from_file_location("frozen_clean_experiment_helpers", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def evaluate_policies(features, samples, states, predictions, names, output):
    """Re-evaluate scores at each actual adopted state; outcome used for reporting only."""
    f_by={r["sample_id"]:r for r in features}
    look={(r["physical_match_id"],r["discovered_mask"],r["acquired_mask"],r["candidate_index"]):r for r in samples}
    st={(r["physical_match_id"],r["discovered_mask"],r["acquired_mask"]):r for r in states}
    group=defaultdict(list)
    for r in samples:
        if f_by[r["sample_id"]]["split"] in ("validation","test"): group[r["physical_match_id"]].append(r)
    details=[]
    for mid, rr in sorted(group.items()):
        n=int(rr[0]["candidate_count"]); dm=(1<<n)-1; split=f_by[rr[0]["sample_id"]]["split"]
        if any((mid,dm,m) not in st for m in range(1<<n)): raise ValueError("Incomplete state grid")
        initial=st[mid,dm,0]
        for budget in (1,2,3):
            k=min(budget,n)
            def finish(strategy, chosen, trace=None):
                after=st[mid,dm,sum(1<<i for i in chosen)]
                return dict(split=split, physical_match_id=mid, budget=budget, strategy=strategy,
                            candidate_count=n, selected_indices=list(chosen),
                            unselected_indices=[i for i in range(n) if i not in chosen],
                            f_updates=len(chosen), budget_compliant=len(chosen)<=budget,
                            p_before=initial["p"], p_after=after["p"], final_log_loss=after["loss"],
                            realized_gain=initial["loss"]-after["loss"], trace=trace or [],
                            monetary_cost=None, source_cost_already_incurred=True,
                            cost_note="Full-text pool and embeddings already materialized; no claimed acquisition savings")
            details += [finish("no_information",[]),finish("all_candidates",range(n)),finish("fixed_order",range(k))]
            orders=[finish("random_exact",p) for p in itertools.permutations(range(n),k)]
            random=dict(orders[0]); random.update(selected_indices=None,unselected_indices=None,p_after=None,
                final_log_loss=float(np.mean([r["final_log_loss"] for r in orders])),
                realized_gain=float(np.mean([r["realized_gain"] for r in orders])),
                trace=[dict(order=r["selected_indices"],loss=r["final_log_loss"]) for r in orders])
            details.append(random)
            best=min((m for m in range(1<<n) if m.bit_count()<=k),key=lambda m:(st[mid,dm,m]["loss"],m.bit_count(),m))
            details.append(finish("oracle_analysis_only",[i for i in range(n) if best&(1<<i)]))
            for name in names:
                chosen=[]; trace=[]
                while len(chosen)<k:
                    mask=sum(1<<i for i in chosen)
                    scored=[(i, predictions[name][look[mid,dm,mask,i]["sample_id"]]) for i in range(n) if i not in chosen]
                    best_i,value=max(scored,key=lambda x:(x[1],-x[0]))
                    trace.append(dict(acquired_mask=mask,scores=[dict(index=i,predicted_gain=v) for i,v in scored],selected=best_i if value>0 else None))
                    if value<=0: break
                    chosen.append(best_i)
                details.append(finish(name,chosen,trace))
    aggregates=[]
    for split,budget,strategy in sorted({(r["split"],r["budget"],r["strategy"]) for r in details}):
        rr=[r for r in details if (r["split"],r["budget"],r["strategy"])==(split,budget,strategy)]
        aggregates.append(dict(split=split,budget=budget,strategy=strategy,matches=len(rr),
            mean_final_log_loss=float(np.mean([r["final_log_loss"] for r in rr])),
            mean_realized_gain=float(np.mean([r["realized_gain"] for r in rr])),
            mean_f_updates=float(np.mean([r["f_updates"] for r in rr])),
            all_budget_compliant=all(r["budget_compliant"] for r in rr)))
    save(output/"strategy-comparison.json",dict(aggregates=aggregates,per_match=details))
    csv_save(output/"strategy-comparison.csv",aggregates)
    return aggregates


def render(output, metrics, curves, predictions, selected, policies):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names=sorted({r["model"] for r in curves})
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for ax,name in zip(axes.flat,names):
        rr=[r for r in curves if r["model"]==name]
        ax.plot([r["epoch"] for r in rr],[r["train_match_mse"] for r in rr],label="train")
        ax.plot([r["epoch"] for r in rr],[r["validation_match_mse"] for r in rr],label="validation")
        ax.set(title=name,yscale="log",xlabel="Epoch",ylabel="Match macro MSE");ax.legend()
    fig.suptitle("Frozen Qwen embedding, new g heads; original grouped split")
    fig.tight_layout();fig.savefig(output/"training-curves.png",dpi=160);plt.close(fig)
    rr=[r for r in predictions if r["model"]==selected]
    low=min(min(r["realized_gain"],r["predicted_gain"]) for r in rr)-.01
    high=max(max(r["realized_gain"],r["predicted_gain"]) for r in rr)+.01
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    for ax,split in zip(axes,SPLITS):
        points=[r for r in rr if r["split"]==split]
        ax.scatter([r["realized_gain"] for r in points],[r["predicted_gain"] for r in points],s=22,alpha=.65)
        ax.plot([low,high],[low,high],"k--",label="y=x reference, NOT a fitted line")
        ax.axhline(0,color="gray",lw=.5);ax.axvline(0,color="gray",lw=.5)
        ax.set(xlim=(low,high),ylim=(low,high),xlabel="Realized signed gain",ylabel="g predicted gain",title=f"{split}: {len(points)} transitions")
        ax.legend(fontsize=8)
    fig.suptitle(selected+"; test reused for exploratory comparison")
    fig.tight_layout();fig.savefig(output/"predicted-vs-realized.png",dpi=160);plt.close(fig)
    compare=["zero","old_fulltext_mlp","lexical_fulltext_mlp","lexical_title_ridge",selected]
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,split in zip(axes,("validation","test")):
        found=[next(r for r in metrics if r["model"]==n and r["split"]==split) for n in compare]
        ax.bar(range(len(compare)),[r["match_macro_mse"] for r in found],color=["#aaa","#bbb","#84a1ad","#c8b19a","#1b8066"])
        ax.set_xticks(range(len(compare)),compare,rotation=30,ha="right",fontsize=8)
        ax.set(title=split,ylabel="Match macro gain MSE (lower is better)")
    fig.tight_layout();fig.savefig(output/"embedding-comparison.png",dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    for name in ["no_information","all_candidates","fixed_order","random_exact","lexical_fulltext_mlp",selected]:
        rr=[r for r in policies if r["split"]=="test" and r["strategy"]==name and r["all_budget_compliant"]]
        ax.plot([r["budget"] for r in rr],[r["mean_final_log_loss"] for r in rr],marker="o",label=name)
    ax.set(xlabel="Maximum f updates",ylabel="Mean final log loss",title="Same six previously viewed test matches; lower is better")
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(output/"budget-performance.png",dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,default=OUTPUT)
    args=parser.parse_args();out=args.output/"training"
    if out.exists(): raise FileExistsError("Independent training directory required")
    def no_network(*a,**k): raise RuntimeError("Head training has no network access")
    socket.socket.connect=socket.create_connection=socket.getaddrinfo=no_network
    helper=load_helpers()
    features=rows(DATA/"features.jsonl");validate_feature_rows(features)
    labels={r["sample_id"]:r for r in rows(DATA/"labels.jsonl")}
    if len(labels)!=len(features) or set(labels)!={r["sample_id"] for r in features}: raise ValueError("Label IDs not aligned")
    summary=read(args.output/"embedding-report.json")
    if summary["source_sha256"]!=sha(DATA/"features.jsonl"): raise ValueError("Embedding/data version mismatch")
    if summary["vectors_file_sha256"]!=sha(args.output/"embeddings.npz"): raise ValueError("Embedding digest mismatch")
    with np.load(args.output/"embeddings.npz",allow_pickle=False) as z: vectors={k:z[k] for k in z.files}
    designs={dim:build_semantic_design(features,vectors,dim) for dim in (32,256)}
    mids=np.array([r["physical_match_id"] for r in features]);idx={s:np.array([i for i,r in enumerate(features) if r["split"]==s]) for s in SPLITS}
    if any(set(mids[idx[a]])&set(mids[idx[b]]) for a,b in itertools.combinations(SPLITS,2)): raise ValueError("Cross-split match")
    y=np.array([labels[r["sample_id"]]["realized_gain"] for r in features],dtype=float)
    if not np.isfinite(y).all(): raise ValueError("Nonfinite target")
    f_shas={r["f_model_sha256"] for r in labels.values()}
    if len(f_shas)!=1: raise ValueError("f version mixture")
    f_path=ROOT/"var/f-entity-semantic-f-shadow-materialization-20261011-v1/model.json"
    if sha(f_path)!=next(iter(f_shas)): raise ValueError("Frozen f digest mismatch")
    sources=[DATA/n for n in ("features.jsonl","labels.jsonl","samples.jsonl","states.jsonl","split.json","documents.jsonl")]
    sources += [DATA/"training/models.joblib",DATA/"training/predictions.jsonl",DATA/"training/model.json",f_path]
    source_hashes={str(p):sha(p) for p in sources}
    out.mkdir()
    protocol=dict(schema="postread-qwen-semantic-g-experiment-v1",encoder=asdict(EncoderContract()),
        dimensions=[32,256],state_dimension=57,claim_dimension=29,input_dimensions={"32":214,"256":1110},
        mlp_architectures={"32":[128,64],"256":[32]},seeds=list(SEEDS),
        alpha=.1,ridge_alpha=10,learning_rate=.001,max_epochs=600,patience=60,
        controls="32-dimensional same architecture/seed lexical comparison; 256-dimensional similar parameter count",
        candidate_claim_ablation="256-dimensional Ridge without candidate semantic_numeric; existing state claims retained",
        selection="validation match macro MSE; all results retained; no test selection",
        target_scaling="weighted train mean/sd",sample_weight="equal match then equal transition",
        scaler="StandardScaler, weighted train fit, original experiment convention unchanged",
        splits={s:dict(matches=len(set(mids[ii])),rows=len(ii)) for s,ii in idx.items()},
        test_role="previously_inspected_exploratory_not_new_blind_test",f_model_sha256=next(iter(f_shas)),
        source_hashes=source_hashes,runtime_modified=False,production_ready=False,
        source_or_provider_http=0,encoder_training=False)
    save(out/"protocol.json",protocol)
    ti,vi=idx["train"],idx["validation"];w=helper.match_weights(mids[ti])
    mean=float(np.average(y[ti],weights=w));scale=float(np.sqrt(np.average((y[ti]-mean)**2,weights=w)))
    yt=(y[ti]-mean)/scale
    fitted={"zero":dict(kind="zero"),"train_mean":dict(kind="constant",value=mean)}
    checkpoints={};curves=[]
    def fit_ridge(name,dim,key):
        x=designs[dim][key];scaler=StandardScaler().fit(x[ti],sample_weight=w)
        est=Ridge(alpha=10).fit(scaler.transform(x[ti]),yt,sample_weight=w)
        fitted[name]=dict(kind="ridge",dimension=dim,input_key=key,scaler=scaler,estimator=est)
    fit_ridge("state_ridge",32,"state");fit_ridge("claims_state_ridge",32,"semantic")
    for dim in (32,256): fit_ridge(f"qwen{dim}_ridge",dim,"fulltext_state")
    fit_ridge("qwen256_without_candidate_claims_ridge",256,"fulltext_state_without_candidate_claims")
    with threadpool_limits(limits=2):
        for dim,hidden in ((32,(128,64)),(256,(32,))):
            x=designs[dim]["fulltext_state"];scaler=StandardScaler().fit(x[ti],sample_weight=w)
            xt=scaler.transform(x[ti]);xv=scaler.transform(x[vi])
            for seed in SEEDS:
                name=f"qwen{dim}_mlp_seed{seed}";best_loss=float("inf");best_epoch=0;best=None
                est=MLPRegressor(hidden_layer_sizes=hidden,activation="relu",solver="adam",alpha=.1,
                    learning_rate_init=.001,batch_size=64,max_iter=1,early_stopping=False,shuffle=True,random_state=seed)
                for epoch in range(1,601):
                    est.partial_fit(xt,yt,sample_weight=w)
                    pt=est.predict(xt)*scale+mean;pv=est.predict(xv)*scale+mean
                    train_loss=helper.macro_mean((pt-y[ti])**2,mids[ti]);val_loss=helper.macro_mean((pv-y[vi])**2,mids[vi])
                    curves.append(dict(model=name,epoch=epoch,train_match_mse=train_loss,validation_match_mse=val_loss))
                    if val_loss<best_loss-1e-12: best_loss=val_loss;best_epoch=epoch;best=deepcopy(est)
                    if epoch-best_epoch>=60: break
                fitted[name]=dict(kind="mlp",dimension=dim,input_key="fulltext_state",scaler=scaler,estimator=best)
                checkpoints[name]=dict(selected_epoch=best_epoch,epochs_run=epoch,hidden=list(hidden),
                    trainable_parameters=sum(a.size for a in best.coefs_)+sum(a.size for a in best.intercepts_),validation_mse=best_loss)
                print(json.dumps(dict(stage="head_trained",model=name,**checkpoints[name])),flush=True)
    bundle=dict(fitted=fitted,target_mean=mean,target_scale=scale,encoder_contract=asdict(EncoderContract()),
                state_feature_names=features[0]["state_feature_names"],semantic_feature_names=features[0]["semantic_feature_names"])
    # Only validation labels enter selection. Old predictions are subsequently compared, not used for training.
    val_scores={name:helper.macro_mean((predict_frozen_head(bundle,[features[i] for i in vi],vectors,name)-y[vi])**2,mids[vi]) for name in fitted}
    selected=min((n for n in val_scores if n.startswith("qwen")),key=val_scores.get)
    overall=min(val_scores,key=val_scores.get);bundle["selected_semantic_model"]=selected
    save(out/"selection-frozen.json",dict(selected_semantic_model=selected,selected_including_baselines=overall,
        validation_match_mse=val_scores,test_used_for_this_selection=False,test_previously_inspected=True))
    joblib.dump(bundle,out/"models.joblib",compress=3);loaded=joblib.load(out/"models.joblib")
    preds={};metrics=[];pred_rows=[];reload_diff=0.
    for name in fitted:
        pp=predict_frozen_head(bundle,features,vectors,name)
        reload_diff=max(reload_diff,float(np.max(abs(pp-predict_frozen_head(loaded,features,vectors,name)))))
        preds[name]={r["sample_id"]:float(p) for r,p in zip(features,pp)}
    old_specs=[(DATA/"training/predictions.jsonl","mlp_128_64_seed20261026","lexical_fulltext_mlp"),
               (DATA/"training/predictions.jsonl","title_local_ridge","lexical_title_ridge"),
               (Path("E:/FootballMind/experiments/unified-f-postread-g-expansion-20261025-v1/training/predictions.jsonl"),"mlp_128_64_seed20261027","old_fulltext_mlp")]
    # The old best model's actual name is taken from its saved validation scores.
    old_training=old_specs[-1][0].parent
    old_selection=read(old_training/"selection-frozen.json")
    old_vals=old_selection["validation_match_mse"]
    old_name=min((n for n in old_vals if n.startswith("mlp_")),key=old_vals.get)
    old_specs[-1]=(old_specs[-1][0],old_name,"old_fulltext_mlp")
    old_specs.extend((DATA/"training/predictions.jsonl",f"mlp_128_64_seed{seed}",f"lexical_fulltext_seed{seed}") for seed in SEEDS)
    baseline_alignment=[]
    for path,name,alias in old_specs:
        rr=[r for r in rows(path) if r["model"]==name];by={r["sample_id"]:r for r in rr}
        if len(by)!=len(rr): raise ValueError("Duplicate baseline prediction IDs")
        valid={r["sample_id"] for r in features if r["sample_id"] in by}
        for sid in valid:
            if abs(by[sid]["realized_gain"]-labels[sid]["realized_gain"])>1e-12: raise ValueError("Baseline target differs")
        for split in ("validation","test"):
            if any(features[i]["sample_id"] not in valid for i in idx[split]): raise ValueError("Missing baseline holdout")
        preds[alias]={sid:by[sid]["predicted_gain"] for sid in valid}
        baseline_alignment.append(dict(alias=alias,saved_model=name,path=str(path),sha256=sha(path),matched_rows=len(valid),labels_verified=True))
    for name,by in preds.items():
        for split,ii in idx.items():
            use=np.array([i for i in ii if features[i]["sample_id"] in by],dtype=int)
            if len(use)!=len(ii): continue # No invented predictions for old model's extra training rows.
            pp=np.array([by[features[i]["sample_id"]] for i in use])
            metrics.append(dict(model=name,split=split,**helper.metrics(y[use],pp,mids[use])))
        pred_rows.extend(dict(model=name,sample_id=r["sample_id"],physical_match_id=r["physical_match_id"],split=r["split"],
            predicted_gain=by[r["sample_id"]],realized_gain=float(y[i])) for i,r in enumerate(features) if r["sample_id"] in by)
    save(out/"metrics.json",metrics);csv_save(out/"metrics.csv",metrics);save_rows(out/"predictions.jsonl",pred_rows)
    save_rows(out/"training-curves.jsonl",curves);save(out/"baseline-alignment.json",baseline_alignment)
    names=["old_fulltext_mlp","lexical_fulltext_mlp","lexical_title_ridge",selected]
    policies=evaluate_policies(features,rows(DATA/"samples.jsonl"),rows(DATA/"states.jsonl"),preds,names,out)
    render(out,metrics,curves,pred_rows,selected,policies)
    save(out/"model.json",dict(schema="postread-qwen-semantic-g-v1",selected_semantic_model=selected,
        selected_including_baselines=overall,encoder=asdict(EncoderContract()),feature_dimensions=protocol["input_dimensions"],
        checkpoints=checkpoints,model_sha256=sha(out/"models.joblib"),f_model_sha256=next(iter(f_shas)),
        state_feature_names=features[0]["state_feature_names"],semantic_feature_names=features[0]["semantic_feature_names"],
        production_ready=False,formal_runtime_integrated=False,embedding_file=str(args.output/"embeddings.npz"),
        limitations=["Six validation and six reused test matches; not new blind evidence",
                     "Existing manually reviewed claim inputs retained except declared ablation",
                     "Pretrained model corpus and historical football results exposure unknown",
                     "Frozen f limitations and original timestamp assumptions persist",
                     "Semantic embeddings do not themselves resolve contradiction or prove predictive value"]))
    unchanged=all(sha(Path(p))==h for p,h in source_hashes.items())
    if not unchanged or reload_diff>1e-12: raise AssertionError("Source mutation or serialized inference mismatch")
    save(out/"verification.json",dict(source_hashes_unchanged=unchanged,serialized_prediction_max_difference=reload_diff,
        grouped_split_disjoint=True,feature_label_ids_aligned=True,all_gain_signs_retained=dict(Counter("positive" if v>1e-12 else "negative" if v< -1e-12 else "zero" for v in y)),
        test_not_used_in_this_checkpoint_selection=True,source_http=0,provider_api_calls=0,f_unchanged=True))
    print(json.dumps(dict(stage="complete",selected=selected,selected_including_baselines=overall,
        metrics=[r for r in metrics if r["model"] in names+["zero"] and r["split"]!="train"],output=str(out))),flush=True)


if __name__=="__main__": main()
