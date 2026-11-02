"""Frozen-Qwen paired split/seed stability study, never a new blind test.

Reuses exact existing corrected assignments. Text preprocessing for the lexical
control is refitted within each training split; pretrained Qwen is never refit.
No new corpus fetches, embedding inference, labels, f updates or runtime changes.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import socket
import sys
import time

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"tools"))
import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from football_research.acquisition.postread_semantic_embedding import build_semantic_design,validate_feature_rows
import train_postread_qwen as prior

DATA=prior.DATA
EMBED=prior.OUTPUT
PLAN=Path("E:/FootballMind/experiments/postread-reversal-corrected-20261025-v2/experiment-plan.json")
OUT=Path("E:/FootballMind/experiments/postread-qwen-stability-20261025-v1")
SEEDS=(20261025,20261026,20261027)
SPLITS=("train","validation","test")


def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def rows(p):return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").split("\n") if x.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    with Path(p).open("x",encoding="utf-8") as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def save_rows(p,rr):
    with Path(p).open("x",encoding="utf-8") as f:
        for r in rr:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+"\n")
def csv_save(p,rr):
    import csv
    with Path(p).open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)


def audit_assignment(assignments,features,documents):
    expected={r["physical_match_id"] for r in features}
    if set(assignments)!=expected or set(assignments.values())-set((*SPLITS,"unused")):
        raise ValueError("Existing exact whole-match assignment required")
    overlaps={}
    for field in ("text_sha256","source_url"):
        locations=defaultdict(set)
        for d in documents:
            split=assignments[d["physical_match_id"]]
            if split!="unused":locations[d[field]].add(split)
        overlaps[field]=[k for k,v in locations.items() if len(v)>1]
        if overlaps[field]:raise ValueError("Cross-split article overlap")
    return dict(counts={s:dict(matches=sum(v==s for v in assignments.values()),
        rows=sum(assignments[r["physical_match_id"]]==s for r in features)) for s in (*SPLITS,"unused")},
        exact_article_cross_split=overlaps)


def predictions(spec,x,mean,scale):
    if spec["kind"]=="zero":return np.zeros(len(x["state"]))
    if spec["kind"]=="constant":return np.full(len(x["state"]),spec["value"])
    return spec["estimator"].predict(spec["scaler"].transform(x[spec["input_key"]]))*scale+mean


def build_design(features,ti,vectors,helper):
    train=[features[i] for i in ti]
    enc=helper.fit_text_encoder([r["candidate_text"] for r in train]+[r["adopted_text"] for r in train],SEEDS[0])
    b=helper.encode(enc,[r["candidate_text"] for r in features]);a=helper.encode(enc,[r["adopted_text"] for r in features])
    if b.shape[1]!=32:raise ValueError("Same-shape control needs 32 lexical dimensions")
    q32=build_semantic_design(features,vectors,32);q256=build_semantic_design(features,vectors,256)
    state=q32["state"];sem=np.array([r["semantic_numeric"] for r in features])
    return dict(state=state,claims_state=q32["semantic"],
        lexical32=np.c_[state,sem,b,a,abs(b-a),b*a],
        qwen32=q32["fulltext_state"],qwen256=q256["fulltext_state"],
        qwen32_body=q32["body"],qwen32_no_candidate_claims=q32["fulltext_state_without_candidate_claims"],
        qwen32_no_adopted_text=np.c_[state,sem,q32["body"]]),enc


def fit_fold(name,assignment,all_features,label_by,samples,states,documents,vectors,helper,output,neural):
    directory=output/name
    if (directory/"summary.json").exists():
        summary=read(directory/"summary.json")
        if read(directory/"assignments.json")!=assignment:raise ValueError("Resume split changed")
        for relative,digest in read(directory/"manifest.json").items():
            if sha(directory/relative)!=digest:raise ValueError("Completed fold digest changed")
        print(json.dumps(dict(stage="reused_completed_fold",fold=name)),flush=True)
        return summary
    if directory.exists():raise FileExistsError("Incomplete fold requires investigation: "+str(directory))
    directory.mkdir(parents=True)
    save(directory/"assignments.json",assignment)
    audit=audit_assignment(assignment,all_features,documents);save(directory/"split-audit.json",audit)
    features=[dict(r,split=assignment[r["physical_match_id"]]) for r in all_features if assignment[r["physical_match_id"]]!="unused"]
    mids=np.array([r["physical_match_id"] for r in features])
    y=np.array([label_by[r["sample_id"]]["realized_gain"] for r in features],float)
    idx={s:np.array([i for i,r in enumerate(features) if r["split"]==s],dtype=int) for s in SPLITS}
    ti,vi=idx["train"],idx["validation"]
    if any(not len(i) for i in idx.values()):raise ValueError("Empty split")
    designs,encoder=build_design(features,ti,vectors,helper)
    w=helper.match_weights(mids[ti]);mean=float(np.average(y[ti],weights=w));scale=max(1e-8,float(np.sqrt(np.average((y[ti]-mean)**2,weights=w))))
    yt=(y[ti]-mean)/scale
    fitted=dict(zero=dict(kind="zero"),train_mean=dict(kind="constant",value=mean))
    for key in designs:
        scaler=StandardScaler().fit(designs[key][ti],sample_weight=w)
        estimator=Ridge(alpha=10).fit(scaler.transform(designs[key][ti]),yt,sample_weight=w)
        fitted[key+"_ridge"]=dict(kind="ridge",input_key=key,scaler=scaler,estimator=estimator)
    curves=[];checkpoints={}
    if neural:
        for key in ("lexical32","qwen32"):
            scaler=StandardScaler().fit(designs[key][ti],sample_weight=w)
            xt,xv=scaler.transform(designs[key][ti]),scaler.transform(designs[key][vi])
            for seed in SEEDS:
                model_name=f"{key}_mlp_seed{seed}"
                est=MLPRegressor(hidden_layer_sizes=(128,64),activation="relu",solver="adam",alpha=.1,
                    learning_rate_init=.001,batch_size=min(64,len(ti)),max_iter=1,early_stopping=False,
                    shuffle=True,random_state=seed)
                best_loss=float("inf");best_epoch=0;best=None
                for epoch in range(1,601):
                    est.partial_fit(xt,yt,sample_weight=w)
                    pt,pv=est.predict(xt)*scale+mean,est.predict(xv)*scale+mean
                    tl=helper.macro_mean((pt-y[ti])**2,mids[ti]);vl=helper.macro_mean((pv-y[vi])**2,mids[vi])
                    curves.append(dict(model=model_name,epoch=epoch,train_mse=tl,validation_mse=vl))
                    if vl<best_loss-1e-12:best_loss=vl;best_epoch=epoch;best=deepcopy(est)
                    if epoch-best_epoch>=60:break
                fitted[model_name]=dict(kind="mlp",input_key=key,scaler=scaler,estimator=best)
                checkpoints[model_name]=dict(selected_epoch=best_epoch,epochs_run=epoch,validation_mse=best_loss,
                    parameters=sum(v.size for v in best.coefs_)+sum(v.size for v in best.intercepts_))
    validation={n:helper.macro_mean((predictions(spec,designs,mean,scale)[vi]-y[vi])**2,mids[vi]) for n,spec in fitted.items()}
    selected={}
    if neural:
        for family in ("lexical32","qwen32"):
            selected[family+"_validation_selected_mlp"]=min((n for n in fitted if n.startswith(family+"_mlp_")),key=validation.get)
    save(directory/"selection-frozen.json",dict(validation_mse=validation,selected=selected,
        selection_inputs="validation only; test cohort has been used previously in development"))
    pred={n:predictions(spec,designs,mean,scale) for n,spec in fitted.items()}
    for alias,n in selected.items():pred[alias]=pred[n]
    metric_rows=[];prediction_rows=[];per_match=[]
    for model,p in pred.items():
        for split,ii in idx.items():
            metric_rows.append(dict(fold=name,model=model,split=split,**helper.metrics(y[ii],p[ii],mids[ii])))
            for mid in sorted(set(mids[ii])):
                jj=ii[mids[ii]==mid]
                per_match.append(dict(fold=name,model=model,split=split,physical_match_id=str(mid),rows=len(jj),
                    mse=float(np.mean((p[jj]-y[jj])**2)),mae=float(np.mean(abs(p[jj]-y[jj])))))
        prediction_rows.extend(dict(model=model,sample_id=r["sample_id"],physical_match_id=r["physical_match_id"],
            split=r["split"],predicted_gain=float(p[i]),realized_gain=float(y[i])) for i,r in enumerate(features))
    save(directory/"metrics.json",metric_rows);csv_save(directory/"metrics.csv",metric_rows)
    save_rows(directory/"predictions.jsonl",prediction_rows);csv_save(directory/"per-match-metrics.csv",per_match)
    save_rows(directory/"training-curves.jsonl",curves)
    score_maps={n:{r["sample_id"]:float(p[i]) for i,r in enumerate(features)} for n,p in pred.items()}
    # Policy helper only receives models' precomputed pre-decision scores and actual
    # adopted-state lookups; it does not sum marginal empty-state gains.
    active={r["physical_match_id"] for r in features}
    active_samples=[r for r in samples if r["physical_match_id"] in active]
    policy_names=["state_ridge","claims_state_ridge","lexical32_ridge","qwen32_ridge","qwen256_ridge",*selected]
    policies=prior.evaluate_policies(features,active_samples,states,score_maps,policy_names,directory)
    reload_diff=None
    if neural:
        bundle=dict(fitted=fitted,lexical_encoder=encoder,target_mean=mean,target_scale=scale,
            assignments=assignment,selected=selected,encoder_sha256=sha(EMBED/"embeddings.npz"),checkpoints=checkpoints)
        joblib.dump(bundle,directory/"models.joblib",compress=3)
        loaded=joblib.load(directory/"models.joblib")
        reload_diff=max(float(np.max(abs(pred[n]-predictions(spec,designs,loaded["target_mean"],loaded["target_scale"])))) for n,spec in loaded["fitted"].items())
        if reload_diff>1e-12:raise AssertionError("Serialized model predictions differ")
    summary=dict(fold=name,neural=neural,counts=audit["counts"],selected=selected,checkpoints=checkpoints,
        metrics=metric_rows,policies=policies,serialized_max_error=reload_diff)
    save(directory/"summary.json",summary)
    save(directory/"manifest.json",{p.name:sha(p) for p in sorted(directory.iterdir()) if p.is_file() and p.name!="manifest.json"})
    print(json.dumps(dict(stage="fold_complete",fold=name,selected=selected,
        test_mse={r["model"]:r["match_macro_mse"] for r in metric_rows if r["split"]=="test" and r["model"] in ["zero","lexical32_ridge","qwen32_ridge","qwen256_ridge",*selected]})),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,default=OUT)
    parser.add_argument("--named-only",action="store_true");args=parser.parse_args()
    def no_network(*a,**k):raise RuntimeError("Stability study is offline")
    socket.socket.connect=socket.create_connection=socket.getaddrinfo=no_network
    features=rows(DATA/"features.jsonl");validate_feature_rows(features)
    labels={r["sample_id"]:r for r in rows(DATA/"labels.jsonl")}
    if set(labels)!={r["sample_id"] for r in features}:raise ValueError("Feature/label ID mismatch")
    samples=rows(DATA/"samples.jsonl");states=rows(DATA/"states.jsonl");docs=rows(DATA/"documents.jsonl")
    plan=read(PLAN);er=read(EMBED/"embedding-report.json")
    if er["source_sha256"]!=sha(DATA/"features.jsonl") or er["vectors_file_sha256"]!=sha(EMBED/"embeddings.npz"):raise ValueError("Frozen embedding/source mismatch")
    with np.load(EMBED/"embeddings.npz",allow_pickle=False) as z:vectors={k:z[k] for k in z.files}
    f_path=ROOT/"var/f-entity-semantic-f-shadow-materialization-20261011-v1/model.json"
    if {r["f_model_sha256"] for r in labels.values()}!={sha(f_path)}:raise ValueError("Fixed-f label version differs")
    inputs=[DATA/n for n in ("features.jsonl","labels.jsonl","samples.jsonl","states.jsonl","documents.jsonl","split.json")]
    inputs += [PLAN,EMBED/"embeddings.npz",EMBED/"embedding-report.json",f_path]
    source_hashes={str(p):sha(p) for p in inputs}
    protocol=dict(schema="postread-qwen-paired-stability-v1",source_hashes=source_hashes,
        named_assignments=plan["named_splits"],random_assignments=plan["random_splits"],seeds=list(SEEDS),
        primary_contrast="Qwen32 vs train-only TFIDF/SVD32, same214 input,128x64 MLP, same seed and optimizer",
        additional_contrast="Ridge alpha10: lexical32, Qwen32, Qwen256, state/claim/body and adopted-text ablations",
        max_epochs=600,patience=60,learning_rate=.001,mlp_alpha=.1,target_weighting="equal match then equal transition",
        fixed_semantic_embeddings=True,new_embedding_inference=0,new_labels=0,new_independent_matches=0,
        split_scope="six existing corrected named splits +30 existing within18-cohort random resplits; correlated development evidence",
        test_used_for_selection=False,previous_test_exposure=True,production_ready=False)
    args.output.mkdir(parents=True,exist_ok=True)
    pp=args.output/"experiment-plan.json"
    if pp.exists():
        if read(pp)!=protocol:raise ValueError("Resume protocol differs")
    else:save(pp,protocol)
    helper=prior.load_helpers();result=[];started=time.perf_counter()
    with threadpool_limits(limits=2):
        for name,assignment in plan["named_splits"].items():
            result.append(fit_fold(name,assignment,features,labels,samples,states,docs,vectors,helper,args.output,True))
        if not args.named_only:
            for name,assignment in plan["random_splits"].items():
                result.append(fit_fold(name,assignment,features,labels,samples,states,docs,vectors,helper,args.output,False))
    if any(sha(Path(p))!=h for p,h in source_hashes.items()):raise AssertionError("Frozen input changed")
    save(args.output/("named-completion.json" if args.named_only else "training-completion.json"),
        dict(completed_folds=[r["fold"] for r in result],elapsed_seconds=time.perf_counter()-started,
            source_hashes_unchanged=True,source_http=0,embedding_inference=0))
    print(json.dumps(dict(stage="all_folds_completed",folds=len(result),elapsed_seconds=time.perf_counter()-started)),flush=True)


if __name__=="__main__":main()
