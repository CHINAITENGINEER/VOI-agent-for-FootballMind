"""Independent read-only checks of saved postread semantic stability results.

Does not import the training runner or fit any estimator/preprocessor. Recreates
numeric designs from the frozen vector file and serialized lexical transform.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
import joblib
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
DATA=Path("E:/FootballMind/experiments/postread-clean-expansion-20261025-v1")
VECTORS=Path("E:/FootballMind/experiments/postread-qwen-semantic-20261025-v1")
ORIGINAL=Path("E:/FootballMind/experiments/postread-reversal-corrected-20261025-v2/experiment-plan.json")
OUTPUT=Path("E:/FootballMind/experiments/postread-qwen-stability-20261025-v1")
SPLITS=("train","validation","test")


def read(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def rows(p): return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").split("\n") if x.strip()]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def text_sha(t): return hashlib.sha256(t.encode("utf-8")).hexdigest()
def check(condition,message):
    if not condition: raise AssertionError(message)


def aggregate_predictions(rr,feature_by,label_by,assignment):
    """Prove alignment through IDs before independently aggregating errors."""
    seen=set();groups=defaultdict(list);absolute=[]
    for r in rr:
        sid=r["sample_id"]
        check(sid not in seen,"Duplicate prediction ID")
        seen.add(sid)
        check(sid in feature_by and sid in label_by,"Unknown prediction ID")
        check(r["physical_match_id"]==feature_by[sid]["physical_match_id"],"Prediction match differs from ID")
        check(r["split"]==assignment[r["physical_match_id"]],"Prediction split differs from assignment")
        check(abs(r["realized_gain"]-label_by[sid]["realized_gain"])<1e-12,"Saved target differs from label")
        check(np.isfinite([r["predicted_gain"],r["realized_gain"]]).all(),"Nonfinite prediction or label")
        residual=r["predicted_gain"]-r["realized_gain"]
        groups[r["physical_match_id"]].append((residual**2,abs(residual)))
        absolute.append((residual**2,abs(residual)))
    return dict(rows=len(rr),matches=len(groups),
        row_mse=float(np.mean([v[0] for v in absolute])),row_mae=float(np.mean([v[1] for v in absolute])),
        match_macro_mse=float(np.mean([np.mean([v[0] for v in bucket]) for bucket in groups.values()])),
        match_macro_mae=float(np.mean([np.mean([v[1] for v in bucket]) for bucket in groups.values()])))


def independent_design(features,encoder,vectors):
    def normalize(x):
        x=np.asarray(x);norm=np.linalg.norm(x,axis=1,keepdims=True)
        return np.divide(x,norm,out=np.zeros_like(x),where=norm>0)
    def lexical(field):
        x=encoder["vectorizer"].transform([r[field] for r in features])
        x=encoder["svd"].transform(x) if encoder["svd"] is not None else x.toarray()
        return normalize(x)
    def qwen(field,dim):
        # Prefix normalization occurs in float32 in the actual representation.
        result=[]
        for r in features:
            v=vectors[text_sha(r[field])][:dim].copy() if r[field].strip() else np.zeros(dim,np.float32)
            norm=np.linalg.norm(v)
            result.append(v/norm if norm>0 else v)
        return np.stack(result)
    state=np.array([r["state_numeric"] for r in features],float)
    sem=np.array([r["semantic_numeric"] for r in features],float)
    b,a=lexical("candidate_text"),lexical("adopted_text")
    qb,qa=qwen("candidate_text",32),qwen("adopted_text",32)
    qb2,qa2=qwen("candidate_text",256),qwen("adopted_text",256)
    return dict(state=state,claims_state=np.c_[state,sem],
        lexical32=np.c_[state,sem,b,a,abs(b-a),b*a],
        qwen32=np.c_[state,sem,qb,qa,abs(qb-qa),qb*qa],
        qwen256=np.c_[state,sem,qb2,qa2,abs(qb2-qa2),qb2*qa2],
        qwen32_body=qb,qwen32_no_candidate_claims=np.c_[state,qb,qa,abs(qb-qa),qb*qa],
        qwen32_no_adopted_text=np.c_[state,sem,qb])


def check_bundle(directory,features,label_by,assignment,vectors,pred_by):
    bundle=joblib.load(directory/"models.joblib")
    check(bundle["assignments"]==assignment,"Serialized split differs")
    ti=np.array([i for i,r in enumerate(features) if r["split"]=="train"])
    ids=[features[i]["physical_match_id"] for i in ti];counts=Counter(ids)
    w=np.array([len(ids)/(len(counts)*counts[m]) for m in ids])
    y=np.array([label_by[r["sample_id"]]["realized_gain"] for r in features])
    mean=float(np.average(y[ti],weights=w));sd=float(np.sqrt(np.average((y[ti]-mean)**2,weights=w)))
    check(abs(mean-bundle["target_mean"])<1e-12 and abs(max(sd,1e-8)-bundle["target_scale"])<1e-12,"Target scaling not train-only weighted stats")
    enc=bundle["lexical_encoder"]
    texts=sorted({features[i][k] for i in ti for k in ("candidate_text","adopted_text") if features[i][k].strip()})
    check(set(enc["train_text_sha256"])=={text_sha(t) for t in texts},"Lexical fit text scope differs from train split")
    vec=enc["vectorizer"];vocab=vec.get_feature_names_out();ana=vec.build_analyzer()
    count=Counter(term for t in texts for term in set(ana(t)))
    expected_idf=np.log((1+len(texts))/(1+np.array([count[t] for t in vocab])))+1
    idf_error=float(np.max(abs(expected_idf-vec.idf_)))
    check(idf_error<1e-12,"TFIDF IDF differs from training-only document counts")
    design=independent_design(features,enc,vectors);checks=[]
    for name,spec in bundle["fitted"].items():
        if "scaler" not in spec:continue
        x=design[spec["input_key"]];scaler=spec["scaler"]
        # sklearn validates sample_weight with the input dtype before its
        # float64 accumulator. Body-only arrays are float32; fused arrays float64.
        scaler_weights=w.astype(x.dtype).astype(np.float64)
        expected_mean=np.average(x[ti].astype(np.float64),axis=0,weights=scaler_weights)
        expected_var=np.average((x[ti].astype(np.float64)-expected_mean)**2,axis=0,weights=scaler_weights)
        check(np.allclose(scaler.mean_,expected_mean,rtol=1e-10,atol=1e-10),"Scaler mean not train-only weighted mean")
        check(np.allclose(scaler.var_,expected_var,rtol=1e-9,atol=1e-10),"Scaler variance not train-only weighted variance")
        check(np.isfinite(scaler.var_).all() and np.isfinite(scaler.scale_).all(),"Nonfinite scaler state")
        transformed=scaler.transform(x)
        check(np.isfinite(transformed).all(),"Nonfinite transformed values")
        p=spec["estimator"].predict(transformed)*bundle["target_scale"]+bundle["target_mean"]
        pp=np.array([pred_by[name][r["sample_id"]]["predicted_gain"] for r in features])
        error=float(np.max(abs(p-pp)))
        check(error<1e-9,"Independent serialized inference differs")
        negative=np.flatnonzero(scaler.var_<0)
        negative_details=[dict(index=int(i),feature=(features[0]["state_feature_names"][i] if i<len(features[0]["state_feature_names"]) else "text_or_claim_coordinate"),
            var=float(scaler.var_[i]),scale=float(scaler.scale_[i]),training_unique=np.unique(x[ti,i]).tolist(),independent_variance=float(expected_var[i])) for i in negative]
        if len(negative):
            check(np.max(abs(scaler.var_[negative]))<1e-12 and np.all(scaler.scale_[negative]==1),"Negative variance is not harmless near-zero constant handling")
        checks.append(dict(model=name,input_shape=list(x.shape),max_prediction_difference=error,
            scaler_input_dtype=str(x.dtype),sample_weight_cast_matches_sklearn_input_dtype=True,
            max_scaler_mean_difference=float(np.max(abs(expected_mean-scaler.mean_))),
            max_scaler_variance_difference=float(np.max(abs(expected_var-scaler.var_))),
            all_transformed_finite=True,negative_variance=negative_details))
    return dict(train_target_mean=mean,train_target_sd=sd,unique_training_texts=len(texts),idf_max_difference=idf_error,models=checks)


def check_policy(directory,pred_by,features,samples,states):
    policies=read(directory/"strategy-comparison.json")
    lookup={(r["physical_match_id"],r["discovered_mask"],r["acquired_mask"],r["candidate_index"]):r["sample_id"] for r in samples}
    st={(r["physical_match_id"],r["discovered_mask"],r["acquired_mask"]):r for r in states}
    checked=0;trace_scores=0
    for r in policies["per_match"]:
        mid=r["physical_match_id"];n=r["candidate_count"];dm=(1<<n)-1
        initial=st[mid,dm,0]
        if r["strategy"]=="random_exact":
            possible=[st[mid,dm,sum(1<<i for i in c)]["loss"] for c in itertools.combinations(range(n),min(n,r["budget"]))]
            expected=float(np.mean(possible))
        else:
            chosen=r["selected_indices"]
            check(len(chosen)==len(set(chosen)) and all(0<=i<n for i in chosen),"Duplicate/invalid policy adoption")
            after=st[mid,dm,sum(1<<i for i in chosen)]
            expected=after["loss"]
            check(np.allclose(r["p_after"],after["p"],atol=1e-12,rtol=0),"Policy after probability differs from frozen state")
        check(abs(expected-r["final_log_loss"])<1e-12,"Policy final log loss differs")
        check(abs(initial["loss"]-expected-r["realized_gain"])<1e-12,"Policy realized gain differs")
        check(r["budget_compliant"]==(r["f_updates"]<=r["budget"]),"Budget compliance flag wrong")
        if r["strategy"] in pred_by:
            model=r["strategy"];chosen=[]
            for trace in r["trace"]:
                mask=sum(1<<i for i in chosen)
                check(trace["acquired_mask"]==mask,"Policy state order differs")
                expected_scores={i:pred_by[model][lookup[mid,dm,mask,i]]["predicted_gain"] for i in range(n) if i not in chosen}
                check({x["index"] for x in trace["scores"]}==set(expected_scores),"Incomplete policy menu")
                for item in trace["scores"]:
                    check(abs(item["predicted_gain"]-expected_scores[item["index"]])<1e-12,"Policy score differs from sample ID prediction")
                    trace_scores+=1
                best=max(expected_scores,key=lambda i:(expected_scores[i],-i))
                expected_choice=best if expected_scores[best]>0 else None
                check(trace["selected"]==expected_choice,"Policy choice not max predecision score/stop")
                if expected_choice is not None:chosen.append(expected_choice)
            check(chosen==r["selected_indices"],"Policy adopted list differs from trace")
        checked+=1
    for a in policies["aggregates"]:
        rr=[r for r in policies["per_match"] if all(r[k]==a[k] for k in ("split","budget","strategy"))]
        check(len(rr)==a["matches"],"Policy aggregate match count differs")
        for raw,field in (("final_log_loss","mean_final_log_loss"),("realized_gain","mean_realized_gain"),("f_updates","mean_f_updates")):
            check(abs(float(np.mean([r[raw] for r in rr]))-a[field])<1e-12,"Policy aggregate differs")
    return dict(paths_checked=checked,predecision_scores_checked=trace_scores,aggregate_rows=len(policies["aggregates"]))


def verify(output,allow_partial=False):
    plan=read(output/"experiment-plan.json");original=read(ORIGINAL)
    check(plan["named_assignments"]==original["named_splits"],"Named split assignments changed")
    check(plan["random_assignments"]==original["random_splits"],"Random split assignments changed")
    source_checks={p:sha(Path(p))==digest for p,digest in plan["source_hashes"].items()}
    check(all(source_checks.values()),"Frozen source modified")
    features=rows(DATA/"features.jsonl");feature_by={r["sample_id"]:r for r in features}
    labels={r["sample_id"]:r for r in rows(DATA/"labels.jsonl")}
    check(len(feature_by)==len(features)==len(labels),"Duplicate or missing feature/label ID")
    samples=rows(DATA/"samples.jsonl");states=rows(DATA/"states.jsonl")
    with np.load(VECTORS/"embeddings.npz",allow_pickle=False) as z:vectors={k:z[k] for k in z.files}
    results=[];pending=[];metric_count=0
    checkpoints=output/"verification-independent/fold-checks"
    checkpoints.mkdir(parents=True,exist_ok=True)
    verifier_sha=sha(Path(__file__))
    for name,assignment in {**plan["named_assignments"],**plan["random_assignments"]}.items():
        directory=output/name
        if not (directory/"manifest.json").exists():pending.append(name);continue
        cache_file=checkpoints/(name+".json")
        if cache_file.exists():
            cached=read(cache_file)
            if cached.get("verifier_sha256")==verifier_sha and cached.get("fold_manifest_sha256")==sha(directory/"manifest.json"):
                results.append(cached["result"]);metric_count+=cached["result"]["metric_groups"]
                continue
        check(read(directory/"assignments.json")==assignment,"Fold assignment differs")
        manifest=read(directory/"manifest.json")
        check(all(sha(directory/k)==v for k,v in manifest.items()),"Fold artifact digest changed")
        rr=rows(directory/"predictions.jsonl");model_rows=defaultdict(list)
        for r in rr:model_rows[r["model"]].append(r)
        expected_ids={sid for sid,r in feature_by.items() if assignment[r["physical_match_id"]]!="unused"}
        pred_by={};recomputed=[]
        for model,records in model_rows.items():
            check(len(records)==len(expected_ids) and {r["sample_id"] for r in records}==expected_ids,"Prediction ID coverage differs")
            pred_by[model]={r["sample_id"]:r for r in records}
            for split in SPLITS:
                values=[r for r in records if r["split"]==split]
                recomputed.append(dict(model=model,split=split,**aggregate_predictions(values,feature_by,labels,assignment)))
        saved={(r["model"],r["split"]):r for r in read(directory/"metrics.json")}
        errors=[]
        for r in recomputed:
            s=saved[r["model"],r["split"]]
            for field in ("row_mse","row_mae","match_macro_mse","match_macro_mae"):
                error=abs(r[field]-s[field]);errors.append(error)
                check(error<1e-12,"Saved metric differs from independent recomputation")
            check(r["rows"]==s["rows"] and r["matches"]==s["independent_matches"],"Saved count differs")
        selection=read(directory/"selection-frozen.json")
        rv={(r["model"],r["split"]):r for r in recomputed}
        for model,value in selection["validation_mse"].items():
            check(abs(value-rv[model,"validation"]["match_macro_mse"])<1e-12,"Validation selection score differs")
        for alias,selected in selection["selected"].items():
            family=alias.split("_validation_selected")[0]
            best=min((k for k in selection["validation_mse"] if k.startswith(family+"_mlp_seed")),key=selection["validation_mse"].get)
            check(best==selected,"Selected seed is not validation optimum")
            check(all(pred_by[alias][sid]["predicted_gain"]==pred_by[selected][sid]["predicted_gain"] for sid in expected_ids),"Selected prediction alias differs")
        active=[dict(r,split=assignment[r["physical_match_id"]]) for r in features if assignment[r["physical_match_id"]]!="unused"]
        bundle_check=check_bundle(directory,active,labels,assignment,vectors,pred_by) if (directory/"models.joblib").exists() else None
        policy_check=check_policy(directory,pred_by,features,samples,states)
        summary=read(directory/"summary.json")
        cc=rows(directory/"training-curves.jsonl")
        for model,checkpoint in summary["checkpoints"].items():
            curve=[r for r in cc if r["model"]==model];best_loss=float("inf");best_epoch=0
            for r in curve:
                if r["validation_mse"]<best_loss-1e-12:best_loss=r["validation_mse"];best_epoch=r["epoch"]
            check(best_epoch==checkpoint["selected_epoch"] and len(curve)==checkpoint["epochs_run"],"Checkpoint does not reproduce validation-only early stopping")
        result=dict(fold=name,models=len(model_rows),prediction_rows=len(rr),metric_groups=len(recomputed),
            max_metric_difference=max(errors),manifest_files=len(manifest),bundle_checks=bundle_check,policy_checks=policy_check)
        results.append(result)
        cache_file.write_text(json.dumps(dict(verifier_sha256=verifier_sha,fold_manifest_sha256=sha(directory/"manifest.json"),result=result),ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        metric_count+=len(recomputed)
        print(json.dumps(dict(stage="fold_verified",fold=name,metric_groups=len(recomputed))),flush=True)
    check(allow_partial or not pending,"Not all 36 folds complete")
    reproduction=[]
    if (output/"P0_original/manifest.json").exists():
        current=rows(output/"P0_original/predictions.jsonl")
        lookup={(r["model"],r["sample_id"]):r for r in current}
        for old_path,prefix in ((DATA/"training/predictions.jsonl","lexical"),(VECTORS/"training/predictions.jsonl","qwen")):
            old=rows(old_path)
            for seed in (20261025,20261026,20261027):
                old_name=f"mlp_128_64_seed{seed}" if prefix=="lexical" else f"qwen32_mlp_seed{seed}"
                new_name=f"{prefix}32_mlp_seed{seed}"
                matched=[r for r in old if r["model"]==old_name]
                check(len(matched)==289,"P0 previous predictions incomplete")
                diff=max(abs(r["predicted_gain"]-lookup[new_name,r["sample_id"]]["predicted_gain"]) for r in matched)
                if prefix=="qwen":check(diff<1e-12,"P0 frozen Qwen numerical reproduction differs")
                elif seed!=20261025:check(diff<1e-8,"P0 selected/other lexical seed reproduction differs")
                reproduction.append(dict(model=new_name,rows=len(matched),maximum_prediction_difference=diff,
                    within_1e8=bool(diff<1e-8),classification="numerically_reproduced" if diff<1e-8 else "refit_numerical_drift"))
    drift=[]
    if any(not r["within_1e8"] for r in reproduction):
        old=joblib.load(DATA/"training/models.joblib");new=joblib.load(output/"P0_original/models.joblib")
        a=old["fitted"]["mlp_128_64_seed20261025"];b=new["fitted"]["lexical32_mlp_seed20261025"]
        om=read(DATA/"training/model.json")["mlp_checkpoints"]["mlp_128_64_seed20261025"]
        nm=read(output/"P0_original/summary.json")["checkpoints"]["lexical32_mlp_seed20261025"]
        check(a["estimator"].get_params()==b["estimator"].get_params(),"Lexical seed25 estimator config changed")
        check(om["selected_epoch"]==nm["selected_epoch"] and om["epochs_run"]==nm["epochs_run"],"Lexical seed25 checkpoint changed")
        check(old["target_mean"]==new["target_mean"] and old["target_scale"]==new["target_scale"],"Lexical seed25 target scale changed")
        check(np.array_equal(old["body_encoder"]["vectorizer"].get_feature_names_out(),new["lexical_encoder"]["vectorizer"].get_feature_names_out()),"Lexical vocabulary changed")
        ometrics=read(DATA/"training/metrics.json");nmetrics=read(output/"P0_original/metrics.json")
        mse_diffs=[]
        for split in SPLITS:
            o=next(r for r in ometrics if r["model"]=="mlp_128_64_seed20261025" and r["split"]==split)
            n=next(r for r in nmetrics if r["model"]=="lexical32_mlp_seed20261025" and r["split"]==split)
            mse_diffs.append(dict(split=split,old_mse=o["match_macro_mse"],new_mse=n["match_macro_mse"],difference=n["match_macro_mse"]-o["match_macro_mse"]))
        drift.append(dict(model="lexical32_mlp_seed20261025",classification="refit_numerical_drift",
            target_scale_identical=True,estimator_config_identical=True,vocabulary_identical=True,
            selected_epoch=nm["selected_epoch"],epochs_run=nm["epochs_run"],
            idf_max_difference=float(np.max(abs(old["body_encoder"]["vectorizer"].idf_-new["lexical_encoder"]["vectorizer"].idf_))),
            svd_max_difference=float(np.max(abs(old["body_encoder"]["svd"].components_-new["lexical_encoder"]["svd"].components_))),
            scaler_mean_max_difference=float(np.max(abs(a["scaler"].mean_-b["scaler"].mean_))),
            scaler_scale_max_difference=float(np.max(abs(a["scaler"].scale_-b["scaler"].scale_))),
            mse_differences=mse_diffs,
            interpretation="Not an exact refit reproduction. Tiny preprocessing differences are observed; nonlinear optimization amplification is a plausible explanation, not a separately established causal result."))
    report=dict(status="partial" if pending else "passed_with_recorded_refit_drift" if drift else "passed",completed_folds=len(results),pending_folds=pending,
        independently_recomputed_metric_groups=metric_count,folds=results,source_unchanged=source_checks,
        P0_reproduction=reproduction,refit_drift=drift,training_performed=False,embedding_inference=False,network_requests=0,
        verification_limits=["These are repeated retrospective development splits, not independent external tests",
            "Thirty random test sets overlap; their counts are not new independent matches",
            "Numerical correctness and training-only preprocessing do not establish football predictive value"])
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,default=OUTPUT);parser.add_argument("--allow-partial",action="store_true")
    args=parser.parse_args()
    with threadpool_limits(limits=2):report=verify(args.output,args.allow_partial)
    dest=args.output/"verification-independent";dest.mkdir(exist_ok=True)
    name="partial-verification.json" if args.allow_partial else "verification.json"
    (dest/name).write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(dict(status=report["status"],folds=report["completed_folds"],metric_groups=report["independently_recomputed_metric_groups"],pending=report["pending_folds"],output=str(dest/name)),ensure_ascii=False))


if __name__=="__main__":main()
