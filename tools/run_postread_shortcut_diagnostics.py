"""Controlled title ablation and split sensitivity on the existing frozen-f panel.

Creates a new offline experiment. Does not change datasets, previous experiments,
f, production g, runtime or the frontend. Alternative splits refit all learned
preprocessing and the same Ridge / 128x64 MLP family; no test-based selection.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import socket
import sys
import time

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, normalize
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from football_research.acquisition.postread_selector import FEATURE_FIELDS

EXP = Path("E:/FootballMind/experiments")
DATA = EXP/"postread-clean-expansion-20261025-v1"
EXPANSION_PLAN = EXP/"unified-f-postread-g-expansion-20261025-v1/expansion-plan.json"
AUDIT = EXP/"postread-validation-test-reversal-audit-20261025-v1"
SEEDS = (20261025, 20261026, 20261027)
SPLITS = ("train", "validation", "test")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def save_rows(path, values):
    with Path(path).open("x", encoding="utf-8") as f:
        for value in values:
            f.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+"\n")


def save_csv(path, values):
    with Path(path).open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)


def forbid_network(*args, **kwargs):
    raise RuntimeError("This diagnostic experiment is offline")


def macro(values, mids):
    grouped = defaultdict(list)
    for value, mid in zip(values, mids):
        grouped[mid].append(float(value))
    return float(np.mean([np.mean(v) for v in grouped.values()]))


def match_weights(mids):
    count = Counter(mids)
    return np.array([len(mids)/(len(count)*count[m]) for m in mids])


def metrics(y, p, mids):
    rho = float(spearmanr(y, p).statistic) if np.ptp(p)>0 and np.ptp(y)>0 else None
    return dict(matches=len(set(mids)), rows=len(y), mse=macro((p-y)**2, mids),
                mae=macro(abs(p-y), mids), row_spearman=rho if rho is None or math.isfinite(rho) else None,
                prediction_squared=macro(p*p, mids), prediction_target_product=macro(p*y, mids),
                excess_mse_over_zero=macro((p-y)**2-y*y, mids))


def source_components(documents):
    """Group whole matches by shared domain, exact body or URL, transitively.

    Conservative source grouping is independent of outcomes and embedding fits.
    It guarantees source separation, not unseen football teams or causal effects.
    """
    mids = sorted({d["physical_match_id"] for d in documents})
    parent = {m:m for m in mids}
    def find(m):
        while parent[m] != m:
            parent[m] = parent[parent[m]]; m = parent[m]
        return m
    def union(a, b):
        aa, bb = find(a), find(b)
        if aa != bb:
            parent[max(aa, bb)] = min(aa, bb)
    indexes = {}
    for d in documents:
        mid = d["physical_match_id"]
        for field in ("source_domain", "text_sha256", "source_url"):
            if not d.get(field):
                raise ValueError("Source grouping needs explicit "+field)
            key = field, d[field]
            if key in indexes:
                union(mid, indexes[key])
            else:
                indexes[key] = mid
    grouped = defaultdict(list)
    for mid in mids:
        grouped[find(mid)].append(mid)
    return dict(grouped)


def match_dates(samples):
    dates = {}
    for s in samples:
        mid = s["physical_match_id"]
        if mid.startswith("fd:"):
            day = mid.split(":")[3]
        else:
            clock = s.get("kickoff") or s.get("clocks", {}).get("kickoff")
            if not clock:
                raise ValueError("No actual match date: "+mid)
            day = clock[:10]
        datetime.strptime(day, "%Y-%m-%d")
        if mid in dates and dates[mid] != day:
            raise ValueError("Conflicting dates")
        dates[mid] = day
    return dates


def plan_splits(features, samples, documents, expansion):
    original = {f["physical_match_id"]: f["split"] for f in features}
    if any(original[f["physical_match_id"]] != f["split"] for f in features):
        raise ValueError("Original match crosses partitions")
    dates = match_dates(samples)
    plans = {"original_51": dict(assignments=original, purpose="reproduce current cohort; previously viewed"),
             "original_18": dict(assignments={r["physical_match_id"]:r["split"] for r in expansion["groups"]},
                                 purpose="same 18-match cohort reference without 33 background matches")}
    by_club = defaultdict(list)
    for r in expansion["groups"]:
        by_club[r["club_source"]].append(r["physical_match_id"])
    chronological = {}
    for club, mids in sorted(by_club.items()):
        if len(mids) != 6:
            raise ValueError("Expected six source-club cohort matches")
        for i, mid in enumerate(sorted(mids, key=lambda m:(dates[m],m))):
            chronological[mid] = SPLITS[i//2]
    plans["within_source_time_18"] = dict(assignments=chronological,
        purpose="each source club first 2 train, middle 2 validation, last 2 test; not global time separation")
    ordered = sorted(original, key=lambda m:(dates[m], m))
    day_order = sorted({dates[m] for m in ordered})
    # Cut at day boundaries, never split simultaneous match dates by row index.
    cut1 = dates[ordered[int(.6*len(ordered))-1]]
    cut2 = dates[ordered[int(.8*len(ordered))-1]]
    global_time = {m:("train" if dates[m]<=cut1 else "validation" if dates[m]<=cut2 else "test") for m in ordered}
    plans["global_time_51"] = dict(assignments=global_time, date_boundaries=[cut1,cut2],
        purpose="global date-forward g split; fixed f and historical snapshots remain retrospective")
    components = source_components(documents)
    membership = {mid:root for root,mids in components.items() for mid in mids}
    domains = {root:sorted({d["source_domain"] for d in documents if d["physical_match_id"] in mids})
               for root,mids in components.items()}
    clubs = ("arsenal", "chelsea", "liverpool")
    club_roots = {club:next(root for root,dd in domains.items() if f"www.{club}fc.com" in dd or f"www.{club}.com" in dd)
                  for club in clubs}
    if len(set(club_roots.values())) != 3:
        raise ValueError("Source components connect the three intended held-out sources")
    for i, club in enumerate(clubs):
        test_root, val_root = club_roots[club], club_roots[clubs[(i+1)%3]]
        assignment = {m:("test" if membership[m]==test_root else "validation" if membership[m]==val_root else "train") for m in original}
        plans["source_holdout_"+club] = dict(assignments=assignment,
            purpose="leave connected source-domain group out; not leave all participating teams out",
            held_out_domains=domains[test_root], validation_domains=domains[val_root])
    for name, plan in plans.items():
        group = plan["assignments"]
        if not all(any(v==s for v in group.values()) for s in SPLITS):
            raise ValueError("Empty split: "+name)
        plan["counts"] = {s:dict(matches=sum(v==s for v in group.values()),
            rows=sum(group.get(f["physical_match_id"])==s for f in features)) for s in SPLITS}
        for field in ("text_sha256", "source_url"):
            split_by = defaultdict(set)
            for d in documents:
                if d["physical_match_id"] in group:
                    split_by[d[field]].add(group[d["physical_match_id"]])
            if any(len(v)>1 for v in split_by.values()):
                raise ValueError(name+": cross-split duplicate "+field)
        if name.startswith("source_holdout"):
            split_by = defaultdict(set)
            for d in documents:
                split_by[d["source_domain"]].add(group[d["physical_match_id"]])
            if any(len(v)>1 for v in split_by.values()):
                raise ValueError("Source domain crosses split")
    return plans, dict(components=components, domains=domains, match_dates=dates)


def fit_encoder(texts):
    unique = sorted({t for t in texts if t.strip()})
    tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=4000, sublinear_tf=True, norm="l2", lowercase=True)
    x = tfidf.fit_transform(unique)
    dim = min(32, x.shape[0]-1, x.shape[1]-1)
    if dim < 1:
        raise ValueError("Not enough training texts")
    svd = TruncatedSVD(n_components=dim, random_state=SEEDS[0]).fit(x)
    return dict(vectorizer=tfidf, svd=svd, dimension=dim, train_text_sha256=[text_sha(t) for t in unique])


def encode(enc, texts):
    return normalize(enc["svd"].transform(enc["vectorizer"].transform(texts)), norm="l2")


def designs(features, body_enc, title_enc, sample_by, domains):
    if any(set(f) != FEATURE_FIELDS for f in features):
        raise ValueError("Feature allowlist differs; labels are not inputs")
    body = encode(body_enc, [r["candidate_text"] for r in features])
    adopted = encode(body_enc, [r["adopted_text"] for r in features])
    title = encode(title_enc, [r["candidate_title"] for r in features])
    state, sem = np.array([r["state_numeric"] for r in features]), np.array([r["semantic_numeric"] for r in features])
    src = np.array([[float(sample_by[r["sample_id"]]["source_domain"]==d) for d in domains] for r in features])
    return dict(state=state, body=body, title_local=np.c_[state,title], semantic=np.c_[state,sem],
        fulltext_state=np.c_[state,sem,body,adopted,abs(body-adopted),body*adopted], source_state=np.c_[state,src])


def predict(spec, design, mean, scale):
    if spec["kind"] == "zero":
        return np.zeros(len(design["state"]))
    if spec["kind"] == "constant":
        return np.full(len(design["state"]), spec["value"])
    return spec["estimator"].predict(spec["scaler"].transform(design[spec["input_key"]]))*scale+mean


def policy_evaluation(features, samples, states, scores, assignments):
    """Follow counterfactual states verified with paired f; no sum of standalone gains."""
    lookup = {(s["physical_match_id"],s["discovered_mask"],s["acquired_mask"],s["candidate_index"]):s["sample_id"] for s in samples}
    state_by = {(s["physical_match_id"],s["discovered_mask"],s["acquired_mask"]):s for s in states}
    ss = {s["physical_match_id"]:s for s in samples}
    detail = []
    for mid, split in assignments.items():
        if split != "test":
            continue
        n = ss[mid]["candidate_count"]; dm=(1<<n)-1
        if not all((mid,dm,mask) in state_by for mask in range(1<<n)):
            raise ValueError("Full state grid missing: "+mid)
        initial = state_by[mid,dm,0]
        for budget in (1,2,3):
            def result(name, selected, reason, score_trace=None):
                mask = sum(1<<i for i in selected)
                last = state_by[mid,dm,mask]
                return dict(physical_match_id=mid, budget=budget, strategy=name, selected_indices=selected,
                    unselected_indices=[i for i in range(n) if i not in selected], f_updates=len(selected),
                    p_before=initial["p"], p_after=last["p"], initial_loss=initial["loss"], final_loss=last["loss"],
                    loss_change=last["loss"]-initial["loss"], stop_reason=reason, score_trace=score_trace or [],
                    historical_total_cost=None, incremental_adoption_cost_assumption=0., g_read_pool_size=n,
                    budget_compliant=len(selected)<=budget)
            detail.extend([result("no_information",[],"baseline"),result("all_candidates",list(range(n)),"all_used"),
                           result("fixed_order",list(range(min(n,budget))),"budget_or_pool_exhausted")])
            losses=[state_by[mid,dm,sum(1<<i for i in subset)]["loss"] for subset in itertools.combinations(range(n), min(n,budget))]
            detail.append(dict(physical_match_id=mid,budget=budget,strategy="random_exact",selected_indices=None,
                unselected_indices=None, f_updates=min(n,budget),p_before=initial["p"],p_after=None,
                initial_loss=initial["loss"],final_loss=float(np.mean(losses)),loss_change=float(np.mean(losses))-initial["loss"],
                stop_reason="budget_or_pool_exhausted", score_trace=[],historical_total_cost=None,
                incremental_adoption_cost_assumption=0.,g_read_pool_size=n,budget_compliant=True))
            for name, pred in scores.items():
                mask=0;selected=[];trace=[];reason="budget_exhausted"
                for _ in range(min(n,budget)):
                    feasible = [i for i in range(n) if not mask&(1<<i)]
                    values = {i:pred[lookup[mid,dm,mask,i]] for i in feasible}
                    i = max(feasible,key=lambda k:(values[k],-k))
                    trace.append(dict(acquired_mask=mask,scores=values))
                    if values[i] <= 0:
                        reason="predicted_nonpositive";break
                    selected.append(i);mask|=1<<i
                detail.append(result(name,selected,reason,trace))
    aggregate=[]
    for budget in (1,2,3):
        for name in sorted({r["strategy"] for r in detail}):
            rr=[r for r in detail if r["budget"]==budget and r["strategy"]==name]
            aggregate.append(dict(budget=budget,strategy=name,matches=len(rr),mean_final_loss=float(np.mean([r["final_loss"] for r in rr])),
                mean_loss_change=float(np.mean([r["loss_change"] for r in rr])),mean_f_updates=float(np.mean([r["f_updates"] for r in rr]))))
    return detail, aggregate


def fit_fold(name, plan, all_features, label_by, sample_by, samples, states, destination, f_sha):
    destination.mkdir()
    assignments=plan["assignments"]
    features=[{**r,"split":assignments[r["physical_match_id"]]} for r in all_features if r["physical_match_id"] in assignments]
    idx={s:np.array([i for i,r in enumerate(features) if r["split"]==s]) for s in SPLITS}
    ti,vi=idx["train"],idx["validation"]
    train=[features[i] for i in ti]
    body=fit_encoder([r["candidate_text"] for r in train]+[r["adopted_text"] for r in train])
    title=fit_encoder([r["candidate_title"] for r in train])
    domains=sorted({sample_by[r["sample_id"]]["source_domain"] for r in train})
    design=designs(features,body,title,sample_by,domains)
    mids=np.array([r["physical_match_id"] for r in features]);y=np.array([label_by[r["sample_id"]]["realized_gain"] for r in features])
    weights=match_weights(mids[ti]);mean=float(np.average(y[ti],weights=weights))
    scale=max(float(np.sqrt(np.average((y[ti]-mean)**2,weights=weights))),1e-8);yt=(y[ti]-mean)/scale
    fitted={"zero":{"kind":"zero"},"train_mean":{"kind":"constant","value":mean}}
    for key in ("state","body","title_local","semantic","fulltext_state","source_state"):
        scaler=StandardScaler().fit(design[key][ti],sample_weight=weights)
        estimator=Ridge(alpha=10.).fit(scaler.transform(design[key][ti]),yt,sample_weight=weights)
        fitted[key+"_ridge"]=dict(kind="ridge",input_key=key,scaler=scaler,estimator=estimator)
    scaler=StandardScaler().fit(design["fulltext_state"][ti],sample_weight=weights)
    xt,xv=scaler.transform(design["fulltext_state"][ti]),scaler.transform(design["fulltext_state"][vi])
    curves=[];checkpoints={}
    for seed in SEEDS:
        model_name=f"mlp_128_64_seed{seed}"
        est=MLPRegressor(hidden_layer_sizes=(128,64),activation="relu",solver="adam",alpha=.1,
            learning_rate_init=.001,batch_size=min(64,len(ti)),max_iter=1,early_stopping=False,shuffle=True,random_state=seed)
        best_loss=math.inf;best_epoch=0;best=None
        for epoch in range(1,601):
            est.partial_fit(xt,yt,sample_weight=weights)
            p_train,p_val=est.predict(xt)*scale+mean,est.predict(xv)*scale+mean
            train_mse=macro((y[ti]-p_train)**2,mids[ti]);val_mse=macro((y[vi]-p_val)**2,mids[vi])
            curves.append(dict(model=model_name,epoch=epoch,train_mse=train_mse,validation_mse=val_mse))
            if val_mse<best_loss-1e-12:
                best_loss=val_mse;best_epoch=epoch;best=deepcopy(est)
            if epoch-best_epoch>=60:break
        fitted[model_name]=dict(kind="mlp",input_key="fulltext_state",scaler=scaler,estimator=best)
        checkpoints[model_name]=dict(selected_epoch=best_epoch,epochs_run=epoch,best_validation_mse=best_loss,
            parameters=sum(v.size for v in best.coefs_+best.intercepts_))
    val={k:macro((y[vi]-predict(v,design,mean,scale)[vi])**2,mids[vi]) for k,v in fitted.items()}
    # A source dummy is a shortcut probe, not a contender for the deployed model.
    contenders={k:v for k,v in val.items() if k!="source_state_ridge"}
    selected=min(contenders,key=contenders.get)
    best_full=min((k for k in contenders if k.startswith("mlp") or k=="fulltext_state_ridge"),key=contenders.get)
    selection=dict(selected_model=selected,best_fulltext=best_full,validation_mse=val,test_used=False,
                   source_dummy_excluded_from_selection=True)
    save(destination/"selection-frozen.json",selection)
    bundle=dict(schema="postread-shortcut-diagnostic-models-v1",f_model_sha256=f_sha,fitted=fitted,
        body_encoder=body,title_encoder=title,source_vocabulary=domains,target_mean=mean,target_scale=scale,
        selected_model=selected,split=plan,production_ready=False)
    joblib.dump(bundle,destination/"models.joblib",compress=3)
    reloaded=joblib.load(destination/"models.joblib")
    design2=designs(features,reloaded["body_encoder"],reloaded["title_encoder"],sample_by,reloaded["source_vocabulary"])
    metrics_rows=[];predrows=[];scores={};reload_error=0.
    for model_name,spec in fitted.items():
        p=predict(spec,design,mean,scale)
        reload_error=max(reload_error,float(np.max(abs(p-predict(reloaded["fitted"][model_name],design2,mean,scale)))))
        scores[model_name]={r["sample_id"]:float(v) for r,v in zip(features,p)}
        for split,ii in idx.items():
            metrics_rows.append(dict(fold=name,model=model_name,split=split,**metrics(y[ii],p[ii],mids[ii])))
        predrows.extend(dict(fold=name,model=model_name,sample_id=r["sample_id"],physical_match_id=r["physical_match_id"],
            split=r["split"],predicted_gain=float(v),realized_gain=float(target)) for r,v,target in zip(features,p,y))
    assert reload_error <= 1e-12
    scores["validation_selected"] = scores[selected]
    scores["validation_selected_fulltext"] = scores[best_full]
    detail,policy=policy_evaluation(features,samples,states,scores,assignments)
    save_rows(destination/"predictions.jsonl",predrows);save_rows(destination/"training-curves.jsonl",curves)
    save(destination/"metrics.json",metrics_rows);save_rows(destination/"policy-paths.jsonl",detail)
    save(destination/"policy-comparison.json",policy)
    # Independent train-only provenance checks for every learned transform.
    train_body_hashes={text_sha(t) for r in train for t in (r["candidate_text"],r["adopted_text"]) if t.strip()}
    train_title_hashes={text_sha(r["candidate_title"]) for r in train if r["candidate_title"].strip()}
    assert set(body["train_text_sha256"])==train_body_hashes and set(title["train_text_sha256"])==train_title_hashes
    scaler_errors={k:float(np.max(abs(spec["scaler"].mean_-np.average(design[spec["input_key"]][ti],axis=0,weights=weights))))
                   for k,spec in fitted.items() if spec["kind"] not in ("zero","constant")}
    assert max(scaler_errors.values())<1e-12
    save(destination/"verification.json",dict(train_only_encoder_hashes=True,train_only_scaler_max_errors=scaler_errors,
        serialized_reload_max_error=reload_error,model_sha256=sha(destination/"models.joblib"),f_model_sha256=f_sha,
        checkpoints=checkpoints,partition_counts=plan["counts"],selection_frozen_before_test=True,
        test_status="retrospective_resplit_of_previously_seen_data_not_new_blind_evaluation"))
    return metrics_rows,[dict(fold=name,**r) for r in policy],selection


def fixed_title_ablation(features, labels, sample_by, original_bundle, output):
    bundle=original_bundle
    design=designs(features,bundle["body_encoder"],bundle["title_encoder"],sample_by,[])
    spec=bundle["fitted"]["title_local_ridge"]
    base=design["title_local"].copy();boundary=len(features[0]["state_numeric"])
    mean,scale=bundle["target_mean"],bundle["target_scale"]
    variants={"original_title_ridge":base,
        "title_replaced_by_training_mean":np.c_[base[:,:boundary],np.tile(spec["scaler"].mean_[boundary:],(len(base),1))],
        "title_raw_zero":np.c_[base[:,:boundary],np.zeros_like(base[:,boundary:])],
        "state_replaced_by_training_mean":np.c_[np.tile(spec["scaler"].mean_[:boundary],(len(base),1)),base[:,boundary:]]}
    result=[];preds=[]
    mids=np.array([r["physical_match_id"] for r in features]);y=np.array([labels[r["sample_id"]]["realized_gain"] for r in features])
    for name,xx in variants.items():
        p=spec["estimator"].predict(spec["scaler"].transform(xx))*scale+mean
        for split in SPLITS:
            ii=np.array([i for i,r in enumerate(features) if r["split"]==split])
            result.append(dict(variant=name,split=split,**metrics(y[ii],p[ii],mids[ii])))
        preds.extend(dict(variant=name,sample_id=r["sample_id"],physical_match_id=r["physical_match_id"],split=r["split"],
                          predicted_gain=float(v)) for r,v in zip(features,p))
    # Permute whole article title vectors, consistently over its multiple states.
    # Within-domain permutation retains source cues; global permutation breaks them.
    keys=[(r["physical_match_id"],sample_by[r["sample_id"]]["candidate_index"]) for r in features]
    first={key:keys.index(key) for key in set(keys)}
    for scope in ("within_source", "across_sources"):
        for seed in range(20261025,20261045):
            rng=np.random.default_rng(seed);xx=base.copy();groups=defaultdict(list)
            for key,i in sorted(first.items()):
                r=features[i];domain=sample_by[r["sample_id"]]["source_domain"] if scope=="within_source" else "all"
                groups[r["split"],domain].append(key)
            mapping={}
            for group in groups.values():
                shuffled=list(group);rng.shuffle(shuffled);mapping.update(zip(group,shuffled))
            for i,key in enumerate(keys):
                xx[i,boundary:]=base[first[mapping[key]],boundary:]
            p=spec["estimator"].predict(spec["scaler"].transform(xx))*scale+mean
            for split in SPLITS:
                ii=np.array([i for i,r in enumerate(features) if r["split"]==split])
                result.append(dict(variant="permute_"+scope,seed=seed,split=split,**metrics(y[ii],p[ii],mids[ii])))
    save(output/"fixed-title-ablation.json",dict(results=result,
        interpretation="Frozen-model intervention, not retrained feature removal. Mean replacement zeros standardized block; raw zero is different and may be off-manifold.",
        permutation="Whole article title vector shuffled within each split; each article remains identical over states; not used for selecting models"))
    save_rows(output/"fixed-title-ablation-predictions.jsonl",preds)
    return result


def figures(output, metrics_rows, policy_rows, ablations):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    variants=["original_title_ridge","title_replaced_by_training_mean","title_raw_zero","state_replaced_by_training_mean"]
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,split in zip(axes,("validation","test")):
        values=[next(r["mse"] for r in ablations if r["variant"]==v and r["split"]==split) for v in variants]
        ax.bar(range(4),values,color=["#197b62","#89b9a4","#aab7c3","#d0aa79"])
        ax.set(xticks=range(4),xticklabels=["Original","Title -> train mean","Title -> raw zero","State -> train mean"],
               ylabel="Match-weighted gain MSE (lower better)",title=split+": same 6 matches")
        ax.tick_params(axis="x",rotation=20);ax.grid(axis="y",alpha=.2)
    fig.suptitle("Fixed title model ablation: mean replacement removes standardized block\nA sensitivity test, not proof of a causal source shortcut")
    fig.tight_layout();fig.savefig(output/"title-ablation.png",dpi=170);plt.close(fig)
    folds=list(dict.fromkeys(r["fold"] for r in metrics_rows))
    models=["state_ridge","title_local_ridge","semantic_ridge","fulltext_state_ridge","source_state_ridge"]
    fig,ax=plt.subplots(figsize=(12,5))
    grid=[]
    for name in models:
        grid.append([next(r["mse"] for r in metrics_rows if (r["fold"],r["model"],r["split"])==(fold,name,"test"))/
                     next(r["mse"] for r in metrics_rows if (r["fold"],r["model"],r["split"])==(fold,"zero","test")) for fold in folds])
    im=ax.imshow(grid,cmap="RdYlGn_r",vmin=0,vmax=2)
    for i,row in enumerate(grid):
        for j,value in enumerate(row):ax.text(j,i,f"{value:.2f}",ha="center",va="center",fontsize=9)
    ax.set(xticks=range(len(folds)),xticklabels=folds,yticks=range(len(models)),yticklabels=models,
           title="Test gain MSE / zero baseline; <1 is better\nRefit within each split; different columns are different cohorts")
    ax.tick_params(axis="x",rotation=25);fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(output/"split-sensitivity.png",dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,5))
    for name in ("title_local_ridge","fulltext_state_ridge","validation_selected_fulltext","random_exact","all_candidates"):
        values=[next(r["mean_loss_change"] for r in policy_rows if (r["fold"],r["strategy"],r["budget"])==(fold,name,2)) for fold in folds]
        ax.plot(range(len(folds)),values,marker="o",label=name)
    ax.axhline(0,color="black",lw=.8);ax.set(xticks=range(len(folds)),xticklabels=folds,
        ylabel="Final log loss minus no-information loss (negative is better)",
        title="Policy diagnostic, maximum 2 f updates; all-candidate reference may exceed budget")
    ax.tick_params(axis="x",rotation=25);ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/"policy-sensitivity.png",dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(17,8));axes=axes.ravel()
    for ax,fold in zip(axes,folds):
        cc=rows(output/fold/"training-curves.jsonl")
        for seed in SEEDS:
            r=[v for v in cc if v["model"]==f"mlp_128_64_seed{seed}"]
            ax.plot([v["epoch"] for v in r],[max(v["train_mse"],1e-12) for v in r],linestyle="--",alpha=.5)
            ax.plot([v["epoch"] for v in r],[max(v["validation_mse"],1e-12) for v in r],label=str(seed))
        ax.set_yscale("log");ax.set(title=fold,xlabel="Epoch",ylabel="MSE");ax.legend(fontsize=6)
    for ax in axes[len(folds):]:ax.axis("off")
    fig.suptitle("Same 128x64 MLP, 3 fixed seeds per split; dashed=train, solid=validation\nEarly stopping uses validation only; no architecture expansion")
    fig.tight_layout();fig.savefig(output/"training-curves.png",dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,default=DATA)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--finalize-existing", action="store_true", help="Finish reporting of completed fits without refitting")
    args=parser.parse_args();output=args.output
    socket.socket.connect=socket.socket.connect_ex=socket.getaddrinfo=socket.create_connection=forbid_network
    start=time.perf_counter();dataset=args.dataset
    if args.finalize_existing:
        if (output/"verification.json").exists():
            raise FileExistsError("Completed experiments must not be rewritten")
        frozen=read(output/"experiment-plan.json")
        if not all(sha(p)==h for p,h in frozen["source_hashes"].items()):
            raise ValueError("Source changed since the experiment plan was frozen")
        finalize(output,dataset,frozen["plans"],read(output/"selections.json"),frozen["source_hashes"],
                 read(output/"metrics.json"),read(output/"policy-comparison.json"),
                 read(output/"fixed-title-ablation.json")["results"],start)
        return
    output.mkdir(parents=True,exist_ok=False)
    features=rows(dataset/"features.jsonl");samples=rows(dataset/"samples.jsonl");docs=rows(dataset/"documents.jsonl")
    if any(set(r)!=FEATURE_FIELDS for r in features):raise ValueError("Unexpected feature fields")
    expansion=read(EXPANSION_PLAN)
    plans,group_audit=plan_splits(features,samples,docs,expansion)
    source_paths=[dataset/n for n in ("features.jsonl","labels.jsonl","samples.jsonl","states.jsonl","documents.jsonl","split.json")]
    source_paths += list((dataset/"training").glob("*.*"))+[EXPANSION_PLAN,AUDIT/"manifest.json",AUDIT/"readable-diagnosis.txt"]
    source_hashes={str(p):sha(p) for p in source_paths if p.is_file()}
    save(output/"experiment-plan.json",dict(schema="postread-shortcut-ablation-plan-v1",created_at=datetime.now(timezone.utc).isoformat(),
        frozen_before_training=True,plans=plans,seeds=list(SEEDS),ridge_alpha=10.,mlp_hidden=[128,64],mlp_alpha=.1,
        epochs=600,patience=60,source_onehot="diagnostic Ridge only; train-only categories; unseen source all zeros; not a promotion candidate",
        text_encoder="same TFIDF1-2 max4000/SVD<=32; refit train only in every split",test_selection_allowed=False,
        source_hashes=source_hashes,test_status="all historical data previously viewed; split sensitivity, not new blind test"))
    save(output/"grouping-audit.json",group_audit)
    labels={r["sample_id"]:r for r in rows(dataset/"labels.jsonl")}
    sample_by={r["sample_id"]:r for r in samples};states=rows(dataset/"states.jsonl")
    if set(labels)!={r["sample_id"] for r in features} or len(labels)!=len(features):raise ValueError("ID alignment differs")
    f_shas={r["f_model_sha256"] for r in labels.values()}
    if len(f_shas)!=1:raise ValueError("Labels mix f models")
    f_sha=next(iter(f_shas));bundle=joblib.load(dataset/"training/models.joblib")
    if f_sha!=read(dataset/"training/model.json")["f_model_sha256"]:raise ValueError("Frozen f binding differs")
    all_metrics=[];all_policy=[];selections={}
    with threadpool_limits(limits=1):
        ablations=fixed_title_ablation(features,labels,sample_by,bundle,output)
        for name,plan in plans.items():
            print(json.dumps(dict(starting_fold=name,counts=plan["counts"])),flush=True)
            mm,pp,selection=fit_fold(name,plan,features,labels,sample_by,samples,states,output/name,f_sha)
            all_metrics.extend(mm);all_policy.extend(pp);selections[name]=selection
            print(json.dumps(dict(completed_fold=name,selected=selection["selected_model"],best_fulltext=selection["best_fulltext"])),flush=True)
    save(output/"metrics.json",all_metrics);save_csv(output/"metrics.csv",all_metrics)
    save(output/"policy-comparison.json",all_policy);save_csv(output/"policy-comparison.csv",all_policy)
    save(output/"selections.json",selections)
    finalize(output,dataset,plans,selections,source_hashes,all_metrics,all_policy,ablations,start)


def finalize(output,dataset,plans,selections,source_hashes,all_metrics,all_policy,ablations,start):
    # Original-cohort Ridge reproduction is a control for this new experiment code.
    old={(r["model"],r["split"]):r["match_macro_mse"] for r in read(dataset/"training/metrics.json")}
    parity={f"{r['model']}/{r['split']}":abs(r["mse"]-old[r["model"],r["split"]]) for r in all_metrics
            if r["fold"]=="original_51" and (r["model"],r["split"]) in old}
    linear_parity={k:v for k,v in parity.items() if not k.startswith("mlp_")}
    if max(linear_parity.values())>1e-10:raise ValueError("Original-cohort linear fit reproduction differs")
    # Frozen-estimator inference parity and nonlinear retraining parity are
    # different checks. Preserve every refit discrepancy rather than claiming
    # bitwise neural training reproduction across numerical execution settings.
    warnings=[]
    for fold in plans:
        bundle=joblib.load(output/fold/"models.joblib")
        for name,spec in bundle["fitted"].items():
            if "scaler" not in spec:continue
            var,scale=spec["scaler"].var_,spec["scaler"].scale_
            if not np.isfinite(scale).all() or np.any(scale<=0):
                raise ValueError("Nonfinite or invalid learned scale")
            if np.any(var<0):
                if var.min() < -1e-12:raise ValueError("Material negative learned variance")
                warnings.append(dict(fold=fold,model=name,minimum_stored_variance=float(var.min()),
                    count=int((var<0).sum()),effective_scale=scale[var<0].tolist(),
                    reason="floating-point variance below zero on constant feature; sklearn already sets effective scale to 1; saved estimator unchanged"))
    save(output/"numerical-notes.json",dict(original_neural_refit_metric_differences={k:v for k,v in parity.items() if k.startswith("mlp_")},
        exact_neural_retraining_claimed=False,roundoff_warnings=warnings))
    figures(output,all_metrics,all_policy,ablations)
    assert all(sha(p)==h for p,h in source_hashes.items())
    verification=dict(status="passed_with_reported_neural_refit_drift",original_fit_max_metric_error=max(parity.values()),
        original_linear_fit_max_metric_error=max(linear_parity.values()),original_fit_metric_errors=parity,
        old_sources_unchanged=True,network_requests=0,external_api_calls=0,f_retrained=False,
        research_g_refitted=True,model_families_expanded=False,formal_runtime_modified=False,
        folds=len(plans),new_independent_matches=0,elapsed_seconds=time.perf_counter()-start)
    save(output/"verification.json",verification)
    report=["标题／来源消融与替代分组诊断", "", "固定同一个 f、289 条既有转移；全部为历史诊断，不是新独立盲测。",
        "每个替代分组均只用其训练集拟合文本编码器、特征标准化和标签尺度，再拟合同结构 Ridge/128×64 MLP。",
        "验证集选择 checkpoint；测试指标在选择冻结后计算。三种 MLP 随机种子全部保留。",
        "所有来源共有的域名、正文摘要或 URL 形成连通分组；来源分组不是陌生球队验证。",
        "标题替换为训练均值与替换为原始零向量分别报告，不能把两者当作同一种消融。", "",
        "原模型标题消融："]
    for r in ablations:
        if r["split"] in ("validation","test") and "seed" not in r:
            report.append(f"{r['split']} | {r['variant']} | MSE={r['mse']:.10f} | E[p*y]={r['prediction_target_product']:.10f}")
    report += ["", "替代分组逐项结果（不同分组的比赛不同，不直接比较绝对loss大小）："]
    for fold,selection in selections.items():
        report.append(f"{fold}：{plans[fold]['counts']}，验证选择={selection['selected_model']}，最佳全文={selection['best_fulltext']}")
        for model in ("zero","state_ridge","title_local_ridge","source_state_ridge","fulltext_state_ridge",selection["best_fulltext"]):
            r=next(x for x in all_metrics if (x["fold"],x["model"],x["split"])==(fold,model,"test"))
            report.append(f"  {model}: test MSE={r['mse']:.10f}, excess_over_zero={r['excess_mse_over_zero']:.10f}")
    report += ["", "推断限制：",
        "改变切分也会改变训练量、目标分布和球队覆盖。反转减弱不能单独证明来源捷径，反转保留也不能证明任何正确模型都会失败。",
        "目标方差更低只说明恒零基线更容易；不能据此排除条件预测更难。训练误差极低不是可泛化信号的证明。",
        "不把测试结果用于改选冠军或修改模型；成本金额仍未知，只比较相同f更新预算。",
        "f 固定且部分数据在 f 拟合日期之前；历史正文版本也未完全证明，仍不能称真实前瞻回测。",
        "没有覆盖或修改旧数据、旧模型、正式runtime和前端；新模型仅在本独立目录。"]
    (output/"readable-report.txt").write_text("\n".join(report)+"\n",encoding="utf-8")
    save(output/"manifest.json",{str(p.relative_to(output)):sha(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name!="manifest.json"})
    print(json.dumps(verification),flush=True)


if __name__=="__main__":main()
