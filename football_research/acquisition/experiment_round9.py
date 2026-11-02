"""Frozen joint prediction plus forward-OOF empirical update-gain calibration."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import timedelta
import itertools
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np

from football_research.baselines.api import canonical_json
from football_research.evaluation.splits import rolling_origin_splits
from .contextual_partial import ContextualPartialPredictor
from .data import ControlledEpisode, load_episodes
from .experiment import losses, outcome, state_after
from .experiment_round4 import write_json, _sem_time
from .experiment_round6 import _hash, _jsonl
from .experiment_round8 import run_episode as raw_run_episode, select_lambda as raw_select_lambda, STRATEGIES as RAW_STRATEGIES
from .joint_returns import JointReturnPredictor, extract_oof_contexts
from .round9_residual import ResidualEVSICalibrator, BIN_EDGES, evsi_bin

VERSION = "residual-evsi-calibration-round9-v1"
STRATEGIES = (*RAW_STRATEGIES, "calibrated-mean", "calibrated-lcb")
NEW_STRATEGIES = ("calibrated-mean", "calibrated-lcb")


def joint_transitions(rows,model,actions,provenance):
    records=[]
    ids=[a["action_id"] for a in actions]
    for row in rows:
        for acquired in ((),(ids[0],),(ids[1],)):
            state=state_after(row,actions,acquired)
            before=model.predict_partial(state)["probabilities"]
            for action in actions:
                if action["action_id"] in acquired:
                    continue
                # Enumeration precedes access to the actual unpurchased return.
                model_evsi=model.enumerate_action(state,action)["evsi"]
                after=model.predict_partial(state_after(row,actions,(*acquired,action["action_id"])))["probabilities"]
                gain=losses(before,outcome(row))["log_loss"]-losses(after,outcome(row))["log_loss"]
                records.append({"match_id":row["match_id"],"kickoff":row["kickoff"],"result_available_at":row["result_available_at"],
                    "state":state,"action":action,"acquired_action_ids":list(acquired),"prediction_before":before,"prediction_after":after,
                    "delta":gain,"model_evsi":max(0.,model_evsi),"residual":gain-max(0.,model_evsi),
                    "oof_predictor_version":model.model_version,**provenance})
    return records


def forward_joint_oof(rows,actions,contexts,source,output):
    manifest=json.loads((source/"oof-manifest.json").read_text(encoding="utf-8"))
    lookup={r["match_id"]:r for r in rows}
    samples,folds,skipped=[],[],[]
    for index,chunk in enumerate(manifest["folds"]):
        cutoff=_sem_time(chunk["cutoff"])
        train=[lookup[mid] for mid in chunk["training_match_ids"]]
        eligible=[r for r in train if r["match_id"] in contexts and _sem_time(r["result_available_at"]) <= cutoff]
        if not eligible:
            skipped.append({"source_chunk":index,"cutoff":chunk["cutoff"],"prediction_match_ids":chunk["prediction_match_ids"],
                "base_training_match_ids":chunk["training_match_ids"],"joint_training_match_ids":[],
                "reason":"no_prior_matches_with_forward_oof_p0_and_known_outcome"})
            continue
        source_base=source/chunk["artifact"]
        if _hash(source_base) != chunk["sha256"]:
            raise ValueError("Source OOF base artifact hash mismatch")
        base=ContextualPartialPredictor.load(source_base)
        if set(base.metadata["training_match_ids"]) != set(chunk["training_match_ids"]):
            raise ValueError("OOF base training membership mismatch")
        local_base=output/"oof-models"/f"chunk-{index}-base.json"
        local_base.parent.mkdir(exist_ok=True)
        shutil.copy2(source_base,local_base)
        model=JointReturnPredictor(base).fit(train,contexts,as_of=cutoff,dataset_version=rows[0]["dataset_version"],actions=actions)
        artifact=output/"oof-models"/f"chunk-{index}-joint.json"
        model.save(artifact)
        model=JointReturnPredictor.load(artifact)
        digest=_hash(artifact)
        targets=[lookup[mid] for mid in chunk["prediction_match_ids"]]
        if set(model.metadata["training_match_ids"]) & set(chunk["prediction_match_ids"]):
            raise ValueError("Joint OOF target leakage")
        record={"source_chunk":index,"artifact":str(artifact.relative_to(output)),"sha256":digest,"cutoff":chunk["cutoff"],
            "model_version":model.model_version,"training_match_ids":chunk["training_match_ids"],
            "base_training_match_ids":chunk["training_match_ids"],"joint_training_match_ids":model.metadata["training_match_ids"],
            "prediction_match_ids":chunk["prediction_match_ids"],"base_artifact":str(local_base.relative_to(output)),
            "base_sha256":_hash(local_base),"source_base_artifact":chunk["artifact"],"base_model_version":base.model_version}
        folds.append(record)
        samples.extend(joint_transitions(targets,model,actions,{"oof_train_cutoff":chunk["cutoff"],
            "oof_train_ids":chunk["training_match_ids"],"joint_training_match_ids":model.metadata["training_match_ids"],
            "oof_manifest_ref":"sha256:"+digest}))
        print(f"round9 OOF {index}: base={len(train)} joint={len(model.records)} targets={len(targets)}",flush=True)
    write_json(output/"oof-manifest.json",{"folds":folds,"skipped_chunks":skipped,"samples":len(samples),
        "independent_matches":len({r["match_id"] for r in samples}),"old_gain_targets_reused":False,
        "base_predictors":"frozen_round7_forward_oof_models","joint_models":"new_fit_at_each_past_cutoff",
        "training_target":"actual_fixed_joint_predictor_one_step_loss_gain_minus_model_evsi"})
    (output/"oof-supervision.jsonl").write_text("".join(canonical_json(r)+"\n" for r in samples),encoding="utf-8")
    return samples


def split_residual_training(samples,fraction=.2):
    matches={r["match_id"]:r for r in samples}
    ordered=sorted(matches.values(),key=lambda r:(_sem_time(r["kickoff"]),r["match_id"]))
    if len(ordered)<3:
        raise ValueError("Insufficient matches for chronological residual diagnostic split")
    boundary=_sem_time(ordered[max(1,int(len(ordered)*(1-fraction)))]["kickoff"])
    tail=[r for r in samples if _sem_time(r["kickoff"]) >= boundary]
    fit_cutoff=min(_sem_time(r["state"]["prediction_cutoff"]) for r in tail)-timedelta(microseconds=1)
    train=[r for r in samples if _sem_time(r["kickoff"]) < boundary and _sem_time(r["result_available_at"]) <= fit_cutoff]
    kept={r["match_id"] for r in train+tail}
    return train,tail,{"rule":"first 80 percent of chronological distinct matches; keep equal kickoff together, purge unavailable labels",
        "boundary":boundary.isoformat(),"fit_cutoff":fit_cutoff.isoformat(),"fraction":fraction,
        "training_match_ids":sorted({r["match_id"] for r in train}),"calibration_match_ids":sorted({r["match_id"] for r in tail}),
        "purged_match_ids":sorted(set(matches)-kept)}


def run_episode(row,model,actions,*,strategy,budget_money,lambda_money,seed,design,calibrator=None):
    if strategy not in NEW_STRATEGIES:
        return raw_run_episode(row,model,actions,strategy=strategy,budget_money=budget_money,lambda_money=lambda_money,seed=seed,design=design)
    if calibrator is None or calibrator.model.model_version != model.model_version:
        raise ValueError("Calibrated policy requires bound reference predictor")
    risk="mean" if strategy == "calibrated-mean" else "conservative"
    env=ControlledEpisode(row,actions)
    spent={unit:0. for unit in ("requests","money","latency_ms","tokens")}
    initial=model.predict_partial(env.visible_state())
    events=[]
    predictor_ms=estimator_ms=0.
    calls=0
    for step in range(len(actions)+1):
        state={**env.visible_state(),"action_history":[{"action_id":aid,"status":"acquired_or_source_missing"} for aid in env.performed_actions]}
        started=time.perf_counter()
        prediction=model.predict_partial(state)
        predictor_ms+=(time.perf_counter()-started)*1000
        candidates=env.legal_actions()
        started=time.perf_counter()
        decision=calibrator.plan(state,candidates,{"limits":{"money":budget_money,"requests":design["request_budget"]},"spent":spent},
            risk_level=risk,lambda_cost={"money":lambda_money})
        estimator_ms+=(time.perf_counter()-started)*1000
        calls+=1
        event={"step":step,"visible_state":deepcopy(state),"prediction":deepcopy(prediction),"decision":decision}
        events.append(event)
        if decision["decision"] == "stop":
            break
        selected=next(a for a in candidates if a["action_id"] == decision["action_id"])
        env.reveal(selected["action_id"])
        event["charged_cost"]=deepcopy(selected["cost"])
        for unit in spent:
            spent[unit]+=selected["cost"].get(unit,0.)
    return {"match_id":row["match_id"],"kickoff":row["kickoff"],"prediction_cutoff":row["prediction_cutoff"],"strategy":strategy,
        "seed":seed,"budget_money":budget_money,"lambda_money":lambda_money,"cost":spent,"final_probabilities":prediction["probabilities"],
        "initial_probabilities":initial["probabilities"],"final_mask":prediction["mask"],"actions":list(env.performed_actions),
        "stop_reason":decision["reason"],"model_version":model.model_version,"base_model_version":model.base_predictor.model_version,
        "gain_version":calibrator.version,"risk_level":risk,"predictor_latency_ms":predictor_ms,"estimator_latency_ms":estimator_ms,
        "estimator_calls":calls,"actual_external_requests":0,"actual_llm_tokens":0,"compute_money":None,"compute_money_status":"not_metered_local_cpu",
        **losses(prediction["probabilities"],outcome(row))},events


def select_lambda(rows,model,actions,strategy,budget,design,calibrator):
    if strategy not in NEW_STRATEGIES:
        return raw_select_lambda(rows,model,actions,strategy,budget,design)
    candidates=[]
    for lam in design["lambda_money_grid"]:
        results=[run_episode(row,model,actions,strategy=strategy,budget_money=budget,lambda_money=lam,seed=0,design=design,calibrator=calibrator)[0] for row in rows]
        utility=float(np.mean([r["log_loss"]+design["validation_utility_money_weight"]*r["cost"]["money"] for r in results]))
        candidates.append({"lambda_money":lam,"validation_utility":utility,"validation_matches":len(rows)})
    selected=min(candidates,key=lambda r:(r["validation_utility"],-r["lambda_money"]))["lambda_money"]
    return selected,{"selected":selected,"candidates":candidates,"validation_match_ids":[r["match_id"] for r in rows],
        "utility_money_weight":design["validation_utility_money_weight"]}


def summarize(results,design):
    groups=defaultdict(lambda:defaultdict(list))
    for row in results:
        groups[(row["strategy"],row["budget_money"])][row["match_id"]].append(row)
    means,metrics={},{}
    for (strategy,budget),matches in groups.items():
        records={}
        for mid,rows in matches.items():
            record={key:float(np.mean([r[key] for r in rows])) for key in ("log_loss","brier_score","accuracy")}
            record.update({u:float(np.mean([r["cost"][u] for r in rows])) for u in ("money","requests")})
            record.update(kickoff=rows[0]["kickoff"],utility_loss=record["log_loss"]+design["validation_utility_money_weight"]*record["money"])
            records[mid]=record
        means[(strategy,budget)]=records
        all_rows=[r for rows in matches.values() for r in rows]
        metrics[f"{strategy}/money={budget:g}"]={"independent_matches":len(records),"episodes":len(all_rows),
            **{key:float(np.mean([r[key] for r in records.values()])) for key in ("log_loss","brier_score","accuracy","money","requests","utility_loss")},
            "stop_reasons":dict(Counter(r["stop_reason"] for r in all_rows)),"action_counts":dict(Counter(a for r in all_rows for a in r["actions"])),
            "aggregation":"average seeds within match, then equal weight matches"}
    comparisons={}
    for budget,strategy in itertools.product(design["money_budgets"],STRATEGIES):
        references=tuple(s for s in STRATEGIES if s != strategy) if strategy in NEW_STRATEGIES else ("base-only",)
        for reference in references:
            if reference == strategy:
                continue
            a,b=means[(strategy,budget)],means[(reference,budget)]
            if a.keys() != b.keys():
                raise ValueError("Paired identities differ")
            blocks=defaultdict(list)
            for mid in sorted(a):
                blocks[_sem_time(a[mid]["kickoff"]).isocalendar()[:2]].append([a[mid][k]-b[mid][k] for k in ("log_loss","brier_score","money","utility_loss")])
            arrays=[np.asarray(v) for v in blocks.values()]
            sums,counts=np.asarray([v.sum(axis=0) for v in arrays]),np.asarray([len(v) for v in arrays])
            draws=np.random.default_rng(design["bootstrap_seed"]).integers(0,len(arrays),size=(design["bootstrap_samples"],len(arrays)))
            boot=sums[draws].sum(axis=1)/counts[draws].sum(axis=1)[:,None]
            diff=sums.sum(axis=0)/counts.sum()
            comparisons[f"{strategy}-vs-{reference}/money={budget:g}"]={"strategy":strategy,"reference":reference,"independent_matches":int(counts.sum()),"weeks":len(arrays),
                "mean_log_loss_difference":float(diff[0]),"confidence_interval_95":np.quantile(boot[:,0],[.025,.975]).tolist(),
                "mean_brier_difference":float(diff[1]),"mean_money_difference":float(diff[2]),"mean_utility_loss_difference":float(diff[3]),
                "utility_confidence_interval_95":np.quantile(boot[:,3],[.025,.975]).tolist(),
                "method":"paired UTC-week percentile bootstrap; random seeds first averaged within match",
                "limitations":"previously explored test season; descriptive, no research-selection/model-fit correction"}
    return metrics,comparisons


def diagnostics_summary(rows):
    supported=[r for r in rows if r["expected_gain"] is not None]
    def summary(values):
        return {"transitions":len(values),"independent_matches":len({r["match_id"] for r in values}),
            "mean_model_evsi":float(np.mean([r["model_evsi"] for r in values])) if values else None,
            "mean_estimated_gain":float(np.mean([r["expected_gain"] for r in values])) if values else None,
            "mean_realized_gain":float(np.mean([r["gain"] for r in values])) if values else None,
            "mean_estimation_error":float(np.mean([r["expected_gain"]-r["gain"] for r in values])) if values else None,
            "mean_squared_error":float(np.mean([(r["expected_gain"]-r["gain"])**2 for r in values])) if values else None,
            "raw_model_mean_squared_error":float(np.mean([(r["model_evsi"]-r["gain"])**2 for r in values])) if values else None}
    grouped=defaultdict(list)
    for row in supported:
        grouped["action_mask_bin:"+row["context_key"]+":"+str(row["evsi_bin"])].append(row)
        grouped["calibrated_sign:"+("positive" if row["expected_gain"]>0 else "nonpositive")].append(row)
        grouped["support:"+row["support_status"]].append(row)
    return {**summary(supported),"total_transitions":len(rows),"unsupported":len(rows)-len(supported),
        "support_counts":dict(Counter(r["support_status"] for r in rows)),"groups":{k:summary(v) for k,v in sorted(grouped.items())},
        "scope":"descriptive complete-transition update-gain diagnostics; dependent transitions share matches"}


def _episode_key(row):
    return row["fold"],row["match_id"],row["strategy"],row["budget_money"],row["seed"]


def run(bundle,earlier_bundle,source_round7,source_round8,output):
    source7,source8,output=Path(source_round7),Path(source_round8),Path(output)
    output.mkdir(parents=True,exist_ok=False)
    original_design=json.loads((source8/"design.json").read_text(encoding="utf-8"))
    design={**original_design,"experiment_version":VERSION,"classification":"exploratory_residual_calibration_previously_explored_history",
        "strategies":STRATEGIES,"source_round7":str(source7.resolve()),"source_round8":str(source8.resolve()),
        "predictor_policy":"all outer joint/base/quantile/q artifacts copied byte-for-byte from frozen round8",
        "new_oof":"new historical joint models with same fixed hyperparameters and frozen round7 OOF bases; no old delta reuse",
        "residual_calibration":{"bin_edges":list(BIN_EDGES),"min_support":12,"shrinkage":20.,"risk_multiplier":1.64,
            "split":"chronological match 80/20 with tied kickoff and label-availability purge", "tail_usage":"diagnostic_only"},
        "calibrated_policy":"one-step empirical predictor-update gain; descriptive LCB is not a two-step plan bound",
        "parameter_selection":"all residual/model hyperparameters frozen; only lambda chosen on original outer validation",
        "test_status":"same 290 previously explored matches, not independent confirmation"}
    write_json(output/"design.json",design)
    earlier,ea=load_episodes(earlier_bundle,lead_hours=24)
    current,ca=load_episodes(bundle,lead_hours=24)
    rows=earlier+current
    write_json(output/"data-audit.json",{"current":ca,"earlier":ea,"input_matches":len(rows)})
    shutil.copy2(source8/"catalog.json",output/"catalog.json")
    actions=json.loads((output/"catalog.json").read_text(encoding="utf-8"))["actions"]
    source_report=json.loads((source8/"report.json").read_text(encoding="utf-8"))
    contexts=extract_oof_contexts(_jsonl(source7/"oof-supervision.jsonl"))
    (output/"training-oof-contexts.jsonl").write_text("".join(canonical_json(r)+"\n" for _,r in sorted(contexts.items())),encoding="utf-8")
    source_hashes={"round7":{name:_hash(source7/name) for name in ("oof-manifest.json","oof-supervision.jsonl","report.json")},
        "round8":{name:_hash(source8/name) for name in ("report.json","design.json","episodes.jsonl","trajectories.jsonl","implementation.json","catalog.json")}}
    write_json(output/"source-artifacts.json",source_hashes)
    names=("round9_residual.py","experiment_round9.py","joint_returns.py","joint_planner.py","experiment_round8.py","contextual_partial.py",
        "contextual_identity.py","experiment_round4.py","experiment.py","data.py","state.py","value.py")
    implementation={name:_hash(Path(__file__).parent/name) for name in names}
    write_json(output/"implementation.json",implementation)
    oof=forward_joint_oof(rows,actions,contexts,source7,output)
    previous={_episode_key(r):r for r in _jsonl(source8/"episodes.jsonl")}
    previous_trace={_episode_key(r):canonical_json(r) for r in _jsonl(source8/"trajectories.jsonl")}
    folds=rolling_origin_splits(rows,initial_validation_start=design["validation_start"],validation_window=timedelta(days=30),test_window=timedelta(days=30))
    results,fold_reports,diagnostics=[],[],[]
    parity=0
    (output/"models").mkdir(exist_ok=True)
    with (output/"trajectories.jsonl").open("w",encoding="utf-8") as traces:
        for number,fold in enumerate(folds):
            old=next((f for f in source_report["folds"] if f["fold"] == number),None)
            if old is None:
                continue
            for key,part in (("training_match_ids",fold.train),("validation_match_ids",fold.validation),("test_match_ids",fold.test)):
                if set(old[key]) != {r["match_id"] for r in part}:
                    raise ValueError("Frozen outer fold identity mismatch")
            print(f"round9 fold {number}: calibrating with outer train={len(fold.train)}",flush=True)
            joint_path=output/"models"/f"fold-{number}-joint.json"
            shutil.copy2(source8/old["joint_model"]["path"],joint_path)
            joint_hash=_hash(joint_path)
            if joint_hash != old["joint_model"]["sha256"]:
                raise ValueError("Frozen reference joint hash mismatch")
            model=JointReturnPredictor.load(joint_path)
            train_ids={r["match_id"] for r in fold.train}
            prior=[r for r in oof if r["match_id"] in train_ids and _sem_time(r["result_available_at"]) <= fold.train_cutoff]
            train,tail,split=split_residual_training(prior)
            calibrator=ResidualEVSICalibrator().fit(train,as_of=split["fit_cutoff"],diagnostics_as_of=fold.train_cutoff,
                dataset_version=rows[0]["dataset_version"],joint_model=model,joint_artifact_sha256=joint_hash,calibration_samples=tail,metadata=split)
            path=output/"models"/f"fold-{number}-calibrator.json"
            calibrator.save(path)
            calibrator=ResidualEVSICalibrator.load(path,model=model,model_sha256=joint_hash)
            selected={}
            for strategy,budget in itertools.product(("evsi-one-step","evsi-two-step",*NEW_STRATEGIES),design["money_budgets"]):
                selected[(strategy,budget)]=select_lambda(fold.validation,model,actions,strategy,budget,design,calibrator)
                if strategy not in NEW_STRATEGIES and selected[(strategy,budget)][1] != old["lambda_selection"][f"{strategy}/{budget:g}"]:
                    raise ValueError("Frozen raw-EVSI lambda selection changed")
            fr={"fold":number,"split":fold.manifest(),**{k:old[k] for k in ("training_match_ids","validation_match_ids","test_match_ids")},
                "joint_training_match_ids":model.metadata["training_match_ids"],
                "joint_model":{"path":str(joint_path.relative_to(output)),"sha256":joint_hash,"version":model.model_version,"metadata":model.metadata},
                "calibrator":{"path":str(path.relative_to(output)),"sha256":_hash(path),"version":calibrator.version,"manifest":calibrator.manifest},
                "residual_split":split,"lambda_selection":{f"{s}/{b:g}":record for (s,b),(_,record) in selected.items()},
                "training_diagnostics":diagnostics_summary([calibrator.diagnose_record(r) for r in calibrator.records]),
                "calibration_diagnostics":diagnostics_summary(calibrator.calibration["records"])}
            for row in fold.test:
                for strategy,budget in itertools.product(STRATEGIES,design["money_budgets"]):
                    lam=selected[(strategy,budget)][0] if (strategy,budget) in selected else 0.
                    for seed in design["random_seeds"] if strategy == "random" else [0]:
                        result,trace=run_episode(row,model,actions,strategy=strategy,budget_money=budget,lambda_money=lam,seed=seed,design=design,calibrator=calibrator)
                        result["fold"]=number
                        trajectory={"fold":number,"match_id":row["match_id"],"strategy":strategy,"budget_money":budget,"seed":seed,"events":trace}
                        if strategy in RAW_STRATEGIES:
                            identity=_episode_key(result)
                            if any(result[k] != v for k,v in previous[identity].items() if k not in {"predictor_latency_ms","estimator_latency_ms"}):
                                raise ValueError("Frozen round8 episode changed")
                            if canonical_json(trajectory) != previous_trace[identity]:
                                raise ValueError("Frozen round8 complete decision trace changed")
                            parity+=1
                        results.append(result)
                        traces.write(canonical_json(trajectory)+"\n")
            evaluated=joint_transitions(fold.test,model,actions,{"oof_train_cutoff":str(fold.train_cutoff),
                "oof_train_ids":old["training_match_ids"],"oof_manifest_ref":"evaluation:"+joint_hash})
            fold_diagnostics=[]
            for r in evaluated:
                e=calibrator.estimate_value(r["state"],[r["action"]])[0]
                fold_diagnostics.append({"fold":number,"match_id":r["match_id"],"kickoff":r["kickoff"],"gain":r["delta"],
                    "prediction_before":r["prediction_before"],"prediction_after":r["prediction_after"],**e})
            diagnostics.extend(fold_diagnostics)
            fr["test_diagnostics"]=diagnostics_summary(fold_diagnostics)
            fold_reports.append(fr)
            write_json(output/"fold-progress.json",fold_reports)
            print(f"round9 fold {number}: completed",flush=True)
    metrics,comparisons=summarize(results,design)
    report={"status":"OK","experiment_version":VERSION,"evidence_grade":design["classification"],"input_matches":len(rows),
        "test_matches":len({r["match_id"] for r in results}),"episode_runs":len(results),"folds":fold_reports,"metrics":metrics,
        "paired_comparisons":comparisons,"update_gain_diagnostics":diagnostics_summary(diagnostics),
        "oof_samples":len(oof),"oof_matches":len({r["match_id"] for r in oof}),
        "raw_round8_parity":{"episodes":parity,"all_non_latency_episode_fields_exact":True,"complete_decision_traces_exact":True,"validation_selection_exact":True},
        "model_artifacts":[{"path":str(p.relative_to(output)),"sha256":_hash(p)} for directory in ("models","oof-models") for p in sorted((output/directory).glob("*.json"))],
        "implementation_sha256":implementation,"source_artifacts_sha256":source_hashes,"external_requests":0,"llm_tokens":0,
        "cost_interpretation":design["cost_interpretation"],"limitations":["Same 290 repeatedly explored matches; not independent confirmation",
            "Empirical fixed-predictor update gain may be negative; this is not Bayes-optimal value with optional evidence ignoring",
            "Residual calibration conditioned only on action, acquired mask, and fixed EVSI bins",
            "Descriptive clustered mean lower summary excludes joint parameter uncertainty, drift and post-selection coverage",
            "Training OOF joint models have earlier/smaller populations than deployed frozen outer models",
            "Historical complete-return transition enumeration avoids action selection censoring here but does not establish live missingness representativeness",
            "Tail diagnostics do not fit residuals or choose parameters/lambda; outer validation alone chooses lambda",
            "No new football priors, sources, lineups or tactical-context features; offline reconstructed timestamps and simulated costs remain"]}
    for name,records in (("episodes",results),("update-gain-diagnostics",diagnostics)):
        (output/(name+".jsonl")).write_text("".join(canonical_json(r)+"\n" for r in records),encoding="utf-8")
    write_json(output/"report.json",report)
    return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("bundle","earlier-bundle","source-round7","source-round8","output"):
        parser.add_argument("--"+name,required=True)
    args=parser.parse_args()
    report=run(args.bundle,args.earlier_bundle,args.source_round7,args.source_round8,args.output)
    print(json.dumps({k:report[k] for k in ("status","input_matches","test_matches","episode_runs","oof_samples")}))
