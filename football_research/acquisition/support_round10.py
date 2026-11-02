"""Read-only support audit of frozen residual calibration and executed queries.

All support flags use public model outputs, timestamps and past training IDs.
Outcomes appear only in explicitly separate retrospective diagnostics. Changes
in calendar time and fitted-model sample size are not causally separated here.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from .data import load_episodes
from .experiment import losses,outcome,state_after
from .joint_returns import JointReturnPredictor,entropy
from .round9_residual import ResidualEVSICalibrator,context_key,evsi_bin
from .state import parse_time
from .verify_round4 import _rows

VERSION="round10-frozen-support-audit-v1"
STAGES=("fit","diagnostic_tail","outer_validation","test")


def distribution(values):
    values=np.asarray(list(values),dtype=float)
    if len(values)==0:
        return {"n":0,"min":None,"p05":None,"median":None,"mean":None,"p95":None,"max":None}
    if not np.isfinite(values).all():
        raise ValueError("Finite diagnostic values required")
    return {"n":len(values),"min":float(values.min()),"p05":float(np.quantile(values,.05)),
        "median":float(np.median(values)),"mean":float(values.mean()),"p95":float(np.quantile(values,.95)),"max":float(values.max())}


def _range(value,reference):
    if not reference:
        return {"in_minmax":None,"in_p05_p95":None,"below_min":None,"above_max":None}
    stats=distribution(reference)
    return {"in_minmax":stats["min"]-1e-12<=value<=stats["max"]+1e-12,
        "in_p05_p95":stats["p05"]-1e-12<=value<=stats["p95"]+1e-12,
        "below_min":value<stats["min"]-1e-12,"above_max":value>stats["max"]+1e-12}


def support_features(row,fit_rows,*,min_support=12,leave_one_match_out=False):
    """Label-free context lookup, fixed range checks and descriptive p0 distance."""
    key=context_key(row["action_id"],row["acquired_action_ids"])
    bucket=evsi_bin(row["model_evsi"])
    parents=[r for r in fit_rows if context_key(r["action_id"],r["acquired_action_ids"])==key]
    if len({r["match_id"] for r in parents}) != len(parents):
        raise ValueError("Fit reference must contain one record per match/action/acquired context")
    local=[r for r in parents if evsi_bin(r["model_evsi"])==bucket]
    p0=np.asarray(row["p0"],dtype=float)
    if p0.shape!=(3,) or not np.isfinite(p0).all() or (p0<0).any() or abs(float(p0.sum())-1)>1e-8:
        raise ValueError("Public p0 must be a normalized three-class vector")
    neighbors=[r for r in local if not leave_one_match_out or r["match_id"]!=row["match_id"]]
    if neighbors:
        distances=np.linalg.norm(np.asarray([r["p0"] for r in neighbors])-p0,axis=1)
        order=np.argsort(distances,kind="stable")
        distance=float(distances[order[0]])
        neighbor=neighbors[int(order[0])]["match_id"]
    else:
        distance,neighbor=None,None
    local_range=_range(row["model_evsi"],[r["model_evsi"] for r in local])
    parent_range=_range(row["model_evsi"],[r["model_evsi"] for r in parents])
    free_entropy_range=_range(row["p0_entropy"],[r["p0_entropy"] for r in local])
    ranges=[_range(float(p0[y]),[r["p0"][y] for r in local]) for y in range(3)]
    support_status="unsupported_parent" if len(parents)<min_support else "supported_parent" if len(local)<min_support else "supported_local"
    return {"context_key":key,"evsi_bin":bucket,"local_support":len(local),"parent_support":len(parents),
        "support_status":support_status,"low_local_support":len(local)<min_support,"low_parent_support":len(parents)<min_support,
        "evsi_local_range":local_range,"evsi_parent_range":parent_range,"p0_entropy_local_range":free_entropy_range,
        "p0_coordinate_envelope":all(r["in_minmax"] for r in ranges) if local else None,
        "p0_coordinate_envelope_note":"coordinatewise range only, not convex-hull membership or joint coverage",
        "nearest_fit_p0_distance":distance,"nearest_fit_p0_match_id":neighbor,
        "nearest_distance_scope":"same fixed action/acquired/bin cell; leave-one-match-out for fit" if leave_one_match_out else "same fixed action/acquired/bin fit cell",
        "local_fit_evsi_distribution":distribution(r["model_evsi"] for r in local)}


def _summary(rows):
    report={"transitions":len(rows),"unique_matches":len({r["match_id"] for r in rows}),
        "model_versions":sorted({r["joint_model_version"] for r in rows}),
        "model_training_cutoff_min":min((r["joint_training_cutoff"] for r in rows),key=parse_time,default=None),
        "model_training_cutoff_max":max((r["joint_training_cutoff"] for r in rows),key=parse_time,default=None),
        "kickoff_min":min((r["kickoff"] for r in rows),key=parse_time,default=None),
        "kickoff_max":max((r["kickoff"] for r in rows),key=parse_time,default=None)}
    for field in ("model_evsi","p0_entropy","current_entropy","joint_training_matches","base_training_matches","training_cutoff_age_days",
                  "joint_training_size_ratio_to_deployed"):
        report[field]=distribution(r[field] for r in rows)
    report["p0_by_class"]={name:distribution(r["p0"][i] for r in rows) for i,name in enumerate(("home","draw","away"))}
    report["support_status_counts"]=dict(Counter(r["support"]["support_status"] for r in rows))
    flags={"outside_local_evsi_minmax":("evsi_local_range","in_minmax"),
        "outside_parent_evsi_minmax":("evsi_parent_range","in_minmax"),
        "outside_local_evsi_p05_p95":("evsi_local_range","in_p05_p95"),
        "outside_local_p0_entropy_minmax":("p0_entropy_local_range","in_minmax")}
    for name,(a,b) in flags.items():
        checked=[r for r in rows if r["support"][a][b] is not None]
        n=sum(not r["support"][a][b] for r in checked)
        report[name]={"count":n,"checked":len(checked),"fraction":n/len(checked) if checked else None}
    report["nearest_fit_p0_distance"]=distribution(r["support"]["nearest_fit_p0_distance"] for r in rows if r["support"]["nearest_fit_p0_distance"] is not None)
    gains=[r for r in rows if r.get("actual_gain") is not None]
    report["offline_outcomes"]={"scope":"retrospective diagnostics only; excluded from support-feature construction",
        "gain":distribution(r["actual_gain"] for r in gains),
        "estimated_gain":distribution(r["estimated_gain"] for r in gains if r.get("estimated_gain") is not None)}
    return report


def _public_record(row,stage,fold,p0,before,evsi,aid,acquired,model_meta,deployed_n):
    cutoff=parse_time(model_meta["training_cutoff"])
    prediction=parse_time(row["prediction_cutoff"])
    # The deployment contract permits the declared fit cutoff to equal the
    # prediction cutoff. Actual training labels must still be strictly earlier.
    latest_label=parse_time(model_meta["latest_training_label_at"])
    if cutoff>prediction or latest_label>=prediction or latest_label>cutoff:
        raise ValueError("Predictor cutoff/actual training labels cross historical prediction boundary")
    return {"fold":fold,"stage":stage,"match_id":row["match_id"],"kickoff":row["kickoff"],
        "prediction_cutoff":row["prediction_cutoff"],"action_id":aid,"acquired_action_ids":sorted(acquired),"model_evsi":float(evsi),
        "evsi_bin":evsi_bin(evsi),"p0":list(p0),"p0_entropy":entropy(p0),"current_entropy":entropy(before),
        "joint_model_version":model_meta["version"],"joint_training_cutoff":model_meta["training_cutoff"],
        "latest_training_label_at":model_meta["latest_training_label_at"],
        "joint_training_matches":model_meta["joint_training_matches"],"base_training_matches":model_meta["base_training_matches"],
        "training_cutoff_age_days":(prediction-cutoff).total_seconds()/86400,
        "joint_training_size_ratio_to_deployed":model_meta["joint_training_matches"]/deployed_n}


def _identity(row):
    return row["fold"],row["match_id"],tuple(sorted(row["acquired_action_ids"])),row["action_id"]


def audit(run,bundle,earlier_bundle,output):
    run,output=Path(run),Path(output)
    if output.exists():
        raise FileExistsError(output)
    report=json.loads((run/"report.json").read_text(encoding="utf-8"))
    if report.get("status")!="OK":
        raise ValueError("Completed frozen round9 report required")
    actions=json.loads((run/"catalog.json").read_text(encoding="utf-8"))["actions"]
    source_audit=json.loads((run/"data-audit.json").read_text(encoding="utf-8"))
    current,ca=load_episodes(bundle,lead_hours=24)
    earlier,ea=load_episodes(earlier_bundle,lead_hours=24)
    for part,fresh in (("current",ca),("earlier",ea)):
        if fresh["platform_manifest_sha256"]!=source_audit[part]["platform_manifest_sha256"]:
            raise ValueError("Frozen episode source manifest changed")
    rows={r["match_id"]:r for r in earlier+current}
    contexts={r["match_id"]:r for r in _rows(run/"training-oof-contexts.jsonl")}
    supervision=list(_rows(run/"oof-supervision.jsonl"))
    manifests=json.loads((run/"oof-manifest.json").read_text(encoding="utf-8"))
    chunks={"sha256:"+r["sha256"]:r for r in manifests["folds"]}
    source_files=("report.json","catalog.json","data-audit.json","oof-supervision.jsonl","oof-manifest.json","training-oof-contexts.jsonl",
                  "update-gain-diagnostics.jsonl","episodes.jsonl","trajectories.jsonl")
    source_hashes={name:sha256((run/name).read_bytes()).hexdigest() for name in source_files}
    artifact_hashes={}
    for chunk in chunks.values():
        for key,hashkey in (("artifact","sha256"),("base_artifact","base_sha256")):
            path=run/chunk[key]
            actual=sha256(path.read_bytes()).hexdigest()
            if actual!=chunk[hashkey]:
                raise ValueError("Frozen OOF model hash changed")
            artifact_hashes[chunk[key]]=actual
    test_diagnostics={(r["fold"],r["match_id"],tuple(r["acquired_action_ids"]),r["action_id"]):r for r in _rows(run/"update-gain-diagnostics.jsonl")}
    all_records,fold_reports=[],[]
    for fold in report["folds"]:
        number=fold["fold"]
        model_path=run/fold["joint_model"]["path"]
        calibrator_path=run/fold["calibrator"]["path"]
        for meta in (fold["joint_model"],fold["calibrator"]):
            actual=sha256((run/meta["path"]).read_bytes()).hexdigest()
            if actual!=meta["sha256"]:
                raise ValueError("Frozen outer artifact changed")
            artifact_hashes[meta["path"]]=actual
        model=JointReturnPredictor.load(model_path)
        calibrator=ResidualEVSICalibrator.load(calibrator_path,model=model,model_sha256=fold["joint_model"]["sha256"])
        deployment_n=len(model.records)
        stages={stage:[] for stage in STAGES}
        for stage,idkey,availability in (("fit","training_match_ids",calibrator.manifest["trained_as_of"]),
                ("diagnostic_tail","calibration_match_ids",calibrator.manifest["diagnostics_as_of"])):
            ids=set(calibrator.manifest[idkey])
            for sample in supervision:
                if sample["match_id"] not in ids:
                    continue
                chunk=chunks[sample["oof_manifest_ref"]]
                prior=contexts[sample["match_id"]]
                if (prior["oof_manifest_ref"]!="sha256:"+chunk["base_sha256"] or prior["oof_predictor_version"]!=chunk["base_model_version"]
                        or sample["oof_predictor_version"]!=chunk["model_version"]):
                    raise ValueError("OOF prior/joint artifact identity mismatch")
                if parse_time(sample["result_available_at"])>parse_time(availability):
                    raise ValueError("Outcome unavailable to historical fit/diagnostic cutoff")
                base_row=rows[sample["match_id"]]
                record=_public_record(base_row,stage,number,prior["probabilities"],sample["prediction_before"],sample["model_evsi"],
                    sample["action"]["action_id"],sample["acquired_action_ids"],{"training_cutoff":chunk["cutoff"],"version":chunk["model_version"],
                    "joint_training_matches":len(chunk["joint_training_match_ids"]),"base_training_matches":len(chunk["base_training_match_ids"]),
                    "latest_training_label_at":max((rows[mid]["result_available_at"] for mid in chunk["base_training_match_ids"]),key=parse_time)},deployment_n)
                estimate=calibrator.estimate_numbers(record["action_id"],record["acquired_action_ids"],record["model_evsi"])
                record.update(actual_gain=sample["delta"],estimated_gain=estimate["expected_gain"],result_available_at=sample["result_available_at"])
                stages[stage].append(record)
        for stage,idkey in (("outer_validation","validation_match_ids"),("test","test_match_ids")):
            for mid in fold[idkey]:
                row=rows[mid]
                if stage=="outer_validation" and parse_time(row["result_available_at"])>parse_time(fold["split"]["validation_label_cutoff"]):
                    raise ValueError("Validation result unavailable at original selection cutoff")
                p0=model.predict_partial(state_after(row,actions,()))["probabilities"]
                for acquired in ((),("results_view",),("xg_view",)):
                    state=state_after(row,actions,acquired)
                    before=model.predict_partial(state)["probabilities"]
                    for action in actions:
                        aid=action["action_id"]
                        if aid in acquired:
                            continue
                        evsi=model.enumerate_action(state,action)["evsi"]
                        record=_public_record(row,stage,number,p0,before,evsi,aid,acquired,
                            {"training_cutoff":model.metadata["training_as_of"],"version":model.model_version,
                             "joint_training_matches":deployment_n,"base_training_matches":len(model.base_predictor.metadata["training_match_ids"]),
                             "latest_training_label_at":max((rows[mid]["result_available_at"] for mid in model.base_predictor.metadata["training_match_ids"]),key=parse_time)},deployment_n)
                        # Evaluation-only outcome follows the label-free state/EVSI projection.
                        after=model.predict_partial(state_after(row,actions,(*acquired,aid)))["probabilities"]
                        gain=losses(before,outcome(row))["log_loss"]-losses(after,outcome(row))["log_loss"]
                        estimate=calibrator.estimate_numbers(aid,acquired,evsi)
                        if stage=="test":
                            frozen=test_diagnostics[_identity(record)]
                            if any(abs(a-b)>1e-11 for a,b in ((evsi,frozen["model_evsi"]),(gain,frozen["gain"]),(estimate["expected_gain"],frozen["expected_gain"]))):
                                raise ValueError("Frozen complete-test transition changed")
                        record.update(actual_gain=gain,estimated_gain=estimate["expected_gain"],result_available_at=row["result_available_at"])
                        stages[stage].append(record)
        fit_rows=stages["fit"]
        fit_parents=defaultdict(list)
        for row in fit_rows:
            fit_parents[context_key(row["action_id"],row["acquired_action_ids"])].append(row)
        for stage,records in stages.items():
            for row in records:
                row["support"]=support_features(row,fit_parents[context_key(row["action_id"],row["acquired_action_ids"])],leave_one_match_out=stage=="fit")
            all_records.extend(records)
        keys=sorted({(context_key(r["action_id"],r["acquired_action_ids"]),r["evsi_bin"]) for records in stages.values() for r in records})
        cells=[]
        for key,bucket in keys:
            aid,acquired=json.loads(key)
            cells.append({"action_id":aid,"acquired_action_ids":acquired,"evsi_bin":bucket,
                "stages":{stage:_summary([r for r in records if context_key(r["action_id"],r["acquired_action_ids"])==key and r["evsi_bin"]==bucket])
                    for stage,records in stages.items()}})
        fold_reports.append({"fold":number,"deployed_joint_training_matches":deployment_n,
            "deployed_base_training_matches":len(model.base_predictor.metadata["training_match_ids"]),
            "deployed_joint_training_cutoff":model.metadata["training_as_of"],"calibrator_fit_cutoff":calibrator.manifest["trained_as_of"],
            "calibrator_diagnostic_cutoff":calibrator.manifest["diagnostics_as_of"],"validation_label_cutoff":fold["split"]["validation_label_cutoff"],
            "stages":{stage:_summary(records) for stage,records in stages.items()},"cells":cells})
        print(f"round10 support fold {number}: "+", ".join(f"{s}={len(r)}" for s,r in stages.items()),flush=True)
    test_records=[r for r in all_records if r["stage"]=="test"]
    test_lookup={_identity(r):r for r in test_records}
    episode_rows=list(_rows(run/"episodes.jsonl"))
    seed_counts=Counter((r["strategy"],r["budget_money"],r["match_id"]) for r in episode_rows)
    grouped_queries=defaultdict(list)
    for trajectory in _rows(run/"trajectories.jsonl"):
        if trajectory["strategy"] not in {"calibrated-mean","calibrated-lcb"}:
            continue
        acquired=[]
        for event in trajectory["events"]:
            if "charged_cost" not in event:
                continue
            aid=event["decision"]["action_id"]
            ident=(trajectory["fold"],trajectory["match_id"],tuple(sorted(acquired)),aid)
            row=test_lookup[ident]
            key=(trajectory["strategy"],trajectory["budget_money"])
            grouped_queries[key].append({"fold":trajectory["fold"],"match_id":trajectory["match_id"],"action_id":aid,
                "acquired_action_ids":list(acquired),"seed":trajectory["seed"],"weight":1/seed_counts[key[0],key[1],trajectory["match_id"]],
                "support":row["support"],"model_evsi":row["model_evsi"],"p0":row["p0"],"p0_entropy":row["p0_entropy"],
                "actual_gain_offline":row["actual_gain"],"estimated_gain":row["estimated_gain"]})
            acquired.append(aid)
    queries={}
    for strategy,budget in sorted({(r["strategy"],r["budget_money"]) for r in episode_rows if r["strategy"] in {"calibrated-mean","calibrated-lcb"}}):
        records=grouped_queries[strategy,budget]
        weight=sum(r["weight"] for r in records)
        def rate(predicate):
            return sum(r["weight"] for r in records if predicate(r))/weight if weight else None
        queries[f"{strategy}/money={budget:g}"]={"actual_query_runs":len(records),"seed_averaged_query_count":weight,
            "queried_unique_matches":len({r["match_id"] for r in records}),
            "unique_query_states":len({(r["match_id"],r["action_id"],tuple(r["acquired_action_ids"])) for r in records}),
            "outside_local_evsi_minmax_rate":rate(lambda r:r["support"]["evsi_local_range"]["in_minmax"] is False),
            "outside_parent_evsi_minmax_rate":rate(lambda r:r["support"]["evsi_parent_range"]["in_minmax"] is False),
            "outside_local_evsi_p05_p95_rate":rate(lambda r:r["support"]["evsi_local_range"]["in_p05_p95"] is False),
            "low_local_support_rate":rate(lambda r:r["support"]["low_local_support"]),
            "low_parent_support_rate":rate(lambda r:r["support"]["low_parent_support"]),
            "p0_outside_coordinate_envelope_rate":rate(lambda r:r["support"]["p0_coordinate_envelope"] is False),
            "weighted_mean_nearest_p0_distance":sum(r["weight"]*r["support"]["nearest_fit_p0_distance"] for r in records if r["support"]["nearest_fit_p0_distance"] is not None)/sum(r["weight"] for r in records if r["support"]["nearest_fit_p0_distance"] is not None)
                if any(r["support"]["nearest_fit_p0_distance"] is not None for r in records) else None,
            "scope":"each query weighted by inverse seeds for its match/policy/budget; queried-state proportions, not new independent samples",
            "records":records}
    result={"status":"AUDITED_DESCRIPTIVE","schema":VERSION,"run":str(run.resolve()),"new_fitted_models":0,
        "test_matches":len({r["match_id"] for r in test_records}),"test_transitions":len(test_records),
        "folds":fold_reports,"complete_test_summary":_summary(test_records),"test_records":test_records,
        "outer_validation_records":[r for r in all_records if r["stage"]=="outer_validation"],"selected_query_support":queries,
        "design":{"fixed_evsi_bins":[0,.02,.05,.10,"infinity"],"min_local_and_parent_support":12,
            "ranges":"fit-cell min/max and 5/95 percentiles, tolerance 1e-12 only; no selected threshold",
            "p0_distance":"Euclidean nearest fitted p0 in same action/acquired/bin; leave-one-match-out for fit; no threshold",
            "p0_envelope":"coordinatewise min/max, explicitly not convex hull",
            "support_features_use_outcomes":False,"validation_outcomes":"checked available by original validation_label_cutoff; diagnostic only",
            "new_strategy_or_parameter_search":False},
        "limitations":["Time and joint-model training size vary together; this audit does not identify drift as the cause",
            "Range membership is descriptive coverage, not conditional calibration or causal transportability",
            "Five/ninety-five percentiles intentionally exclude tails even under a stable process",
            "Nearest-p0 distance alone is not distributional density or sufficiency of p0",
            "Fit/tail overlap across folds; four transitions share matches; repeated budgets and random seeds add no independent matches",
            "Historical reconstructed feature availability and simulated acquisition costs remain",
            "Actual outcomes used only for retrospective gain summaries, never for support flags or thresholds",
            "Same 290 previously explored matches, no independent confirmation"],
        "source_sha256":source_hashes,"model_artifact_sha256":artifact_hashes,
        "implementation_sha256":sha256(Path(__file__).read_bytes()).hexdigest()}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("run","bundle","earlier-bundle","output"):
        parser.add_argument("--"+name,required=True)
    args=parser.parse_args()
    result=audit(args.run,args.bundle,args.earlier_bundle,args.output)
    print(json.dumps({k:result[k] for k in ("status","test_matches","test_transitions","new_fitted_models")}))
