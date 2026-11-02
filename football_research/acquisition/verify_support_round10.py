"""Independently recompute frozen fit-range and nearest-prior support claims."""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .verify_round4 import _rows
from .verify_round8 import close,require
from .verify_round9 import reference_bin


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def key(row):
    return row["fold"],row["match_id"],tuple(row["acquired_action_ids"]),row["action_id"]


def verify(run, audit_path, contrast_run):
    run,audit_path,contrast_run=Path(run),Path(audit_path),Path(contrast_run)
    audit=json.loads(audit_path.read_bytes())
    original=json.loads((run/"report.json").read_bytes())
    folds={f["fold"]:f for f in original["folds"]}
    for name,h in audit["source_sha256"].items():require(digest(run/name)==h,"Support source changed")
    for name,h in audit["model_artifact_sha256"].items():require(digest(run/name)==h,"Support model changed")
    require(digest(Path(__file__).with_name("support_round10.py"))==audit["implementation_sha256"],"Support auditor changed")
    contrast_verified=json.loads((contrast_run.parent/(contrast_run.name+"-verification.json")).read_bytes())
    require(contrast_verified["status"]=="VERIFIED","Independent empty-state forecast audit missing")
    for name,h in contrast_verified["source_artifacts"].items():require(digest(contrast_run/name)==h,"Verified fixed-alpha source changed")
    p0={}
    for partition,filename in (("test","episodes.jsonl"),("outer_validation","validation-episodes.jsonl")):
        for row in _rows(contrast_run/filename):
            if row["strategy"]=="fixed-alpha-mean" and row["budget_money"]==0:
                require(not row["actions"],"Free-state reference acquired evidence")
                p0[partition,row["fold"],row["match_id"]]=row["final_probabilities"]
    contexts={r["match_id"]:r for r in _rows(run/"training-oof-contexts.jsonl")}
    chunks=json.loads((run/"oof-manifest.json").read_bytes())["folds"]
    sizes={c["model_version"]:len(c["joint_training_match_ids"]) for c in chunks}
    parents,calibrators={},{}
    for number,fold in folds.items():
        saved=json.loads((run/fold["calibrator"]["path"]).read_bytes())
        # Artifact payload format is read directly rather than loading the new
        # support auditor or using its helper to produce expected flags.
        payload=saved.get("payload",saved)
        records=payload["records"]
        calibrators[number]=payload
        groups=defaultdict(list)
        for row in records:
            groups[row["action_id"],tuple(row["acquired_action_ids"])].append(row)
        parents[number]=groups
    diagnostic={(r["fold"],r["match_id"],tuple(r["acquired_action_ids"]),r["action_id"]):r for r in _rows(run/"update-gain-diagnostics.jsonl")}
    stage_records={"test":audit["test_records"],"outer_validation":audit["outer_validation_records"]}
    all_test={}
    for stage,records in stage_records.items():
        partition="validation" if stage=="outer_validation" else "test"
        expected={(number,mid,acq,aid) for number,f in folds.items() for mid in f[partition+"_match_ids"]
            for acq,aid in (((),"results_view"),((),"xg_view"),(("results_view",),"xg_view"),(("xg_view",),"results_view"))}
        require(len(records)==len(expected) and {key(r) for r in records}==expected,"Support public-state coverage differs")
        for row in records:
            number=row["fold"]
            before=p0[stage,number,row["match_id"]]
            for a,b in zip(row["p0"],before):close(a,b,"Independently verified empty-state prior")
            broad=parents[number][row["action_id"],tuple(row["acquired_action_ids"])]
            require(len({r["match_id"] for r in broad})==len(broad),"Support duplicates a fitted match")
            local=[r for r in broad if reference_bin(r["model_evsi"])==reference_bin(row["model_evsi"])]
            s=row["support"]
            require(s["local_support"]==len(local) and s["parent_support"]==len(broad),"Support counts differ")
            status="unsupported_parent" if len(broad)<12 else "supported_parent" if len(local)<12 else "supported_local"
            require(s["support_status"]==status and s["low_local_support"]==(len(local)<12) and s["low_parent_support"]==(len(broad)<12),"Support status differs")
            for field,population in (("evsi_local_range",local),("evsi_parent_range",broad)):
                values=[r["model_evsi"] for r in population]
                if not values:
                    require(all(v is None for v in s[field].values()),"Empty support range became numeric")
                    continue
                lo,hi=min(values),max(values);qlo,qhi=np.quantile(values,[.05,.95]);value=row["model_evsi"]
                wanted={"in_minmax":bool(lo-1e-12<=value<=hi+1e-12),"in_p05_p95":bool(qlo-1e-12<=value<=qhi+1e-12),
                        "below_min":bool(value<lo-1e-12),"above_max":bool(value>hi+1e-12)}
                require(s[field]==wanted,"Independent EVSI range differs")
            if local:
                vectors=np.asarray([contexts[r["match_id"]]["probabilities"] for r in local])
                p=np.asarray(before);dist=np.linalg.norm(vectors-p,axis=1)
                close(s["nearest_fit_p0_distance"],float(dist.min()),"Independent nearest fitted prior")
                nearest={r["match_id"] for r,d in zip(local,dist) if abs(d-dist.min())<1e-10}
                require(s["nearest_fit_p0_match_id"] in nearest,"Nearest prior identity differs")
                inside=bool(((p>=vectors.min(axis=0)-1e-12)&(p<=vectors.max(axis=0)+1e-12)).all())
                require(s["p0_coordinate_envelope"]==inside,"Coordinatewise prior range differs")
            require(row["joint_training_matches"]==len(folds[number]["joint_training_match_ids"]),"Deployed training size differs")
            if stage=="test":
                truth=diagnostic[key(row)]
                for field,tfield in (("model_evsi","model_evsi"),("actual_gain","gain"),("estimated_gain","expected_gain")):
                    close(row[field],truth[tfield],"Frozen test support diagnostic")
                all_test[key(row)]=row
    for fold in audit["folds"]:
        records=calibrators[fold["fold"]]["records"]
        values=[sizes[r["oof_predictor_version"]] for r in records]
        stats=fold["stages"]["fit"]["joint_training_matches"]
        require(stats["n"]==len(records),"Fit-size transition count differs")
        for name,value in (("min",min(values)),("max",max(values)),("mean",float(np.mean(values)))):
            close(stats[name],value,"Historical supervised model-size distribution")
    selected_count=0
    for record in audit["selected_query_support"].values():
        queries=record["records"];selected_count+=len(queries)
        require(record["actual_query_runs"]==len(queries),"Selected support count differs")
        for row in queries:
            require(row["support"]==all_test[key(row)]["support"],"Selected query support differs from complete state")
        total=sum(r["weight"] for r in queries)
        for field,get in (
            ("outside_local_evsi_minmax_rate",lambda s:s["evsi_local_range"]["in_minmax"] is False),
            ("outside_parent_evsi_minmax_rate",lambda s:s["evsi_parent_range"]["in_minmax"] is False),
            ("outside_local_evsi_p05_p95_rate",lambda s:s["evsi_local_range"]["in_p05_p95"] is False),
            ("low_local_support_rate",lambda s:s["low_local_support"]),
            ("p0_outside_coordinate_envelope_rate",lambda s:s["p0_coordinate_envelope"] is False)):
            value=sum(r["weight"] for r in queries if get(r["support"]))/total if total else None
            if value is None:require(record[field] is None,"No-query support rate must be undefined")
            else:close(record[field],value,"Selected query support proportion")
    return {"status":"VERIFIED","round":10,"test_support_records":len(audit["test_records"]),
        "validation_support_records":len(audit["outer_validation_records"]),"selected_query_support_records":selected_count,
        "fitted_model_size_distributions":len(folds),"audit_sha256":digest(audit_path),"verifier_sha256":digest(__file__),
        "new_external_requests":0,"scope":"Independent fit counts, min/max and quantile flags, nearest p0 and selected-query proportions; no calibration coverage guarantee"}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("run","audit","contrast-run","output"):parser.add_argument("--"+name,required=True)
    args=parser.parse_args();output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    result=verify(args.run,args.audit,args.contrast_run)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(result))
