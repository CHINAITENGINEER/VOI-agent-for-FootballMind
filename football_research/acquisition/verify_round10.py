"""Independent frozen-alpha replay and accounting verification."""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from .data import load_episodes
from .joint_returns import JointReturnPredictor
from .round9_residual import ResidualEVSICalibrator
from .state import parse_time
from .verify_round4 import _rows
from .verify_round6 import KEYS
from .verify_round8 import close, require
from .verify_round9 import IndependentEvaluator, verify_calibrated_trace, verify_value_cells


FIXED = {"fixed-alpha-mean":"calibrated-mean","fixed-alpha-lcb":"calibrated-lcb"}


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def audit_accounting_record(row, alpha):
    """No calls to the production accounting helper."""
    h,m,g,c,lam=(row[k] for k in ("mean_estimate","decision_estimate","actual_gain","cost","policy_lambda"))
    expected={"decision_margin":m-lam*c,"cost_weight_difference":(lam-alpha)*c,
              "mean_minus_decision":h-m,"calibration_residual":g-h,
              "actual_utility_gain":g-alpha*c,"reported_estimated_net_gain":h-alpha*c}
    for field,value in expected.items():
        close(row[field],value,"Independent accounting "+field)
    close(sum(row[k] for k in ("decision_margin","cost_weight_difference","mean_minus_decision","calibration_residual")),
          row["actual_utility_gain"],"Independent accounting identity")
    bucket=sum(row["decision_margin"]>=edge for edge in (-.05,-.02,-.01,0.,.01,.02,.05))
    require(row["margin_bucket"]==bucket,"Boundary diagnostic bucket differs")


def verify(run, bundle, earlier_bundle, analysis_path):
    run=Path(run)
    read=lambda name:json.loads((run/name).read_bytes())
    report,design=read("report.json"),read("design.json")
    require(report["status"]=="OK","Incomplete fixed-alpha run")
    source=Path(design["source_round9"])
    old_report=json.loads((source/"report.json").read_bytes())
    old_design=json.loads((source/"design.json").read_bytes())
    old_audit=json.loads((source.parent/(source.name+"-verification.json")).read_bytes())
    require(old_audit["status"]=="VERIFIED","Original source audit incomplete")
    for name,h in old_audit["artifacts"].items():
        require(digest(source/name)==h,"Frozen round9 artifact changed")
    for name,h in read("source-artifacts.json").items():
        require(digest(source/name)==h,"Fixed-alpha source binding changed")
    for name,h in read("implementation.json").items():
        require(digest(Path(__file__).with_name(name))==h,"Frozen-alpha implementation changed")
    require(report["source_hashes"]==read("source-artifacts.json"),"Report upstream hash manifest differs")
    require(report["implementation_sha256"]==read("implementation.json"),"Report implementation manifest differs")
    require(design["frozen_alpha"]==old_design["validation_utility_money_weight"]==.5,"Prespecified fixed alpha changed")
    require(design["fixed_strategy_mapping"]==FIXED,"Fixed policy mapping differs")
    for field in ("money_budgets","request_budget","bootstrap_samples","bootstrap_seed"):
        require(design[field]==old_design[field],"Prespecified design differs")
    design_note=Path(__file__).resolve().parents[2]/"docs/research/acquisition/round10-design.md"
    require(digest(design_note)==design["design_note_sha256"],"Written diagnostic design changed")
    actions=json.loads((source/"catalog.json").read_bytes())["actions"]
    current,ca=load_episodes(bundle,lead_hours=old_design["lead_hours"])
    earlier,ea=load_episodes(earlier_bundle,lead_hours=old_design["lead_hours"])
    audit=json.loads((source/"data-audit.json").read_bytes())
    require(ca["platform_manifest_sha256"]==audit["current"]["platform_manifest_sha256"],"Current export changed")
    require(ea["platform_manifest_sha256"]==audit["earlier"]["platform_manifest_sha256"],"Earlier export changed")
    sources={r["match_id"]:r for r in earlier+current}
    folds={f["fold"]:f for f in old_report["folds"]}
    models,calibrators,parents={},{},{}
    for number,fold in folds.items():
        joint=source/fold["joint_model"]["path"]
        cal=source/fold["calibrator"]["path"]
        require(digest(joint)==fold["joint_model"]["sha256"] and digest(cal)==fold["calibrator"]["sha256"],"Frozen predictors changed")
        model=JointReturnPredictor.load(joint)
        calibrator=ResidualEVSICalibrator.load(cal,model=model,model_sha256=digest(joint))
        p,_=verify_value_cells(calibrator)
        models[number],calibrators[number],parents[number]=model,calibrator,p
    evaluator=IndependentEvaluator(sources,actions,models,calibrators,parents)
    key=lambda row:tuple(row[k] for k in KEYS)
    test,validation=list(_rows(run/"episodes.jsonl")),list(_rows(run/"validation-episodes.jsonl"))
    require(len(test)==report["test_episode_runs"] and len(validation)==report["validation_episode_runs"],"New replay count differs")
    for partition,rows in (("test",test),("validation",validation)):
        expected={(number,mid,strategy,budget,0) for number,fold in folds.items() for mid in fold[partition+"_match_ids"]
                  for strategy in FIXED for budget in design["money_budgets"]}
        require(len(rows)==len(expected) and {key(r) for r in rows}==expected,"Fixed-alpha partition coverage differs")
        for row in rows:
            require(row["partition"]==partition and row["reference_strategy"]==FIXED[row["strategy"]],"Fixed episode source mapping differs")
            require(row["lambda_money"]==design["frozen_alpha"],"Fixed policy uses selected lambda")
            model=models[row["fold"]]
            require(row["model_version"]==model.model_version and row["gain_version"]==calibrators[row["fold"]].version,"Episode model binding differs")
            require(parse_time(model.metadata["training_as_of"])<=parse_time(row["prediction_cutoff"]),"Prediction uses future model")
            require(max(parse_time(sources[mid]["result_available_at"]) for mid in folds[row["fold"]]["training_match_ids"])
                    <parse_time(row["prediction_cutoff"]),"Prediction uses unavailable actual training label")
            reference=evaluator.episode(row["fold"],row["match_id"],risk_level=row["risk_level"],budget_money=row["budget_money"],
                lambda_money=design["frozen_alpha"],request_budget=design["request_budget"])
            require(row["actions"]==reference["actions"],"Independent fixed-alpha actions differ")
            for a,b in zip(row["final_probabilities"],reference["probabilities"]):
                close(a,b,"Independent fixed-alpha posterior")
            for field in ("log_loss","brier_score","accuracy"):
                close(row[field],reference[field],"Independent fixed-alpha score")
            for field in ("money","requests"):
                close(row["cost"][field],reference[field],"Independent fixed-alpha cost")
    validation_scores=defaultdict(list)
    for row in validation:
        validation_scores[row["fold"],row["strategy"],row["budget_money"]].append(row["log_loss"]+.5*row["cost"]["money"])
    for (number,strategy,budget),values in validation_scores.items():
        candidate=next(c for c in folds[number]["lambda_selection"][f"{FIXED[strategy]}/{budget:g}"]["candidates"] if c["lambda_money"]==.5)
        close(candidate["validation_utility"],float(np.mean(values)),"Historical fixed-alpha validation candidate")
    indexed={key(r):r for r in test}
    seen=set();totals={"events":0,"charges":0,"candidates":0}
    for trace in _rows(run/"trajectories.jsonl"):
        identity=key(trace)
        require(identity in indexed and identity not in seen,"Unexpected fixed-alpha trace")
        seen.add(identity)
        row=deepcopy(indexed[identity]);row["strategy"]=FIXED[row["strategy"]]
        counts=verify_calibrated_trace(row,trace,sources[row["match_id"]],actions,evaluator,design["request_budget"])
        for field in totals:totals[field]+=counts[field]
    require(seen==set(indexed),"Missing fixed-alpha trajectories")
    print(f"round10 verified {len(test)} test + {len(validation)} validation episodes",flush=True)

    analysis_path=Path(analysis_path);analysis=json.loads(analysis_path.read_bytes())
    for name,h in analysis["source_hashes"].items():
        require(digest(name)==h,"Accounting analysis source changed")
    require(digest(Path(__file__).with_name("experiment_round10.py"))==analysis["analyzer_sha256"],"Analysis implementation changed")
    require(digest(Path(__file__).with_name("decision_boundary.py"))==analysis["accounting_sha256"],"Accounting implementation changed")
    source_episodes=[r for r in _rows(source/"episodes.jsonl") if r["strategy"] in ("base-only","calibrated-mean","calibrated-lcb")]
    all_indexed={key(r):r for r in test+source_episodes}
    base={(r["fold"],r["match_id"],r["budget_money"]):r for r in source_episodes if r["strategy"]=="base-only"}
    gain_lookup={(r["fold"],r["match_id"],tuple(r["acquired_action_ids"]),r["action_id"]):r for r in _rows(source/"update-gain-diagnostics.jsonl")}
    grouped=defaultdict(list)
    def check_diagnostic_binding(row):
        identity=row["fold"],row["match_id"],row["strategy"],row["budget_money"],0
        require(identity in all_indexed,"Diagnostic references an unknown policy episode")
        episode=all_indexed[identity]
        close(row["policy_lambda"],episode["lambda_money"],"Diagnostic execution penalty")
        truth=gain_lookup[row["fold"],row["match_id"],tuple(row["acquired_action_ids"]),row["action_id"]]
        close(row["actual_gain"],truth["gain"],"Diagnostic empirical transition gain")
        close(row["mean_estimate"],truth["expected_gain"],"Diagnostic empirical mean estimate")
        used=truth["lower_gain"] if episode["risk_level"]=="conservative" else truth["expected_gain"]
        close(row["decision_estimate"],used,"Diagnostic estimate actually used")
        action=next(a for a in actions if a["action_id"]==row["action_id"])
        close(row["cost"],action["cost"]["money"],"Diagnostic single-query cost")
    for row in analysis["selected_query_records"]:
        audit_accounting_record(row,.5)
        check_diagnostic_binding(row)
        truth=gain_lookup[row["fold"],row["match_id"],tuple(row["acquired_action_ids"]),row["action_id"]]
        close(row["actual_gain"],truth["gain"],"Accounting query empirical gain")
        close(row["mean_estimate"],truth["expected_gain"],"Accounting query estimated mean")
        aid=row["action_id"]
        action=next(a for a in actions if a["action_id"]==aid)
        close(row["cost"],action["cost"]["money"],"Accounting query unit cost")
        grouped[row["fold"],row["match_id"],row["strategy"],row["budget_money"],0].append(row)
    for row in analysis["initial_candidate_records"]:
        audit_accounting_record(row,.5)
        check_diagnostic_binding(row)
        require(row["step"]==0 and row["acquired_action_ids"]==[],"Initial-candidate population contains later state")
    accounting_seen=set()
    for row in analysis["episode_accounting_records"]:
        identity=row["fold"],row["match_id"],row["strategy"],row["budget_money"],0
        require(identity in all_indexed and identity not in accounting_seen,"Accounting episode coverage error")
        accounting_seen.add(identity)
        episode=all_indexed[identity];queries=sorted(grouped[identity],key=lambda q:q["step"])
        require([q["action_id"] for q in queries]==episode["actions"],"Accounting chosen actions differ")
        require(row["queries"]==len(queries),"Accounting query count differs")
        for field in ("decision_margin","cost_weight_difference","mean_minus_decision","calibration_residual","actual_gain","actual_utility_gain"):
            close(row[field],math.fsum(q[field] for q in queries),"Accounting episode contribution")
        prior=base[episode["fold"],episode["match_id"],episode["budget_money"]]
        close(row["actual_gain"],prior["log_loss"]-episode["log_loss"],"Independent gain telescoping")
        close(row["actual_utility_gain"],prior["log_loss"]-episode["log_loss"]-.5*episode["cost"]["money"],"Independent utility telescoping")
    require(accounting_seen=={key(r) for r in test+source_episodes if r["strategy"]!="base-only"},"Missing accounting episodes")
    return {"status":"VERIFIED","round":10,"test_matches":len({r["match_id"] for r in test}),
        "new_test_episodes":len(test),"new_validation_episodes":len(validation),
        "visible_test_events":totals["events"],"test_charges":totals["charges"],"affordable_test_candidates":totals["candidates"],
        "independent_joint_states":len(evaluator.states),"accounting_episodes":len(accounting_seen),
        "chosen_query_decompositions":len(analysis["selected_query_records"]),"initial_candidate_decompositions":len(analysis["initial_candidate_records"]),
        "source_artifacts":{name:digest(run/name) for name in ("report.json","design.json","episodes.jsonl","validation-episodes.jsonl","trajectories.jsonl","folds.json")},
        "analysis_sha256":digest(analysis_path),"verifier_sha256":digest(__file__),
        "new_independent_matches":0,"external_requests":0,"scope":"Implementation, provenance, accounting verification; exploratory existing games"}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("run","bundle","earlier-bundle","analysis","output"):
        parser.add_argument("--"+name,required=True)
    args=parser.parse_args();output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    result=verify(args.run,args.bundle,args.earlier_bundle,args.analysis)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="source_artifacts"}))
