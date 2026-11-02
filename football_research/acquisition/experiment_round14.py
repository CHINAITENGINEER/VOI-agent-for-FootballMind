"""Frozen empirical-gain adaptive versus committed two-step acquisition."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
import shutil

from football_research.baselines.api import canonical_json
from .analyze_round7 import match_means, paired
from .calibrated_continuation import CalibratedContinuationPlanner, VERSION as ESTIMATOR_KIND
from .data import load_episodes
from .experiment import losses, outcome
from .experiment_round4 import write_json
from .experiment_round9 import run_episode as old_run_episode
from .experiment_round11 import digest, write_lines, summarize as reference_summarize
from .joint_returns import JointReturnPredictor
from .round9_residual import ResidualEVSICalibrator
from .state import parse_time
from .verify_round4 import _rows

VERSION = "calibrated-continuation-ablation-round14-v1"
STRATEGIES = {"calibrated-adaptive-two-step":"adaptive", "calibrated-commit-two-step":"commit"}
REFERENCE = {"adaptive":"frozen-version-mean", "commit":"content-marginal-mean"}
DESIGN_SHA256 = "9739f3f33abdb7fcdefd532b2f63f2f027a93ced692cb785ef01114e4e8eefc7"


def run_episode(row,model,actions,*,strategy,budget_money,seed,design,planner):
    if STRATEGIES.get(strategy) != planner.mode:
        raise ValueError("Registered strategy and planner mode required")
    result,events = old_run_episode(row,model,actions,strategy="calibrated-mean",budget_money=budget_money,
        lambda_money=design["fixed_lambda_money"],seed=seed,design=design,calibrator=planner)
    result.update(strategy=strategy,mode=planner.mode,estimator_kind=ESTIMATOR_KIND,source_calibrator_version=planner.calibrator.version)
    return result,events


def realized_queries(row,result,events,*,alpha):
    """Outcome is consulted here only after the entire policy trace is fixed."""
    answer=[]
    for index,event in enumerate(events):
        decision=event["decision"]
        if decision["decision"] != "acquire":
            continue
        selected=next(c for c in decision["candidates"] if c["action_id"]==decision["action_id"])
        before,after=event["prediction"]["probabilities"],events[index+1]["prediction"]["probabilities"]
        gain=losses(before,outcome(row))["log_loss"]-losses(after,outcome(row))["log_loss"]
        answer.append({"match_id":row["match_id"],"kickoff":row["kickoff"],"strategy":result["strategy"],"mode":result["mode"],
            "budget_money":result["budget_money"],"step":event["step"],"action_id":decision["action_id"],
            "probabilities_before":before,"probabilities_after":after,"realized_gain":gain,
            "actual_cost":deepcopy(event["charged_cost"]),"realized_net_gain":gain-alpha*event["charged_cost"].get("money",0.),
            "immediate_expected_gain":selected["immediate_expected_gain"],"immediate_net_gain":selected["immediate_net_gain"],
            "selected_negative_immediate_net":selected["immediate_net_gain"] < 0,
            "planned_gain":selected["expected_gain"],"planned_net_gain":selected["mean_net_gain"],
            "continuation_bonus":selected["continuation_bonus"],"scope":"retrospective_real_path_only_not_runtime_feature"})
    base_loss=losses(result["initial_probabilities"],outcome(row))["log_loss"]
    if abs(math.fsum(r["realized_gain"] for r in answer)-(base_loss-result["log_loss"])) > 1e-10:
        raise ValueError("Realized query gains do not telescope to total log-loss change")
    if abs(math.fsum(r["realized_net_gain"] for r in answer)-(base_loss-result["log_loss"]-alpha*result["cost"]["money"])) > 1e-10:
        raise ValueError("Realized query net gains do not telescope to utility")
    return answer


def compare_paths(episodes,references):
    index={(r["fold"],r["match_id"],r["budget_money"],r["strategy"]):r for r in references}
    result={"episodes":len(episodes),"low_budget_episodes":0,"low_budget_identical_paths_costs_probabilities":0,
        "initial_action_changes":[],"initial_stop_to_query":[],"changed_path_ids":[]}
    for row in episodes:
        old=index[row["fold"],row["match_id"],row["budget_money"],REFERENCE[row["mode"]]]
        identity={"fold":row["fold"],"match_id":row["match_id"],"budget_money":row["budget_money"],"mode":row["mode"],
            "original_actions":old["actions"],"new_actions":row["actions"]}
        if row["initial_probabilities"] != old["initial_probabilities"]:
            raise ValueError("Frozen base prediction changed")
        if row["budget_money"] < .05:
            result["low_budget_episodes"] += 1
            for key in ("actions","cost","final_probabilities","log_loss","brier_score"):
                if row[key] != old[key]:
                    raise ValueError("No-continuation-budget path differs from one-step reference: "+key)
            result["low_budget_identical_paths_costs_probabilities"] += 1
        if row["actions"][:1] != old["actions"][:1]:
            result["initial_action_changes"].append(identity)
        if row["actions"] and not old["actions"]:
            result["initial_stop_to_query"].append(identity)
        if row["actions"] != old["actions"]:
            result["changed_path_ids"].append(identity)
    return result


def summarize(episodes,design):
    metrics,_=reference_summarize(episodes,design)
    means=match_means(episodes,design["validation_utility_money_weight"])
    pairs=[("calibrated-adaptive-two-step","calibrated-commit-two-step"),
        ("calibrated-adaptive-two-step","frozen-version-mean"),("calibrated-commit-two-step","content-marginal-mean")]
    pairs += [(s,ref) for s in STRATEGIES for ref in ("base-only","stage-prior-mean","tool-prior-mean")]
    comparisons={}
    for left,right in pairs:
        for budget in design["money_budgets"]:
            comparisons[f"{left}-vs-{right}/money={budget:g}"]={metric:paired(means[left,budget],means[right,budget],metric,
                design["bootstrap_samples"],design["bootstrap_seed"]) for metric in ("log_loss","money","utility_loss")}
    behavior={}
    for strategy in STRATEGIES:
        for budget in design["money_budgets"]:
            for fold in sorted({r["fold"] for r in episodes}):
                rows=[r for r in episodes if r["strategy"]==strategy and r["budget_money"]==budget and r["fold"]==fold]
                behavior[f"{strategy}/money={budget:g}/fold={fold}"]={"matches":len(rows),
                    "first_actions":dict(Counter(r["actions"][0] if r["actions"] else "STOP" for r in rows)),
                    "query_count_histogram":dict(Counter(len(r["actions"]) for r in rows)),
                    "paths":dict(Counter(" -> ".join([*r["actions"],"STOP"]) for r in rows))}
    return metrics,comparisons,behavior


def run(source,source_round12,source_round13,bundle,earlier_bundle,output,design_note):
    source,source_round12,source_round13,output,design_note=map(Path,(source,source_round12,source_round13,output,design_note))
    if output.exists():raise FileExistsError(output)
    if digest(design_note)!=DESIGN_SHA256:raise ValueError("Frozen round-fourteen design changed")
    roots={11:source,12:source_round12,13:source_round13}
    reports={k:json.loads((p/"report.json").read_text(encoding="utf-8")) for k,p in roots.items()}
    old_design=json.loads((source/"design.json").read_text(encoding="utf-8"))
    if any(r["status"]!="OK" for r in reports.values()):raise ValueError("Completed frozen references required")
    for k in (12,13):
        if Path(reports[k]["source_round11"]).resolve()!=source.resolve():raise ValueError("Common round-eleven source required")
    for name,h in reports[13]["implementation_sha256"].items():
        if digest(Path(__file__).with_name(name))!=h:raise ValueError("Frozen implementation changed: "+name)
    for k,p in roots.items():
        for item in reports[k]["model_artifacts"]:
            if digest(p/item["path"])!=item["sha256"]:raise ValueError("Frozen source model changed")
    output.mkdir(parents=True)
    design={**old_design,"experiment_version":VERSION,"classification":"exploratory_calibrated_continuation_surrogate_on_reused_history",
        "source_round11":str(source.resolve()),"source_round12":str(source_round12.resolve()),"source_round13":str(source_round13.resolve()),
        "strategies":[*reports[13]["metrics"].keys()],"new_strategies":STRATEGIES,"fixed_lambda_money":.5,
        "parameter_selection":"none; frozen joint and calibration; exact branchwise or committed continuation",
        "source_usage":"read original round11/12/13 references; execute only two new policies",
        "primary_comparison":"calibrated-adaptive-two-step minus calibrated-commit-two-step utility loss at budget .05",
        "continuation_semantics":"approximate calibrated empirical-gain surrogate, not coherent Bayes risk",
        "unsupported_rule":"any unsupported future branch removes continuation bonus; supported immediate value remains eligible",
        "design_note_path":str(design_note.resolve()),"design_note_sha256":digest(design_note)}
    design["strategies"]=[*old_design["strategies"],"tool-prior-mean","stage-prior-mean","content-marginal-mean",*STRATEGIES]
    for key in ("arm_matching","scope","predictor_policy","calibration_inner_split"):design.pop(key,None)
    write_json(output/"design.json",design)
    current,ca=load_episodes(bundle,lead_hours=24);earlier,ea=load_episodes(earlier_bundle,lead_hours=24)
    lookup={r["match_id"]:r for r in earlier+current}
    old_audit=json.loads((source/"data-audit.json").read_text(encoding="utf-8"))
    for name,audit in (("current",ca),("earlier",ea)):
        if audit["platform_manifest_sha256"]!=old_audit[name]["platform_manifest_sha256"]:raise ValueError("Frozen data snapshot changed")
    write_json(output/"data-audit.json",{"current":ca,"earlier":ea,"input_matches":len(lookup)})
    shutil.copy2(source/"catalog.json",output/"catalog.json")
    actions=json.loads((output/"catalog.json").read_text(encoding="utf-8"))["actions"]
    ref={partition:[row for p in roots.values() for row in _rows(p/("episodes.jsonl" if partition=="test" else "validation-episodes.jsonl"))]
        for partition in ("test","validation")}
    source_names=("report.json","design.json","catalog.json","data-audit.json","episodes.jsonl","trajectories.jsonl",
        "validation-episodes.jsonl","validation-trajectories.jsonl","transition-diagnostics.jsonl","implementation.json")
    source_hashes={name:digest(source/name) for name in source_names}
    for k in (12,13):
        source_hashes.update({f"round{k}/"+name:digest(roots[k]/name) for name in ("report.json","design.json","episodes.jsonl","validation-episodes.jsonl","implementation.json")})
    for k in (12,13):
        if reports[k]["source_artifacts_sha256"]["episodes.jsonl"]!=source_hashes["episodes.jsonl"]:raise ValueError("Earlier reference episode hash mismatch")
    write_json(output/"source-artifacts.json",source_hashes)
    implementation={name:digest(Path(__file__).with_name(name)) for name in (*reports[13]["implementation_sha256"],"calibrated_continuation.py","experiment_round14.py")}
    write_json(output/"implementation.json",implementation)
    (output/"models").mkdir()
    output_rows={"test":[],"validation":[]};folds=[];realized=[]
    with (output/"trajectories.jsonl").open("w",encoding="utf-8") as tf,(output/"validation-trajectories.jsonl").open("w",encoding="utf-8") as vf:
        for target in reports[11]["folds"]:
            number=target["fold"]
            joint_path=output/"models"/f"fold-{number}-joint.json";cal_path=output/"models"/f"fold-{number}-same-version-calibrator.json"
            shutil.copy2(source/target["joint_model"]["path"],joint_path);shutil.copy2(source/target["calibrators"]["same_version"]["path"],cal_path)
            model=JointReturnPredictor.load(joint_path);jhash,chash=digest(joint_path),digest(cal_path)
            calibrator=ResidualEVSICalibrator.load(cal_path,model=model,model_sha256=jhash)
            planners,artifacts={},{}
            for mode in STRATEGIES.values():
                planner=CalibratedContinuationPlanner(model,calibrator,mode=mode,model_sha256=jhash,calibrator_sha256=chash)
                path=output/"models"/f"fold-{number}-{mode}-continuation.json";planner.save(path)
                planners[mode]=CalibratedContinuationPlanner.load(path,model=model,model_sha256=jhash,calibrator=calibrator,calibrator_sha256=chash)
                artifacts[mode]={"path":str(path.relative_to(output)),"sha256":digest(path),"version":planner.version,"manifest":planner.manifest}
            for partition,stream in (("validation",vf),("test",tf)):
                for mid in target[partition+"_match_ids"]:
                    row=lookup[mid]
                    if partition=="validation" and parse_time(row["result_available_at"])>parse_time(target["split"]["validation_label_cutoff"]):raise ValueError("Unavailable diagnostic validation label")
                    for strategy,mode in STRATEGIES.items():
                        for budget in design["money_budgets"]:
                            result,events=run_episode(row,model,actions,strategy=strategy,budget_money=budget,seed=0,design=design,planner=planners[mode])
                            result.update(fold=number,reference_fold=target["reference_fold"],partition=partition);output_rows[partition].append(result)
                            stream.write(canonical_json({"fold":number,"reference_fold":target["reference_fold"],"partition":partition,
                                "match_id":mid,"strategy":strategy,"budget_money":budget,"seed":0,"events":events})+"\n")
                            realized.extend({**r,"fold":number,"partition":partition} for r in realized_queries(row,result,events,alpha=.5))
            fold={key:deepcopy(target[key]) for key in ("fold","reference_fold","split","training_match_ids","calibration_match_ids","validation_match_ids","test_match_ids")}
            fold.update(joint_model={"path":str(joint_path.relative_to(output)),"sha256":jhash,"version":model.model_version,"metadata":model.metadata},
                source_calibrator={"path":str(cal_path.relative_to(output)),"sha256":chash,"version":calibrator.version,"manifest":calibrator.manifest},
                planners=artifacts,fixed_lambda_money=.5)
            folds.append(fold);write_json(output/"fold-progress.json",folds)
            print(f"round14 fold {number}: both frozen continuation policies complete",flush=True)
    invariants={p:compare_paths(output_rows[p],ref[p]) for p in output_rows}
    write_json(output/"path-invariants.json",invariants)
    write_lines(output/"episodes.jsonl",output_rows["test"]);write_lines(output/"validation-episodes.jsonl",output_rows["validation"])
    write_lines(output/"realized-transitions.jsonl",realized)
    metrics,comparisons,behavior=summarize(ref["test"]+output_rows["test"],design)
    valmetrics,_,valbehavior=summarize(ref["validation"]+output_rows["validation"],design)
    grouped=defaultdict(list)
    for r in realized:grouped[(r["partition"],r["strategy"],r["budget_money"],r["step"])].append(r)
    realized_summary={f"{p}/{s}/money={b:g}/step={step}":{"queries":len(rows),"matches":len({r['match_id'] for r in rows}),
        "mean_realized_gain":math.fsum(r["realized_gain"] for r in rows)/len(rows),
        "sum_realized_gain":math.fsum(r["realized_gain"] for r in rows),"sum_realized_net_gain":math.fsum(r["realized_net_gain"] for r in rows),
        "selected_negative_immediate_net":sum(r["selected_negative_immediate_net"] for r in rows)} for (p,s,b,step),rows in grouped.items()}
    report={"status":"OK","experiment_version":VERSION,"evidence_grade":design["classification"],
        **{f"source_round{k}":str(p.resolve()) for k,p in roots.items()},"input_matches":len(lookup),"test_matches":len({r['match_id'] for r in output_rows['test']}),
        "episode_runs":len(output_rows["test"]),"new_policy_episode_runs":len(output_rows["test"]),"reference_episode_runs":len(ref["test"]),
        "combined_episode_runs":len(ref["test"])+len(output_rows["test"]),"validation_episode_runs":len(output_rows["validation"]),
        "reference_validation_episode_runs":len(ref["validation"]),"validation_unique_matches":len({r['match_id'] for r in output_rows['validation']}),
        "folds":folds,"metrics":metrics,"paired_comparisons":comparisons,
        "primary_comparison":comparisons["calibrated-adaptive-two-step-vs-calibrated-commit-two-step/money=0.05"],
        "policy_behavior_by_fold":behavior,"validation_metrics_diagnostic_only":valmetrics,"validation_behavior_diagnostic_only":valbehavior,
        "path_invariants":invariants,"realized_query_diagnostics":realized_summary,
        "model_artifacts":[{"path":str(p.relative_to(output)),"sha256":digest(p)} for p in sorted((output/"models").glob("*.json"))],
        "implementation_sha256":implementation,"source_artifacts_sha256":source_hashes,
        "new_outcome_predictors":0,"new_fitted_estimators":0,"new_wrappers":len(folds)*2,"new_independent_test_matches":0,
        "external_requests":0,"llm_tokens":0,"fixed_lambda_money":.5,
        "limitations":["Approximate empirical reward planning, not a coherent Bayes-risk guarantee",
            "200 previously explored games; no independent effectiveness confirmation",
            "Frozen return probabilities and empirical rewards may both be misspecified",
            "Expected future spending is a surrogate planning quantity, not an actual charge",
            "Model-only branches are not observed outcomes or extra independent games",
            "Simulated costs and two cached-source historical projections",
            "Week intervals omit fitting, repeated research, multiplicity and temporal shift uncertainty"]}
    write_json(output/"report.json",report)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("source","source-round12","source-round13","bundle","earlier-bundle","output","design-note"):parser.add_argument("--"+name,required=True)
    args=parser.parse_args()
    result=run(args.source,args.source_round12,args.source_round13,args.bundle,args.earlier_bundle,args.output,args.design_note)
    print(json.dumps({k:result[k] for k in ("status","test_matches","episode_runs","validation_episode_runs")}))
