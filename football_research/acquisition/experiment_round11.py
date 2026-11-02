"""Same frozen predictor for held-out calibration and later acquisition decisions."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from copy import deepcopy
from hashlib import sha256
import itertools
import json
from pathlib import Path
import shutil

import numpy as np

from football_research.baselines.api import canonical_json
from .analyze_round7 import match_means,paired
from .data import load_episodes
from .experiment import state_after
from .experiment_round4 import write_json
from .experiment_round9 import run_episode as old_run_episode,joint_transitions,diagnostics_summary
from .frozen_version import select_matched_window,matched_supervision,fit_calibrator
from .joint_returns import JointReturnPredictor
from .round9_residual import ResidualEVSICalibrator
from .state import parse_time
from .verify_round4 import _rows

VERSION="matched-frozen-predictor-calibration-round11-v1"
STRATEGIES=("base-only","all-available","random","entropy","evsi-one-step", "matched-mixed-mean","matched-mixed-lcb","frozen-version-mean","frozen-version-lcb")
CALIBRATED={"matched-mixed-mean":("matched_mixed","calibrated-mean"),"matched-mixed-lcb":("matched_mixed","calibrated-lcb"),
    "frozen-version-mean":("same_version","calibrated-mean"),"frozen-version-lcb":("same_version","calibrated-lcb")}
ARMS=("same_version","matched_mixed")


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write_lines(path,records):
    with Path(path).open("w",encoding="utf-8") as stream:
        for row in records:
            stream.write(canonical_json(row)+"\n")


def run_episode(row,model,actions,*,strategy,budget_money,seed,design,calibrators):
    if strategy not in STRATEGIES:
        raise ValueError("Unregistered frozen-version strategy")
    arm,underlying=CALIBRATED.get(strategy,(None,strategy))
    calibrator=calibrators[arm] if arm else None
    lam=design["fixed_lambda_money"] if arm or strategy=="evsi-one-step" else 0.
    result,trace=old_run_episode(row,model,actions,strategy=underlying,budget_money=budget_money,lambda_money=lam,seed=seed,design=design,calibrator=calibrator)
    result.update(strategy=strategy,calibrator_arm=arm)
    return result,trace


def summarize(episodes,design):
    means=match_means(episodes,design["validation_utility_money_weight"])
    metrics={}
    for (strategy,budget),records in sorted(means.items()):
        raw=[r for r in episodes if r["strategy"]==strategy and r["budget_money"]==budget]
        metrics[f"{strategy}/money={budget:g}"]={"independent_matches":len(records),"episodes":len(raw),
            **{key:float(np.mean([r[key] for r in records.values()])) for key in ("log_loss","brier_score","accuracy","money","requests","utility_loss")},
            "action_counts":dict(Counter(a for r in raw for a in r["actions"])),"stop_reasons":dict(Counter(r["stop_reason"] for r in raw)),
            "aggregation":"average random seeds within each match; then equal-weight matches"}
    comparisons={}
    for strategy in CALIBRATED:
        for budget in design["money_budgets"]:
            refs=("base-only","all-available","random","entropy","evsi-one-step","matched-mixed-mean","matched-mixed-lcb")
            for reference in refs:
                if strategy==reference:
                    continue
                comparisons[f"{strategy}-vs-{reference}/money={budget:g}"]={metric:paired(means[strategy,budget],means[reference,budget],metric,
                    design["bootstrap_samples"],design["bootstrap_seed"]) for metric in ("log_loss","money","utility_loss")}
    return metrics,comparisons


def transition_diagnostics(rows,model,actions,calibrators,*,fold,partition,joint_sha256):
    records=[]
    transitions=joint_transitions(rows,model,actions,{"oof_train_cutoff":model.metadata["training_as_of"],
        "oof_train_ids":model.base_predictor.metadata["training_match_ids"],"oof_manifest_ref":"sha256:"+joint_sha256})
    for sample in transitions:
        estimates={arm:calibrators[arm].estimate_value(sample["state"],[sample["action"]])[0] for arm in ARMS}
        records.append({"fold":fold,"partition":partition,"match_id":sample["match_id"],"kickoff":sample["kickoff"],
            "prediction_cutoff":sample["state"]["prediction_cutoff"],"action_id":sample["action"]["action_id"],
            "acquired_action_ids":sample["acquired_action_ids"],"prediction_before":sample["prediction_before"],
            "prediction_after":sample["prediction_after"],"model_evsi":sample["model_evsi"],"actual_gain":sample["delta"],
            "joint_model_version":model.model_version,"estimates":estimates,
            "scope":"one shared actual gain from the same reference predictor; two dependent calibration estimates"})
    return records


def diagnostic_summaries(records):
    return {arm:diagnostics_summary([{"match_id":r["match_id"],"gain":r["actual_gain"],**r["estimates"][arm]} for r in records]) for arm in ARMS}


def run(source,bundle,earlier_bundle,output,design_note):
    source,output,design_note=Path(source),Path(output),Path(design_note)
    if output.exists():
        raise FileExistsError(output)
    source_report=json.loads((source/"report.json").read_text(encoding="utf-8"))
    old_design=json.loads((source/"design.json").read_text(encoding="utf-8"))
    if source_report["status"]!="OK":
        raise ValueError("Completed source experiment required")
    for name,h in source_report["implementation_sha256"].items():
        if digest(Path(__file__).with_name(name))!=h:
            raise ValueError("Frozen source implementation changed: "+name)
    for model in source_report["model_artifacts"]:
        if digest(source/model["path"])!=model["sha256"]:
            raise ValueError("Frozen source model changed")
    output.mkdir(parents=True)
    design={**old_design,"experiment_version":VERSION,"classification":"exploratory_matched_version_control_on_reused_history",
        "source_round9":str(source.resolve()),"target_folds":[2,3,4,5,6],"reference_rule":"target fold minus two, fixed in advance",
        "strategies":STRATEGIES,"fixed_lambda_money":old_design["validation_utility_money_weight"],
        "lambda_selection":"none; alpha fixed from prior reporting objective","calibration_inner_split":"none",
        "validation_usage":"diagnostic-only; no parameter, lambda, reference, or strategy selection",
        "arm_matching":"same calibration match IDs and four transitions; generating predictor for EVSI and gain differs together",
        "parameter_selection":"none; bins/min-support/shrinkage/risk-multiplier unchanged",
        "primary_family":"all strategies use same earlier frozen reference joint predictor within target fold",
        "predictor_policy":"byte-identical source joint from k-2 for calibration same-version and all evaluation predictions",
        "test_status":"200 target matches from previously explored 290, no independent confirmation",
        "design_note_path":str(design_note.resolve()),"design_note_sha256":digest(design_note),
        "scope":"Matched diagnostic control; not an isolated causal effect of model version"}
    # Delete inherited declarations specific to previous OOF/80-20 experiments.
    for key in ("residual_calibration","new_oof","calibration_fraction","source_round7","source_round8","source_usage","round7_comparison",
                "planner_optimality","joint_model","quantile_fitting","lambda_money_grid","calibrated_policy"):
        design.pop(key,None)
    write_json(output/"design.json",design)
    current,ca=load_episodes(bundle,lead_hours=24)
    earlier,ea=load_episodes(earlier_bundle,lead_hours=24)
    rows=earlier+current
    lookup={r["match_id"]:r for r in rows}
    source_audit=json.loads((source/"data-audit.json").read_text(encoding="utf-8"))
    for part,audit in (("current",ca),("earlier",ea)):
        if audit["platform_manifest_sha256"]!=source_audit[part]["platform_manifest_sha256"]:
            raise ValueError("Source episode snapshot changed")
    write_json(output/"data-audit.json",{"current":ca,"earlier":ea,"input_matches":len(rows)})
    shutil.copy2(source/"catalog.json",output/"catalog.json")
    actions=json.loads((output/"catalog.json").read_text(encoding="utf-8"))["actions"]
    mixed=list(_rows(source/"oof-supervision.jsonl"))
    source_hashes={name:digest(source/name) for name in ("report.json","design.json","catalog.json","data-audit.json","oof-manifest.json","oof-supervision.jsonl")}
    write_json(output/"source-artifacts.json",source_hashes)
    names=("frozen_version.py","experiment_round11.py","round9_residual.py","experiment_round9.py","experiment_round8.py",
        "joint_returns.py","joint_planner.py","data.py","state.py","experiment.py","analyze_round7.py")
    implementation={name:digest(Path(__file__).with_name(name)) for name in names}
    write_json(output/"implementation.json",implementation)
    (output/"models").mkdir(); (output/"supervision").mkdir()
    source_folds={r["fold"]:r for r in source_report["folds"]}
    results,validation_results,all_diagnostics,fold_reports=[],[],[],[]
    with (output/"trajectories.jsonl").open("w",encoding="utf-8") as traces, (output/"validation-trajectories.jsonl").open("w",encoding="utf-8") as validation_traces:
        for number in design["target_folds"]:
            target,reference=source_folds[number],source_folds[number-2]
            joint_path=output/"models"/f"fold-{number}-joint.json"
            shutil.copy2(source/reference["joint_model"]["path"],joint_path)
            joint_hash=digest(joint_path)
            model=JointReturnPredictor.load(joint_path)
            if joint_hash!=reference["joint_model"]["sha256"]:
                raise ValueError("Reference predictor bytes changed")
            selected,matched,selection=select_matched_window(rows,target_fold=target,reference_model=model,mixed_samples=mixed)
            calcutoff=target["split"]["train_cutoff"]
            print(f"round11 target {number}/reference {number-2}: calibration={len(selected)} test={len(target['test_match_ids'])}",flush=True)
            arms=matched_supervision(selected,matched,model,actions,as_of=calcutoff,reference_sha256=joint_hash)
            cal_models,cal_artifacts,supervision_artifacts={},{},{}
            selection_path=output/"supervision"/f"fold-{number}-selection.json"
            write_json(selection_path,selection)
            for arm in ARMS:
                sample_path=output/"supervision"/f"fold-{number}-{arm}.jsonl"
                write_lines(sample_path,arms[arm])
                supervision_artifacts[arm]={"path":str(sample_path.relative_to(output)),"sha256":digest(sample_path),"samples":len(arms[arm]),
                    "match_ids":selection["calibration_match_ids"],"generating_model_versions":sorted({r["oof_predictor_version"] for r in arms[arm]})}
                calibrator=fit_calibrator(arms[arm],kind=arm,as_of=calcutoff,dataset_version=selected[0]["dataset_version"],model=model,model_sha256=joint_hash,
                    metadata={"target_fold":number,"reference_fold":number-2,"selection_path":str(selection_path.relative_to(output)),
                        "supervision_path":str(sample_path.relative_to(output)),"supervision_sha256":digest(sample_path),
                        "matched_calibration_match_ids":selection["calibration_match_ids"],"reference_model_version":model.model_version})
                cal_path=output/"models"/f"fold-{number}-{arm}-calibrator.json"
                calibrator.save(cal_path)
                calibrator=ResidualEVSICalibrator.load(cal_path,model=model,model_sha256=joint_hash)
                cal_models[arm]=calibrator
                cal_artifacts[arm]={"path":str(cal_path.relative_to(output)),"sha256":digest(cal_path),"version":calibrator.version,"manifest":calibrator.manifest}
            latest_reference_label=max(parse_time(lookup[mid]["result_available_at"]) for mid in model.base_predictor.metadata["training_match_ids"])
            latest_calibration_label=max(parse_time(r["result_available_at"]) for r in selected)
            fold_diagnostics={}
            for partition,collection,stream in (("validation",validation_results,validation_traces),("test",results,traces)):
                partition_rows=[lookup[mid] for mid in target[partition+"_match_ids"]]
                for row in partition_rows:
                    prediction_at=parse_time(row["prediction_cutoff"])
                    if latest_reference_label>=prediction_at or latest_calibration_label>=prediction_at or parse_time(calcutoff)>prediction_at:
                        raise ValueError("Reference or calibrator crosses evaluation prediction time")
                    if partition=="validation" and parse_time(row["result_available_at"])>parse_time(target["split"]["validation_label_cutoff"]):
                        raise ValueError("Validation outcome unavailable at original diagnostic cutoff")
                    for strategy,budget in itertools.product(STRATEGIES,design["money_budgets"]):
                        for seed in design["random_seeds"] if strategy=="random" else [0]:
                            result,events=run_episode(row,model,actions,strategy=strategy,budget_money=budget,seed=seed,design=design,calibrators=cal_models)
                            result.update(fold=number,reference_fold=number-2,partition=partition)
                            collection.append(result)
                            stream.write(canonical_json({"fold":number,"reference_fold":number-2,"partition":partition,"match_id":row["match_id"],
                                "strategy":strategy,"budget_money":budget,"seed":seed,"events":events})+"\n")
                diagnostic=transition_diagnostics(partition_rows,model,actions,cal_models,fold=number,partition=partition,joint_sha256=joint_hash)
                all_diagnostics.extend(diagnostic)
                fold_diagnostics[partition]=diagnostic_summaries(diagnostic)
            fold_report={"fold":number,"reference_fold":number-2,"split":target["split"],"reference_source_joint":reference["joint_model"],
                "training_match_ids":target["training_match_ids"],"calibration_match_ids":selection["calibration_match_ids"],
                "validation_match_ids":target["validation_match_ids"],"test_match_ids":target["test_match_ids"],
                "joint_model":{"path":str(joint_path.relative_to(output)),"sha256":joint_hash,"version":model.model_version,"metadata":model.metadata},
                "calibrators":cal_artifacts,"supervision":supervision_artifacts,"selection":{"path":str(selection_path.relative_to(output)),"sha256":digest(selection_path)},
                "fixed_lambda_money":design["fixed_lambda_money"],"diagnostics":fold_diagnostics,
                "fit_diagnostics":{arm:diagnostics_summary([cal_models[arm].diagnose_record(r) for r in cal_models[arm].records]) for arm in ARMS}}
            fold_reports.append(fold_report)
            write_json(output/"fold-progress.json",fold_reports)
            print(f"round11 fold {number}: complete",flush=True)
    metrics,comparisons=summarize(results,design)
    validation_metrics,_=summarize(validation_results,design)
    write_lines(output/"episodes.jsonl",results); write_lines(output/"validation-episodes.jsonl",validation_results)
    write_lines(output/"transition-diagnostics.jsonl",all_diagnostics)
    report={"status":"OK","experiment_version":VERSION,"evidence_grade":design["classification"],"source_round9":str(source.resolve()),
        "input_matches":len(rows),"test_matches":len({r["match_id"] for r in results}),"episode_runs":len(results),
        "validation_episode_runs":len(validation_results),"validation_unique_matches":len({r["match_id"] for r in validation_results}),
        "folds":fold_reports,"metrics":metrics,"paired_comparisons":comparisons,"validation_metrics_diagnostic_only":validation_metrics,
        "transition_diagnostics":{partition:diagnostic_summaries([r for r in all_diagnostics if r["partition"]==partition]) for partition in ("validation","test")},
        "model_artifacts":[{"path":str(path.relative_to(output)),"sha256":digest(path)} for path in sorted((output/"models").glob("*.json"))],
        "supervision_artifacts":[{"path":str(path.relative_to(output)),"sha256":digest(path)} for path in sorted((output/"supervision").glob("*"))],
        "implementation_sha256":implementation,"source_artifacts_sha256":source_hashes,"new_outcome_predictors":0,"new_calibrators":len(fold_reports)*2,
        "external_requests":0,"llm_tokens":0,"new_independent_test_matches":0,"fixed_lambda_money":design["fixed_lambda_money"],
        "limitations":["200 matches reused from previously explored 290; no independent confirmation",
            "Reference rule k-2 frozen in advance; reference predictor older than previous-round deployments",
            "Matched arms differ jointly in gain/EVSI generating models, numerical bins and resulting residual targets; no isolated causal claim",
            "Short calibration windows produce limited local support; parent fallback and descriptive uncertainty remain",
            "No inner 80/20 split; all calibration outcomes fit residuals; later original validation reports diagnostics only",
            "No lambda optimization; existing reporting alpha 0.5 fixed for every empirical/raw value policy",
            "Same source projections, offline availability reconstruction and simulated tool costs",
            "Calibration and test records overlap across folds; transition counts and repeated budgets are not independent samples",
            "Week bootstrap excludes fitted-model, repeated-research, multiplicity and drift uncertainty"]}
    write_json(output/"report.json",report)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("source","bundle","earlier-bundle","output","design-note"):
        parser.add_argument("--"+name,required=True)
    args=parser.parse_args()
    result=run(args.source,args.bundle,args.earlier_bundle,args.output,args.design_note)
    print(json.dumps({k:result[k] for k in ("status","test_matches","episode_runs","validation_episode_runs")}))
