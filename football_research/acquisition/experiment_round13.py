"""Execute a content-marginal query-value ablation with frozen predictors."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil

from football_research.baselines.api import canonical_json
from .analyze_round7 import match_means, paired
from .content_marginal import ContentMarginalGainEstimator, VERSION as ESTIMATOR_KIND
from .data import load_episodes
from .experiment_round4 import write_json
from .experiment_round9 import run_episode as old_run_episode
from .experiment_round11 import digest, write_lines, summarize as reference_summarize
from .joint_returns import JointReturnPredictor
from .round9_residual import ResidualEVSICalibrator
from .state import parse_time
from .verify_round4 import _rows

VERSION = "content-marginal-acquisition-ablation-round13-v1"
STRATEGY = "content-marginal-mean"
DESIGN_SHA256 = "dc26af59df5fc7533130c48a2767e643a7d86835f8b8610c16c71ac724309054"


def run_episode(row,model,actions,*,budget_money,seed,design,estimator):
    result,events = old_run_episode(row,model,actions,strategy="calibrated-mean",budget_money=budget_money,
        lambda_money=design["fixed_lambda_money"],seed=seed,design=design,calibrator=estimator)
    result.update(strategy=STRATEGY,estimator_kind=ESTIMATOR_KIND,erasure_mode=estimator.manifest["erasure_mode"],
        source_calibrator_version=estimator.calibrator.version)
    return result,events


def compare_paths(episodes,references):
    index = {(r["fold"],r["match_id"],r["budget_money"]):r for r in references if r["strategy"] == "frozen-version-mean"}
    result = {"episodes":len(episodes),"initial_action_equal":0,"initial_stop_preserved":0,
        "low_budget_identical_paths_costs_probabilities":0,"low_budget_episodes":0,
        "changed_path_ids":[],"same_initial_gains_scope":"native initial candidates are checked by independent trajectory verifier"}
    for row in episodes:
        old = index[row["fold"],row["match_id"],row["budget_money"]]
        first = row["actions"][:1]
        if first != old["actions"][:1] or row["initial_probabilities"] != old["initial_probabilities"]:
            raise ValueError("Content marginalization changed initial decision or free prediction")
        result["initial_action_equal"] += 1
        if not old["actions"]:
            if row["actions"]:
                raise ValueError("Initial STOP was not preserved")
            result["initial_stop_preserved"] += 1
        if row["budget_money"] < .05:
            result["low_budget_episodes"] += 1
            for key in ("actions","cost","final_probabilities","log_loss","brier_score"):
                if row[key] != old[key]:
                    raise ValueError("Unaffordable-second-query path differs: "+key)
            result["low_budget_identical_paths_costs_probabilities"] += 1
        if row["actions"] != old["actions"]:
            result["changed_path_ids"].append({"fold":row["fold"],"match_id":row["match_id"],"budget_money":row["budget_money"],
                "original_actions":old["actions"],"content_marginal_actions":row["actions"]})
    return result


def summarize(episodes,design):
    metrics,_ = reference_summarize(episodes,design)
    means = match_means(episodes,design["validation_utility_money_weight"])
    comparisons = {}
    pairs = [("frozen-version-mean",STRATEGY)]+[(STRATEGY,s) for s in
        ("base-only","stage-prior-mean","tool-prior-mean","all-available","entropy","evsi-one-step")]
    for left,right in pairs:
        for budget in design["money_budgets"]:
            comparisons[f"{left}-vs-{right}/money={budget:g}"] = {metric:paired(means[left,budget],means[right,budget],metric,
                design["bootstrap_samples"],design["bootstrap_seed"]) for metric in ("log_loss","money","utility_loss")}
    behavior = {}
    for strategy in (STRATEGY,"frozen-version-mean"):
        for budget in design["money_budgets"]:
            for fold in sorted({r["fold"] for r in episodes}):
                rows = [r for r in episodes if r["strategy"]==strategy and r["budget_money"]==budget and r["fold"]==fold]
                behavior[f"{strategy}/money={budget:g}/fold={fold}"] = {"matches":len(rows),
                    "first_actions":dict(Counter(r["actions"][0] if r["actions"] else "STOP" for r in rows)),
                    "query_count_histogram":dict(Counter(len(r["actions"]) for r in rows)),
                    "paths":dict(Counter(" -> ".join([*r["actions"],"STOP"]) for r in rows))}
    return metrics,comparisons,behavior


def run(source,source_round12,bundle,earlier_bundle,output,design_note):
    source,source_round12,output,design_note = map(Path,(source,source_round12,output,design_note))
    if output.exists():
        raise FileExistsError(output)
    if digest(design_note) != DESIGN_SHA256:
        raise ValueError("Frozen round-thirteen design changed")
    previous = json.loads((source/"report.json").read_text(encoding="utf-8"))
    priors_report = json.loads((source_round12/"report.json").read_text(encoding="utf-8"))
    old_design = json.loads((source/"design.json").read_text(encoding="utf-8"))
    if previous["status"] != "OK" or priors_report["status"] != "OK":
        raise ValueError("Completed frozen source experiments required")
    if Path(priors_report["source_round11"]).resolve() != source.resolve():
        raise ValueError("Round-twelve references must share the same round-eleven source")
    for name,h in priors_report["implementation_sha256"].items():
        if digest(Path(__file__).with_name(name)) != h:
            raise ValueError("Frozen source implementation changed: "+name)
    for root,report in ((source,previous),(source_round12,priors_report)):
        for item in report["model_artifacts"]:
            if digest(root/item["path"]) != item["sha256"]:
                raise ValueError("Frozen source model changed")
    output.mkdir(parents=True)
    design = {**old_design,"experiment_version":VERSION,"classification":"exploratory_content_erasure_ablation_on_reused_history",
        "source_round11":str(source.resolve()),"source_round12":str(source_round12.resolve()),
        "strategies":[*old_design["strategies"],"tool-prior-mean","stage-prior-mean",STRATEGY],"new_strategies":[STRATEGY],
        "fixed_lambda_money":.5,"parameter_selection":"none; frozen joint and same-version residual calibrator; exact finite sum",
        "source_usage":"read frozen reference episodes; execute only new content-marginal policy",
        "primary_comparison":"frozen-version-mean minus content-marginal-mean utility loss at budget .05",
        "content_marginalization":"same free prior and acquired set; calibrate branch EVSI first, then free-joint-weighted average",
        "unsupported_rule":"any positive-mass unsupported branch makes whole candidate unsupported; no reweighting",
        "design_note_path":str(design_note.resolve()),"design_note_sha256":digest(design_note)}
    for key in ("arm_matching","scope","predictor_policy","calibration_inner_split"):
        design.pop(key,None)
    write_json(output/"design.json",design)
    current,ca = load_episodes(bundle,lead_hours=24)
    earlier,ea = load_episodes(earlier_bundle,lead_hours=24)
    lookup = {r["match_id"]:r for r in earlier+current}
    old_audit = json.loads((source/"data-audit.json").read_text(encoding="utf-8"))
    for name,audit in (("current",ca),("earlier",ea)):
        if audit["platform_manifest_sha256"] != old_audit[name]["platform_manifest_sha256"]:
            raise ValueError("Frozen data snapshot changed")
    write_json(output/"data-audit.json",{"current":ca,"earlier":ea,"input_matches":len(lookup)})
    shutil.copy2(source/"catalog.json",output/"catalog.json")
    actions = json.loads((output/"catalog.json").read_text(encoding="utf-8"))["actions"]
    ref11, val11 = list(_rows(source/"episodes.jsonl")),list(_rows(source/"validation-episodes.jsonl"))
    ref12, val12 = list(_rows(source_round12/"episodes.jsonl")),list(_rows(source_round12/"validation-episodes.jsonl"))
    source_names = ("report.json","design.json","catalog.json","data-audit.json","episodes.jsonl","trajectories.jsonl",
        "validation-episodes.jsonl","validation-trajectories.jsonl","transition-diagnostics.jsonl","implementation.json")
    source_hashes = {name:digest(source/name) for name in source_names}
    source_hashes.update({"round12/"+name:digest(source_round12/name) for name in ("report.json","design.json","episodes.jsonl","validation-episodes.jsonl","implementation.json")})
    if priors_report["source_artifacts_sha256"]["episodes.jsonl"] != source_hashes["episodes.jsonl"]:
        raise ValueError("Round-eleven episode reference changed after round twelve")
    write_json(output/"source-artifacts.json",source_hashes)
    names = (*priors_report["implementation_sha256"],"content_marginal.py","experiment_round13.py")
    implementation = {name:digest(Path(__file__).with_name(name)) for name in names}
    write_json(output/"implementation.json",implementation)
    (output/"models").mkdir()
    episodes,validation,folds = [],[],[]
    with (output/"trajectories.jsonl").open("w",encoding="utf-8") as tf, (output/"validation-trajectories.jsonl").open("w",encoding="utf-8") as vf:
        for target in previous["folds"]:
            number = target["fold"]
            joint_path = output/"models"/f"fold-{number}-joint.json"
            cal_path = output/"models"/f"fold-{number}-same-version-calibrator.json"
            shutil.copy2(source/target["joint_model"]["path"],joint_path)
            shutil.copy2(source/target["calibrators"]["same_version"]["path"],cal_path)
            model = JointReturnPredictor.load(joint_path)
            jhash,chash = digest(joint_path),digest(cal_path)
            calibrator = ResidualEVSICalibrator.load(cal_path,model=model,model_sha256=jhash)
            estimator = ContentMarginalGainEstimator(model,calibrator,model_sha256=jhash,calibrator_sha256=chash)
            wrapper_path = output/"models"/f"fold-{number}-content-marginal.json"
            estimator.save(wrapper_path)
            estimator = ContentMarginalGainEstimator.load(wrapper_path,model=model,model_sha256=jhash,calibrator=calibrator,calibrator_sha256=chash)
            for partition,results,stream in (("validation",validation,vf),("test",episodes,tf)):
                for mid in target[partition+"_match_ids"]:
                    row = lookup[mid]
                    if partition=="validation" and parse_time(row["result_available_at"]) > parse_time(target["split"]["validation_label_cutoff"]):
                        raise ValueError("Validation diagnostic label unavailable at original cutoff")
                    for budget in design["money_budgets"]:
                        result,events = run_episode(row,model,actions,budget_money=budget,seed=0,design=design,estimator=estimator)
                        result.update(fold=number,reference_fold=target["reference_fold"],partition=partition)
                        results.append(result)
                        stream.write(canonical_json({"fold":number,"reference_fold":target["reference_fold"],"partition":partition,
                            "match_id":mid,"strategy":STRATEGY,"budget_money":budget,"seed":0,"events":events})+"\n")
            fold = {key:deepcopy(target[key]) for key in ("fold","reference_fold","split","training_match_ids","calibration_match_ids","validation_match_ids","test_match_ids")}
            fold.update(joint_model={"path":str(joint_path.relative_to(output)),"sha256":jhash,"version":model.model_version,"metadata":model.metadata},
                source_calibrator={"path":str(cal_path.relative_to(output)),"sha256":chash,"version":calibrator.version,"manifest":calibrator.manifest},
                content_marginal={"path":str(wrapper_path.relative_to(output)),"sha256":digest(wrapper_path),"version":estimator.version,"manifest":estimator.manifest},
                fixed_lambda_money=.5)
            folds.append(fold)
            write_json(output/"fold-progress.json",folds)
            print(f"round13 fold {number}: frozen wrapper and new policy complete",flush=True)
    invariant = {"test":compare_paths(episodes,ref11),"validation":compare_paths(validation,val11)}
    write_json(output/"path-invariants.json",invariant)
    write_lines(output/"episodes.jsonl",episodes); write_lines(output/"validation-episodes.jsonl",validation)
    metrics,comparisons,behavior = summarize(ref11+ref12+episodes,design)
    valmetrics,_,valbehavior = summarize(val11+val12+validation,design)
    report = {"status":"OK","experiment_version":VERSION,"evidence_grade":design["classification"],
        "source_round11":str(source.resolve()),"source_round12":str(source_round12.resolve()),"input_matches":len(lookup),
        "test_matches":len({r["match_id"] for r in episodes}),"episode_runs":len(episodes),"new_policy_episode_runs":len(episodes),
        "reference_episode_runs":len(ref11)+len(ref12),"combined_episode_runs":len(ref11)+len(ref12)+len(episodes),
        "validation_episode_runs":len(validation),"reference_validation_episode_runs":len(val11)+len(val12),
        "validation_unique_matches":len({r["match_id"] for r in validation}),"folds":folds,
        "metrics":metrics,"paired_comparisons":comparisons,"primary_comparison":comparisons["frozen-version-mean-vs-content-marginal-mean/money=0.05"],
        "policy_behavior_by_fold":behavior,"validation_metrics_diagnostic_only":valmetrics,"validation_behavior_diagnostic_only":valbehavior,
        "path_invariants":invariant,"model_artifacts":[{"path":str(p.relative_to(output)),"sha256":digest(p)} for p in sorted((output/"models").glob("*.json"))],
        "implementation_sha256":implementation,"source_artifacts_sha256":source_hashes,
        "new_outcome_predictors":0,"new_fitted_estimators":0,"new_wrappers":len(folds),"new_independent_test_matches":0,
        "external_requests":0,"llm_tokens":0,"fixed_lambda_money":.5,
        "limitations":["Same explored 200 test games; no independent confirmation",
            "Content marginalization uses possibly misspecified frozen return probabilities",
            "One-step calibrated update gain is not Bayes-optimal information value",
            "Hypothetical branches are model-only diagnostics, never real queries or counterfactual outcomes",
            "Simulated costs; two historical projections of one cached source",
            "Confidence intervals omit fitting, repeated research, multiplicity and distribution-shift uncertainty",
            "Validation diagnostic only; no parameters or policies chosen by observed performance"]}
    write_json(output/"report.json",report)
    return report


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source","source-round12","bundle","earlier-bundle","output","design-note"):
        parser.add_argument("--"+name,required=True)
    args = parser.parse_args()
    result = run(args.source,args.source_round12,args.bundle,args.earlier_bundle,args.output,args.design_note)
    print(json.dumps({k:result[k] for k in ("status","test_matches","episode_runs","validation_episode_runs")}))
