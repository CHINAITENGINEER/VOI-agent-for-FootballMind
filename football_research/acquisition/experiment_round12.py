"""Frozen same-version gains: action-only and acquired-set mean controls."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import itertools
import json
from pathlib import Path
import shutil

from .analyze_round7 import match_means, paired
from .data import load_episodes
from .experiment_round4 import write_json
from .experiment_round9 import run_episode as old_run_episode
from .experiment_round11 import digest, write_lines, summarize as reference_summarize
from .gain_prior import GainPriorEstimator
from .joint_returns import JointReturnPredictor
from .state import parse_time
from .verify_round4 import _rows
from football_research.baselines.api import canonical_json

VERSION = "state-erasure-acquisition-ablation-round12-v1"
NEW_STRATEGIES = {"tool-prior-mean": "action_only", "stage-prior-mean": "acquired_set"}
DESIGN_SHA256 = "884ac4fe147f006855acbd959b21e824d78e7169c001f1f70a41759dfe2cf8a8"


def run_episode(row, model, actions, *, strategy, budget_money, seed, design, prior):
    if strategy not in NEW_STRATEGIES or prior.mode != NEW_STRATEGIES[strategy]:
        raise ValueError("Registered strategy and matching erasure mode required")
    result, trace = old_run_episode(row, model, actions, strategy="calibrated-mean", budget_money=budget_money,
        lambda_money=design["fixed_lambda_money"], seed=seed, design=design, calibrator=prior)
    result.update(strategy=strategy, estimator_kind=prior.manifest["estimator_kind"], erasure_mode=prior.mode)
    return result, trace


def summarize(episodes, design):
    metrics, _ = reference_summarize(episodes, design)
    means = match_means(episodes, design["validation_utility_money_weight"])
    comparisons = {}
    pairs = [("frozen-version-mean", "stage-prior-mean"), ("frozen-version-mean", "tool-prior-mean"),
        ("stage-prior-mean", "tool-prior-mean")]
    pairs += [(strategy, ref) for strategy in NEW_STRATEGIES for ref in
        ("base-only", "all-available", "entropy", "random", "evsi-one-step", "matched-mixed-mean")]
    for a,b in pairs:
        for budget in design["money_budgets"]:
            comparisons[f"{a}-vs-{b}/money={budget:g}"] = {metric: paired(means[a,budget], means[b,budget], metric,
                design["bootstrap_samples"], design["bootstrap_seed"]) for metric in ("log_loss", "money", "utility_loss")}
    behavior = {}
    for strategy in (*NEW_STRATEGIES, "frozen-version-mean"):
        for budget in design["money_budgets"]:
            raw = [r for r in episodes if r["strategy"] == strategy and r["budget_money"] == budget]
            for fold in sorted({r["fold"] for r in raw}):
                subset = [r for r in raw if r["fold"] == fold]
                behavior[f"{strategy}/money={budget:g}/fold={fold}"] = {
                    "matches": len(subset), "query_count_histogram": dict(Counter(len(r["actions"]) for r in subset)),
                    "first_actions": dict(Counter(r["actions"][0] if r["actions"] else "STOP" for r in subset)),
                    "paths": dict(Counter(" -> ".join([*r["actions"], "STOP"]) for r in subset))}
    return metrics, comparisons, behavior


def run(source, bundle, earlier_bundle, output, design_note):
    source, output, design_note = Path(source), Path(output), Path(design_note)
    if output.exists():
        raise FileExistsError(output)
    if digest(design_note) != DESIGN_SHA256:
        raise ValueError("Predeclared round-twelve design changed")
    previous = json.loads((source/"report.json").read_text(encoding="utf-8"))
    old_design = json.loads((source/"design.json").read_text(encoding="utf-8"))
    if previous["status"] != "OK":
        raise ValueError("Completed frozen round-eleven source required")
    for name,h in previous["implementation_sha256"].items():
        if digest(Path(__file__).with_name(name)) != h:
            raise ValueError("Frozen implementation changed: "+name)
    for item in previous["model_artifacts"]+previous["supervision_artifacts"]:
        if digest(source/item["path"]) != item["sha256"]:
            raise ValueError("Frozen source artifact changed: "+item["path"])
    output.mkdir(parents=True)
    design = {**old_design, "experiment_version": VERSION,
        "classification": "exploratory_state_erasure_ablation_on_reused_history", "source_round11": str(source.resolve()),
        "new_strategies": NEW_STRATEGIES, "reference_strategies": old_design["strategies"],
        "strategies": [*old_design["strategies"], *NEW_STRATEGIES], "fixed_lambda_money": .5,
        "parameter_selection": "none; unique match minimum support 12; diagnostic week SE only",
        "source_usage": "new prior models and new policy episodes only; immutable round-eleven references read for paired comparison",
        "primary_comparison": "frozen-version-mean minus stage-prior-mean utility loss at budget .05",
        "erasure_controls": "action only uniform two-state historical average; action plus acquired-set historical average",
        "design_note_path": str(design_note.resolve()), "design_note_sha256": digest(design_note)}
    for key in ("arm_matching", "scope", "predictor_policy", "calibration_inner_split"):
        design.pop(key, None)
    write_json(output/"design.json", design)
    current, ca = load_episodes(bundle, lead_hours=24)
    earlier, ea = load_episodes(earlier_bundle, lead_hours=24)
    lookup = {r["match_id"]:r for r in earlier+current}
    old_audit = json.loads((source/"data-audit.json").read_text(encoding="utf-8"))
    for name,audit in (("current",ca),("earlier",ea)):
        if audit["platform_manifest_sha256"] != old_audit[name]["platform_manifest_sha256"]:
            raise ValueError("Source data snapshot changed")
    write_json(output/"data-audit.json", {"current":ca,"earlier":ea,"input_matches":len(lookup)})
    shutil.copy2(source/"catalog.json", output/"catalog.json")
    actions = json.loads((output/"catalog.json").read_text(encoding="utf-8"))["actions"]
    references = list(_rows(source/"episodes.jsonl"))
    validation_references = list(_rows(source/"validation-episodes.jsonl"))
    source_names = ("report.json", "design.json", "catalog.json", "data-audit.json", "episodes.jsonl", "trajectories.jsonl",
        "validation-episodes.jsonl", "validation-trajectories.jsonl", "transition-diagnostics.jsonl", "implementation.json")
    source_hashes = {name:digest(source/name) for name in source_names}
    write_json(output/"source-artifacts.json", source_hashes)
    names = (*previous["implementation_sha256"], "gain_prior.py", "experiment_round12.py")
    implementation = {name:digest(Path(__file__).with_name(name)) for name in names}
    write_json(output/"implementation.json", implementation)
    (output/"models").mkdir()
    episodes, validation_episodes, folds = [], [], []
    with (output/"trajectories.jsonl").open("w",encoding="utf-8") as trace_file, (output/"validation-trajectories.jsonl").open("w",encoding="utf-8") as val_trace_file:
        for target in previous["folds"]:
            number = target["fold"]
            joint_path = output/"models"/f"fold-{number}-joint.json"
            shutil.copy2(source/target["joint_model"]["path"], joint_path)
            model, joint_hash = JointReturnPredictor.load(joint_path), digest(joint_path)
            sample_info = target["supervision"]["same_version"]
            samples = list(_rows(source/sample_info["path"]))
            if digest(source/sample_info["path"]) != sample_info["sha256"]:
                raise ValueError("Same-version supervision changed")
            priors, artifacts = {}, {}
            for mode in NEW_STRATEGIES.values():
                prior = GainPriorEstimator(mode=mode).fit(samples, as_of=target["split"]["train_cutoff"],
                    dataset_version=target["calibrators"]["same_version"]["manifest"]["dataset_version"],
                    joint_model=model, joint_artifact_sha256=joint_hash,
                    metadata={"target_fold":number,"reference_fold":target["reference_fold"],
                        "source_supervision_path":str((source/sample_info["path"]).resolve()),"source_supervision_sha256":sample_info["sha256"]})
                if set(prior.manifest["training_match_ids"]) != set(target["calibration_match_ids"]):
                    raise ValueError("Prior and conditional calibration IDs differ")
                path = output/"models"/f"fold-{number}-{mode}-prior.json"
                prior.save(path)
                priors[mode] = GainPriorEstimator.load(path, model=model, model_sha256=joint_hash)
                artifacts[mode] = {"path":str(path.relative_to(output)),"sha256":digest(path),"version":prior.version,"manifest":prior.manifest}
            for partition, results, stream in (("validation", validation_episodes, val_trace_file), ("test", episodes, trace_file)):
                for mid in target[partition+"_match_ids"]:
                    row = lookup[mid]
                    if partition == "validation" and parse_time(row["result_available_at"]) > parse_time(target["split"]["validation_label_cutoff"]):
                        raise ValueError("Unavailable validation diagnostic outcome")
                    for strategy,budget in itertools.product(NEW_STRATEGIES, design["money_budgets"]):
                        prior = priors[NEW_STRATEGIES[strategy]]
                        result,events = run_episode(row, model, actions, strategy=strategy, budget_money=budget, seed=0, design=design, prior=prior)
                        result.update(fold=number,reference_fold=target["reference_fold"],partition=partition)
                        results.append(result)
                        stream.write(canonical_json({"fold":number,"reference_fold":target["reference_fold"],"partition":partition,
                            "match_id":mid,"strategy":strategy,"budget_money":budget,"seed":0,"events":events})+"\n")
            fold = {key:deepcopy(target[key]) for key in ("fold","reference_fold","split","training_match_ids","calibration_match_ids","validation_match_ids","test_match_ids")}
            fold.update(joint_model={"path":str(joint_path.relative_to(output)),"sha256":joint_hash,"version":model.model_version,"metadata":model.metadata},
                gain_priors=artifacts,source_same_version_supervision={**sample_info,"path":str((source/sample_info["path"]).resolve())},
                source_conditional_calibrator={**target["calibrators"]["same_version"],"path":str((source/target["calibrators"]["same_version"]["path"]).resolve())},
                fixed_lambda_money=.5)
            folds.append(fold)
            write_json(output/"fold-progress.json", folds)
            print(f"round12 fold {number}: {len(target['calibration_match_ids'])} calibration matches; two policies complete", flush=True)
    write_lines(output/"episodes.jsonl", episodes)
    write_lines(output/"validation-episodes.jsonl", validation_episodes)
    metrics, comparisons, behavior = summarize(references+episodes, design)
    val_metrics, _, val_behavior = summarize(validation_references+validation_episodes, design)
    report = {"status":"OK","experiment_version":VERSION,"source_round11":str(source.resolve()),"evidence_grade":design["classification"],
        "input_matches":len(lookup),"test_matches":len({r['match_id'] for r in episodes}),"episode_runs":len(episodes),
        "new_policy_episode_runs":len(episodes),"reference_episode_runs":len(references),"combined_episode_runs":len(references)+len(episodes),
        "validation_episode_runs":len(validation_episodes),"reference_validation_episode_runs":len(validation_references),
        "validation_unique_matches":len({r['match_id'] for r in validation_episodes}),"folds":folds,
        "metrics":metrics,"paired_comparisons":comparisons,"policy_behavior_by_fold":behavior,
        "primary_comparison":comparisons["frozen-version-mean-vs-stage-prior-mean/money=0.05"],
        "validation_metrics_diagnostic_only":val_metrics,"validation_behavior_diagnostic_only":val_behavior,
        "model_artifacts":[{"path":str(p.relative_to(output)),"sha256":digest(p)} for p in sorted((output/"models").glob("*.json"))],
        "implementation_sha256":implementation,"source_artifacts_sha256":source_hashes,
        "new_outcome_predictors":0,"new_gain_priors":len(folds)*2,"new_independent_test_matches":0,"external_requests":0,"llm_tokens":0,
        "fixed_lambda_money":.5,"limitations":["Exploratory reused 200 games; no independent confirmation",
            "Action-only averages two recorded states uniformly, not a deployment occupancy model",
            "State-conditioned versus erased estimators differ in bias and variance; not an isolated causal conditioning effect",
            "Simulated costs and reconstructed historical two-view returns; no live tool or LLM experiment",
            "Week intervals exclude fitting, repeated research, multiplicity and temporal nonstationarity uncertainty",
            "Validation only describes frozen policies; no parameters or winner selected from its results"]}
    write_json(output/"report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source","bundle","earlier-bundle","output","design-note"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args()
    result = run(args.source,args.bundle,args.earlier_bundle,args.output,args.design_note)
    print(json.dumps({k:result[k] for k in ("status","test_matches","episode_runs","validation_episode_runs")}))
