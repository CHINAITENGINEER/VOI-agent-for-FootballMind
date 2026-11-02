"""Frozen-alpha contrast and exact decision-boundary accounting (exploratory)."""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .analyze_round7 import match_means, paired
from .data import load_episodes
from .decision_boundary import MARGIN_EDGES, TERMS, decompose_query, summarize_records, sum_episode
from .experiment_round9 import run_episode
from .joint_returns import JointReturnPredictor
from .round9_residual import ResidualEVSICalibrator
from .state import parse_time
from .verify_round4 import _rows


VERSION = "frozen-alpha-boundary-round10-v1"
FIXED = {"fixed-alpha-mean":"calibrated-mean","fixed-alpha-lcb":"calibrated-lcb"}
REFERENCES = ("base-only","calibrated-mean","calibrated-lcb")


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def write_lines(path, records):
    with Path(path).open("w",encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n")


def episode_key(row):
    return row["fold"],row["match_id"],row["strategy"],row["budget_money"],row["seed"]


def run(source, bundle, earlier_bundle, output):
    source,output = Path(source),Path(output)
    if output.exists():
        raise FileExistsError(output)
    prior = json.loads((source/"report.json").read_bytes())
    frozen = json.loads((source/"design.json").read_bytes())
    if prior["status"] != "OK":
        raise ValueError("Finished frozen round9 source required")
    for name,h in prior["implementation_sha256"].items():
        if digest(Path(__file__).with_name(name)) != h:
            raise ValueError("Frozen source changed: "+name)
    for artifact in prior["model_artifacts"]:
        if digest(source/artifact["path"]) != artifact["sha256"]:
            raise ValueError("Frozen model artifact changed")
    output.mkdir(parents=True)
    design_path = Path(__file__).resolve().parents[2]/"docs/research/acquisition/round10-design.md"
    design = {"version":VERSION,"classification":"exploratory_diagnostic_reused_history_no_refitting",
        "source_round9":str(source.resolve()),"source_report_sha256":digest(source/"report.json"),
        "frozen_alpha":frozen["validation_utility_money_weight"],"fixed_strategy_mapping":FIXED,
        "money_budgets":frozen["money_budgets"],"request_budget":frozen["request_budget"],
        "validation_utility_money_weight":frozen["validation_utility_money_weight"],
        "bootstrap_samples":frozen["bootstrap_samples"],"bootstrap_seed":frozen["bootstrap_seed"],
        "margin_edges":list(MARGIN_EDGES),"margin_side":"right","new_fitted_parameters":0,
        "new_independent_test_matches":0,"design_note_sha256":digest(design_path),
        "contrast":"Only replace validation-selected execution lambda with existing frozen reporting alpha",
        "scope":"Accounting contrast, not proof alpha is optimal or policy is prospectively validated"}
    write_json(output/"design.json",design)
    actions = json.loads((source/"catalog.json").read_bytes())["actions"]
    current,ca = load_episodes(bundle,lead_hours=frozen["lead_hours"])
    earlier,ea = load_episodes(earlier_bundle,lead_hours=frozen["lead_hours"])
    previous_audit = json.loads((source/"data-audit.json").read_bytes())
    for part,audit in (("current",ca),("earlier",ea)):
        if audit["platform_manifest_sha256"] != previous_audit[part]["platform_manifest_sha256"]:
            raise ValueError("Frozen data export changed")
    lookup = {r["match_id"]:r for r in earlier+current}
    episodes,validation,folds = [],[],[]
    with (output/"trajectories.jsonl").open("w",encoding="utf-8") as traces:
        for fold in prior["folds"]:
            number = fold["fold"]
            model = JointReturnPredictor.load(source/fold["joint_model"]["path"])
            calibrator = ResidualEVSICalibrator.load(source/fold["calibrator"]["path"],model=model,
                model_sha256=fold["joint_model"]["sha256"])
            latest_label=max(parse_time(lookup[mid]["result_available_at"]) for mid in fold["training_match_ids"])
            comparisons = {}
            for strategy,reference_strategy in FIXED.items():
                for budget in design["money_budgets"]:
                    validation_records = []
                    for partition,target in (("validation",validation),("test",episodes)):
                        for mid in fold[partition+"_match_ids"]:
                            row = lookup[mid]
                            if (parse_time(model.metadata["training_as_of"]) > parse_time(row["prediction_cutoff"])
                                    or latest_label >= parse_time(row["prediction_cutoff"])):
                                raise ValueError("Frozen model does not precede prediction")
                            result,events = run_episode(row,model,actions,strategy=reference_strategy,budget_money=budget,
                                lambda_money=design["frozen_alpha"],seed=0,design=frozen,calibrator=calibrator)
                            result.update(fold=number,strategy=strategy,reference_strategy=reference_strategy,partition=partition)
                            target.append(result)
                            if partition == "validation":
                                validation_records.append(result)
                            else:
                                trace={k:result[k] for k in ("fold","match_id","strategy","budget_money","seed")}
                                trace["events"] = events
                                traces.write(json.dumps(trace,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n")
                    selection=fold["lambda_selection"][f"{reference_strategy}/{budget:g}"]
                    alpha_record=next(c for c in selection["candidates"] if c["lambda_money"]==design["frozen_alpha"])
                    utility=float(np.mean([r["log_loss"]+design["frozen_alpha"]*r["cost"]["money"] for r in validation_records]))
                    if abs(utility-alpha_record["validation_utility"])>1e-10:
                        raise ValueError("Fixed-alpha validation differs from frozen lambda candidate")
                    best=next(c for c in selection["candidates"] if c["lambda_money"]==selection["selected"])
                    comparisons[f"{strategy}/{budget:g}"]={"validation_matches":len(validation_records),
                        "selected_lambda":selection["selected"],"fixed_alpha":design["frozen_alpha"],
                        "selected_validation_utility":best["validation_utility"],"fixed_validation_utility":utility,
                        "fixed_minus_selected_validation_utility":utility-best["validation_utility"]}
            folds.append({"fold":number,"test_match_ids":fold["test_match_ids"],"validation_match_ids":fold["validation_match_ids"],
                "joint_model":{k:fold["joint_model"][k] for k in ("path","sha256","version")},
                "calibrator":{k:fold["calibrator"][k] for k in ("path","sha256","version")},"comparisons":comparisons})
            print(f"round10 fixed-alpha completed fold {number}",flush=True)
    write_lines(output/"episodes.jsonl",episodes)
    write_lines(output/"validation-episodes.jsonl",validation)
    write_json(output/"folds.json",folds)
    filenames=("report.json","design.json","episodes.jsonl","trajectories.jsonl","update-gain-diagnostics.jsonl","catalog.json")
    source_hashes={name:digest(source/name) for name in filenames}
    write_json(output/"source-artifacts.json",source_hashes)
    implementation={name:digest(Path(__file__).with_name(name)) for name in
        ("experiment_round10.py","decision_boundary.py","experiment_round9.py","round9_residual.py","analyze_round7.py")}
    write_json(output/"implementation.json",implementation)
    report={"status":"OK","version":VERSION,"test_matches":len({r["match_id"] for r in episodes}),
        "test_episode_runs":len(episodes),"validation_episode_runs":len(validation),"new_fitted_models":0,
        "new_independent_test_matches":0,"external_requests":0,"llm_tokens":0,"source_hashes":source_hashes,
        "implementation_sha256":implementation,"frozen_alpha":design["frozen_alpha"],
        "source_round9":str(source.resolve()),"limitations":["Already explored 290 historical matches",
            "No new data or fitting; only the prespecified cost-weight contrast", "Simulated costs and frozen historical returns",
            "Validation superiority need not persist on test; contrast does not identify a causal mechanism"]}
    write_json(output/"report.json",report)
    return report


def analyze(run_path, output):
    run_path,output=Path(run_path),Path(output)
    if output.exists():
        raise FileExistsError(output)
    design=json.loads((run_path/"design.json").read_bytes())
    source=Path(design["source_round9"])
    alpha=design["frozen_alpha"]
    new=list(_rows(run_path/"episodes.jsonl"))
    references=[r for r in _rows(source/"episodes.jsonl") if r["strategy"] in REFERENCES]
    episodes=new+references
    indexed={episode_key(r):r for r in episodes}
    if len(indexed)!=len(episodes):
        raise ValueError("Duplicate comparison episode")
    means=match_means(episodes,alpha)
    metrics={f"{strategy}/money={budget:g}":{"independent_matches":len(records),
        **{k:float(np.mean([r[k] for r in records.values()])) for k in ("log_loss","brier_score","accuracy","money","requests","utility_loss")}}
        for (strategy,budget),records in sorted(means.items())}
    comparisons={}
    for strategy in FIXED:
        for budget in design["money_budgets"]:
            for reference in REFERENCES:
                comparisons[f"{strategy}-vs-{reference}/money={budget:g}"]={metric:paired(means[strategy,budget],means[reference,budget],metric,
                    design["bootstrap_samples"],design["bootstrap_seed"]) for metric in ("log_loss","money","utility_loss")}
    diagnostics={(r["fold"],r["match_id"],tuple(r["acquired_action_ids"]),r["action_id"]):r
                 for r in _rows(source/"update-gain-diagnostics.jsonl")}
    actions={a["action_id"]:a for a in json.loads((source/"catalog.json").read_bytes())["actions"]}
    initial,selected,episode_terms=[],[],[]
    seen=set()
    for path in (source/"trajectories.jsonl",run_path/"trajectories.jsonl"):
        for trace in _rows(path):
            if trace["strategy"] not in (*FIXED,"calibrated-mean","calibrated-lcb"):
                continue
            identity=episode_key(trace)
            if identity in seen:
                raise ValueError("Duplicate diagnostic trajectory")
            seen.add(identity)
            episode=indexed[identity]
            queries=[]
            for event in trace["events"]:
                decision=event["decision"]
                for candidate in decision["candidates"]:
                    if not candidate.get("feasible"):
                        continue
                    aid=candidate["action_id"]
                    acquired=tuple(candidate["acquired_action_ids"])
                    record=diagnostics[trace["fold"],trace["match_id"],acquired,aid]
                    if abs(record["expected_gain"]-candidate["expected_gain"])>1e-12:
                        raise ValueError("Diagnostic and policy estimate differ")
                    cost=actions[aid]["cost"]["money"]
                    terms=decompose_query(gain=record["gain"],mean_estimate=candidate["expected_gain"],
                        decision_estimate=candidate["decision_gain"],cost=cost,policy_lambda=decision["lambda_cost"]["money"],utility_weight=alpha)
                    item={"fold":trace["fold"],"match_id":trace["match_id"],"strategy":trace["strategy"],"budget_money":trace["budget_money"],
                        "kickoff":episode["kickoff"],"step":event["step"],"action_id":aid,"acquired_action_ids":list(acquired),
                        "mean_estimate":candidate["expected_gain"],"decision_estimate":candidate["decision_gain"],"model_evsi":candidate["model_evsi"],
                        "policy_lambda":decision["lambda_cost"]["money"],"cost":cost,"support":candidate["support"],
                        "parent_support":candidate["parent_support"],"support_status":candidate["support_status"],
                        "selected":decision["decision"]=="acquire" and decision["action_id"]==aid,**terms}
                    if event["step"]==0:
                        initial.append(item)
                    if item["selected"]:
                        if cost!=event["charged_cost"]["money"]:
                            raise ValueError("Selected query charge differs")
                        queries.append(item)
                        selected.append(item)
            base=means["base-only",episode["budget_money"]][episode["match_id"]]
            summary=sum_episode(queries,initial_loss=base["log_loss"],final_loss=episode["log_loss"],
                cost=episode["cost"]["money"],utility_weight=alpha)
            episode_terms.append({"fold":episode["fold"],"match_id":episode["match_id"],"kickoff":episode["kickoff"],
                "strategy":episode["strategy"],"budget_money":episode["budget_money"],**summary})
    expected={episode_key(r) for r in episodes if r["strategy"]!="base-only"}
    if seen!=expected:
        raise ValueError("Diagnostic trajectory coverage differs")
    group_records=defaultdict(list)
    for population,records in (("initial_affordable_candidates",initial),("selected_queries",selected)):
        for row in records:
            prefix=f"{population}/{row['strategy']}/money={row['budget_money']:g}"
            for suffix in ("all",f"margin={row['margin_bucket']}",f"action={row['action_id']}",f"fold={row['fold']}"):
                group_records[prefix+"/"+suffix].append(row)
    groups={k:summarize_records(v) for k,v in sorted(group_records.items())}
    ep_groups=defaultdict(list)
    for row in episode_terms:
        ep_groups[f"{row['strategy']}/money={row['budget_money']:g}"].append(row)
    accounting={}
    for key,records in sorted(ep_groups.items()):
        summary={"matches":len(records),"queries":sum(r["queries"] for r in records),
            "initial_stops":sum(r["queries"]==0 for r in records),
            **{field:float(np.mean([r[field] for r in records])) for field in (*TERMS,"actual_gain","actual_utility_gain")}}
        # Every strategy has one trajectory per match: report contributions per
        # match, including the zero contribution of stopped matches.
        strategy,budget=key.split("/money=")
        utility_difference=metrics[f"base-only/money={budget}"]["utility_loss"]-metrics[key]["utility_loss"]
        if abs(summary["actual_utility_gain"]-utility_difference)>1e-10:
            raise ValueError("Mean accounting differs from reported policy utility")
        accounting[key]=summary
    folds=json.loads((run_path/"folds.json").read_bytes())
    comparisons_by_fold=[]
    for fold in folds:
        for key,validation in fold["comparisons"].items():
            strategy,budget=key.split("/");budget=float(budget)
            reference=FIXED[strategy]
            a=[r for r in episodes if r["fold"]==fold["fold"] and r["strategy"]==strategy and r["budget_money"]==budget]
            b=[r for r in episodes if r["fold"]==fold["fold"] and r["strategy"]==reference and r["budget_money"]==budget]
            difference=float(np.mean([r["log_loss"]+alpha*r["cost"]["money"] for r in a])-np.mean([r["log_loss"]+alpha*r["cost"]["money"] for r in b]))
            comparisons_by_fold.append({"fold":fold["fold"],"strategy":strategy,"budget_money":budget,
                **validation,"test_matches":len(a),"fixed_minus_selected_test_utility":difference})
    result={"status":"ANALYZED_EXPLORATORY","metrics":metrics,"paired_comparisons":comparisons,
        "episode_accounting":accounting,"boundary_groups":groups,"validation_test_contrasts":comparisons_by_fold,
        "initial_candidate_records":initial,"selected_query_records":selected,"episode_accounting_records":episode_terms,
        "margin_edges":list(MARGIN_EDGES),"utility_weight":alpha,"telescoping_and_accounting_verified":True,
        "source_hashes":{str(path.resolve()):digest(path) for path in (run_path/"design.json",run_path/"episodes.jsonl",run_path/"trajectories.jsonl",
            source/"episodes.jsonl",source/"trajectories.jsonl",source/"update-gain-diagnostics.jsonl")},
        "analyzer_sha256":digest(__file__),"accounting_sha256":digest(Path(__file__).with_name("decision_boundary.py")),
        "limitations":["Reused historical test matches; descriptive comparisons, no independent confirmation",
            "Decomposition is algebraic and not a causal explanation","One realized gain is not conditional expected information value",
            "Lambda tuning may compensate for model misspecification; lambda-alpha difference is not a software defect",
            "No new fitted parameters, future source data or deployed policy", "Repeated budgets/transitions are dependent"]}
    write_json(output,result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest="command",required=True)
    runner=subs.add_parser("run")
    for name in ("source","bundle","earlier-bundle","output"):
        runner.add_argument("--"+name,required=True)
    analysis=subs.add_parser("analyze")
    analysis.add_argument("--run",required=True)
    analysis.add_argument("--output",required=True)
    args=parser.parse_args()
    if args.command=="run":
        result=run(args.source,args.bundle,args.earlier_bundle,args.output)
        print(json.dumps(result,ensure_ascii=True))
    else:
        result=analyze(args.run,args.output)
        print(json.dumps({"status":result["status"],"selected_queries":len(result["selected_query_records"])}))
