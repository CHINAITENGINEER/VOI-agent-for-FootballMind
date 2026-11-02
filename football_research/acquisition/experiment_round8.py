"""Learned joint returns: coherent Bayes prediction and exact finite EVSI policy.

All primary policies use the SAME joint-return predictor. Frozen round-seven
outcome models supply free-context priors only; old policy scores form a separate
family and are not combined with this experiment's acquisition comparisons.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from copy import deepcopy
from datetime import timedelta
from hashlib import sha256
import itertools
import json
import math
from pathlib import Path
import random
import shutil
import time

import numpy as np

from football_research.baselines.api import canonical_json
from football_research.evaluation.splits import rolling_origin_splits
from .contextual_partial import ContextualPartialPredictor
from .data import ControlledEpisode,load_episodes
from .experiment import losses,outcome,state_after
from .experiment_round4 import DEFAULT_DESIGN,_heuristic_decision,_sem_time,write_json
from .experiment_round6 import _hash,_jsonl
from .joint_planner import JointRiskPlanner
from .joint_returns import JointReturnPredictor,extract_oof_contexts,entropy,ACTION_IDS

VERSION = "learned-joint-returns-exact-evsi-round8-v1"
STRATEGIES = ("base-only","all-available","entropy","fixed-order","random","evsi-one-step","evsi-two-step")


def run_episode(row,model,actions,*,strategy,budget_money,lambda_money,seed,design):
    env = ControlledEpisode(row,actions)
    rng = random.Random(int(sha256(f"round4|{seed}|{row['match_id']}".encode()).hexdigest()[:16],16))
    spent = {unit:0. for unit in ("requests","money","latency_ms","tokens")}
    prediction = initial = model.predict_partial(env.visible_state())
    events = []
    predictor_ms = estimator_ms = 0.
    planner_calls = 0
    planner = JointRiskPlanner(model,horizon=2 if strategy == "evsi-two-step" else 1,lambda_cost={"money":lambda_money})
    for step in range(len(actions)+1):
        state = {**env.visible_state(),"action_history":[{"action_id":aid,"status":"acquired_or_source_missing"} for aid in env.performed_actions]}
        started = time.perf_counter()
        prediction = model.predict_partial(state)
        predictor_ms += (time.perf_counter()-started)*1000
        candidates = env.legal_actions()
        if strategy == "base-only":
            decision = {"decision":"stop","action_id":None,"reason":"base_only"}
        elif not candidates:
            decision = {"decision":"stop","action_id":None,"reason":"no_legal_action"}
        elif strategy in {"evsi-one-step","evsi-two-step"}:
            started = time.perf_counter()
            decision = planner.plan(state,candidates,{"limits":{"money":budget_money,"requests":design["request_budget"]},"spent":spent})
            estimator_ms += (time.perf_counter()-started)*1000
            planner_calls += 1
        else:
            decision = _heuristic_decision(strategy,state,prediction,initial,candidates,
                {"money":max(0.,budget_money-spent["money"]),"requests":max(0.,design["request_budget"]-spent["requests"])},design,rng)
        event = {"step":step,"visible_state":deepcopy(state),"prediction":deepcopy(prediction),"decision":deepcopy(decision)}
        events.append(event)
        if decision["decision"] == "stop":
            break
        if decision["decision"] not in {"action","acquire"}:
            raise ValueError("Unknown real policy action")
        selected = next(a for a in candidates if a["action_id"] == decision["action_id"])
        env.reveal(selected["action_id"])
        event["charged_cost"] = deepcopy(selected["cost"])
        for unit in spent:
            spent[unit] += selected["cost"].get(unit,0.)
    return {"match_id":row["match_id"],"kickoff":row["kickoff"],"prediction_cutoff":row["prediction_cutoff"],
        "strategy":strategy,"seed":seed,"budget_money":budget_money,"lambda_money":lambda_money,"cost":spent,
        "final_probabilities":prediction["probabilities"],"initial_probabilities":initial["probabilities"],"final_mask":prediction["mask"],
        "actions":list(env.performed_actions),"stop_reason":decision["reason"],"model_version":model.model_version,
        "base_model_version":model.base_predictor.model_version,"gain_version":planner.version if strategy.startswith("evsi-") else "not_applicable",
        "predictor_latency_ms":predictor_ms,"estimator_latency_ms":estimator_ms,"estimator_calls":planner_calls,
        "actual_external_requests":0,"actual_llm_tokens":0,"compute_money":None,"compute_money_status":"not_metered_local_cpu",
        **losses(prediction["probabilities"],outcome(row))},events


def select_lambda(validation,model,actions,strategy,budget,design):
    scores = []
    for lam in design["lambda_money_grid"]:
        results = [run_episode(row,model,actions,strategy=strategy,budget_money=budget,lambda_money=lam,seed=0,design=design)[0] for row in validation]
        utility = float(np.mean([r["log_loss"]+design["validation_utility_money_weight"]*r["cost"]["money"] for r in results]))
        scores.append({"lambda_money":lam,"validation_utility":utility,"validation_matches":len(results)})
    selected = min(scores,key=lambda r:(r["validation_utility"],-r["lambda_money"]))["lambda_money"]
    return selected,{"selected":selected,"candidates":scores,"validation_match_ids":[r["match_id"] for r in validation],
        "utility_money_weight":design["validation_utility_money_weight"]}


def summarize(results,design):
    grouped = defaultdict(lambda:defaultdict(list))
    for row in results:
        grouped[(row["strategy"],row["budget_money"])][row["match_id"]].append(row)
    per_match,metrics = {},{}
    for (strategy,budget),matches in grouped.items():
        averaged = {}
        for mid,rows in matches.items():
            record = {key:float(np.mean([r[key] for r in rows])) for key in ("log_loss","brier_score","accuracy")}
            record.update({unit:float(np.mean([r["cost"][unit] for r in rows])) for unit in ("money","requests")})
            record["utility_loss"] = record["log_loss"]+design["validation_utility_money_weight"]*record["money"]
            record["kickoff"] = rows[0]["kickoff"]
            averaged[mid] = record
        per_match[(strategy,budget)] = averaged
        records = [r for rows in matches.values() for r in rows]
        metrics[f"{strategy}/money={budget:g}"] = {"independent_matches":len(averaged),"episodes":len(records),
            **{k:float(np.mean([r[k] for r in averaged.values()])) for k in ("log_loss","brier_score","accuracy","money","requests","utility_loss")},
            "stop_reasons":dict(Counter(r["stop_reason"] for r in records)),"action_counts":dict(Counter(a for r in records for a in r["actions"])),
            "aggregation":"average seeds within match, then equal-weight matches"}
    comparisons = {}
    for budget,strategy in itertools.product(design["money_budgets"],STRATEGIES):
        references = tuple(s for s in STRATEGIES if s != strategy) if strategy.startswith("evsi-") else ("base-only",)
        for reference in references:
            if reference == strategy:
                continue
            a,b = per_match[(strategy,budget)],per_match[(reference,budget)]
            if a.keys() != b.keys():
                raise ValueError("Paired test identities differ")
            blocks = defaultdict(list)
            for mid in sorted(a):
                year,week,_ = _sem_time(a[mid]["kickoff"]).isocalendar()
                blocks[(year,week)].append([a[mid][key]-b[mid][key] for key in ("log_loss","brier_score","money","utility_loss")])
            arrays = [np.asarray(v) for v in blocks.values()]
            sums,counts = np.asarray([v.sum(axis=0) for v in arrays]),np.asarray([len(v) for v in arrays])
            draws = np.random.default_rng(design["bootstrap_seed"]).integers(0,len(arrays),size=(design["bootstrap_samples"],len(arrays)))
            boot = sums[draws].sum(axis=1)/counts[draws].sum(axis=1)[:,None]
            difference = sums.sum(axis=0)/counts.sum()
            comparisons[f"{strategy}-vs-{reference}/money={budget:g}"] = {"strategy":strategy,"reference":reference,
                "independent_matches":int(counts.sum()),"weeks":len(arrays),"mean_log_loss_difference":float(difference[0]),
                "confidence_interval_95":np.quantile(boot[:,0],[.025,.975]).tolist(),"mean_brier_difference":float(difference[1]),
                "mean_money_difference":float(difference[2]),"mean_utility_loss_difference":float(difference[3]),
                "utility_confidence_interval_95":np.quantile(boot[:,3],[.025,.975]).tolist(),
                "method":"paired UTC-week percentile bootstrap; random seeds first averaged per match",
                "limitations":"previously explored historical season; no multiplicity/model-fit/adaptive-research correction"}
    return metrics,comparisons


def return_diagnostics(rows,model,actions):
    transitions,initial = [],[]
    ids = [a["action_id"] for a in actions]
    for row in rows:
        full_state = state_after(row,actions,ids)
        actual = model.observed_categories(full_state)
        initial_state = state_after(row,actions,())
        distribution = model.joint_distribution(initial_state)
        reference = model.joint_distribution(initial_state,condition_on_p0=False)
        ri = distribution["results_categories"].index(actual["results_view"])
        xi = distribution["xg_categories"].index(actual["xg_view"])
        joint,ref_joint = np.asarray(distribution["probabilities"]),np.asarray(reference["probabilities"])
        y = outcome(row)
        marginal = float(joint[:,ri,xi].sum())
        reference_marginal = float(ref_joint[:,ri,xi].sum())
        label_conditional = float(joint[y,ri,xi]/distribution["prior_probabilities"][y])
        ref_label_conditional = float(ref_joint[y,ri,xi]/distribution["prior_probabilities"][y])
        initial.append({"match_id":row["match_id"],"actual_return_categories":actual,
            "joint_return_probability":marginal,"joint_return_log_loss":-math.log(marginal),
            "unconditional_context_reference_probability":reference_marginal,"unconditional_context_reference_log_loss":-math.log(reference_marginal),
            "label_conditional_return_probability":label_conditional,"label_conditional_return_log_loss":-math.log(label_conditional),
            "reference_label_conditional_log_loss":-math.log(ref_label_conditional),
            "label_conditioning_scope":"offline diagnostic only; target result never given to runtime model",
            "q_support":distribution["q_support"]})
        for acquired in ((),(ids[0],),(ids[1],)):
            state = state_after(row,actions,acquired)
            before = model.predict_partial(state)
            for action in actions:
                if action["action_id"] in acquired:
                    continue
                enumeration = model.enumerate_action(state,action)
                token = actual[action["action_id"]]
                chosen = next(b for b in enumeration["branches"] if b["observation"] == token)
                after_state = state_after(row,actions,(*acquired,action["action_id"]))
                after = model.predict_partial(after_state)
                if max(abs(a-b) for a,b in zip(after["probabilities"],chosen["posterior_probabilities"])) > 1e-10:
                    raise ValueError("Actual posterior does not match category branch")
                reference = model.joint_distribution(state,condition_on_p0=False)
                rj = np.asarray(reference["probabilities"])
                reference_probability = float(rj[:,ri,:].sum()) if action["action_id"] == "results_view" else float(rj[:,:,xi].sum())
                realized = losses(before["probabilities"],y)["log_loss"]-losses(after["probabilities"],y)["log_loss"]
                mean_posterior = sum(b["probability"]*np.asarray(b["posterior_probabilities"]) for b in enumeration["branches"])
                transitions.append({"match_id":row["match_id"],"kickoff":row["kickoff"],"acquired_action_ids":list(acquired),"action_id":action["action_id"],
                    "actual_return_category":token,"predicted_return_probability":chosen["probability"],"return_log_loss":-math.log(chosen["probability"]),
                    "reference_return_probability":reference_probability,"reference_return_log_loss":-math.log(reference_probability),
                    "estimated_evsi":enumeration["evsi"],"conditional_mutual_information":enumeration["conditional_mutual_information"],
                    "realized_log_loss_gain":realized,"posterior_martingale_max_error":float(np.max(np.abs(mean_posterior-np.asarray(before["probabilities"])))),
                    "scope":"enumerate without true return; evaluator selects actual category only after value computation"})
    return transitions,initial


def diagnostic_summary(transitions,initial):
    bins = []
    for bucket in np.array_split(sorted(transitions,key=lambda r:r["estimated_evsi"]),5):
        rows = list(bucket)
        if rows:
            bins.append({"transitions":len(rows),"independent_matches":len({r["match_id"] for r in rows}),
                "mean_estimated_evsi":float(np.mean([r["estimated_evsi"] for r in rows])),
                "mean_realized_gain":float(np.mean([r["realized_log_loss_gain"] for r in rows]))})
    return {"initial_joint_return_matches":len(initial),"transitions":len(transitions),"independent_matches":len({r["match_id"] for r in transitions}),
        "initial_joint_return_log_loss":float(np.mean([r["joint_return_log_loss"] for r in initial])),
        "initial_unconditional_context_reference_log_loss":float(np.mean([r["unconditional_context_reference_log_loss"] for r in initial])),
        "label_conditional_return_log_loss":float(np.mean([r["label_conditional_return_log_loss"] for r in initial])),
        "reference_label_conditional_log_loss":float(np.mean([r["reference_label_conditional_log_loss"] for r in initial])),
        "transition_return_log_loss":float(np.mean([r["return_log_loss"] for r in transitions])),
        "transition_reference_return_log_loss":float(np.mean([r["reference_return_log_loss"] for r in transitions])),
        "mean_model_evsi":float(np.mean([r["estimated_evsi"] for r in transitions])),
        "mean_realized_log_loss_gain":float(np.mean([r["realized_log_loss_gain"] for r in transitions])),
        "max_martingale_error":max(r["posterior_martingale_max_error"] for r in transitions),
        "max_cmi_evsi_error":max(abs(r["estimated_evsi"]-r["conditional_mutual_information"]) for r in transitions),
        "bins":bins,"scope":"held-out descriptive model misspecification diagnostics; multiple transitions do not add independent matches"}


def run(bundle,earlier_bundle,source_round7,output):
    source,output = Path(source_round7),Path(output)
    output.mkdir(parents=True,exist_ok=False)
    design = {**DEFAULT_DESIGN,"experiment_version":VERSION,"classification":"exploratory_learned_joint_returns_model_conditional_evsi",
        "strategies":STRATEGIES,"joint_model":{"bandwidth":.2,"dirichlet_mass":5.},
        "return_categories":{"results":"3 Elo-difference quantiles x 3 recent-point-difference quantiles + missing","xg":"3 net-xg-difference quantiles + missing"},
        "quantile_fitting":"training eligible OOF rows only","source_round7":str(source.resolve()),
        "primary_family":"same fitted joint Bayesian predictor for every policy","round7_comparison":"separate predictor family only",
        "source_usage":"frozen free-context outcome predictor and past OOF empty priors, never old OOF gain targets",
        "parameter_selection":"kernel and compression fixed before test; lambda selected on temporal validation",
        "planner_optimality":"one/two remaining acquisitions in fitted finite joint model, branchwise STOP always available"}
    write_json(output/"design.json",design)
    earlier,earlier_audit = load_episodes(earlier_bundle,lead_hours=24)
    current,current_audit = load_episodes(bundle,lead_hours=24)
    episodes = earlier+current
    write_json(output/"data-audit.json",{"current":current_audit,"earlier":earlier_audit,"input_matches":len(episodes),
        "paid_group_feature_fields":"unchanged frozen historical projections","new_external_requests":0})
    actions = json.loads((source/"catalog.json").read_text(encoding="utf-8"))["actions"]
    shutil.copy2(source/"catalog.json",output/"catalog.json")
    source_report = json.loads((source/"report.json").read_text(encoding="utf-8"))
    contexts = extract_oof_contexts(_jsonl(source/"oof-supervision.jsonl"))
    (output/"training-oof-contexts.jsonl").write_text("".join(canonical_json(r)+"\n" for _,r in sorted(contexts.items())),encoding="utf-8")
    source_artifacts = {name:_hash(source/name) for name in ("report.json","catalog.json","oof-supervision.jsonl","oof-manifest.json","episodes.jsonl")}
    write_json(output/"source-artifacts.json",source_artifacts)
    names = ("joint_returns.py","joint_planner.py","experiment_round8.py","contextual_partial.py","contextual_identity.py","experiment_round4.py","experiment.py","data.py","state.py","value.py")
    implementation = {name:_hash(Path(__file__).parent/name) for name in names}
    write_json(output/"implementation.json",implementation)
    folds = rolling_origin_splits(episodes,initial_validation_start=design["validation_start"],validation_window=timedelta(days=30),test_window=timedelta(days=30))
    results,diagnostic_rows,initial_diagnostics,fold_reports = [],[],[],[]
    with (output/"trajectories.jsonl").open("w",encoding="utf-8") as traces:
        for number,fold in enumerate(folds):
            prior = next((f for f in source_report["folds"] if f["fold"] == number),None)
            if prior is None:
                continue
            for key,rows in (("training_match_ids",fold.train),("validation_match_ids",fold.validation),("test_match_ids",fold.test)):
                if set(prior[key]) != {r["match_id"] for r in rows}:
                    raise ValueError("Frozen round-seven split differs")
            print(f"round8 fold {number}: fitting joint training={len(fold.train)}, test={len(fold.test)}",flush=True)
            base_path = output/"models"/f"fold-{number}-base.json"
            base_path.parent.mkdir(exist_ok=True)
            shutil.copy2(source/"models"/f"fold-{number}-predictor.json",base_path)
            base = ContextualPartialPredictor.load(base_path)
            model = JointReturnPredictor(base,**design["joint_model"]).fit(fold.train,contexts,as_of=fold.train_cutoff,
                dataset_version=episodes[0]["dataset_version"],actions=actions)
            joint_path = output/"models"/f"fold-{number}-joint.json"
            model.save(joint_path)
            model = JointReturnPredictor.load(joint_path)
            selected = {}
            for strategy,budget in itertools.product(("evsi-one-step","evsi-two-step"),design["money_budgets"]):
                selected[(strategy,budget)] = select_lambda(fold.validation,model,actions,strategy,budget,design)
            fold_report = {"fold":number,"split":fold.manifest(),"training_match_ids":prior["training_match_ids"],
                "validation_match_ids":prior["validation_match_ids"],"test_match_ids":prior["test_match_ids"],
                "joint_training_match_ids":model.metadata["training_match_ids"],
                "predictor":{"path":str(base_path.relative_to(output)),"sha256":_hash(base_path),"version":base.model_version,"metadata":base.metadata},
                "joint_model":{"path":str(joint_path.relative_to(output)),"sha256":_hash(joint_path),"version":model.model_version,"metadata":model.metadata},
                "lambda_selection":{f"{s}/{b:g}":record for (s,b),(_,record) in selected.items()}}
            for row in fold.test:
                for strategy,budget in itertools.product(STRATEGIES,design["money_budgets"]):
                    lam = selected[(strategy,budget)][0] if strategy.startswith("evsi-") else 0.
                    for seed in design["random_seeds"] if strategy == "random" else [0]:
                        result,trace = run_episode(row,model,actions,strategy=strategy,budget_money=budget,lambda_money=lam,seed=seed,design=design)
                        result["fold"] = number
                        results.append(result)
                        traces.write(canonical_json({"fold":number,"match_id":row["match_id"],"strategy":strategy,"budget_money":budget,"seed":seed,"events":trace})+"\n")
            transition,initial = return_diagnostics(fold.test,model,actions)
            diagnostic_rows.extend({**r,"fold":number} for r in transition)
            initial_diagnostics.extend({**r,"fold":number} for r in initial)
            fold_report["return_diagnostics"] = diagnostic_summary(transition,initial)
            fold_reports.append(fold_report)
            write_json(output/"fold-progress.json",fold_reports)
            print(f"round8 fold {number}: completed",flush=True)
    metrics,comparisons = summarize(results,design)
    previous_base = {(r["match_id"],r["budget_money"]):r["final_probabilities"] for r in _jsonl(source/"episodes.jsonl") if r["strategy"] == "base-only"}
    base_matches = [r for r in results if r["strategy"] == "base-only"]
    if any(r["final_probabilities"] != previous_base[(r["match_id"],r["budget_money"])] for r in base_matches):
        raise ValueError("Empty observation predictions changed from frozen round-seven base")
    report = {"status":"OK","experiment_version":VERSION,"evidence_grade":design["classification"],"input_matches":len(episodes),
        "test_matches":len({r["match_id"] for r in results}),"episode_runs":len(results),"folds":fold_reports,"metrics":metrics,
        "paired_comparisons":comparisons,"return_diagnostics":diagnostic_summary(diagnostic_rows,initial_diagnostics),
        "model_artifacts":[{"path":str(p.relative_to(output)),"sha256":_hash(p)} for p in sorted((output/"models").glob("*.json"))],
        "source_artifacts_sha256":source_artifacts,"implementation_sha256":implementation,"external_requests":0,"llm_tokens":0,
        "base_only_exact_round7_parity":{"rows":len(base_matches),"exact_equal":True},"cost_interpretation":design["cost_interpretation"],
        "limitations":["Same already explored 290-match season; not independent confirmation",
            "Finite lossy numeric summaries, not open-ended live news/lineup tools","q conditional on p0 alone assumes contextual sufficiency without proof",
            "OOF p0 comes from earlier model fits; deployment p0 model population can differ","Kernel/smoothing misspecification and parameter uncertainty not included in exact Bellman risk",
            "Mathematical CMI=EVSI and nonnegative expected information value hold only inside fitted joint model",
            "Positive fitted EVSI does not guarantee positive held-out realized gain or useful real-world tool acquisition",
            "Missing summary is distinct from unmodeled provider failure; failure states rejected",
            "Two tools share the frozen source response; simulated projection costs are not provider bills",
            "Crosswalk identity and historical values reconstructed offline; no true operational timestamp guarantee",
            "No novel-theory claim for CMI or nongreedy acquisition; contribution is auditable learned-joint mechanism and mismatch diagnostics"]}
    write_json(output/"report.json",report)
    for name,rows in (("episodes",results),("return-diagnostics",diagnostic_rows),("initial-return-diagnostics",initial_diagnostics)):
        (output/(name+".jsonl")).write_text("".join(canonical_json(r)+"\n" for r in rows),encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("bundle","earlier-bundle","source-round7","output"):
        parser.add_argument("--"+argument,required=True)
    args = parser.parse_args()
    report = run(args.bundle,args.earlier_bundle,args.source_round7,args.output)
    print(json.dumps({k:report[k] for k in ("status","input_matches","test_matches","episode_runs")}))
