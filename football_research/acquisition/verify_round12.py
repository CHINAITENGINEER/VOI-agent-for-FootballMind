"""Independent state-erased gain means, public-state decisions and cost replay."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path

from .data import ControlledEpisode, load_episodes
from .experiment import losses, outcome
from .joint_returns import JointReturnPredictor
from .state import parse_time
from .verify_round4 import _rows
from .verify_round6 import KEYS, verify_aggregate_metrics
from .verify_round7 import verify_episode_coverage
from .verify_round8 import close, require
from .verify_round9 import IndependentEvaluator

DESIGN_HASH = "884ac4fe147f006855acbd959b21e824d78e7169c001f1f70a41759dfe2cf8a8"
SOURCE_REPORT_HASH = "63a157974e7e4c5f20ee6ad1a779041f11607b1c3c6a1dcb2c4c8267f2df8592"
POLICIES = {"tool-prior-mean":"action_only", "stage-prior-mean":"acquired_set"}
LEGAL = {((),"results_view"),((),"xg_view"),(("results_view",),"xg_view"),(("xg_view",),"results_view")}
UNITS = ("money","requests","latency_ms","tokens")


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def reference_cells(samples, mode):
    """Compute average gains directly, including within-game state averaging."""
    require(mode in ("action_only","acquired_set"), "Unknown erasure mode")
    grouped = defaultdict(lambda:defaultdict(list))
    identities, transitions = set(), defaultdict(set)
    for sample in samples:
        mid, aid, acquired = sample["match_id"],sample["action"]["action_id"],tuple(sorted(sample["acquired_action_ids"]))
        identity = mid,aid,acquired
        require(identity not in identities and (acquired,aid) in LEGAL,"Duplicate/illegal calibration transition")
        identities.add(identity); transitions[mid].add((acquired,aid))
        key = aid, acquired if mode == "acquired_set" else ()
        require(math.isfinite(sample["delta"]), "Nonfinite historical gain")
        grouped[key][mid].append(sample)
    require(all(value == LEGAL for value in transitions.values()), "Incomplete four-state calibration support")
    cells = {}
    for key,matches in grouped.items():
        values, influences, weeks = {},defaultdict(float),{}
        for mid,entries in matches.items():
            require(len(entries) == (2 if mode == "action_only" else 1), "Incorrect within-match state weighting")
            require(len({e["kickoff"] for e in entries}) == 1,"Conflicting match calendar")
            values[mid] = math.fsum(e["delta"] for e in entries)/len(entries)
            weeks[mid] = tuple(parse_time(entries[0]["kickoff"]).isocalendar()[:2])
        n = len(values)
        mean = math.fsum(values.values())/n
        for mid,value in values.items():
            influences[weeks[mid]] += (value-mean)/n
        b = len(influences)
        se = math.sqrt(b/(b-1)*math.fsum(x*x for x in influences.values())) if b >= 2 else None
        cells[key] = {"expected_gain":mean if n >= 12 else None,"raw_mean":mean,
            "standard_error":se,"support":n,"supported":n >= 12,"time_blocks":b,
            "match_values":values,"week_influences":dict(influences)}
    return cells


def reference_estimate(cells, mode, aid, acquired):
    require((tuple(sorted(acquired)),aid) in LEGAL,"Illegal runtime query state")
    key = aid,tuple(sorted(acquired)) if mode == "acquired_set" else ()
    return cells[key]


def verify_numbers(actual, expected):
    for name in ("model_evsi","evsi_bin","lower_gain"):
        require(actual[name] is None,"Erased prior fabricated "+name)
    for name in ("expected_gain","g_hat","decision_gain"):
        if expected["expected_gain"] is None:
            require(actual[name] is None,"Unsupported prior became numeric")
        else:
            close(actual[name],expected["expected_gain"],"Independent historical mean "+name)
    if expected["standard_error"] is None:
        require(actual["standard_error"] is None,"Unavailable week SE became numeric")
    else:
        close(actual["standard_error"],expected["standard_error"],"Independent week-cluster SE")
    require(actual["support"] == expected["support"],"Unique-match support differs")


def reference_choice(cells, mode, actions, acquired, money, requests):
    candidates = []
    for action in actions:
        aid = action["action_id"]
        if aid in acquired or not action.get("available",True):
            continue
        if action["cost"]["money"] > money+1e-12 or action["cost"]["requests"] > requests+1e-12:
            continue
        estimate = reference_estimate(cells,mode,aid,acquired)
        if estimate["expected_gain"] is not None:
            candidates.append((round(estimate["expected_gain"]-.5*action["cost"]["money"],12),aid))
    candidates.sort(key=lambda r:(-r[0],r[1]))
    return candidates[0][1] if candidates and candidates[0][0]>0 else None


def verify_trace(row,trace,source,actions,evaluator,cells,mode,estimator):
    env,spent = ControlledEpisode(source,actions),dict.fromkeys(UNITS,0.)
    events = trace["events"]
    require(bool(events),"Empty prior trajectory")
    require(row["lambda_money"] == .5,"Prior execution coefficient changed")
    n,mid = row["fold"],row["match_id"]
    _,initial,_ = evaluator.state(n,mid,())
    require(row["initial_probabilities"] == initial["probabilities"],"Unequal initial prediction")
    count = Counter()
    for step,event in enumerate(events):
        require(event["step"] == step,"Noncontiguous prior trajectory")
        acquired = tuple(env.performed_actions)
        state = {**env.visible_state(),"action_history":[{"action_id":aid,"status":"acquired_or_source_missing"} for aid in acquired]}
        require(event["visible_state"] == state,"Prior sees hidden information or wrong acquisition history")
        _,prediction,_ = evaluator.state(n,mid,acquired)
        require(event["prediction"]["probabilities"] == prediction["probabilities"],"Predictor update differs")
        decision = event["decision"]
        expected_action = reference_choice(cells,mode,actions,acquired,row["budget_money"]-spent["money"],2-spent["requests"])
        require(decision["action_id"] == expected_action,"Independent prior tool/STOP differs")
        require(decision["decision"] == ("acquire" if expected_action else "stop"),"Prior decision type differs")
        require(decision["risk_level"] == "mean" and decision["horizon"] == 1,"Prior policy mode changed")
        require(decision["estimator_version"] == estimator.version,"Wrong estimator in decision")
        for unit in UNITS:
            close(decision["physical_budget"]["spent"][unit],spent[unit],"Prior running charge")
        close(decision["physical_budget"]["limits"]["money"],row["budget_money"],"Prior physical money budget")
        close(decision["physical_budget"]["limits"]["requests"],2,"Prior physical request budget")
        eligible = {a["action_id"]:a for a in env.legal_actions()}
        actual = {c["action_id"]:c for c in decision["candidates"]}
        require(len(actual) == len(decision["candidates"]) and set(actual) == set(eligible),"Candidate coverage differs")
        for aid,action in eligible.items():
            candidate = actual[aid]
            affordable = (action["cost"]["money"] <= row["budget_money"]-spent["money"]+1e-12
                and action["cost"]["requests"] <= 2-spent["requests"]+1e-12)
            if not affordable:
                require(not candidate["feasible"] and candidate["reason"] == "physical_budget_exhausted","Wrong prior budget gate")
                continue
            expected = reference_estimate(cells,mode,aid,acquired)
            verify_numbers(candidate,expected)
            penalty = .5*action["cost"]["money"]
            close(candidate["cost_penalty"],penalty,"Prior cost conversion")
            require(candidate["feasible"] == expected["supported"],"Unsupported prior selected")
            if expected["supported"]:
                close(candidate["net_gain"],expected["expected_gain"]-penalty,"Prior net value")
                close(candidate["selection_score"],round(expected["expected_gain"]-penalty,12),"Prior selection rounding")
            count["candidates"] += 1
        if expected_action is None:
            require(step == len(events)-1 and "charged_cost" not in event,"Activity after prior STOP")
            require(row["stop_reason"] == decision["reason"],"Final STOP reason differs")
            break
        action = eligible[expected_action]
        require(event["charged_cost"] == action["cost"],"Prior charge differs")
        for unit in UNITS:
            spent[unit] += action["cost"].get(unit,0.)
        env.reveal(expected_action)
        count["charges"] += 1
    require(events[-1]["decision"]["decision"] == "stop","No prior STOP")
    require(row["actions"] == list(env.performed_actions) and row["final_mask"] == prediction["mask"],"Final prior evidence differs")
    require(row["final_probabilities"] == prediction["probabilities"],"Final prior prediction differs")
    for unit in UNITS:
        close(row["cost"][unit],spent[unit],"Prior total cost")
    for metric,value in losses(prediction["probabilities"],outcome(source)).items():
        close(row[metric],value,"Independent prior outcome score")
    count["events"] = len(events)
    return count


def verify(run,bundle,earlier_bundle):
    from .gain_prior import GainPriorEstimator
    from .analyze_round7 import match_means, paired

    run=Path(run)
    read=lambda name:json.loads((run/name).read_bytes())
    report,design=read("report.json"),read("design.json")
    require(report["status"] == "OK","Incomplete prior experiment")
    source=Path(design["source_round11"])
    previous=json.loads((source/"report.json").read_bytes())
    require(digest(source/"report.json") == SOURCE_REPORT_HASH,"Frozen reference report changed")
    require(digest(design["design_note_path"]) == design["design_note_sha256"] == DESIGN_HASH,"Pre-run design changed")
    require(design["new_strategies"] == POLICIES,"Erasure strategies changed")
    require(design["target_folds"] == [2,3,4,5,6],"Target folds changed")
    require(design["fixed_lambda_money"] == design["validation_utility_money_weight"] == .5,"Frozen objective/cost changed")
    require(design["money_budgets"] == [0.,.02,.03,.05] and design["request_budget"] == 2,"Frozen budgets changed")
    require(design["bootstrap_samples"] == 2000 and design["bootstrap_seed"] == 1729,"Frozen interval settings changed")
    require(report["implementation_sha256"] == read("implementation.json") and
        report["source_artifacts_sha256"] == read("source-artifacts.json"),"Source manifest differs")
    for name,h in report["source_artifacts_sha256"].items():
        require(digest(source/name) == h,"Old artifact changed: "+name)
    for name,h in report["implementation_sha256"].items():
        require(digest(Path(__file__).with_name(name)) == h,"Implementation changed: "+name)
    for item in report["model_artifacts"]:
        require(digest(run/item["path"]) == item["sha256"],"Model artifact changed")
    current,ca=load_episodes(bundle,lead_hours=24)
    earlier,ea=load_episodes(earlier_bundle,lead_hours=24)
    sources={r["match_id"]:r for r in current+earlier}
    require(len(sources) == len(current+earlier) == report["input_matches"],"Data identities differ")
    audit=read("data-audit.json")
    for part,actual in (("current",ca),("earlier",ea)):
        require(actual["platform_manifest_sha256"] == audit[part]["platform_manifest_sha256"],"Source snapshot changed")
    require(digest(run/"catalog.json") == digest(source/"catalog.json"),"Catalog changed")
    actions=read("catalog.json")["actions"]
    old_folds={f["fold"]:f for f in previous["folds"]}
    folds={f["fold"]:f for f in report["folds"]}
    require(set(folds) == {2,3,4,5,6} and len(folds) == len(report["folds"]),"Fold coverage differs")
    models,priors,cells,fold_summaries={},{},{},[]
    counts=Counter()
    for number,fold in sorted(folds.items()):
        old=old_folds[number]
        for field in ("reference_fold","split","training_match_ids","calibration_match_ids","validation_match_ids","test_match_ids"):
            require(fold[field] == old[field],"Prior partition changed: "+field)
        path=run/fold["joint_model"]["path"]
        h=digest(path)
        require(h == old["joint_model"]["sha256"] == fold["joint_model"]["sha256"] == digest(source/old["joint_model"]["path"]),
            "Prior uses a different predictor")
        model=models[number]=JointReturnPredictor.load(path)
        prior_samples=fold["source_same_version_supervision"]
        require(digest(prior_samples["path"]) == prior_samples["sha256"] == old["supervision"]["same_version"]["sha256"],"Prior uses different supervision")
        samples=list(_rows(prior_samples["path"]))
        indexed={(s["match_id"],s["action"]["action_id"],tuple(sorted(s["acquired_action_ids"]))):s for s in samples}
        require(len(indexed) == len(samples) == len(fold["calibration_match_ids"])*4,"Source four-transition coverage differs")
        boundary=min(parse_time(sources[mid]["prediction_cutoff"]) for mid in fold["validation_match_ids"]+fold["test_match_ids"])
        require(all(parse_time(s["result_available_at"]) < boundary for s in samples),"Prior label reaches evaluation time")
        for mode in POLICIES.values():
            ref=cells[number,mode]=reference_cells(samples,mode)
            artifact=fold["gain_priors"][mode]
            prior=priors[number,mode]=GainPriorEstimator.load(run/artifact["path"],model=model,model_sha256=h)
            require(prior.version == artifact["version"] and prior.manifest == artifact["manifest"],"Prior manifest differs")
            require(prior.mode == mode and prior.min_support == 12,"Prior mode/support changed")
            require(set(prior.manifest["training_match_ids"]) == set(fold["calibration_match_ids"]),"Prior training matches differ")
            require(parse_time(prior.manifest["trained_as_of"]) == parse_time(fold["split"]["train_cutoff"]),"Prior training cutoff changed")
            require(prior.manifest["runtime_value_inputs"] == ["action_id"]+(["acquired_action_ids"] if mode == "acquired_set" else []),
                "Erased value inputs changed")
            actual={(r["match_id"],r["action_id"],tuple(r["acquired_action_ids"])):r for r in prior.records}
            require(len(actual) == len(prior.records) and actual.keys() == indexed.keys(),"Prior gain rows differ")
            for identity,row in actual.items():
                sample=indexed[identity]
                close(row["gain"],sample["delta"],"Prior uses an altered historical gain")
                for field in ("kickoff","result_available_at"):
                    require(parse_time(row[field]) == parse_time(sample[field]),"Prior record calendar changed")
                require(parse_time(row["prediction_cutoff"]) == parse_time(sample["state"]["prediction_cutoff"]),"Prior cutoff changed")
                require(row["generating_model_version"] == model.model_version and row["generating_model_sha256"] == h,"Prior label model changed")
            require(len(prior.cells) == (2 if mode == "action_only" else 4),"Unexpected prior numerical groups")
            for encoded,cell in prior.cells.items():
                aid,acquired=json.loads(encoded)
                key=aid,tuple(acquired or ())
                expected=ref[key]
                close(cell["sample_mean"],expected["raw_mean"],"Independent prior cell mean")
                require(set(cell["match_values"]) == set(expected["match_values"]),"Prior cell membership differs")
                for mid,value in expected["match_values"].items():
                    close(cell["match_values"][mid]["gain"],value,"Independent within-match gain")
                expected_weeks={f"{y}-W{w:02d}":value for (y,w),value in expected["week_influences"].items()}
                require(cell["week_influences"].keys() == expected_weeks.keys(),"Prior weekly partition differs")
                for week,value in expected_weeks.items():
                    close(cell["week_influences"][week],value,"Independent cluster influence")
                counts["prior_cells"] += 1
            for acquired,aid in LEGAL:
                verify_numbers(prior.estimate_numbers(aid,acquired),reference_estimate(ref,mode,aid,acquired))
            counts["fitted_gain_records"] += len(prior.records)
        fold_summaries.append({"fold":number,"reference_fold":fold["reference_fold"],"calibration_matches":len(fold["calibration_match_ids"]),
            "means":{mode:[{"action_id":aid,"acquired_action_ids":list(acq),"expected_gain":v["expected_gain"],
                "support":v["support"]} for (aid,acq),v in sorted(cells[number,mode].items())] for mode in POLICIES.values()}})
        print(f"round12 audit fold {number}: independent historical means verified",flush=True)
    evaluator=IndependentEvaluator(sources,actions,models,{}, {})
    for partition,prefix in (("test",""),("validation","validation-")):
        rows=list(_rows(run/(prefix+"episodes.jsonl")))
        references=list(_rows(source/(prefix+"episodes.jsonl")))
        fold_ids={n:set(f[partition+"_match_ids"]) for n,f in folds.items()}
        verify_episode_coverage(rows,fold_ids,list(POLICIES),design["money_budgets"],[0])
        require(len(rows) == report["episode_runs" if partition == "test" else "validation_episode_runs"],"Prior episode count differs")
        combined=references+rows
        metrics=report["metrics" if partition == "test" else "validation_metrics_diagnostic_only"]
        verify_aggregate_metrics(combined,metrics)
        for values in metrics.values():
            close(values["utility_loss"],values["log_loss"]+.5*values["money"],"Combined utility")
        indexed={tuple(r[k] for k in KEYS):r for r in rows}
        seen=set()
        for trace in _rows(run/(prefix+"trajectories.jsonl")):
            key=tuple(trace[k] for k in KEYS)
            require(key in indexed and key not in seen,"Duplicate/unknown prior trace")
            seen.add(key); row=indexed[key]
            n,mid=row["fold"],row["match_id"]
            mode=POLICIES[row["strategy"]]
            require(row["partition"] == trace["partition"] == partition and row["reference_fold"] == trace["reference_fold"] == n-2,
                "Prior episode identity differs")
            require(row["gain_version"] == priors[n,mode].version and row["estimator_kind"] == "historical-gain-prior-v1" and
                row["erasure_mode"] == mode,"Prior episode estimator differs")
            require(row["model_version"] == models[n].model_version and row["base_model_version"] == models[n].base_predictor.model_version,
                "Prior predictor identity differs")
            require(row["actual_external_requests"] == row["actual_llm_tokens"] == 0,"Unexpected external use")
            result=verify_trace(row,trace,sources[mid],actions,evaluator,cells[n,mode],mode,priors[n,mode])
            counts.update({partition+"_"+k:v for k,v in result.items()})
        require(seen == set(indexed),"Missing prior trace")
        counts[partition+"_episodes"] = len(rows)
        if partition == "test":
            means=match_means(combined,.5)
            for label,expected in report["paired_comparisons"].items():
                pair,budget=label.rsplit("/money=",1)
                a,b=pair.split("-vs-")
                calculated={metric:paired(means[a,float(budget)],means[b,float(budget)],metric,2000,1729)
                    for metric in ("log_loss","money","utility_loss")}
                require(calculated == expected,"Paired utility comparison differs")
                counts["paired_comparisons"] += 1
        print(f"round12 audit {partition}: {len(rows)} new trajectories and combined metrics verified",flush=True)
    # Invariance is evaluated on every legal observed historical public state,
    # including states that the policy itself would not choose to visit.
    for row in _rows(source/"transition-diagnostics.jsonl"):
        n,mid,aid,acquired=row["fold"],row["match_id"],row["action_id"],row["acquired_action_ids"]
        state,_,_=evaluator.state(n,mid,acquired)
        action=next(a for a in actions if a["action_id"] == aid)
        for mode in POLICIES.values():
            actual=priors[n,mode].estimate_value(state,[action])[0]
            verify_numbers(actual,reference_estimate(cells[n,mode],mode,aid,acquired))
            counts["matched_state_estimates"] += 1
    names=("report.json","design.json","catalog.json","episodes.jsonl","validation-episodes.jsonl",
        "trajectories.jsonl","validation-trajectories.jsonl","implementation.json","source-artifacts.json","data-audit.json")
    return {"status":"VERIFIED","round":12,"counts":dict(counts),"fold_means":fold_summaries,
        "unique_public_joint_states":len(evaluator.states),"test_matches":200,"new_independent_matches":0,"external_requests":0,
        "artifacts":{name:digest(run/name) for name in names},"verifier_sha256":digest(__file__),
        "scope":"Independent source-bound mean/SE calculation, within-match state weighting, all new decisions/costs/STOP, common-state value invariance and paired metrics"}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("run","bundle","earlier-bundle","output"):
        parser.add_argument("--"+name,required=True)
    args=parser.parse_args()
    result=verify(args.run,args.bundle,args.earlier_bundle)
    output=Path(args.output)
    require(not output.exists(),"Verification output exists")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:result[k] for k in ("status","counts","unique_public_joint_states")}))
