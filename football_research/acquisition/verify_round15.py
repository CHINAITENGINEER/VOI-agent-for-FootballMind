"""Verify frozen finite benchmark laws, independent policy optima and replays."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations
import json
import math
from pathlib import Path

from .finite_tree_reference import FiniteTreeReference, STRATEGIES, distorted_population, entropy, frozen_population

DESIGN_SHA256 = "aa70d9bd6ac92c04722935cdf021d29deaaa811e6f496f5d40fd420aab37a64b"
SCENARIOS = ("gated_stop", "gated_route", "complementary", "redundant", "irrelevant")


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def require(value, detail):
    if not value:
        raise AssertionError(detail)


def close(actual, expected, detail, tol=1e-10):
    require(type(actual) in (int,float) and math.isfinite(actual)
            and abs(actual-expected) <= tol, f"{detail}: actual={actual!r}, expected={expected!r}")


def rows_equal(actual, expected):
    def indexed(rows):
        out = {}
        for row in rows:
            key = row["outcome"], tuple(sorted(row["observations"].items()))
            require(key not in out, "Duplicate population cell")
            out[key] = row["mass"]
        return out
    a,b = indexed(actual),indexed(expected)
    require(a.keys() == b.keys(), "Declared population support differs from frozen factors")
    for key in a:
        close(a[key],b[key],"Frozen population cell mass")
    close(math.fsum(a.values()),1.,"Population total mass")


def verify_scenarios(scenarios):
    require(set(scenarios) == set(SCENARIOS), "Scenario coverage differs")
    references = {}
    for name in SCENARIOS:
        scenario = scenarios[name]
        truth,actions,cap = frozen_population(name)
        require(scenario["schema"] == "finite-information-scenario-v1" and scenario["scenario_id"] == name, "Scenario identity")
        rows_equal(scenario["worlds"],truth)
        require(len({r["world_id"] for r in scenario["worlds"]}) == len(truth), "Duplicate or absent world ID")
        costs = {a["action_id"]: a["cost"]["money"] for a in actions}
        require(len(scenario["actions"]) == len(costs), "Action count differs")
        require({a["action_id"] for a in scenario["actions"]} == set(costs), "Scenario action identities")
        for action in scenario["actions"]:
            close(action["cost"]["money"],costs[action["action_id"]],"Predeclared action fee")
            close(action["cost"]["requests"],1.,"Predeclared request fee")
            require(action["cost_status"] == "simulated" and action.get("available",True), "Wrong cost kind or availability")
            for unit in ("latency_ms","tokens"):
                close(action["cost"].get(unit,0.),0.,"Extra resource charge")
        close(scenario["budget_money"],cap,"Predeclared scenario budget")
        require(scenario["request_budget"] == 2 and scenario["lambda_money"] == 1., "Changed loss/budget objective")
        names = {"correct","independent"}|({"swapped_context"} if name.startswith("gated") else set())
        require(set(scenario["models"]) == names, "Model variant coverage differs")
        for model_name,spec in scenario["models"].items():
            require(spec["scenario_id"] == name and spec["model_name"] == model_name, "Model identity changed")
            model_rows = distorted_population(truth,model_name)
            rows_equal(spec["rows"],model_rows)
            ref = references[name,model_name] = FiniteTreeReference(model_rows,actions,cap)
            close(entropy(ref.posterior({})),entropy(FiniteTreeReference(truth,actions,cap).posterior({})),"Preserved prior")
            for row in truth:
                require(ref.posterior(row["observations"])[row["outcome"]] > 0, "No support for a scored true world")
    return references


def verify_plan(plan,reference,observations,horizon):
    expected = reference.solve(observations,horizon)
    p = reference.posterior(observations)
    for actual,wanted in zip(plan["prediction"]["probabilities"],p):
        close(actual,wanted,"Current model posterior")
    require(plan["action_id"] == expected["action_id"], "First action differs from independent finite-tree optimum")
    require(plan["horizon"] == horizon, "Changed planning horizon")
    for field,wanted in (("expected_terminal_risk",expected["expected_terminal_risk"]),
                         ("expected_total_risk",expected["expected_total_risk"]),
                         ("planning_net_gain",expected["net_gain"]),
                         ("selection_score",expected["selection_score"])):
        close(plan[field],wanted,"Independent policy optimum "+field)
    close(plan["expected_plan_cost_vector"]["money"],expected["expected_money"],"Optimum expected cost")
    close(plan["expected_plan_cost_vector"]["requests"],expected["expected_queries"],"Optimum expected calls")
    actual_candidates = {r["action_id"]:r for r in plan["candidates"]}
    require(len(actual_candidates) == len(plan["candidates"]),"Duplicate planning candidate")
    require(set(actual_candidates) == set(reference.costs),"Planning catalog coverage")
    counts = Counter(plans=1,independently_enumerated_trees=expected["enumerated_policy_trees"])
    for aid,candidate in actual_candidates.items():
        wanted = expected["candidate_optima"].get(aid)
        require(candidate["feasible"] == (wanted is not None),"Independent physical feasibility")
        if wanted is None:
            continue
        before = entropy(p)
        branches = reference.branches(observations,aid)
        immediate = before - math.fsum(w*entropy(reference.posterior({**observations,aid:z})) for z,w in branches)
        close(candidate["immediate_evsi"],max(0.,immediate),"Immediate mutual information")
        close(candidate["immediate_penalty"],reference.costs[aid],"Immediate cost")
        for field in ("expected_terminal_risk","expected_total_risk"):
            close(candidate[field],wanted[field],"Candidate best tree "+field)
        close(candidate["expected_plan_cost_vector"]["money"],wanted["expected_money"],"Candidate expected fee")
        close(candidate["expected_plan_cost_vector"]["requests"],wanted["expected_queries"],"Candidate expected requests")
        close(candidate["expected_plan_penalty"],wanted["expected_money"],"Candidate penalty")
        close(candidate["expected_loss_improvement"],before-wanted["expected_terminal_risk"],"Candidate planned loss gain")
        close(candidate["planning_net_gain"],wanted["net_gain"],"Candidate total net value")
        projected = {b["observation"]:b for b in candidate["branches"]}
        require(len(projected) == len(candidate["branches"]) and projected.keys() == dict(branches).keys(),"Model branch coverage")
        for token,probability in branches:
            branch = projected[token]
            close(branch["probability"],probability,"Modeled return probability")
            first = {**observations,aid:token}
            second = wanted["children"][token]
            require(branch["hypothetical"] is True and branch["continuation_action"] == second,"Independent conditional continuation")
            posterior = reference.posterior(first)
            for actual,wanted_probability in zip(branch["posterior_probabilities"],posterior):
                close(actual,wanted_probability,"Branch posterior")
            fee = reference.costs[second] if second else 0.
            risk = (math.fsum(w*entropy(reference.posterior({**first,second:z})) for z,w in reference.branches(first,second))
                    if second else entropy(posterior))
            close(branch["continuation_expected_terminal_risk"],risk,"Branch terminal risk")
            close(branch["continuation_expected_cost_vector"]["money"],fee,"Branch future fee")
            close(branch["continuation_expected_total_risk"],risk+fee,"Branch total risk")
            counts["mathematical_branches"] += 1
        counts["candidate_values"] += 1
    return counts


def all_public_states(reference):
    states = {()}
    for size in range(1,reference.requests+1):
        for subset in combinations(sorted(reference.costs),size):
            if sum(reference.costs[a] for a in subset)>reference.cap+1e-12:
                continue
            states.update(tuple((a,row["observations"][a]) for a in subset) for row in reference.rows)
    return [dict(key) for key in sorted(states)]


def verify_all_model_states(scenarios,references):
    from .finite_information import FiniteJointModel
    from .joint_planner import JointRiskPlanner
    counts = Counter()
    for (name,model_name),reference in references.items():
        scenario = scenarios[name]
        model = FiniteJointModel(scenario["models"][model_name])
        for observations in all_public_states(reference):
            state = {"observations":observations,"action_history":[]}
            for horizon in (1,2):
                planner = JointRiskPlanner(model,horizon=horizon,lambda_cost={"money":1.})
                plan = planner.plan(state,scenario["actions"],{
                    "limits":{"money":reference.cap,"requests":2},
                    "spent":{"money":math.fsum(reference.costs[a] for a in observations),"requests":len(observations)}})
                counts.update(verify_plan(plan,reference,observations,horizon))
            counts["public_states"] += 1
    return counts


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def record_key(row):
    return tuple(row[k] for k in ("scenario_id","model_name","policy","world_id"))


def verify_trace(row,trace,scenario,reference):
    world = next(w for w in scenario["worlds"] if w["world_id"] == row["world_id"])
    require(row["world_observations"] == world["observations"] and row["outcome"] == world["outcome"],"Scoring world identity")
    close(row["mass"],world["mass"],"Scoring population mass")
    close(trace["mass"],world["mass"],"Trajectory population mass")
    observations,history,actions,predictions = {},[],[],[]
    counts = Counter(episodes=1)
    spent = 0.
    events = trace["events"]
    require(bool(events),"Empty policy trace")
    for step,event in enumerate(events):
        require(event["step"] == step and event["visible_state"] == {"observations":observations,"action_history":history},
                "Runtime state contains unacquired data or mismatched receipt history")
        expected_p = reference.posterior(observations)
        actual_p = event["prediction"]["probabilities"]
        require(len(actual_p) == 3 and event["prediction"]["observations"] == observations
                and event["prediction"]["prediction_kind"] == "acquired_evidence_posterior","Actual prediction scope")
        for actual,expected in zip(actual_p,expected_p):
            close(actual,expected,"Actual step posterior")
        predictions.append(actual_p)
        expected_action = reference.action(row["policy"],observations)
        decision = event["decision"]
        require(decision["action_id"] == expected_action,"Executed policy differs from independent reference")
        require(decision["decision"] == ("acquire" if expected_action else "stop"),"Wrong query/STOP decision")
        if row["policy"] in {"myopic","adaptive-two-step"}:
            counts.update(verify_plan(decision,reference,observations,1 if row["policy"] == "myopic" else 2))
            close(decision["physical_budget"]["spent"]["money"],spent,"Only actual costs enter spent budget")
            close(decision["physical_budget"]["spent"]["requests"],len(actions),"Only actual calls consume request budget")
        if expected_action is None:
            require(step == len(events)-1 and "receipt" not in event,"Extra receipt or activity after STOP")
            break
        require(expected_action in reference.legal(observations),"One-shot or physical resource limit violated")
        receipt = event["receipt"]
        require(receipt["action_id"] == expected_action and receipt["observation"] == world["observations"][expected_action],"Actual reveal differs")
        close(receipt["cost"]["money"],reference.costs[expected_action],"Actual charge differs from current query")
        close(receipt["cost"]["requests"],1.,"Actual request charge")
        for unit in ("latency_ms","tokens"):
            close(receipt["cost"][unit],0.,"Unexpected extra actual resource")
        spent += reference.costs[expected_action]
        observations[expected_action] = receipt["observation"]
        history.append({"action_id":expected_action,"status":"observed"})
        actions.append(expected_action)
        counts["actual_synthetic_reveals"] += 1
    require(events[-1]["decision"]["action_id"] is None,"Missing terminal STOP")
    require(row["actions"] == actions and row["observations"] == observations,"Final acquired state differs")
    for actual,expected in zip(row["initial_probabilities"],predictions[0]):
        close(actual,expected,"Initial forecast snapshot")
    for actual,expected in zip(row["final_probabilities"],predictions[-1]):
        close(actual,expected,"Final forecast snapshot")
    close(row["cost"]["money"],spent,"Actual total money")
    close(row["cost"]["requests"],len(actions),"Actual total queries")
    y = world["outcome"]  # Used only after the complete action/observation path has been verified.
    initial_loss,final_loss = (-math.log(predictions[i][y]) for i in (0,-1))
    close(row["initial_log_loss"],initial_loss,"Initial true-world log loss")
    close(row["log_loss"],final_loss,"Final true-world log loss")
    close(row["utility_loss"],final_loss+spent,"True-world risk with actual fees")
    close(row["realized_gain"],initial_loss-final_loss,"Actual path gain")
    close(math.fsum(math.log(b[y])-math.log(a[y]) for a,b in zip(predictions,predictions[1:])),
          row["realized_gain"],"Step gains telescope to initial/final gain")
    close(row["brier_score"],sum((predictions[-1][i]-float(i == y))**2 for i in range(3)),"True-world Brier score")
    counts["actual_prediction_snapshots"] += len(events)
    return counts


def verify(run):
    run = Path(run)
    report = json.loads((run/"report.json").read_bytes())
    spec = json.loads((run/"spec.json").read_bytes())
    require(report["status"] == "OK" and report["design_note_sha256"] == DESIGN_SHA256,"Incomplete/unregistered finite study")
    require(digest(spec["design_note_path"]) == DESIGN_SHA256,"Frozen study design changed")
    require(spec["policies"] == list(STRATEGIES),"Policy coverage changed")
    for name,value in report["implementation_sha256"].items():
        require(digest(Path(__file__).with_name(name)) == value,"Production implementation changed: "+name)
    for name,value in report["artifact_sha256"].items():
        require(digest(run/name) == value,"Production artifact changed: "+name)
    scenarios = json.loads((run/"scenarios.json").read_bytes())
    references = verify_scenarios(scenarios)
    for item in report["scenarios"]:
        path = run/item["path"]
        require(digest(path) == item["sha256"] and json.loads(path.read_bytes()) == scenarios[item["scenario_id"]],"Scenario artifact binding")
    from .finite_information import FiniteJointModel
    for item in report["models"]:
        path = run/item["path"]
        require(digest(path) == item["sha256"],"Model artifact binding")
        model = FiniteJointModel.load(path)
        require(model.model_version == item["version"] and model.spec == scenarios[item["scenario_id"]]["models"][item["model_name"]],"Declared model identity")
    counts = verify_all_model_states(scenarios,references)
    counts["models"] = len(references)
    selections = json.loads((run/"fixed-policy-selections.json").read_bytes())
    expected_metrics,model_metrics = {},{}
    for (name,model_name),reference in references.items():
        fixed = selections[name+"/"+model_name]
        for policy,wanted in (("best-fixed-subset",reference.fixed_subset()),("best-fixed-order-stop",reference.fixed_order_stop())):
            require(fixed[policy]["selected_order"] == wanted["order"],"Pre-return fixed order/subset selection differs")
            close(fixed[policy]["selected_expected_total_risk"],wanted["expected_total_risk"],"Independent fixed-policy expectation")
            counts["fixed_baseline_selections"] += 1
        truth = scenarios[name]["worlds"]
        for policy in STRATEGIES:
            key = name+"/"+model_name+"/"+policy
            expected_metrics[key] = reference.evaluate(policy,truth)
            model_metrics[key] = reference.evaluate(policy)
    episodes,traces = read_lines(run/"episodes.jsonl"),read_lines(run/"trajectories.jsonl")
    indexed = {record_key(row):row for row in traces}
    require(len(indexed) == len(traces),"Duplicate trajectory identity")
    expected_keys = {(name,model,policy,w["world_id"]) for name,s in scenarios.items()
                     for model in s["models"] for policy in STRATEGIES for w in s["worlds"]}
    require(len(episodes) == len({record_key(r) for r in episodes}) and {record_key(r) for r in episodes} == indexed.keys() == expected_keys,
            "Incomplete/duplicate declared-world policy coverage")
    for row in episodes:
        counts.update(verify_trace(row,indexed[record_key(row)],scenarios[row["scenario_id"]],references[row["scenario_id"],row["model_name"]]))
    require(set(report["metrics"]) == set(expected_metrics),"Missing or extra policy metrics")
    declared_records = {"/".join(r[k] for k in ("scenario_id","model_name","policy")):r for r in read_lines(run/"model-expectations.jsonl")}
    require(set(declared_records) == set(expected_metrics),"Missing or extra declared-model expectations")
    for key,wanted in expected_metrics.items():
        actual = report["metrics"][key]
        for field,value in wanted.items():
            close(actual["requests" if field == "queries" else field],value,"Exact true expectation "+key+"/"+field)
        for field,value in model_metrics[key].items():
            close(declared_records[key]["requests" if field == "queries" else field],value,"Exact model expectation "+field)
        close(actual["declared_model_expected_utility"],model_metrics[key]["utility_loss"],"Modeled versus actual distinction")
        name = key.split("/")[0]
        optimum = references[name,"correct"].solve()["expected_total_risk"]
        close(actual["production_optimum_reference"],optimum,"True oracle risk")
        close(actual["regret_vs_true_two_query_optimum"],wanted["utility_loss"]-optimum,"Oracle regret")
        require(wanted["utility_loss"] >= optimum-1e-10,"Policy beat complete true-model policy enumeration")
        counts["weighted_policy_metrics"] += 1
    require(report["episode_runs"] == len(expected_keys) == 1560 and report["true_worlds"] == 120
            and report["distinct_return_tuples"] == 40 and report["scenario_model_pairs"] == 12,"Benchmark count claims")
    require(report["new_independent_football_matches"] == report["external_requests"] == report["llm_tokens"] == report["new_fitted_models"] == 0,
            "Unexpected fitted/live/football evidence claim")
    return {"status":"VERIFIED_FINITE_MECHANISMS_NOT_FOOTBALL_EFFECTIVENESS","round":15,"counts":dict(counts),
        "true_metrics":expected_metrics,"model_metrics":model_metrics,
        "correct_policy_optima":{name:references[name,"correct"].solve() for name in SCENARIOS},
        "report_sha256":digest(run/"report.json"),"verifier_sha256":digest(__file__),
        "reference_sha256":digest(Path(__file__).with_name("finite_tree_reference.py")),
        "design_sha256":DESIGN_SHA256,"new_independent_football_matches":0,
        "scope":"Frozen generative cells, explicit finite policy trees, complete true-world paths and weighted expectations; synthetic mechanisms only"}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",required=True);parser.add_argument("--output",required=True)
    args=parser.parse_args();output=Path(args.output)
    require(not output.exists(),"Verification already exists")
    result=verify(args.run)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"status":result["status"],"counts":result["counts"]}))
