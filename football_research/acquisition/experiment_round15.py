"""Exact finite-population scores for frozen synthetic information mechanisms."""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from itertools import combinations, permutations
import json
import math
from pathlib import Path

from .finite_information import build_scenarios, FiniteJointModel, HYPOTHESIS_KEY
from .joint_planner import JointRiskPlanner, UNITS, _budget, _cost, entropy
from .experiment_round4 import write_json
from .experiment_round11 import digest, write_lines
from football_research.baselines.api import canonical_json

VERSION="finite-information-mechanism-benchmark-round15-v1"
DESIGN_SHA256="aa70d9bd6ac92c04722935cdf021d29deaaa811e6f496f5d40fd420aab37a64b"
POLICIES=("base-only","myopic","adaptive-two-step","best-fixed-subset","best-fixed-order-stop")


def empty_state():return {"observations":{},"action_history":[]}


def budget_for(scenario,spent=None):
    return {"limits":{"money":scenario["budget_money"],"requests":scenario["request_budget"]},
        "spent":dict(spent or {u:0. for u in UNITS})}


def fixed_plan(model,scenario,order,state,budget,*,allow_stop,hypothetical=False):
    """Evaluate a specified fixed list; only STOP, never alternative tools, is allowed."""
    if not hypothetical and HYPOTHESIS_KEY in state:raise ValueError("Real fixed-policy boundary rejects hypothetical state")
    limits,spent=_budget(budget)
    observed=model.observed_actions(state,hypothetical=hypothetical)
    if len(order)!=len(set(order)) or len(order)>2 or any(a not in model.action_ids for a in order):raise ValueError("Fixed order must contain unique modeled actions, length at most two")
    if observed!=sorted(order[:len(observed)]):raise ValueError("Public observations must match a prefix of the selected fixed order")
    remaining=order[len(observed):]
    prediction=model.predict_hypothetical(state) if hypothetical else model.predict_partial(state)
    p=prediction["probabilities"];risk=entropy(p)
    zero={u:0. for u in UNITS}
    stop={"decision":"stop","action_id":None,"reason":"fixed_order_exhausted" if not remaining else "nonpositive_fixed_order_gain",
        "expected_terminal_risk":risk,"expected_plan_cost_vector":zero,"expected_plan_penalty":0.,"expected_total_risk":risk,
        "planning_net_gain":0.,"selection_score":0.,"prediction":prediction,"fixed_order":list(order),"candidates":[]}
    if not remaining:return stop
    action=next(a for a in scenario["actions"] if a["action_id"]==remaining[0]);cost=_cost(action)
    if not action.get("available",True) or any(spent[u]+cost[u]>cap+1e-12 for u,cap in limits.items()):
        return {**stop,"reason":"physical_budget_or_availability"}
    enumeration=model.enumerate_action(state,action,hypothetical=hypothetical)
    children=[]
    next_budget={"limits":limits,"spent":{u:spent[u]+cost[u] for u in UNITS}}
    for branch in enumeration["branches"]:
        following=fixed_plan(model,scenario,order,branch["next_state"],next_budget,allow_stop=allow_stop,hypothetical=True)
        children.append({"observation":branch["observation"],"probability":branch["probability"],"posterior_probabilities":branch["posterior_probabilities"],
            "continuation_action":following["action_id"],"continuation_expected_total_risk":following["expected_total_risk"],"hypothetical":True,
            "_following":following})
    loss=math.fsum(b["probability"]*b["_following"]["expected_terminal_risk"] for b in children)
    expected_cost={u:cost[u]+math.fsum(b["probability"]*b["_following"]["expected_plan_cost_vector"][u] for b in children) for u in UNITS}
    penalty=scenario["lambda_money"]*expected_cost["money"]
    total=loss+penalty;net=risk-total
    for b in children:b.pop("_following")
    candidate={"action_id":action["action_id"],"expected_terminal_risk":loss,"expected_plan_cost_vector":expected_cost,
        "expected_plan_penalty":penalty,"expected_total_risk":total,"planning_net_gain":net,"selection_score":round(net,12),"branches":children}
    if allow_stop and round(net,12)<=0:return {**stop,"candidates":[candidate]}
    return {**stop,**candidate,"decision":"acquire","reason":"fixed_order_query_if_positive" if allow_stop else "execute_preselected_subset",
        "candidates":[candidate]}


def select_fixed_policies(model,scenario):
    aids=sorted(model.action_ids);out={}
    for policy,allow_stop in (("best-fixed-subset",False),("best-fixed-order-stop",True)):
        orders=[()]
        for length in (1,2):
            orders.extend(combinations(aids,length) if not allow_stop else permutations(aids,length))
        candidates=[]
        for order in orders:
            chosen=[next(a for a in scenario["actions"] if a["action_id"]==aid) for aid in order]
            if len(order)>scenario["request_budget"] or math.fsum(a["cost"]["money"] for a in chosen)>scenario["budget_money"]+1e-12:continue
            plan=fixed_plan(model,scenario,order,empty_state(),budget_for(scenario),allow_stop=allow_stop)
            candidates.append({"order":list(order),"expected_total_risk":plan["expected_total_risk"],
                "rounded_expected_total_risk":round(plan["expected_total_risk"],12),
                "expected_terminal_risk":plan["expected_terminal_risk"],"expected_cost":plan["expected_plan_cost_vector"]})
        selected=min(candidates,key=lambda r:(r["rounded_expected_total_risk"],len(r["order"]),tuple(r["order"])))
        ties=[r["order"] for r in candidates if r["rounded_expected_total_risk"]==selected["rounded_expected_total_risk"]]
        out[policy]={"selected_order":selected["order"],"selected_expected_total_risk":selected["expected_total_risk"],
            "tied_best_rounded_orders":ties,"candidates":candidates,"selection":"minimum rounded risk, shorter list, lexicographic list",
            "allow_content_dependent_stop":allow_stop}
    return out


def run_returns(scenario,model,policy,returns,*,fixed_selections):
    """Environment receives actual returns only; the model sees acquired state only."""
    if policy not in POLICIES:raise ValueError("Unregistered finite policy")
    if set(returns)!=set(model.action_ids):raise ValueError("Complete environment return tuple required")
    # Validate categories without asking the predictor to condition on the full tuple.
    if any(returns[a] not in model.categories[a] for a in model.action_ids):raise ValueError("Unsupported environment category")
    state=empty_state();spent={u:0. for u in UNITS};performed=[];events=[]
    planner=JointRiskPlanner(model,horizon=1 if policy=="myopic" else 2,lambda_cost={"money":scenario["lambda_money"]})
    for step in range(scenario["request_budget"]+1):
        prediction=model.predict_partial(state)
        if policy=="base-only":
            decision={"decision":"stop","action_id":None,"reason":"base_only","candidates":[],"expected_total_risk":entropy(prediction["probabilities"])}
        elif policy in ("myopic","adaptive-two-step"):
            decision=planner.plan(state,scenario["actions"],budget_for(scenario,spent))
        else:
            decision=fixed_plan(model,scenario,fixed_selections[policy]["selected_order"],state,budget_for(scenario,spent),allow_stop=policy=="best-fixed-order-stop")
        event={"step":step,"visible_state":deepcopy(state),"prediction":prediction,"decision":decision};events.append(event)
        if decision["decision"]=="stop":break
        aid=decision["action_id"]
        if aid in performed or len(performed)>=scenario["request_budget"]:raise ValueError("One-shot request cap violated")
        action=next(a for a in scenario["actions"] if a["action_id"]==aid);cost=_cost(action)
        if spent["money"]+cost["money"]>scenario["budget_money"]+1e-12:raise ValueError("Actual money cap violated")
        z=returns[aid]  # First access to this actual return, after action selection.
        event["receipt"]={"action_id":aid,"observation":z,"cost":deepcopy(cost)}
        state["observations"][aid]=z;state["action_history"].append({"action_id":aid,"status":"observed"})
        performed.append(aid)
        for unit in UNITS:spent[unit]+=cost[unit]
    else:raise ValueError("Policy did not STOP within request cap")
    return {"actions":performed,"observations":deepcopy(state["observations"]),"initial_probabilities":events[0]["prediction"]["probabilities"],
        "final_probabilities":prediction["probabilities"],"cost":spent,"model_version":model.model_version,
        "stop_reason":decision["reason"]},events


def score_world(result,world,*,lambda_money):
    p=result["final_probabilities"];y=world["outcome"]
    if p[y]<=0 or not math.isfinite(p[y]):raise ValueError("Scored true world has zero or invalid predicted outcome probability")
    loss=-math.log(p[y]);initial=-math.log(result["initial_probabilities"][y])
    return {**result,"mass":world["mass"],"outcome":y,"log_loss":loss,"initial_log_loss":initial,
        "utility_loss":loss+lambda_money*result["cost"]["money"],"realized_gain":initial-loss,
        "brier_score":math.fsum((p[k]-float(k==y))**2 for k in range(3))}


def weighted_metrics(rows):
    mass=math.fsum(r["mass"] for r in rows)
    if abs(mass-1)>1e-10:raise ValueError("Scored population mass must sum to one")
    result={"population_mass":mass,"worlds":len(rows),"observation_patterns":len({tuple(sorted(r["world_observations"].items())) for r in rows}),
        **{key:math.fsum(r["mass"]*r[key] for r in rows) for key in ("log_loss","utility_loss","initial_log_loss","realized_gain","brier_score")},
        **{u:math.fsum(r["mass"]*r["cost"][u] for r in rows) for u in UNITS}}
    for key,index in (("first_action_distribution",0),("second_action_distribution",1)):
        counts=defaultdict(float)
        for r in rows:counts[r["actions"][index] if len(r["actions"])>index else "STOP"]+=r["mass"]
        result[key]=dict(sorted(counts.items()))
    counts=defaultdict(float)
    for r in rows:counts[str(len(r["actions"]))]+=r["mass"]
    result["query_count_distribution"]=dict(sorted(counts.items()))
    result["expectation_scope"]="exact specified finite population, no sampling interval"
    return result


def run(output,design_note):
    output,design_note=Path(output),Path(design_note)
    if output.exists():raise FileExistsError(output)
    if digest(design_note)!=DESIGN_SHA256:raise ValueError("Frozen round-fifteen specification changed")
    scenarios=build_scenarios();output.mkdir(parents=True);(output/"models").mkdir();(output/"scenarios").mkdir()
    spec={"experiment_version":VERSION,"design_note_path":str(design_note.resolve()),"design_note_sha256":DESIGN_SHA256,
        "policies":list(POLICIES),"lambda_money":1.,"request_budget":2,"worlds":"all positive generating masses",
        "model_variants":"correct and independent for all; swapped_context additionally for gated scenarios",
        "baseline_selection":"pre-return model-risk minimization; minimum rounded12 risk, shorter list, lexicographic list",
        "no_fitting":True,"no_bootstrap_or_sampling":True,"new_independent_football_matches":0}
    write_json(output/"spec.json",spec);write_json(output/"scenarios.json",scenarios)
    implementation={name:digest(Path(__file__).with_name(name)) for name in ("finite_information.py","experiment_round15.py","joint_planner.py","value.py","experiment_round4.py","experiment_round11.py")}
    write_json(output/"implementation.json",implementation)
    episodes=[];models=[];scenario_artifacts=[];metrics={};model_metrics=[];selections={};cache={}
    with (output/"trajectories.jsonl").open("w",encoding="utf-8") as traces:
        for sid,scenario in scenarios.items():
            sp=output/"scenarios"/f"{sid}.json";write_json(sp,scenario)
            scenario_artifacts.append({"scenario_id":sid,"path":str(sp.relative_to(output)),"sha256":digest(sp)})
            for name,mspec in scenario["models"].items():
                model=FiniteJointModel(mspec);mp=output/"models"/f"{sid}-{name}.json";model.save(mp);model=FiniteJointModel.load(mp)
                models.append({"scenario_id":sid,"model_name":name,"path":str(mp.relative_to(output)),"sha256":digest(mp),"version":model.model_version})
                selection=select_fixed_policies(model,scenario);selections[sid+"/"+name]=selection
                for policy in POLICIES:
                    true_rows=[];declared_rows=[]
                    for population,worlds in (("true",scenario["worlds"]),("model",model.spec["rows"])):
                        for world in worlds:
                            key=(sid,name,policy,tuple(sorted(world["observations"].items())))
                            if key not in cache:cache[key]=run_returns(scenario,model,policy,world["observations"],fixed_selections=selection)
                            result,events=cache[key]
                            scored=score_world(result,world,lambda_money=scenario["lambda_money"])
                            scored.update(scenario_id=sid,model_name=name,policy=policy,world_observations=deepcopy(world["observations"]))
                            if population=="true":
                                scored["world_id"]=world["world_id"];true_rows.append(scored);episodes.append(scored)
                                traces.write(canonical_json({"scenario_id":sid,"model_name":name,"policy":policy,"world_id":world["world_id"],
                                    "mass":world["mass"],"events":events})+"\n")
                            else:declared_rows.append(scored)
                    key=sid+"/"+name+"/"+policy
                    metrics[key]=weighted_metrics(true_rows)
                    declared=weighted_metrics(declared_rows);model_metrics.append({"scenario_id":sid,"model_name":name,"policy":policy,**declared})
                    metrics[key]["declared_model_expected_utility"]=declared["utility_loss"]
                    metrics[key]["declared_model_expected_log_loss"]=declared["log_loss"]
                print(f"round15 {sid}/{name}: five exact policies complete",flush=True)
    for sid in scenarios:
        optimum=metrics[sid+"/correct/adaptive-two-step"]["utility_loss"]
        for key,result in metrics.items():
            if key.startswith(sid+"/"):
                result["regret_vs_true_two_query_optimum"]=result["utility_loss"]-optimum
                result["production_optimum_reference"]=optimum
                result["optimum_reference_scope"]="correct-model exact horizon-two planner; independently enumerated oracle checked separately"
    write_lines(output/"episodes.jsonl",episodes);write_lines(output/"model-expectations.jsonl",model_metrics)
    write_json(output/"fixed-policy-selections.json",selections)
    report={"status":"OK","experiment_version":VERSION,"population_kind":"synthetic_enumerated_mechanism_benchmark",
        "scenario_count":len(scenarios),"scenario_model_pairs":len(models),"policy_count":len(POLICIES),"episode_runs":len(episodes),
        "true_worlds":sum(len(s["worlds"]) for s in scenarios.values()),"distinct_return_tuples":sum(len({tuple(sorted(w['observations'].items())) for w in s['worlds']}) for s in scenarios.values()),
        "scenarios":scenario_artifacts,"models":models,"metrics":metrics,"implementation_sha256":implementation,
        "design_note_sha256":DESIGN_SHA256,"artifact_sha256":{name:digest(output/name) for name in ("spec.json","scenarios.json","episodes.jsonl","trajectories.jsonl","model-expectations.jsonl","fixed-policy-selections.json")},
        "new_fitted_models":0,"external_requests":0,"llm_tokens":0,"new_independent_football_matches":0,
        "limitations":["Known synthetic distributions test mechanisms, not real football effectiveness",
            "Wrong-model variants change both forecasting and planning; not an identified explanation of previous learned-model failures",
            "All masses are enumerated; repeated outcomes/trajectories do not create independent samples or confidence intervals",
            "Fixed scenarios and costs chosen before evaluation; not a general superiority theorem"]}
    write_json(output/"report.json",report)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",required=True);parser.add_argument("--design-note",required=True)
    args=parser.parse_args();result=run(args.output,args.design_note)
    print(json.dumps({k:result[k] for k in ("status","scenario_count","scenario_model_pairs","episode_runs","distinct_return_tuples")}))
