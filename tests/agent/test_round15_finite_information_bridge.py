"""Mechanism and boundary tests on predeclared synthetic populations."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from uuid import uuid4

import pytest

from football_agent.finite_information_bridge import (VERSION,MODE,CUTOFF,FiniteObservationEnvironment,
    FiniteInformationWorkflow,make_finite_backend,build_finite_cards)
from football_research.acquisition.finite_information import build_scenarios,FiniteJointModel
from football_agent.verify_round15_finite import select_patterns,COUNTS


def configured(tmp_path,sid,returns=None,horizon=2):
    scenario=build_scenarios()[sid]
    model=FiniteJointModel(scenario["models"]["correct"])
    model_path=tmp_path/"model.json";model.save(model_path)
    spec_path=tmp_path/"scenario.json";spec_path.write_text(json.dumps(scenario),encoding="utf-8")
    config={"schema":VERSION,"mode":MODE,"case_id":str(uuid4()),"scenario_id":sid,"prediction_cutoff":CUTOFF,
            "actions":scenario["actions"],"budget_money":scenario["budget_money"],"request_budget":2,"lambda_money":1.,
            "horizon":horizon,"model_version":model.model_version,"model_path":model_path.name,"scenario_spec_path":spec_path.name,
            "model_sha256":sha256(model_path.read_bytes()).hexdigest(),"scenario_spec_sha256":sha256(spec_path.read_bytes()).hexdigest()}
    path=tmp_path/"backend.json";path.write_text(json.dumps(config),encoding="utf-8")
    env=FiniteObservationEnvironment(categories=model.spec["categories"],returns=returns or scenario["worlds"][0]["observations"])
    return make_finite_backend(path,env),path,env


@pytest.mark.parametrize("sid,returns,actions",[
    ("gated_stop",{"context":"0","signal_a":"2"},["context","signal_a"]),
    ("gated_stop",{"context":"1","signal_a":"2"},["context"]),
    ("gated_route",{"context":"0","signal_a":"2","signal_b":"1"},["context","signal_a"]),
    ("gated_route",{"context":"1","signal_a":"2","signal_b":"1"},["context","signal_b"]),
    ("complementary",{"signal_a":"1","signal_b":"2"},["signal_a","signal_b"]),
    ("redundant",{"signal_a":"2","signal_b":"2"},["signal_a"]),
    ("irrelevant",{"signal_a":"0","signal_b":"1"},[])])
def test_durable_actual_routes_stops_receipts_and_cards(tmp_path,sid,returns,actions):
    backend,_,_=configured(tmp_path,sid,returns)
    flow=FiniteInformationWorkflow(backend,tmp_path/"runs")
    run=flow.start(backend.case_id,CUTOFF)
    while not run["history"] and run["state"] not in {"STOPPED","DEFERRED"}:
        run=flow.step(run["run_id"])
    # Reopen after the first real reveal (or initial STOP), preserving the same
    # private environment ledger. Process-restart recovery is not claimed.
    flow=FiniteInformationWorkflow(backend,tmp_path/"runs")
    final=flow.run_to_completion(run["run_id"])
    assert final["state"]=="STOPPED" and [h["action_id"] for h in final["history"]]==actions
    assert final["budget"]["actions"]==len(actions) and final["budget"]["llm_calls"]==0
    costs={a["action_id"]:a["cost"]["money"] for a in backend.catalog}
    assert final["budget"]["simulated_cost"]==pytest.approx(sum(costs[a] for a in actions))
    assert flow.store.verify(final["run_id"])["status"]=="OK"
    card=build_finite_cards(final)
    assert set(card["final_probabilities"])=={"0","1","2"}
    assert card["cards"][0]["current"]["acquired_tokens"]=={}
    revealed={}
    for event in final["events"]:
        if event["type"]=="acquisition_result":
            aid=event["data"]["action_id"];revealed[aid]=returns[aid]
        visible=event["snapshot"].get("visible_state")
        if visible:
            assert {a:g["values"]["category"] for a,g in visible["groups"].items()}==revealed
    for c in card["cards"]:
        if c["receipt"]:
            assert c["receipt"]["observation"]==returns[c["receipt"]["action_id"]]
            assert c["receipt"]["return_kind"]=="actual_synthetic_categorical_return"
            assert c["receipt"]["prediction_after_return"] is not None
        for candidate in c["candidates"]:
            assert all(b["hypothetical"] for b in candidate["plan_value"].get("branches",[]))


@pytest.mark.parametrize("sid",["gated_stop","gated_route","complementary"])
def test_first_context_or_complementary_query_has_zero_immediate_information_but_positive_plan(tmp_path,sid):
    backend,_,_=configured(tmp_path,sid)
    state=backend.start(backend.case_id,CUTOFF)
    values=backend.estimate(state,backend.actions(state))
    best=max(values,key=lambda v:v["net_gain"])
    native=best["research_estimate"]
    assert native["immediate_evsi"]==pytest.approx(0,abs=1e-10)
    assert native["immediate_evsi"]-native["immediate_penalty"]<0<best["net_gain"]
    assert best["cost_in_loss_units"]==native["expected_plan_penalty"]


@pytest.mark.parametrize("mutation",["outcome","world","hypothetical","extra_quality","unacquired","history","budget","request_cap"])
def test_runtime_rejects_hidden_or_forged_evidence_and_budget(tmp_path,mutation):
    backend,_,_=configured(tmp_path,"gated_route")
    state=backend.start(backend.case_id,CUTOFF)
    if mutation=="outcome":state["outcome"]=0
    elif mutation=="world":state["base_facts"]["world_id"]="current"
    elif mutation=="hypothetical":state["joint_return_hypothesis"]={"hypothetical":True,"observations":{}}
    elif mutation=="extra_quality":state["quality"]["hypothetical"]=True
    elif mutation=="unacquired":
        state["groups"]["context"]={"status":"observed","values":{"category":"0"}}
        state["mask"]["context"]=True
        state["state_version"]=1
        state["evidence_versions"]["context"]=backend.model.model_version
        state["action_history"]=[{"action_id":"context","status":"observed"}]
    elif mutation=="history":state["action_history"]=[{"action_id":"context","status":"FAILED"}]
    elif mutation=="budget":state["budget"]["simulated_cost"]=-1
    else:state["budget"]["max_actions"]=3
    with pytest.raises(ValueError):backend.predict(state)


def test_one_shot_environment_and_no_outcome_constructor(tmp_path):
    backend,_,env=configured(tmp_path,"redundant")
    state=backend.start(backend.case_id,CUTOFF)
    receipt=backend.acquire(state,"signal_a",{},"request-1")
    with pytest.raises(ValueError):backend.acquire(receipt["visible_state"],"signal_a",{},"request-2")
    with pytest.raises(ValueError):env.reveal("signal_b","request-1")
    with pytest.raises(ValueError):FiniteObservationEnvironment(categories={"a":["0"]},returns={"a":"unsupported"})
    with pytest.raises(TypeError):FiniteObservationEnvironment(categories={"a":["0"]},returns={"a":"0"},outcome=0)


def test_inconsistent_joint_return_is_explicitly_rejected_without_default_posterior(tmp_path):
    backend,_,_=configured(tmp_path,"redundant",{"signal_a":"0","signal_b":"1"})
    state=backend.start(backend.case_id,CUTOFF)
    first=backend.acquire(state,"signal_a",{},"first")
    # Deliberately invoke an unsupported joint return outside the valid suite.
    # The adaptive policy would never buy this redundant second observation.
    with pytest.raises(ValueError):backend.acquire(first["visible_state"],"signal_b",{},"second")


@pytest.mark.parametrize("mutation",["model_bytes","spec_bytes","catalog","mode","model_version"])
def test_artifact_and_configuration_bindings(tmp_path,mutation):
    _,path,env=configured(tmp_path,"irrelevant")
    config=json.loads(path.read_text(encoding="utf-8"))
    if mutation in {"model_bytes","spec_bytes"}:
        target=tmp_path/config["model_path" if mutation=="model_bytes" else "scenario_spec_path"]
        target.write_text(target.read_text(encoding="utf-8")+"\n",encoding="utf-8")
    elif mutation=="catalog":config["actions"][0]["cost"]["money"]=0
    elif mutation=="mode":config["mode"]="historical_reconstruction"
    else:config["model_version"]="wrong"
    path.write_text(json.dumps(config),encoding="utf-8")
    with pytest.raises(ValueError):make_finite_backend(path,env)


def test_request_budget_one_prevents_complementary_plan_and_mean_only(tmp_path):
    backend,_,_=configured(tmp_path,"complementary")
    flow=FiniteInformationWorkflow(backend,tmp_path/"runs")
    with pytest.raises(ValueError):flow.start(backend.case_id,CUTOFF,risk_mode="conservative")
    run=flow.start(backend.case_id,CUTOFF,budget={"max_actions":1,"max_simulated_cost":.24})
    final=flow.run_to_completion(run["run_id"])
    assert final["state"]=="STOPPED" and final["history"]==[]


def test_all_forty_patterns_selected_without_using_outcomes_or_query_effects():
    counts={}
    for sid,scenario in build_scenarios().items():
        patterns=select_patterns(scenario);counts[sid]=len(patterns)
        changed=deepcopy(scenario)
        for w in changed["worlds"]:w["outcome"]=(w["outcome"]+1)%3
        assert select_patterns(changed)==patterns
        assert len({tuple(sorted(obs.items())) for obs,_ in patterns})==len(patterns)
        assert sum(len(ids) for _,ids in patterns)==len(scenario["worlds"])
    assert counts==COUNTS and sum(counts.values())==40
