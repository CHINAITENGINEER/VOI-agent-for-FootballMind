"""Frozen actual inputs exercise erasure only in the query-value layer."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from football_agent.audit import RunStore
from football_agent.calibrated_joint_bridge import CalibratedJointServices
from football_agent.calibrated_continuation_bridge import (ESTIMATOR_KIND, GATE_KIND, GAIN_SEMANTICS, COST_SEMANTICS, MODEL_AUDIT_SCOPE,
    CalibratedContinuationServices, CalibratedContinuationAcquisitionWorkflow, load_calibrated_continuation_services, build_calibrated_continuation_cards)
from football_agent.research_bridge import ResearchServices, research_state
from football_agent.verify_round14_continuation import select_cases, reused_marginal_cards
from football_agent.verify_round12_state_ablation import reused_conditional_cards
from football_research.acquisition.content_marginal import ContentMarginalGainEstimator
from football_research.acquisition.calibrated_continuation import CalibratedContinuationPlanner
from football_research.acquisition.joint_returns import JointReturnPredictor
from football_research.acquisition.round9_residual import ResidualEVSICalibrator


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "var/parallel/baselines/round11/frozen-version-v1"
AGENT = ROOT / "var/parallel/agent/round11/frozen-version-bridge-v1"


@pytest.fixture(scope="module")
def inputs():
    if not (SOURCE / "report.json").exists() or not (AGENT / "acceptance.json").exists():
        pytest.skip("Frozen round-eleven fixtures unavailable in this checkout")
    report = json.loads((SOURCE / "report.json").read_text(encoding="utf-8"))
    fold = report["folds"][0]
    model_path = SOURCE / fold["joint_model"]["path"]
    cal_path = SOURCE / fold["calibrators"]["same_version"]["path"]
    model_sha, cal_sha = sha256(model_path.read_bytes()).hexdigest(), sha256(cal_path.read_bytes()).hexdigest()
    model = JointReturnPredictor.load(model_path)
    cal = ResidualEVSICalibrator.load(cal_path, model=model, model_sha256=model_sha)
    wrapper = CalibratedContinuationPlanner(model, cal, mode="adaptive", model_sha256=model_sha, calibrator_sha256=cal_sha)
    acceptance = json.loads((AGENT / "acceptance.json").read_text(encoding="utf-8"))
    case = next(c for c in acceptance["checks"] if c["fold"] == 2 and c["strategy"] == "frozen-version-mean")
    run = RunStore(AGENT / "runs").load(case["run_id"])
    event = next(e for e in run["events"] if e["type"] == "value_estimated")
    return {"fold": fold, "model": model, "calibrator": cal, "wrapper": wrapper, "model_path": model_path,
            "cal_path": cal_path, "model_sha": model_sha, "cal_sha": cal_sha,
            "initial": event["snapshot"]["visible_state"], "after": run["visible_state"],
            "catalog": [a for a in event["data"]["candidates"] if a["status"] == "AVAILABLE"],
            "card": json.loads((AGENT / case["decision_card"]["path"]).read_text(encoding="utf-8"))}


def service(inputs,mode="adaptive"):
    planner=CalibratedContinuationPlanner(inputs["model"],inputs["calibrator"],mode=mode,model_sha256=inputs["model_sha"],calibrator_sha256=inputs["cal_sha"])
    return CalibratedContinuationServices(inputs["model"],planner,lambda_cost={"money":.5})


def config(tmp_path, inputs):
    value = tmp_path / "wrapper.json"
    inputs["wrapper"].save(value)
    settings = {"partial_predictor": str(inputs["model_path"]), "partial_predictor_kind": "learned-joint-returns-v1",
                "source_calibrator": str(inputs["cal_path"]), "source_calibrator_kind": "residual-evsi-calibrator-v1",
                "value_estimator": value.name, "value_estimator_kind": ESTIMATOR_KIND,
                "gate": {"kind": GATE_KIND, "risk_level": "mean", "mode": "adaptive"}, "lambda_cost": {"money": .5}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(settings), encoding="utf-8")
    return path, settings


def test_distinct_wrapper_source_and_model_bindings_without_old_factory_change(tmp_path, inputs):
    path, settings = config(tmp_path, inputs)
    source_path = tmp_path / "same-version-source.json"
    source_path.write_bytes(inputs["cal_path"].read_bytes())
    settings["source_calibrator"] = source_path.name
    path.write_text(json.dumps(settings), encoding="utf-8")
    svc = load_calibrated_continuation_services(path)
    cfg = svc.configuration()
    assert cfg["value_estimator_kind"] == ESTIMATOR_KIND and "calibrated_gate" not in cfg
    assert cfg["continuation_gate"]["horizon"] == 2 and cfg["continuation_gate"]["mode"] == "adaptive"
    assert cfg["reference_source_calibrator"]["sha256"] == inputs["cal_sha"]
    assert cfg["artifact_hashes"]["predictor"] == inputs["model_sha"]
    assert cfg["artifact_hashes"]["source_calibrator"] == inputs["cal_sha"]
    with pytest.raises(ValueError, match="Unrecognized value estimator"):
        ResearchServices.load(path)


@pytest.mark.parametrize("mutation", ["kind", "source_kind", "risk", "horizon", "mode", "model_hash", "source_hash", "wrapper_binding"])
def test_loading_rejects_wrong_kinds_gates_and_independently_changed_bytes(tmp_path, inputs, mutation):
    path, settings = config(tmp_path, inputs)
    if mutation == "kind": settings["value_estimator_kind"] = "residual-evsi-calibrator-v1"
    elif mutation == "source_kind": settings["source_calibrator_kind"] = ESTIMATOR_KIND
    elif mutation == "risk": settings["gate"]["risk_level"] = "conservative"
    elif mutation == "horizon": settings["gate"]["horizon"] = 1
    elif mutation == "mode": settings["gate"]["mode"] = "commit"
    elif mutation in {"model_hash", "source_hash"}:
        key = "partial_predictor" if mutation == "model_hash" else "source_calibrator"
        changed = tmp_path / (key + ".json")
        changed.write_text(Path(settings[key]).read_text(encoding="utf-8") + "\n", encoding="utf-8")
        settings[key] = changed.name
    else:
        changed = tmp_path / settings["value_estimator"]
        raw = json.loads(changed.read_text(encoding="utf-8"))
        raw["manifest"]["source_calibrator_sha256"] = "0"*64
        changed.write_text(json.dumps(raw), encoding="utf-8")
    path.write_text(json.dumps(settings), encoding="utf-8")
    with pytest.raises(ValueError): load_calibrated_continuation_services(path)


@pytest.mark.parametrize("mode",["adaptive","commit"])
@pytest.mark.parametrize("budget",[0,.02,.03,.05])
def test_predeclared_budgets_totals_and_low_budget_one_step_equivalence(inputs,mode,budget):
    state=deepcopy(inputs["initial"])
    state["budget"]={"max_simulated_cost":budget,"simulated_cost":0.,"max_actions":2,"actions":0}
    original=CalibratedJointServices(inputs["model"],inputs["calibrator"],lambda_cost={"money":.5},risk_level="mean")
    old={x["action_id"]:x for x in original.estimate(state,inputs["catalog"])}
    for value in service(inputs,mode).estimate(state,inputs["catalog"]):
        native=value["research_estimate"]
        if value["status"] != "OK":
            assert old[value["action_id"]]["status"] == "UNAVAILABLE"
            continue
        assert native["expected_gain"] == pytest.approx(native["immediate_expected_gain"]+native["expected_continuation_gain"])
        assert native["cost_penalty"] == pytest.approx(native["immediate_cost_penalty"]+native["expected_continuation_cost_penalty"])
        assert value["net_gain"] == pytest.approx(native["mean_net_gain"])
        assert native["model_evsi"] is None and native["standard_error"] is None
        if budget < .05:
            assert native["continuation_bonus"]==0 and native["branches"]==[]
            assert value["expected_loss_improvement"]==old[value["action_id"]]["expected_loss_improvement"]
            assert value["net_gain"]==old[value["action_id"]]["net_gain"]
        for branch in native["branches"]:
            assert branch["physical_budget"]["spent_after_first"]["money"]+branch["cost_vector"]["money"] <= budget+1e-12
            assert branch["physical_budget"]["spent_after_first"]["requests"]+branch["cost_vector"]["requests"] <= 2


@pytest.mark.parametrize("mutation",["hidden","hypothetical","failure","future_fit","mode","negative_budget","overspent_budget"])
def test_runtime_rejects_hidden_hypothetical_failed_future_and_invalid_budget(inputs,mutation):
    state=deepcopy(inputs["initial"])
    if mutation=="hidden":state["base_facts"]["outcome"]=0
    elif mutation=="hypothetical":state["joint_return_hypothesis"]={"observations":{"xg_view":"missing"}}
    elif mutation=="failure":state["action_history"]=[{"action_id":"xg_view","status":"PROVIDER_ERROR"}]
    elif mutation=="future_fit":state["prediction_cutoff"]="2020-01-01T00:00:00Z"
    elif mutation=="mode":state["information_mode"]="operational"
    else:state["budget"]={"max_simulated_cost":-.01 if mutation=="negative_budget" else .01,
                          "simulated_cost":0. if mutation=="negative_budget" else .02,"max_actions":2,"actions":0}
    with pytest.raises(ValueError):service(inputs).estimate(state,inputs["catalog"])


@pytest.mark.parametrize("mode",["adaptive","commit"])
def test_after_actual_return_uses_declared_conditional_or_marginal_rule(inputs,mode):
    state=deepcopy(inputs["after"])
    changed=deepcopy(state)
    for key,value in changed["groups"]["xg"]["values"].items():
        if type(value) in (int,float):changed["groups"]["xg"]["values"][key]=4. if key.startswith("home_") else .1
    svc=service(inputs,mode)
    assert svc.predict(state)["probabilities"]!=svc.predict(changed)["probabilities"]
    old=CalibratedJointServices(inputs["model"],inputs["calibrator"],lambda_cost={"money":.5},risk_level="mean")
    marginal=ContentMarginalGainEstimator(inputs["model"],inputs["calibrator"],model_sha256=inputs["model_sha"],calibrator_sha256=inputs["cal_sha"])
    values=[]
    for current in (state,changed):
        native=svc.estimate(current,inputs["catalog"][:1])[0]["research_estimate"]
        if mode=="adaptive":expected=old.estimate(current,inputs["catalog"][:1])[0]["expected_loss_improvement"]
        else:expected=marginal.estimate_value(research_state(current),[{"action_id":"results_view","cost":inputs["catalog"][0]["cost_vector"],"cost_status":"simulated"}])[0]["expected_gain"]
        assert native["expected_gain"]==expected and native["continuation_bonus"]==0
        values.append(native)
    if mode=="commit":assert values[0]==values[1]


def test_zero_lambda_keeps_requests_and_money_as_physical_limits(inputs):
    svc=service(inputs)
    svc.lambda_cost={u:0. for u in svc.lambda_cost}
    state=deepcopy(inputs["initial"])
    state["budget"]={"max_simulated_cost":.05,"simulated_cost":0.,"max_actions":1,"actions":0}
    for value in svc.estimate(state,inputs["catalog"]):
        assert value["cost_in_loss_units"]==0
        assert value["research_estimate"]["continuation_bonus"]==0
    state["budget"]["max_actions"]=2
    state["budget"]["max_simulated_cost"]=.02
    assert svc.estimate(state,inputs["catalog"])[1]["status"]=="UNAVAILABLE"


@pytest.mark.parametrize("mode",["adaptive","commit"])
def test_unsupported_future_disables_bonus_but_preserves_supported_first_query(inputs,monkeypatch,mode):
    original=inputs["calibrator"].estimate_numbers
    def constructed_support(aid,acquired,evsi,*,risk_level):
        value=original(aid,acquired,evsi,risk_level=risk_level)
        gain=None if acquired else .1
        value.update(expected_gain=gain,g_hat=gain,decision_gain=gain)
        return value
    monkeypatch.setattr(inputs["calibrator"],"estimate_numbers",constructed_support)
    values=service(inputs,mode).estimate(inputs["initial"],inputs["catalog"])
    for value in values:
        native=value["research_estimate"]
        assert value["status"]=="OK" and value["net_gain"]>0
        assert native["continuation_bonus"]==native["expected_continuation_gain"]==0
        assert native["expected_gain"]==native["immediate_expected_gain"]==.1
        assert native["branches"] and sum(z["probability"] for z in native["branches"])==pytest.approx(1)
        assert any(z["observation"]=="missing" for z in native["branches"])
        assert all(z["expected_gain"] is None and z["continuation_decision"]=="stop" for z in native["branches"])


class ScriptedReturnBackend:
    """Constructed receipt test, not an extra historical policy episode."""
    def __init__(self,inputs,svc,status="NO_DATA"):self.inputs,self.svc,self.status=inputs,svc,status
    def configuration(self):return self.svc.configuration()
    def start(self,*args):
        state=deepcopy(self.inputs["initial"])
        state["quality"].pop("protocol_version",None)
        return state
    def predict(self,state):return self.svc.predict(state)
    def prediction_conditioning(self,state):return self.svc.predictor.observed_categories(research_state(state))
    def actions(self,state):
        used={x["action_id"] for x in state["action_history"]}
        return [a for a in self.inputs["catalog"] if a["action_id"] not in used]
    def estimate(self,state,actions):return self.svc.estimate(state,actions)
    def acquire(self,state,action_id,parameters,request_id):
        action=next(a for a in self.inputs["catalog"] if a["action_id"]==action_id)
        updated=deepcopy(state)
        for group in action["group_ids"]:updated["quality"]["group_statuses"][group]="source_missing"
        updated["action_history"].append({"action_id":action_id,"status":self.status})
        return {"status":self.status,"action_id":action_id,"request_id":request_id,"visible_state":updated,
                "cost":{"amount":action["cost_vector"]["money"],"status":"simulated","unit":"simulation_credit"}}


@pytest.mark.parametrize("mode",["adaptive","commit"])
@pytest.mark.parametrize("status",["NO_DATA","PROVIDER_ERROR"])
def test_negative_immediate_positive_plan_executes_and_charges_only_real_requests(tmp_path,inputs,monkeypatch,mode,status):
    original=inputs["calibrator"].estimate_numbers
    def constructed_gain(aid,acquired,evsi,*,risk_level):
        estimate=original(aid,acquired,evsi,risk_level=risk_level)
        gain=-.005 if not acquired else .1
        estimate.update(expected_gain=gain,g_hat=gain,decision_gain=gain,gate_reason="constructed_test_gain")
        return estimate
    monkeypatch.setattr(inputs["calibrator"],"estimate_numbers",constructed_gain)
    svc=service(inputs,mode)
    values=svc.estimate(inputs["initial"],inputs["catalog"])
    for value in values:
        assert value["research_estimate"]["immediate_net_gain"] < 0 < value["net_gain"]
        assert value["cost_in_loss_units"]==pytest.approx(.025)
    flow=CalibratedContinuationAcquisitionWorkflow(ScriptedReturnBackend(inputs,svc,status),tmp_path/"runs")
    state=inputs["initial"]
    with pytest.raises(ValueError,match="mean-only"):
        flow.start(state["match_id"],state["prediction_cutoff"],risk_mode="conservative")
    run=flow.start(state["match_id"],state["prediction_cutoff"],budget={"max_simulated_cost":.05,"max_actions":2})
    final=flow.run_to_completion(run["run_id"])
    expected_count=2 if status=="NO_DATA" else 1
    assert len(final["history"])==expected_count and final["history"][0]["action_id"]=="results_view"
    assert final["budget"]["simulated_cost"]==pytest.approx(.05 if status=="NO_DATA" else .02)
    assert len([e for e in final["events"] if e["type"]=="prediction_updated"])==(3 if status=="NO_DATA" else 1)
    assert final["state"]==("STOPPED" if status=="NO_DATA" else "DEFERRED")
    assert flow.store.verify(final["run_id"])["status"]=="OK"
    cards=build_calibrated_continuation_cards(final)
    first=cards["cards"][0]
    candidate=next(c for c in first["candidates"] if c["action_id"]=="results_view")
    assert candidate["plan_value"]["immediate_net_gain"]<0<candidate["selection_score"]
    assert candidate["gain_semantics"]==GAIN_SEMANTICS and candidate["cost_semantics"]==COST_SEMANTICS
    assert candidate["plan_value"]["expected_plan_cost_vector"]["money"]==pytest.approx(.05)
    assert first["receipt"]["charged_cost"]["amount"]==.02 and "branches" not in first["receipt"]
    assert candidate["model_only_value_audit"]["counts_as_acquisition"] is False
    assert all(b["hypothetical"] for b in candidate["model_only_value_audit"]["branches"])


def test_fixed_ten_cases_and_two_sets_of_five_frozen_references(inputs):
    report=json.loads((SOURCE/"report.json").read_text(encoding="utf-8"))
    schedule=[json.loads(s) for s in (SOURCE/"episodes.jsonl").read_text(encoding="utf-8").splitlines()]
    episodes=[]
    for f in report["folds"]:
        f["source_calibrator"],f["planners"]=f["calibrators"]["same_version"],{"adaptive":{},"commit":{}}
        first=min((r for r in schedule if r["fold"]==f["fold"] and r["strategy"]=="base-only" and r["budget_money"]==.05),key=lambda r:(r["kickoff"],r["match_id"]))
        for mode in ("adaptive","commit"):episodes.append({**first,"strategy":f"calibrated-{mode}-two-step","lambda_money":.5})
    cases=select_cases(report,episodes,schedule)
    assert len(cases)==10
    conditional=reused_conditional_cards(AGENT,cases)
    marginal=reused_marginal_cards(ROOT/"var/parallel/agent/round13/content-marginal-bridge-v1",cases)
    assert len(conditional["cards"])==len(marginal["cards"])==5
    assert all(r["execution_reused_not_rerun"] for r in conditional["cards"]+marginal["cards"])
    episodes[-1]["lambda_money"]=0
    with pytest.raises(ValueError):select_cases(report,episodes,schedule)


