"""Frozen actual inputs exercise erasure only in the query-value layer."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from football_agent.audit import RunStore
from football_agent.calibrated_joint_bridge import CalibratedJointServices
from football_agent.content_marginal_bridge import (ESTIMATOR_KIND, GATE_KIND, ERASURE_SCOPE, MODEL_AUDIT_SCOPE,
    ContentMarginalServices, ContentMarginalAcquisitionWorkflow, load_content_marginal_services, build_content_marginal_cards)
from football_agent.research_bridge import ResearchServices, research_state
from football_agent.verify_round13_content_marginal import compare_initial_query_card, select_cases
from football_research.acquisition.content_marginal import ContentMarginalGainEstimator
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
    wrapper = ContentMarginalGainEstimator(model, cal, model_sha256=model_sha, calibrator_sha256=cal_sha)
    acceptance = json.loads((AGENT / "acceptance.json").read_text(encoding="utf-8"))
    case = next(c for c in acceptance["checks"] if c["fold"] == 2 and c["strategy"] == "frozen-version-mean")
    run = RunStore(AGENT / "runs").load(case["run_id"])
    event = next(e for e in run["events"] if e["type"] == "value_estimated")
    return {"fold": fold, "model": model, "calibrator": cal, "wrapper": wrapper, "model_path": model_path,
            "cal_path": cal_path, "model_sha": model_sha, "cal_sha": cal_sha,
            "initial": event["snapshot"]["visible_state"], "after": run["visible_state"],
            "catalog": [a for a in event["data"]["candidates"] if a["status"] == "AVAILABLE"],
            "card": json.loads((AGENT / case["decision_card"]["path"]).read_text(encoding="utf-8"))}


def service(inputs):
    return ContentMarginalServices(inputs["model"], inputs["wrapper"], lambda_cost={"money": .5})


def config(tmp_path, inputs):
    value = tmp_path / "wrapper.json"
    inputs["wrapper"].save(value)
    settings = {"partial_predictor": str(inputs["model_path"]), "partial_predictor_kind": "learned-joint-returns-v1",
                "source_calibrator": str(inputs["cal_path"]), "source_calibrator_kind": "residual-evsi-calibrator-v1",
                "value_estimator": value.name, "value_estimator_kind": ESTIMATOR_KIND,
                "gate": {"kind": GATE_KIND, "risk_level": "mean"}, "lambda_cost": {"money": .5}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(settings), encoding="utf-8")
    return path, settings


def test_distinct_wrapper_source_and_model_bindings_without_old_factory_change(tmp_path, inputs):
    path, settings = config(tmp_path, inputs)
    source_path = tmp_path / "same-version-source.json"
    source_path.write_bytes(inputs["cal_path"].read_bytes())
    settings["source_calibrator"] = source_path.name
    path.write_text(json.dumps(settings), encoding="utf-8")
    svc = load_content_marginal_services(path)
    cfg = svc.configuration()
    assert cfg["value_estimator_kind"] == ESTIMATOR_KIND and "calibrated_gate" not in cfg
    assert cfg["content_marginal_gate"]["erasure_scope"] == ERASURE_SCOPE
    assert cfg["reference_source_calibrator"]["sha256"] == inputs["cal_sha"]
    assert cfg["artifact_hashes"]["predictor"] == inputs["model_sha"]
    assert cfg["artifact_hashes"]["source_calibrator"] == inputs["cal_sha"]
    with pytest.raises(ValueError, match="Unrecognized value estimator"):
        ResearchServices.load(path)


@pytest.mark.parametrize("mutation", ["kind", "source_kind", "risk", "horizon", "model_hash", "source_hash", "wrapper_binding"])
def test_loading_rejects_wrong_kinds_gates_and_independently_changed_bytes(tmp_path, inputs, mutation):
    path, settings = config(tmp_path, inputs)
    if mutation == "kind": settings["value_estimator_kind"] = "residual-evsi-calibrator-v1"
    elif mutation == "source_kind": settings["source_calibrator_kind"] = ESTIMATOR_KIND
    elif mutation == "risk": settings["gate"]["risk_level"] = "conservative"
    elif mutation == "horizon": settings["gate"]["horizon"] = 2
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
    with pytest.raises(ValueError): load_content_marginal_services(path)


def test_empty_set_preserves_original_numerical_gains_and_first_decision(inputs):
    old = CalibratedJointServices(inputs["model"], inputs["calibrator"], lambda_cost={"money": .5}, risk_level="mean")
    before = old.estimate(inputs["initial"], inputs["catalog"])
    after = service(inputs).estimate(inputs["initial"], inputs["catalog"])
    for a,b in zip(before,after):
        for key in ("expected_loss_improvement", "net_gain", "cost_in_loss_units"):
            assert a[key] == b[key]
        assert a["research_estimate"]["planned_action"] == b["research_estimate"]["planned_action"]
        assert b["research_estimate"]["model_evsi"] is None and b["standard_error"] is None
        assert len(b["research_estimate"]["branches"]) == 1


def test_content_changes_actual_prediction_but_not_marginal_weights_gains_or_query(inputs):
    svc = service(inputs)
    original, changed = deepcopy(inputs["after"]), deepcopy(inputs["after"])
    for key,value in changed["groups"]["xg"]["values"].items():
        if type(value) in (int,float): changed["groups"]["xg"]["values"][key] = 4. if key.startswith("home_") else .1
    assert svc.predict(original)["probabilities"] != svc.predict(changed)["probabilities"]
    a = svc.estimate(original, inputs["catalog"][:1])[0]["research_estimate"]
    b = svc.estimate(changed, inputs["catalog"][:1])[0]["research_estimate"]
    assert a == b and len(a["branches"]) == 4
    assert any(branch["observation"] == "missing" for branch in a["branches"])
    assert sum(branch["probability"] for branch in a["branches"]) == pytest.approx(1)
    assert a["expected_gain"] == pytest.approx(sum(z["probability"]*z["expected_gain"] for z in a["branches"]))


@pytest.mark.parametrize("mutation", ["hidden", "hypothetical", "failure", "future_fit", "mode", "negative_budget", "overspent_budget"])
def test_erasure_keeps_public_time_failure_and_physical_budget_guards(inputs, mutation):
    state = deepcopy(inputs["initial"])
    if mutation == "hidden": state["base_facts"]["outcome"] = 0
    elif mutation == "hypothetical": state["joint_return_hypothesis"] = {"observations": {"xg_view": "missing"}}
    elif mutation == "failure": state["action_history"] = [{"action_id": "xg_view", "status": "PROVIDER_ERROR"}]
    elif mutation == "future_fit": state["prediction_cutoff"] = "2020-01-01T00:00:00Z"
    elif mutation == "mode": state["information_mode"] = "operational"
    else:
        state["budget"] = {"max_simulated_cost": -.01 if mutation == "negative_budget" else .01,
                           "simulated_cost": 0. if mutation == "negative_budget" else .02, "max_actions": 2, "actions": 0}
    with pytest.raises(ValueError): service(inputs).estimate(state, inputs["catalog"])


def test_zero_lambda_keeps_physical_money_limit(inputs):
    svc = ContentMarginalServices(inputs["model"], inputs["wrapper"], lambda_cost={"money": 0})
    state = deepcopy(inputs["initial"])
    state["budget"] = {"max_simulated_cost": .02, "simulated_cost": 0., "max_actions": 1, "actions": 0}
    values = svc.estimate(state, inputs["catalog"])
    assert values[0]["status"] == "OK" and values[0]["cost_in_loss_units"] == 0
    assert values[1]["status"] == "UNAVAILABLE" and values[1]["reason"] == "physical_budget_exhausted"


class MissingBackend:
    def __init__(self, inputs, status): self.inputs,self.svc,self.status = inputs,service(inputs),status
    def configuration(self): return self.svc.configuration()
    def start(self,*args):
        state = deepcopy(self.inputs["initial"])
        state["quality"].pop("protocol_version",None)
        return state
    def predict(self,state): return self.svc.predict(state)
    def prediction_conditioning(self,state): return self.svc.predictor.observed_categories(research_state(state))
    def actions(self,state): return [self.inputs["catalog"][1]]
    def estimate(self,state,actions): return self.svc.estimate(state,actions)
    def acquire(self,state,action_id,parameters,request_id):
        updated = deepcopy(state)
        updated["quality"]["group_statuses"]["xg"] = "source_missing"
        updated["action_history"].append({"action_id":action_id,"status":self.status})
        return {"status":self.status,"action_id":action_id,"request_id":request_id,"visible_state":updated,
                "cost":{"amount":.03,"status":"simulated","unit":"simulation_credit"}}


@pytest.mark.parametrize("status", ["NO_DATA", "PROVIDER_ERROR"])
def test_actual_workflow_missing_refresh_failure_and_separate_model_only_card(tmp_path, inputs, status):
    flow = ContentMarginalAcquisitionWorkflow(MissingBackend(inputs,status),tmp_path/"runs")
    state = inputs["initial"]
    with pytest.raises(ValueError,match="mean-only"):
        flow.start(state["match_id"],state["prediction_cutoff"],risk_mode="conservative")
    run = flow.start(state["match_id"],state["prediction_cutoff"],budget={"max_actions":1,"max_simulated_cost":.03})
    final = flow.run_to_completion(run["run_id"])
    assert len(final["history"]) == 1 and final["history"][0]["action_id"] == "xg_view"
    updates = [e for e in final["events"] if e["type"] == "prediction_updated"]
    assert len(updates) == (2 if status == "NO_DATA" else 1)
    assert final["state"] == ("STOPPED" if status == "NO_DATA" else "DEFERRED")
    assert flow.store.verify(final["run_id"])["status"] == "OK"
    card = build_content_marginal_cards(final)
    assert card["erasure_scope"] == ERASURE_SCOPE
    first = card["cards"][0]
    assert first["receipt"]["status"] == status
    audit = first["candidates"][0]["model_only_value_audit"]
    assert audit["scope"] == MODEL_AUDIT_SCOPE and audit["counts_as_acquisition"] is False
    assert audit["branches"][0]["hypothetical"] is True
    assert first["candidates"][0]["model_evsi"] is None
    assert "branches" not in first["receipt"]


def test_initial_card_comparison_detects_source_and_first_decision_changes(inputs):
    card = deepcopy(inputs["card"])
    assert compare_initial_query_card(card,inputs["card"],inputs["cal_sha"])["initial_candidates_and_query_equal"]
    with pytest.raises(ValueError): compare_initial_query_card(card,inputs["card"],"0"*64)
    card["cards"][0]["decision"]["selected_action"] = "results_view"
    with pytest.raises(ValueError): compare_initial_query_card(card,inputs["card"],inputs["cal_sha"])


def test_fixed_case_selection_uses_frozen_schedule_not_branch_or_gain_filters(inputs):
    report = json.loads((SOURCE/"report.json").read_text(encoding="utf-8"))
    schedule = [json.loads(s) for s in (SOURCE/"episodes.jsonl").read_text(encoding="utf-8").splitlines()]
    episodes=[]
    for f in report["folds"]:
        f["source_calibrator"],f["content_marginal"] = f["calibrators"]["same_version"],{}
        first=min((r for r in schedule if r["fold"]==f["fold"] and r["strategy"]=="base-only" and r["budget_money"]==.05),key=lambda r:(r["kickoff"],r["match_id"]))
        episodes.append({**first,"strategy":"content-marginal-mean","lambda_money":.5})
    assert len(select_cases(report,episodes,schedule))==5
    episodes[-1]["lambda_money"]=0
    with pytest.raises(ValueError): select_cases(report,episodes,schedule)
