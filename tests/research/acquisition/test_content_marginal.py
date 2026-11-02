"""Mechanism fixtures isolate the new wrapper; full real joint replay is audited separately."""
from copy import deepcopy
import json

import pytest

from football_research.acquisition.content_marginal import ContentMarginalGainEstimator, VERSION
from football_research.acquisition.joint_returns import JointReturnPredictor, HYPOTHESIS_KEY
from football_research.acquisition.round9_residual import ResidualEVSICalibrator, context_key
from football_research.acquisition.state import GROUP_FIELDS, sanitize_state
from football_research.acquisition.value import _canonical


ACTIONS = [
    {"action_id":"results_view","group_ids":["elo","recent_form"],"cost_status":"simulated","cost":{"money":.02,"requests":1}},
    {"action_id":"xg_view","group_ids":["xg"],"cost_status":"simulated","cost":{"money":.03,"requests":1}}]
BUDGET = {"limits":{"money":.05,"requests":2},"spent":{}}


def state(acquired=(), value=1.):
    groups={g:{"status":"not_acquired","values":{}} for g in GROUP_FIELDS}
    for action in ACTIONS:
        if action["action_id"] in acquired:
            for g in action["group_ids"]:
                groups[g]={"status":"observed","available_at":"2025-05-01T00:00:00Z","values":{f:value for f in GROUP_FIELDS[g]}}
    return {"match_id":"fixture","prediction_cutoff":"2025-05-01T00:00:00Z","information_mode":"historical_reconstruction",
        "base":{"competition_id":"c","home_team_id":"h","away_team_id":"a","kickoff":"2025-05-02T00:00:00Z"},"groups":groups}


class BranchFixture(JointReturnPredictor):
    """Specified free weights versus three EVSIs, including missing; not a football fit."""
    def __init__(self):
        super().__init__()
        self.model_version="fixed-joint-fixture"
        self.metadata={"training_as_of":"2025-04-01T00:00:00Z","information_mode":"historical_reconstruction"}
        self.weights=(.5,.3,.2)
        self.values=(.01,.04,.11)
        self.calls=[]
    def observed_actions(self,s,**kwargs):
        clean=sanitize_state(s)
        return [a["action_id"] for a in ACTIONS if all(clean["groups"][g]["status"] in {"observed","source_missing"} for g in a["group_ids"])]
    def predict_partial(self,s):
        if HYPOTHESIS_KEY in s: raise ValueError("Runtime hypothetical token")
        sanitize_state(s)
        return {"probabilities":[.5,.3,.2]}
    def enumerate_action(self,s,action,*,hypothetical=False):
        clean=sanitize_state(s)
        self.calls.append((action["action_id"],hypothetical,deepcopy(s)))
        if hypothetical:
            token=next(iter(s[HYPOTHESIS_KEY]["observations"].values()))
            assert all(g["status"]=="not_acquired" for g in clean["groups"].values())
            return {"evsi":dict(zip(("low","high","missing"),self.values))[token]}
        # Any attempt to weight by the current real posterior is a test failure.
        assert not self.observed_actions(s)
        branches=[]
        for token,weight in zip(("low","high","missing"),self.weights):
            nxt=deepcopy(clean)
            nxt[HYPOTHESIS_KEY]={"hypothetical":True,"observations":{action["action_id"]:token}}
            branches.append({"observation":token,"probability":weight,"next_state":nxt})
        return {"evsi":.025,"branches":branches}


def fixture():
    model=BranchFixture();cal=ResidualEVSICalibrator()
    cal.model=model;cal.version="source-residual-fixture"
    cal.records=[{"result_available_at":"2025-04-20T00:00:00Z"}]
    cal.manifest={"dataset_version":"fixture","trained_as_of":"2025-04-30T00:00:00Z","training_as_of":"2025-04-30T00:00:00Z",
        "diagnostics_as_of":"2025-04-30T00:00:00Z","information_mode":"historical_reconstruction","training_match_ids":["past"],
        "training_matches":30,"joint_model_version":model.model_version,"joint_artifact_sha256":"joint-hash",
        "joint_training_as_of":model.metadata["training_as_of"],"metadata":{"supervision_kind":"same_version_heldout_fixed_predictor"}}
    for action in ACTIONS:
        aid=action["action_id"];other=next(a["action_id"] for a in ACTIONS if a["action_id"]!=aid)
        for acquired in ([],[other]):
            for b in range(4):
                cal.cells[_canonical([context_key(aid,acquired),b])]={"support":30,"parent_support":30,"support_matches":30,
                    "effective_support":30.,"time_blocks":4,"residual_mean":0. if b==0 else .2,"standard_error":.05,
                    "support_status":"supported_local","fallback_reason":None,"local_match_ids":[],"parent_match_ids":[],"week_influences":{}}
    return model,cal,ContentMarginalGainEstimator(model,cal,model_sha256="joint-hash",calibrator_sha256="cal-hash")


def test_empty_state_exact_source_values_and_initial_decision():
    model,cal,wrapper=fixture()
    original=cal.estimate_value(state(),ACTIONS)
    new=wrapper.estimate_value(state(),ACTIONS)
    assert [r["expected_gain"] for r in new]==[r["expected_gain"] for r in original]
    old_plan=cal.plan(state(),ACTIONS,BUDGET,lambda_cost={"money":.5})
    plan=wrapper.plan(state(),ACTIONS,BUDGET,lambda_cost={"money":.5})
    assert plan["action_id"]==old_plan["action_id"]
    assert [r["selection_score"] for r in plan["candidates"]]==[r["selection_score"] for r in old_plan["candidates"]]
    assert all(r["branches"][0]["probability"]==1 for r in new)


def test_branch_calibration_precedes_weighting_and_keeps_missing():
    model,cal,wrapper=fixture()
    result=wrapper.estimate_value(state(["results_view"]),[ACTIONS[1]])[0]
    expected=.5*.01+.3*.24+.2*.31
    assert result["expected_gain"]==pytest.approx(expected)
    assert result["marginal_model_evsi"]==pytest.approx(.5*.01+.3*.04+.2*.11)
    wrong=cal.estimate_numbers("xg_view",["results_view"],result["marginal_model_evsi"])["expected_gain"]
    assert abs(result["expected_gain"]-wrong)>.05
    assert [b["observation"] for b in result["branches"]]==["low","high","missing"]
    assert result["model_evsi"] is result["evsi_bin"] is result["lower_gain"] is result["standard_error"] is None


def test_real_content_hidden_payload_and_outcome_do_not_enter_weights_or_value():
    class Poison:
        def __deepcopy__(self,memo): raise AssertionError("Hidden payload copied")
        def __str__(self): raise AssertionError("Hidden payload inspected")
    model,cal,wrapper=fixture()
    a=state(["results_view"],1.);b=state(["results_view"],999.)
    b["outcome"]=Poison();b["groups"]["xg"].update(values=Poison(),reason=Poison())
    one=wrapper.estimate_value(a,[ACTIONS[1]])[0]
    two=wrapper.estimate_value(b,[ACTIONS[1]])[0]
    assert one==two
    for aid,hypothetical,passed in model.calls:
        assert "outcome" not in passed
        assert all(g["status"]=="not_acquired" for g in passed["groups"].values())


def test_positive_mass_unsupported_missing_branch_stops_without_renormalizing():
    model,cal,wrapper=fixture()
    cell=cal.cells[_canonical([context_key("xg_view",["results_view"]),3])]
    cell.update(residual_mean=None,support_status="unsupported",support=0,effective_support=0.)
    result=wrapper.estimate_value(state(["results_view"]),[ACTIONS[1]])[0]
    assert result["expected_gain"] is None
    assert result["unsupported_observations"]==["missing"]
    assert [b["probability"] for b in result["branches"]]==[.5,.3,.2]
    assert result["branch_probability_mass"]==1.
    assert wrapper.plan(state(["results_view"]),ACTIONS,BUDGET,lambda_cost={"money":0})["decision"]=="stop"


def test_source_missing_is_acquired_and_stage_must_be_correct():
    _,_,wrapper=fixture()
    missing=state();missing["groups"]["xg"]={"status":"source_missing"}
    result=wrapper.estimate_value(missing,[ACTIONS[0]])[0]
    assert result["acquired_action_ids"]==["xg_view"] and result["positive_mass_branches"]==3
    with pytest.raises(ValueError,match="unpurchased"):
        wrapper.estimate_value(missing,[ACTIONS[1]])
    partial=state();partial["groups"]["elo"]={"status":"source_missing"}
    with pytest.raises(ValueError,match="bundled"):
        wrapper.plan(partial,ACTIONS,BUDGET)


def test_runtime_token_failure_future_and_mean_only_boundaries():
    _,_,wrapper=fixture()
    bads=[]
    token=state();token[HYPOTHESIS_KEY]={};bads.append(token)
    failed=state();failed["action_history"]=[{"action_id":"xg_view","status":"TIMEOUT"}];bads.append(failed)
    future=state();future["prediction_cutoff"]="2025-04-29T00:00:00Z";bads.append(future)
    for s in bads:
        with pytest.raises(ValueError): wrapper.plan(s,[],{"limits":{"money":0,"requests":0},"spent":{}})
    with pytest.raises(ValueError,match="only mean"):
        wrapper.plan(state(),[],BUDGET,risk_level="conservative")
    with pytest.raises(ValueError,match="Duplicate"):
        wrapper.plan(state(),[ACTIONS[0],ACTIONS[0]],BUDGET)
    with pytest.raises(ValueError):
        wrapper.estimate_value(state(),ACTIONS,prediction={"probabilities":[1.]})
    with pytest.raises(ValueError,match="differs"):
        wrapper.estimate_value(state(),ACTIONS,prediction={"probabilities":[.2,.3,.5]})


def test_no_affordable_second_query_no_imagined_acquisition():
    _,_,wrapper=fixture()
    result=wrapper.plan(state(["results_view"]),ACTIONS,
        {"limits":{"money":.03,"requests":2},"spent":{"money":.02,"requests":1}},lambda_cost={"money":.5})
    assert result["decision"]=="stop"
    assert [r["reason"] for r in result["candidates"]]==["already_attempted","physical_budget_exhausted"]
    assert all("branches" not in r for r in result["candidates"])


def test_serialization_binds_both_source_versions_hashes_and_metadata(tmp_path):
    model,cal,wrapper=fixture();path=tmp_path/"wrapper.json";wrapper.save(path)
    loaded=ContentMarginalGainEstimator.load(path,model=model,model_sha256="joint-hash",calibrator=cal,calibrator_sha256="cal-hash")
    assert loaded.version==wrapper.version
    assert loaded.estimate_value(state(["results_view"]),[ACTIONS[1]])==wrapper.estimate_value(state(["results_view"]),[ACTIONS[1]])
    with pytest.raises(ValueError,match="binding"):
        ContentMarginalGainEstimator.load(path,model=model,model_sha256="joint-hash",calibrator=cal,calibrator_sha256="other")
    raw=json.loads(path.read_text(encoding="utf-8"));assert raw["schema"]==VERSION
    raw["manifest"]["trained_as_of"]="2025-04-29T00:00:00Z"
    path.write_text(json.dumps(raw),encoding="utf-8")
    with pytest.raises(ValueError,match="binding"):
        ContentMarginalGainEstimator.load(path,model=model,model_sha256="joint-hash",calibrator=cal,calibrator_sha256="cal-hash")


def test_branch_probabilities_must_be_complete_and_nonnegative():
    model,_,wrapper=fixture();model.weights=(.5,.3,.1)
    with pytest.raises(ValueError,match="normalized"):
        wrapper.estimate_value(state(["results_view"]),[ACTIONS[1]])
    model.weights=(.5,.6,-.1)
    with pytest.raises(ValueError,match="mass"):
        wrapper.estimate_value(state(["results_view"]),[ACTIONS[1]])
