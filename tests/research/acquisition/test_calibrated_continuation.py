"""Fixed planning-reward fixture isolates continuation and cost accounting."""
from copy import deepcopy
import json

import pytest

from football_research.acquisition.calibrated_continuation import CalibratedContinuationPlanner, VERSION
from football_research.acquisition.joint_returns import JointReturnPredictor, HYPOTHESIS_KEY
from football_research.acquisition.round9_residual import ResidualEVSICalibrator
from football_research.acquisition.state import GROUP_FIELDS, sanitize_state

ACTIONS=[{"action_id":"results_view","group_ids":["elo","recent_form"],"cost_status":"simulated","cost":{"money":.02,"requests":1}},
         {"action_id":"xg_view","group_ids":["xg"],"cost_status":"simulated","cost":{"money":.03,"requests":1}}]
BUDGET={"limits":{"money":.05,"requests":2},"spent":{}}


def state(acquired=None,value=1.):
    groups={g:{"status":"not_acquired","values":{}} for g in GROUP_FIELDS}
    if acquired:
        action=next(a for a in ACTIONS if a["action_id"]==acquired)
        for g in action["group_ids"]:
            groups[g]={"status":"observed","available_at":"2025-05-01T00:00:00Z","values":{f:value for f in GROUP_FIELDS[g]}}
    return {"match_id":"fixture","prediction_cutoff":"2025-05-01T00:00:00Z","information_mode":"historical_reconstruction",
        "base":{"competition_id":"c","home_team_id":"h","away_team_id":"a","kickoff":"2025-05-02T00:00:00Z"},"groups":groups}


class Model(JointReturnPredictor):
    def __init__(self):
        super().__init__();self.model_version="planning-mechanism-fixture"
        self.metadata={"training_as_of":"2025-04-01T00:00:00Z","information_mode":"historical_reconstruction"}
    def predict_partial(self,s):
        if HYPOTHESIS_KEY in s: raise ValueError("hypothetical at runtime")
        sanitize_state(s);return {"probabilities":[.5,.3,.2]}
    def enumerate_action(self,s,action,*,hypothetical=False):
        clean=sanitize_state(s)
        if hypothetical:
            token=next(iter(s[HYPOTHESIS_KEY]["observations"].values()))
            return {"evsi":.01 if token=="good" else .03}
        observed=[g for g in clean["groups"].values() if g["status"] in {"observed","source_missing"}]
        if observed:
            good=any(v is not None and v>0 for g in observed for v in g["values"].values())
            return {"evsi":.01 if good else .03}
        branches=[]
        for token in ("good","missing"):
            nxt=deepcopy(clean);nxt[HYPOTHESIS_KEY]={"observations":{action["action_id"]:token},"hypothetical":True}
            branches.append({"observation":token,"probability":.5,"next_state":nxt})
        return {"evsi":.02,"branches":branches}


class Calibrator(ResidualEVSICalibrator):
    def __init__(self,model):
        super().__init__();self.model=model;self.version="fixed-reward-fixture"
        self.initial={"results_view":-.005,"xg_view":-.02}
        self.future={"results_view":{.01:-.1,.03:-.1},"xg_view":{.01:.1,.03:-.1}}
        self.records=[{"result_available_at":"2025-04-20T00:00:00Z"}]
        self.manifest={"dataset_version":"fixture","trained_as_of":"2025-04-30T00:00:00Z","training_as_of":"2025-04-30T00:00:00Z",
            "diagnostics_as_of":"2025-04-30T00:00:00Z","information_mode":"historical_reconstruction","training_match_ids":["past"],
            "training_matches":30,"joint_model_version":model.model_version,"joint_artifact_sha256":"joint-hash",
            "joint_training_as_of":model.metadata["training_as_of"],"metadata":{"supervision_kind":"same_version_heldout_fixed_predictor"}}
    def estimate_numbers(self,aid,acquired,evsi,*,risk_level="mean"):
        gain=self.future[aid][evsi] if acquired else self.initial[aid]
        return {"action_id":aid,"acquired_action_ids":list(acquired),"model_evsi":evsi,"evsi_bin":0,
            "expected_gain":gain,"g_hat":gain,"decision_gain":gain,"lower_gain":None,"upper_gain":None,
            "support":30 if gain is not None else 0,"support_matches":30,"parent_support":30,"effective_support":30.,"time_blocks":4,
            "standard_error":.01,"support_status":"supported_local" if gain is not None else "unsupported","fallback_reason":None}


def fixture(mode="adaptive"):
    model=Model();cal=Calibrator(model)
    return model,cal,CalibratedContinuationPlanner(model,cal,mode=mode,model_sha256="joint-hash",calibrator_sha256="cal-hash")


def candidate(plan,aid="results_view"):
    return next(r for r in plan["candidates"] if r["action_id"]==aid)


def test_negative_immediate_positive_plan_selected_and_expected_cost_is_not_charge():
    _,_,planner=fixture()
    p=planner.plan(state(),ACTIONS,BUDGET,lambda_cost={"money":.5});c=candidate(p)
    assert p["action_id"]=="results_view"
    assert c["immediate_net_gain"]==pytest.approx(-.015)
    assert c["continuation_bonus"]==pytest.approx(.0425)
    assert c["mean_net_gain"]==pytest.approx(.0275)
    assert c["expected_gain"]==pytest.approx(.045)
    assert c["cost_penalty"]==pytest.approx(.0175)
    assert c["expected_continuation_cost"]["money"]==pytest.approx(.015)
    assert c["expected_plan_cost_vector"]["requests"]==pytest.approx(1.5)
    assert c["cost_vector"]["money"]==.02
    assert c["gate_reason"]=="positive_total_plan_gain"
    assert c["model_evsi"] is c["standard_error"] is c["lower_gain"] is None


def test_adaptive_commit_gap_and_branch_query_stop():
    model,cal,adaptive=fixture()
    commit=CalibratedContinuationPlanner(model,cal,mode="commit",model_sha256="joint-hash",calibrator_sha256="cal-hash")
    a=candidate(adaptive.plan(state(),ACTIONS,BUDGET,lambda_cost={"money":.5}))
    c=candidate(commit.plan(state(),ACTIONS,BUDGET,lambda_cost={"money":.5}))
    assert a["adaptive_bonus"]==c["adaptive_bonus"]==pytest.approx(.0425)
    assert a["commit_bonus"]==c["commit_bonus"]==0
    assert a["surrogate_option_gap"]==pytest.approx(.0425)
    assert [b["continuation_decision"] for b in a["branches"]]==["acquire","stop"]
    assert all(b["continuation_decision"]=="stop" for b in c["branches"])
    assert commit.plan(state(),ACTIONS,BUDGET,lambda_cost={"money":.5})["decision"]=="stop"


@pytest.mark.parametrize("limits",[{"money":.03,"requests":2},{"money":.05,"requests":1},{"money":.05,"requests":2,"tokens":1}])
def test_each_physical_branch_budget_prevents_bonus(limits):
    _,cal,planner=fixture();cal.initial["results_view"]=.02
    actions=deepcopy(ACTIONS)
    for a in actions:a["cost"]["tokens"]=1
    c=candidate(planner.plan(state(),actions,{"limits":limits,"spent":{}},lambda_cost={"money":.5}))
    assert c["continuation_bonus"]==0 and c["branches"]==[]
    assert c["expected_plan_cost_vector"]["requests"]==1
    assert c["mean_net_gain"]==pytest.approx(.01)


def test_unavailable_or_duplicate_evidence_cannot_create_continuation():
    _,cal,planner=fixture();cal.initial["results_view"]=.02
    for actions in ([ACTIONS[0],{**ACTIONS[1],"available":False}],[{**a,"evidence_id":"same"} for a in ACTIONS]):
        c=candidate(planner.plan(state(),actions,BUDGET,lambda_cost={"money":.5}))
        assert c["branches"]==[] and c["continuation_bonus"]==0


def test_positive_mass_unsupported_disables_all_future_but_immediate_may_run():
    for mode in ("adaptive","commit"):
        _,cal,planner=fixture(mode);cal.initial["results_view"]=.03;cal.future["xg_view"][.03]=None
        p=planner.plan(state(),ACTIONS,BUDGET,lambda_cost={"money":.5});c=candidate(p)
        assert p["action_id"]=="results_view" and c["feasible"]
        assert c["unsupported_continuation_observations"]==["missing"]
        assert [b["probability"] for b in c["branches"]]==[.5,.5]
        assert c["branches"][1]["expected_gain"] is None
        assert c["continuation_bonus"]==c["expected_continuation_gain"]==0
        assert all(b["continuation_decision"]=="stop" for b in c["branches"])
        cal.initial["results_view"]=None
        assert not candidate(planner.plan(state(),ACTIONS,BUDGET,lambda_cost={"money":.5}))["feasible"]


def test_after_actual_return_adaptive_is_conditional_commit_is_content_erased():
    model,cal,adaptive=fixture();commit=CalibratedContinuationPlanner(model,cal,mode="commit",model_sha256="joint-hash",calibrator_sha256="cal-hash")
    budget={"limits":{"money":.05,"requests":2},"spent":{"money":.02,"requests":1}}
    good,bad=state("results_view",1.),state("results_view",-1.)
    assert adaptive.plan(good,ACTIONS,budget,lambda_cost={"money":.5})["action_id"]=="xg_view"
    assert adaptive.plan(bad,ACTIONS,budget,lambda_cost={"money":.5})["decision"]=="stop"
    assert commit.plan(good,ACTIONS,budget,lambda_cost={"money":.5})["decision"]=="stop"
    assert candidate(commit.plan(good,ACTIONS,budget,lambda_cost={"money":.5}),"xg_view")["expected_gain"]==candidate(commit.plan(bad,ACTIONS,budget,lambda_cost={"money":.5}),"xg_view")["expected_gain"]


def test_runtime_missing_failure_token_hidden_payload_and_mean_only():
    _,_,planner=fixture()
    missing=state();missing["groups"]["xg"]={"status":"source_missing"}
    p=planner.plan(missing,ACTIONS,BUDGET)
    assert candidate(p,"xg_view")["reason"]=="already_attempted"
    dirty=state();dirty["label"]=object()
    for g in dirty["groups"].values():g.update(values={"hidden":object()},reason="TIMEOUT")
    assert planner.plan(dirty,ACTIONS,BUDGET)==planner.plan(state(),ACTIONS,BUDGET)
    for field,value in ((HYPOTHESIS_KEY,{}),("action_history",[{"action_id":"xg_view","status":"TIMEOUT"}])):
        s=state();s[field]=value
        with pytest.raises(ValueError):planner.plan(s,[],BUDGET)
    with pytest.raises(ValueError):planner.plan(state(),[],BUDGET,risk_level="conservative")


def test_artifact_binds_mode_and_both_source_versions(tmp_path):
    model,cal,planner=fixture();p=tmp_path/"plan.json";planner.save(p)
    loaded=CalibratedContinuationPlanner.load(p,model=model,model_sha256="joint-hash",calibrator=cal,calibrator_sha256="cal-hash")
    assert loaded.plan(state(),ACTIONS,BUDGET)==planner.plan(state(),ACTIONS,BUDGET)
    with pytest.raises(ValueError):
        CalibratedContinuationPlanner.load(p,model=model,model_sha256="joint-hash",calibrator=cal,calibrator_sha256="other")
    raw=json.loads(p.read_text(encoding="utf-8"));assert raw["schema"]==VERSION
    raw["manifest"]["mode"]="commit";p.write_text(json.dumps(raw),encoding="utf-8")
    with pytest.raises(ValueError):CalibratedContinuationPlanner.load(p,model=model,model_sha256="joint-hash",calibrator=cal,calibrator_sha256="cal-hash")
