from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import math
from types import SimpleNamespace

import pytest

from football_research.acquisition.gain_prior import GainPriorEstimator, gain_summary, VERSION
from football_research.acquisition.joint_returns import JointReturnPredictor, HYPOTHESIS_KEY
from football_research.acquisition.state import GROUP_FIELDS


ACTIONS = [
    {"action_id":"results_view","group_ids":["elo","recent_form"],"cost_status":"simulated","cost":{"money":.02,"requests":1}},
    {"action_id":"xg_view","group_ids":["xg"],"cost_status":"simulated","cost":{"money":.03,"requests":1}}]
START = datetime(2025,1,10,tzinfo=timezone.utc)


class BoundaryModel(JointReturnPredictor):
    def __init__(self):
        super().__init__()
        self.model_version = "fixed-boundary-fixture"
        self.metadata = {"training_as_of":"2025-01-01T00:00:00+00:00","information_mode":"historical_reconstruction"}
        self.base_predictor = SimpleNamespace(metadata={"training_match_ids":["past"]})
    def predict_partial(self, *args, **kwargs):
        raise AssertionError("Current prediction is not a prior-value input")
    def enumerate_action(self, *args, **kwargs):
        raise AssertionError("EVSI is not a prior-value input")


def state(i=30, acquired=()):
    at = START+timedelta(days=i)
    groups = {g:{"status":"not_acquired","values":{}} for g in GROUP_FIELDS}
    for action in ACTIONS:
        if action["action_id"] in acquired:
            for g in action["group_ids"]:
                groups[g] = {"status":"observed","available_at":at.isoformat(),"values":{f:1. for f in GROUP_FIELDS[g]}}
    return {"match_id":f"m{i}","prediction_cutoff":at.isoformat(),"information_mode":"historical_reconstruction",
        "base":{"kickoff":(at+timedelta(days=1)).isoformat(),"competition_id":"c","home_team_id":"h","away_team_id":"a"},"groups":groups}


def samples(n=14):
    output=[]
    for i in range(n):
        for action in ACTIONS:
            other=next(a["action_id"] for a in ACTIONS if a["action_id"] != action["action_id"])
            for acquired in ([],[other]):
                s=state(i,acquired)
                # Tool action-only means .2/.3; stage means .1/.2 then .3/.4.
                gain=.1 + (.1 if action["action_id"]=="xg_view" else 0) + (.2 if acquired else 0) + i*.01
                output.append({"state":s,"match_id":s["match_id"],"kickoff":s["base"]["kickoff"],
                    "result_available_at":(START+timedelta(days=i+1,hours=2)).isoformat(),"action":action,
                    "acquired_action_ids":acquired,"delta":gain,"model_evsi":999,
                    "supervision_kind":"same_version_heldout_fixed_predictor","oof_predictor_version":"fixed-boundary-fixture",
                    "oof_manifest_ref":"sha256:fixture-hash","oof_train_cutoff":"2025-01-01T00:00:00+00:00"})
    return output


def fit(mode="action_only",n=14):
    return GainPriorEstimator(mode=mode).fit(samples(n),as_of="2025-02-01T00:00:00+00:00",dataset_version="fixture",
        joint_model=BoundaryModel(),joint_artifact_sha256="fixture-hash")


def test_match_weighting_and_stage_mean_do_not_use_evsi():
    tool,stage=fit(),fit("acquired_set")
    assert tool.estimate_numbers("results_view",[])["expected_gain"]==pytest.approx(.265)
    assert tool.estimate_numbers("results_view",["xg_view"])["expected_gain"]==pytest.approx(.265)
    assert stage.estimate_numbers("results_view",[])["expected_gain"]==pytest.approx(.165)
    assert stage.estimate_numbers("results_view",["xg_view"])["expected_gain"]==pytest.approx(.365)
    for estimate in (tool.estimate_numbers("results_view",[]),stage.estimate_numbers("xg_view",[])):
        assert estimate["support"]==14
        assert estimate["model_evsi"] is estimate["evsi_bin"] is estimate["lower_gain"] is None


def test_public_return_identity_probability_and_hidden_payload_invariance():
    for mode in ("action_only","acquired_set"):
        prior=fit(mode); before=state(acquired=["results_view"]); after=deepcopy(before)
        after["base"].update(home_team_id="other",away_team_id="third",competition_id="another")
        for g in ("elo","recent_form"):
            after["groups"][g]["values"]={f:900. for f in GROUP_FIELDS[g]}
        after["groups"]["xg"].update(values={"secret":object()},reason="TIMEOUT",label=2)
        after["outcome"]=2
        a=prior.estimate_value(before,[ACTIONS[1]],prediction={"probabilities":[.9,.05,.05]})[0]
        b=prior.estimate_value(after,[ACTIONS[1]],prediction={"probabilities":[.05,.05,.9]})[0]
        assert a==b


def test_week_cluster_se_counts_matches_after_two_state_average():
    prior=fit()
    cell=next(c for c in prior.cells.values() if c["sample_mean"]==pytest.approx(.265))
    influence={}
    for i in range(14):
        week=(START+timedelta(days=i+1)).isocalendar()[:2]
        influence[week]=influence.get(week,0)+(.2+i*.01-.265)/14
    b=len(influence)
    expected=math.sqrt(b/(b-1)*sum(x*x for x in influence.values()))
    assert cell["standard_error"]==pytest.approx(expected)
    assert cell["support"]==len(cell["match_values"])==14


def test_duplicate_incomplete_and_wrong_version_training_rejected():
    for changed in (samples()+[deepcopy(samples()[0])],samples()[1:]):
        with pytest.raises(ValueError):
            GainPriorEstimator(mode="action_only").fit(changed,as_of="2025-02-01T00:00:00+00:00",dataset_version="x",
                joint_model=BoundaryModel(),joint_artifact_sha256="fixture-hash")
    bad=samples();bad[0]["oof_predictor_version"]="different"
    with pytest.raises(ValueError,match="binding"):
        GainPriorEstimator(mode="action_only").fit(bad,as_of="2025-02-01T00:00:00+00:00",dataset_version="x",
            joint_model=BoundaryModel(),joint_artifact_sha256="fixture-hash")


def test_missing_acquired_and_physical_budget_mean_only():
    prior=fit("acquired_set"); s=state(acquired=["results_view"])
    for g in ("elo","recent_form"):
        s["groups"][g]={"status":"source_missing"}
    result=prior.plan(s,ACTIONS,{"limits":{"money":.05,"requests":2},"spent":{"money":.02,"requests":1}},lambda_cost={"money":.5})
    assert result["action_id"]=="xg_view"
    assert result["candidates"][0]["reason"]=="already_attempted"
    assert result["candidates"][1]["expected_gain"]==pytest.approx(.465)
    assert prior.plan(s,ACTIONS,{"limits":{"money":0,"requests":2},"spent":{}},lambda_cost={"money":.5})["decision"]=="stop"
    with pytest.raises(ValueError,match="only mean"):
        prior.plan(s,[],{"limits":{"money":0,"requests":2},"spent":{}},risk_level="conservative")


def test_runtime_rejects_future_failure_hypothesis_invalid_actions_and_probabilities():
    prior=fit()
    bads=[]
    future=state();future["prediction_cutoff"]="2025-01-30T00:00:00+00:00";bads.append(future)
    failure=state();failure["action_history"]=[{"action_id":"results_view","status":"TIMEOUT"}];bads.append(failure)
    fake=state();fake[HYPOTHESIS_KEY]={};bads.append(fake)
    for s in bads:
        with pytest.raises(ValueError):
            prior.plan(s,[],{"limits":{"money":0,"requests":2},"spent":{}})
    with pytest.raises(ValueError,match="Duplicate"):
        prior.plan(state(),[ACTIONS[0],ACTIONS[0]],{"limits":{"money":0,"requests":2},"spent":{}})
    with pytest.raises(ValueError):
        prior.plan(state(),[{**ACTIONS[0],"available":1}],{"limits":{"money":0,"requests":2},"spent":{}})
    with pytest.raises(ValueError):
        prior.estimate_value(state(),ACTIONS,prediction={"probabilities":[1.]})


def test_unsupported_support_stops_and_one_week_mean_remains_available():
    prior=fit(n=11)
    assert prior.plan(state(),ACTIONS,{"limits":{"money":.05,"requests":2},"spent":{}})["decision"]=="stop"
    assert all(c["support_status"]=="unsupported" for c in prior.plan(state(),ACTIONS,{"limits":{"money":.05,"requests":2},"spent":{}})["candidates"])
    one=gain_summary([{"match_id":str(i),"kickoff":"2025-01-10T00:00:00+00:00","gain":.1} for i in range(12)],min_support=12)
    assert one["standard_error"] is None and one["expected_gain"]==pytest.approx(.1)


def test_serialization_explicit_new_kind_binding_and_tamper(tmp_path):
    prior=fit();path=tmp_path/"prior.json";prior.save(path)
    loaded=GainPriorEstimator.load(path,model=BoundaryModel(),model_sha256="fixture-hash")
    assert loaded.version==prior.version
    assert loaded.plan(state(),ACTIONS,{"limits":{"money":.05,"requests":2},"spent":{}})==prior.plan(state(),ACTIONS,{"limits":{"money":.05,"requests":2},"spent":{}})
    with pytest.raises(ValueError,match="binding"):
        GainPriorEstimator.load(path,model=BoundaryModel(),model_sha256="other")
    raw=json.loads(path.read_text(encoding="utf-8"));assert raw["schema"]==VERSION
    raw["cells"][next(iter(raw["cells"]))]["expected_gain"]+=.01
    path.write_text(json.dumps(raw),encoding="utf-8")
    with pytest.raises(ValueError,match="hash"):
        GainPriorEstimator.load(path,model=BoundaryModel(),model_sha256="fixture-hash")
