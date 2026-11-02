from copy import deepcopy
from datetime import datetime,timedelta,timezone
import json
import math

import pytest

from football_research.acquisition.round9_residual import ResidualEVSICalibrator, residual_summary, evsi_bin
from football_research.acquisition.joint_returns import JointReturnPredictor
from football_research.acquisition.state import GROUP_FIELDS,sanitize_state
from football_research.acquisition.experiment_round9 import split_residual_training


ACTION={"action_id":"results_view","group_ids":["elo","recent_form"],"available":True,"cost_status":"simulated","cost":{"money":.02,"requests":1.}}
XG={"action_id":"xg_view","group_ids":["xg"],"available":True,"cost_status":"simulated","cost":{"money":.03,"requests":1.}}


def public(cutoff="2025-05-01T00:00:00+00:00",mid="future"):
    t=datetime.fromisoformat(cutoff)
    return {"match_id":mid,"prediction_cutoff":cutoff,"information_mode":"historical_reconstruction",
        "base":{"home_team_id":"a","away_team_id":"b","competition_id":"c","kickoff":(t+timedelta(days=1)).isoformat()},
        "groups":{g:{"status":"not_acquired","values":{}} for g in GROUP_FIELDS}}


class MechanismModel(JointReturnPredictor):
    """Fixed EVSI fixture isolates calibrator math; real joint bridge tested separately."""
    def __init__(self):
        super().__init__()
        self.model_version="mechanism-test-only"
        self.metadata={"training_as_of":"2025-04-01T00:00:00+00:00","information_mode":"historical_reconstruction"}
    def predict_partial(self,state):
        if "joint_return_hypothesis" in state:
            raise ValueError("Hypothetical token")
        sanitize_state(state)
        return {"probabilities":[.5,.3,.2]}
    def observed_actions(self,state,**kwargs):
        state=sanitize_state(state)
        return [aid for aid,groups in (("results_view",("elo","recent_form")),("xg_view",("xg",)))
                if all(state["groups"][g]["status"] in {"observed","source_missing"} for g in groups)]
    def enumerate_action(self,state,action,**kwargs):
        self.predict_partial(state)
        return {"evsi":.03}


def samples(n=30):
    result=[]
    for i in range(n):
        cutoff=datetime(2025,1,1,tzinfo=timezone.utc)+timedelta(days=i*2)
        mid=f"m{i}"
        state=public(cutoff.isoformat(),mid)
        value=.03 if i<15 else .07
        gain=-.05 if i<15 else .02
        result.append({"match_id":mid,"state":state,"kickoff":state["base"]["kickoff"],
            "result_available_at":(cutoff+timedelta(days=1,hours=3)).isoformat(),"acquired_action_ids":[],"action":ACTION,
            "delta":gain,"model_evsi":value,"oof_train_cutoff":"2024-12-01T00:00:00+00:00","oof_train_ids":["past"],
            "oof_manifest_ref":"sha256:fixture","oof_predictor_version":"past-joint-fixture"})
    return result


def fit(rows=None):
    return ResidualEVSICalibrator().fit(rows or samples(),as_of="2025-03-15T00:00:00+00:00",diagnostics_as_of="2025-04-01T00:00:00+00:00",
        dataset_version="test",joint_model=MechanismModel(),joint_artifact_sha256="fixture-hash")


def test_fixed_bins_and_tolerance():
    assert [evsi_bin(x) for x in (-1e-12,0.,.019999,.02,.05,.10,1.)] == [0,0,0,1,2,3,3]
    for value in (-.01,float("nan"),float("inf"),True):
        with pytest.raises(ValueError):
            evsi_bin(value)


def test_local_shrinkage_parent_fallback_and_cluster_overlap():
    rows=[{"match_id":str(i),"kickoff":(datetime(2025,1,1,tzinfo=timezone.utc)+timedelta(days=i)).isoformat(),"residual":i/100} for i in range(30)]
    result=residual_summary(rows[:15],rows)
    w=15/35
    assert result["local_weight"] == pytest.approx(w)
    assert result["residual_mean"] == pytest.approx(w*.07+(1-w)*.145)
    influences={}
    for i,row in enumerate(rows):
        week=datetime.fromisoformat(row["kickoff"]).isocalendar()[:2]
        influences[week]=influences.get(week,0)+(1-w)*(row["residual"]-.145)/30+(w*(row["residual"]-.07)/15 if i<15 else 0)
    expected=math.sqrt(len(influences)/(len(influences)-1)*sum(v*v for v in influences.values()))
    assert result["standard_error"] == pytest.approx(expected)
    fallback=residual_summary(rows[:11],rows)
    assert fallback["local_weight"] == 0
    assert fallback["residual_mean"] == pytest.approx(.145)
    assert fallback["support_status"] == "supported_parent"
    assert residual_summary(rows[:11],rows[:11])["residual_mean"] is None


def test_duplicate_matches_do_not_inflate_support_and_conflicts_reject():
    model=fit(samples()+samples())
    assert len(model.records) == 30
    assert model.manifest["training_matches"] == 30
    duplicate=deepcopy(samples()[0]); duplicate["delta"]+=1
    with pytest.raises(ValueError,match="duplicate"):
        fit(samples()+[duplicate])


def test_negative_gain_and_unsupported_stop_even_with_zero_lambda():
    model=fit()
    estimate=model.estimate_value(public(),[ACTION])[0]
    assert estimate["model_evsi"]>0 and estimate["expected_gain"]<0
    plan=model.plan(public(),[ACTION,XG],{"limits":{"money":.05,"requests":2},"spent":{}},lambda_cost={"money":0})
    assert plan["decision"] == "stop"
    assert plan["candidates"][0]["feasible"]
    assert not plan["candidates"][1]["feasible"]


def test_single_week_lcb_unavailable_mean_retained():
    rows=samples(12)
    for row in rows:
        row["kickoff"]="2025-01-02T00:00:00+00:00"
        row["result_available_at"]="2025-01-02T03:00:00+00:00"
        row["state"]=public("2025-01-01T00:00:00+00:00",row["match_id"])
        row["delta"]=.1
    model=fit(rows)
    assert model.estimate_value(public(),[ACTION])[0]["expected_gain"] == pytest.approx(.1)
    conservative=model.estimate_value(public(),[ACTION],risk_level="conservative")[0]
    assert conservative["lower_gain"] is None and conservative["gate_reason"] == "stop_uncertainty_unavailable"


def test_temporal_leakage_and_training_target_overlap_rejected():
    rows=samples()
    rows[0]["oof_train_ids"].append(rows[0]["match_id"])
    with pytest.raises(ValueError,match="OOF target"):
        fit(rows)
    rows=samples(); rows[0]["result_available_at"]="2026-01-01T00:00:00+00:00"
    with pytest.raises(ValueError,match="strict forward"):
        fit(rows)


def test_tail_is_diagnostic_only_and_time_purge():
    train,tail,split=split_residual_training(samples())
    assert not {r["match_id"] for r in train} & {r["match_id"] for r in tail}
    assert all(datetime.fromisoformat(r["result_available_at"])<=datetime.fromisoformat(split["fit_cutoff"]) for r in train)
    kwargs=dict(as_of=split["fit_cutoff"],diagnostics_as_of="2025-04-01T00:00:00+00:00",dataset_version="fixture",
        joint_model=MechanismModel(),joint_artifact_sha256="fixture-hash")
    a=ResidualEVSICalibrator().fit(train,calibration_samples=tail,**kwargs)
    dirty=deepcopy(tail)
    for row in dirty: row["delta"]+=100
    b=ResidualEVSICalibrator().fit(train,calibration_samples=dirty,**kwargs)
    assert a.cells == b.cells and a.records == b.records
    assert a.calibration != b.calibration
    assert all("estimator_version" not in r for r in a.calibration["records"])


def test_missing_action_mask_is_distinct_and_hidden_payload_inert():
    rows=samples()
    for row in rows:
        row["acquired_action_ids"]=["xg_view"]
        row["state"]["groups"]["xg"]={"status":"source_missing","values":{}}
    model=fit(rows)
    missing=public(); missing["groups"]["xg"]={"status":"source_missing","values":{}}
    assert model.estimate_value(missing,[ACTION])[0]["parent_support"] == 30
    assert model.estimate_value(public(),[ACTION])[0]["parent_support"] == 0
    class Poison:
        def __str__(self): raise AssertionError("hidden inspected")
        def __deepcopy__(self,memo): raise AssertionError("hidden copied")
    clean=fit(); dirty=public()
    for group in dirty["groups"].values(): group.update(values=Poison(),reason=Poison())
    dirty["label"]=Poison()
    assert clean.estimate_value(dirty,[ACTION]) == clean.estimate_value(public(),[ACTION])


def test_roundtrip_reference_hash_and_tamper_detection(tmp_path):
    model=fit(); path=tmp_path/"calibrator.json"; model.save(path)
    loaded=ResidualEVSICalibrator.load(path,model=model.model,model_sha256="fixture-hash")
    assert loaded.estimate_value(public(),[ACTION]) == model.estimate_value(public(),[ACTION])
    with pytest.raises(ValueError,match="binding"):
        ResidualEVSICalibrator.load(path,model=model.model,model_sha256="other")
    raw=json.loads(path.read_text()); raw["records"][0]["residual"]+=1; path.write_text(json.dumps(raw))
    with pytest.raises(ValueError,match="hash mismatch"):
        ResidualEVSICalibrator.load(path,model=model.model,model_sha256="fixture-hash")


def test_plan_boundary_validation_with_no_budget_and_prediction_shape():
    model=fit(); budget={"limits":{"money":0,"requests":0},"spent":{}}
    with pytest.raises(ValueError,match="Risk level"):
        model.plan(public(),[],budget,risk_level="arbitrary")
    with pytest.raises(ValueError,match="Boolean"):
        model.plan(public(),[{**ACTION,"available":"false"}],budget)
    with pytest.raises(ValueError,match="Duplicate"):
        model.plan(public(),[ACTION,ACTION],budget)
    for p in ([.5],[float("nan"),.3,.2],[.5,.3,.1]):
        with pytest.raises(ValueError):
            model.estimate_value(public(),[ACTION],prediction={"probabilities":p})
