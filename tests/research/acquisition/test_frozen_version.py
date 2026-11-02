from copy import deepcopy
from datetime import datetime,timedelta,timezone
from hashlib import sha256
import json
from types import SimpleNamespace

import pytest

from football_research.acquisition.frozen_version import (select_matched_window,matched_supervision,validate_samples,
    fit_calibrator,index_transitions,LEGAL_TRANSITIONS)
from football_research.acquisition.experiment import losses,outcome,state_after
from football_research.acquisition.joint_returns import JointReturnPredictor
from football_research.acquisition.round9_residual import ResidualEVSICalibrator
from football_research.acquisition.state import GROUP_FIELDS,sanitize_state


ACTIONS=[{"action_id":"results_view","group_ids":["elo","recent_form"],"cost_status":"simulated","cost":{"money":.02,"requests":1}},
         {"action_id":"xg_view","group_ids":["xg"],"cost_status":"simulated","cost":{"money":.03,"requests":1}}]


class MechanismModel(JointReturnPredictor):
    def __init__(self):
        super().__init__()
        self.model_version="fixed-mechanism-test"
        self.metadata={"training_as_of":"2025-01-03T00:00:00+00:00","training_match_ids":["m0"],"information_mode":"historical_reconstruction"}
        self.base_predictor=SimpleNamespace(metadata={"training_match_ids":["m0"]},model_version="base-fixture")
    def observed_actions(self,state,**kwargs):
        s=sanitize_state(state)
        return [a["action_id"] for a in ACTIONS if all(s["groups"][g]["status"] in {"observed","source_missing"} for g in a["group_ids"])]
    def predict_partial(self,state):
        acquired=self.observed_actions(state)
        p=[.4,.3,.3] if not acquired else [.5,.25,.25] if len(acquired)==1 else [.6,.2,.2]
        return {"probabilities":p}
    def enumerate_action(self,state,action,**kwargs):
        return {"evsi":.03 if not self.observed_actions(state) else .06}


def rows():
    start=datetime(2025,1,1,tzinfo=timezone.utc)
    output=[]
    for i in range(14):
        kickoff=start+timedelta(days=i*2)
        cutoff=kickoff-timedelta(days=1)
        row={"match_id":f"m{i}","kickoff":kickoff.isoformat(),"prediction_cutoff":cutoff.isoformat(),
            "result_available_at":(kickoff+timedelta(hours=3)).isoformat(),"features_available_at":cutoff.isoformat(),"fixture_available_at":cutoff.isoformat(),
            "competition_id":"c","home_team_id":"a","away_team_id":"b","feature_version":"test","information_mode":"historical_reconstruction",
            "home_goals":1,"away_goals":0,"dataset_version":"fixture"}
        row.update({f:1. for names in GROUP_FIELDS.values() for f in names})
        output.append(row)
    return output


def source_samples(source_rows):
    samples=[]
    for row in source_rows:
        for acquired,aid in sorted(LEGAL_TRANSITIONS):
            action=next(a for a in ACTIONS if a["action_id"]==aid)
            before,after=[.45,.3,.25],[.48,.3,.22]
            gain=losses(before,outcome(row))["log_loss"]-losses(after,outcome(row))["log_loss"]
            samples.append({"match_id":row["match_id"],"kickoff":row["kickoff"],"result_available_at":row["result_available_at"],
                "state":state_after(row,ACTIONS,acquired),"action":action,"acquired_action_ids":list(acquired),"prediction_before":before,"prediction_after":after,
                "model_evsi":.04,"delta":gain,"residual":gain-.04,"oof_train_cutoff":"2024-12-01T00:00:00+00:00",
                "oof_train_ids":["past"],"oof_manifest_ref":"sha256:mixed","oof_predictor_version":"mixed-other"})
    return samples


def target():
    return {"training_match_ids":[f"m{i}" for i in range(9)],"validation_match_ids":["m10","m11"],"test_match_ids":["m12","m13"],
        "split":{"train_cutoff":"2025-01-20T00:00:00+00:00"}}


def test_common_ids_inherit_original_train_purge_and_drop_missing_both_arms():
    source=rows(); mixed=source_samples(source)
    mixed=[s for s in mixed if s["match_id"]!="m5"]
    selected,matching,manifest=select_matched_window(source,target_fold=target(),reference_model=MechanismModel(),mixed_samples=mixed)
    assert {r["match_id"] for r in selected}=={"m2","m3","m4","m6","m7","m8"}
    assert len(matching)==4*len(selected)
    exclusions={r["match_id"]:r["reasons"] for r in manifest["excluded"]}
    assert "excluded_by_original_target_training_split" in exclusions["m9"]
    assert "matched_mixed_supervision_missing" in exclusions["m5"]
    assert "prediction_not_strictly_after_reference_fit" in exclusions["m1"]
    assert "member_of_reference_base_training" in exclusions["m0"]


def test_strict_calibration_label_before_first_evaluation_even_when_declared_cutoff_equal():
    source=rows(); source[8]["result_available_at"]=source[10]["prediction_cutoff"]
    chosen,_,manifest=select_matched_window(source,target_fold=target(),reference_model=MechanismModel(),mixed_samples=source_samples(source))
    assert "m8" not in {r["match_id"] for r in chosen}
    bad=next(r for r in manifest["excluded"] if r["match_id"]=="m8")
    assert "outcome_not_strictly_before_evaluation_predictions" in bad["reasons"]


def test_incomplete_four_transition_control_is_common_exclusion_duplicate_is_error():
    source=rows(); samples=source_samples(source)
    incomplete=[r for r in samples if not (r["match_id"]=="m5" and r["acquired_action_ids"]==["xg_view"])]
    chosen,_,manifest=select_matched_window(source,target_fold=target(),reference_model=MechanismModel(),mixed_samples=incomplete)
    assert "m5" not in {r["match_id"] for r in chosen}
    assert any("matched_mixed_four_transition_set_incomplete" in r["reasons"] for r in manifest["excluded"])
    with pytest.raises(ValueError,match="Duplicate"):
        index_transitions(samples+[deepcopy(samples[0])])


def test_same_version_recomputes_predictions_gains_and_explicit_provenance():
    source=rows(); model=MechanismModel()
    selected,mixed,manifest=select_matched_window(source,target_fold=target(),reference_model=model,mixed_samples=source_samples(source))
    arms=matched_supervision(selected,mixed,model,ACTIONS,as_of=manifest["calibration_cutoff"],reference_sha256="reference-hash")
    assert set(index_transitions(arms["same_version"]))==set(index_transitions(arms["matched_mixed"]))
    same=arms["same_version"][0]
    assert same["prediction_before"]==[.4,.3,.3]
    assert same["delta"]==pytest.approx(__import__('math').log(.5/.4))
    assert same["supervision_kind"]=="same_version_heldout_fixed_predictor"
    assert same["oof_predictor_version"]==model.model_version
    assert {r["oof_predictor_version"] for r in arms["matched_mixed"]}=={"mixed-other"}


def test_reject_forged_reference_version_wrong_gain_and_wrong_public_state():
    source=rows(); model=MechanismModel()
    chosen,mixed,manifest=select_matched_window(source,target_fold=target(),reference_model=model,mixed_samples=source_samples(source))
    arm=matched_supervision(chosen,mixed,model,ACTIONS,as_of=manifest["calibration_cutoff"],reference_sha256="h")["same_version"]
    for mutator in (lambda r:r.update(oof_predictor_version="other"),lambda r:r.update(delta=r["delta"]+1),
                    lambda r:r["state"]["base"].update(home_team_id="wrong")):
        bad=deepcopy(arm); mutator(bad[0])
        with pytest.raises(ValueError):
            validate_samples(bad,rows=chosen,as_of=manifest["calibration_cutoff"],reference_model=model,kind="same_version",reference_sha256="h")


def test_calibrator_retains_legacy_kind_but_accurate_same_version_metadata(tmp_path):
    source=rows(); model=MechanismModel()
    chosen,mixed,manifest=select_matched_window(source,target_fold=target(),reference_model=model,mixed_samples=source_samples(source))
    arm=matched_supervision(chosen,mixed,model,ACTIONS,as_of=manifest["calibration_cutoff"],reference_sha256="h")["same_version"]
    fitted=fit_calibrator(arm,kind="same_version",as_of=manifest["calibration_cutoff"],dataset_version="test",model=model,model_sha256="h",metadata={})
    assert fitted.manifest["calibration_matches"]==0
    assert fitted.manifest["metadata"]["supervision_kind"]=="same_version_heldout_fixed_predictor"
    assert fitted.manifest["training_joint_versions"]==[model.model_version]
    path=tmp_path/"calibrator.json";fitted.save(path)
    loaded=ResidualEVSICalibrator.load(path,model=model,model_sha256="h")
    assert loaded.version==fitted.version
    assert loaded.manifest==fitted.manifest
