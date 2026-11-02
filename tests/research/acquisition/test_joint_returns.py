from copy import deepcopy
from datetime import datetime,timedelta,timezone
import json

import numpy as np
import pytest

from football_research.acquisition.contextual_identity import build_crosswalk
from football_research.acquisition.contextual_partial import ContextualPartialPredictor
from football_research.acquisition.joint_returns import JointReturnPredictor, HYPOTHESIS_KEY, RESULT_CATEGORIES, XG_CATEGORIES
from football_research.acquisition.joint_planner import JointRiskPlanner
from football_research.acquisition.state import GROUP_FIELDS


ACTIONS = [{"action_id":"results_view","group_ids":["elo","recent_form"],"available":True,"cost_status":"simulated",
            "cost":{"money":.02,"requests":1.,"latency_ms":0.,"tokens":0.}},
           {"action_id":"xg_view","group_ids":["xg"],"available":True,"cost_status":"simulated",
            "cost":{"money":.03,"requests":1.,"latency_ms":0.,"tokens":0.}}]


@pytest.fixture
def fitted(tmp_path):
    mapping = [{"source":"understat","entity_type":"team","source_id":str(i+70),"entity_id":f"t{i}","status":"source_scoped","valid_to":None} for i in range(3)]
    mapping.append({"source":"understat","entity_type":"competition","source_id":"EPL","entity_id":"epl","status":"source_scoped","valid_to":None})
    (tmp_path/"id_mappings.jsonl").write_text("\n".join(json.dumps(r) for r in mapping),encoding="utf-8")
    walk = build_crosswalk([tmp_path])
    rows,contexts = [],{}
    for i in range(60):
        kickoff = datetime(2025,1,1,tzinfo=timezone.utc)+timedelta(days=i)
        cutoff = (kickoff-timedelta(days=1)).isoformat()
        row = {"match_id":f"m{i}","competition_id":"epl","home_team_id":"t0","away_team_id":"t1",
            "kickoff":kickoff.isoformat(),"prediction_cutoff":cutoff,"result_available_at":(kickoff+timedelta(hours=3)).isoformat(),
            "fixture_available_at":cutoff,"features_available_at":cutoff,"information_mode":"historical_reconstruction",
            "home_goals":2 if i%3 == 0 else 0,"away_goals":2 if i%3 == 2 else 0,"score_period":"regulation"}
        row.update({f:float(i%3) for fields in GROUP_FIELDS.values() for f in fields})
        row.update(home_elo=1200+200*(i%3),away_elo=1300,home_rolling_points=float(i%3),away_rolling_points=1.,
            home_rolling_xg_for=float(i%3)+.5,home_rolling_xg_against=1.,away_rolling_xg_for=1.,away_rolling_xg_against=1.)
        if i == 0:
            row["home_rolling_xg_for"] = None
        rows.append(row)
        contexts[row["match_id"]] = {"match_id":row["match_id"],"probabilities":[.45+.002*(i%4),.3,.25-.002*(i%4)],
            "prediction_cutoff":cutoff,"kickoff":row["kickoff"],"oof_train_cutoff":"2024-12-01T00:00:00Z",
            "oof_train_ids":["past"],"oof_manifest_ref":"sha256:synthetic_mechanism_fixture","oof_predictor_version":"synthetic"}
    base = ContextualPartialPredictor(walk).fit(rows,as_of="2025-04-01T00:00:00Z",dataset_version="synthetic")
    model = JointReturnPredictor(base).fit(rows,contexts,as_of="2025-04-01T00:00:00Z",dataset_version="synthetic",actions=ACTIONS)
    return model,rows,contexts


def state():
    return {"match_id":"future","prediction_cutoff":"2025-05-01T00:00:00Z","information_mode":"historical_reconstruction",
        "base":{"home_team_id":"t0","away_team_id":"t1","competition_id":"epl","kickoff":"2025-05-02T00:00:00Z"},
        "groups":{g:{"status":"not_acquired","values":{}} for g in GROUP_FIELDS}}


def test_unobserved_runtime_probabilities_exactly_match_frozen_base(fitted):
    model,_,_ = fitted
    assert model.predict_partial(state())["probabilities"] == model.base_predictor.predict_partial(state())["probabilities"]


def test_hidden_paid_payload_is_inert_for_joint_and_expected_value(fitted):
    model,_,_ = fitted
    public,dirty = state(),state()
    dirty.update(result="away",home_goals=999,full_snapshot={"home_elo":99999})
    for group in dirty["groups"].values():
        group["values"] = {"home_elo":float("nan"),"home_rolling_points":object()}
        group["reason"] = "TIMEOUT"
    assert model.joint_distribution(public) == model.joint_distribution(dirty)
    assert model.enumerate_action(public,ACTIONS[0])["evsi"] == model.enumerate_action(dirty,ACTIONS[0])["evsi"]


def test_hypothetical_branch_never_copies_hidden_or_label_payload(fitted):
    class Poison:
        def __deepcopy__(self,memo):
            raise AssertionError("Unpurchased payload copied")
        def __str__(self):
            raise AssertionError("Unpurchased payload inspected")
    model,_,_ = fitted
    dirty = state()
    dirty["label"] = Poison()
    for group in dirty["groups"].values():
        group["values"] = Poison()
        group["reason"] = Poison()
    enumeration = model.enumerate_action(dirty,ACTIONS[0])
    for branch in enumeration["branches"]:
        assert "label" not in branch["next_state"]
        assert all("reason" not in group for group in branch["next_state"]["groups"].values())
        model.predict_hypothetical(branch["next_state"])


def test_martingale_cmi_nonnegative_evsi_and_correlated_joint(fitted):
    model,_,_ = fitted
    distribution = model.joint_distribution(state())
    joint = np.asarray(distribution["probabilities"])
    np.testing.assert_allclose(joint.sum(axis=(1,2)),distribution["prior_probabilities"],atol=1e-12)
    q = joint/np.asarray(distribution["prior_probabilities"])[:,None,None]
    assert np.max(np.abs(q-q.sum(axis=2)[:,:,None]*q.sum(axis=1)[:,None,:])) > .01
    for action in ACTIONS:
        enumeration = model.enumerate_action(state(),action)
        mean = sum(b["probability"]*np.asarray(b["posterior_probabilities"]) for b in enumeration["branches"])
        np.testing.assert_allclose(mean,enumeration["current_probabilities"],atol=1e-12)
        assert enumeration["evsi"] >= -1e-12
        assert enumeration["evsi"] == pytest.approx(enumeration["conditional_mutual_information"],abs=1e-12)


def test_hypothetical_tokens_rejected_at_runtime_and_planner_root(fitted):
    model,_,_ = fitted
    branch = model.enumerate_action(state(),ACTIONS[0])["branches"][0]
    with pytest.raises(ValueError,match="Hypothetical"):
        model.predict_partial(branch["next_state"])
    with pytest.raises(ValueError,match="hypothetical"):
        JointRiskPlanner(model).plan(branch["next_state"],ACTIONS,{"limits":{"money":.05,"requests":2},"spent":{}})
    assert model.predict_hypothetical(branch["next_state"])["probabilities"] == pytest.approx(branch["posterior_probabilities"])


def test_joint_prior_and_kernel_never_change_after_hypothetical_acquisition(fitted):
    model,_,_ = fitted
    before = model.joint_distribution(state())
    for branch in model.enumerate_action(state(),ACTIONS[0])["branches"]:
        after = model.joint_distribution(branch["next_state"],hypothetical=True)
        assert after["prior_probabilities"] == before["prior_probabilities"]
        assert after["q_support"] == before["q_support"]
        second = model.enumerate_action(branch["next_state"],ACTIONS[1],hypothetical=True)
        mean = sum(b["probability"]*np.asarray(b["posterior_probabilities"]) for b in second["branches"])
        np.testing.assert_allclose(mean,after["current_probabilities"],atol=1e-12)


def test_actual_missing_consumes_action_but_provider_failure_is_rejected(fitted):
    model,_,_ = fitted
    missing = state()
    missing["groups"]["xg"] = {"status":"source_missing","values":{}}
    assert model.observed_categories(missing) == {"xg_view":"missing"}
    with pytest.raises(ValueError,match="already observed"):
        model.enumerate_action(missing,ACTIONS[1])
    missing["groups"]["xg"]["reason"] = "PROVIDER_ERROR"
    with pytest.raises(ValueError,match="failure"):
        model.predict_partial(missing)


def test_oof_self_training_and_future_label_are_rejected(fitted):
    model,rows,contexts = fitted
    bad = deepcopy(contexts)
    bad[rows[0]["match_id"]]["oof_train_ids"].append(rows[0]["match_id"])
    with pytest.raises(ValueError,match="target occurs"):
        JointReturnPredictor(model.base_predictor).fit(rows,bad,as_of="2025-04-01T00:00:00Z",dataset_version="synthetic",actions=ACTIONS)
    bad_rows = deepcopy(rows)
    bad_rows[0]["result_available_at"] = "2025-06-01T00:00:00Z"
    with pytest.raises(ValueError,match="unavailable"):
        JointReturnPredictor(model.base_predictor).fit(bad_rows,contexts,as_of="2025-04-01T00:00:00Z",dataset_version="synthetic",actions=ACTIONS)


def test_model_serialization_integrity_and_exact_planner_executes(fitted,tmp_path):
    model,_,_ = fitted
    path = tmp_path/"joint.json"
    model.save(path)
    loaded = JointReturnPredictor.load(path)
    assert loaded.joint_distribution(state()) == model.joint_distribution(state())
    plan = JointRiskPlanner(loaded,horizon=2,lambda_cost={"money":.5}).plan(state(),ACTIONS,{"limits":{"money":.05,"requests":2},"spent":{}})
    assert plan["decision"] in {"stop","acquire"}
    assert plan["expected_terminal_risk"] >= 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["parameters"]["bandwidth"] = .99
    path.write_text(json.dumps(payload),encoding="utf-8")
    with pytest.raises(ValueError,match="hash"):
        JointReturnPredictor.load(path)
