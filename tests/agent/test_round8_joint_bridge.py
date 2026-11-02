from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json

import pytest

from football_agent.acquisition import AcquisitionWorkflow
from football_agent.joint_bridge import JointPlanningServices, physical_budget
from football_agent.joint_workflow import JointAcquisitionWorkflow
from football_agent.research_bridge import ResearchServices, PublicStateError, research_state
from football_agent.verify_round8_joint import compare_candidate_trace, select_cases
from football_research.acquisition.contextual_partial import ContextualPartialPredictor
from football_research.acquisition.joint_returns import JointReturnPredictor
from football_research.acquisition.state import GROUP_FIELDS
from football_research.acquisition.value import _canonical


ACTIONS = [{"action_id": aid, "group_ids": groups, "available": True, "cost_status": "simulated",
            "cost": {"money": money, "requests": 1., "latency_ms": 0., "tokens": 0.}}
           for aid, groups, money in (("results_view", ["elo", "recent_form"], .02), ("xg_view", ["xg"], .03))]


@pytest.fixture(scope="module")
def model():
    walk = {"provider": "understat", "aliases": {"team": {"home": "provider:understat:team:1", "away": "provider:understat:team:2"},
            "competition": {"epl": "provider:understat:competition:EPL"}}, "sources": [],
            "time_interpretation": "constructed_bridge_test"}
    walk["version"] = "provider-identity-crosswalk:" + sha256(_canonical(walk).encode()).hexdigest()[:20]
    rows, contexts = [], {}
    for i in range(36):
        kickoff = datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)
        cutoff = (kickoff - timedelta(days=1)).isoformat()
        row = {"match_id": f"m{i}", "home_team_id": "home", "away_team_id": "away", "competition_id": "epl",
               "kickoff": kickoff.isoformat(), "prediction_cutoff": cutoff, "fixture_available_at": cutoff,
               "features_available_at": cutoff, "result_available_at": (kickoff + timedelta(hours=3)).isoformat(),
               "information_mode": "historical_reconstruction", "home_goals": [2, 1, 0][i % 3], "away_goals": [0, 1, 2][i % 3]}
        row.update({f: float(i % 3) for fields in GROUP_FIELDS.values() for f in fields})
        row.update(home_elo=1000 + 100 * (i % 3), away_elo=1100, home_rolling_points=float(i % 3), away_rolling_points=1.,
                   home_rolling_xg_for=.5 + i % 3, home_rolling_xg_against=1., away_rolling_xg_for=1., away_rolling_xg_against=1.)
        rows.append(row)
        contexts[row["match_id"]] = {"match_id": row["match_id"], "probabilities": [.4, .3, .3], "prediction_cutoff": cutoff,
                                    "kickoff": row["kickoff"], "oof_train_cutoff": "2024-12-01T00:00:00Z",
                                    "oof_train_ids": ["past"], "oof_manifest_ref": "constructed:prior", "oof_predictor_version": "constructed"}
    base = ContextualPartialPredictor(walk).fit(rows, "2025-03-01T00:00:00Z", "constructed")
    return JointReturnPredictor(base).fit(rows, contexts, as_of="2025-03-01T00:00:00Z", dataset_version="constructed", actions=ACTIONS)


def visible():
    return {"match_id": "future", "prediction_cutoff": "2025-04-01T12:00:00Z", "information_mode": "historical_reconstruction",
            "base_facts": {"competition_id": "epl", "home_team_id": "home", "away_team_id": "away", "kickoff": "2025-04-02T12:00:00Z"},
            "groups": {}, "mask": {g: False for g in GROUP_FIELDS}, "quality": {}, "action_history": [],
            "budget": {"max_simulated_cost": .05, "simulated_cost": 0., "max_actions": 2, "actions": 0},
            "evidence_versions": {}, "state_version": "constructed"}


def catalog():
    return [{"action_id": a["action_id"], "group_ids": a["group_ids"], "status": "AVAILABLE", "available": True,
             "parameters": {}, "cost_vector": deepcopy(a["cost"]),
             "cost": {"amount": a["cost"]["money"], "unit": "simulation_credit", "status": "simulated"},
             "version": "constructed", "evidence_version": "constructed"} for a in ACTIONS]


def config(tmp_path, model, *, horizon=2):
    model.save(tmp_path / "joint.json")
    value = {"partial_predictor": "joint.json", "value_estimator": "joint.json",
             "partial_predictor_kind": "learned-joint-returns-v1", "value_estimator_kind": "joint-risk-planner-v1",
             "planner": {"kind": "finite-joint-bellman-v1", "horizon": horizon}, "lambda_cost": {"money": 0}}
    path = tmp_path / "backend.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path, value


def test_joint_loading_preserves_model_identity_and_explicit_planning_scope(tmp_path, model):
    path, _ = config(tmp_path, model)
    services = ResearchServices.load(path)
    assert isinstance(services, JointPlanningServices)
    assert services.predictor.model_version == model.model_version
    assert services.configuration()["planner"]["horizon"] == 2
    assert services.configuration()["planner"]["supported_risk_modes"] == ["mean"]
    assert services.artifact_hashes["predictor"] == services.artifact_hashes["estimator"]


@pytest.mark.parametrize("field,value", [("partial_predictor_kind", "contextual-partial-v1"),
                                         ("planner", {"kind": "finite-joint-bellman-v1", "horizon": 3}),
                                         ("planner", {"kind": "guessed", "horizon": 2})])
def test_joint_load_rejects_implicit_or_invalid_configuration(tmp_path, model, field, value):
    path, settings = config(tmp_path, model)
    settings[field] = value
    path.write_text(json.dumps(settings), encoding="utf-8")
    with pytest.raises(ValueError):
        ResearchServices.load(path)


def test_different_predictor_and_value_joint_artifacts_are_rejected(tmp_path, model):
    path, settings = config(tmp_path, model)
    (tmp_path / "different.json").write_text("{}", encoding="utf-8")
    settings["value_estimator"] = "different.json"
    path.write_text(json.dumps(settings), encoding="utf-8")
    with pytest.raises(ValueError, match="same frozen joint"):
        ResearchServices.load(path)


@pytest.mark.parametrize("horizon", [1, 2])
def test_agent_candidate_values_equal_shared_budget_dependent_planner(model, horizon):
    services = JointPlanningServices(model, horizon=horizon, lambda_cost={"money": .5})
    state = visible()
    native = services.estimator.plan(research_state(state), ACTIONS, physical_budget(state))
    actual = services.estimate(state, catalog())
    for value, expected in zip(actual, native["candidates"]):
        assert value["net_gain"] == expected["selection_score"]
        assert abs(value["net_gain"] - expected["planning_net_gain"]) <= 5.1e-13
        assert value["expected_loss_improvement"] == expected["expected_loss_improvement"]
        assert value["cost_in_loss_units"] == expected["expected_plan_penalty"]
        assert value["lower_gain"] is None
        assert value["uncertainty"] is None
        assert value["net_gain"] == pytest.approx(value["expected_loss_improvement"] - value["cost_in_loss_units"])


def test_money_request_budget_stays_physical_with_zero_lambda(model):
    services = JointPlanningServices(model, horizon=2, lambda_cost={"money": 0})
    state = visible()
    state["budget"].update(max_simulated_cost=.02, max_actions=1)
    values = services.estimate(state, catalog())
    affordable = next(v for v in values if v["action_id"] == "results_view")
    unaffordable = next(v for v in values if v["action_id"] == "xg_view")
    assert affordable["cost_in_loss_units"] == 0
    assert affordable["research_estimate"]["expected_plan_cost_vector"]["money"] == .02
    assert unaffordable["status"] == "UNAVAILABLE"


def test_hypothetical_tokens_do_not_cross_real_agent_boundary(model):
    state = visible()
    state["joint_return_hypothesis"] = {"hypothetical": True, "observations": {"xg_view": "xg0"}}
    with pytest.raises(PublicStateError, match="Unexpected state"):
        JointPlanningServices(model, horizon=2, lambda_cost={}).predict(state)


def test_conservative_workflow_stops_explicitly_without_fabricated_confidence(tmp_path, model):
    services = JointPlanningServices(model, horizon=2, lambda_cost={"money": 0})
    class Backend:
        def configuration(self):
            return services.configuration()
        def start(self, match_id, cutoff, seed=0):
            return visible()
        def predict(self, state):
            return services.predict(state)
        def actions(self, state):
            return catalog()
        def estimate(self, state, actions):
            return services.estimate(state, actions)
        def acquire(self, *args, **kwargs):
            raise AssertionError("Unsupported conservative mode must not acquire")
    workflow = AcquisitionWorkflow(Backend(), tmp_path / "runs")
    run = workflow.start("future", visible()["prediction_cutoff"], risk_mode="conservative",
                         budget={"max_actions": 2, "max_simulated_cost": .05})
    final = workflow.run_to_completion(run["run_id"])
    assert final["state"] == "STOPPED"
    assert final["reason"] == "ESTIMATE_UNAVAILABLE"
    assert final["budget"]["actions"] == 0
    values = [c["value"] for e in final["events"] if e["type"] == "value_estimated" for c in e["data"]["candidates"]]
    assert values and all(v["conservative_status"] == "UNSUPPORTED_JOINT_PARAMETER_UNCERTAINTY" for v in values)
    assert workflow.store.verify(final["run_id"])["status"] == "OK"


def test_trace_verifier_checks_contingent_candidate_costs_not_just_final_prediction(model):
    services = JointPlanningServices(model, horizon=2, lambda_cost={"money": .5})
    state = visible()
    plan = services.estimator.plan(research_state(state), ACTIONS, physical_budget(state))
    values = services.estimate(state, catalog())
    run = {"events": [{"type": "value_estimated", "data": {"candidates": [
        {"action_id": v["action_id"], "value": v} for v in values]}}]}
    research = [{"step": 0, "decision": plan}]
    assert compare_candidate_trace(run, research)["candidate_values_checked"] == 2
    corrupted = deepcopy(run)
    corrupted["events"][0]["data"]["candidates"][0]["value"]["research_estimate"]["expected_plan_cost_vector"]["money"] += .01
    with pytest.raises(ValueError, match="numerical mismatch"):
        compare_candidate_trace(corrupted, research)


def test_one_and_two_step_cases_use_fixed_schedule_and_each_validation_lambda():
    fold = {"fold": 0, "joint_model": {"path": "models/fold-0-joint.json"}, "lambda_selection": {
        "evsi-one-step/0.05": {"selected": .25}, "evsi-two-step/0.05": {"selected": 1.0}}}
    rows = [{"fold": 0, "strategy": strategy, "match_id": mid, "budget_money": .05,
             "kickoff": "2025-05-01T18:00:00Z", "prediction_cutoff": "2025-04-30T18:00:00Z", "log_loss": loss}
            for strategy in ("base-only", "evsi-one-step", "evsi-two-step")
            for mid, loss in (("z-best", 0.0), ("a-first", 99.0))]
    selected = select_cases({"folds": [fold]}, rows)
    assert [c["first"]["match_id"] for c in selected] == ["a-first", "a-first"]
    assert [c["horizon"] for c in selected] == [1, 2]
    assert [c["lambda_money"] for c in selected] == [.25, 1.0]


@pytest.mark.parametrize("status", ["NO_DATA", "PROVIDER_ERROR"])
def test_missing_return_refreshes_prediction_and_provider_failure_defers(tmp_path, model, status):
    services = JointPlanningServices(model, horizon=1, lambda_cost={"money": 0})
    class Backend:
        def configuration(self):
            return services.configuration()
        def start(self, match_id, cutoff, seed=0):
            return visible()
        def predict(self, state):
            return services.predict(state)
        def prediction_conditioning(self, state):
            return model.observed_categories(research_state(state))
        def actions(self, state):
            return [catalog()[1]]
        def estimate(self, state, actions):
            return services.estimate(state, actions)
        def acquire(self, state, action_id, parameters, request_id):
            updated = deepcopy(state)
            updated["quality"]["group_statuses"] = {"xg": "source_missing"}
            updated["action_history"].append({"action_id": action_id, "status": status, "group_ids": ["xg"]})
            return {"status": status, "visible_state": updated, "cost": catalog()[1]["cost"],
                    "request_id": request_id, "action_id": action_id}
    workflow = JointAcquisitionWorkflow(Backend(), tmp_path / "runs")
    run = workflow.start("future", visible()["prediction_cutoff"], budget={"max_actions": 1, "max_simulated_cost": .03})
    final = workflow.run_to_completion(run["run_id"])
    if status == "NO_DATA":
        assert final["state"] == "STOPPED"
        assert final["prediction"]["observed_return_categories"] == {"xg_view": "missing"}
        assert sum(e["type"] == "prediction_updated" for e in final["events"]) == 2
        assert any(e["type"] == "joint_conditioning_changed" for e in final["events"])
    else:
        assert final["state"] == "DEFERRED"
        assert final["reason"] == "JOINT_OBSERVATION_MODEL_MISMATCH"
    assert final["budget"]["actions"] == 1
    assert workflow.store.verify(final["run_id"])["status"] == "OK"
