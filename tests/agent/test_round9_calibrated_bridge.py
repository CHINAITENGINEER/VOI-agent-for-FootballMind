"""Bridge tests use explicitly constructed training records, not real OOF evidence."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json

import pytest

from football_agent.calibrated_joint_bridge import CalibratedJointServices, load_calibrated_joint_services, GATE_KIND
from football_agent.calibrated_joint_workflow import CalibratedJointAcquisitionWorkflow
from football_agent.joint_bridge import physical_budget
from football_agent.research_bridge import PublicStateError, research_state
from football_agent.verify_round9_calibrated import compare_candidate_trace, select_cases
from football_research.acquisition.contextual_partial import ContextualPartialPredictor
from football_research.acquisition.joint_returns import JointReturnPredictor
from football_research.acquisition.round9_residual import ResidualEVSICalibrator
from football_research.acquisition.state import GROUP_FIELDS
from football_research.acquisition.value import _canonical


ACTIONS = [{"action_id": aid, "group_ids": groups, "available": True, "cost_status": "simulated",
            "cost": {"money": money, "requests": 1., "latency_ms": 0., "tokens": 0.}}
           for aid, groups, money in (("results_view", ["elo", "recent_form"], .02), ("xg_view", ["xg"], .03))]


@pytest.fixture(scope="module")
def model():
    walk = {"provider": "understat", "aliases": {"team": {"home": "provider:understat:team:1", "away": "provider:understat:team:2"},
            "competition": {"epl": "provider:understat:competition:EPL"}}, "sources": [], "time_interpretation": "constructed_test"}
    walk["version"] = "provider-identity-crosswalk:" + sha256(_canonical(walk).encode()).hexdigest()[:20]
    rows, contexts = [], {}
    for i in range(36):
        kickoff = datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)
        cutoff = (kickoff - timedelta(days=1)).isoformat()
        row = {"match_id": f"constructed-{i}", "home_team_id": "home", "away_team_id": "away", "competition_id": "epl",
               "kickoff": kickoff.isoformat(), "prediction_cutoff": cutoff, "fixture_available_at": cutoff,
               "features_available_at": cutoff, "result_available_at": (kickoff + timedelta(hours=3)).isoformat(),
               "information_mode": "historical_reconstruction", "home_goals": [2, 1, 0][i % 3], "away_goals": [0, 1, 2][i % 3]}
        row.update({f: float(i % 3) for fields in GROUP_FIELDS.values() for f in fields})
        row.update(home_elo=1000 + 100 * (i % 3), away_elo=1100, home_rolling_points=float(i % 3), away_rolling_points=1.,
                   home_rolling_xg_for=.5 + i % 3, home_rolling_xg_against=1., away_rolling_xg_for=1., away_rolling_xg_against=1.)
        rows.append(row)
        contexts[row["match_id"]] = {"match_id": row["match_id"], "probabilities": [.4, .3, .3], "prediction_cutoff": cutoff,
                                    "kickoff": row["kickoff"], "oof_train_cutoff": "2024-12-01T00:00:00Z",
                                    "oof_train_ids": ["constructed-past"], "oof_manifest_ref": "constructed", "oof_predictor_version": "constructed"}
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


def calibration(model, *, model_hash="constructed", residual=.2, variation=0., size=36, same_week=False):
    samples = []
    for i in range(size):
        kickoff = datetime(2025, 1, 1, tzinfo=timezone.utc) + (timedelta(hours=i) if same_week else timedelta(days=i))
        cutoff = (kickoff - timedelta(days=1)).isoformat()
        state = research_state(visible())
        state.update(match_id=f"constructed-{i}", prediction_cutoff=cutoff)
        state["base"]["kickoff"] = kickoff.isoformat()
        for action in ACTIONS:
            samples.append({"match_id": state["match_id"], "state": state, "kickoff": kickoff.isoformat(),
                            "result_available_at": (kickoff + timedelta(hours=3)).isoformat(),
                            "oof_train_cutoff": "2024-12-01T00:00:00Z", "oof_train_ids": ["constructed-past"],
                            "oof_manifest_ref": "constructed-not-real-OOF", "oof_predictor_version": "constructed",
                            "acquired_action_ids": [], "action": action, "model_evsi": .01,
                            "delta": .01 + residual + (variation if i < size / 2 else -variation)})
    return ResidualEVSICalibrator().fit(samples, as_of="2025-03-01T00:00:00Z", diagnostics_as_of="2025-03-02T00:00:00Z",
        dataset_version="constructed", joint_model=model, joint_artifact_sha256=model_hash)


def service(model, *, risk="mean", **kwargs):
    return CalibratedJointServices(model, calibration(model, **kwargs), lambda_cost={"money": 0}, risk_level=risk)


def config(tmp_path, model, risk="mean"):
    model_path = tmp_path / "joint.json"
    model.save(model_path)
    calibration(model, model_hash=sha256(model_path.read_bytes()).hexdigest()).save(tmp_path / "value.json")
    settings = {"partial_predictor": "joint.json", "partial_predictor_kind": "learned-joint-returns-v1",
                "value_estimator": "value.json", "value_estimator_kind": "residual-evsi-calibrator-v1",
                "gate": {"kind": GATE_KIND, "risk_level": risk}, "lambda_cost": {"money": 0}}
    path = tmp_path / "backend.json"
    path.write_text(json.dumps(settings), encoding="utf-8")
    return path, settings


def test_independent_artifacts_load_with_explicit_reference_binding(tmp_path, model):
    path, _ = config(tmp_path, model)
    loaded = load_calibrated_joint_services(path)
    assert loaded.artifact_hashes["predictor"] != loaded.artifact_hashes["estimator"]
    assert loaded.configuration()["reference_joint"]["joint_artifact_sha256"] == loaded.artifact_hashes["predictor"]
    assert loaded.configuration()["calibrated_gate"]["horizon"] == 1
    assert "without_coverage_guarantee" in loaded.configuration()["calibrated_gate"]["uncertainty_kind"]


@pytest.mark.parametrize("field,value", [("partial_predictor_kind", "contextual-partial-v1"),
    ("value_estimator_kind", "joint-risk-planner-v1"), ("gate", {"kind": GATE_KIND, "risk_level": "mean", "horizon": 2}),
    ("gate", {"kind": GATE_KIND, "risk_level": "optimistic"}), ("value_estimator", "joint.json")])
def test_mistyped_or_multistep_configuration_is_rejected(tmp_path, model, field, value):
    path, settings = config(tmp_path, model)
    settings[field] = value
    path.write_text(json.dumps(settings), encoding="utf-8")
    with pytest.raises(ValueError):
        load_calibrated_joint_services(path)


def test_same_model_version_with_changed_artifact_bytes_breaks_binding(tmp_path, model):
    path, _ = config(tmp_path, model)
    with (tmp_path / "joint.json").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="reference joint version/hash"):
        load_calibrated_joint_services(path)


def test_reference_training_cutoff_mismatch_is_rejected(model):
    value = calibration(model)
    value.manifest["joint_training_as_of"] = "2025-02-01T00:00:00Z"
    with pytest.raises(ValueError, match="training cutoff"):
        CalibratedJointServices(model, value, lambda_cost={}, risk_level="mean")


@pytest.mark.parametrize("risk", ["mean", "conservative"])
def test_all_native_candidate_calibration_fields_survive_bridge(model, risk):
    svc = service(model, risk=risk)
    state = visible()
    plan = svc.estimator.plan(research_state(state), ACTIONS, physical_budget(state), risk_level=risk, lambda_cost=svc.lambda_cost)
    values = svc.estimate(state, catalog())
    for value, expected in zip(values, plan["candidates"]):
        assert {k: value["research_estimate"][k] for k in expected} == expected
        assert value["lower_gain"] == expected["lower_gain"]
        assert value["expected_loss_improvement"] == expected["g_hat"]
        assert value["uncertainty"] is None
        assert value["cost_in_loss_units"] == 0
    run = {"risk_mode": risk, "candidates": [{"action_id": a["action_id"]} for a in catalog()]}
    wf = object.__new__(CalibratedJointAcquisitionWorkflow)
    wf._merge_values(run, catalog(), values)
    run["events"] = [{"type": "value_estimated", "data": {"candidates": run["candidates"]}}]
    assert compare_candidate_trace(run, [{"step": 0, "decision": plan}])["candidate_values_checked"] == 2
    corrupted = deepcopy(run)
    corrupted["events"][0]["data"]["candidates"][0]["value"]["research_estimate"]["parent_support"] += 1
    with pytest.raises(ValueError, match="numerical mismatch"):
        compare_candidate_trace(corrupted, [{"step": 0, "decision": plan}])


class Backend:
    def __init__(self, svc, *, return_status="NO_DATA", only_xg=False):
        self.svc, self.return_status, self.only_xg = svc, return_status, only_xg
    def configuration(self):
        return self.svc.configuration()
    def start(self, match_id, cutoff, seed=0):
        return visible()
    def predict(self, state):
        return self.svc.predict(state)
    def prediction_conditioning(self, state):
        return self.svc.predictor.observed_categories(research_state(state))
    def actions(self, state):
        return catalog()[1:] if self.only_xg else catalog()
    def estimate(self, state, actions):
        return self.svc.estimate(state, actions)
    def acquire(self, state, action_id, parameters, request_id):
        action = next(a for a in catalog() if a["action_id"] == action_id)
        updated = deepcopy(state)
        updated["quality"]["group_statuses"] = {g: "source_missing" for g in action["group_ids"]}
        updated["action_history"].append({"action_id": action_id, "status": self.return_status, "group_ids": action["group_ids"]})
        return {"status": self.return_status, "visible_state": updated, "cost": action["cost"],
                "request_id": request_id, "action_id": action_id}


@pytest.mark.parametrize("residual,size,expected", [(-2., 36, "VALUE_STOP_NONPOSITIVE_NET_GAIN"),
                                                  (.2, 11, "ESTIMATE_UNAVAILABLE")])
def test_negative_gain_and_missing_support_stop_before_any_acquisition(tmp_path, model, residual, size, expected):
    svc = service(model, residual=residual, size=size)
    workflow = CalibratedJointAcquisitionWorkflow(Backend(svc), tmp_path / "runs")
    run = workflow.start("future", visible()["prediction_cutoff"])
    final = workflow.run_to_completion(run["run_id"])
    assert final["state"] == "STOPPED" and final["reason"] == expected
    assert not final["history"] and final["budget"]["simulated_cost"] == 0
    native = [c["value"]["research_estimate"] for e in final["events"] if e["type"] == "value_estimated" for c in e["data"]["candidates"]]
    assert all(v["expected_gain"] < 0 for v in native) if size >= 12 else all(v["expected_gain"] is None for v in native)
    assert workflow.store.verify(final["run_id"])["status"] == "OK"


def test_mean_and_descriptive_lcb_drive_different_actual_workflows(tmp_path, model):
    evsi = model.enumerate_action(research_state(visible()), ACTIONS[1])["evsi"]
    for risk, actions in (("mean", 1), ("conservative", 0)):
        svc = service(model, risk=risk, residual=.05 - evsi, variation=2.)
        workflow = CalibratedJointAcquisitionWorkflow(Backend(svc, only_xg=True), tmp_path / risk)
        run = workflow.start("future", visible()["prediction_cutoff"], budget={"max_actions": 1, "max_simulated_cost": .03})
        final = workflow.run_to_completion(run["run_id"])
        assert final["risk_mode"] == risk and final["budget"]["actions"] == actions
        assert final["budget"]["simulated_cost"] == pytest.approx(.03 * actions)
        assert workflow.store.verify(final["run_id"])["status"] == "OK"


def test_workflow_rejects_a_different_risk_mode_than_the_loaded_configuration(tmp_path, model):
    workflow = CalibratedJointAcquisitionWorkflow(Backend(service(model)), tmp_path / "runs")
    with pytest.raises(ValueError, match="risk mode differs"):
        workflow.start("future", visible()["prediction_cutoff"], risk_mode="conservative")


def test_parent_fallback_is_usable_and_single_week_lcb_is_explicitly_unavailable(tmp_path, model):
    mean = service(model, same_week=True)
    values = mean.estimate(visible(), catalog())
    assert all(v["status"] == "OK" and v["parent_support"] == 36 for v in values)
    assert all(v["standard_error"] is None and v["lower_gain"] is None for v in values)
    assert any(v["support_status"] == "supported_parent" and v["fallback_reason"] == "local_support_below_minimum_use_parent" for v in values)
    lcb = service(model, risk="conservative", same_week=True)
    workflow = CalibratedJointAcquisitionWorkflow(Backend(lcb), tmp_path / "runs")
    run = workflow.start("future", visible()["prediction_cutoff"], budget={"max_actions": 2, "max_simulated_cost": .05})
    final = workflow.run_to_completion(run["run_id"])
    assert final["reason"] == "ESTIMATE_UNAVAILABLE" and not final["history"]
    plan = lcb.estimator.plan(research_state(visible()), ACTIONS, physical_budget(visible()), risk_level="conservative", lambda_cost=lcb.lambda_cost)
    parity = compare_candidate_trace(final, [{"step": 0, "decision": plan}])
    assert parity["unsupported_candidates_checked"] == 2
    values = [c["value"] for e in final["events"] if e["type"] == "value_estimated" for c in e["data"]["candidates"]]
    assert all(v["gate_reason"] == "stop_uncertainty_unavailable" for v in values)


def test_acquired_missing_response_is_not_queried_twice_or_given_empty_mask_support(model):
    state = visible()
    state["quality"]["group_statuses"] = {"xg": "source_missing"}
    state["action_history"] = [{"action_id": "xg_view", "status": "NO_DATA", "group_ids": ["xg"]}]
    values = service(model).estimate(state, catalog())
    assert values[1]["reason"] == "already_attempted" and values[1]["status"] == "UNAVAILABLE"
    result = values[0]["research_estimate"]
    assert result["acquired_action_ids"] == ["xg_view"]
    assert result["expected_gain"] is None and result["gate_reason"] == "stop_no_support"


def test_scores_round_identically_at_stop_boundary_without_changing_raw_gains():
    value = {"action_id": "xg_view", "status": "OK", "support_status": "supported_parent", "gate_reason": "eligible",
             "expected_loss_improvement": 2e-13, "lower_gain": 1e-13, "cost_in_loss_units": 0.,
             "net_gain": 2e-13, "estimator_version": "constructed"}
    run = {"candidates": [{"action_id": "xg_view"}]}
    merged = object.__new__(CalibratedJointAcquisitionWorkflow)._merge_values(run, catalog()[1:], [value])
    assert merged[0]["net_gain"] == 0 and merged[0]["conservative_net_gain"] == 0
    assert merged[0]["expected_loss_improvement"] == 2e-13 and merged[0]["lower_gain"] == 1e-13


def test_zero_lambda_preserves_cash_and_request_affordability(model):
    state = visible()
    state["budget"].update(max_simulated_cost=.02, max_actions=1)
    values = service(model).estimate(state, catalog())
    assert values[0]["status"] == "OK" and values[0]["cost_in_loss_units"] == 0
    assert values[1]["status"] == "UNAVAILABLE"
    assert values[1]["reason"] == "physical_budget_exhausted"
    state["budget"]["max_actions"] = 0
    assert all(v["status"] == "UNAVAILABLE" for v in service(model).estimate(state, catalog()))


@pytest.mark.parametrize("status", ["NO_DATA", "PROVIDER_ERROR"])
def test_missing_refreshes_belief_while_failure_defers(tmp_path, model, status):
    workflow = CalibratedJointAcquisitionWorkflow(Backend(service(model), return_status=status, only_xg=True), tmp_path / "runs")
    run = workflow.start("future", visible()["prediction_cutoff"], budget={"max_actions": 1, "max_simulated_cost": .03})
    final = workflow.run_to_completion(run["run_id"])
    assert final["budget"]["actions"] == 1
    if status == "NO_DATA":
        assert final["prediction"]["observed_return_categories"] == {"xg_view": "missing"}
        assert sum(e["type"] == "prediction_updated" for e in final["events"]) == 2
    else:
        assert final["state"] == "DEFERRED" and final["reason"] == "JOINT_OBSERVATION_MODEL_MISMATCH"
    assert workflow.store.verify(final["run_id"])["status"] == "OK"


@pytest.mark.parametrize("kind", ["hidden", "token", "nested_token", "failure", "http_failure", "future_fit"])
def test_real_runtime_rejects_hidden_hypothetical_failed_and_future_inputs(model, kind):
    state = visible()
    if kind == "hidden":
        state["base_facts"]["target"] = 0
    elif kind == "token":
        state["joint_return_hypothesis"] = {"observations": {"xg_view": "missing"}}
    elif kind == "nested_token":
        state["base_facts"]["joint_return_hypothesis"] = {}
    elif kind in {"failure", "http_failure"}:
        state["action_history"] = [{"action_id": "xg_view", "status": "PROVIDER_ERROR" if kind == "failure" else "HTTP_ERROR:503"}]
    else:
        state["prediction_cutoff"] = "2025-02-01T00:00:00Z"
    with pytest.raises((ValueError, PublicStateError)):
        service(model).predict(state)
    with pytest.raises((ValueError, PublicStateError)):
        service(model).estimate(state, catalog())


def test_fixed_schedule_selection_ignores_effectiveness_and_uses_each_lambda():
    fold = {"fold": 0, "joint_model": {}, "calibrator": {}, "lambda_selection": {
        "calibrated-mean/0.05": {"selected": .25}, "calibrated-lcb/0.05": {"selected": 1.}}}
    rows = [{"fold": 0, "strategy": strategy, "match_id": mid, "budget_money": .05,
             "kickoff": "2025-05-01T18:00:00Z", "prediction_cutoff": "2025-04-30T18:00:00Z", "log_loss": loss}
            for strategy in ("base-only", "calibrated-mean", "calibrated-lcb") for mid, loss in (("z-best", 0.), ("a-first", 99.))]
    cases = select_cases({"folds": [fold]}, rows)
    assert [c["first"]["match_id"] for c in cases] == ["a-first", "a-first"]
    assert [c["lambda_money"] for c in cases] == [.25, 1.]
