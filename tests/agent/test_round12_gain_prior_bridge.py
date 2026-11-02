"""Adapter invariance tests use frozen historical inputs, never external calls."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from football_agent.audit import RunStore
from football_agent.gain_prior_bridge import (ESTIMATOR_KIND, GATE_KIND, GainPriorServices, GainPriorAcquisitionWorkflow,
    build_gain_prior_cards, load_gain_prior_services)
from football_agent.research_bridge import ResearchServices, research_state
from football_agent.verify_round12_state_ablation import reused_conditional_cards, select_cases
from football_research.acquisition.gain_prior import GainPriorEstimator
from football_research.acquisition.joint_returns import JointReturnPredictor


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "var/parallel/baselines/round11/frozen-version-v1"
AGENT = ROOT / "var/parallel/agent/round11/frozen-version-bridge-v1"


@pytest.fixture(scope="module")
def inputs():
    if not (SOURCE / "report.json").exists() or not (AGENT / "acceptance.json").exists():
        pytest.skip("Frozen round-eleven historical fixtures are unavailable in this checkout")
    report = json.loads((SOURCE / "report.json").read_text(encoding="utf-8"))
    fold = report["folds"][0]
    path = SOURCE / fold["joint_model"]["path"]
    model = JointReturnPredictor.load(path)
    sample_path = SOURCE / fold["supervision"]["same_version"]["path"]
    samples = [json.loads(s) for s in sample_path.read_text(encoding="utf-8").splitlines()]
    acceptance = json.loads((AGENT / "acceptance.json").read_text(encoding="utf-8"))
    ref = next(c for c in acceptance["checks"] if c["fold"] == 2 and c["strategy"] == "frozen-version-mean")
    run = RunStore(AGENT / "runs").load(ref["run_id"])
    event = next(e for e in run["events"] if e["type"] == "value_estimated")
    return {"fold": fold, "model": model, "path": path, "sha256": sha256(path.read_bytes()).hexdigest(), "samples": samples,
            "initial": event["snapshot"]["visible_state"], "after": run["visible_state"],
            "catalog": [a for a in event["data"]["candidates"] if a["status"] == "AVAILABLE"]}


def estimator(inputs, mode, *, constructed_gain=None, min_support=12):
    samples = deepcopy(inputs["samples"])
    if constructed_gain is not None:
        for row in samples:
            row["delta"] = constructed_gain
    return GainPriorEstimator(mode=mode, min_support=min_support).fit(samples,
        as_of=inputs["fold"]["calibrators"]["same_version"]["manifest"]["trained_as_of"],
        dataset_version="constructed_gain_test" if constructed_gain is not None else inputs["model"].metadata["dataset_version"],
        joint_model=inputs["model"], joint_artifact_sha256=inputs["sha256"],
        metadata={"purpose": "adapter_test", "gains_constructed": constructed_gain is not None})


def service(inputs, mode="action_only", **kwargs):
    return GainPriorServices(inputs["model"], estimator(inputs, mode, **kwargs), lambda_cost={"money": .5}, erasure_mode=mode)


def config(tmp_path, inputs, *, mode="action_only"):
    value_path = tmp_path / "gain-prior.json"
    estimator(inputs, mode).save(value_path)
    settings = {"partial_predictor": str(inputs["path"]), "partial_predictor_kind": "learned-joint-returns-v1",
                "value_estimator": str(value_path), "value_estimator_kind": ESTIMATOR_KIND,
                "gate": {"kind": GATE_KIND, "risk_level": "mean", "erasure_mode": mode}, "lambda_cost": {"money": .5}}
    path = tmp_path / "backend.json"
    path.write_text(json.dumps(settings), encoding="utf-8")
    return path, settings


@pytest.mark.parametrize("mode", ["action_only", "acquired_set"])
def test_distinct_kind_and_erasure_are_loaded_without_changing_legacy_factory(tmp_path, inputs, mode):
    path, _ = config(tmp_path, inputs, mode=mode)
    svc = load_gain_prior_services(path)
    conf = svc.configuration()
    assert conf["value_estimator_kind"] == ESTIMATOR_KIND
    assert conf["gain_prior_gate"]["erasure_mode"] == mode
    assert "calibrated_gate" not in conf and conf["gain_prior_gate"]["model_evsi"] is None
    assert conf["reference_joint"]["joint_artifact_sha256"] == inputs["sha256"]
    with pytest.raises(ValueError, match="Unrecognized value estimator"):
        ResearchServices.load(path)


@pytest.mark.parametrize("mutation", ["residual_kind", "conservative", "wrong_erasure", "wrong_predictor", "horizon2"])
def test_configuration_rejects_disguised_residual_lcb_and_mismatched_modes(tmp_path, inputs, mutation):
    path, settings = config(tmp_path, inputs)
    if mutation == "residual_kind":
        settings["value_estimator_kind"] = "residual-evsi-calibrator-v1"
    elif mutation == "conservative":
        settings["gate"]["risk_level"] = "conservative"
    elif mutation == "wrong_erasure":
        settings["gate"]["erasure_mode"] = "acquired_set"
    elif mutation == "wrong_predictor":
        settings["partial_predictor_kind"] = "contextual-partial-v1"
    else:
        settings["gate"]["horizon"] = 2
    path.write_text(json.dumps(settings), encoding="utf-8")
    with pytest.raises(ValueError):
        load_gain_prior_services(path)


@pytest.mark.parametrize("mode", ["action_only", "acquired_set"])
def test_numeric_return_and_team_identity_changes_do_not_change_value_but_predictor_still_updates(inputs, mode, monkeypatch):
    svc = service(inputs, mode)
    original = deepcopy(inputs["after"])
    changed = deepcopy(original)
    changed["base_facts"]["home_team_id"], changed["base_facts"]["away_team_id"] = changed["base_facts"]["away_team_id"], changed["base_facts"]["home_team_id"]
    changed["base_facts"]["home_team_name"] = "admissible_changed_public_name"
    for key, value in changed["groups"]["xg"]["values"].items():
        if type(value) in (int, float):
            changed["groups"]["xg"]["values"][key] = 4. if key.startswith("home_") else .1
    remaining = [a for a in inputs["catalog"] if a["action_id"] == "results_view"]
    before_prediction = svc.predict(original)["probabilities"]
    changed_prediction = svc.predict(changed)["probabilities"]
    assert before_prediction != changed_prediction
    def forbidden(*args, **kwargs):
        raise AssertionError("Value erasure must not call current prediction or EVSI")
    with monkeypatch.context() as patch:
        patch.setattr(inputs["model"], "predict_partial", forbidden)
        patch.setattr(inputs["model"], "enumerate_action", forbidden)
        first = svc.estimate(original, remaining)[0]
        second = svc.estimate(changed, remaining)[0]
    assert first["research_estimate"] == second["research_estimate"]
    assert first["model_evsi"] is None and first["lower_gain"] is None
    assert first["estimator_kind"] == ESTIMATOR_KIND and first["erasure_mode"] == mode


def test_acquired_set_changes_stage_gain_but_action_only_average_remains_constant(inputs):
    remaining = [a for a in inputs["catalog"] if a["action_id"] == "results_view"]
    gains = {}
    for mode in ("action_only", "acquired_set"):
        svc = service(inputs, mode)
        gains[mode] = [svc.estimate(deepcopy(state), remaining)[0]["expected_loss_improvement"] for state in (inputs["initial"], inputs["after"])]
    assert gains["action_only"][0] == gains["action_only"][1]
    assert gains["acquired_set"][0] != gains["acquired_set"][1]


@pytest.mark.parametrize("mutation", ["hidden", "hypothetical", "failure", "future_fit", "mode"])
def test_erasure_preserves_runtime_public_failure_time_and_information_guards(inputs, mutation):
    state = deepcopy(inputs["initial"])
    if mutation == "hidden":
        state["base_facts"]["outcome"] = 0
    elif mutation == "hypothetical":
        state["joint_return_hypothesis"] = {"observations": {"xg_view": "missing"}}
    elif mutation == "failure":
        state["action_history"] = [{"action_id": "xg_view", "status": "PROVIDER_ERROR"}]
    elif mutation == "future_fit":
        state["prediction_cutoff"] = "2020-01-01T00:00:00Z"
    else:
        state["information_mode"] = "operational"
    with pytest.raises(ValueError):
        service(inputs).estimate(state, inputs["catalog"])


def test_zero_lambda_does_not_release_physical_budget(inputs):
    svc = service(inputs)
    svc.lambda_cost = {k: 0. for k in svc.lambda_cost}
    state = deepcopy(inputs["initial"])
    state["budget"] = {"max_simulated_cost": .02, "simulated_cost": 0., "max_actions": 1, "actions": 0}
    values = svc.estimate(state, inputs["catalog"])
    assert values[0]["status"] == "OK" and values[0]["cost_in_loss_units"] == 0
    assert values[1]["status"] == "UNAVAILABLE" and values[1]["reason"] == "physical_budget_exhausted"


class Backend:
    def __init__(self, inputs, svc, status="NO_DATA"):
        self.inputs, self.svc, self.status = inputs, svc, status
    def configuration(self):
        return self.svc.configuration()
    def start(self, *args):
        state = deepcopy(self.inputs["initial"])
        state["quality"].pop("protocol_version", None)  # Constructed return test uses generic public projection.
        return state
    def predict(self, state):
        return self.svc.predict(state)
    def prediction_conditioning(self, state):
        return self.svc.predictor.observed_categories(research_state(state))
    def actions(self, state):
        return [self.inputs["catalog"][1]]
    def estimate(self, state, actions):
        return self.svc.estimate(state, actions)
    def acquire(self, state, action_id, parameters, request_id):
        updated = deepcopy(state)
        updated["quality"]["group_statuses"]["xg"] = "source_missing"
        updated["action_history"].append({"action_id": action_id, "status": self.status})
        return {"status": self.status, "action_id": action_id, "request_id": request_id, "visible_state": updated,
                "cost": {"amount": .03, "status": "simulated", "unit": "simulation_credit"}}


@pytest.mark.parametrize("status", ["NO_DATA", "PROVIDER_ERROR"])
def test_new_workflow_keeps_missing_update_failure_semantics_and_explicit_card_kind(tmp_path, inputs, status):
    svc = service(inputs, constructed_gain=.1)
    flow = GainPriorAcquisitionWorkflow(Backend(inputs, svc, status), tmp_path / "runs")
    state = inputs["initial"]
    run = flow.start(state["match_id"], state["prediction_cutoff"], budget={"max_actions": 1, "max_simulated_cost": .03})
    final = flow.run_to_completion(run["run_id"])
    assert final["budget"]["actions"] == 1
    if status == "NO_DATA":
        assert final["prediction"]["observed_return_categories"] == {"xg_view": "missing"}
        assert sum(e["type"] == "prediction_updated" for e in final["events"]) == 2
    else:
        assert final["state"] == "DEFERRED" and final["reason"] == "JOINT_OBSERVATION_MODEL_MISMATCH"
    assert flow.store.verify(final["run_id"])["status"] == "OK"
    cards = build_gain_prior_cards(final)
    assert cards["value_estimator_kind"] == ESTIMATOR_KIND
    assert cards["state_erasure"]["erasure_mode"] == "action_only"
    for card in cards["cards"]:
        for candidate in card["candidates"]:
            assert candidate["model_evsi"] is None and candidate["lower_gain"] is None
            assert candidate["estimator_kind"] == ESTIMATOR_KIND


def test_no_support_stops_and_conservative_mode_is_rejected(tmp_path, inputs):
    flow = GainPriorAcquisitionWorkflow(Backend(inputs, service(inputs, min_support=10000)), tmp_path / "runs")
    state = inputs["initial"]
    with pytest.raises(ValueError, match="mean-only"):
        flow.start(state["match_id"], state["prediction_cutoff"], risk_mode="conservative")
    run = flow.start(state["match_id"], state["prediction_cutoff"])
    final = flow.run_to_completion(run["run_id"])
    assert final["reason"] == "ESTIMATE_UNAVAILABLE" and not final["history"]


def test_fixed_ten_cases_and_five_conditional_cards_are_reused_without_rerunning():
    if not (SOURCE / "report.json").exists():
        pytest.skip("Frozen round-eleven release unavailable")
    report = json.loads((SOURCE / "report.json").read_text(encoding="utf-8"))
    episodes = [json.loads(s) for s in (SOURCE / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
    selected = []
    for f in report["folds"]:
        f["gain_priors"] = {"action_only": {}, "acquired_set": {}}
        base = min((r for r in episodes if r["fold"] == f["fold"] and r["strategy"] == "base-only" and r["budget_money"] == .05), key=lambda r:(r["kickoff"],r["match_id"]))
        selected.append(base)
        for strategy in ("tool-prior-mean", "stage-prior-mean"):
            selected.append({**base, "strategy": strategy, "lambda_money": .5})
    cases = select_cases(report, selected)
    reference = reused_conditional_cards(AGENT, cases)
    assert len(cases) == 10 and len(reference["cards"]) == 5
    assert all(c["execution_reused_not_rerun"] for c in reference["cards"])
    selected[-1]["lambda_money"] = 0.
    with pytest.raises(ValueError, match="fixed at .5"):
        select_cases(report, selected)
