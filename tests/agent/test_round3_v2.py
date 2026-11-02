"""Acceptance checks for the formal active-information-v2 bridge.

The fixture is a frozen historical reconstruction: all charges are explicitly
simulated reveal credits and no network call is made by these tests.
"""
import json
from pathlib import Path

import pytest

from football_agent.acquisition import AcquisitionBudget, AcquisitionWorkflow
from football_agent.research_bridge import make_backend, research_state
from football_research.acquisition.policy import select_or_stop


ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "var/parallel/agent/round3/real-v2/backend-fold-2.json"
BUNDLE = ROOT / "var/parallel/collectors/round2/exports/understat-2025-historical_reconstruction-a3b3d45811fd5cda"


@pytest.fixture(scope="module")
def backend():
    if not CONFIG.exists() or not BUNDLE.exists():
        pytest.skip("formal v2 research artifacts are not present")
    return make_backend(bundle_path=BUNDLE, model_artifact=CONFIG)


def fixture(backend):
    item = backend.matches()[0]
    return item["match_id"], item["prediction_cutoff"]


def test_formal_versions_and_budget_are_pinned(backend):
    cfg = backend.configuration()
    assert cfg["mode"] == "historical_controlled_reveal"
    assert cfg["recommended_budget"] == {"max_simulated_cost": 0.05, "max_actions": 2}
    assert cfg["lambda_cost"]["money"] == pytest.approx(0.25)
    assert cfg["dataset_version"].startswith("platform-history-research-v1.0:")
    assert cfg["estimator_version"].startswith("empirical-mask-action-shrinkage-v1:")


def test_numeric_selection_matches_research_policy(backend):
    match_id, cutoff = fixture(backend)
    state = backend.start(match_id, cutoff, seed=7)
    actions = [item for item in backend.actions(state) if item["status"] == "AVAILABLE"]
    values = backend.estimate(state, actions)
    direct = select_or_stop(
        research_state(state),
        [value["research_estimate"] for value in values],
        {"money": 0.05, "requests": 2},
        backend._services.lambda_cost,
    )
    assert direct["decision"] == "action"
    assert direct["action_id"] == "xg_view"
    selected = max(values, key=lambda value: (value["net_gain"], value["action_id"]))
    assert selected["action_id"] == direct["action_id"]
    assert selected["net_gain"] == pytest.approx(direct["net_gain"])


@pytest.mark.parametrize("strategy,expected_actions,expected_cost", [("fixed", 2, 0.05), ("all_candidates", 1, 0.03)])
def test_real_workflow_and_direct_prediction(strategy, expected_actions, expected_cost, backend, tmp_path):
    match_id, cutoff = fixture(backend)
    workflow = AcquisitionWorkflow(backend, tmp_path)
    run = workflow.start(match_id, cutoff, strategy, AcquisitionBudget(max_actions=2, max_simulated_cost=0.05), seed=7)
    run = workflow.run_to_completion(run["run_id"])
    assert run["state"] == "STOPPED"
    assert run["budget"]["actions"] == expected_actions
    assert run["budget"]["simulated_cost"] == pytest.approx(expected_cost)
    direct = backend.predict(run["visible_state"])
    assert run["prediction"]["probabilities"] == direct["probabilities"]
    assert run["visible_state"]["information_mode"] == "historical_reconstruction"
    assert all(event.get("cost_status", "simulated") != "actual_network" for event in run["events"])


def test_config_does_not_mix_v1_model_namespace():
    data = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert "active-information-v2" in data["research_release"]["directory"]
    assert "active-information-v1" not in str(data["partial_predictor"])
    assert "active-information-v1" not in str(data["value_estimator"])
