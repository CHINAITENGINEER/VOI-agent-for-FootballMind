from copy import deepcopy

import pytest

from football_agent.research_bridge import (
    BackendUnavailable, PublicStateError, SyntheticBackend, make_backend,
    public_state, validate_parameters, platform_state, restore_platform_state, research_state,
    ResearchServices, platform_actions,
)


def started(backend):
    match = backend.matches()[0]
    return backend.start(match["match_id"], match["prediction_cutoff"], seed=17)


@pytest.mark.parametrize("field", ["label", "target", "hidden_groups", "archive_path"])
def test_nested_private_information_rejected(field):
    state = started(SyntheticBackend())
    state["base_facts"][field] = "private"
    with pytest.raises(PublicStateError):
        public_state(state)


def test_unrevealed_group_rejected_and_outputs_detached():
    state = started(SyntheticBackend())
    state["groups"]["xg"] = {"value": 20}
    with pytest.raises(PublicStateError):
        public_state(state)
    state["mask"]["xg"] = True
    projected = public_state(state)
    projected["groups"]["xg"]["value"] = 99
    assert state["groups"]["xg"]["value"] == 20


def test_synthetic_restore_uses_only_visible_state():
    first = SyntheticBackend()
    state = started(first)
    result = first.acquire(state, "fixture.reveal.strength", {}, "request-1")
    restored = SyntheticBackend()
    assert first.predict(result["visible_state"]) == restored.predict(result["visible_state"])
    assert first.actions(result["visible_state"]) == restored.actions(result["visible_state"])
    assert state["mask"]["strength"] is False
    assert result["visible_state"]["mask"]["strength"] is True


def test_no_data_only_disables_attempted_action():
    backend = SyntheticBackend("no_data")
    result = backend.acquire(started(backend), "fixture.reveal.strength", {}, "request-1")
    assert result["status"] == "NO_DATA"
    actions = {item["action_id"]: item for item in backend.actions(result["visible_state"])}
    assert actions["fixture.reveal.strength"]["status"] == "UNAVAILABLE"
    assert actions["fixture.reveal.form"]["status"] == "AVAILABLE"
    assert result["cost"] == {"amount": 1.0, "unit": "simulation_credit", "status": "simulated"}


def test_value_fixture_selects_different_actions_after_reveal():
    backend = SyntheticBackend()
    initial = started(backend)
    values = backend.estimate(initial, backend.actions(initial)[:3])
    assert max(values, key=lambda x: x["net_gain"])["action_id"] == "fixture.reveal.strength"
    next_state = backend.acquire(initial, "fixture.reveal.strength", {}, "request-1")["visible_state"]
    legal = [a for a in backend.actions(next_state) if a["status"] == "AVAILABLE"]
    values = backend.estimate(next_state, legal)
    assert max(values, key=lambda x: x["net_gain"])["action_id"] == "fixture.reveal.xg"


@pytest.mark.parametrize("action_id,parameters", [("file.read", {}), ("fixture.reveal.form", {"path": "private.json"})])
def test_out_of_catalog_or_extra_arguments_rejected(action_id, parameters):
    backend = SyntheticBackend()
    with pytest.raises(ValueError):
        backend.acquire(started(backend), action_id, parameters, "request-1")


def test_real_missing_interfaces_explicit():
    with pytest.raises(BackendUnavailable, match="INTERFACE_NOT_READY"):
        make_backend()
    assert make_backend(mode="synthetic_test").configuration()["mode"] == "synthetic_test"


@pytest.mark.parametrize("parameters", [{"n": True}, {"n": -1}, {"n": 4}, {"n": "1"}, {}])
def test_argument_schema_bounds(parameters):
    action = {"parameters_schema": {"properties": {"n": {"type": "integer", "minimum": 0, "maximum": 3}}, "required": ["n"]}}
    with pytest.raises(ValueError):
        validate_parameters(action, parameters)


def platform_fixture(revealed=False):
    from football_platform.information import VisibleState, EvidenceGroup, EvidenceTimes
    group = EvidenceGroup(group_id="strength", values={"home_elo": 1510, "away_elo": 1490}, source="understat",
                          times=EvidenceTimes(feature_available_at="2025-03-01T00:00:00Z", basis="historical_assumption"),
                          quality="VALIDATED", version="feature-v1", evidence_refs=["sha256:" + "a" * 64])
    return VisibleState(episode_id="public-episode", match_id="canonical-1", prediction_cutoff="2025-03-02T00:00:00Z",
                        information_mode="historical_reconstruction", fixture={"match_id": "canonical-1", "competition_id": "league",
                        "home_team_id": "h", "away_team_id": "a", "kickoff": "2025-03-03T00:00:00Z"},
                        base_prior={"home": 0.4, "draw": 0.3, "away": 0.3}, groups={"strength": group} if revealed else {},
                        mask={"strength": "OBSERVED" if revealed else "UNACQUIRED", "recent_form": "SOURCE_MISSING", "rolling_xg": "UNACQUIRED"},
                        catalog_version="catalog-v1", cost_config_version="cost-v1")


@pytest.mark.parametrize("revealed", [True, False])
def test_actual_public_contract_roundtrip(revealed):
    source = platform_fixture(revealed)
    projected = platform_state(source)
    assert projected["mask"]["elo"] is revealed
    assert restore_platform_state(projected).model_dump(mode="json") == source.model_dump(mode="json")
    research = research_state(projected)
    assert research["groups"]["recent_form"] == {"status": "source_missing", "values": {}}
    assert research["mask"] == (["elo"] if revealed else [])


def test_forged_future_observation_stopped_before_research():
    state = platform_state(platform_fixture(True))
    state["groups"]["elo"]["available_at"] = "2025-03-04T00:00:00Z"
    with pytest.raises(ValueError, match="unavailable"):
        research_state(state)


def test_actual_public_cost_catalog_preserves_dependency_and_simulation():
    from football_research.acquisition.platform_bridge import experiment_actions
    _, manifest = experiment_actions()
    actions = platform_actions(manifest["catalog"], platform_state(platform_fixture()), {"money": 0.5})
    assert actions[0]["action_id"] == "results_view"
    assert actions[0]["group_ids"] == ["elo", "recent_form"]
    assert actions[0]["cost"]["unit"] == "simulation_credit"
    assert actions[0]["cost"]["status"] == "simulated"
    assert actions[0]["cost_in_loss_units"] == 0.01
    assert actions[0]["dependency_key"] == "understat_league_response"


def test_service_delegates_values_without_replacing_research_gain():
    class Predictor:
        calls = 0
        def predict_partial(self, state):
            self.calls += 1
            assert "fixture" not in state and "base" in state
            return {"probabilities": [0.4, 0.3, 0.3], "mask": []}
    class Estimator:
        def estimate(self, state, actions, prediction=None):
            assert prediction["probabilities"] == [0.4, 0.3, 0.3]
            return [{"action_id": action["action_id"], "cost": action["cost"], "gain": -0.02,
                     "supported": True, "standard_error": 0.01, "lower_gain": -0.04,
                     "support_matches": 25, "reason": "estimated", "version": "fixture-estimator",
                     "estimator_cost": {"requests": 0, "money": 0, "latency_ms": 1, "tokens": 0}}
                    for action in actions]
    predictor = Predictor()
    service = ResearchServices(predictor, Estimator(), {"money": 0.5})
    visible = platform_state(platform_fixture())
    from football_research.acquisition.platform_bridge import experiment_actions
    _, manifest = experiment_actions()
    actions = platform_actions(manifest["catalog"], visible, service.lambda_cost)
    service.predict(visible)
    values = service.estimate(visible, actions)
    assert predictor.calls == 1
    assert values[0]["expected_loss_improvement"] == -0.02
    assert values[0]["net_gain"] == pytest.approx(-0.03)
    assert values[0]["support"] == 25
