"""Synthetic provider outputs; never charge an external LLM in regression tests."""
from copy import deepcopy
import json
from urllib.error import HTTPError

import pytest

from football_agent.proposer import LLMProposer, legal_actions, public_state


STATE = {"episode_id": "e1", "match_id": "m1", "prediction_cutoff": "2026-01-01T00:00:00Z",
         "information_mode": "historical_reconstruction", "base_facts": {"home_team": "A"},
         "groups": {}, "mask": {"strength": False}, "action_history": [], "budget": {"requests": 5}}
ACTIONS = [{"action_id": "reveal:strength", "group_ids": ["strength"], "status": "available",
            "parameters_schema": {"type": "object", "properties": {"window": {"type": "integer", "enum": [5, 10]}}, "required": ["window"]}}]
ANSWER = {"proposals": [{"action_id": "reveal:strength", "parameters": {"window": 5}, "reason": "Strength remains unrevealed"}], "stop": False}


def response(answer=None, **kwargs):
    return dict({"id": "provider-request", "model": "fixture-model", "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
                 "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(ANSWER if answer is None else answer)}}]}, **kwargs)


def proposer(tmp_path, transport=None, **kwargs):
    return LLMProposer(model="fixture-model", api_key="test-secret", ledger_path=tmp_path / "ledger.json",
                       transport=transport or (lambda _: response()), **kwargs)


def test_missing_configuration_is_zero_calls(tmp_path, monkeypatch):
    for name in ["FOOTBALL_LLM_MODEL", "FOOTBALL_LLM_API_KEY", "OPENAI_MODEL", "OPENAI_API_KEY"]:
        monkeypatch.delenv(name, raising=False)
    result = LLMProposer.from_env(ledger_path=tmp_path / "ledger.json").propose(STATE, ACTIONS)
    assert result["status"] == "NEEDS_CONFIGURATION"
    assert result["requests"] == 0 and result["actual_fee"]["status"] == "not_incurred"
    assert not (tmp_path / "ledger.json").exists()


def test_prompt_has_only_public_state_no_tools_and_capped_output(tmp_path):
    captured = []
    instance = proposer(tmp_path, lambda p: captured.append(p) or response())
    result = instance.propose(STATE, ACTIONS)
    assert result["status"] == "OK" and result["reason_is_value_estimate"] is False
    assert result["usage"]["total_count"] == 30
    assert result["actual_fee"]["status"] == "unknown"
    assert captured[0]["max_completion_tokens"] == 512
    assert "tools" not in captured[0] and "test-secret" not in json.dumps(captured)
    assert "ledger_path" not in json.dumps(instance.configuration())
    assert "test-secret" not in (tmp_path / "ledger.json").read_text()
    assert json.loads((tmp_path / "ledger.json").read_text())["calls"][result["request_id"]]["result"]["usage"]["total_count"] == 30


@pytest.mark.parametrize("extra", [{"hidden_groups": {"strength": 999}}, {"label": "home"}, {"features": {"all": 9}}, {"archive_path": "private"}])
def test_private_input_rejected_before_provider(tmp_path, extra):
    called = []
    result = proposer(tmp_path, lambda p: called.append(p)).propose(dict(STATE, **extra), ACTIONS)
    assert result["status"] == "INVALID_INPUT" and not called


def test_private_nested_label_and_mask_violation_rejected():
    with pytest.raises(ValueError, match="PRIVATE"):
        public_state(dict(STATE, groups={"x": {"target_label": 1}}))
    with pytest.raises(ValueError, match="UNREVEALED"):
        public_state(dict(STATE, groups={"strength": {"elo": 1700}}))
    assert public_state(dict(STATE, groups={"strength": {"elo": 1700}}, mask={"strength": True}))["groups"]["strength"]["elo"] == 1700


@pytest.mark.parametrize("mutation", [
    lambda a: a["proposals"][0].update(action_id="read:archive"),
    lambda a: a["proposals"][0]["parameters"].update(file="archive.json"),
    lambda a: a["proposals"][0]["parameters"].update(window=9),
    lambda a: a["proposals"][0]["parameters"].update(window=True),
    lambda a: a["proposals"][0].update(expected_gain=.5),
    lambda a: a.update(net_gain=.5),
    lambda a: a.update(stop=True),
    lambda a: a.update(proposals=[]),
])
def test_invalid_proposals_are_charged_but_not_executed(tmp_path, mutation):
    answer = deepcopy(ANSWER)
    mutation(answer)
    result = proposer(tmp_path, lambda _: response(answer)).propose(STATE, ACTIONS)
    assert result["status"] == "INVALID_RESPONSE"
    assert result["proposals"] == [] and result["requests"] == 1
    assert result["usage"]["total_count"] == 30


def test_unavailable_action_excluded_and_no_actions_skips_llm(tmp_path):
    action = dict(ACTIONS[0], status="NOT_CONFIGURED")
    called = []
    result = proposer(tmp_path, lambda p: called.append(p)).propose(STATE, [action])
    assert result["status"] == "NO_ACTIONS" and result["stop"] and not called


def test_budget_durable_across_instances(tmp_path):
    calls = []
    instance = proposer(tmp_path, lambda p: calls.append(p) or response(), max_calls=2)
    assert instance.propose(STATE, ACTIONS, request_id="a")["requests"] == 1
    resumed = proposer(tmp_path, lambda p: calls.append(p) or response(), max_calls=2)
    replay = resumed.propose(STATE, ACTIONS, request_id="a")
    assert replay["replayed"] and replay["requests"] == 0 and replay["original_requests"] == 1
    assert resumed.propose(STATE, ACTIONS, request_id="b")["status"] == "OK"
    assert resumed.propose(STATE, ACTIONS, request_id="c")["status"] == "LLM_BUDGET_EXHAUSTED"
    assert len(calls) == 2


def test_parent_budget_limits_even_fresh_ledger(tmp_path):
    assert proposer(tmp_path).propose(STATE, ACTIONS, remaining_calls=0)["status"] == "LLM_BUDGET_EXHAUSTED"


def test_pending_crash_is_not_retried(tmp_path):
    def crash(_):
        raise KeyboardInterrupt()
    instance = proposer(tmp_path, crash)
    with pytest.raises(KeyboardInterrupt):
        instance.propose(STATE, ACTIONS, request_id="a")
    result = proposer(tmp_path).propose(STATE, ACTIONS, request_id="a")
    assert result["status"] == "INTERRUPTED_OUTCOME_UNKNOWN" and result["requests"] == 0
    assert result["reservation"]["requests"] == 1


def test_request_id_with_changed_input_is_rejected(tmp_path):
    instance = proposer(tmp_path)
    instance.propose(STATE, ACTIONS, request_id="a")
    result = instance.propose(dict(STATE, state_version=2), ACTIONS, request_id="a")
    assert result["status"] == "REQUEST_ID_CONFLICT" and result["requests"] == 0


def test_http_rate_limit_retains_unknown_fee_and_no_auto_retry(tmp_path):
    calls = []
    def rate_limit(payload):
        calls.append(payload)
        raise HTTPError("https://fixture.invalid", 429, "secret-body", {}, None)
    result = proposer(tmp_path, rate_limit).propose(STATE, ACTIONS)
    assert result["status"] == "RATE_LIMITED" and result["requests"] == 1
    assert result["actual_fee"]["status"] == "unknown" and len(calls) == 1
    assert "secret-body" not in json.dumps(result)


def test_truncated_output_is_not_a_valid_proposal(tmp_path):
    answer = response()
    answer["choices"][0]["finish_reason"] = "length"
    result = proposer(tmp_path, lambda _: answer).propose(STATE, ACTIONS)
    assert result["status"] == "INVALID_RESPONSE"


def test_remote_schema_ref_rejected_without_request(tmp_path):
    action = dict(ACTIONS[0], parameters_schema={"$ref": "https://private.example/schema"})
    result = proposer(tmp_path).propose(STATE, [action])
    assert result["status"] == "INVALID_INPUT"


def test_explicit_stop(tmp_path):
    result = proposer(tmp_path, lambda _: response({"stop": True, "proposals": [], "reason": "No useful candidate"})).propose(STATE, ACTIONS)
    assert result["status"] == "OK" and result["stop"]


def test_local_schema_ref_supported(tmp_path):
    action = deepcopy(ACTIONS[0])
    action["parameters_schema"]["$defs"] = {"window": {"type": "integer", "const": 5}}
    action["parameters_schema"]["properties"]["window"] = {"$ref": "#/$defs/window"}
    assert proposer(tmp_path).propose(STATE, [action])["status"] == "OK"


def test_env_reuse_does_not_expose_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-value")
    monkeypatch.setenv("OPENAI_MODEL", "configured-model")
    instance = LLMProposer.from_env()
    assert instance.configuration()["model"] == "configured-model"
    assert instance.configuration()["configured"] is True
    assert "secret-value" not in json.dumps(instance.configuration())


@pytest.mark.parametrize("kwargs", [{"max_calls": 6}, {"max_calls": -1}, {"max_output_tokens": 10000}, {"base_url": "https://user:secret@example.org/v1"}])
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        LLMProposer(**kwargs)


def test_native_platform_catalog_unimplemented_excluded():
    actions = [{"action_id": "weather", "capability_status": "NOT_IMPLEMENTED"},
               {"action_id": "results_view", "capability_status": "AVAILABLE", "expected_cost": {"money": {"basis": "simulated", "amount": .02, "unit": "USD"}}}]
    assert [a["action_id"] for a in legal_actions(actions)] == ["results_view"]


def test_native_platform_visible_state_supported():
    state = dict(STATE, protocol_version="information-v1", base_prior={"home": 1/3},
                 prior_basis="declared_uniform_prior_not_fitted", experiment_role="exploratory", cost_config_version="c1",
                 groups={"strength": {"values": {"home_elo": 1500}}}, mask={"strength": "OBSERVED"})
    assert public_state(state)["groups"]["strength"]["values"]["home_elo"] == 1500
