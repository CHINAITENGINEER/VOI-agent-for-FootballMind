import json

import pytest

from football_agent.decision_value import simulate_episode, synthetic_scenarios
from football_agent.voi_trace import VoITrace, VoITraceLedger


def episode_event(*, stop=False):
    model = synthetic_scenarios()["qualified_rotation_unknown"]["model"]
    episode = simulate_episode(model, seed=500, budget=.08, horizon=2, max_actions=2)
    event = episode["trace"][-1] if stop else episode["trace"][0]
    if stop and event["decision"]["action_id"] != "STOP":
        event = {**event, "decision": {**event["decision"], "action_id": "STOP", "reason": "test_stop",
                                         "current_risk": event["decision"]["current_risk"],
                                         "prediction": event["decision"]["prediction"]}}
    return event


def test_action_trace_preserves_pre_action_value_and_posterior_without_evaluator_result():
    event = episode_event()
    trace = VoITrace.from_episode_event(event, episode_id="episode-1", target_key="match-1",
                                        request_id="request-1", model_version="finite-v1",
                                        captured_at="2026-09-27T08:00:00Z")
    assert trace.action_id == event["decision"]["action_id"]
    assert trace.expected_information_value is not None
    assert trace.expected_cost >= 0
    assert trace.belief_after is not None and trace.prediction_after is not None
    assert trace.observed_outcome == event["observation"]
    assert "evaluator_only" not in trace.as_dict()
    assert trace.as_dict()["hidden_result_included"] is False


def test_stop_trace_records_reason_without_inventing_information_value():
    event = episode_event(stop=True)
    trace = VoITrace.from_episode_event(event, episode_id="episode-stop", target_key="match-1",
                                        request_id="request-stop", model_version="finite-v1")
    assert trace.action_id == "STOP"
    assert trace.candidate_id == "STOP"
    assert trace.stop_reason == event["decision"]["reason"]
    assert trace.expected_information_value is None
    assert trace.belief_after is None


def test_trace_ledger_roundtrips_checksum_and_episode_filter(tmp_path):
    event = episode_event()
    trace = VoITrace.from_episode_event(event, episode_id="episode-1", target_key="match-1",
                                        request_id="request-1", model_version="finite-v1")
    ledger = VoITraceLedger(tmp_path)
    ledger.append([trace])
    assert ledger.entries()[0].trace_id == trace.trace_id
    assert ledger.for_episode("episode-1", target_key="match-1")[0].action_id == trace.action_id
    envelope = json.loads((tmp_path / "voi-trace.jsonl").read_text(encoding="utf-8"))
    assert envelope["record_sha256"]


def test_trace_ledger_rejects_duplicate_and_tampered_records(tmp_path):
    event = episode_event()
    trace = VoITrace.from_episode_event(event, episode_id="episode-1", target_key="match-1",
                                        request_id="request-1", model_version="finite-v1")
    ledger = VoITraceLedger(tmp_path)
    ledger.append([trace])
    with pytest.raises(ValueError, match="append-only"):
        ledger.append([trace])
    path = tmp_path / "voi-trace.jsonl"
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["record"]["expected_cost"] = 999
    path.write_text(json.dumps(envelope) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        ledger.entries()


def test_trace_rejects_non_normalized_belief_and_stop_observation():
    event = episode_event()
    with pytest.raises(ValueError, match="normalized"):
        VoITrace(
            trace_id="t", episode_id="e", target_key="m", step=0, action_id="query", candidate_id="query",
            request_id="r", selection_basis="policy", model_version="v", expected_information_value=0.1,
            expected_cost=0.01, expected_net_value=0.09, risk_before=0.5, expected_risk_after=0.4,
            observed_risk_after=0.4,
            belief_before=(.7, .7), prediction_before=(.5, .5), belief_after=(.5, .5),
            prediction_after=(.5, .5), observed_outcome="yes", stop_reason=None,
            captured_at="2026-09-27T08:00:00Z")
    with pytest.raises(ValueError, match="STOP traces cannot"):
        VoITrace(
            trace_id="t2", episode_id="e", target_key="m", step=0, action_id="STOP", candidate_id="STOP",
            request_id="r2", selection_basis="stop:test", model_version="v", expected_information_value=None,
            expected_cost=0, expected_net_value=0, risk_before=.5, expected_risk_after=None,
            observed_risk_after=None,
            belief_before=(.5, .5), prediction_before=(.5, .5), belief_after=None,
            prediction_after=None, observed_outcome="unexpected", stop_reason="test",
            captured_at="2026-09-27T08:00:00Z")
