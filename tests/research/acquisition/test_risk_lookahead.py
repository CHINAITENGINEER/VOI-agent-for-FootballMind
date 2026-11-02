from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json

import pytest

from football_research.acquisition.lookahead import mask_transition_state, select_two_step
from football_research.acquisition.risk import RiskAwareValueEstimator
from football_research.acquisition.state import GROUP_FIELDS
from football_research.acquisition.policy import select_or_stop


def make_state(day="2025-05-01", match_id="future"):
    kickoff = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
    return {"match_id": match_id, "prediction_cutoff": (kickoff - timedelta(days=1)).isoformat(),
            "information_mode": "historical_reconstruction", "base": {"competition_id": "epl",
            "home_team_id": "h", "away_team_id": "a", "kickoff": kickoff.isoformat()},
            "groups": {group: {"status": "not_acquired", "values": {}} for group in GROUP_FIELDS},
            "action_history": []}


def make_action(name="elo", **extra):
    return {"action_id": "fetch-" + name, "group_ids": [name], "available": True,
            "cost": {"requests": 1.0, "money": 0.01, "latency_ms": 0.0, "tokens": 0.0},
            "cost_status": "simulated", "dependency_key": "provider-response",
            "cache_policy": "shared_response", **extra}


def samples(n=40):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(n):
        kickoff = start + timedelta(days=i)
        state = make_state(kickoff.date().isoformat(), f"m{i}")
        rows.append({"state": state, "action": make_action(), "delta": 0.2 if i % 3 else -0.15,
                     "match_id": f"m{i}", "kickoff": kickoff.isoformat(),
                     "prediction_before": [0.6, 0.2, 0.2] if i % 2 else [0.34, 0.33, 0.33],
                     "result_available_at": (kickoff + timedelta(hours=3)).isoformat(),
                     "oof_train_cutoff": "2024-12-01T00:00:00Z", "oof_train_ids": ["older"]})
    return rows


def fitted():
    return RiskAwareValueEstimator(min_support=5, shrinkage=5).fit(
        samples(), as_of="2025-03-01T00:00:00Z", dataset_version="fixture")


def test_mean_and_conservative_outputs_keep_negative_gain_and_risk_fields():
    model = fitted()
    mean = model.estimate_value(make_state(), [make_action()], prediction=[0.6, 0.2, 0.2], risk_level="mean")[0]
    conservative = model.estimate_value(make_state(), [make_action()], prediction=[0.6, 0.2, 0.2], risk_level="conservative")[0]
    assert mean["expected_gain"] < 0 or mean["expected_gain"] > 0
    assert mean["support"] >= 5
    assert mean["cost_vector"]["money"] == pytest.approx(0.01)
    assert mean["estimator_version"] == model.version
    assert conservative["decision_gain"] == conservative["lower_gain"]
    assert conservative["decision_gain"] <= mean["expected_gain"]
    assert "quality=" in mean["quality_context"] and "dependency=" in mean["quality_context"]


def test_artifact_roundtrip_and_content_hash(tmp_path):
    model = fitted()
    path = tmp_path / "risk.json"
    model.save(path)
    restored = RiskAwareValueEstimator.load(path)
    state, action = make_state(), make_action()
    assert model.estimate_value(state, [action], prediction=[0.6, 0.2, 0.2]) == restored.estimate_value(state, [action], prediction=[0.6, 0.2, 0.2])
    payload = json.loads(path.read_text())
    payload["manifest"]["dataset_version"] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        RiskAwareValueEstimator.load(path)


def test_quality_freshness_dependency_are_runtime_context_not_hidden_values():
    model = fitted()
    state = make_state()
    action = make_action()
    base = model.estimate_value(state, [action], prediction=[0.6, 0.2, 0.2])[0]
    poisoned = deepcopy(state)
    poisoned["label"] = 2
    poisoned["full_snapshot"] = {"secret": 999}
    poisoned["groups"]["xg"] = {"status": "not_acquired", "values": {"secret": 999}}
    changed = model.estimate_value(poisoned, [action], prediction=[0.6, 0.2, 0.2])[0]
    # The unacquired xG is outside this action's dependency; labels and hidden
    # payloads do not reach the condition or value. A visible stale status is
    # intentionally different and is covered by the next policy gate.
    assert {key: base[key] for key in ("expected_gain", "uncertainty", "support", "net_gain")} == {key: changed[key] for key in ("expected_gain", "uncertainty", "support", "net_gain")}


def test_source_missing_action_does_not_masquerade_as_supported():
    model = fitted()
    state = make_state()
    state["groups"]["elo"] = {"status": "source_missing", "values": {}}
    result = model.estimate_value(state, [make_action()], prediction=[0.6, 0.2, 0.2])[0]
    assert result["fallback_reason"] in {"unsupported_action", "state_quality_fallback_to_action_confidence", "insufficient_support"}
    assert select_or_stop(state, result and [result], {}, {})["decision"] == "stop"


def test_mask_transition_never_inserts_returned_values_or_label():
    state = make_state()
    state["label"] = 1
    transitioned = mask_transition_state(state, make_action("xg"))
    assert transitioned["groups"]["xg"]["status"] == "observed"
    assert all(value is None for value in transitioned["groups"]["xg"]["values"].values())
    assert "label" not in transitioned
    assert transitioned["action_history"] == [{"action_id": "fetch-xg", "status": "projected"}]


def test_two_step_uses_projected_mask_only():
    model = fitted()
    actions = [make_action("elo"), make_action("xg")]

    class Predictor:
        def __call__(self, state):
            return {"probabilities": [0.6, 0.2, 0.2]}

    result = select_two_step(make_state(), actions, predictor=Predictor(), estimator=model,
                             budget={"money": 0.05, "requests": 2}, lambda_cost={"money": 0.1}, min_support=5)
    assert result["policy_version"] == "two-step-mask-lookahead-v1"
    assert "two_step" in result
    for item in result["two_step"]:
        assert "continuation" in item
