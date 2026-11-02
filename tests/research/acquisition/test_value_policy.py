from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import math

import pytest

from football_research.acquisition.value import ValueEstimator
from football_research.acquisition.policy import select_or_stop


def state(day="2025-05-01", match_id="future"):
    kickoff = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
    return {"match_id": match_id, "prediction_cutoff": (kickoff - timedelta(days=1)).isoformat(),
            "information_mode": "historical_reconstruction", "base": {"competition_id": "epl",
            "home_team_id": "h", "away_team_id": "a", "kickoff": kickoff.isoformat()},
            "groups": {g: {"status": "not_acquired", "values": {}} for g in ("elo", "recent_form", "xg")}}


def action(name="elo", **kwargs):
    return {"action_id": "fetch-" + name, "group_ids": [name], "available": True,
            "cost": {"requests": 1, "money": 0, "latency_ms": 10, "tokens": 0},
            "cost_status": "simulated", **kwargs}


def samples(n=30):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    result = []
    for i in range(n):
        kickoff = start + timedelta(days=i)
        s = state(kickoff.date().isoformat(), f"m{i}")
        result.append({"state": s, "action": action(), "delta": 0.2 if i % 2 else -0.1,
                       "match_id": f"m{i}", "kickoff": kickoff.isoformat(),
                       "prediction_before": [0.6, 0.2, 0.2] if i % 2 else [0.34, 0.33, 0.33],
                       "result_available_at": (kickoff + timedelta(hours=3)).isoformat(),
                       "oof_train_cutoff": "2024-12-01T00:00:00Z", "oof_train_ids": ["earlier1"]})
    return result


def fitted(rows=None):
    return ValueEstimator(min_support=5, shrinkage=5).fit(rows or samples(), as_of="2025-03-01T00:00:00Z", dataset_version="synthetic-test")


def semantic(rows):
    return [{k: v for k, v in row.items() if not k.startswith("estimator_")} for row in rows]


def estimate(gain=0.2, **changes):
    return {"action": action(), "gain": gain, "lower_gain": gain - 0.1,
            "support_matches": 30, "reason": "estimated", **changes}


def test_current_visible_probability_conditions_gain_and_negative_is_preserved():
    model = fitted()
    uncertain = model.estimate(state(), [action()], prediction=[0.34, 0.33, 0.33])[0]
    confident = model.estimate(state(), [action()], prediction=[0.6, 0.2, 0.2])[0]
    assert uncertain["gain"] < 0 < confident["gain"]
    assert uncertain["conditional_support_matches"] == confident["conditional_support_matches"] == 15
    assert uncertain["uncertainty_kind"].endswith("no_coverage_guarantee")
    assert uncertain["estimator_cost"]["requests"] == 0
    assert uncertain["estimator_batch_latency_ms"] >= 0


def test_unknown_payloads_target_and_identity_do_not_change_inference_or_action():
    model = fitted()
    before = state()
    after = deepcopy(before)
    after["match_id"] = "different-unseen-id"
    after["label"] = 2
    after["groups"]["xg"]["values"] = {"secret": float("inf"), "home_goals": 999}
    after["groups"]["xg"]["available_at"] = "invalid-hidden-time"
    a = model.estimate(before, [action()], [0.6, 0.2, 0.2])
    b = model.estimate(after, [action(returned_values={"home_goals": 1})], [0.6, 0.2, 0.2])
    assert semantic(a) == semantic(b)
    assert select_or_stop(before, a, {"requests": 3}, {"requests": 0.001}, min_support=5) == select_or_stop(after, b, {"requests": 3}, {"requests": 0.001}, min_support=5)


def test_repeat_snapshots_do_not_inflate_independent_match_support():
    rows = samples()
    a = fitted(rows).estimate(state(), [action()], [0.6, 0.2, 0.2])[0]
    b = fitted(rows * 4).estimate(state(), [action()], [0.6, 0.2, 0.2])[0]
    for key in ("gain", "standard_error", "support_matches", "parent_support_matches"):
        assert a[key] == b[key]


def test_sparse_probability_bin_explicitly_falls_back_to_mask_support():
    rows = samples()
    for row in rows:
        row["prediction_before"] = [0.6, 0.2, 0.2]
    estimate = fitted(rows).estimate(state(), [action()], [0.34, 0.33, 0.33])[0]
    assert estimate["reason"] == "conditional_support_fallback_to_mask"
    assert estimate["conditional_support_matches"] == 0
    assert estimate["support_matches"] == 30


def test_unsupported_state_action_never_masquerades_as_supported_parent():
    s = state()
    s["groups"]["xg"]["status"] = "source_missing"
    result = fitted().estimate(s, [action()], [0.6, 0.2, 0.2])
    assert result[0]["support_matches"] == 0
    assert result[0]["reason"] == "unsupported_state_action"
    assert select_or_stop(s, result, {}, {}, min_support=5)["reason"] == "insufficient_support"


@pytest.mark.parametrize("mutation", ["future_label", "future_oof", "target_in_train", "wrong_id", "bad_delta"])
def test_oof_time_and_identity_validation(mutation):
    rows = samples()
    if mutation == "future_label": rows[0]["result_available_at"] = "2030-01-01T00:00:00Z"
    if mutation == "future_oof": rows[0]["oof_train_cutoff"] = "2025-01-02T00:00:00Z"
    if mutation == "target_in_train": rows[0]["oof_train_ids"].append("m0")
    if mutation == "wrong_id": rows[0]["match_id"] = "wrong"
    if mutation == "bad_delta": rows[0]["delta"] = float("nan")
    with pytest.raises(ValueError): fitted(rows)


def test_future_estimator_and_invalid_probabilities_are_rejected():
    model = fitted()
    with pytest.raises(ValueError, match="after requested"):
        model.estimate(state("2025-02-01"), [action()])
    with pytest.raises(ValueError, match="normalized"):
        model.estimate(state(), [action()], [1, 1, 1])
    changed = state()
    changed["information_mode"] = "operational"
    with pytest.raises(ValueError, match="information mode"):
        model.estimate(changed, [action()])


def test_single_time_block_does_not_claim_estimable_interval():
    model = ValueEstimator(min_support=1, shrinkage=0).fit(samples(3), as_of="2025-03-01T00:00:00Z", dataset_version="synthetic")
    result = model.estimate(state(), [action()], [0.6, 0.2, 0.2])[0]
    assert result["time_blocks"] == 1
    assert result["lower_gain"] is None
    assert select_or_stop(state(), [result], {}, {}, conservative=True, min_support=1)["reason"] == "uncertainty_unavailable"


def test_artifact_roundtrip_and_content_tamper_detection(tmp_path):
    model = fitted()
    path = tmp_path / "value.json"
    model.save(path)
    restored = ValueEstimator.load(path)
    assert semantic(model.estimate(state(), [action()])) == semantic(restored.estimate(state(), [action()]))
    payload = json.loads(path.read_text())
    payload["manifest"]["dataset_version"] = "tampered"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash"):
        ValueEstimator.load(path)


@pytest.mark.parametrize("candidate,budget,expected", [
    (estimate(action=action(available=False)), {}, "no_capability"),
    (estimate(), {"requests": 0}, "budget"),
    (estimate(), {"deadline_ms_remaining": 1}, "deadline"),
    (estimate(gain=-0.01), {}, "nonpositive_net_gain"),
    (estimate(support_matches=1), {}, "insufficient_support"),
    (estimate(action=action(cost_status="unknown")), {}, "unknown_cost"),
])
def test_stop_reasons_are_distinct(candidate, budget, expected):
    assert select_or_stop(state(), [candidate], budget, {})["reason"] == expected


def test_unit_converted_cost_and_conservative_selection():
    regular = select_or_stop(state(), [estimate()], {}, {"requests": 0.15})
    conservative = select_or_stop(state(), [estimate()], {}, {"requests": 0.15}, conservative=True)
    assert regular["decision"] == "action"
    assert regular["net_gain"] == pytest.approx(0.05)
    assert conservative["reason"] == "nonpositive_net_gain"
    assert regular["lambda_units"]["latency_ms"] == "log_loss_nats_per_millisecond"


def test_source_missing_retry_and_exhaustion_are_bounded():
    s = state()
    s["groups"]["elo"]["status"] = "source_missing"
    assert select_or_stop(s, [estimate()], {}, {})["reason"] == "source_missing_no_retry"
    retry = estimate(action=action(retry_allowed=True, max_attempts=2))
    s["action_history"] = [{"action_id": "fetch-elo"}]
    assert select_or_stop(s, [retry], {}, {})["decision"] == "action"
    s["action_history"] *= 2
    assert select_or_stop(s, [retry], {}, {})["reason"] == "attempt_limit"


def test_bundled_action_can_reveal_new_group_without_double_request():
    s = state()
    s["groups"]["elo"] = {"status": "observed", "values": {}, "available_at": "2025-01-01T00:00:00Z"}
    bundled = action(group_ids=["elo", "xg"])
    decision = select_or_stop(s, [estimate(action=bundled)], {"requests": 1}, {})
    assert decision["decision"] == "action"
    assert decision["selected_estimate"]["cost"]["requests"] == 1
    assert select_or_stop(s, [estimate()], {}, {})["reason"] == "already_observed"


def test_complementary_xor_information_exposes_one_step_stop_limit():
    # Offline mechanism fixture only: each bit is marginally useless for XOR,
    # while seeing both bits determines the binary target. Runtime sees no bits.
    outcomes = [(a, b, a ^ b) for a in (0, 1) for b in (0, 1)]
    base_loss = -sum(math.log(0.5) for _ in outcomes) / len(outcomes)
    single_loss = base_loss
    joint_loss = -math.log(0.99)
    candidates = [estimate(gain=base_loss - single_loss), estimate(gain=0, action=action("xg"))]
    result = select_or_stop(state(), candidates, {"requests": 2}, {"requests": 0.01})
    assert result["reason"] == "nonpositive_net_gain"
    assert base_loss - joint_loss - 0.02 > 0
