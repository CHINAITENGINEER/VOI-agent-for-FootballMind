from datetime import datetime, timedelta, timezone
import json

import pytest

from football_research.acquisition.risk_calibrated import CalibratedRiskAwareValueEstimator
from football_research.acquisition.state import GROUP_FIELDS


def _state(kickoff: datetime, match_id: str):
    return {"match_id": match_id,
            "prediction_cutoff": (kickoff - timedelta(days=1)).isoformat(),
            "information_mode": "historical_reconstruction",
            "base": {"competition_id": "epl", "home_team_id": "h", "away_team_id": "a",
                     "kickoff": kickoff.isoformat()},
            "groups": {group: {"status": "not_acquired", "values": {}}
                       for group in GROUP_FIELDS}, "action_history": []}


def _action(name="elo"):
    return {"action_id": "fetch-" + name, "group_ids": [name], "available": True,
            "cost": {"requests": 1.0, "money": .01, "latency_ms": 0.0, "tokens": 0.0},
            "cost_status": "simulated", "dependency_key": "provider-response",
            "cache_policy": "shared_response"}


def _rows(start_day=1, count=20, delta=.12):
    start = datetime(2025, 1, start_day, tzinfo=timezone.utc)
    rows = []
    for index in range(count):
        kickoff = start + timedelta(days=index)
        rows.append({"state": _state(kickoff, f"m{start_day}-{index}"),
                     "action": _action(), "delta": delta if index % 3 else -delta / 2,
                     "match_id": f"m{start_day}-{index}", "kickoff": kickoff.isoformat(),
                     "prediction_before": [0.6, .2, .2],
                     "result_available_at": (kickoff + timedelta(hours=3)).isoformat(),
                     "oof_train_cutoff": "2024-12-01T00:00:00Z", "oof_train_ids": ["old"]})
    return rows


def _fitted():
    train = _rows(1, 20)
    calibration = _rows(25, 8, .08)
    return CalibratedRiskAwareValueEstimator(min_support=5, shrinkage=5).fit(
        train, calibration_samples=calibration, as_of="2025-02-10T00:00:00Z",
        dataset_version="fixture", calibration_target_coverage=.8)


def test_time_boundary_and_calibration_metadata_are_explicit():
    model = _fitted()
    assert model.manifest["temporal_boundary"]["train_matches"] == 20
    assert model.manifest["temporal_boundary"]["calibration_matches"] == 8
    assert model.calibration["standardized_residuals"] >= 1
    assert model.calibration["scope"].startswith("inner chronological tail")


def test_overlap_or_nonchronological_calibration_is_rejected():
    train = _rows(1, 10)
    overlapping = [dict(_rows(25, 1)[0], match_id=train[0]["match_id"], state=train[0]["state"],
                        kickoff=train[0]["kickoff"])]
    with pytest.raises(ValueError, match="overlap"):
        CalibratedRiskAwareValueEstimator(min_support=2).fit(
            train, calibration_samples=overlapping, as_of="2025-02-10T00:00:00Z",
            dataset_version="fixture")


def test_zero_support_is_visible_and_never_masquerades_as_parent_support():
    model = _fitted()
    state = _state(datetime(2025, 3, 1, tzinfo=timezone.utc), "future")
    unknown = _action("xg")
    result = model.estimate_value(state, [unknown], prediction=[.6, .2, .2], risk_level="conservative")[0]
    assert result["support"] == 0
    assert result["support_status"] == "no_support"
    assert result["expected_gain"] is None
    assert result["gate_reason"] == "stop_no_support"


def test_mean_and_conservative_expose_gate_reason_and_interval():
    model = _fitted()
    state = _state(datetime(2025, 3, 1, tzinfo=timezone.utc), "future")
    mean = model.estimate_value(state, [_action()], prediction=[.6, .2, .2], risk_level="mean")[0]
    conservative = model.estimate_value(state, [_action()], prediction=[.6, .2, .2], risk_level="conservative")[0]
    assert mean["support"] == 20
    assert mean["variance"] is not None
    assert mean["upper_gain"] >= mean["lower_gain"]
    assert conservative["decision_gain"] == conservative["lower_gain"]
    assert conservative["gate_reason"].startswith(("eligible", "stop_"))


def test_artifact_hash_roundtrip_and_tamper_detection(tmp_path):
    model = _fitted()
    path = tmp_path / "risk-v2.json"
    model.save(path)
    restored = CalibratedRiskAwareValueEstimator.load(path)
    state = _state(datetime(2025, 3, 1, tzinfo=timezone.utc), "future")
    assert model.estimate_value(state, [_action()]) == restored.estimate_value(state, [_action()])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["manifest"]["dataset_version"] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        CalibratedRiskAwareValueEstimator.load(path)


def test_single_game_residual_calibration_never_inflates_mean_interval():
    def fit(delta):
        return CalibratedRiskAwareValueEstimator(min_support=5).fit(
            _rows(1, 20), calibration_samples=_rows(25, 8, delta),
            as_of="2025-02-10T00:00:00Z", dataset_version="fixture")
    small, large = fit(.01), fit(3.)
    assert small.risk_multiplier == large.risk_multiplier == 1.64
    assert large.calibration["predictive_radius"] > small.calibration["predictive_radius"]
    for key in small.cells:
        assert small.cells[key]["lower_gain"] == large.cells[key]["lower_gain"]


def test_duplicate_training_records_do_not_inflate_support_or_precision():
    train = _rows()
    def fit(rows):
        return CalibratedRiskAwareValueEstimator(min_support=5).fit(
            rows, as_of="2025-02-10T00:00:00Z", dataset_version="fixture")
    a, b = fit(train), fit(train*5)
    assert a.cells == b.cells


def test_delayed_labels_are_excluded_before_calibration_tail():
    from football_research.acquisition.experiment_round5 import split_calibration
    rows = _rows(1, 20) + _rows(25, 8)
    rows[0]["result_available_at"] = "2025-02-09T00:00:00Z"
    train, tail, audit = split_calibration(rows, .2)
    assert rows[0]["match_id"] not in {r["match_id"] for r in train}
    assert audit["purged_matches"] >= 1
    assert max(r["result_available_at"] for r in train) < min(r["state"]["prediction_cutoff"] for r in tail)
