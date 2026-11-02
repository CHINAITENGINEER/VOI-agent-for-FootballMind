from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from football_research.acquisition.partial import PartialPredictor
from football_research.acquisition.state import GROUP_FIELDS, GROUP_IDS, sanitize_state


def training_rows(n=30):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(n):
        kickoff = start + timedelta(days=i)
        row = {"match_id": f"m{i}", "competition_id": "league", "kickoff": kickoff.isoformat(),
               "prediction_cutoff": (kickoff - timedelta(hours=24)).isoformat(),
               "result_available_at": (kickoff + timedelta(hours=3)).isoformat(),
               "information_mode": "historical_reconstruction", "home_goals": [2, 1, 0][i % 3],
               "away_goals": [0, 1, 2][i % 3]}
        for j, field in enumerate(field for fields in GROUP_FIELDS.values() for field in fields):
            row[field] = float((i * (j + 1)) % 13)
        rows.append(row)
    return rows


def runtime(mask=0):
    state = {"match_id": "future", "prediction_cutoff": "2025-03-02T00:00:00Z",
             "information_mode": "historical_reconstruction", "base": {"competition_id": "league",
             "home_team_id": "h", "away_team_id": "a", "kickoff": "2025-03-03T00:00:00Z"}, "groups": {}}
    for i, (group, fields) in enumerate(GROUP_FIELDS.items()):
        state["groups"][group] = {"status": "observed" if mask & (1 << i) else "not_acquired",
            "values": {f: float(j + 2) for j, f in enumerate(fields)}, "available_at": "2025-02-01T00:00:00Z",
            "version": "features-v1", "evidence_refs": ["sha256:" + "a" * 64]}
    return state


@pytest.fixture(scope="module")
def model():
    return PartialPredictor().fit(training_rows(), as_of="2025-02-01T00:00:00Z", dataset_version="unit-fixture")


@pytest.mark.parametrize("mask", range(8))
def test_each_subset_is_its_own_normalized_model(model, mask):
    result = model.predict_partial(runtime(mask))
    assert result["mask_key"] == mask
    assert sum(result["probabilities"]) == pytest.approx(1)
    assert min(result["probabilities"]) > 0
    assert result["mask"] == [name for i, name in enumerate(GROUP_IDS) if mask & (1 << i)]
    if mask:
        assert model.models[mask].feature_names == [f for name in result["mask"] for f in GROUP_FIELDS[name]]


@pytest.mark.parametrize("status", ["not_acquired", "source_missing", "stale"])
def test_unobserved_values_and_labels_never_affect_predictions(model, status):
    state = runtime(1)
    state["groups"]["xg"]["status"] = status
    before = model.predict_partial(state)
    state["groups"]["xg"]["values"] = {"home_rolling_xg_for": float("inf"), "home_goals": 999}
    state["groups"]["xg"]["available_at"] = "invalid hidden timestamp"
    state["home_goals"] = 999
    state["input_manifest"] = {"full_feature_path": "hidden.json"}
    state["mask"] = ["elo", "xg"]
    assert model.predict_partial(state) == before
    assert sanitize_state(state)["groups"]["xg"] == {"status": status, "values": {}}


def test_nulls_use_only_subset_training_statistics(model):
    state = runtime(1)
    state["groups"]["elo"]["values"]["home_elo"] = None
    pred = model.predict_partial(state)
    assert sum(pred["probabilities"]) == pytest.approx(1)
    assert pred["mask"] == ["elo"]


def test_future_observation_and_stale_model_are_rejected(model):
    state = runtime(1)
    state["groups"]["elo"]["available_at"] = "2025-03-02T00:00:01Z"
    with pytest.raises(ValueError, match="unavailable"):
        model.predict_partial(state)
    state = runtime()
    state["prediction_cutoff"] = "2025-01-01T00:00:00Z"
    with pytest.raises(ValueError, match="training cutoff"):
        model.predict_partial(state)


def test_training_time_and_mode_validation():
    rows = training_rows()
    rows[-1]["result_available_at"] = "2025-02-01T00:00:00Z"
    with pytest.raises(ValueError, match="unavailable"):
        PartialPredictor().fit(rows, "2025-02-01T00:00:00Z", "test")
    rows = training_rows()
    rows[-1]["features_available_at"] = rows[-1]["kickoff"]
    with pytest.raises(ValueError, match="features are unavailable"):
        PartialPredictor().fit(rows, "2025-02-01T00:00:00Z", "test")
    rows = training_rows()
    rows[-1]["information_mode"] = "operational"
    with pytest.raises(ValueError, match="one information mode"):
        PartialPredictor().fit(rows, "2025-02-01T00:00:00Z", "test")


def test_artifact_roundtrip_and_mutation_detection(model, tmp_path):
    path = tmp_path / "partial.json"
    model.save(path)
    restored = PartialPredictor.load(path)
    for mask in range(8):
        assert model.predict_partial(runtime(mask)) == restored.predict_partial(runtime(mask))
    payload = model.to_state()
    payload["metadata"]["dataset_version"] = "tampered"
    with pytest.raises(ValueError, match="hash mismatch"):
        PartialPredictor.from_state(payload)
    assert model.metadata["dataset_version"] == "unit-fixture"


def test_low_support_falls_back_to_frozen_prior():
    tiny = PartialPredictor().fit(training_rows(3), "2025-02-01T00:00:00Z", "small")
    assert tiny.predict_partial(runtime(7))["probabilities"] == tiny.predict_partial(runtime())["probabilities"]
    assert tiny.predict_partial(runtime(7))["fallback_reason"] == "insufficient_training_matches"


def test_refs_have_no_file_capability_and_modes_are_isolated(model):
    state = runtime(1)
    state["groups"]["elo"]["evidence_refs"] = ["D:/data/full-snapshot.json"]
    with pytest.raises(ValueError, match="inert identifiers"):
        sanitize_state(state)
    state = runtime()
    state["information_mode"] = "operational"
    with pytest.raises(ValueError, match="modes differ"):
        model.predict_partial(state)


def test_training_manifest_lists_ids_and_cutoffs_for_each_subset(model):
    for mask in range(8):
        training = model.metadata["subsets"][str(mask)]["training"]
        assert len(training) == 30
        assert {r["match_id"] for r in training} == {f"m{i}" for i in range(30)}
        assert all(r["result_available_at"] < model.metadata["training_as_of"] for r in training)
