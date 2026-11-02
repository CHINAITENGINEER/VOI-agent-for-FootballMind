"""Fixed-gate neighborhood mechanisms, never real football efficacy evidence."""
from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from football_research.acquisition.text_returns import TextReturnModel, _digest
from football_research.acquisition.text_history import (
    HistoryNeighborhoodTextReturnModel, load_text_model, history_swap_distance,
    VERSION, STATE_VERSION, NEIGHBOR_WEIGHT,
)
from football_research.acquisition.text_risk import TextRiskCalibrator


P = [1/3, 1/3, 1/3]
ACTIONS = [{"tool": "read_url", "url": "https://club.example/"+word+"-"+word}
           for word in ("alpha", "beta", "gamma", "delta")]
RESPONSES = [{"status": "OK", "kind": "article", "text": word+" "+word}
             for word in ("amber", "cobalt", "violet")]


def records(per_class=12):
    data = []
    for outcome in range(3):
        for match in range(per_class):
            base = {"match_id": f"training-{outcome}-{match}", "outcome": outcome,
                "prior_probabilities": P, "prediction_cutoff": "2026-01-01T10:00:00Z",
                "result_available_at": "2026-01-02T10:00:00Z"}
            for action, response, history in ((ACTIONS[0], RESPONSES[0], []),
                    (ACTIONS[1], RESPONSES[1], []),
                    (ACTIONS[1], RESPONSES[1], [{"action": ACTIONS[0], "response": RESPONSES[0]}]),
                    (ACTIONS[0], RESPONSES[0], [{"action": ACTIONS[1], "response": RESPONSES[1]}]),
                    (ACTIONS[2], RESPONSES[outcome], [{"action": ACTIONS[0], "response": RESPONSES[0]},
                                                   {"action": ACTIONS[1], "response": RESPONSES[1]}])):
                data.append({**base, "action": action, "response": response, "history": history})
    return data


def fit(cls=HistoryNeighborhoodTextReturnModel, data=None, **kwargs):
    return cls().fit(records() if data is None else data, as_of="2026-02-01T00:00:00Z",
        dataset_version="constructed-history-neighborhood-only", evidence_origin="constructed_mechanism",
        text_clusters=3, action_clusters=3, min_matches=kwargs.pop("min_matches", 12),
        min_class_matches=1., **kwargs)


@pytest.fixture(scope="module")
def models():
    return fit(TextReturnModel), fit()


def state(model, order=()):
    current = model.start(P, "unseen-match", "2026-03-01T10:00:00Z")
    for index in order:
        result = model.update(current, ACTIONS[index], RESPONSES[index])
        assert result["status"] == "updated"
        current = result["state"]
    return current


def test_unseen_permitted_order_has_neighbor_support_without_erasing_actual_order(models):
    exact, nearby = models
    exact_state, nearby_state = state(exact, (1, 0)), state(nearby, (1, 0))
    missing = exact.enumerate(exact_state, ACTIONS[2])
    result = nearby.enumerate(nearby_state, ACTIONS[2])
    assert missing["reason"] == "no_matching_acquisition_history"
    assert result["status"] == "supported" and result["value"]["gross_evsi"] > 0
    assert nearby_state["schema"] == STATE_VERSION
    assert [entry["action"] for entry in nearby_state["history"]] == [ACTIONS[1], ACTIONS[0]]
    assert [entry["action"] for entry in exact_state["history"]] == [ACTIONS[1], ACTIONS[0]]
    assert result["support"]["history_length"] == 2
    assert result["support"]["history_neighbor_transition_count"] == 36
    assert result["support"]["history_weighted_match_mass"] == pytest.approx(36*NEIGHBOR_WEIGHT)
    # Training records retain AB; borrowing BA does not synthesize training BA-C.
    target = [r for r in nearby.records if r["action_digest"] == _digest(ACTIONS[2])]
    assert all(r["history_key"] == nearby._history_key(state(nearby, (0, 1))["history"]) for r in target)


def test_enumeration_update_and_state_replay_use_identical_neighborhood_likelihood(models):
    model = models[1]
    before = state(model, (1, 0))
    decision = model.enumerate(before, ACTIONS[2], cost=.02)
    prior = before["current_probabilities"]
    for outcome in range(3):
        assert math.fsum(b["probability"]*b["posterior"][outcome] for b in decision["branches"]) == pytest.approx(prior[outcome])
    for response in RESPONSES:
        result = model.update(before, ACTIONS[2], response, decision=decision)
        branch = next(b for b in decision["branches"] if b["branch_id"] == result["observed_token"])
        assert result["prediction_after"] == branch["posterior"]
        assert model.predict(result["state"])["probabilities"] == branch["posterior"]
        assert len(result["state"]["history"]) == 3
    tampered = deepcopy(before)
    tampered["history"].reverse()
    with pytest.raises(ValueError, match="frozen model/state/action"):
        model.update(tampered, ACTIONS[2], RESPONSES[0], decision=decision)


def test_scale_invariant_ess_cannot_erase_neighbor_support_penalty():
    model = fit(data=records(per_class=5))
    result = model.enumerate(state(model, (1, 0)), ACTIONS[2])
    assert result["support"]["effective_matches"] == pytest.approx(15)
    assert result["support"]["history_weighted_match_mass"] == pytest.approx(15*NEIGHBOR_WEIGHT)
    assert result["reason"] == "insufficient_history_neighbor_support"
    assert result["value"]["gross_evsi"] is None


def test_repeated_rows_are_not_extra_independent_history_support(models):
    repeated = fit(data=records()*3)
    current = repeated.enumerate(state(repeated, (1, 0)), ACTIONS[2])
    reference = models[1].enumerate(state(models[1], (1, 0)), ACTIONS[2])
    assert current == reference
    summary, _ = repeated._history_support([("one-match", 1.), ("one-match", NEIGHBOR_WEIGHT)]*5)
    assert summary["history_weighted_match_mass"] == pytest.approx((1.+NEIGHBOR_WEIGHT)/2)


def test_unknown_observation_poisoning_is_not_dropped_by_neighbor_lookup(models):
    model = models[1]
    current = state(model, (1, 0))
    result = model.update(current, ACTIONS[2], {"status": "OK", "kind": "article", "text": "xenolith unknownword"})
    assert result["status"] == "unknown" and result["state"]["history"][-1]["unsupported"]
    assert result["prediction_after"] == current["current_probabilities"]
    assert model.enumerate(result["state"], ACTIONS[0])["reason"] == "unsupported_observed_history"
    failed = model.update(state(model), ACTIONS[0], {"status": "FAILED", "kind": "unknown", "text": ""})
    assert failed["status"] == "unknown"
    assert model.enumerate(failed["state"], ACTIONS[1])["reason"] == "unsupported_observed_history"


def test_history_gates_preserve_multiplicity_request_revision_order_and_neutral_order():
    a, b, c = ["read_url:club.example", "0", "text:0"], ["read_url:club.example", "1", "text:1"], ["read_url:club.example", "2", "text:2"]
    revised = ["read_url:club.example", "0", "text:1"]
    failed = ["read_url:other.example", "0", "failure"]
    discovery = ["search_web:open_web", "0", "discovery"]
    assert history_swap_distance([a, b], [b, a]) == 1
    assert history_swap_distance([a, a, b], [a, b, a]) == 1
    assert history_swap_distance([a, a], [a]) is None
    assert history_swap_distance([a, revised], [revised, a]) is None
    assert history_swap_distance([failed, discovery], [discovery, failed]) is None
    assert history_swap_distance([a, failed], [failed, a]) == 1
    assert history_swap_distance([a], [a, failed]) is None
    assert history_swap_distance([a, b, c], [c, b, a]) is None
    assert history_swap_distance([a, b], [a, b]) == 0


def test_semantic_outcome_class_support_is_still_required():
    data = [row for row in records() if row["outcome"] != 2]
    model = fit(data=data)
    initial = model.enumerate(state(model), ACTIONS[0])
    assert initial["reason"] == "insufficient_conditional_support"
    assert initial["value"]["gross_evsi"] is None


def test_factory_freeze_restore_and_cross_schema_rejection(models, tmp_path):
    for index, model in enumerate(models):
        path = tmp_path/f"model-{index}.json"
        model.save(path)
        restored = load_text_model(path)
        assert type(restored) is type(model)
        assert restored.model_version == model.model_version
        assert restored.enumerate(state(restored, (1, 0)), ACTIONS[2]) == model.enumerate(state(model, (1, 0)), ACTIONS[2])
        restored.save(tmp_path/f"restored-{index}.json")
        assert path.read_bytes() == (tmp_path/f"restored-{index}.json").read_bytes()
    with pytest.raises(ValueError, match="schema"):
        TextReturnModel.load(tmp_path/"model-1.json")
    with pytest.raises(ValueError, match="schema"):
        HistoryNeighborhoodTextReturnModel.load(tmp_path/"model-0.json")
    with pytest.raises(ValueError, match="version mismatch"):
        models[1].predict(state(models[0]))


def test_rehashed_unregistered_spec_cannot_change_frozen_algorithm(models, tmp_path):
    path = tmp_path/"changed.json"
    models[1].save(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["metadata"]["history_kernel"]["max_adjacent_swaps"] = 2
    payload = {key: value for key, value in data.items() if key not in {"model_version", "content_sha256"}}
    data.update(content_sha256=_digest(payload), model_version=VERSION+":"+_digest(payload)[:24])
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="specification"):
        load_text_model(path)


def test_old_empirical_risk_cannot_silently_change_its_fixed_predictor(models):
    with pytest.raises(ValueError, match="content no longer matches its version"):
        TextRiskCalibrator().fit([], text_model=models[1], as_of="2026-04-01T00:00:00Z",
            dataset_version="not-valid", evidence_origin="constructed_mechanism")


def _resign(data, path):
    payload = {key: value for key, value in data.items() if key not in {"model_version", "content_sha256"}}
    data.update(content_sha256=_digest(payload), model_version=VERSION+":"+_digest(payload)[:24])
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.parametrize("key,value", [("min_matches", 0), ("min_matches", -1), ("min_matches", False),
    ("min_class_matches", 0), ("bandwidth", 0), ("dirichlet_mass", -1),
    ("text_clusters", 0), ("action_clusters", 33), ("action_clusters", True),
    ("min_action_similarity", 0), ("min_action_similarity", 1.01)])
def test_rehashed_impossible_fit_parameters_rejected(models, tmp_path, key, value):
    path = tmp_path/"changed.json"
    models[1].save(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["parameters"][key] = value
    _resign(data, path)
    with pytest.raises(ValueError, match="parameters"):
        load_text_model(path)


@pytest.mark.parametrize("change", ["late_label", "training_ids", "conflicting_outcome", "duplicate_row"])
def test_rehashed_inconsistent_training_metadata_rejected(models, tmp_path, change):
    path = tmp_path/"changed.json"
    models[1].save(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if change == "late_label":
        data["records"][0]["result_available_at"] = "2026-03-01T00:00:00Z"
    elif change == "training_ids":
        data["metadata"]["training_match_ids"] = []
    elif change == "conflicting_outcome":
        data["records"][0]["outcome"] = 2
    else:
        data["records"].append(deepcopy(data["records"][0]))
    _resign(data, path)
    with pytest.raises(ValueError, match="[Ff]rozen history-neighborhood"):
        load_text_model(path)


@pytest.mark.parametrize("location,key", [("history", "decision_at"), ("history", "available_at"),
    ("history_response", "observed_at"), ("current_response", "available_at")])
def test_provided_late_history_or_response_clock_cannot_train(location, key):
    data = deepcopy(records())
    row = next(item for item in data if item["history"])
    target = (row["history"][0] if location == "history" else row["history"][0]["response"]
              if location == "history_response" else row["response"])
    target[key] = "2026-01-03T00:00:00Z"
    with pytest.raises(ValueError, match="clock exceeds prediction cutoff"):
        fit(data=data)


def test_known_history_availability_must_precede_following_decision():
    data = deepcopy(records())
    row = next(item for item in data if len(item["history"]) == 2)
    row["history"][0]["available_at"] = "2026-01-01T09:30:00Z"
    row["history"][1]["decision_at"] = "2026-01-01T09:00:00Z"
    with pytest.raises(ValueError, match="not available before the following decision"):
        fit(data=data)


def test_existing_v1_frozen_artifact_is_byte_identical_after_loader_round_trip(tmp_path):
    path = Path(__file__).resolve().parents[3]/"var/parallel/agent/round178/evaluation/frozen-text-model.json"
    if not path.exists():
        pytest.skip("Old Round178 artifact is not included in this checkout")
    model = load_text_model(path)
    output = tmp_path/"same-v1.json"
    model.save(output)
    assert output.read_bytes() == path.read_bytes()
    assert "history_kernel" not in model.metadata
