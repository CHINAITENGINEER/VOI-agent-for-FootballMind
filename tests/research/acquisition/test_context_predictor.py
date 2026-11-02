"""Constructed model/serialization checks, not real information-value results."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import socket

import numpy as np
import pytest

from football_research.acquisition.context_predictor import ContextPredictor, training_readiness, PARAMETERS, _state
from football_research.acquisition.visible_state import encode_visible_state


AS_OF = "2025-01-15T00:00:00+00:00"
WORDS = ("Available home squad alpha forward returning to training.",
         "Balanced evenly matched teams beta cautious preparation.",
         "Away squad gamma returning players and training confidence.")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Context predictor must not use the network")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)


def visible(mid="provider:future", *, day=20, text=WORDS[0], source="www.uefa.com", headline=False):
    start = datetime(2025, 1, day, 10, tzinfo=timezone.utc)
    base = {"state": {"match_id": mid, "prediction_cutoff": (start+timedelta(hours=8)).isoformat(),
                      "base": {"kickoff": (start+timedelta(hours=9)).isoformat()}, "action_history": []},
            "decision_at": (start+timedelta(seconds=1)).isoformat(), "documents": [], "attempts": [],
            "prediction": {"probabilities": [.45, .3, .25], "probability_basis": "observed_market_snapshot"}}
    if text is not None:
        doc = {"document_id": "doc-"+mid, "raw_sha256": "a"*64, "url": "https://"+source+"/news/story/",
            "status": "OK", "kind": "article", "text": text, "links": [],
            "observed_at": start.isoformat(), "available_at": start.isoformat(), "source_published_at": None,
            "structured_news": {"version": "structured-news-discovery-v3", "role": "article",
                "segments": [{"text": text, "start": 0, "end": len(text),
                              "content_source": "json_ld_article_body", "json_pointer": "script[0].articleBody"}]}}
        if headline:
            doc.update(kind="index", text="UNSTRUCTURED_NAVIGATION")
            doc.pop("structured_news")
            doc["links"] = [{"url": "https://"+source+"/news/unread/", "title": text,
                "observed_anchors": [{"anchor_index": 0, "text": text, "text_truncated": False}],
                "observed_anchors_truncated": False}]
        base["documents"].append(doc)
    return encode_visible_state(base)


def records():
    result = []
    for i in range(12):
        for text in (None, WORDS[i % 3]):
            result.append({"physical_match_id": f"official:{i}",
                "visible_state": visible(f"provider:{i}", day=i+1, text=text),
                "outcome": i % 3, "result_available_at": f"2025-01-{i+1:02}T22:00:00+00:00"})
    return result


@pytest.fixture(scope="module")
def model():
    return ContextPredictor().fit(records(), as_of=AS_OF, dataset_version="constructed-unit-input-v1",
                                  evidence_origin="constructed_mechanism")


def test_fixed_true_independent_match_gate_and_no_class_imputation():
    assert not training_readiness([], as_of=AS_OF, evidence_origin="real_observational")["ready_to_fit"]
    six = records()[:12]
    labels = [1, 1, 2, 2, 2, 2]
    for i, row in enumerate(six):
        row["outcome"] = labels[i//2]
    report = training_readiness(six*10, as_of=AS_OF, evidence_origin="real_observational")
    assert report["physical_matches"] == 6 and report["distinct_prefixes"] == 12
    assert report["class_counts"] == {"0": 0, "1": 2, "2": 4}
    assert len(report["reasons"]) == 2
    unfitted = ContextPredictor()
    with pytest.raises(ValueError, match="not_ready"):
        unfitted.fit(six, as_of=AS_OF, dataset_version="six-match-test", evidence_origin="real_observational")
    assert not unfitted.model_version and not unfitted.coefficients
    assert PARAMETERS["min_physical_matches"] == 12 and PARAMETERS["regularization"] == 10.


def test_market_offset_predicts_positive_normalized_probabilities(model):
    result = model.predict(visible(), physical_match_id="official:future")
    assert abs(sum(result["probabilities"])-1) < 1e-12 and min(result["probabilities"]) > 0
    assert result["prediction_kind"] == "regularized_context_forecast"
    assert result["probabilities"] != [.45, .3, .25]
    expected = np.exp(np.log(result["prior_probabilities"])+result["logit_residual"])
    np.testing.assert_allclose(result["probabilities"], expected/expected.sum(), rtol=0, atol=1e-15)
    assert result["evidence_origin"] == "constructed_mechanism" and not result["production_ready"]
    assert result["calibration_status"] == "not_established"


@pytest.mark.parametrize("text,reason", [(None, "no_visible_text"), ("龍龘麤靐龖龗", "all_text_out_of_vocabulary")])
def test_empty_and_oov_are_exact_prior_with_zero_correction(model, text, reason):
    state = visible(text=text)
    state["before"]["history"]["attempts"] = [{"status": "FAILED"}, {"status": "NO_DATA"}]
    result = model.predict(state, physical_match_id="official:unseen")
    assert result["probabilities"] == state["before"]["prior_probabilities"]
    assert result["logit_residual"] == [0., 0., 0.]
    assert result["fallback_reason"] == reason and result["support_status"] == "prior_only"


def test_visible_parent_labels_can_be_encoded_without_unread_body(model):
    state = visible(text=WORDS[0], headline=True)
    assert state["before"]["contents"] == []
    result = model.predict(state, physical_match_id="official:headline")
    assert result["known_character_features"] > 0 and result["fallback_reason"] is None
    state["candidates"]["unread"] = {"future_body": WORDS[2], "hidden_outcome": 2, "expected_gain": 100}
    assert model.predict(state, physical_match_id="official:headline") == result


def test_current_posterior_cost_and_fixture_identifier_are_not_text_features(model):
    state = visible()
    result = model.predict(state, physical_match_id="official:future")
    changed = deepcopy(state)
    changed["before"]["current_probabilities"] = [.999, .0005, .0005]
    changed["before"]["budget"] = {"limits": {"requests": 999, "money": 999}, "spent": {}}
    changed["before"]["fixture"].update(home_team_id="hidden-winner", away_team_id="future-score")
    changed["outcome"] = 2
    changed["future_returns"] = {"unread-body": WORDS[2]}
    assert model.predict(changed, physical_match_id="official:future") == result


def test_unseen_source_is_explicit_extrapolation_not_calibrated_support(model):
    result = model.predict(visible(source="unknown-source.example"), physical_match_id="official:new-source")
    assert result["unknown_sources"] == ["unknown-source.example"]
    assert result["support_status"] == "source_extrapolation"
    assert result["calibration_status"] == "not_established"


def test_equivalent_duplicate_prefixes_and_runs_do_not_reweight_fixture(model):
    rows = records()
    duplicated = rows+[deepcopy(rows[1]) for _ in range(30)]
    other = ContextPredictor().fit(duplicated, as_of=AS_OF, dataset_version="duplicates-test",
                                   evidence_origin="constructed_mechanism")
    assert other.encoder == model.encoder
    np.testing.assert_allclose(other.coefficients, model.coefficients, rtol=0, atol=1e-14)
    assert other.metadata["readiness"]["distinct_prefixes"] == 24
    assert other.metadata["readiness"]["physical_matches"] == 12


def test_physical_aliases_stay_one_fixture_and_conflicting_aliases_are_rejected():
    rows = records()
    alias = deepcopy(rows[0])
    alias["visible_state"]["before"]["match_id"] = "other-provider:first"
    report = training_readiness(rows+[alias], as_of=AS_OF, evidence_origin="constructed_mechanism")
    assert report["physical_matches"] == 12 and report["distinct_prefixes"] == 24
    alias["physical_match_id"] = "official:other"
    alias["visible_state"]["before"]["match_id"] = rows[0]["visible_state"]["before"]["match_id"]
    with pytest.raises(ValueError, match="provider identity"):
        training_readiness(rows+[alias], as_of=AS_OF, evidence_origin="constructed_mechanism")


@pytest.mark.parametrize("mutation,reason", [
    (lambda r: r.update(purpose="confirmation"), "f_train"),
    (lambda r: r.update(result_available_at="2025-01-16T00:00:00Z"), "Settlement"),
    (lambda r: r["visible_state"]["before"].update(decision_at="2025-01-01T20:00:00Z"), "prematch"),
    (lambda r: r["visible_state"]["before"]["documents"][0].update(available_at="2025-01-02T00:00:00Z"), "clock"),
])
def test_fit_rejects_wrong_cohort_and_unavailable_inputs(mutation, reason):
    rows = records()
    mutation(rows[1])
    with pytest.raises(ValueError, match=reason):
        ContextPredictor().fit(rows, as_of=AS_OF, dataset_version="bad", evidence_origin="constructed_mechanism")


def test_prediction_rejects_training_physical_or_provider_identity_and_time(model):
    with pytest.raises(ValueError, match="Training physical"):
        model.predict(visible(), physical_match_id="official:0")
    with pytest.raises(ValueError, match="Training physical"):
        model.predict(visible(mid="provider:0"), physical_match_id="renamed-physical")
    with pytest.raises(ValueError, match="training cutoff"):
        model.predict(visible(day=13), physical_match_id="official:earlier")


def test_same_fixture_inconsistent_labels_are_not_more_supervision():
    rows = records()
    rows[1]["outcome"] = 2
    with pytest.raises(ValueError, match="Conflicting outcome"):
        training_readiness(rows, as_of=AS_OF, evidence_origin="constructed_mechanism")


def test_frozen_model_load_preserves_probabilities_and_refuses_mutation(model, tmp_path):
    file = tmp_path/"f.json"
    model.save(file)
    loaded = ContextPredictor.load(file)
    assert loaded.model_version == model.model_version and loaded.encoder == model.encoder
    assert loaded.predict(visible(), physical_match_id="official:future") == model.predict(visible(), physical_match_id="official:future")
    with pytest.raises(FileExistsError):
        model.save(file)
    payload = json.loads(file.read_text(encoding="utf-8"))
    payload["coefficients"][0][0] += .1
    file.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="version mismatch"):
        ContextPredictor.load(file)


def test_dictionary_does_not_include_future_tokens_and_original_input_is_unchanged(model):
    old = deepcopy(model.encoder)
    sample = visible(text="FUTURE_QZXWV_TOKEN unseen contents.")
    original = deepcopy(sample)
    model.predict(sample, physical_match_id="official:future")
    assert model.encoder == old and sample == original
    assert all(token not in model.encoder["vocabulary"] for token in ("qzx", "zxw", "xwv"))


def test_objective_has_equal_fixture_mass_not_equal_prefix_mass():
    rows = records()
    for text in (WORDS[0]+" additional public sentence.", WORDS[0]+" another public sentence."):
        extra = deepcopy(rows[1])
        extra["visible_state"] = visible("provider:0", day=1, text=text)
        rows.append(extra)
    model = ContextPredictor().fit(rows, as_of=AS_OF, dataset_version="weighted-prefixes", evidence_origin="constructed_mechanism")
    losses = {}
    for row in rows:
        state = _state(row["visible_state"])
        v = model._vector(state)[0]
        logits = np.log(state["prior"])+v@np.asarray(model.coefficients)
        p = np.exp(logits-np.max(logits)); p /= p.sum()
        losses.setdefault(row["physical_match_id"], []).append(-np.log(p[row["outcome"]]))
    expected = np.mean([np.mean(values) for values in losses.values()])+5*np.square(model.coefficients).sum()
    assert abs(expected-model.metadata["optimizer"]["objective"]) < 1e-12
