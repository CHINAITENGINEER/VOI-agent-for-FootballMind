"""Constructed mechanisms, never evidence of real football accuracy."""
from copy import deepcopy
import json

import pytest

from football_research.acquisition.text_returns import TextReturnModel


FIRST = {"tool": "read_url", "url": "https://club.example/news/first-report"}
SECOND = {"tool": "read_url", "url": "https://club.example/news/second-report"}
TEXTS = ["amber cedar midfield amber cedar", "cobalt birch backline cobalt birch",
         "violet willow warmup violet willow"]
P = [1/3, 1/3, 1/3]


def response(token):
    return {"status": "OK", "kind": "article", "text": TEXTS[token]}


def record(match_id, y, token, action=FIRST, history=None):
    return {"match_id": match_id, "prediction_cutoff": "2026-01-01T10:00:00+00:00",
            "result_available_at": "2026-01-02T10:00:00+00:00", "prior_probabilities": P,
            "outcome": y, "action": action, "response": response(token), "history": history or []}


def rows(*, sequential=True, neutral=True):
    data = []
    for y in range(3):
        for j in range(20):
            z = y if j < 12 else (y+1+j % 2) % 3
            item = record(f"match-{y}-{j}", y, z)
            data.append(item)
            if sequential:
                data.append(record(item["match_id"], y, z, SECOND,
                                   [{"action": FIRST, "response": response(z)}]))
        if neutral:
            for status, kind in [("FAILED", "article"), ("NO_DATA", "article"), ("OK", "index"), ("OK", "discovery")]:
                item = record(f"neutral-{y}-{status}-{kind}", y, y)
                item["response"] = {"status": status, "kind": kind, "text": ""}
                data.append(item)
    return data


def fit(data=None, **kwargs):
    return TextReturnModel().fit(rows() if data is None else data,
        as_of="2026-02-01T00:00:00+00:00", dataset_version="constructed-test-only",
        evidence_origin="constructed_mechanism", text_clusters=3, action_clusters=2,
        min_matches=3, min_class_matches=.5, **kwargs)


@pytest.fixture(scope="module")
def model():
    return fit()


def state(model):
    return model.start(P, "future-fixture", "2026-03-01T10:00:00+00:00")


def test_branch_mixture_and_actual_update_share_model(model):
    before = state(model)
    decision = model.enumerate(before, FIRST, cost=.01)
    assert decision["status"] == "supported"
    assert decision["value"]["gross_evsi"] > 0
    assert sum(b["probability"] for b in decision["branches"]) == pytest.approx(1)
    assert [sum(b["probability"]*b["posterior"][y] for b in decision["branches"]) for y in range(3)] == pytest.approx(P)
    result = model.update(before, FIRST, response(0), decision=decision)
    branch = next(b for b in decision["branches"] if b["branch_id"] == result["observed_token"])
    assert result["status"] == "updated"
    assert result["prediction_after"] == branch["posterior"]
    assert model.predict(result["state"])["probabilities"] == branch["posterior"]
    assert before["history"] == []


def test_already_read_information_changes_next_value(model):
    before = state(model)
    first = model.enumerate(before, FIRST)
    after = model.update(before, FIRST, response(0), decision=first)["state"]
    second = model.enumerate(after, SECOND, cost=.01)
    assert second["status"] == "supported"
    assert second["value"]["gross_evsi"] == pytest.approx(0, abs=1e-12)
    assert second["value"]["net_voi"] == pytest.approx(-.01)
    assert second["value"]["gross_evsi"] < first["value"]["gross_evsi"]
    assert after["prior_probabilities"] == P


def test_single_response_records_do_not_support_a_later_history():
    single = fit(rows(sequential=False))
    before = state(single)
    after = single.update(before, FIRST, response(0))["state"]
    next_value = single.enumerate(after, FIRST)
    assert next_value["status"] == "unknown"
    assert next_value["reason"] == "no_matching_acquisition_history"
    assert next_value["value"]["gross_evsi"] is None


@pytest.mark.parametrize("status,kind,token", [("FAILED", "article", "failure"),
    ("NO_DATA", "article", "no_data"), ("OK", "index", "index"), ("OK", "discovery", "discovery")])
def test_operational_returns_keep_prior_but_record_history(model, status, kind, token):
    before = state(model)
    decision = model.enumerate(before, FIRST)
    branch = next(b for b in decision["branches"] if b["branch_id"] == token)
    assert branch["posterior"] == pytest.approx(P)
    after = model.update(before, FIRST, {"status": status, "kind": kind, "text": ""}, decision=decision)
    assert after["status"] == "updated"
    assert after["prediction_after"] == pytest.approx(P)
    assert len(after["state"]["history"]) == 1
    assert model.enumerate(after["state"], FIRST)["status"] == "unknown"


def test_oov_response_preserves_probability_and_poisoned_history_never_falls_back(model):
    before = state(model)
    after = model.update(before, FIRST, {"status": "OK", "kind": "article", "text": "quartz xenolith zephyr"})
    assert after["status"] == "unknown"
    assert after["prediction_after"] == P
    assert len(after["state"]["history"]) == 1
    assert after["state"]["history"][0]["unsupported"]
    assert model.enumerate(after["state"], SECOND)["reason"] == "unsupported_observed_history"
    assert "xenolith" not in model.text_codebook.payload["vocabulary"]


def test_unseen_source_and_oov_request_are_unknown(model):
    before = state(model)
    unseen = model.enumerate(before, {"tool": "read_url", "url": "https://unseen.example/news/first-report"})
    assert unseen["reason"] == "unseen_action_source"
    oov = model.enumerate(before, {"tool": "read_url", "url": "https://club.example/xenolith"})
    assert oov["reason"] == "out_of_vocabulary_action"
    assert oov["value"]["gross_evsi"] is None


def test_unseen_operational_return_is_not_silently_given_a_branch():
    articles = fit(rows(neutral=False))
    result = articles.update(state(articles), FIRST, {"status": "FAILED", "kind": "article", "text": ""})
    assert result["status"] == "unknown"
    assert result["state"]["history"][0]["unsupported"]


def test_frozen_binding_rejects_other_action_changed_state_and_tampered_branch(model):
    before = state(model)
    decision = model.enumerate(before, FIRST)
    with pytest.raises(ValueError, match="frozen model/state/action"):
        model.update(before, SECOND, response(0), decision=decision)
    changed = deepcopy(before)
    changed["match_id"] = "another-future-fixture"
    with pytest.raises(ValueError, match="frozen model/state/action"):
        model.update(changed, FIRST, response(0), decision=decision)
    changed_decision = deepcopy(decision)
    changed_decision["branches"][0]["probability"] = .99
    with pytest.raises(ValueError, match="digest"):
        model.update(before, FIRST, response(0), decision=changed_decision)


def test_probability_changes_require_observed_frozen_return_history(model):
    before = state(model)
    before["current_probabilities"] = [.8, .1, .1]
    with pytest.raises(ValueError, match="disagree with frozen observed history"):
        model.predict(before)
    after = model.update(state(model), FIRST, response(0))["state"]
    after["current_probabilities"] = [.8, .1, .1]
    with pytest.raises(ValueError, match="disagree with frozen observed history"):
        model.enumerate(after, SECOND)


def test_each_match_gets_at_most_one_support_weight(model):
    assert model.enumerate(state(model), FIRST)["support"]["max_match_weight"] <= 1+1e-12
    duplicate = fit(rows()*3)
    assert duplicate.metadata["training_records"] == model.metadata["training_records"]
    assert duplicate.enumerate(state(duplicate), FIRST)["branches"] == model.enumerate(state(model), FIRST)["branches"]


def test_smoothing_cannot_invent_missing_outcome_support():
    sparse = fit([r for r in rows() if r["outcome"] != 2], dirichlet_mass=1000.)
    result = sparse.enumerate(state(sparse), FIRST)
    assert result["status"] == "unknown"
    assert result["value"]["gross_evsi"] is None
    assert result["support"]["smoothing_counts_as_support"] is False


def test_unsettled_or_late_records_cannot_train():
    data = rows()
    data[0]["outcome"] = None
    with pytest.raises(ValueError, match="independently settled"):
        fit(data)
    data = rows()
    data[0]["result_available_at"] = "2027-01-01T00:00:00Z"
    with pytest.raises(ValueError, match="Settlement"):
        fit(data)
    data = rows()
    data[0]["observed_at"] = "2026-01-03T00:00:00Z"
    with pytest.raises(ValueError, match="Late text"):
        fit(data)
    with pytest.raises(ValueError, match="Independent settled"):
        fit([])


def test_unknown_page_role_cannot_enter_semantic_training():
    data = rows()
    data[0]["response"]["kind"] = "unknown"
    with pytest.raises(ValueError, match="Unclassified response"):
        fit(data)


def test_future_model_and_training_match_cannot_run(model):
    with pytest.raises(ValueError, match="training match"):
        model.start(P, "match-0-0", "2026-03-01T00:00:00Z")
    with pytest.raises(ValueError, match="trained after"):
        model.start(P, "new", "2026-01-01T00:00:00Z")


def test_save_load_exact_and_never_claim_production_readiness(model, tmp_path):
    destination = tmp_path/"text.json"
    model.save(destination)
    restored = TextReturnModel.load(destination)
    assert restored.model_version == model.model_version
    assert restored.enumerate(state(restored), FIRST) == model.enumerate(state(model), FIRST)
    assert restored.metadata["evidence_origin"] == "constructed_mechanism"
    assert restored.metadata["production_ready"] is False
    payload = json.loads(destination.read_text(encoding="utf-8"))
    payload["records"][0]["outcome"] = 2
    destination.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="content mismatch"):
        TextReturnModel.load(destination)
