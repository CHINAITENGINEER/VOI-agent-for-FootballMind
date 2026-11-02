"""Input isolation and real f-update regressions for the postread entry point."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from football_research.acquisition.entity_semantic_f import FEATURES, _predict
from football_research.acquisition.postread_selector import (
    BodyCandidate, FEATURE_FIELDS, PostreadSelector, PostreadSession, sha256, text_sha256,
    SCHEMA, run_request,
)
from football_research.acquisition.semantic_f import EXTRACTION_VERSION, public_input


def record(text="Player One will miss this match through injury.", *, claims=True, mid="match1", value="unavailable"):
    inp = public_input(dict(physical_match_id=mid, home_name="Home", away_name="Away", kickoff=None),
                       dict(url="https://club.example/news/team", title="Team news", text=text), "body")
    cs = [dict(team="home", entity="Player One", kind="availability", value=value, scope="this_match",
               statement="Explicit source statement", quotes=[text])] if claims else []
    return dict(input=inp, response=dict(schema=EXTRACTION_VERSION, input_id=inp["input_id"], claims=cs))


def candidate(name="a", **kwargs):
    r = record(**kwargs)
    return BodyCandidate.from_record(candidate_id=name, action_id="act-"+name,
                                     body_sha256=text_sha256(r["input"]["text"]), record=r)


class FixedSelector:
    f_sha256 = "test-f"
    g_sha256 = "test-g"
    model_names = ("positive", "negative")

    def predict_f(self, projection):
        # Hand-set nonzero f weights, not a trained research model.
        coef = np.zeros((192, 3))
        coef[48+FEATURES.index("home:availability:unavailable")] = [-.2, .1, .1]
        return _predict(coef, projection).tolist()

    def score(self, row, *, model_name):
        assert set(row) == FEATURE_FIELDS
        return .02 if model_name == "positive" else -.02


def session(candidates=None, **kwargs):
    return PostreadSession(FixedSelector(), physical_match_id="match1", prior=[.5,.3,.2],
                           candidates=candidates or [candidate()], **kwargs)


def test_body_read_is_not_f_adoption_and_update_is_executed():
    s = session()
    initial = s.prediction()
    assert initial["used_input_ids"] == []
    d = s.choose(model_name="positive", remaining_updates=1)
    assert d["g_read_body_ids"] == ["a"] and d["f_adopted_body_ids"] == []
    assert d["selected_action_id"] == "act-a"
    result = s.adopt("a")
    assert result["probability_l1"] > 0
    assert result["p_after"][0] < result["p_before"][0]
    assert result["used_input_ids_after"] == [candidate().input_id]
    assert result["realized_gain"] is None and result["outcome"] is None
    assert s.choose(model_name="positive", remaining_updates=1)["stop_reason"] == "no_unadopted_read_candidate"


def test_completed_empty_claims_is_a_real_f_evaluation_not_failure():
    s = session([candidate(claims=False)])
    update = s.adopt("a")
    assert update["modeled_claims"] == 0
    assert update["f_executed"] and update["used_input_ids_after"]
    np.testing.assert_allclose(update["p_before"], update["p_after"], atol=1e-15)


def test_duplicate_and_conflicting_claims_are_state_dependent():
    a, duplicate = candidate(), candidate("b")
    conflict = candidate("c", text="Player One is available for this match.", value="available")
    s = session([a, duplicate, conflict])
    s.adopt("a")
    duplicate_features = s.feature("b")["semantic_numeric"]
    assert duplicate_features[2] == 1 and sum(duplicate_features[5:]) == 0
    assert s.feature("c")["semantic_numeric"][4] == 1
    update = s.adopt("b")
    assert update["probability_l1"] == 0


def test_unread_or_reused_body_cannot_be_selected_or_adopted():
    s = session(unread_candidate_ids=["unread"])
    for cid in ("unread", "missing"):
        with pytest.raises(ValueError): s.feature(cid)
        with pytest.raises(ValueError): s.adopt(cid)
    s.adopt("a")
    with pytest.raises(ValueError): s.adopt("a")


def test_invalid_claim_is_rejected_not_an_empty_body():
    r = record()
    r["response"]["claims"][0]["value"] = "unmodeled"
    with pytest.raises(ValueError, match="predicate/value"):
        BodyCandidate.from_record(candidate_id="a", action_id="a", body_sha256=text_sha256(r["input"]["text"]), record=r)


def test_input_id_and_body_sha_are_bound():
    r = record()
    with pytest.raises(ValueError, match="Body SHA"):
        BodyCandidate.from_record(candidate_id="a", action_id="a", body_sha256="wrong", record=r)
    r["response"]["input_id"] = "wrong"
    with pytest.raises(ValueError, match="binding differs"):
        BodyCandidate.from_record(candidate_id="a", action_id="a", body_sha256=text_sha256(r["input"]["text"]), record=r)


def test_cross_match_records_and_duplicate_ids_are_rejected():
    with pytest.raises(ValueError, match="another physical match"):
        session([candidate(mid="other")])
    with pytest.raises(ValueError, match="unique"):
        session([candidate(), candidate()])


def test_known_title_slot_cannot_hide_an_unselected_body():
    with pytest.raises(ValueError, match="Only title records"):
        session(known_title_records=[record()])


def test_negative_g_is_only_an_inference_stop_and_does_not_erase_candidate():
    s = session()
    d = s.choose(model_name="negative", remaining_updates=2)
    assert d["stop_reason"] == "nonpositive_predicted_net_gain"
    assert len(d["candidates"]) == 1 and not s.trace
    # A controlled exploration caller may still exercise a rejected candidate.
    assert s.adopt("a")["f_executed"]


def test_budget_and_incremental_cost_are_explicit():
    s = session()
    assert s.choose(model_name="positive", remaining_updates=0)["stop_reason"] == "budget_exhausted"
    assert s.choose(model_name="positive", remaining_updates=1, incremental_cost=.03)["selected_candidate_id"] is None
    for bad in (-1, True, .5):
        with pytest.raises(ValueError): s.choose(model_name="positive", remaining_updates=bad)


def test_failed_f_update_rolls_back_without_a_fake_after(monkeypatch):
    s = session()
    good = s.selector.predict_f
    def fail_when_adopted(projection):
        if sum(projection["novel_body"]) > 0:
            raise ValueError("intentional failed f")
        return good(projection)
    monkeypatch.setattr(s.selector, "predict_f", fail_when_adopted)
    with pytest.raises(ValueError, match="intentional"):
        s.adopt("a")
    assert s.adopted_ids == [] and s.trace == []


def test_wrong_f_binding_is_rejected_before_loading_pickle(tmp_path):
    f = tmp_path/"f.json"; f.write_text("{}", encoding="utf-8")
    (tmp_path/"model.json").write_text(json.dumps(dict(schema="unified-f-postread-g-experiment-v1",
                                                       f_model_sha256="wrong")), encoding="utf-8")
    with pytest.raises(ValueError, match="Paired f SHA"):
        PostreadSelector(tmp_path, f)


@pytest.fixture(scope="module")
def real_selector():
    root = Path("E:/FootballMind/experiments")
    g = root/"postread-clean-expansion-20261025-v1/training"
    f = root/"unified-f-postread-g-20261025-v1/replay/frozen-f.json"
    if not g.exists() or not f.exists():
        pytest.skip("Existing frozen research models not mounted")
    return PostreadSelector(g, f)


@pytest.mark.parametrize("field", ["outcome_class", "prediction_after", "realized_gain", "loss_after"])
def test_real_g_refuses_target_fields(real_selector, field):
    row = session().feature("a")
    row[field] = 0
    with pytest.raises(ValueError, match="allowlist"):
        real_selector.score(row, model_name="fulltext_state_ridge")


def test_encoder_changes_fulltext_design_without_changing_title_local(real_selector):
    row = session().feature("a")
    changed = deepcopy(row)
    changed["candidate_text"] = "The goalkeeper injury suspension away match squad team unavailable"
    first, second = real_selector.design(row), real_selector.design(changed)
    np.testing.assert_array_equal(first["title_local"], second["title_local"])
    assert not np.array_equal(first["fulltext_state"], second["fulltext_state"])


def test_f_model_on_current_online_path_cannot_be_silently_paired(real_selector):
    root = Path(__file__).resolve().parents[2]
    online_f = root/"var/reviews/fm-historical-transition-g-training-20261014-v2/f-labeler.json"
    if not online_f.exists():
        pytest.skip("Old online model not mounted")
    assert sha256(online_f) != real_selector.f_sha256
    with pytest.raises(ValueError, match="Paired f SHA"):
        PostreadSelector("E:/FootballMind/experiments/postread-clean-expansion-20261025-v1/training", online_f)


def request():
    c = candidate()
    return dict(schema=SCHEMA, physical_match_id="match1", prior=[.5,.3,.2], candidates=[dict(
        candidate_id=c.candidate_id, action_id=c.action_id, body_sha256=c.body_sha256, record=c.record, clocks={})],
        known_title_records=[], hours_to_kickoff=None, evidence_order=["a"], unread_candidate_ids=[],
        model_name="positive", max_updates=1, incremental_cost_loss_units=0.)


def test_label_free_single_match_entry_executes_and_stops():
    result = run_request(FixedSelector(), request())
    assert len(result["f_updates"]) == 1
    assert result["p_after"] != result["p_before"]
    assert result["stop_reason"] == "budget_exhausted"
    assert result["outcome"] is None and result["realized_gain"] is None


def test_single_match_entry_refuses_outcome_even_when_ignored_by_policy():
    req = request(); req["outcome_class"] = 0
    with pytest.raises(ValueError, match="allowlist"):
        run_request(FixedSelector(), req)
