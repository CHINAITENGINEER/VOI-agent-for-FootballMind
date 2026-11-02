from copy import deepcopy
from datetime import datetime,timedelta,timezone
import json

import pytest

from football_research.acquisition.contextual_identity import build_crosswalk
from football_research.acquisition.contextual_partial import ContextualPartialPredictor
from football_research.acquisition.state import GROUP_FIELDS


def _walk(tmp_path):
    rows = [{"source":"understat","entity_type":"team","source_id":str(i+70),"entity_id":f"t{i}","status":"source_scoped","valid_to":None} for i in range(5)]
    rows += [{"source":"understat","entity_type":"competition","source_id":"EPL","entity_id":"epl","status":"source_scoped","valid_to":None}]
    (tmp_path/"id_mappings.jsonl").write_text("\n".join(json.dumps(r) for r in rows),encoding="utf-8")
    return build_crosswalk([tmp_path])


def _rows():
    records = []
    start = datetime(2025,1,1,tzinfo=timezone.utc)
    for i in range(45):
        kickoff = start+timedelta(days=i)
        row = {"match_id":f"m{i}","competition_id":"epl","home_team_id":"t0" if i%3 == 0 else "t1",
            "away_team_id":"t2","kickoff":kickoff.isoformat(),"prediction_cutoff":(kickoff-timedelta(days=1)).isoformat(),
            "result_available_at":(kickoff+timedelta(hours=3)).isoformat(),"information_mode":"historical_reconstruction",
            "home_goals":2 if i%3 == 0 else 0,"away_goals":0 if i%3 != 2 else 2,"score_period":"regulation"}
        row.update({f:float(2+i%5) for fields in GROUP_FIELDS.values() for f in fields})
        records.append(row)
    return records


def _state(home="t0"):
    return {"match_id":"future","prediction_cutoff":"2025-04-01T00:00:00Z","information_mode":"historical_reconstruction",
        "base":{"competition_id":"epl","home_team_id":home,"away_team_id":"t2","kickoff":"2025-04-02T00:00:00Z"},
        "groups":{g:{"status":"not_acquired","values":{}} for g in GROUP_FIELDS}}


def _fit(tmp_path,rows=None):
    return ContextualPartialPredictor(_walk(tmp_path)).fit(rows or _rows(),as_of="2025-03-01T00:00:00Z",dataset_version="synthetic-test")


def test_empty_mask_uses_free_identity_and_all_masks_share_vocabulary(tmp_path):
    model = _fit(tmp_path)
    strong,weak = model.predict_partial(_state("t0")),model.predict_partial(_state("t1"))
    assert strong["mask_key"] == weak["mask_key"] == 0
    assert strong["home_win_probability"] > weak["home_win_probability"]+.1
    assert all(h["categorical_fit"] == model.heads["0"]["categorical_fit"] for h in model.heads.values())
    assert "home_elo" not in model.heads["0"]["numeric_names"]
    assert "home_elo" in model.heads["1"]["numeric_names"]


def test_hidden_or_outcome_payload_never_changes_empty_prediction(tmp_path):
    model = _fit(tmp_path)
    clean,dirty = _state(),_state()
    dirty.update(home_goals=999,result="home",label=0,rotation_intent="known")
    dirty["base"]["home_elo"] = 999999
    for group in dirty["groups"].values():
        group["values"] = {"home_elo":object(),"home_rolling_xg_for":float("nan")}
    assert model.predict_partial(clean) == model.predict_partial(dirty)


def test_unobserved_numeric_training_fields_do_not_change_empty_head(tmp_path):
    rows = _rows()
    walk = _walk(tmp_path)
    a = ContextualPartialPredictor(walk).fit(rows,as_of="2025-03-01T00:00:00Z",dataset_version="synthetic")
    modified = deepcopy(rows)
    for row in modified:
        for fields in GROUP_FIELDS.values():
            for field in fields:
                row[field] = 12345.
    b = ContextualPartialPredictor(walk).fit(modified,as_of="2025-03-01T00:00:00Z",dataset_version="synthetic")
    assert a.heads["0"] == b.heads["0"]
    assert a.predict_partial(_state())["probabilities"] == b.predict_partial(_state())["probabilities"]


def test_train_only_vocabulary_even_when_crosswalk_contains_other_team(tmp_path):
    model = _fit(tmp_path)
    result = model.predict_partial(_state("t4"))
    assert result["unknown_identity_fields"] == ["home_team_id"]
    assert result["identity_projection"]["unmapped_fields"] == []
    assert "provider:understat:team:74" not in model.heads["0"]["categorical_fit"]["home_team_id"]


def test_observed_future_information_and_training_future_labels_rejected(tmp_path):
    walk = _walk(tmp_path)
    rows = _rows()
    rows[0]["result_available_at"] = "2025-05-01T00:00:00Z"
    with pytest.raises(ValueError,match="unavailable"):
        ContextualPartialPredictor(walk).fit(rows,as_of="2025-03-01T00:00:00Z",dataset_version="synthetic")
    model = ContextualPartialPredictor(walk).fit(_rows(),as_of="2025-03-01T00:00:00Z",dataset_version="synthetic")
    s = _state()
    s["groups"]["elo"] = {"status":"observed","values":{"home_elo":1200.,"away_elo":1200.},"available_at":"2025-04-03T00:00:00Z"}
    with pytest.raises(ValueError,match="unavailable"):
        model.predict_partial(s)
    s = _state()
    s["prediction_cutoff"] = "2025-02-01T00:00:00Z"
    with pytest.raises(ValueError,match="precedes"):
        model.predict_partial(s)


def test_explicit_fallback_and_model_serialization(tmp_path):
    model = _fit(tmp_path,rows=_rows()[:9])
    assert model.predict_partial(_state())["fallback_reason"] == "insufficient_training_matches"
    path = tmp_path/"model.json"
    model.save(path)
    restored = ContextualPartialPredictor.load(path)
    assert model.predict_partial(_state()) == restored.predict_partial(_state())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["metadata"]["regularization"] = 999
    path.write_text(json.dumps(payload),encoding="utf-8")
    with pytest.raises(ValueError,match="hash"):
        ContextualPartialPredictor.load(path)


def test_current_predictor_probabilities_not_reused_as_free_feature(tmp_path):
    model = _fit(tmp_path)
    assert all(not any(name.startswith("p_") or name == "entropy" for name in h["numeric_names"]) for h in model.heads.values())
