from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json

import pytest

from football_agent.research_bridge import ResearchServices
from football_agent.verify_round6_contextual import select_cases
from football_research.acquisition.contextual_identity import SourceIdentityValueEstimator
from football_research.acquisition.contextual_value import ContextualValueEstimator
from football_research.acquisition.partial import PartialPredictor
from football_research.acquisition.state import GROUP_FIELDS
from football_research.acquisition.value import _canonical


KINDS = ("contextual-value-v1", "source-identity-value-v1")
ACTION = {"action_id": "xg", "group_ids": ["xg"], "available": True,
          "cost": {"requests": 1., "money": .03, "latency_ms": 0., "tokens": 0.}, "cost_status": "simulated"}


def state(day=100, match_id="future", home="h0"):
    kickoff = datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=day)
    return {"match_id": match_id, "prediction_cutoff": (kickoff - timedelta(days=1)).isoformat(),
            "information_mode": "historical_reconstruction",
            "base": {"competition_id": "epl", "home_team_id": home, "away_team_id": "away", "kickoff": kickoff.isoformat()},
            "groups": {g: {"status": "not_acquired"} for g in GROUP_FIELDS}}


def samples():
    rows = []
    for i in range(1, 25):
        s = state(i, f"sample-{i}", f"h{i % 2}")
        rows.append({"state": s, "action": ACTION, "delta": .3 if i % 2 else .1,
                     "match_id": s["match_id"], "kickoff": s["base"]["kickoff"], "prediction_before": [.5, .3, .2],
                     "result_available_at": (datetime.fromisoformat(s["base"]["kickoff"]) + timedelta(hours=3)).isoformat(),
                     "oof_train_cutoff": "2024-12-01T00:00:00Z", "oof_train_ids": ["earlier"]})
    return rows


def crosswalk():
    payload = {"provider": "understat", "aliases": {
        "team": {"h0": "provider:understat:team:1", "h1": "provider:understat:team:2", "away": "provider:understat:team:3"},
        "competition": {"epl": "provider:understat:competition:EPL"}},
        "sources": [], "time_interpretation": "constructed_bridge_test_not_historical_verification"}
    payload["version"] = "provider-identity-crosswalk:" + sha256(_canonical(payload).encode()).hexdigest()[:20]
    return payload


class Predictor:
    model_version = "constructed-predictor"
    metadata = {"training_as_of": "2025-03-01T00:00:00Z", "dataset_version": "constructed-bridge-test"}

    def predict_partial(self, visible):
        return {"probabilities": [.5, .3, .2], "model_version": self.model_version}


def configuration(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(PartialPredictor, "load", lambda path: Predictor())
    model = (ContextualValueEstimator(min_support=2) if kind == KINDS[0]
             else SourceIdentityValueEstimator(crosswalk(), min_support=2))
    model.fit(samples(), as_of="2025-03-01T00:00:00Z", dataset_version="constructed-bridge-test")
    model.save(tmp_path / "value.json")
    (tmp_path / "predictor.json").write_text("{}", encoding="utf-8")
    config = {"partial_predictor": "predictor.json", "value_estimator": "value.json",
              "value_estimator_kind": kind, "lambda_cost": {"money": .5}}
    path = tmp_path / "backend.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return model, path, config


def visible():
    s = state()
    return {"match_id": s["match_id"], "prediction_cutoff": s["prediction_cutoff"],
            "information_mode": s["information_mode"], "base_facts": s["base"],
            "groups": {}, "mask": {g: False for g in GROUP_FIELDS}, "quality": {},
            "action_history": [], "budget": {}, "evidence_versions": {}, "state_version": "test"}


def bridge_action():
    return {"action_id": ACTION["action_id"], "group_ids": ACTION["group_ids"], "status": "AVAILABLE",
            "cost_vector": deepcopy(ACTION["cost"]), "cost": {"status": "simulated"}}


@pytest.mark.parametrize("kind", KINDS)
def test_kind_dispatch_loads_matching_schema_and_preserves_artifact_identity(tmp_path, monkeypatch, kind):
    model, path, config = configuration(tmp_path, monkeypatch, kind)
    service = ResearchServices.load(path)
    assert type(service.estimator) is type(model)
    assert service.estimator.version == model.version
    assert service.configuration()["artifact_hashes"]["estimator"] == sha256((tmp_path / "value.json").read_bytes()).hexdigest()
    assert service.configuration()["lambda_cost"]["money"] == .5


@pytest.mark.parametrize("kind", KINDS)
def test_mismatched_kind_does_not_silently_load_different_estimator(tmp_path, monkeypatch, kind):
    _, path, config = configuration(tmp_path, monkeypatch, kind)
    config["value_estimator_kind"] = KINDS[1 - KINDS.index(kind)]
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        ResearchServices.load(path)


def test_unknown_kind_is_rejected_without_fallback(tmp_path, monkeypatch):
    _, path, config = configuration(tmp_path, monkeypatch, KINDS[0])
    config["value_estimator_kind"] = "contextual-typo"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="Unrecognized value estimator kind"):
        ResearchServices.load(path)


@pytest.mark.parametrize("kind", KINDS)
def test_bridge_preserves_native_contextual_gain_cost_and_identity_audit(tmp_path, monkeypatch, kind):
    model, path, _ = configuration(tmp_path, monkeypatch, kind)
    service = ResearchServices.load(path)
    direct = model.estimate_value(state(), [ACTION], prediction=[.5, .3, .2])[0]
    result = service.estimate(visible(), [bridge_action()])[0]
    assert result["expected_loss_improvement"] == direct["gain"]
    assert result["net_gain"] == pytest.approx(direct["gain"] - .015)
    assert result["cost_in_loss_units"] == .015
    assert result["status"] == "OK"
    assert result["support"] == direct["support_matches"]
    assert result["lower_gain"] == direct["lower_gain"]
    assert result["estimator_version"] == model.version
    if kind == KINDS[1]:
        assert result["research_estimate"]["identity_projection"]["unmapped_fields"] == []
    assert "additional_prediction_compute_cost" in result


@pytest.mark.parametrize("kind", KINDS)
def test_unknown_action_support_remains_unavailable(tmp_path, monkeypatch, kind):
    _, path, _ = configuration(tmp_path, monkeypatch, kind)
    action = bridge_action()
    action["action_id"], action["group_ids"] = "untrained", ["recent_form"]
    result = ResearchServices.load(path).estimate(visible(), [action])[0]
    assert result["status"] == "UNAVAILABLE"
    assert result["net_gain"] is None
    assert result["support"] == 0


def fixture_records():
    rows = []
    for strategy in ("base-only", "contextual-visible-context", "contextual-source-identity"):
        for mid, day in (("later-higher-gain", "2025-05-02"), ("first", "2025-05-01")):
            rows.append({"fold": 0, "strategy": strategy, "match_id": mid, "budget_money": .05,
                         "kickoff": day + "T18:00:00Z", "prediction_cutoff": day + "T12:00:00Z",
                         "gain": 999 if mid != "first" else -999})
    report = {"folds": [{"fold": 0, "models": {"visible-context": {}, "source-identity": {}},
                         "lambda_selection": {"contextual-visible-context/0.05": {"selected": .25},
                                              "contextual-source-identity/0.05": {"selected": 1.0}}}]}
    return report, rows


def test_bridge_fixture_selection_uses_schedule_and_strategy_specific_lambda():
    report, rows = fixture_records()
    selected = select_cases(report, rows)
    assert [c["first"]["match_id"] for c in selected] == ["first", "first"]
    assert [c["lambda_money"] for c in selected] == [.25, 1.0]
    assert [c["kind"] for c in selected] == list(KINDS)


def test_bridge_missing_or_duplicated_reference_result_is_rejected():
    report, rows = fixture_records()
    broken = [r for r in rows if not (r["strategy"] == "contextual-source-identity" and r["match_id"] == "first")]
    with pytest.raises(ValueError, match="exactly one"):
        select_cases(report, broken)
    with pytest.raises(ValueError, match="exactly one"):
        select_cases(report, rows + [rows[-1]])
