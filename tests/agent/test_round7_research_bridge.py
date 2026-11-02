from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json

import pytest

from football_agent.research_bridge import ResearchServices
from football_agent.verify_round7_contextual import artifact_hashes, select_cases
from football_research.acquisition.contextual_partial import ContextualPartialPredictor
from football_research.acquisition.contextual_identity import SourceIdentityValueEstimator
from football_research.acquisition.partial import PartialPredictor
from football_research.acquisition.state import GROUP_FIELDS
from football_research.acquisition.value import _canonical


def training_rows():
    rows = []
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    for i in range(36):
        kickoff = start + timedelta(days=i)
        row = {"match_id": f"bridge-{i}", "competition_id": "league", "home_team_id": f"home-{i % 3}",
               "away_team_id": "away", "kickoff": kickoff.isoformat(),
               "prediction_cutoff": (kickoff - timedelta(hours=24)).isoformat(),
               "result_available_at": (kickoff + timedelta(hours=3)).isoformat(),
               "information_mode": "historical_reconstruction", "home_goals": [2, 1, 0][i % 3],
               "away_goals": [0, 1, 2][i % 3]}
        for fields in GROUP_FIELDS.values():
            row.update({field: float(i % 5) for field in fields})
        rows.append(row)
    return rows


def public(home="home-0", *, observed=False):
    return {"match_id": "future", "prediction_cutoff": "2025-04-01T12:00:00Z",
            "information_mode": "historical_reconstruction",
            "base_facts": {"competition_id": "league", "home_team_id": home, "away_team_id": "away",
                           "kickoff": "2025-04-02T12:00:00Z"},
            "groups": ({"elo": {"status": "observed", "values": {"home_elo": 2.0, "away_elo": 3.0},
                                "available_at": "2025-03-01T12:00:00Z", "evidence_refs": []}} if observed else {}),
            "mask": {name: observed and name == "elo" for name in GROUP_FIELDS}, "quality": {},
            "action_history": [], "budget": {}, "evidence_versions": {}, "state_version": "constructed-bridge-test"}


def research_state(visible):
    return {"match_id": visible["match_id"], "prediction_cutoff": visible["prediction_cutoff"],
            "information_mode": visible["information_mode"], "base": visible["base_facts"],
            "groups": {name: deepcopy(visible["groups"].get(name, {"status": "not_acquired"})) for name in GROUP_FIELDS}}


class GainEstimator:
    """Test only that the adapter forwards the actual fitted prediction."""
    min_support = 2
    version = "constructed-value-forwarding"
    manifest = {"trained_as_of": "2025-03-01T00:00:00Z"}

    def estimate_value(self, visible_state, candidates, prediction=None):
        gain = prediction["probabilities"][0] * .1
        return [{"action_id": c["action_id"], "cost": c["cost"], "cost_status": "simulated", "gain": gain,
                 "standard_error": .01, "lower_gain": gain - .01, "support_matches": 24,
                 "support_status": "supported", "reason": "constructed_prediction_forwarding", "version": self.version}
                for c in candidates]


@pytest.fixture(scope="module")
def predictors():
    rows = training_rows()
    crosswalk = {"provider": "understat", "aliases": {
        "team": {**{f"home-{i}": f"provider:understat:team:{i + 1}" for i in range(3)},
                 "away": "provider:understat:team:4"},
        "competition": {"league": "provider:understat:competition:EPL"}},
        "sources": [], "time_interpretation": "constructed_bridge_test_not_historical_verification"}
    crosswalk["version"] = "provider-identity-crosswalk:" + sha256(_canonical(crosswalk).encode()).hexdigest()[:20]
    return {"partial-subset-v1": PartialPredictor().fit(rows, "2025-03-01T00:00:00Z", "constructed-bridge-test"),
            "contextual-partial-v1": ContextualPartialPredictor(crosswalk).fit(rows, "2025-03-01T00:00:00Z", "constructed-bridge-test")}


def config_path(tmp_path, monkeypatch, predictor, kind):
    monkeypatch.setattr(SourceIdentityValueEstimator, "load", lambda path: GainEstimator())
    predictor.save(tmp_path / "predictor.json")
    (tmp_path / "value.json").write_text("{}", encoding="utf-8")
    config = {"partial_predictor": "predictor.json", "value_estimator": "value.json",
              "value_estimator_kind": "source-identity-value-v1", "lambda_cost": {"money": .5}}
    if kind is not None:
        config["partial_predictor_kind"] = kind
    path = tmp_path / "backend.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


@pytest.mark.parametrize("kind", [None, "partial-subset-v1", "contextual-partial-v1"])
def test_explicit_predictor_dispatch_and_omitted_kind_backward_compatibility(tmp_path, monkeypatch, predictors, kind):
    predictor = predictors[kind or "partial-subset-v1"]
    path = config_path(tmp_path, monkeypatch, predictor, kind)
    service = ResearchServices.load(path)
    assert type(service.predictor) is type(predictor)
    assert service.configuration()["predictor_version"] == predictor.model_version
    assert service.configuration()["artifact_hashes"]["predictor"] == sha256((tmp_path / "predictor.json").read_bytes()).hexdigest()


@pytest.mark.parametrize("actual,requested", [("partial-subset-v1", "contextual-partial-v1"),
                                              ("contextual-partial-v1", "partial-subset-v1"),
                                              ("contextual-partial-v1", None)])
def test_predictor_artifact_family_is_not_silently_guessed(tmp_path, monkeypatch, predictors, actual, requested):
    path = config_path(tmp_path, monkeypatch, predictors[actual], requested)
    with pytest.raises(ValueError):
        ResearchServices.load(path)


def test_unknown_predictor_kind_rejected(tmp_path, monkeypatch, predictors):
    path = config_path(tmp_path, monkeypatch, predictors["partial-subset-v1"], "contextual-typo")
    with pytest.raises(ValueError, match="Unrecognized partial predictor kind"):
        ResearchServices.load(path)


@pytest.mark.parametrize("observed", [False, True])
def test_actual_contextual_prediction_matches_direct_model_through_bridge(tmp_path, monkeypatch, predictors, observed):
    predictor = predictors["contextual-partial-v1"]
    path = config_path(tmp_path, monkeypatch, predictor, "contextual-partial-v1")
    visible = public(observed=observed)
    direct = predictor.predict_partial(research_state(visible))
    result = ResearchServices.load(path).predict(visible)
    assert [result["probabilities"][k] for k in ("home", "draw", "away")] == direct["probabilities"]
    assert result["mask_key"] == direct["mask_key"]
    assert result["model_version"] == direct["model_version"]
    assert result["free_context"] == direct["free_context"]


def test_free_identity_context_changes_empty_mask_and_estimator_receives_current_probabilities(tmp_path, monkeypatch, predictors):
    path = config_path(tmp_path, monkeypatch, predictors["contextual-partial-v1"], "contextual-partial-v1")
    service = ResearchServices.load(path)
    action = {"action_id": "xg", "group_ids": ["xg"], "status": "AVAILABLE", "cost": {"status": "simulated"},
              "cost_vector": {"money": .03, "requests": 1., "latency_ms": 0., "tokens": 0.}}
    values = []
    for team in ("home-0", "home-2"):
        visible = public(home=team)
        prediction = service.predict(visible)
        estimate = service.estimate(visible, [action])[0]
        assert prediction["mask_key"] == 0
        assert estimate["expected_loss_improvement"] == prediction["probabilities"]["home"] * .1
        assert estimate["net_gain"] == pytest.approx(estimate["expected_loss_improvement"] - .015)
        values.append(estimate["expected_loss_improvement"])
    assert abs(values[0] - values[1]) > .001


def selection_fixture():
    modes = ("prediction-only", "source-identity")
    fold = {"fold": 0, "models": {mode: {} for mode in modes},
            "lambda_selection": {"contextual-prediction-only/0.05": {"selected": .25},
                                 "contextual-source-identity/0.05": {"selected": 1.0}}}
    rows = [{"fold": 0, "strategy": strategy, "match_id": mid, "budget_money": .05,
             "kickoff": "2025-05-01T18:00:00Z", "prediction_cutoff": "2025-04-30T18:00:00Z",
             "log_loss": loss} for strategy in ("base-only", "contextual-prediction-only", "contextual-source-identity")
            for mid, loss in (("z-best-result", 0.0), ("a-first", 99.0))]
    return {"folds": [fold]}, rows


def test_round7_selection_tie_breaks_by_identity_not_result_and_uses_actual_lambdas():
    report, rows = selection_fixture()
    selected = select_cases(report, rows)
    assert [r["first"]["match_id"] for r in selected] == ["a-first", "a-first"]
    assert [r["lambda_money"] for r in selected] == [.25, 1.0]
    assert [r["kind"] for r in selected] == ["contextual-value-v1", "source-identity-value-v1"]


def test_round7_missing_or_inconsistent_reference_rows_fail():
    report, rows = selection_fixture()
    with pytest.raises(ValueError, match="exactly one"):
        select_cases(report, rows[:-1])
    broken = deepcopy(rows)
    broken[-1]["prediction_cutoff"] = "2025-04-29T18:00:00Z"
    with pytest.raises(ValueError, match="timeline differs"):
        select_cases(report, broken)


def test_model_artifact_hash_lookup_normalizes_platform_separators_and_rejects_alias_duplicates():
    entries = [{"path": "models\\fold-0-predictor.json", "sha256": "digest"}]
    assert artifact_hashes(entries) == {"models/fold-0-predictor.json": "digest"}
    with pytest.raises(ValueError, match="Duplicate normalized"):
        artifact_hashes(entries + [{"path": "models/fold-0-predictor.json", "sha256": "changed"}])
