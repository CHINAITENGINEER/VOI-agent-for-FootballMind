"""The submission entry must execute real acquisition without exposing labels."""
import json

import pytest

from football_agent import historical_analysis as history


@pytest.fixture(scope="module")
def inventory():
    return history._load(history.ROOT)


@pytest.fixture
def frozen_inventory(monkeypatch, inventory):
    monkeypatch.setattr(history, "_load", lambda root: inventory)
    return inventory


def _walk(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def test_real_catalog_contains_registered_out_of_sample_matches_only(frozen_inventory):
    catalog = history.load_catalog()
    assert len(catalog) == 290
    assert len({r["match_id"] for r in catalog}) == 290
    assert {r["fold"] for r in catalog} == set(range(7))
    assert len(frozen_inventory["skipped"]) == 90
    assert not {"home_goals", "away_goals", "outcome", "label"}.intersection(_walk(catalog))


def test_real_match_executes_frozen_context_choices_and_persists_trace(tmp_path, frozen_inventory):
    result = history.analyze_match("Bournemouth vs Chelsea", output=tmp_path)
    assert result["match"]["match_id"] == "54db3e02-8a94-5163-b0f8-0d46274f74d1"
    assert result["actions"] == ["xg_view", "results_view"]
    assert result["configuration"]["lambda_money"] == 0.5
    assert result["cost"]["requests"] == 2
    assert result["terminal_selection"]["stop_certificate"] is None
    assert result["network_requests"] == 0
    assert result["model_refitted"] is False
    assert result["execution_completed"] is True
    assert result["awaiting_background_execution"] is False
    assert result["data_scope"]["live_news_value_model"] is False
    assert not {"home_goals", "away_goals", "outcome", "outcome_index", "label"}.intersection(_walk(result))
    saved = json.loads((tmp_path / "trajectory.json").read_text(encoding="utf-8"))
    assert saved == result["trajectory"]
    assert saved["cycles"][0]["state_before"]["groups"]["xg"]["status"] == "not_acquired"
    assert saved["cycles"][0]["state_after"]["groups"]["xg"]["status"] == "observed"
    for cycle in saved["cycles"]:
        selected = cycle["decision"]["selection"].get("candidate_id")
        if selected:
            candidates = [c for c in cycle["decision"]["candidates"] if c["feasible"] and c["value"]["status"] == "COMPUTED"]
            chosen = next(c for c in candidates if c["action_id"] == selected)
            assert chosen["value"]["net_voi"] > 0
            assert chosen["value"]["net_voi"] == max(c["value"]["net_voi"] for c in candidates)
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "预期净价值" in text
    assert "真实历史" in text
    assert "最终预测" in text


def test_query_budget_zero_retains_forecast_without_faking_sufficiency(tmp_path, frozen_inventory):
    result = history.analyze_match("28918", output=tmp_path, max_queries=0)
    assert result["actions"] == []
    assert result["initial_probabilities"] == result["final_probabilities"]
    assert result["cost"]["requests"] == 0
    assert result["terminal_selection"]["reason"] == "physical_budget_exhausted"
    assert result["terminal_selection"]["stop_certificate"] is None
    assert result["configuration"]["explicit_sensitivity_override"] is True


def test_hidden_result_does_not_change_actor_and_missing_stays_explicit(tmp_path, frozen_inventory, monkeypatch):
    mid = "54db3e02-8a94-5163-b0f8-0d46274f74d1"
    before = history.analyze_match(mid, output=tmp_path / "original")
    altered = dict(frozen_inventory)
    altered["rows"] = dict(altered["rows"])
    altered["rows"][mid] = dict(altered["rows"][mid], home_goals=18, away_goals=17)
    monkeypatch.setattr(history, "_load", lambda root: altered)
    after = history.analyze_match(mid, output=tmp_path / "changed-label")
    assert before["trajectory"] == after["trajectory"]
    from football_research.acquisition.state import GROUP_FIELDS
    for field in GROUP_FIELDS["xg"]:
        altered["rows"][mid][field] = None
    missing = history.analyze_match(mid, output=tmp_path / "missing")
    assert missing["actions"][0] == "xg_view"
    assert missing["cycles"][0]["state_after"]["groups"]["xg"] == {"status": "source_missing", "values": {}}
    assert missing["terminal_selection"]["stop_certificate"] is None


def test_eligibility_rejects_training_overlap_and_wrong_clock(inventory):
    row = inventory["rows"][inventory["catalog"][0]["match_id"]]
    model = inventory["models"][0]
    fold = {"training_match_ids": [row["match_id"]]}
    assert history._eligibility(row, model, fold) == "overlaps_model_training_or_selection"
    changed = dict(row, prediction_cutoff="2024-01-01T00:00:00+00:00")
    assert history._eligibility(changed, model, {}) == "incompatible_model_or_match_chronology"


def test_resolver_never_guesses_unknown_or_ambiguous_fixture():
    catalog = [{"match_id": "first", "source_id": "1", "home_team": "Home", "away_team": "Away", "kickoff": "2025-01-01T00:00:00+00:00"},
               {"match_id": "second", "source_id": "2", "home_team": "Home", "away_team": "Away", "kickoff": "2025-02-01T00:00:00+00:00"}]
    with pytest.raises(ValueError, match="歧义"):
        history._resolve("Home vs Away", catalog)
    assert history._resolve({"home_team": "Home", "away_team": "Away", "date": "2025-02-01"}, catalog)["match_id"] == "second"
    with pytest.raises(ValueError, match="不在"):
        history._resolve("Unknown vs Away", catalog)


@pytest.mark.parametrize("kwargs", [{"max_queries": -1}, {"max_queries": True}, {"max_queries": 3},
                                   {"lambda_money": float("nan")}, {"lambda_money": -1}, {"lambda_money": True}])
def test_invalid_budget_fails_before_loading_or_writing(tmp_path, monkeypatch, kwargs):
    monkeypatch.setattr(history, "_load", lambda root: pytest.fail("invalid input loaded frozen data"))
    with pytest.raises(ValueError):
        history.analyze_match("whatever", output=tmp_path, **kwargs)
    assert not list(tmp_path.iterdir())


def test_existing_analysis_is_preserved(tmp_path, monkeypatch):
    (tmp_path / "analysis.json").write_text("existing user record", encoding="utf-8")
    monkeypatch.setattr(history, "_load", lambda root: pytest.fail("existing output should be checked first"))
    with pytest.raises(FileExistsError):
        history.analyze_match("whatever", output=tmp_path)
    assert (tmp_path / "analysis.json").read_text(encoding="utf-8") == "existing user record"


def test_batch_session_loads_once_and_separates_match_state(tmp_path, monkeypatch, inventory):
    calls = []
    def counted(root):
        calls.append(root)
        return inventory
    monkeypatch.setattr(history, "_load", counted)
    session = history.HistoricalAnalysisSession()
    catalog = session.load_catalog()
    catalog[0]["home_team"] = "caller modification"
    assert session.load_catalog()[0]["home_team"] != "caller modification"
    first = session.analyze_match("Bournemouth vs Chelsea", output=tmp_path / "first")
    second = session.analyze_match("292ec40c-b1ad-5ed0-b75b-258f41f1b80f", output=tmp_path / "second")
    assert len(calls) == 1
    assert first["actions"] == ["xg_view", "results_view"]
    assert second["actions"] == ["results_view"]
    assert second["cost"]["requests"] == 1
    assert all(group["status"] == "not_acquired" for group in second["cycles"][0]["state_before"]["groups"].values())
    first["match"]["home_team"] = "modified report"
    first["configuration"]["original_limits"]["requests"] = 99
    assert session.load_catalog()[next(i for i, r in enumerate(session.load_catalog()) if r["match_id"] == first["match"]["match_id"])]["home_team"] == "Bournemouth"
    assert inventory["limits"]["requests"] == 2
