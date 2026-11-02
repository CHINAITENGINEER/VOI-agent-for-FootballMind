"""Regression cases found by independent feature-consumer integration review."""
from datetime import datetime, timedelta, timezone
import json

import pytest

from football_research.baselines.api import PredictionService, canonical_json, content_hash


def supervised_rows():
    rows = []
    for number in range(9):
        kickoff = datetime(2025, 1, 2, 12, tzinfo=timezone.utc) + timedelta(days=number)
        cutoff = kickoff - timedelta(hours=24)
        rows.append({"match_id": f"constructed:{number}", "competition_id": "constructed:league",
            "home_team_id": "constructed:home", "away_team_id": "constructed:away", "kickoff": kickoff,
            "fixture_available_at": cutoff - timedelta(days=1), "result_available_at": kickoff + timedelta(hours=3),
            "home_goals": number % 3, "away_goals": 1, "status": "finished", "score_period": "regulation",
            "information_mode": "historical_reconstruction", "prediction_cutoff": cutoff,
            "features_available_at": cutoff, "feature_version": "constructed-v1",
            "feature_input_manifest": {"test_only_source": f"history:{number}"},
            "input_manifest": ["constructed-only"], "home_elo": 1500. + number, "away_elo": 1500.})
    return rows


def train(rows, parameters=None):
    return PredictionService.train("logistic", rows, as_of="2025-01-12T00:00:00Z",
        dataset_version="constructed-only", information_mode="historical_reconstruction", parameters=parameters)


def test_partly_missing_feature_version_cannot_be_declared_as_one_known_version():
    rows = supervised_rows()
    rows[0].pop("feature_version")
    with pytest.raises(ValueError, match="[Vv]ersion"):
        train(rows)


@pytest.mark.parametrize("name", ["forecast", "home_xg", "final_season_points"])
def test_public_training_api_rejects_target_or_final_feature_names(name):
    rows = supervised_rows()
    for row in rows:
        row[name] = float(row["home_goals"])
    with pytest.raises(ValueError):
        train(rows, {"feature_names": [name]})


def test_cli_configured_historical_feature_is_passed_to_reader_allowlist(tmp_path, capsys):
    from football_research.baselines.__main__ import main
    from football_research.data_adapters import save_jsonl
    from football_research.data_adapters.types import ResearchDataset, ResearchMatch

    records, snapshots = [], []
    for row in supervised_rows():
        records.append(ResearchMatch(**{name: row[name] for name in (
            "match_id", "competition_id", "home_team_id", "away_team_id", "kickoff", "fixture_available_at",
            "result_available_at", "home_goals", "away_goals", "status", "score_period", "information_mode", "input_manifest")},
            season="constructed"))
        manifest = {name: row[name] for name in ("match_id", "prediction_cutoff", "information_mode", "feature_version")}
        snapshots.append({**manifest, "features_available_at": row["prediction_cutoff"],
            "input_manifest": manifest, "input_manifest_hash": content_hash(manifest),
            "features": {"home_history_xg_avg": row["home_elo"] / 1000}})
    dataset = tmp_path / "dataset.jsonl"
    save_jsonl(ResearchDataset(tuple(records), (), "constructed-only", ("constructed-only",)), dataset)
    features = tmp_path / "features.jsonl"
    features.write_text("".join(canonical_json(row) + "\n" for row in snapshots), encoding="utf-8")
    parameters = tmp_path / "parameters.json"
    parameters.write_text(json.dumps({"feature_names": ["home_history_xg_avg"]}), encoding="utf-8")
    artifact = tmp_path / "model.json"
    main(["train", "--dataset", str(dataset), "--model", "logistic", "--as-of", "2025-01-12T00:00:00Z",
        "--feature-file", str(features), "--feature-version", "constructed-v1", "--lead-hours", "24",
        "--model-parameters", str(parameters), "--output", str(artifact)])
    assert PredictionService.load(artifact).model.feature_names == ["home_history_xg_avg"]
    assert json.loads(capsys.readouterr().out)["training_matches"] == 9
