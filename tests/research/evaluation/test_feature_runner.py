"""Synthetic mechanisms; these matches do not enter real round-two evidence."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json

from football_research.baselines.api import canonical_json, content_hash
from football_research.data_adapters.batch_features import FeatureIndex
from football_research.evaluation.runner import evaluate


def mechanism_data(n=90):
    matches, features = [], []
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for i in range(n):
        kickoff = start + timedelta(days=i)
        row = dict(match_id=f"test:{i}", competition_id="test", home_team_id=f"team:{i % 4}",
                   away_team_id=f"team:{(i + 1) % 4}", kickoff=kickoff, fixture_available_at=kickoff - timedelta(days=5),
                   result_available_at=kickoff + timedelta(hours=3), home_goals=i % 3, away_goals=1,
                   score_period="regulation", status="finished", information_mode="historical_reconstruction",
                   input_manifest=["synthetic-mechanism"])
        matches.append(row)
        for lead in (24, 1):
            cutoff = (kickoff - timedelta(hours=lead)).isoformat()
            manifest = dict(match_id=row["match_id"], prediction_cutoff=cutoff,
                            information_mode=row["information_mode"], feature_version="synthetic-v1", history_match_ids=[])
            features.append({**manifest, "features_available_at": cutoff, "input_manifest": manifest,
                "input_manifest_hash": content_hash(manifest), "features": {
                    "home_elo": 1490. + i % 7 + lead, "away_elo": 1510. - i % 5,
                    "home_rolling_xg_for": 1. + (i % 9) * .1, "away_rolling_xg_for": 1.2}})
    return matches, features


def run(matches, features, output, models=("frequency", "elo", "logistic", "tree")):
    return evaluate(matches, dataset_version="synthetic-only", information_mode="historical_reconstruction",
        output_dir=output, initial_validation_start="2024-02-01T00:00:00Z", validation_days=10, test_days=10,
        models=models, lead_hours=(24, 1), min_train_matches=10,
        feature_index=FeatureIndex.from_rows(features), feature_version="synthetic-v1")


def test_all_feature_models_run_at_each_lead_with_training_only_preprocessing(tmp_path):
    matches, features = mechanism_data()
    report = run(matches, features, tmp_path)
    assert report["test_matches"] == 49
    assert report["test_predictions"] == 49 * 2 * 4
    assert all(v["common_matches"] == 49 for v in report["common_match_metrics"].values())
    for entry in report["fold_runs"]:
        artifact = json.loads(Path(entry["model_artifact"]).read_text(encoding="utf-8"))
        ids = artifact["training_manifest"]["training_match_ids"]
        assert len(ids) == len(set(ids))
        if entry["model"] != "frequency":
            for feature in artifact["training_manifest"]["training_feature_inputs"]:
                match = matches[int(feature["match_id"].split(":")[-1])]
                assert match["kickoff"] - datetime.fromisoformat(feature["prediction_cutoff"]) == timedelta(hours=entry["lead_hours"])


def test_future_poison_does_not_change_predictions(tmp_path):
    matches, features = mechanism_data()
    run(matches, features, tmp_path / "clean", models=("logistic",))
    poisoned = [dict(row, features_available_at="2030-01-01T00:00:00Z", features={"home_elo": 9999, "away_elo": 0}) for row in features]
    run(matches, features + poisoned, tmp_path / "future", models=("logistic",))
    assert (tmp_path / "clean/predictions.jsonl").read_bytes() == (tmp_path / "future/predictions.jsonl").read_bytes()


def test_missing_features_report_coverage_without_disabling_score_baseline(tmp_path):
    matches, _ = mechanism_data()
    report = run(matches, [], tmp_path, models=("frequency", "elo"))
    assert report["model_coverage"]["elo/24h"]["test_predictions"] == 0
    assert report["model_coverage"]["frequency/24h"]["test_predictions"] == 49
    assert report["common_match_metrics"]["24h"]["unavailable_models"] == ["elo"]
