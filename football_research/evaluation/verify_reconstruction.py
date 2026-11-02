"""Validate frozen real-run provenance and reload a prediction per fitted model."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from football_research.baselines.api import PredictionService, canonical_json, utc
from football_research.data_adapters import load_jsonl
from football_research.data_adapters.batch_features import load_feature_index
from .runner import FEATURE_MODELS, model_feature_names


def verify(directory):
    root = Path(directory)
    report = json.loads((root / "backtest/report.json").read_bytes())
    dataset = load_jsonl(root / "dataset.jsonl")
    matches = {m.match_id: asdict(m) for m in dataset.matches}
    index = load_feature_index(root / "features.jsonl")
    predictions = [json.loads(line) for line in (root / "backtest/predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    versions, checks = [], []
    probabilities = ("home_win_probability", "draw_probability", "away_win_probability")
    for row in predictions:
        assert all(0 <= row[field] <= 1 for field in probabilities)
        assert abs(sum(row[field] for field in probabilities) - 1) < 1e-12
        assert utc(row["prediction_cutoff"]) < utc(row["kickoff"])
    for run in report["fold_runs"]:
        path = Path(run["model_artifact"])
        service = PredictionService.load(path)
        params = report["configuration"]["model_parameters"].get(run["model"], {})
        training_ids = service.training_manifest["training_match_ids"]
        for mid in training_ids:
            assert utc(matches[mid]["result_available_at"]) <= service.trained_at
        candidates = [r for r in predictions if r["fold"] == run["fold"] and r["model"] == run["model"]
                      and r["lead_hours"] == run["lead_hours"] and r["partition"] == "test"]
        for prediction in candidates:
            assert prediction["match_id"] not in training_ids
            assert service.trained_at < utc(prediction["prediction_cutoff"])
        if candidates:
            row = candidates[0]
            fixture = matches[row["match_id"]]
            features = index.join(fixture, prediction_cutoff=row["prediction_cutoff"],
                information_mode=row["information_mode"], feature_version=report["feature_version"],
                required_features=model_feature_names(run["model"], params),
                allow_missing_values=run["model"] in {"logistic", "tree"}) if run["model"] in FEATURE_MODELS else None
            actual = service.predict(fixture, prediction_cutoff=row["prediction_cutoff"], features=features).model_dump(mode="json")
            assert all(actual[field] == row[field] for field in probabilities)
            checks.append({"artifact": str(path), "match_id": row["match_id"], "probabilities_exactly_equal": True})
        versions.append({"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest(), **service.describe()})
    return {"status": "VERIFIED", "input_matches": len(matches), "prediction_rows_checked": len(predictions),
        "test_prediction_rows": report["test_predictions"], "loaded_models": len(versions),
        "model_reload_predictions": len(checks), "checks": checks, "model_metadata": versions,
        "feature_audit": index.report(), "training_cutoff_and_target_exclusion": True,
        "validation_scope": "all probabilities and fitted-label cutoffs; exact model reload probability for first test fixture per model/lead/fold; feature temporal lineage checked on load"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = verify(args.run_dir)
    Path(args.output).write_text(canonical_json(result) + "\n", encoding="utf-8")
    print(canonical_json({k: result[k] for k in ("status", "input_matches", "prediction_rows_checked", "loaded_models", "model_reload_predictions")}))


if __name__ == "__main__":
    main()
