"""Recheck actual artifacts and optionally consume an existing Elo export.

No network calls, fixture fabrication, warehouse writes, or Elo updates occur.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from football_research.baselines.api import Prediction, PredictionService, canonical_json, utc
from football_research.data_adapters import load_jsonl, load_spftll_archive, load_feature_snapshot


def verify(run_dir):
    root = Path(run_dir)
    dataset_path = root / "dataset.jsonl"
    dataset = load_jsonl(dataset_path)
    matches = {m.match_id: m for m in dataset.matches}
    predictions = [json.loads(line) for line in (root / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    for prediction in predictions:
        Prediction.model_validate({k: prediction[k] for k in Prediction.model_fields})
        cutoff = utc(prediction["prediction_cutoff"])
        manifest = prediction["input_manifest"]
        train_cutoff = utc(manifest["train_cutoff"])
        assert train_cutoff <= cutoff < matches[prediction["match_id"]].kickoff
        assert matches[prediction["match_id"]].fixture_available_at <= cutoff
        for mid in manifest["training_match_ids"]:
            assert mid != prediction["match_id"]
            assert matches[mid].result_available_at < train_cutoff
        refs = manifest["training_inputs"] + manifest["target_input"]
        if manifest.get("feature_input"):
            refs += manifest["feature_input"].get("quote_refs", [])
        for ref in refs:
            if "|at:" in ref:
                assert utc(ref.split("|at:", 1)[1]) <= cutoff
    artifacts = list((root / "models").glob("*.json"))
    for artifact in artifacts:
        PredictionService.load(artifact)
    return {"status": "VERIFIED", "dataset_version": dataset.dataset_version,
            "dataset_file_sha256": sha256(dataset_path.read_bytes()).hexdigest(),
            "predictions_file_sha256": sha256((root / "predictions.jsonl").read_bytes()).hexdigest(),
            "predictions": len(predictions), "matches": len({p["match_id"] for p in predictions}),
            "model_counts": dict(Counter(p["model"] for p in predictions)), "verified_model_artifacts": len(artifacts),
            "checks": ["probability_contract", "training_results_strictly_before_freeze", "target_not_in_training",
                       "known_prematch_fixture", "all_observation_refs_at_or_before_prediction", "model_artifact_hashes"]}


def verify_elo(feature_path, identity_report_path, archive_root, output_dir):
    feature = load_feature_snapshot(feature_path)
    identity_path = Path(identity_report_path)
    target = json.loads(identity_path.read_text(encoding="utf-8"))["target"]
    if target["match_id"] != feature["match_id"]:
        raise ValueError("Explicit platform identity report does not match feature canonical match ID")
    dataset = load_spftll_archive(archive_root, as_of=feature["prediction_cutoff"])
    match = next(m for m in dataset.matches if m.match_id == "spftll:match:" + target["source_id"])
    if match.kickoff != utc(target["kickoff"]):
        raise ValueError("Identity report kickoff differs from observed fixture")
    fixture = dict(asdict(match), match_id=feature["match_id"])
    feature["input_manifest"] = {"feature_manifest": list(feature["input_manifest"]),
        "identity_mapping": {"source_match_id": match.match_id, "canonical_match_id": feature["match_id"],
                             "evidence_report_sha256": sha256(identity_path.read_bytes()).hexdigest()}}
    service = PredictionService.fixed_elo_mapping(as_of=feature["prediction_cutoff"], dataset_version=dataset.dataset_version)
    prediction = service.predict(fixture, prediction_cutoff=feature["prediction_cutoff"], features=feature)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    service.save(output / "elo-fixed-model.json")
    (output / "elo-prediction.json").write_text(canonical_json(prediction.model_dump(mode="json")) + "\n", encoding="utf-8")
    assert PredictionService.load(output / "elo-fixed-model.json").predict(
        fixture, prediction_cutoff=feature["prediction_cutoff"], features=feature) == prediction
    return {"status": "FUNCTIONAL_VALIDATION", "mapping": "fixed_parameters_not_fitted",
            "rating_owner": "existing_FeatureStore_export", "feature_file": str(Path(feature_path).resolve()),
            "feature_cutoff": feature["prediction_cutoff"].isoformat(),
            "home_elo": feature["home_elo"], "away_elo": feature["away_elo"],
            "home_win_probability": prediction.home_win_probability, "draw_probability": prediction.draw_probability,
            "away_win_probability": prediction.away_win_probability, "json_reload_equivalent": True,
            "limitation": "Existing export consumed and manifest checked; warehouse Elo replay and forecast accuracy not evaluated."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--elo-feature")
    parser.add_argument("--identity-report", default="football_spider/platform_data/reports/real-acceptance.json")
    parser.add_argument("--archive-root", default="football_spider/platform_data/raw")
    args = parser.parse_args()
    report = verify(args.run_dir)
    if args.elo_feature:
        report["elo"] = verify_elo(args.elo_feature, args.identity_report, args.archive_root, Path(args.output).parent)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_json(report) + "\n", encoding="utf-8")
    print(canonical_json(report))


if __name__ == "__main__":
    main()
