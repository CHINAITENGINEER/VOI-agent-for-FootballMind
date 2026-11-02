"""One completed platform export -> fixed-design historical reconstruction.

All real results retain today's observation and explicit historical visibility
assumptions. Prospective ledgers and the first-round artifacts are never touched.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from football_research.baselines.api import canonical_json
from football_research.data_adapters import save_jsonl
from football_research.data_adapters.batch_features import load_feature_index
from football_research.data_adapters.platform_history import load_completed_history_export, verify_completed_export
from football_research.data_adapters.settlement import export_results
from .ledger import ForecastLedger
from .runner import evaluate


def run_reconstruction(export_path, output_dir, design_path, feature_version):
    config_bytes = Path(design_path).read_bytes()
    config = json.loads(config_bytes)
    verified = verify_completed_export(export_path)
    dataset = load_completed_history_export(export_path)
    if {m.information_mode for m in dataset.matches} != {"historical_reconstruction"}:
        raise ValueError("This experiment requires explicit historical_reconstruction input")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "experiment-design.json").write_bytes(config_bytes)
    (output / "platform-manifest.json").write_text(canonical_json(verified.manifest) + "\n", encoding="utf-8")
    save_jsonl(dataset, output / "dataset.jsonl")
    if "features" not in verified.payloads:
        raise ValueError("Completed export does not contain batch features")
    (output / "features.jsonl").write_bytes(verified.payloads["features"])
    allowed = {field for params in config.get("model_parameters", {}).values() for field in params.get("feature_names", [])}
    index = load_feature_index(output / "features.jsonl", allowed_features=allowed)
    (output / "feature-audit.json").write_text(canonical_json(index.report()) + "\n", encoding="utf-8")
    report = evaluate([asdict(m) for m in dataset.matches], dataset_version=dataset.dataset_version,
        information_mode="historical_reconstruction", output_dir=output / "backtest",
        initial_validation_start=config["initial_validation_start"], validation_days=config["validation_days"],
        test_days=config["test_days"], lead_hours=config["lead_hours"], min_train_matches=config["min_train_matches"],
        models=config["models"], model_parameters=config.get("model_parameters"), feature_index=index,
        feature_version=feature_version, dataset_audit=dataset.audit, evidence_grade="conditional_historical_reconstruction")
    ledger = ForecastLedger(output / "ledger-reconstruction")
    forecasts = [json.loads(line) for line in (output / "backtest/predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    test_forecasts = [r for r in forecasts if r["partition"] == "test"]
    (output / "test-predictions.jsonl").write_text("".join(canonical_json(r) + "\n" for r in test_forecasts), encoding="utf-8")
    for prediction in test_forecasts:
        ledger.import_forecast(prediction, kickoff=prediction["kickoff"], source_reference="frozen_temporal_test_predictions")
    frozen_hash = ledger.score()["forecast_snapshot_sha256"]
    results, result_audit = export_results(dataset, match_ids={r["match_id"] for r in test_forecasts})
    (output / "results.jsonl").write_text("".join(canonical_json(r) + "\n" for r in results), encoding="utf-8")
    ledger.settle_results(results)
    score = ledger.score()
    repeated = [result["status"] for result in ledger.settle_results(results)]
    score_repeat = ledger.score()
    assert all(status == "duplicate" for status in repeated)
    assert frozen_hash == score["forecast_snapshot_sha256"] == score_repeat["forecast_snapshot_sha256"]
    assert score["groups"] == score_repeat["groups"] and score["forecasts"] == score_repeat["forecasts"]
    (output / "settlement-score.json").write_text(canonical_json(score) + "\n", encoding="utf-8")
    summary = {"status": report["status"], "information_mode": "historical_reconstruction", "registration_kind": "replay",
        "dataset_version": dataset.dataset_version, "platform_manifest_sha256": verified.manifest_sha256,
        "design_sha256": sha256(config_bytes).hexdigest(), "input_matches": len(dataset.matches),
        "feature_version": feature_version, "feature_index": index.report(),
        "test_matches": report["test_matches"], "test_predictions": report["test_predictions"],
        "common_match_metrics": report["common_match_metrics"], "model_coverage": report["model_coverage"],
        "result_export": result_audit, "settlement_status": score["status_counts"],
        "result_repeat_status": {"duplicate": len(repeated)}, "repeat_metrics_equal": True,
        "frozen_forecast_hash_unchanged": True, "forecast_snapshot_sha256": frozen_hash,
        "limits": dataset.audit, "output_dir": str(output.resolve())}
    (output / "summary.json").write_text(canonical_json(summary) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform-export", required=True)
    parser.add_argument("--design", required=True)
    parser.add_argument("--feature-version", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = run_reconstruction(args.platform_export, args.output, args.design, args.feature_version)
    print(canonical_json({k: summary[k] for k in ("status", "input_matches", "test_matches", "test_predictions", "settlement_status", "output_dir")}))


if __name__ == "__main__":
    main()
