"""Reproduce a real existing FeatureStore -> PredictionService -> Agent run."""
from hashlib import sha256
import json
from pathlib import Path

from .audit import atomic_write
from .engine import Workflow
from .frozen_features import FrozenFeatureTools
from .model_catalog import ModelCatalog
from .types import Budget


def main():
    from football_research.baselines.api import PredictionService
    folder = Path("var/parallel/agent/round2/frozen-feature").resolve()
    folder.mkdir(parents=True, exist_ok=True)
    report_path = Path("football_spider/platform_data/reports/real-acceptance.json").resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    parquet = Path("football_spider/platform_data/features") / (report["feature"]["snapshot_id"] + ".parquet")
    binding = {"schema": "agent-frozen-platform-binding-v1",
               "platform_report": {"path": str(report_path), "sha256": sha256(report_path.read_bytes()).hexdigest()},
               "feature_snapshot": {"path": str(parquet.resolve()), "sha256": sha256(parquet.read_bytes()).hexdigest()}}
    binding_path = folder / "binding.json"
    atomic_write(binding_path, binding)
    cutoff = report["feature"]["manifest"]["prediction_cutoff"]
    service = PredictionService.fixed_elo_mapping(as_of=cutoff, dataset_version="platform-existing-feature:" + report["feature"]["manifest_hash"])
    model_path = service.save(folder / "elo-fixed.json")
    catalog_path = folder / "catalog.json"
    atomic_write(catalog_path, {"models": [{"model_id": "elo-platform-frozen", "model_name": "elo",
        "model_artifact": str(model_path), "artifact_sha256": service.to_artifact()["artifact_sha256"],
        "information_mode": "operational", "competition_id": None}]})
    tools = FrozenFeatureTools(export_path=binding_path, model_id="elo-platform-frozen", model_catalog=ModelCatalog(catalog_path))
    workflow = Workflow(tools, "var/parallel/agent/round2/runs")
    run = workflow.start(report["target"]["match_id"], cutoff, Budget(max_queries=0))
    for _ in range(4):
        run = workflow.step(run["run_id"])
    if run["state"] != "PREDICTED":
        raise RuntimeError(run["reason"])
    direct = service.predict(run["match"], prediction_cutoff=cutoff, features=run["features"]).model_dump(mode="json")
    assert run["prediction"] == direct
    run = workflow.step(run["run_id"], action="stop")
    atomic_write(folder / "report.json", run)
    result = {"kind": "real_existing_postgresql_featurestore_export", "run_id": run["run_id"],
              "match_id": run["match_id"], "prediction_cutoff": cutoff, "direct_prediction_equal": True,
              "model_version": direct["model_version"], "model_hash": service.to_artifact()["artifact_sha256"],
              "model_kind": "fixed_unfitted_elo_mapping;real_upstream_elo_values",
              "feature_version": run["features"]["feature_version"], "feature_sha256": binding["feature_snapshot"]["sha256"],
              "feature_manifest_hash": run["features"]["input_manifest_hash"],
              "source_to_canonical_mapping": {"source": tools.source_id, "canonical": tools.target_id,
                                               "basis": "platform_actual_acceptance_target;single_source_only"},
              "audit": workflow.store.verify(run["run_id"]), "network_requests": 0,
              "binding_path": str(binding_path), "catalog_path": str(catalog_path)}
    atomic_write(folder / "validation.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
