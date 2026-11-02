"""Existing real frozen platform snapshot; read-only, no network or database."""
import json
from pathlib import Path

import pytest

from demo.server import DemoApplication
from football_agent.frozen_features import FrozenFeatureTools
from football_agent.model_catalog import ModelCatalog


ROOT = Path("var/parallel/agent/round2/frozen-feature")


def tools():
    return FrozenFeatureTools(export_path=ROOT / "binding.json", model_id="elo-platform-frozen",
                             model_catalog=ModelCatalog(ROOT / "catalog.json"))


def test_real_feature_cutoff_id_and_mode_gates():
    adapter = tools()
    cutoff = adapter.snapshot["prediction_cutoff"].isoformat()
    assert adapter.execute("query_matches", {"match_id": adapter.target_id, "prediction_cutoff": cutoff}).status == "OK"
    assert adapter.execute("query_matches", {"match_id": "wrong", "prediction_cutoff": cutoff}).status == "NO_DATA"
    assert adapter.execute("query_matches", {"match_id": adapter.target_id, "prediction_cutoff": "2026-09-20T11:45:00Z"}).status == "NO_DATA"
    assert adapter.execute("query_matches", {"information_mode": "historical_reconstruction"}).status == "FAILED"


def test_real_saved_run_restores_model_and_prediction():
    validation = json.loads((ROOT / "validation.json").read_text(encoding="utf-8"))
    app = DemoApplication(model_catalog=ModelCatalog(ROOT / "catalog.json"), platform_export=ROOT / "binding.json")
    run = app.get("/api/runs/" + validation["run_id"], {})
    assert run["tool_configuration"]["artifact_sha256"] == validation["model_hash"]
    assert run["features"]["parquet_sha256"] == validation["feature_sha256"]
    assert run["prediction"]["model_version"] == "elo-davidson-mapping-1.0"


def test_reference_changed_is_rejected(tmp_path):
    binding = json.loads((ROOT / "binding.json").read_text(encoding="utf-8"))
    binding["feature_snapshot"]["sha256"] = "0" * 64
    target = tmp_path / "binding.json"
    target.write_text(json.dumps(binding), encoding="utf-8")
    with pytest.raises(ValueError, match="hash changed"):
        FrozenFeatureTools(export_path=target, model_id="elo-platform-frozen", model_catalog=ModelCatalog(ROOT / "catalog.json"))


def test_frozen_snapshot_rejects_live_mode():
    with pytest.raises(ValueError, match="archive_replay"):
        FrozenFeatureTools(export_path=ROOT / "binding.json", time_mode="live")
