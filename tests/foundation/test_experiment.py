"""Pinned replay planning and execution without remote sources or training."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest

from footballmind.config import load_config
from footballmind import experiment as ex


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def setup(tmp_path):
    ws = load_config(environ={}, workspace_root=tmp_path)
    release = ws.release_dir
    release.mkdir(parents=True)
    bodies = {
        "footballmind_defense.py": b"raise AssertionError('planning must not import this file')",
        "data/embeddings.npz": b"pinned-vectors-not-loaded-by-plan",
        "models/f/model.pt": b"pinned-f",
        "models/g/g-models.joblib": b"pinned-g",
    }
    fs = hashlib.sha256(bodies["models/f/model.pt"]).hexdigest()
    gs = hashlib.sha256(bodies["models/g/g-models.joblib"]).hexdigest()
    ws = replace(ws, f_sha256=fs, g_sha256=gs)
    cases = [{"case_id": c, "physical_match_id": "match-" + c,
              "split_role": "g_dev", "documents": [{"document_id": c + "-body"}]}
             for c in ["b", "a"]]
    bodies["data/cases.json"] = json.dumps(cases).encode()
    bodies["bundle-config.json"] = json.dumps({"f_model_sha256": fs, "g_model_sha256": gs,
        "f_model_path": "models/f", "g_model_path": "models/g/g-models.joblib",
        "models": {h: {} for h in ex.HEADS}}).encode()
    for name, data in bodies.items():
        path = release / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    manifest = {"bundle_id": ws.release_id,
                "files_sha256": {name: hashlib.sha256(b).hexdigest() for name, b in bodies.items()}}
    write_json(release / "manifest.json", manifest)
    raw = {"schema": ex.SCHEMA, "experiment_id": "test-replay",
           "expected_manifest_sha256": ex._sha(release / "manifest.json"),
           "dataset_role": "g_dev", "heads": ["state_ridge", "fixed_order"], "max_reads": 3}
    config = tmp_path / "replay.json"
    write_json(config, raw)
    return ws, config, raw


def test_plan_is_deterministic_and_has_no_side_effects_or_weight_load(setup, monkeypatch):
    ws, config, raw = setup
    monkeypatch.setattr(ex, "_load_backend", lambda *a: pytest.fail("plan loaded weights"))
    first = ex.experiment_plan(config, ws)
    second = ex.experiment_plan(config, ws)
    assert first == second
    assert first["case_ids"] == ["a", "b"]
    assert first["physical_match_ids"] == ["match-a", "match-b"]
    assert first["scope"] == "frozen_development_bundle_replay_not_new_blind_evaluation"
    assert first["training_calls"] == first["network_requests"] == 0
    assert not first["weights_loaded"]
    assert not ws.experiments_dir.exists()


@pytest.mark.parametrize("patch", [{"max_reads": True}, {"max_reads": 0}, {"max_reads": 21},
    {"heads": ["state_ridge", "state_ridge"]}, {"heads": ["unknown"]},
    {"case_ids": ["unknown"]}, {"case_ids": ["a", "a"]}, {"case_ids": []},
    {"settle": 1}, {"dataset_role": "test"}, {"unexpected": 1}])
def test_invalid_configuration_is_not_silently_corrected(setup, patch):
    ws, config, raw = setup
    write_json(config, {**raw, **patch})
    with pytest.raises(ValueError):
        ex.experiment_plan(config, ws)
    assert not ws.experiments_dir.exists()


def test_manifest_and_input_sha_are_checked_before_execution(setup):
    ws, config, raw = setup
    write_json(config, {**raw, "expected_manifest_sha256": "a" * 64})
    with pytest.raises(ValueError, match="manifest SHA"):
        ex.experiment_plan(config, ws)
    write_json(config, raw)
    (ws.release_dir / "models/f/model.pt").write_bytes(b"changed-model")
    with pytest.raises(ValueError, match="input SHA mismatch"):
        ex.experiment_plan(config, ws)


def test_manifest_path_escape_is_rejected(setup):
    ws, config, raw = setup
    manifest_path = ws.release_dir / "manifest.json"
    manifest = ex._read(manifest_path)
    manifest["files_sha256"]["../escape"] = "a" * 64
    write_json(manifest_path, manifest)
    write_json(config, {**raw, "expected_manifest_sha256": ex._sha(manifest_path)})
    with pytest.raises(ValueError, match="escapes"):
        ex.experiment_plan(config, ws)


def test_output_never_overwrites_inputs_or_existing_results(setup):
    ws, config, raw = setup
    for output in [str(ws.release_dir), "../outside", "."]:
        write_json(config, {**raw, "output_dir": output})
        with pytest.raises(ValueError):
            ex.experiment_plan(config, ws)
    # Also catch the inverse overlap when a configured experiment parent is
    # broader than the frozen release directory.
    write_json(config, {**raw, "output_dir": "var"})
    with pytest.raises(ValueError, match="overlaps frozen"):
        ex.experiment_plan(config, replace(ws, experiments_dir=ws.workspace_root))
    write_json(config, raw)
    output = ws.experiments_dir / raw["experiment_id"]
    output.mkdir(parents=True)
    (output / "keep.txt").write_text("original")
    with pytest.raises(FileExistsError):
        ex.run_experiment(config, ws)
    assert (output / "keep.txt").read_text() == "original"


def install_backend(monkeypatch, *, fail=False, invalid=False):
    calls = []
    backend = SimpleNamespace()
    def run(backend, case, head, maximum):
        calls.append(("predict", case, head, maximum))
        if fail and head == "fixed_order":
            raise ValueError("protocol rejected")
        return {"status": "STOPPED", "case_id": case, "physical_match_id": "match-" + case,
                "run_id": case + head, "p_before": [.4, .3, .3],
                "p_after": [float("nan"), .3, .3] if invalid else [.4, .3, .3],
                "selected_ids": [], "body_adoptions": 0, "stop_reason": "threshold"}
    def settle(result):
        assert len([c for c in calls if c[0] == "predict"]) == 4
        calls.append(("settle", result["case_id"]))
        return {"status": "settled_historical_diagnostic", "loss_before": 1.0,
                "loss_after": 1.0, "realized_gain": 0.0}
    backend.settle_run = settle
    monkeypatch.setattr(ex, "_load_backend", lambda *a: backend)
    monkeypatch.setattr("demo.defense_budget.run_with_limit", run)
    return calls


def test_run_records_unchanged_valid_predictions_without_reading_outcomes(setup, monkeypatch):
    ws, config, raw = setup
    calls = install_backend(monkeypatch)
    summary = ex.run_experiment(config, ws)
    assert summary["status"] == "COMPLETED"
    assert summary["completed_predictions"] == 4 and summary["failed_predictions"] == 0
    assert all(c[0] == "predict" for c in calls)
    output = Path(summary["output_dir"])
    rows = [json.loads(r) for r in (output / "predictions.jsonl").read_text().splitlines()]
    assert [(r["case_id"], r["head"]) for r in rows] == [(c, h) for c in ["a", "b"] for h in raw["heads"]]
    assert rows[0]["prediction"]["p_before"] == rows[0]["prediction"]["p_after"]
    assert not (output / "settlements.jsonl").exists()
    assert ex._read(output / "config.json") == raw
    hashes = ex._read(output / "artifacts.json")
    assert all(ex._sha(output / name) == sha for name, sha in hashes.items())
    with pytest.raises(FileExistsError):
        ex.run_experiment(config, ws)


def test_protocol_failure_is_retained_without_zero_gain(setup, monkeypatch):
    ws, config, raw = setup
    install_backend(monkeypatch, fail=True)
    summary = ex.run_experiment(config, ws)
    assert summary["status"] == "COMPLETED_WITH_FAILURES"
    assert summary["completed_predictions"] == summary["failed_predictions"] == 2
    failures = [json.loads(r) for r in (Path(summary["output_dir"]) / "failures.jsonl").read_text().splitlines()]
    assert all(r["gain"] is None and r["p_after"] is None for r in failures)


def test_nonfinite_prediction_is_a_failure_not_a_valid_after(setup, monkeypatch):
    ws, config, raw = setup
    install_backend(monkeypatch, invalid=True)
    summary = ex.run_experiment(config, ws)
    assert summary["failed_predictions"] == 4 and summary["completed_predictions"] == 0
    assert not (Path(summary["output_dir"]) / "predictions.jsonl").read_text()


def test_settlement_is_opt_in_and_follows_all_inference(setup, monkeypatch):
    ws, config, raw = setup
    write_json(config, {**raw, "settle": True})
    calls = install_backend(monkeypatch)
    summary = ex.run_experiment(config, ws)
    assert summary["settled_predictions"] == 4
    assert [c[0] for c in calls] == ["predict"] * 4 + ["settle"] * 4
    assert summary["historical_diagnostic_metrics"]["state_ridge"]["mean_realized_gain"] == 0.0


def test_backend_load_failure_is_persisted_and_network_restored(setup, monkeypatch):
    ws, config, raw = setup
    original = socket.create_connection
    def fail(*args):
        with pytest.raises(RuntimeError, match="prohibits network"):
            socket.create_connection(("example.com", 443))
        raise FileNotFoundError("missing frozen file")
    monkeypatch.setattr(ex, "_load_backend", fail)
    result = ex.run_experiment(config, ws)
    assert result["status"] == "FAILED" and result["completed_predictions"] == 0
    assert socket.create_connection is original
    assert ex._read(Path(result["output_dir"]) / "summary.json")["error_type"] == "FileNotFoundError"
