"""Offline contract tests for the entity-semantic f collection preflight."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import socket

import pytest

from tools.audit_entity_semantic_f_preflight import (
    _network_denied,
    audit_preflight,
)


def _stamp(seconds: float) -> str:
    return f"2026-01-01T00:00:{seconds:04.1f}+00:00"


def _processing(input_at: str, available_at: str, effective_at: str,
                *, status: str = "ready", reason=None) -> dict:
    return {"schema": "entity-semantic-prediction-processing-v1", "status": status,
            "reason": reason, "input_at": input_at, "available_at": available_at,
            "effective_at": effective_at}


def _source(tmp_path: Path, *, failed: bool = False, zero_gain: bool = False,
            bad_pipeline: bool = False, bad_menu: bool = False) -> Path:
    root = tmp_path / "capture"
    run_dir = root / "batch" / "match-1"
    archive = run_dir / "source-archive" / "raw"
    archive.mkdir(parents=True)
    ids = ["web-a", "web-b"]
    public_prediction = {"probabilities": [0.5, 0.3, 0.2]}
    menu = [{"action_id": item, "selected": item == "web-a"} for item in ids]
    raw_ids = ["web-b", "web-a"] if bad_menu else ids
    raw = json.dumps({"generated_actions": [{"action_id": item} for item in raw_ids]},
                     separators=(",", ":")).encode()
    raw_path = archive / "menu.body.gz"
    raw_path.write_bytes(gzip.compress(raw))
    raw_digest = hashlib.sha256(raw).hexdigest()
    before_input, before_available, before_effective, before_observed = _stamp(1), _stamp(.5), _stamp(1), _stamp(2)
    after_input, after_available, after_effective, after_observed = _stamp(5), _stamp(4), _stamp(5), _stamp(6)
    pipeline_before = "entity-market-f:test-pipeline"
    pipeline_after = "entity-market-f:other-pipeline" if bad_pipeline else pipeline_before
    after_status = "not_updated" if failed else "ready"
    after_processing = (_processing(after_input, after_available, None, status=after_status,
                                    reason="semantic_extraction_failed") if failed else
                        _processing(after_input, after_available, after_effective))
    cycle = {
        "step": 0, "generated_actions": [{"action_id": item} for item in ids], "menu": menu,
        "selected_action": {"action_id": "web-a"}, "menu_raw_observation_id": "obs-menu",
        "menu_raw_path": "raw/menu.body.gz", "menu_raw_sha256": raw_digest,
        "public_before": {"prediction": public_prediction},
        "prediction_before": [0.5, 0.3, 0.2], "prediction_after": [0.5, 0.3, 0.2],
        "before_state_timing": {
            "observed_at": before_observed, "forecast_as_of": before_effective,
            "prediction_pipeline_version": pipeline_before,
            "prediction_processing": _processing(before_input, before_available, before_effective),
            "forecast_eligible": True, "recording_scope": "pre_match_context"},
        "after_state_timing": {
            "observed_at": after_observed, "forecast_as_of": None if failed else after_effective,
            "prediction_pipeline_version": pipeline_after,
            "prediction_processing": after_processing,
            "forecast_eligible": not failed,
            "recording_scope": "audit_only" if failed else "pre_match_context"},
        "receipt_available_at": after_input,
        "semantic_prediction": {"available_at": after_effective},
        "update": {"status": "not_updated" if failed else "updated",
                    "model_version": "f-test-v1" if not failed else None},
    }
    if zero_gain:
        cycle["realized_signed_gain"] = 0.0
    run = {"schema": "prospective-acquisition-trajectory-v1", "run_id": "run-1",
           "cycles": [cycle], "outcome": None, "policy_arm": "random_explore",
           "numeric_voi_training_eligible": False}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps(run), encoding="utf-8")
    (run_dir / "configuration.json").write_text(json.dumps({"prediction_pipeline_version": pipeline_before}), encoding="utf-8")
    return run_dir


def test_valid_v4_preflight_is_offline_and_preserves_label_placeholders(tmp_path):
    report = audit_preflight(_source(tmp_path), require_v4=True)
    assert report["status"] == "PASS", report
    assert report["offline"] and report["network_requests"] == 0
    assert report["v4_clocks"]["consistent"]
    assert report["labels"]["run_outcome"] is None
    assert report["f_consumed"][0]["f_consumed"] is True


def test_not_updated_context_is_audit_only_and_not_zero_gain(tmp_path):
    report = audit_preflight(_source(tmp_path, failed=True), require_v4=True)
    assert report["status"] == "PASS", report
    assert report["f_consumed"][0]["f_consumed"] is False
    assert report["labels"]["rows"][0]["realized_signed_gain"] is None
    assert report["labels"]["rows"][0]["label_status"] == "audit_only"


def test_failed_context_with_zero_gain_is_blocked(tmp_path):
    report = audit_preflight(_source(tmp_path, failed=True, zero_gain=True), require_v4=True)
    assert report["status"] == "BLOCKED"
    assert any("carries a label" in error for error in report["errors"])


def test_after_processing_must_bind_to_receipt_available_clock(tmp_path):
    source = _source(tmp_path)
    path = source / "run.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["cycles"][0]["receipt_available_at"] = _stamp(5.5)
    path.write_text(json.dumps(payload), encoding="utf-8")
    report = audit_preflight(source, require_v4=True)
    assert report["status"] == "BLOCKED"
    assert any("input_at must equal receipt_available_at" in error for error in report["errors"])


@pytest.mark.parametrize("kwargs,needle", [
    ({"bad_pipeline": True}, "pipeline_version changes"),
    ({"bad_menu": True}, "raw menu action order differs"),
])
def test_preflight_blocks_frozen_contract_changes(tmp_path, kwargs, needle):
    report = audit_preflight(_source(tmp_path, **kwargs), require_v4=True)
    assert report["status"] == "BLOCKED"
    assert any(needle in error for error in report["errors"]), report["errors"]


def test_declared_cycle_pipeline_cannot_disagree_with_timing(tmp_path):
    source = _source(tmp_path)
    path = source / "run.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["cycles"][0]["prediction_pipeline_version"] = "entity-market-f:wrong"
    path.write_text(json.dumps(payload), encoding="utf-8")
    report = audit_preflight(source, require_v4=True)
    assert report["status"] == "BLOCKED"
    assert any("declared prediction pipeline differs" in error for error in report["errors"])


def test_network_guard_is_hard_failure():
    with _network_denied():
        with pytest.raises(AssertionError, match="offline"):
            socket.getaddrinfo("example.invalid", 443)
