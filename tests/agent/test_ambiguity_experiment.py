"""The entrypoint freezes inputs and executes the declared numerical policy."""
from datetime import datetime, timezone
import hashlib
import json

import pytest
import requests

from football_agent.ambiguity_experiment import run_experiment
from football_agent.inquiry_provider import CodexInquiryProvider


def test_complete_experiment_preserves_supported_and_impossible_branches_without_network(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("The constructed experiment must not call HTTP or a real provider")
    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    monkeypatch.setattr(CodexInquiryProvider, "decide", forbidden)
    root = tmp_path/"artifact"
    now = datetime(2026, 9, 27, 2, tzinfo=timezone.utc)
    result = run_experiment(root, clock=lambda: now)
    comparison = result["comparison"]
    assert set(comparison["point_actions"].values()) == {"STOP"}
    assert comparison["shared_stop_loss"] == pytest.approx(0.2)
    assert comparison["fill_j_worst_total_risk"] == pytest.approx(0.22)
    assert comparison["verify_k_worst_total_risk"] == pytest.approx(0.03)
    assert comparison["verify_k_robust_risk_reduction"] == pytest.approx(0.17)
    assert result["scripted_provider_calls"] == result["constructed_tool_calls"] == 2
    assert result["real_llm_calls"] == result["external_http_requests"] == result["football_policy_gain_evidence"] == 0
    supported, outside = result["cases"]
    assert supported["state"] == "STOPPED"
    assert supported["terminal_decision"]["decision"] == "K_no__J_yes"
    assert supported["active_model_ids_after_return"] == ["assume_no"]
    assert outside["state"] == "DEFERRED" and outside["terminal_decision"] is None
    assert outside["model_set_status_after_return"] == "MODEL_SET_MISSPECIFIED"
    for case in result["cases"]:
        assert case["proposed_candidate"] == "fill_j" and case["executed_candidate"] == "verify_k"
        assert case["budget"]["max_llm_calls"] == case["budget"]["llm_calls"] == 1
        proof = json.loads((root/case["case_id"]/"verification.json").read_text(encoding="utf-8"))
        assert proof["frozen_terminal_decisions_replayed"] == 1
    manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    assert manifest["private_return_tapes_are_policy_input"] is False
    assert manifest["runtime"]["external_source_snapshot"] is False
    snapshots = {record["snapshot"].replace("\\", "/") for record in manifest["sources"]}
    assert "execution-source/football_agent/inquiry.py" in snapshots
    assert "execution-source/football_agent/ambiguity_value.py" in snapshots
    assert "execution-source/football_spider/storage/raw_archive.py" in snapshots
    assert "execution-source/football_platform/collectors/http_transport.py" in snapshots
    for record in manifest["sources"]:
        assert hashlib.sha256((root/record["snapshot"]).read_bytes()).hexdigest() == record["sha256"]
    for record in manifest["frozen_inputs"]:
        assert hashlib.sha256((root/record["path"]).read_bytes()).hexdigest() == record["sha256"]


def test_existing_root_is_preserved_and_rejected_before_execution(tmp_path):
    marker = tmp_path/"previous-result.txt"
    marker.write_text("Preserve this prior attempt", encoding="utf-8")
    with pytest.raises(ValueError, match="existing artifacts are never overwritten"):
        run_experiment(tmp_path)
    assert marker.read_text(encoding="utf-8") == "Preserve this prior attempt"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["previous-result.txt"]
