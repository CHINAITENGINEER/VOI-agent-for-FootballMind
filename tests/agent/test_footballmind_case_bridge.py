"""Offline case bridge acceptance; synthetic values are not football evidence."""
import json
from pathlib import Path

from tools.check_footballmind_case_bridge import (
    check_source_case, run_synthetic_bridge,
)


def test_synthetic_case_bridge_runs_h1_h2_report_and_replay(tmp_path):
    summary = run_synthetic_bridge(tmp_path / "bridge")
    assert summary["status"] == "passed"
    assert summary["claims_boundary"].startswith("not_real_match_evidence")
    xor = summary["checks"]["two_step_xor"]
    assert xor["h1_selected"] is None
    assert xor["h2_selected"]
    assert xor["first_immediate_net"] < 0
    assert xor["first_planning_net"] > 0
    assert xor["observations"] == 2
    assert summary["checks"]["math_replay"]["status"] == "passed"
    assert summary["resources"]["actual_network"] is False
    assert summary["resources"]["real_provider_calls"] == 0
    assert Path(summary["evidence_paths"]["report"]).exists()
    run = json.loads(Path(summary["evidence_paths"]["run"]).read_text(encoding="utf-8"))
    assert all(item["returned"]["test_only"] is True and
               item["returned"]["real_provider_call"] is False
               for item in run["http_receipts"])
    assert all(item["receipt"]["test_only"] is True and
               item["receipt"]["real_provider_call"] is False
               for item in run["semantic_calls"])


def test_existing_source_case_conversion_does_not_inject_synthetic_value(tmp_path):
    source = Path("var/submission/p1-single-match/initial-state")
    summary = check_source_case(source, tmp_path / "source-case")
    assert summary["status"] == "passed"
    assert summary["claims_boundary"].startswith("conversion_integrity_only")
    case = json.loads(Path(summary["evidence_paths"]["case"]).read_text(encoding="utf-8"))
    original = json.loads(source.joinpath("case.json").read_text(encoding="utf-8"))
    assert case["structured_baseline"]["probabilities"] == original["structured_baseline"]["probabilities"]
    assert case["source_profile"]["bootstrap_url"] in case["source_profile"]["seed_urls"]
    assert summary["resources"]["http_requests"] == 0
    assert summary["resources"]["semantic_calls"] == 0
