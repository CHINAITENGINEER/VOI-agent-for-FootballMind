import json
from pathlib import Path

import pytest

from football_agent.policy_benchmark import POLICIES, compare_policies
from football_agent.round133_policy_integration import fixture_to_benchmark_episode
from football_agent.round134_policy_report import (
    TRACKED_METRICS,
    VERSION,
    build_report,
    render_markdown,
    write_report,
)
from football_agent.scenario_fixture_pack import get_scenario_fixture


def benchmark():
    episodes = [
        fixture_to_benchmark_episode(get_scenario_fixture("qualified-but-strong-performance")),
        fixture_to_benchmark_episode(get_scenario_fixture("weather-low-value")),
    ]
    return compare_policies(episodes, policies=POLICIES, seed=7)


def test_report_projects_each_episode_and_preserves_observed_denominators():
    report = build_report(benchmark())
    assert report["schema_version"] == VERSION
    assert report["source"]["episode_count"] == 2
    current = report["policies"]["current_voi_gate"]
    assert len(current["episodes"]) == 2
    assert current["summary"]["episode_count"] == 2
    for metric in TRACKED_METRICS:
        coverage = current["summary"]["metric_coverage"][metric]
        assert coverage["observed_count"] + coverage["unknown_count"] == 2
        assert coverage["denominator"] == 2
    assert current["summary"]["regret"]["observed_denominator"] == current["summary"]["metric_coverage"]["regret_vs_oracle"]["observed_count"]
    assert current["summary"]["stop_quality"]["observed_denominator"] == 0


def test_unknowns_remain_explicit_and_stop_reason_is_visible():
    report = build_report(benchmark())
    row = report["policies"]["oracle_upper_bound"]["episodes"][0]
    assert row["query_count"] is None
    assert row["resource_cost"] is None
    assert row["stop_reason"] == "oracle_ineligible"
    assert "regret_vs_oracle" in row["unknown_metrics"]
    summary = report["policies"]["oracle_upper_bound"]["summary"]
    assert summary["stop_reason_counts"] == {"oracle_ineligible": 2}
    assert summary["metric_coverage"]["regret_vs_oracle"]["unknown_count"] == 2


def test_report_does_not_copy_evaluator_or_scenario_private_material():
    report = build_report(benchmark())
    encoded = json.dumps(report, ensure_ascii=False).lower()
    for private_key in ("candidate_evaluations", "evaluation_sidecar", "expected_policy", "evaluation_reference", "true_outcome", "3-0"):
        assert private_key not in encoded
    assert report["evidence_boundary"]["match_result_improvement_claim"] is False
    assert report["evidence_boundary"]["numeric_voi_training_eligible"] is False

    malicious = benchmark()
    malicious["aggregate"]["fixed_all"]["evaluation_sidecar"] = {"secret": True}
    encoded = json.dumps(build_report(malicious), ensure_ascii=False).lower()
    assert "evaluation_sidecar" not in encoded


def test_markdown_exposes_cost_stop_unknown_and_boundary():
    markdown = render_markdown(build_report(benchmark()))
    assert "mean query count (observed denominator)" in markdown
    assert "stop reason" in markdown
    assert "unknown" in markdown
    assert "match-result improvement claim: disabled" in markdown
    assert "regret_vs_oracle" in markdown
    assert "candidate_evaluations" not in markdown


def test_report_rejects_non_offline_contract_and_aggregate_drift():
    bad_contract = benchmark()
    bad_contract["contract"]["synthetic_or_offline_only"] = False
    with pytest.raises(ValueError, match="offline"):
        build_report(bad_contract)

    bad_denominator = benchmark()
    bad_denominator["aggregate"]["fixed_all"]["final_loss_observed_denominator"] = 99
    with pytest.raises(ValueError, match="inconsistent"):
        build_report(bad_denominator)


def test_report_rejects_misaligned_episode_rows_and_roundtrips(tmp_path: Path):
    bad = benchmark()
    bad["rows"]["fixed_all"][0]["episode_id"] = "wrong"
    with pytest.raises(ValueError, match="ordering mismatch"):
        build_report(bad)

    report = build_report(benchmark())
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    write_report(report, json_path, markdown_path)
    restored = json.loads(json_path.read_text(encoding="utf-8"))
    assert restored == report
    assert "FootballMind" in markdown_path.read_text(encoding="utf-8")
