"""No-network report contracts for explicit v5 runs without a market prior."""
from copy import deepcopy
from hashlib import sha256
import json

import pytest

from football_agent.match_analysis_report import build_analysis_report, write_analysis_report


MATCH = {"match_id": "evidence-only-match", "home_team": "Home", "away_team": "Away"}
NOTE = "缺少可用市场先验，仅整理已获取证据，不提供胜平负概率或数值信息增益"


def evidence_run():
    action = {"action_id": "read-article", "tool": "read_url", "source": "club",
              "url": "https://club.example/news/preview", "cost": {"requests": 1}}
    return {
        "schema": "learned-voi-open-execution-v5", "run_id": "evidence-only",
        "prior_status": "prior_unavailable", "analysis_mode": "evidence_only",
        "initial_probabilities": None, "final_probabilities": None,
        "terminal_selection": {"decision": "defer", "reason": "defer_prior_unavailable", "stop_certificate": None},
        "measured_resources": {"requests": 1}, "unmeasured_resource_receipts": {"requests": 0},
        "attempts": [{"action_id": action["action_id"], "status": "OK"}],
        "cycles": [{"selected_action": action, "generated_actions": [action],
                    "selection_basis": "bounded_unknown_information_exploration",
                    "execution_status": "returned", "prediction_before": None, "prediction_after": None,
                    "strict_decision": {"candidates": [{"action_id": action["action_id"], "value": {
                        "status": "UNKNOWN", "net_voi": None, "gross_evsi": None, "reason": "prior_unavailable"}}]}}],
        "evidence": {"match_id": MATCH["match_id"], "evidence": [{
            "evidence_id": "quote-1", "topics": ["availability"],
            "applicability": "match_scoped_source_assertion", "modality": "uncertain_source_statement",
            "anchor": {"url": action["url"], "document_id": "doc-1", "quote": "The player is doubtful."},
        }], "state": {"coverage_complete": True}},
    }


@pytest.mark.parametrize("decision,reason", [
    ("defer", "defer_prior_unavailable"), ("halt", "physical_budget_exhausted"),
    ("halt", "forecast_cutoff_elapsed"), ("halt", "step_limit"),
])
def test_finished_evidence_run_writes_gaps_without_probabilities_or_gain(tmp_path, decision, reason):
    run = evidence_run()
    run["terminal_selection"].update(decision=decision, reason=reason)
    original = deepcopy(run)
    written = write_analysis_report(run, match=MATCH, output=tmp_path)
    assert run == original
    report = json.loads((tmp_path / "analysis.json").read_text(encoding="utf-8"))
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert written["status"] == report["status"] == "completed_with_gaps"
    assert report["analysis_mode"] == "evidence_only" and report["prior_status"] == "prior_unavailable"
    assert report["gaps"] == ["prior_unavailable"] and report["incomplete_reasons"] == []
    assert report["probabilities"]["initial"] is report["probabilities"]["final"] is None
    assert report["probabilities"]["changed"] is None
    assert report["probabilities"]["initial_probability_basis"] == "unavailable"
    assert report["decisions"][0]["prediction_before"] is report["decisions"][0]["prediction_after"] is None
    assert report["decisions"][0]["value"]["net_voi"] is None
    assert report["evidence"][0]["source_quote"] == "The player is doubtful."
    assert report["budget"]["measured_resources"]["requests"]["total"] == 1
    assert report["semantics"]["numeric_information_gain_available"] is False
    assert report["analysis_note_zh"] == NOTE and NOTE in markdown
    assert "概率沿用" not in markdown and "本轮分析尚未完整完成" not in markdown
    assert "主胜 33.33%" not in markdown and "club.example/news/preview" in markdown


@pytest.mark.parametrize("execution,reason", [
    ("execution_outcome_unknown", "execution_reconciliation_required"),
    ("receipt_rejected", "receipt_reconciliation_required"),
    ("returned", "actual_resource_budget_overrun"),
])
def test_execution_failures_remain_incomplete(execution, reason):
    run = evidence_run()
    run["cycles"][0]["execution_status"] = execution
    run["terminal_selection"].update(decision="halt", reason=reason)
    report = build_analysis_report(run, match=MATCH)
    assert report["status"] == "incomplete" and reason in report["incomplete_reasons"]
    assert "forecast_missing" not in report["incomplete_reasons"]
    if execution != "returned":
        run["terminal_selection"].update(reason="step_limit")
        run["cycles"][0]["selected_action"] = None
        assert build_analysis_report(run, match=MATCH)["status"] == "incomplete"


@pytest.mark.parametrize("key", ["initial_probabilities", "final_probabilities", "prediction_before", "prediction_after"])
def test_evidence_only_cannot_display_fabricated_probabilities(key):
    run = evidence_run()
    target = run["cycles"][0] if key.startswith("prediction_") else run
    target[key] = [1 / 3, 1 / 3, 1 / 3]
    with pytest.raises(ValueError, match="all predictions to be None"):
        build_analysis_report(run, match=MATCH)


@pytest.mark.parametrize("gain_source", ["computed", "empirical", "context", "continuation"])
def test_evidence_only_cannot_display_numeric_gains(gain_source):
    run = evidence_run()
    candidate = run["cycles"][0]["strict_decision"]["candidates"][0]
    if gain_source == "computed":
        candidate["value"].update(status="COMPUTED", net_voi=.1, gross_evsi=.2)
    elif gain_source == "empirical":
        candidate["empirical_risk_estimate"] = {"expected_gain": .1}
    elif gain_source == "context":
        candidate["value"].update(schema_version="context-empirical-query-value-v1", expected_signed_gain=.1)
    else:
        candidate["continuation_enumeration"] = {}
        candidate["response_enumeration"] = {"value": {"gross_evsi": .2, "net_voi": .1}}
    with pytest.raises(ValueError, match="numeric information gain"):
        build_analysis_report(run, match=MATCH)


def test_evidence_only_requires_explicit_markers_and_cannot_claim_numeric_stop():
    run = evidence_run()
    run.pop("prior_status")
    with pytest.raises(ValueError, match="explicit prior_unavailable"):
        build_analysis_report(run, match=MATCH)
    run["prior_status"] = "prior_unavailable"
    run["terminal_selection"]["stop_certificate"] = {"claimed": True}
    with pytest.raises(ValueError, match="numeric stop certificate"):
        build_analysis_report(run, match=MATCH)


@pytest.mark.parametrize("version", [1, 2, 3, 4])
def test_old_schema_outputs_stay_identical_even_with_evidence_only_markers(tmp_path, version):
    run = {"schema": f"learned-voi-open-execution-v{version}", "run_id": "compatibility",
           "prior_status": "prior_unavailable", "analysis_mode": "evidence_only",
           "initial_probabilities": None, "final_probabilities": None, "cycles": [],
           "terminal_selection": {"decision": "halt", "reason": "physical_budget_exhausted"}}
    write_analysis_report(run, match=MATCH, output=tmp_path)
    report = json.loads((tmp_path / "analysis.json").read_text(encoding="utf-8"))
    assert report["status"] == "incomplete" and "forecast_missing" in report["incomplete_reasons"]
    encoded = json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    # Captured from the unchanged report implementation before the v5 branch.
    assert sha256(encoded).hexdigest() == "119b30b57ed75caedb2d0afe5d2192f4c85dc09050be8b0ac5e695d0ade02d66"
    assert sha256((tmp_path / "report.md").read_bytes()).hexdigest() == "bb954ffb0ce459dc4c4c78793808101a5008e3cb17e8940ef45b670da25923a9"
