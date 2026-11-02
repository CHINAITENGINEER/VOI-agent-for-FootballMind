"""Constructed report fixtures only; these tests are not football evidence."""
from copy import deepcopy
import json

import pytest

from football_agent.semantic_scenario_model import PLANNING_POLICY_VERSION, SemanticScenarioModel
from football_agent.semantic_scenario_report import write_semantic_report


def constructed_run(planning_horizon=1):
    proposal = {"schema": "semantic-scenario-proposal-v1", "hypotheses": [
        {"id": "h1", "description": "构造情景一", "prior": .5,
         "outcome_probabilities": [.8, .1, .1]},
        {"id": "h2", "description": "构造情景二", "prior": .5,
         "outcome_probabilities": [.1, .1, .8]}],
        "actions": [{"action_id": action, "question": f"构造问题 {action} 是否成立？",
                     "observation_group": action, "likelihoods": [
                         {"hypothesis_id": "h1", "probabilities": [.8, .1, .1]},
                         {"hypothesis_id": "h2", "probabilities": [.1, .8, .1]}]}
                    for action in ("a", "b")],
        "assumptions": ["所有内容均为构造测试，不是足球实证。"]}
    baseline = [.45, .1, .45]
    model = SemanticScenarioModel(proposal, baseline, planning_horizon=planning_horizon)
    bindings = {action: {"action_id": action, "tool": "read_url", "url": f"https://example.invalid/{action}",
                         "request_cost": 1, "return_contract": "one_actual_article_body_or_no_answer"}
                for action in ("a", "b")}
    observations, cycles, receipts = [], [], []
    for index, outcome in enumerate(("no_answer", "supports", None)):
        decision = model.evaluate(observations, 4 - index)
        cycles.append({"index": index, "decision": decision, "model": model.manifest(),
                       "observations": deepcopy(observations), "candidate_bindings": bindings,
                       "remaining_requests": 4 - index, "requests_spent": index})
        if outcome is None:
            break
        action = decision["selected_action_id"]
        assert action is not None
        document = {"document_id": f"doc-{action}", "url": bindings[action]["url"],
                    "status": "OK", "kind": "article", "title": f"构造来源 {action}",
                    "text": "这是构造测试的连续正文，不是现实比赛证据。", "links": [],
                    "observed_at": "2026-01-01T00:00:00Z", "available_at": "2026-01-01T00:00:00Z",
                    "source_published_at": "2025-12-31T00:00:00Z"}
        observations.append({"action_id": action, "outcome": outcome, "reason": "构造分类结果。",
                             "applicability": "unresolved" if outcome == "no_answer" else "match_specific",
                             "evidence": [] if outcome == "no_answer" else [
                                 {"document_id": document["document_id"], "quote": "这是构造测试的连续正文"}],
                             "returned_document_ids": [document["document_id"]]})
        receipts.append({"index": index, "candidate": bindings[action], "admitted": document,
                         "purpose": "voi_selected_body_read", "returned": {
                             "http_attempts": 1, "latency_seconds": .1,
                             "observation": {"http_status": 200, "content_hash": "constructed-test-hash"}}})
    return {"run_id": "constructed-report-test", "status": "completed", "configuration": {
        "scope": "constructed_report_test_not_football_evidence", "baseline_scope": "constructed",
        "match": {"home_name": "构造主队", "away_name": "构造客队", "match_id": "constructed",
                  "kickoff": "2026-01-02T12:00:00Z", "prediction_cutoff": "2026-01-02T00:00:00Z"},
        "request_loss_cost": .001, "contamination_levels": [0., .25, .5],
        "planning_horizon": planning_horizon, "planning_policy": PLANNING_POLICY_VERSION,
        "request_budget": 4, "semantic_call_budget": 3},
        "initial_prediction": baseline, "final_prediction": cycles[-1]["decision"]["prediction_nominal"],
        "proposal": proposal, "observations": observations, "cycles": cycles, "http_receipts": receipts,
        "terminal": cycles[-1]["decision"], "semantic_calls": [{"index": 0, "status": "accepted",
            "protocol": "semantic-scenario-discovery-v1", "validated": {
                "schema": "semantic-scenario-discovery-v1", "queries": [
                    {"query": "constructed query", "information_need": "构造信息需求"}]},
            "receipt": {"usage": [{"input_tokens": 100, "output_tokens": 20, "cached_input_tokens": 30}],
                        "real_provider_call": False}}],
        "resources": {"http_requests": 2, "semantic_calls": 1, "http_latency_seconds": .2,
                      "semantic_latency_seconds": 0., "actual_money": None},
        "evidence_scope": {"real_source_acquisition": False, "future_outcome_loaded": False}}


def test_report_preserves_saved_values_quotes_and_separates_assumptions(tmp_path):
    run = constructed_run()
    before = deepcopy(run)
    summary = write_semantic_report(run, tmp_path)
    analysis = json.loads((tmp_path / "analysis.json").read_text(encoding="utf-8"))
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert run == before
    assert analysis["initial_prediction"] == run["initial_prediction"]
    assert analysis["final_prediction"] == run["final_prediction"]
    assert analysis["semantic_assumptions"]["hypotheses"] == run["proposal"]["hypotheses"]
    assert analysis["semantic_assumptions"]["actions"] == run["proposal"]["actions"]
    assert analysis["conditional_voi"]["cycles"][0]["candidates"][0]["net"] == run["cycles"][0]["decision"]["candidates"][0]["net"]
    assert analysis["no_answer_checks"][0]["unchanged_within_1e_12"] is True
    assert analysis["resources"]["reported_token_totals"] == {
        "input_tokens": 100, "output_tokens": 20, "cached_input_tokens": 30}
    assert "构造测试数据" in markdown and "不是足球实证" in markdown
    assert "constructed query" in markdown and "构造信息需求" in markdown
    assert "> 这是构造测试的连续正文" in markdown
    assert "不是置信区间" in markdown and "不是统计校准" in markdown
    assert "未报告，不能视为零" in markdown
    assert "不是全开放世界停止证书" in markdown
    assert summary["decision_cycles"] == 3
    assert summary["global_stop_certificate"] is False
    assert analysis["directional_evidence"]["available"] is False
    assert analysis["directional_evidence"]["used_for_voi"] is False
    assert "不自动解释为主队或客队利空/利好" in markdown


def test_two_step_report_exposes_shared_branch_policy_and_separate_values(tmp_path):
    run = constructed_run(planning_horizon=2)
    before = deepcopy(run)
    write_semantic_report(run, tmp_path)
    analysis = json.loads((tmp_path / "analysis.json").read_text(encoding="utf-8"))
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert run == before
    assert analysis["conditional_voi"]["meaning"] == (
        "branchwise_robust_shared_two_step_policy_conditional_on_declared_assumptions")
    candidates = analysis["conditional_voi"]["cycles"][0]["candidates"]
    for saved, rendered in zip(run["cycles"][0]["decision"]["candidates"], candidates):
        assert rendered["planning_net"] == saved["planning_net"]
        assert rendered["immediate_lower"] == saved["immediate_lower"]
        assert rendered["planning_lower"] == saved["planning_lower"]
        assert rendered["continuation_policy"] == saved["continuation_policy"]
        for branch in saved["continuation_policy"]:
            assert f"{branch['observation']} → {branch['action_id']}" in markdown
    assert "一步下包络 | 两步名义净值 | 两步下包络" in markdown
    assert "查询前的条件计划，不是已执行记录" in markdown
    assert "不声称是全局最优策略树" in markdown
    assert "同一观测组只尝试一次" not in markdown
    assert "无答案后的后备来源" in markdown


def test_report_preserves_opt_in_directional_annotation_without_using_it_for_voi(tmp_path):
    run = constructed_run()
    run["observations"][1].update({
        "schema": "semantic-directional-observation-v1",
        "direction": "unknown", "subject": "unknown",
        "direction_basis": "availability_only",
        "directional_claim": "正文有健康状态信息，但没有影响方向。",
    })
    write_semantic_report(run, tmp_path)
    analysis = json.loads((tmp_path / "analysis.json").read_text(encoding="utf-8"))
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert analysis["directional_evidence"]["available"] is True
    assert analysis["directional_evidence"]["used_for_voi"] is False
    assert analysis["directional_evidence"]["records"][0]["direction"] == "unknown"
    assert "availability_only" in markdown
    assert "未进入后验或 VoI" in markdown


def test_no_answer_checks_all_model_posteriors_and_does_not_hide_mismatch(tmp_path):
    run = constructed_run()
    run["cycles"][1]["decision"]["model_posteriors"]["contamination_0.5"]["h1"] += .03
    write_semantic_report(run, tmp_path)
    analysis = json.loads((tmp_path / "analysis.json").read_text(encoding="utf-8"))
    check = analysis["no_answer_checks"][0]
    assert check["unchanged_within_1e_12"] is False
    assert check["max_posterior_delta"] == pytest.approx(.03)
    assert "不一致，需检查" in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_incomplete_run_does_not_invent_no_answer_verification_or_execution(tmp_path):
    run = constructed_run()
    run["cycles"] = run["cycles"][:1]
    run["status"] = "incomplete"
    run["terminal"] = {"stop_reason": "runtime_incomplete"}
    run["observations"] = run["observations"][:1]
    write_semantic_report(run, tmp_path)
    analysis = json.loads((tmp_path / "analysis.json").read_text(encoding="utf-8"))
    assert analysis["no_answer_checks"][0]["unchanged_within_1e_12"] is None
    assert "记录不足，未核对" in (tmp_path / "report.md").read_text(encoding="utf-8")
    run["observations"] = []
    write_semantic_report(run, tmp_path)
    assert "没有保存该动作的完成观测" in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_failed_proposal_still_reports_missing_probabilities_and_costs(tmp_path):
    run = {"status": "incomplete", "configuration": {"scope": "constructed_report_test"},
           "proposal": None, "terminal": {"stop_reason": "runtime_incomplete"},
           "semantic_calls": [{"index": 0, "status": "failed", "error": "constructed failure"}]}
    summary = write_semantic_report(run, tmp_path)
    assert summary["initial_prediction"] is None and summary["final_prediction"] is None
    analysis = json.loads((tmp_path / "analysis.json").read_text(encoding="utf-8"))
    assert analysis["resources"]["calls_without_token_usage"] == 1
    assert analysis["semantic_assumptions"]["hypotheses"] == []
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "未取得已验证提案" in markdown
    assert "没有保存完整数学决策轮次" in markdown


def test_zero_cost_exercise_is_labeled_and_unknown_http_count_is_not_zero(tmp_path):
    run = constructed_run()
    run["configuration"].update(request_loss_cost=0,
        study_role="zero_loss_cost_mechanism_exercise_not_policy_success")
    run["http_receipts"].append({"index": 2, "purpose": "constructed_failure",
        "candidate": {"tool": "read_url", "url": "https://example.invalid/failure"},
        "status": "reconciliation_required", "reserved_requests": 1,
        "returned": {"http_attempts": None}, "error": "constructed projection failure"})
    run["resources"].update(request_budget_reserved=3, http_accounting_complete=False)
    summary = write_semantic_report(run, tmp_path)
    analysis = json.loads((tmp_path / "analysis.json").read_text(encoding="utf-8"))
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert summary["http_requests"] is None
    assert summary["http_requests_confirmed"] == 2
    assert summary["http_accounting_complete"] is False
    assert analysis["source_facts"]["sources"][-1]["admitted_body_characters"] is None
    assert "零损失成本机制诊断，不能作为主策略成功证据" in markdown
    assert "实际 HTTP 请求总量未知" in markdown
    assert "缺失回执不能视为零次请求" in markdown
    assert "constructed projection failure" in markdown
