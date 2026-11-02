"""Catalog visibility and continuation provenance are audit scopes, not evidence."""
from copy import deepcopy
import hashlib
import json

from football_agent.semantic_scenario_report import write_semantic_report


def _legacy_run():
    return {
        "run_id": "legacy-catalog-report-compatibility", "status": "incomplete",
        "configuration": {"scope": "constructed_report_test",
                          "match": {"home_name": "A", "away_name": "B"}},
        "terminal": {"stop_reason": "runtime_incomplete"},
    }


def _discovery():
    return {
        "proposal": {"schema": "semantic-scenario-discovery-v2", "queries": [],
                     "article_candidates": [
                         {"action_id": "read-001", "information_need": "确认本场出场可能性"},
                         {"action_id": "read-002", "information_need": "确认另一个不确定因素"}]},
        "menu": [{"action_id": "read-001", "tool": "read_url",
                  "url": "https://example.invalid/news/one", "title": "已见文章标题",
                  "request_cost": 1, "information_need": "确认本场出场可能性"}],
        "observed_article_candidates": 18,
        "menu_cap": 1, "omitted_visible_candidates": 17,
        "ordering": "semantic_nomination_from_observed_catalog_then_numerical_voi",
        "catalog_coverage": {
            "available_articles": 18, "shown_articles": 12, "omitted_articles": 6,
            "catalog_limit": 12, "scope": "initial_observed_catalog",
            "global_completeness": False, "source_projection_truncated": True},
        "unmodeled_article_candidates": 17,
    }


def _read(output):
    return (json.loads((output / "analysis.json").read_text(encoding="utf-8")),
            (output / "report.md").read_text(encoding="utf-8"))


def test_absent_catalog_and_resume_preserve_legacy_output(tmp_path):
    """Digests were captured from the renderer before this opt-in extension."""
    write_semantic_report(_legacy_run(), tmp_path)
    analysis, markdown = _read(tmp_path)
    assert "candidate_discovery" not in analysis
    assert "resume_provenance" not in analysis
    assert hashlib.sha256(markdown.encode()).hexdigest() == (
        "b2dbcf7c4c9877fcbdb3645d64a7c0a37ce8b7192f19a77ab5fa06fe2cc3d069")
    assert hashlib.sha256((tmp_path / "analysis.json").read_text(encoding="utf-8").encode()).hexdigest() == (
        "263639dc5670bf2a5932575172be8ec740d0b5eb71fa6eaf9abb9f045564df90")


def test_initial_v2_catalog_discloses_nomination_scope_and_omissions(tmp_path):
    run = {**_legacy_run(), "candidate_discovery": _discovery()}
    before = deepcopy(run)
    write_semantic_report(run, tmp_path)
    analysis, markdown = _read(tmp_path)
    assert run == before
    assert analysis["candidate_discovery"] == before["candidate_discovery"]
    assert "已见目录 → 语义提名 → 数学估值" in markdown
    assert "可用文章 18，本次向语义模型展示 12，未展示 6，展示上限 12" in markdown
    assert "上游来源目录是否截断：`True`" in markdown
    assert "未建模文章候选 17" in markdown
    assert "未建模不等于低价值" in markdown
    assert "不证明全目录覆盖，也不构成开放世界停止证明" in markdown
    assert "| read-001 | 确认本场出场可能性 | 是 |" in markdown
    assert "| read-002 | 确认另一个不确定因素 | 否 |" in markdown
    assert "入选菜单不等于已执行" in markdown
    assert "没有保存完整数学决策轮次" in markdown


def test_v1_or_null_discovery_never_claims_observed_catalog_nomination(tmp_path):
    for discovery in (None, {"proposal": {"schema": "semantic-scenario-discovery-v1",
                                        "queries": []}}):
        run = {**_legacy_run(), "candidate_discovery": discovery, "resume_provenance": None}
        write_semantic_report(run, tmp_path)
        analysis, markdown = _read(tmp_path)
        assert analysis["candidate_discovery"] == discovery
        assert analysis["resume_provenance"] is None
        assert "初始候选目录、语义提名与覆盖范围" not in markdown
        assert "从已保存运行恢复" not in markdown


def test_empty_nomination_is_disclosed_without_inventing_valued_candidates(tmp_path):
    discovery = _discovery()
    discovery["proposal"]["article_candidates"] = []
    discovery["menu"] = []
    discovery["unmodeled_article_candidates"] = 18
    write_semantic_report({**_legacy_run(), "candidate_discovery": discovery}, tmp_path)
    analysis, markdown = _read(tmp_path)
    assert "本次没有保存已验证的文章提名" in markdown
    assert "未建模文章候选 18" in markdown
    assert analysis["conditional_voi"]["cycles"] == []


def test_resumed_run_records_cumulative_budget_and_failed_calls(tmp_path):
    provenance = {
        "source_run_path": "var/submission/original/run.json", "source_run_sha256": "a" * 64,
        "retained_semantic_call_indices": [0, 1, 2], "retained_failed_semantic_calls": 2,
        "cumulative_budget_no_refund": True, "source_files_modified": False,
    }
    run = {**_legacy_run(), "resume_provenance": provenance}
    before = deepcopy(run)
    write_semantic_report(run, tmp_path)
    analysis, markdown = _read(tmp_path)
    assert run == before
    assert analysis["resume_provenance"] == provenance
    assert "继续同一累计预算，不是新的独立实验" in markdown
    assert "保留语义调用索引：`[0, 1, 2]`" in markdown
    assert "保留的失败语义调用数：2" in markdown
    assert "累计预算不退还：`True`" in markdown
    assert "原文件是否修改：`False`" in markdown
    assert provenance["source_run_sha256"] in markdown
