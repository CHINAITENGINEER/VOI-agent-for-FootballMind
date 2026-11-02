"""Render a reviewer-facing report from autonomous-loop replay artifacts.

The renderer selects fields explicitly instead of copying replay JSON wholesale.
This keeps evaluator-only realizations and raw tool returns out of the demo
surface while preserving the reasoning trace needed for an AIC review.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any


VERSION = "autonomous-loop-report-v1"


def _json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("replay artifact must contain a mapping")
    return dict(value)


def _delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    keys = set(before) | set(after)
    return {key: after[key] for key in sorted(keys) if before.get(key) != after.get(key) and key in after}


def build_report(loop_replay: Mapping[str, Any], policy_replay: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if not isinstance(loop_replay, Mapping):
        raise ValueError("loop_replay must be a mapping")
    projection = loop_replay.get("projection", {})
    loop = loop_replay.get("loop", {})
    if not isinstance(projection, Mapping) or not isinstance(loop, Mapping):
        raise ValueError("loop replay is missing projection or loop")
    decisions = loop.get("decisions", ())
    if not isinstance(decisions, list):
        raise ValueError("loop decisions must be a list")
    steps = []
    for index, item in enumerate(decisions, 1):
        if not isinstance(item, Mapping):
            raise ValueError("each loop decision must be a mapping")
        context = item.get("context", {})
        plan = item.get("plan", {})
        decision = item.get("decision", {})
        board = plan.get("board", {}) if isinstance(plan, Mapping) else {}
        eligible = plan.get("eligible_candidate_ids", []) if isinstance(plan, Mapping) else []
        assessments = decision.get("assessments", []) if isinstance(decision, Mapping) else []
        selected = decision.get("candidate_id") if isinstance(decision, Mapping) else None
        chosen_assessment = next((row for row in assessments if row.get("candidate_id") == selected), None)
        steps.append({
            "step": index,
            "known_context_keys": sorted(context.get("values", {}).keys()) if isinstance(context, Mapping) else [],
            "eligible_candidates": list(eligible),
            "selected_candidate": selected,
            "action": decision.get("action") if isinstance(decision, Mapping) else None,
            "selection_basis": decision.get("selection_basis") if isinstance(decision, Mapping) else None,
            "selected_expected_value": None if chosen_assessment is None else chosen_assessment.get("expected_value"),
            "selected_cost": None if chosen_assessment is None else chosen_assessment.get("cost"),
            "value_status": None if chosen_assessment is None else chosen_assessment.get("value_status"),
            "stop_reason": None if decision.get("stop_certificate") is None else decision["stop_certificate"].get("reason"),
            "hidden_result_included": False,
        })
    final_context = loop.get("final_context", {})
    if not isinstance(final_context, Mapping):
        raise ValueError("final_context must be a mapping")
    for index, step in enumerate(steps):
        before = decisions[index].get("context", {}).get("values", {})
        after = decisions[index + 1].get("context", {}).get("values", {}) if index + 1 < len(decisions) else final_context
        step["context_delta_after_step"] = _delta(before, after)
    comparison = []
    if isinstance(policy_replay, Mapping):
        for row in policy_replay.get("report", {}).get("scenarios", []):
            if not isinstance(row, Mapping):
                continue
            scenario = row.get("scenario", {})
            policies = row.get("policies", {})
            if scenario.get("scenario_id") not in {"qualification-high-value", "weather-low-value"}:
                continue
            for policy in ("fixed_all", "first_query", "voi_gate"):
                item = policies.get(policy, {})
                comparison.append({
                    "scenario_id": scenario.get("scenario_id"), "policy": policy,
                    "query_count": item.get("query_count"), "acquisition_cost": item.get("acquisition_cost"),
                    "final_loss": item.get("final_loss"), "net_utility": item.get("net_utility"),
                })
    evidence_refs = projection.get("accepted_evidence_refs", [])
    return {
        "schema_version": VERSION,
        "title": "FootballMind：开放信息环境下的自主信息选择回放",
        "source_scope": loop_replay.get("source_scope"),
        "initial_context": projection.get("next_context", {}).get("values", {}),
        "steps": steps,
        "final_context": dict(final_context),
        "transition_events": list(loop.get("transition_events", [])),
        "evidence_refs": list(evidence_refs),
        "policy_comparison": comparison,
        "hidden_result_included": False,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    if report.get("schema_version") != VERSION:
        raise ValueError("unexpected report schema")
    lines = [
        f"# {report.get('title', 'FootballMind autonomous information report')}", "",
        "这是一条可回放的 Agent 信息选择轨迹。展示面只包含截止点前的公开上下文、候选路线、选择依据、成本和证据引用。",
        "",
        "## 初始公开上下文", "",
    ]
    for key, value in sorted(report.get("initial_context", {}).items()):
        lines.append(f"- `{key}` = `{value}`")
    lines.extend(["", "## 查询选择轨迹", "", "| 步骤 | 已知上下文键 | 候选 | Agent 选择 | 选择依据 | 预期价值 | 成本 |", "|---:|---|---|---|---|---:|---:|"])
    for step in report.get("steps", []):
        lines.append(
            f"| {step['step']} | {', '.join(step['known_context_keys']) or '（空）'} | "
            f"{', '.join(step['eligible_candidates']) or '—'} | {step['selected_candidate'] or 'STOP'} | "
            f"{step['selection_basis'] or step['stop_reason'] or '—'} | "
            f"{step['selected_expected_value'] if step['selected_expected_value'] is not None else '—'} | "
            f"{step['selected_cost'] if step['selected_cost'] is not None else '—'} |"
        )
        delta = step.get("context_delta_after_step", {})
        if delta:
            lines.append(f"|  | 更新 | `{json.dumps(delta, ensure_ascii=False, sort_keys=True)}` |  |  |  |  |")
    lines.extend(["", "## 最终公开状态", ""])
    for key, value in sorted(report.get("final_context", {}).items()):
        lines.append(f"- `{key}` = `{value}`")
    lines.extend(["", "## 固定特征与 VoI 对照", "", "| 场景 | 策略 | 查询数 | 获取成本 | 最终损失 | 净效用 |", "|---|---|---:|---:|---:|---:|"])
    for row in report.get("policy_comparison", []):
        lines.append(f"| {row['scenario_id']} | {row['policy']} | {row['query_count']} | {row['acquisition_cost']} | {row['final_loss']} | {row['net_utility']} |")
    lines.extend(["", "## 证据引用", "", *[f"- `{ref}`" for ref in report.get("evidence_refs", [])]])
    lines.extend(["", "## 解释边界", "", "这份回放验证信息选择、截止点和重规划机制；合成查询返回不代表真实比赛结论，真实结果标签留在评估层。"])
    return "\n".join(lines) + "\n"


__all__ = ["VERSION", "build_report", "render_markdown"]
