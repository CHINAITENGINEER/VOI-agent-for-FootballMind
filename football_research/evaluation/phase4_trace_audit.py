"""Offline audit for the Phase 4 historical acquisition trace.

This evaluator deliberately does *not* estimate VoI, ``g``, prediction loss,
or causal effect.  The Phase 4 pilot is a historical reconstruction with a
manual/predeclared menu, no policy propensities, and no model configured.  The
report therefore measures the observable information-selection mechanics:
candidate exposure, action/cost accounting, response yield, failures, and the
gate that stopped expansion.  It is useful for the AIC demonstration because
it makes the missing ingredients for a numeric VoI claim explicit instead of
turning collection success into an invented information gain.

All inputs are read-only.  The CLI writes only the requested output files.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any, Mapping


VERSION = "phase4-trace-voi-audit-v1"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _status(row: Mapping[str, Any]) -> str:
    return str(row.get("status") or row.get("result_status") or "unknown").lower()


def _candidate_count(discovery: Mapping[str, Any]) -> int:
    value = discovery.get("visible_candidates")
    return len(value) if isinstance(value, list) else 0


def _read_candidate_ids(attempts: list[Mapping[str, Any]]) -> set[str]:
    return {
        str(row["candidate_id"])
        for row in attempts
        if row.get("candidate_id") not in (None, "")
    }


def _latest_document_rows(documents: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Use the gate's convention: local-reparse-v2 supersedes its base row."""
    latest: dict[str, Mapping[str, Any]] = {}
    for row in documents:
        doc_id = str(row.get("document_id") or "")
        base = doc_id.split(":local-reparse-v2", 1)[0]
        latest[base] = row
    return list(latest.values())


def _fixture_report(
    fixture_id: str,
    discoveries: list[Mapping[str, Any]],
    attempts: list[Mapping[str, Any]],
    documents: list[Mapping[str, Any]],
    gate_fixture: Mapping[str, Any] | None,
) -> dict[str, Any]:
    statuses = Counter(_status(row) for row in attempts)
    discovery_statuses = Counter(_status(row) for row in discoveries)
    candidate_count = sum(_candidate_count(row) for row in discoveries)
    read_attempts = [row for row in attempts if row.get("action_id") == "read_url"]
    read_ids = _read_candidate_ids(read_attempts)
    nonempty_docs = [row for row in documents if str(row.get("body") or "")]
    # Match the original (network) read to a document through its raw path.
    # The local-reparse rows deliberately share that path but are not new
    # requests, so they are excluded from this original-yield measure.
    docs_by_raw = defaultdict(list)
    for doc in documents:
        if ":local-reparse-v2" not in str(doc.get("document_id") or ""):
            docs_by_raw[str(doc.get("raw_path") or "")].append(doc)
    original_read_nonempty = sum(
        any(str(doc.get("body") or "") for doc in docs_by_raw.get(str(row.get("raw_path") or ""), []))
        for row in read_attempts
    )
    latest_docs = _latest_document_rows(documents)
    latest_nonempty = [row for row in latest_docs if str(row.get("body") or "")]
    latencies = [float(row["latency_ms"]) for row in attempts if isinstance(row.get("latency_ms"), (int, float))]
    requested = sum(int(row.get("request_count") or 0) for row in attempts)
    returned = sum(1 for row in attempts if _status(row) == "returned")
    nonempty_success = sum(1 for row in read_attempts if _status(row) == "returned" and any(
        str(doc.get("attempt_id")) == str(row.get("attempt_id")) and str(doc.get("body") or "")
        for doc in documents
    ))
    # Some historical exports omit attempt_id on document rows.  The gate is
    # authoritative for the reconciled latest count, while this local metric
    # stays conservative and is labelled as an observable lower bound.
    if gate_fixture is not None:
        latest_nonempty_count = int(gate_fixture.get("distinct_nonempty_bodies") or 0)
        latest_doc_count = int(gate_fixture.get("latest_document_count") or 0)
    else:
        latest_nonempty_count = len(latest_nonempty)
        latest_doc_count = len(latest_docs)
    return {
        "physical_match_id": fixture_id,
        "discovery_calls": len(discoveries),
        "discovery_status_counts": dict(discovery_statuses),
        "candidate_exposures": candidate_count,
        "candidate_ids_read": len(read_ids),
        "candidate_to_read_rate": (len(read_ids) / candidate_count if candidate_count else None),
        "attempts": len(attempts),
        "charged_requests": requested,
        "attempt_status_counts": dict(statuses),
        "read_attempts": len(read_attempts),
        "returned_attempts": returned,
        "attempt_return_rate": (returned / len(attempts) if attempts else None),
        "nonempty_document_rows_raw": len(nonempty_docs),
        "original_read_nonempty_bodies": original_read_nonempty,
        "original_nonempty_body_yield_per_read": (
            original_read_nonempty / len(read_attempts) if read_attempts else None
        ),
        "latest_document_rows": latest_doc_count,
        "latest_nonempty_bodies": latest_nonempty_count,
        "nonempty_body_yield_per_read_lower_bound": (
            nonempty_success / len(read_attempts) if read_attempts else None
        ),
        "latency_ms": {
            "median": median(latencies) if latencies else None,
            "mean": (sum(latencies) / len(latencies)) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "selection_basis": sorted({str(row.get("selection_basis")) for row in attempts}),
        "selection_probability_values": sorted({row.get("selection_probability") for row in attempts}, key=str),
        "historical_reconstruction_only": all(
            row.get("information_mode") == "historical_reconstruction" for row in attempts
        ),
    }


def build_report(panel: str | Path) -> dict[str, Any]:
    root = Path(panel).resolve()
    fixtures = _read_jsonl(root / "fixtures.jsonl")
    discoveries = _read_jsonl(root / "discovery.jsonl")
    attempts = _read_jsonl(root / "attempts.jsonl")
    documents = _read_jsonl(root / "documents.jsonl")
    plan = _read_json(root / "frozen-plan.json")
    gate = _read_json(root / "reports" / "pilot-gate-reconciled.json")
    validation = _read_json(root / "reports" / "postpilot-validation.json")
    journal_ref = _read_json(root / "journal-reference.json")
    run_id = journal_ref.get("journal_run_id") or journal_ref.get("run_id")
    run_path = root / "journal" / str(run_id) / "run.json"
    run = _read_json(run_path) if run_path.exists() else {}

    fixture_ids = [str(row.get("physical_match_id")) for row in fixtures if row.get("physical_match_id")]
    fixture_ids = list(dict.fromkeys(fixture_ids))
    gate_rows = {str(row.get("physical_match_id")): row for row in gate.get("fixtures", [])}
    per_fixture = []
    for fixture_id in fixture_ids:
        per_fixture.append(_fixture_report(
            fixture_id,
            [row for row in discoveries if str(row.get("physical_match_id")) == fixture_id],
            [row for row in attempts if str(row.get("physical_match_id")) == fixture_id],
            [row for row in documents if str(row.get("physical_match_id")) == fixture_id],
            gate_rows.get(fixture_id),
        ))

    event_types = Counter(str(event.get("type")) for event in run.get("events", []))
    selection_basis = Counter(str(row.get("selection_basis")) for row in attempts)
    policy_trace_proven = any(bool(row.get("original_policy_trace")) for row in discoveries)
    propensities_present = any(row.get("selection_probability") is not None for row in attempts + discoveries)
    model_statuses = sorted({str(row.get("model_status")) for row in fixtures + documents})
    statuses = Counter(_status(row) for row in attempts)
    charged = sum(int(row.get("request_count") or 0) for row in attempts)
    read_attempt_count = sum(1 for row in attempts if row.get("action_id") == "read_url")
    read_candidate_count = len(_read_candidate_ids(attempts))
    original_nonempty = sum(
        int(row.get("original_read_nonempty_bodies") or 0) for row in per_fixture
    )
    latest_nonempty = int(gate.get("latest_nonempty_documents") or 0)
    max_fixture = max((int(row.get("read_budget_per_match") or 0) for row in [plan]), default=0)
    pilot_cap = int(plan.get("pilot_request_budget") or plan.get("pilot_cap") or 0)
    global_cap = int(plan.get("request_budget") or 0)
    # The frozen plan in this release names request_budget and a pilot list;
    # the actual pilot cap is 32 in the coordinator receipt when omitted here.
    if not pilot_cap:
        pilot_cap = 32

    auditability = {
        "numeric_voi_estimable": False,
        "numeric_voi_blockers": [
            "model_status is NOT_CONFIGURED; no baseline prediction before/after",
            "selection_probability/propensity is absent, so IPS/DR policy evaluation is invalid",
            "available_at is null and all rows are historical_reconstruction",
            "no evaluator-side realized gain/loss was attached to the trace",
            "selection_basis=manual_predeclared; original autonomous policy trace is unproven",
        ],
        "safe_proxy_metrics": [
            "candidate_exposures and candidate_to_read_rate",
            "charged_requests and latency distribution",
            "returned/failed/timeout/http_error/parse_error rates",
            "nonempty body yield and cost per nonempty body",
            "per-fixture stopping and expansion gate",
            "evidence lineage and historical reconstruction flags",
        ],
        "next_online_fields_required": [
            "state_hash_before and state_hash_after",
            "candidate_expected_gain or calibrated VoI estimate",
            "decision_cost and request budget before/after",
            "selection_probability under the frozen policy",
            "prediction_before and prediction_after",
            "independent realized loss/gain sidecar",
            "available_at <= prediction_cutoff and source version ID",
        ],
    }
    report = {
        "schema": VERSION,
        "panel": str(root),
        "input_sha256": {
            name: _sha(root / name)
            for name in ("fixtures.jsonl", "discovery.jsonl", "attempts.jsonl", "documents.jsonl", "frozen-plan.json")
            if (root / name).exists()
        },
        "scope": {
            "information_mode": "historical_reconstruction",
            "network_requests_performed_by_audit": 0,
            "original_inputs_modified": False,
            "purpose": "offline mechanics and evidence-lineage audit; not a VoI or outcome evaluation",
        },
        "counts": {
            "fixtures": len(fixture_ids),
            "pilot_fixtures": len(plan.get("pilot_order", [])),
            "discovery_rows": len(discoveries),
            "candidate_exposures": sum(_candidate_count(row) for row in discoveries),
            "attempts": len(attempts),
            "charged_requests": charged,
            "documents_raw": len(documents),
            "documents_latest_nonempty": int(gate.get("latest_nonempty_documents") or 0),
            "g_training_eligible": int(gate.get("g_training_eligible_latest_rows") or 0),
        },
        "observable_efficiency": {
            "read_attempts": read_attempt_count,
            "distinct_candidates_read": read_candidate_count,
            "candidate_to_read_rate": (
                read_candidate_count / sum(_candidate_count(row) for row in discoveries)
                if sum(_candidate_count(row) for row in discoveries) else None
            ),
            "original_nonempty_bodies": original_nonempty,
            "latest_nonempty_bodies": latest_nonempty,
            "charged_requests_per_original_nonempty_body": (
                charged / original_nonempty if original_nonempty else None
            ),
            "charged_requests_per_latest_nonempty_body": (
                charged / latest_nonempty if latest_nonempty else None
            ),
            "interpretation": (
                "mechanics-only proxy; local reparses added no HTTP and are not treated as new acquisition value"
            ),
        },
        "budget": {
            "per_fixture_request_cap": max_fixture,
            "pilot_request_cap": pilot_cap,
            "global_request_cap": global_cap,
            "pilot_utilization": charged / pilot_cap if pilot_cap else None,
            "global_utilization": charged / global_cap if global_cap else None,
        },
        "status_counts": dict(statuses),
        "journal_event_counts": dict(event_types),
        "selection_regime": {
            "basis_counts": dict(selection_basis),
            "propensity_present": propensities_present,
            "original_policy_trace_proven": policy_trace_proven,
            "model_statuses": model_statuses,
            "all_manual_or_predeclared": set(selection_basis).issubset({"manual_predeclared", "None"}),
        },
        "per_fixture": per_fixture,
        "gate": {
            "passed": bool(gate.get("passed")),
            "expansion_authorized": bool(gate.get("expansion_authorized")),
            "rule": gate.get("rule"),
            "reason": gate.get("reason"),
        },
        "auditability": auditability,
        "validation": {
            "postpilot_validation_passed": bool(validation.get("passed")),
            "checks": validation.get("checks", {}),
        },
    }
    return report


def markdown(report: Mapping[str, Any]) -> str:
    counts = report["counts"]
    budget = report["budget"]
    gate = report["gate"]
    efficiency = report["observable_efficiency"]
    cost_original = efficiency["charged_requests_per_original_nonempty_body"]
    cost_original_text = f"{cost_original:.1f}" if cost_original is not None else "—"
    lines = [
        "# FootballMind Phase 4 信息选择轨迹审计",
        "",
        "本报告由冻结的 Phase 4 pilot 产物离线生成；审计本身发起 0 次 HTTP，且不修改任何原始采集文件。",
        "它衡量候选暴露、动作成本、响应产出、失败类型和停止门控，不把历史重建数据解释为数值 VoI 或模型收益。",
        "",
        "## 运行概况",
        "",
        f"- 冻结比赛数：{counts['fixtures']}；先导比赛数：{counts['pilot_fixtures']}；发现记录：{counts['discovery_rows']}；候选暴露：{counts['candidate_exposures']}",
        f"- 尝试数/计费请求：{counts['attempts']} / {counts['charged_requests']}",
        f"- 原始文档行：{counts['documents_raw']}；最新非空正文：{counts['documents_latest_nonempty']}",
        f"- 可观测正文产出：原始读取得到 {efficiency['original_nonempty_bodies']} 篇，最新版本 {efficiency['latest_nonempty_bodies']} 篇；"
        f"每篇原始正文约 {cost_original_text} 次请求",
        f"- g 可训练行：{counts['g_training_eligible']}；模型收益评估：未配置",
        f"- 预算使用：pilot {budget['pilot_utilization']:.3f}，全局 {budget['global_utilization']:.3f}",
        f"- 扩展门控：{'通过' if gate['passed'] else '未通过'}；允许扩展：{'是' if gate['expansion_authorized'] else '否'}",
        "",
        "## 每场轨迹代理指标",
        "",
        "| 比赛 | 发现/候选 | 尝试/请求 | 返回率 | 最新非空正文 | 读候选率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["per_fixture"]:
        rate = row["attempt_return_rate"]
        read_rate = row["candidate_to_read_rate"]
        lines.append(
            f"| {row['physical_match_id']} | {row['discovery_calls']}/{row['candidate_exposures']} "
            f"| {row['attempts']}/{row['charged_requests']} | "
            f"{rate:.3f} | {row['latest_nonempty_bodies']} | "
            f"{read_rate:.3f} |" if rate is not None and read_rate is not None else
            f"| {row['physical_match_id']} | {row['discovery_calls']}/{row['candidate_exposures']} "
            f"| {row['attempts']}/{row['charged_requests']} | — | {row['latest_nonempty_bodies']} | — |"
        )
    lines += [
        "",
        "## AIC 展示可用结论",
        "",
        "1. 系统已经留下了可审计的‘发现菜单 → 选择尝试 → 返回/失败 → 正文版本 → 停止门控’链路。",
        "2. 当前 pilot 的选择是 `manual_predeclared`，没有原始 Agent policy trace、selection probability 或模型预测更新；因此不能宣称 Agent 已经学会了数值 VoI。",
        "3. 最稳妥的演示是展示成本约束和停止规则：三个 fixture 没有可用正文时继续搜索的边际收益无法由当前历史证据证明，门控因此拒绝扩展。",
        "4. 下一次在线运行必须同时记录 state hash、候选 VoI、选择概率、prediction before/after 和独立 realized gain，才能从机制审计升级为数值 VoI 评估。",
        "",
        "## 明确限制",
        "",
        *[f"- {item}" for item in report["auditability"]["numeric_voi_blockers"]],
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output")
    args = parser.parse_args()
    report = build_report(args.panel)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_output:
        path = Path(args.markdown_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"schema": VERSION, "output": str(output), "http_requests": 0,
                      "numeric_voi_estimable": report["auditability"]["numeric_voi_estimable"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
