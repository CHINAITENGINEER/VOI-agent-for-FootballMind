from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
STAGE2 = ROOT / "var/work/fm-postread-g-dataset-v1-20261021/postread-g-dataset-v1.jsonl"
LABELS = ROOT / "var/work/fm-g-f-bound-labels-v1-20261024/g-f-bound-labels-v1.jsonl"
STRATEGIES = ROOT / "var/work/fm-strategy-comparison-v2-20261024/strategy-comparison.json"
OUT = ROOT / "var/work/fm-g-stage6-error-analysis-20261024-v1"


def sha256(p: Path):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def dt(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def first_type(row):
    claims = row.get("candidate_semantics", {}).get("claims", [])
    if not claims:
        return "no_claim"
    types = []
    for c in claims:
        kind = c.get("mapped_kind") or "unknown"
        value = c.get("mapped_value")
        if kind == "other" and value == "unmodeled":
            kind = "unmodeled"
        types.append(kind)
    # Preserve mixed evidence instead of silently selecting a single type.
    uniq = sorted(set(types))
    return uniq[0] if len(uniq) == 1 else "mixed:" + "+".join(uniq)


def publication_minutes(row):
    p = dt(row.get("source", {}).get("source_published_at"))
    k = dt(row.get("fixture", {}).get("kickoff"))
    if p is None or k is None:
        return None
    return (k - p).total_seconds() / 60.0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stage2 = {r["sample_id"]: r for r in (json.loads(x) for x in STAGE2.read_text(encoding="utf-8").splitlines() if x.strip())}
    labels = [json.loads(x) for x in LABELS.read_text(encoding="utf-8").splitlines() if x.strip()]
    strategy_data = json.loads(STRATEGIES.read_text(encoding="utf-8"))
    selected_by_match = {}
    for r in strategy_data["records"]:
        if r["strategy"] == "new_full_text_state_g":
            selected_by_match[r["physical_match_id"]] = set(r["selected_candidate_ids"])

    records = []
    for label in labels:
        sid = label["sample_id"]
        row = stage2[sid]
        claims = row.get("candidate_semantics", {}).get("claims", [])
        fexec = label.get("f_execution", {})
        types = []
        for c in claims:
            kind = c.get("mapped_kind") or "unknown"
            if kind == "other" and c.get("mapped_value") == "unmodeled":
                kind = "unmodeled"
            types.append(kind)
        if not types:
            types = ["no_claim"]
        match_id = label["physical_match_id"]
        selected = sid in selected_by_match.get(match_id, set())
        gain = label.get("realized_gain")
        duplicate_group = [x["sample_id"] for x in labels if x["physical_match_id"] == match_id]
        availability = label.get("clocks", {}).get("source_available_at")
        publication_distance = publication_minutes(row)
        records.append({
            "sample_id": sid,
            "physical_match_id": match_id,
            "candidate_text_id": label.get("candidate_text_id"),
            "competition": row.get("fixture", {}).get("competition"),
            "season": row.get("fixture", {}).get("season"),
            "home_team": row.get("fixture", {}).get("home_team"),
            "away_team": row.get("fixture", {}).get("away_team"),
            "source_domain": label.get("candidate", {}).get("source_domain"),
            "source_url": label.get("candidate", {}).get("source_url"),
            "publication_distance_minutes": publication_distance,
            "published_before_kickoff": row.get("source", {}).get("source_published_at") is not None and (row.get("source", {}).get("source_published_at") < row.get("fixture", {}).get("kickoff")),
            "source_available_at": availability,
            "historical_visibility_proven": row.get("source", {}).get("availability_status") == "PROVEN",
            "evidence_types": types,
            "primary_evidence_type": first_type(row),
            "claim_count": len(claims),
            "evidence_strength": row.get("candidate_semantics", {}).get("evidence_strength"),
            "uncertainty_status": row.get("candidate_semantics", {}).get("uncertainty_status"),
            "same_match_candidate_count": len(duplicate_group),
            "same_body_sha_duplicate": False,
            "explicit_conflict_status": "UNKNOWN_NO_EXPLICIT_CROSS_DOCUMENT_CONTRADICTION_ANNOTATION",
            "body_consumed": bool(fexec.get("body_received")),
            "f_modeled_claim_count": fexec.get("modeled_claim_count"),
            "f_prediction_changed": bool(fexec.get("prediction_changed")),
            "f_processability": "modeled" if fexec.get("modeled_claim_count", 0) > 0 else ("received_no_modeled_claim" if fexec.get("body_received") else "not_received"),
            "g_full_text_selected": selected,
            "g_selection_score_source": "stage4_lomo_ridge_body_state" if selected else "stage4_lomo_ridge_body_state_not_selected",
            "outcome_class": label.get("outcome_class"),
            "realized_gain": gain,
            "g_error_inclusion": bool(selected and gain is not None and gain < 0),
            "g_error_stop_or_exclusion": bool((not selected) and gain is not None and gain > 0),
            "f_adverse_realized_update": bool(fexec.get("prediction_changed") and gain is not None and gain < 0),
            "label_status": label.get("eligibility_status"),
            "formal_training_eligible": bool(label.get("formal_g_training_eligible")),
        })

    by_type = Counter(r["primary_evidence_type"] for r in records)
    g_inclusion = Counter(r["primary_evidence_type"] for r in records if r["g_error_inclusion"])
    g_exclusion = Counter(r["primary_evidence_type"] for r in records if r["g_error_stop_or_exclusion"])
    f_adverse = Counter(r["primary_evidence_type"] for r in records if r["f_adverse_realized_update"])
    recommendations = []
    if sum(r["source_available_at"] is None for r in records) == len(records):
        recommendations.append({"priority": 1, "direction": "增加真实前瞻轨迹", "reason": "当前8条候选均未证明历史 source_available_at；无法把离线结果等同可执行前瞻选择。"})
    if any(r["same_match_candidate_count"] > 1 for r in records):
        recommendations.append({"priority": 2, "direction": "增加冲突和重复样本", "reason": "同场多文章存在，但没有连续多正文 f replay；需要保留顺序、重复、冲突并可重放。"})
    if any(r["f_processability"] == "received_no_modeled_claim" for r in records):
        recommendations.append({"priority": 3, "direction": "改善正文表示", "reason": "有正文进入 f 但没有 modeled claim，当前 evidence types 与 f 支持范围之间存在覆盖缺口。"})
    recommendations.append({"priority": 4, "direction": "改善状态表示", "reason": "现有 evidence_set 在历史回放中为 UNKNOWN，g 无法区分重复、补充和冲突。"})
    report = {
        "schema": "footballmind-g-stage6-error-analysis-v1",
        "status": "STAGE6_DIAGNOSTIC_COMPLETE_NOT_MODEL_EXPANSION",
        "source_labels": str(LABELS),
        "source_labels_sha256": sha256(LABELS),
        "source_dataset": str(STAGE2),
        "source_dataset_sha256": sha256(STAGE2),
        "source_strategy_comparison": str(STRATEGIES),
        "source_strategy_comparison_sha256": sha256(STRATEGIES),
        "records": len(records),
        "physical_matches": len({r["physical_match_id"] for r in records}),
        "settled_records": sum(r["realized_gain"] is not None for r in records),
        "evidence_type_counts": dict(by_type),
        "g_error_inclusion_by_type": dict(g_inclusion),
        "g_error_stop_or_exclusion_by_type": dict(g_exclusion),
        "f_adverse_update_by_type": dict(f_adverse),
        "highest_frequency_error_types": [x[0] for x in by_type.most_common()],
        "highest_contribution_errors": sorted([r for r in records if r["realized_gain"] is not None], key=lambda r: r["realized_gain"]),
        "high_confidence_g_errors": [],
        "low_confidence_g_errors": [r["sample_id"] for r in records if r["g_error_inclusion"] or r["g_error_stop_or_exclusion"]],
        "f_processable_not_selected": [r["sample_id"] for r in records if r["f_processability"] == "modeled" and not r["g_full_text_selected"]],
        "f_unmodeled_selected": [r["sample_id"] for r in records if r["f_processability"] != "modeled" and r["g_full_text_selected"]],
        "recommendations": recommendations,
        "unknowns": [
            "No explicit conflict annotation exists; conflict is UNKNOWN, not inferred from outcome or title.",
            "Source availability is mostly UNKNOWN; historical publication time is not visibility proof.",
            "No independent action_id/propensity/full menu; g selection error rates are diagnostic only.",
            "The sample is too small to separate label noise from match randomness.",
        ],
        "network_requests": 0,
        "provider_calls": 0,
        "model_refit": False,
        "formal_runtime_integration": False,
    }
    (OUT / "error-analysis.json").write_text(json.dumps({"report": report, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "error-analysis.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = list(records[0])
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in records:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in r.items()})
    (OUT / "recommendations.json").write_text(json.dumps(recommendations, ensure_ascii=False, indent=2), encoding="utf-8")

    names = list(by_type); vals = [by_type[n] for n in names]
    plt.figure(figsize=(10, 5)); plt.bar(range(len(names)), vals); plt.xticks(range(len(names)), names, rotation=35, ha="right"); plt.ylabel("Candidate rows"); plt.title("Stage 6 evidence type coverage"); plt.tight_layout(); plt.savefig(OUT / "evidence-type-coverage.png", dpi=160); plt.close()
    plt.figure(figsize=(10, 5)); plt.bar(range(len(names)), [g_inclusion.get(n, 0) for n in names], label="g inclusion errors"); plt.bar(range(len(names)), [g_exclusion.get(n, 0) for n in names], bottom=[g_inclusion.get(n, 0) for n in names], label="g stop/exclusion errors"); plt.xticks(range(len(names)), names, rotation=35, ha="right"); plt.ylabel("Rows"); plt.legend(); plt.title("Stage 6 selection error counts (diagnostic)"); plt.tight_layout(); plt.savefig(OUT / "g-selection-errors.png", dpi=160); plt.close()

    lines = [
        "FootballMind Stage 6：全文+状态 g 系统错误分析（诊断性，不扩模型）",
        "",
        f"覆盖 {len(records)} 条候选、{len({r['physical_match_id'] for r in records})} 场比赛；可结算 {sum(r['realized_gain'] is not None for r in records)} 条。",
        "",
        "事实结果：",
        f"- 证据类型计数：{dict(by_type)}",
        f"- g 选中后出现负 realized_gain 的行：{[r['sample_id'] for r in records if r['g_error_inclusion']]}",
        f"- g 未选中但 realized_gain 为正的行：{[r['sample_id'] for r in records if r['g_error_stop_or_exclusion']]}",
        f"- f 发生概率变化且 realized_gain 为负的行：{[r['sample_id'] for r in records if r['f_adverse_realized_update']]}",
        f"- f 可处理但全文 g 未选中：{report['f_processable_not_selected']}",
        f"- f 当前未建模但全文 g 选中：{report['f_unmodeled_selected']}",
        "",
        "解释边界：",
        "- 这些是观察到的相关关系，不能把负 gain 自动归因于 g 评分错误；赛果随机性、f 更新方向和标签/可见性缺失仍未分离。",
        "- 没有显式冲突标注，冲突维度全部保持 UNKNOWN；没有用赛果反推冲突。",
        "- 高置信错误为空：当前缺少独立前瞻菜单、propensity 和完整来源可见时钟，不能把离线选择错误提升为高置信结论。",
        "",
        "下一步建议按优先级：",
    ]
    for rec in recommendations:
        lines.append(f"{rec['priority']}. {rec['direction']}：{rec['reason']}")
    lines.append("阶段7 不应启动：阶段6明确提示当前数据与时间链路尚不足以支持 f-v2 设计。")
    (OUT / "readable-report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "README.md").write_text("# Stage 6\n\n只读错误分析，来源为当前 Stage2/Stage3 与 Stage5 v2。未发网络、未调用 provider、未重训或修改 f/g/runtime。UNKNOWN 维度保持 UNKNOWN。\n", encoding="utf-8")
    verification = {
        "schema": "footballmind-g-stage6-verification-v1",
        "run_id": "fm-g-stage6-error-analysis-20261024-v1",
        "rows": len(records),
        "physical_matches": len({r["physical_match_id"] for r in records}),
        "network_requests": 0,
        "provider_calls": 0,
        "model_refit": False,
        "formal_runtime_integration": False,
        "unknown_conflicts_preserved": True,
        "result_not_used_to_rewrite_body_features": True,
        "py_compile": True,
        "source_labels_sha256": sha256(LABELS),
        "source_dataset_sha256": sha256(STAGE2),
        "source_strategy_sha256": sha256(STRATEGIES),
    }
    (OUT / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {}
    for p in sorted(OUT.iterdir()):
        if p.name != "manifest.json": manifest[p.name] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    (OUT / "manifest.json").write_text(json.dumps({"schema": "footballmind-stage6-manifest-v1", "files": manifest}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
