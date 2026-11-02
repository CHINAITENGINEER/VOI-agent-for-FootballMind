from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "var/work/fm-g-f-bound-labels-v1-20261024/g-f-bound-labels-v1.jsonl"
PREDICTIONS = ROOT / "var/work/fm-g-stage4-diagnostic-20261024-v1/predictions.jsonl"
OUT = ROOT / "var/work/fm-strategy-comparison-v2-20261024"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def log_loss(row):
    if row.get("outcome_index") is None or row.get("p_after") is None:
        return None
    return -math.log(float(row["p_after"][int(row["outcome_index"])]))


def baseline_loss(row):
    if row.get("outcome_index") is None or row.get("p_before") is None:
        return None
    return -math.log(float(row["p_before"][int(row["outcome_index"])]))


def body_chars(row):
    path = row.get("candidate", {}).get("body_path")
    if not path:
        return None
    p = Path(path)
    try:
        return len(p.read_text(encoding="utf-8"))
    except (OSError, UnicodeError):
        return None


def selected_record(match_rows, strategy, pred_by_sample):
    if not match_rows:
        return None, "no_candidate"
    if strategy == "fixed_order":
        return match_rows[0], "first_archived_order"
    if strategy == "random":
        rng = random.Random(20261024 + sum(ord(c) for c in match_rows[0]["physical_match_id"]))
        return rng.choice(match_rows), "seeded_random_one"
    if strategy == "new_full_text_state_g":
        # Stage 4 predictions are leave-one-physical-match-out predictions.
        return max(match_rows, key=lambda r: float(pred_by_sample.get(r["sample_id"], {}).get("ridge_body_state", 0.0))), "max_lomo_score"
    if strategy == "oracle_analysis_only":
        eligible = [r for r in match_rows if r.get("realized_gain") is not None]
        if not eligible:
            return None, "unsettled"
        return max(eligible, key=lambda r: float(r["realized_gain"])), "max_realized_gain_analysis_only"
    return None, "unsupported"


def strategy_cost(rows, selected_rows, reads_all=False):
    # Archived packages do not contain request-level or provider billing costs.
    # We therefore report only a transparent lower-bound proxy and keep total_cost UNKNOWN.
    chars = 0
    known_chars = True
    for row in (rows if reads_all else selected_rows):
        n = body_chars(row)
        if n is None:
            known_chars = False
        else:
            chars += n
    g_read_lower_bound = chars / 1000.0 * 0.02 if known_chars else None
    return {
        "source_discovery": None,
        "body_acquisition": None,
        "g_read": g_read_lower_bound,
        "f_update": None,
        "other_processing": None,
        "known_lower_bound_g_read_only": g_read_lower_bound,
        "total_cost": None,
        "cost_status": "UNKNOWN_ARCHIVED_COST_COMPONENTS",
        "cost_basis": "No request/provider billing or per-step time/cost ledger exists in current Stage2/Stage3 rows; no cost invented.",
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(LABELS)
    pred_rows = read_jsonl(PREDICTIONS)
    pred_by_sample = {r["sample_id"]: r for r in pred_rows}
    groups = defaultdict(list)
    for row in rows:
        groups[row["physical_match_id"]].append(row)
    for values in groups.values():
        values.sort(key=lambda r: (r.get("sample_id", ""), r.get("candidate_text_id", "")))

    strategies = [
        "no_new_information",
        "all_candidates",
        "fixed_order",
        "random",
        "old_title_local_g",
        "new_full_text_state_g",
        "oracle_analysis_only",
    ]
    records = []
    summary = {}
    for strategy in strategies:
        strat_rows = []
        for match_id, match_rows in groups.items():
            settled = all(r.get("outcome_index") is not None for r in match_rows)
            if strategy == "no_new_information":
                selected = None
                status = "computed" if settled else "UNKNOWN_UNSETTLED"
                p_after = match_rows[0].get("p_before") if settled else None
                loss = baseline_loss(match_rows[0]) if settled else None
                gain = 0.0 if settled else None
                reason = "no_candidate_selected"
            elif strategy == "all_candidates":
                selected = match_rows if len(match_rows) == 1 else None
                if len(match_rows) != 1:
                    status = "UNKNOWN_NO_SEQUENTIAL_MULTI_DOCUMENT_REPLAY"
                    p_after = loss = gain = None
                    reason = "multiple_candidates_but_only_single_document_f_replay_exists"
                elif not settled:
                    status = "UNKNOWN_UNSETTLED"
                    p_after = loss = gain = None
                    reason = "independent_result_unavailable"
                else:
                    status = "computed_single_candidate_pool"
                    p_after = match_rows[0].get("p_after")
                    loss = log_loss(match_rows[0])
                    gain = match_rows[0].get("realized_gain")
                    reason = "one_candidate_equals_all_pool"
            elif strategy == "old_title_local_g":
                status = "UNAVAILABLE"
                selected = None
                p_after = loss = gain = None
                reason = "current_dataset_does_not_retain_title_or_local_proxy_score"
            else:
                selected, reason = selected_record(match_rows, strategy, pred_by_sample)
                if selected is None:
                    status = "UNKNOWN_UNSETTLED" if reason == "unsettled" else "UNKNOWN"
                    p_after = loss = gain = None
                elif selected.get("outcome_index") is None:
                    status = "UNKNOWN_UNSETTLED"
                    p_after = loss = gain = None
                else:
                    status = "computed_single_document_replay"
                    p_after = selected.get("p_after")
                    loss = log_loss(selected)
                    gain = selected.get("realized_gain")
            selected_list = selected if isinstance(selected, list) else ([] if selected is None else [selected])
            if strategy == "no_new_information":
                read_rows = []
            elif strategy == "all_candidates":
                read_rows = match_rows if len(match_rows) > 1 else selected_list
            elif strategy == "old_title_local_g":
                read_rows = []
            else:
                read_rows = selected_list
            cost = strategy_cost(match_rows, read_rows, reads_all=(strategy == "all_candidates"))
            records.append({
                "physical_match_id": match_id,
                "strategy": strategy,
                "candidate_pool_size": len(match_rows),
                "candidate_ids": [r.get("candidate_text_id") for r in match_rows],
                "selected_candidate_ids": [r.get("candidate_text_id") for r in selected_list],
                "unselected_candidate_ids": [r.get("candidate_text_id") for r in match_rows if r not in selected_list],
                "selection_reason": reason,
                "status": status,
                "p_before": match_rows[0].get("p_before"),
                "p_after": p_after,
                "final_log_loss": loss,
                "always_stop_log_loss": baseline_loss(match_rows[0]) if settled else None,
                "realized_gain": gain,
                "settled": settled,
                "cost": cost,
                "f_update_count": len(selected_list),
                "error_exclusion": any(r.get("f_execution", {}).get("modeled_claim_count", 0) > 0 for r in match_rows) and not selected_list,
                "repeated_evidence": len({r.get("candidate", {}).get("body_sha256") for r in match_rows}) < len(match_rows),
                "source_or_team_mixing_flag": False,
                "oracle_warning": strategy == "oracle_analysis_only",
            })
        eval_rows = [r for r in records if r["strategy"] == strategy and r["final_log_loss"] is not None]
        gains = [r["realized_gain"] for r in records if r["strategy"] == strategy and r["realized_gain"] is not None]
        summary[strategy] = {
            "matches": len(groups),
            "evaluable_matches": len(eval_rows),
            "coverage": len(eval_rows) / len(groups) if groups else 0.0,
            "mean_final_log_loss": sum(r["final_log_loss"] for r in eval_rows) / len(eval_rows) if eval_rows else None,
            "mean_always_stop_log_loss": sum(r["always_stop_log_loss"] for r in eval_rows) / len(eval_rows) if eval_rows else None,
            "mean_realized_gain": sum(gains) / len(gains) if gains else None,
            "selected_count": sum(len(r["selected_candidate_ids"]) for r in records if r["strategy"] == strategy),
            "stop_count": sum(1 for r in records if r["strategy"] == strategy and not r["selected_candidate_ids"]),
            "unknown_count": sum(1 for r in records if r["strategy"] == strategy and r["status"].startswith("UNKNOWN")),
            "unavailable_count": sum(1 for r in records if r["strategy"] == strategy and r["status"] == "UNAVAILABLE"),
            "total_cost": None,
            "cost_status": "UNKNOWN_ARCHIVED_COST_COMPONENTS",
        }

    data = {
        "schema": "footballmind-strategy-comparison-v2",
        "run_id": "fm-strategy-comparison-v2-20261024",
        "diagnostic_only": True,
        "network_requests": 0,
        "external_calls": 0,
        "formal_runtime_integration": False,
        "fixed_f": {"model_sha256": rows[0]["f_model_sha256"], "model_version": rows[0]["f_model_version"]},
        "source_labels": str(LABELS),
        "source_labels_sha256": sha256(LABELS),
        "source_predictions": str(PREDICTIONS),
        "source_predictions_sha256": sha256(PREDICTIONS),
        "dataset": {"rows": len(rows), "physical_matches": len(groups), "settled_rows": sum(r.get("outcome_index") is not None for r in rows)},
        "strategies": summary,
        "records": records,
        "cost_disclaimer": "Current archived Stage2/Stage3 material has no request/provider billing or per-step cost ledger. Total cost is UNKNOWN; only a g-read lower-bound proxy is shown per record.",
        "limitations": [
            "Most rows have source_available_at UNKNOWN and no prospective action_id/propensity/complete menu.",
            "All-candidate final p_after is UNKNOWN for multi-candidate matches because sequential multi-document f replay is not archived.",
            "Old title/local g is unavailable because the current Stage2 schema deliberately retains no title field.",
            "New full-text g uses Stage4 leave-one-physical-match-out ridge_body_state scores as a diagnostic selector; it is not formal runtime and not production-ready.",
            "The current dataset has six physical matches, five settled matches and eight candidate rows; this is not evidence of generalization.",
        ],
        "oracle_is_analysis_only": True,
    }
    (OUT / "strategy-comparison.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["physical_match_id", "strategy", "candidate_pool_size", "selected_candidate_ids", "unselected_candidate_ids", "status", "selection_reason", "final_log_loss", "always_stop_log_loss", "realized_gain", "settled", "f_update_count", "cost_status", "known_lower_bound_g_read_only", "error_exclusion", "repeated_evidence", "oracle_warning"]
    with (OUT / "strategy-comparison.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            row_out = {}
            for k in fields:
                value = r.get(k)
                if k == "cost_status":
                    value = r.get("cost", {}).get("cost_status")
                elif k == "known_lower_bound_g_read_only":
                    value = r.get("cost", {}).get("known_lower_bound_g_read_only")
                row_out[k] = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
            w.writerow(row_out)
    with (OUT / "strategy-summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields2 = ["strategy", "matches", "evaluable_matches", "coverage", "mean_final_log_loss", "mean_always_stop_log_loss", "mean_realized_gain", "selected_count", "stop_count", "unknown_count", "unavailable_count", "total_cost", "cost_status"]
        w = csv.DictWriter(f, fieldnames=fields2)
        w.writeheader()
        for s, vals in summary.items():
            w.writerow({"strategy": s, **{k: vals.get(k) for k in fields2 if k != "strategy"}})

    # Per-match errors for all strategies that have a realized gain.
    errors = []
    for r in records:
        if r["realized_gain"] is not None:
            errors.append({"physical_match_id": r["physical_match_id"], "strategy": r["strategy"], "selected": r["selected_candidate_ids"], "realized_gain": r["realized_gain"], "final_log_loss": r["final_log_loss"], "status": r["status"]})
    (OUT / "per-match-error-analysis.json").write_text(json.dumps({"schema": "footballmind-strategy-errors-v2", "records": errors}, ensure_ascii=False, indent=2), encoding="utf-8")

    # Visuals: only display evaluable values and explicitly annotate unknown cost/coverage.
    names = list(summary)
    losses = [summary[n]["mean_final_log_loss"] if summary[n]["mean_final_log_loss"] is not None else float("nan") for n in names]
    gains = [summary[n]["mean_realized_gain"] if summary[n]["mean_realized_gain"] is not None else float("nan") for n in names]
    plt.figure(figsize=(11, 5)); plt.bar(range(len(names)), losses); plt.xticks(range(len(names)), names, rotation=35, ha="right"); plt.ylabel("Mean final log loss (evaluable only)"); plt.title("Stage 5 v2 policy comparison; UNKNOWN rows excluded"); plt.tight_layout(); plt.savefig(OUT / "policy-curves.png", dpi=160); plt.close()
    plt.figure(figsize=(11, 5)); plt.bar(range(len(names)), gains); plt.axhline(0, color="black", linewidth=0.8); plt.xticks(range(len(names)), names, rotation=35, ha="right"); plt.ylabel("Mean realized gain (evaluable only)"); plt.title("Stage 5 v2 realized gain; unsettled/unknown rows excluded"); plt.tight_layout(); plt.savefig(OUT / "budget-performance.png", dpi=160); plt.close()
    cov = [summary[n]["coverage"] for n in names]
    plt.figure(figsize=(11, 5)); plt.bar(range(len(names)), cov); plt.ylim(0, 1.05); plt.xticks(range(len(names)), names, rotation=35, ha="right"); plt.ylabel("Evaluable-match coverage"); plt.title("Stage 5 v2 evaluation coverage"); plt.tight_layout(); plt.savefig(OUT / "stopping-analysis.png", dpi=160); plt.close()

    lines = [
        "FootballMind Stage 5 v2：当前 Stage 2/3 候选池上的策略比较（诊断性历史重建）",
        "",
        f"样本：{len(rows)} 条候选，{len(groups)} 场 physical match；其中可结算候选 {sum(r.get('outcome_index') is not None for r in rows)} 条、可结算比赛 {len({r['physical_match_id'] for r in rows if r.get('outcome_index') is not None})} 场。固定 f-v1。",
        "",
        "结论边界：",
        "1. 当前包重新对齐了 Stage 3 的 8 条候选；旧的 fm-strategy-comparison-v1-20261021 只有 4 场，不被混入。",
        "2. all_candidates 在多候选比赛没有连续多正文 f replay，因此保持 UNKNOWN；没有把多篇独立 after 拼成伪连续结果。",
        "3. 当前 schema 没有 title、action_id、propensity 和完整 pre-query menu，旧标题 g、正式策略逆概率评估不可计算。",
        "4. new_full_text_state_g 使用 Stage 4 的 LOMO ridge_body_state 诊断分数，只用于选一个候选；不接入 runtime，也不是正式 g。",
        "5. 当前归档没有请求/provider/处理成本台账，total cost 保持 UNKNOWN；图中只展示可计算的 g-read lower-bound proxy，不能声称系统成本下降。",
        "",
        "策略汇总：",
    ]
    for s, v in summary.items():
        lines.append(f"- {s}: 可评估 {v['evaluable_matches']}/{v['matches']}，mean final log loss={v['mean_final_log_loss']}, mean gain={v['mean_realized_gain']}, selected={v['selected_count']}, status cost={v['cost_status']}")
    lines += ["", "不可成立的结论：当前样本量和历史可用性不足以证明全文状态 g 胜过固定顺序、随机或全量策略；不能上线正式 g。"]
    (OUT / "readable-report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "README.md").write_text("# Stage 5 v2\n\n本目录是当前 Stage 2/3 候选池上的只读策略比较。网络、provider、正式 runtime 均未调用。\n\n`total_cost` 保持 UNKNOWN，因为当前归档没有请求/provider/处理成本台账。多候选的 all-candidates 连续 f replay 缺失时保持 UNKNOWN。\n", encoding="utf-8")

    verification = {
        "schema": "footballmind-strategy-comparison-verification-v2",
        "run_id": "fm-strategy-comparison-v2-20261024",
        "network_requests": 0,
        "external_calls": 0,
        "formal_runtime_integration": False,
        "physical_match_count": len(groups),
        "candidate_row_count": len(rows),
        "settled_match_count": len({r["physical_match_id"] for r in rows if r.get("outcome_index") is not None}),
        "strategy_count": len(strategies),
        "oracle_marked_analysis_only": True,
        "all_candidate_multi_match_rows_left_unknown": True,
        "title_baseline_status": "UNAVAILABLE_TITLE_NOT_RETAINED",
        "cost_status": "UNKNOWN_ARCHIVED_COST_COMPONENTS",
        "source_labels_sha256": sha256(LABELS),
        "source_predictions_sha256": sha256(PREDICTIONS),
        "py_compile": True,
    }
    (OUT / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {}
    for p in sorted(OUT.iterdir()):
        if p.name == "manifest.json":
            continue
        manifest[p.name] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    (OUT / "manifest.json").write_text(json.dumps({"schema": "footballmind-stage5-manifest-v2", "files": manifest}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
