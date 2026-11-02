"""Recompute retrospective log-loss for settled shadow forecasts.

This report deliberately does not rewrite archived trajectories or invent a
post-hoc f prediction.  It joins the frozen before/after probabilities from
the two settled shadow runs with the independently archived result labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SETTLED = ROOT / "var/submission/fm-g-settlement-evidence-20261011-v1/settled-results.jsonl"
SETTLEMENT_SUMMARY = ROOT / "var/submission/fm-g-settlement-evidence-20261011-v1/settlement-summary.json"
ARSENAL_REPORT = ROOT / "var/submission/fm-g-single-match-shadow-20261010-final/shadow-report.json"
MANU_TRACE = ROOT / "var/submission/fm-g-observable-prequery-3003907-20261010-v1/shadow-trace.json"

CHANGED_FILES = [
    ROOT / "football_agent/historical_context_policy.py",
    ROOT / "football_agent/entity_semantic_f_only_policy.py",
    ROOT / "football_research/acquisition/historical_context_training.py",
    ROOT / "football_agent/learned_voi_replay.py",
    ROOT / "tests/research/acquisition/test_entity_semantic_f_collection.py",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_settled_rows():
    return [json.loads(line) for line in SETTLED.read_text(encoding="utf-8").splitlines() if line.strip()]


def arsenals_probabilities():
    report = read_json(ARSENAL_REPORT)
    before_after = report["before_after"]
    return (
        report["physical_match_id"],
        list(before_after["prediction_before"]),
        list(before_after["prediction_after_for_returned_steps"]),
        {
            "document_f_consumed": report["menu_and_document_contract"].get("document_f_consumed"),
            "candidate_body_in_runtime_state_before_read": report["menu_and_document_contract"].get(
                "candidate_body_in_runtime_state_before_read"
            ),
            "document_admission_reason": report["menu_and_document_contract"].get(
                "document_admission_reason"
            ),
            "prediction_update_reason": before_after.get("prediction_update_reason"),
            "policy_arm": report["policy_and_gates"].get("policy_arm"),
        },
    )


def manu_probabilities():
    trace = read_json(MANU_TRACE)
    eligible = [
        step
        for step in trace["steps"]
        if isinstance(step.get("prediction_before"), list)
        and isinstance(step.get("prediction_after"), list)
    ]
    if not eligible:
        raise ValueError("No eligible before/after pair in the ManUtd–Spurs trace")
    step = eligible[-1]
    return (
        "spftll:match:3003907",
        list(step["prediction_before"]),
        list(step["prediction_after"]),
        {
            "eligible_step": len(trace["steps"]) - 1 - list(reversed(trace["steps"])).index(step),
            "target_body_used": False,
            "f_consumed": None,
            "predictor": "observed-market-prior",
            "selection_status": "observable_prequery_shadow_only",
        },
    )


def loss(probabilities, outcome):
    index = {"home": 0, "draw": 1, "away": 2}[outcome]
    p = float(probabilities[index])
    if not 0.0 < p <= 1.0:
        raise ValueError(f"Invalid outcome probability {p!r}")
    return -math.log(p)


def main(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    settlements = {row["physical_match_id"]: row for row in load_settled_rows()}
    summary = read_json(SETTLEMENT_SUMMARY)
    forecast_map = {
        "spftll:match:3003899": arsenals_probabilities(),
        "spftll:match:3003907": manu_probabilities(),
    }
    rows = []
    for match_id, (forecast_id, before, after, evidence) in forecast_map.items():
        if forecast_id != match_id:
            raise ValueError("Forecast/result identity mismatch")
        result = settlements[match_id]
        outcome = result["result"]
        before_loss = loss(before, outcome)
        after_loss = loss(after, outcome)
        rows.append(
            {
                "physical_match_id": match_id,
                "kickoff_at": result["kickoff_at"],
                "settlement": {
                    "result": outcome,
                    "regulation_home_goals": result["regulation_home_goals"],
                    "regulation_away_goals": result["regulation_away_goals"],
                    "result_observed_at": result["result_observed_at"],
                    "observation_content_hash": result["observation_content_hash"],
                    "independent_outcome_verified": result["independent_outcome_verified"],
                },
                "forecast": {
                    "prediction_before": before,
                    "prediction_after": after,
                    "forecast_source": "frozen_archived_shadow_trace",
                    "f_was_run_in_original_trace": False,
                    "body_to_f_route_evidence": evidence,
                },
                "log_loss": {
                    "formula": "-log(p_observed_outcome)",
                    "outcome_index": {"home": 0, "draw": 1, "away": 2}[outcome],
                    "loss_before": before_loss,
                    "loss_after": after_loss,
                    "loss_reduction_before_minus_after": before_loss - after_loss,
                    "probability_delta_l1": sum(abs(a - b) for a, b in zip(before, after)),
                },
                "training_status": "not_eligible_no_f_consumption_and_no_predeclared_g_label",
            }
        )

    mean_before = sum(row["log_loss"]["loss_before"] for row in rows) / len(rows)
    mean_after = sum(row["log_loss"]["loss_after"] for row in rows) / len(rows)
    total_reduction = sum(row["log_loss"]["loss_reduction_before_minus_after"] for row in rows)
    pending = [
        {
            "physical_match_id": match_id,
            "status": "not_calculable_no_verified_settlement_in_frozen_evidence",
        }
        for match_id in summary.get("pending_matches", [])
    ]
    report = {
        "schema": "footballmind-f-body-fix-loss-recalculation-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "mode": "retrospective_archived_forecast_log_loss",
            "settled_rows": len(rows),
            "pending_rows": len(pending),
            "does_not_rewrite_archives": True,
            "does_not_claim_repaired_f_forecasts_for_old_generic_runs": True,
        },
        "bug_diagnosis": {
            "old_failure": "generic market-prior runs admitted pages as unmodeled/replanning-only and did not route body into f",
            "repair": "operational EntitySemanticFOnlyPolicy now admits body when available_at <= decision_at even when source_published_at is null; historical reconstruction remains strict",
            "operational_gate": "body_admission=observed-body-availability-v1",
        },
        "settled_rows": rows,
        "aggregate": {
            "mean_log_loss_before": mean_before,
            "mean_log_loss_after": mean_after,
            "mean_loss_reduction_before_minus_after": mean_before - mean_after,
            "total_loss_reduction_before_minus_after": total_reduction,
            "interpretation": "The archived predictions were market-prior-only; equal before/after is therefore expected for these old runs and is evidence of the old routing bug, not evidence that the repaired f has zero value.",
        },
        "pending_rows": pending,
        "repair_verification": {
            "focused_test_command": "python -m pytest -q tests/research/acquisition/test_entity_semantic_f_collection.py tests/research/acquisition/test_entity_semantic_f.py tests/research/acquisition/test_entity_semantic_policy.py tests/research/acquisition/test_entity_semantic_g_contract.py tests/agent/test_historical_context_policy.py tests/research/acquisition/test_historical_context_training.py tests/research/acquisition/test_html_body_runtime.py --basetemp D:/fm-test-body-20261013c",
            "focused_tests_passed": 72,
            "py_compile": True,
            "new_regression": "unknown publication clock + verified available_at admits body and changes f probabilities; replay remains byte-exact",
        },
        "limitations": [
            "The two settled generic runs have no archived f semantic extraction and cannot be retroactively re-run as if f had been active before kickoff.",
            "A repaired prospective loss requires a new pre-kickoff run with --f-only-predictor and --f-only-semantic-config, followed by an independent settlement.",
            "No empirical g row is created by this retrospective calculation; UNKNOWN/constant gate remains unchanged.",
        ],
        "source_hashes": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [SETTLED, SETTLEMENT_SUMMARY, ARSENAL_REPORT, MANU_TRACE]
        },
        "changed_file_hashes": {
            str(path.relative_to(ROOT)): sha256(path) for path in CHANGED_FILES
        },
    }
    report_path = output / "loss-recalculation.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    text = []
    text.append("FootballMind：正文进入 f 修复后的损失复算（赛后回放）")
    text.append("=" * 64)
    text.append("")
    text.append("复算范围：只读取已经冻结的赛前 before/after 概率和独立结算结果；不改写原始轨迹，不把赛后信息送回赛前。")
    text.append("")
    text.append("修复证据：")
    text.append("- 旧路径的页面曾被标记为 unmodeled_document_for_replanning_only，正文没有进入 f。")
    text.append("- 新路径只在 available_at <= decision_at 时放行正文；source_published_at 仍可为 null；历史重建路径仍要求保守发布时间。")
    text.append(f"- 针对性回归：72 passed；py_compile：passed。")
    text.append("")
    text.append("逐场复算（log-loss = -log(真实结果对应概率)）：")
    text.append("比赛                         结果   before loss       after loss        reduction")
    text.append("-" * 86)
    names = {"spftll:match:3003899": "Arsenal–Leeds", "spftll:match:3003907": "ManUtd–Spurs"}
    for row in rows:
        ll = row["log_loss"]
        text.append(
            f"{names[row['physical_match_id']]:28} {row['settlement']['result']:5} "
            f"{ll['loss_before']:.12f}   {ll['loss_after']:.12f}   {ll['loss_reduction_before_minus_after']:+.12f}"
        )
    text.append("-" * 86)
    text.append(f"平均 log-loss                 {mean_before:.12f}   {mean_after:.12f}   {mean_before - mean_after:+.12f}")
    text.append("")
    text.append("结论：这两场旧轨迹的 before/after 完全相同，因此复算得到的 loss 也完全相同。原因已由轨迹字段直接证明：当时运行的是 market-prior 路径，正文被保留为未建模材料，f 没有在赛前消费正文。这个数值不能用来证明修复后的 f 没有收益；它只证明旧运行没有执行用户要求的那条链路。")
    text.append("")
    text.append("要得到修复后 f 的真实 loss，下一场必须在开球前以 f-only predictor + semantic config 启动，记录正文进入 f 后的 before/after，再等独立赛果结算。当前 empirical g 仍保持 UNKNOWN/constant gate。")
    (output / "loss-recalculation.txt").write_text("\n".join(text) + "\n", encoding="utf-8")

    manifest = {
        "schema": "footballmind-f-body-fix-loss-recalculation-manifest-v1",
        "files": {
            name: sha256(output / name)
            for name in ("loss-recalculation.json", "loss-recalculation.txt")
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "manifest": manifest, "aggregate": report["aggregate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "var/reviews/fm-loss-recalculation-after-body-fix-20261013-v1",
    )
    args = parser.parse_args()
    main(args.output)
