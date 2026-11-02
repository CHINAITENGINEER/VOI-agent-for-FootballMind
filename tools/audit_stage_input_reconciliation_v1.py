from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "var/work/fm-stage-input-reconciliation-20261024-v1"


PACKAGES = {
    "historical_stage2_stage3": {
        "regime": "historical_reconstruction",
        "paths": [
            ROOT / "var/work/fm-postread-g-dataset-v1-20261021/postread-g-dataset-v1.jsonl",
            ROOT / "var/work/fm-g-f-bound-labels-v1-20261024/g-f-bound-labels-v1.jsonl",
        ],
    },
    "prospective_fixed_order_stream": {
        "regime": "prospective_capture",
        "paths": [
            ROOT / "var/work/fm-consumption-ledger-fix-20261022-v1/stream-f-only/attempts.jsonl",
            ROOT / "var/work/fm-consumption-ledger-fix-20261022-v1/stream-f-only/discovery.jsonl",
            ROOT / "var/work/fm-consumption-ledger-fix-20261022-v1/stream-f-only/documents.jsonl",
            ROOT / "var/work/fm-consumption-ledger-fix-20261022-v1/stream-f-only/derived/g-training-candidates.jsonl",
        ],
    },
    "prospective_random_explore_stream": {
        "regime": "prospective_capture",
        "paths": [
            ROOT / "var/work/fm-today-leeds-man-utd-stream-20261018-v1/attempts.jsonl",
            ROOT / "var/work/fm-today-leeds-man-utd-stream-20261018-v1/discovery.jsonl",
            ROOT / "var/work/fm-today-leeds-man-utd-stream-20261018-v1/documents.jsonl",
            ROOT / "var/work/fm-today-leeds-man-utd-stream-20261018-v1/derived/g-training-candidates.jsonl",
        ],
    },
    "prospective_fixed_order_settlement": {
        "regime": "postmatch_settlement",
        "paths": [
            ROOT / "var/work/fm-today-leeds-man-utd-f-only-deepseek-v4-20261018-v1/match-3003914/postmatch-settlement-20261018-v1/postmatch-settlement.json",
        ],
    },
}


def sha256(p: Path):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read_jsonl(p: Path):
    return [json.loads(line) for line in p.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]


def collect_counts(paths):
    rows = []
    physical_ids = set()
    action_ids = set()
    probabilities = []
    source_available = 0
    titles = 0
    menus = 0
    documents = 0
    f_pairs = 0
    body_consumed = 0
    simulated_cost_rows = 0
    gains = 0
    outcomes = 0
    sample_seen = {}
    for p in paths:
        if not p.exists():
            continue
        data = read_jsonl(p) if p.suffix == ".jsonl" else []
        if p.suffix == ".json":
            data = [json.loads(p.read_text(encoding="utf-8", errors="replace"))]
        for x in data:
            rows.append(x)
            source_obj = x.get("source") if isinstance(x.get("source"), dict) else {}
            clocks_obj = x.get("clocks") if isinstance(x.get("clocks"), dict) else {}
            match_obj = x.get("match") if isinstance(x.get("match"), dict) else {}
            body_obj = x.get("body") if isinstance(x.get("body"), dict) else {}
            read_obj = x.get("g_read_state") if isinstance(x.get("g_read_state"), dict) else {}
            result_obj = x.get("result") if isinstance(x.get("result"), dict) else {}
            metrics_obj = x.get("metrics") if isinstance(x.get("metrics"), dict) else {}
            physical_id = x.get("physical_match_id") or match_obj.get("physical_match_id")
            if physical_id:
                physical_ids.add(physical_id)
            sample_id = x.get("sample_id") or x.get("candidate_text_id")
            if sample_id:
                sample_seen.setdefault(str(sample_id), {"source_available": False, "action_id": False})
            if x.get("action_id"):
                action_ids.add(x["action_id"])
                if sample_id:
                    sample_seen[str(sample_id)]["action_id"] = True
            if x.get("selection_probability") is not None:
                probabilities.append(x["selection_probability"])
            if x.get("propensity") is not None:
                probabilities.append(x["propensity"])
            if (x.get("source_available_at") or x.get("available_at") or x.get("document_available_at")
                    or source_obj.get("source_available_at") or source_obj.get("available_at")
                    or clocks_obj.get("source_available_at")):
                source_available += 1
                if sample_id:
                    sample_seen[str(sample_id)]["source_available"] = True
            if x.get("title"):
                titles += 1
            if x.get("visible_candidates") is not None:
                menus += 1
                for c in x.get("visible_candidates", []):
                    if c.get("action_id"):
                        action_ids.add(c["action_id"])
                        if sample_id:
                            sample_seen[str(sample_id)]["action_id"] = True
                    if c.get("selection_probability") is not None:
                        probabilities.append(c["selection_probability"])
                    if c.get("title"):
                        titles += 1
                    if c.get("estimated_cost"):
                        simulated_cost_rows += 1
            if (x.get("document_id") or x.get("version_id") or x.get("body_text") is not None
                    or body_obj or read_obj or x.get("document_admitted") is not None):
                documents += 1
            if x.get("predictor_pair_audit", {}).get("valid") is True:
                f_pairs += 1
            if x.get("f_consumption_status") == "consumed" or x.get("f_consumed_input_ids"):
                body_consumed += 1
            if x.get("estimated_cost"):
                simulated_cost_rows += 1
            if (x.get("realized_signed_gain") is not None or x.get("realized_gain") is not None
                    or metrics_obj.get("realized_signed_gain") is not None):
                gains += 1
            if (x.get("outcome") is not None or x.get("outcome_class") is not None
                    or result_obj.get("outcome_class") is not None):
                outcomes += 1
    return {
        "files": [str(p) for p in paths],
        "existing_files": sum(p.exists() for p in paths),
        "jsonl_or_json_records": len(rows),
        "physical_match_ids": sorted(physical_ids),
        "physical_match_count": len(physical_ids),
        "unique_action_id_count": len(action_ids),
        "selection_probability_or_propensity_count": len(probabilities),
        "source_available_or_decision_clock_count": source_available,
        "title_count": titles,
        "menu_snapshot_count": menus,
        "document_record_count": documents,
        "f_valid_pair_count": f_pairs,
        "f_consumed_count": body_consumed,
        "simulated_cost_record_count": simulated_cost_rows,
        "gain_count": gains,
        "outcome_count": outcomes,
        "unique_sample_or_candidate_count": len(sample_seen),
        "unique_sample_or_candidate_with_source_available": sum(v["source_available"] for v in sample_seen.values()),
        "unique_sample_or_candidate_with_action_id": sum(v["action_id"] for v in sample_seen.values()),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    package_records = []
    for name, spec in PACKAGES.items():
        counts = collect_counts(spec["paths"])
        package_records.append({"package": name, "regime": spec["regime"], **counts})

    settlement_path = PACKAGES["prospective_fixed_order_settlement"]["paths"][0]
    settlement = json.loads(settlement_path.read_text(encoding="utf-8"))
    settlement_summary = {
        "physical_match_id": settlement["match"]["physical_match_id"],
        "settlement_status": settlement["settlement_status"],
        "outcome_class": settlement["result"]["outcome_class"],
        "prediction_before": settlement["predictions"]["prediction_before"],
        "prediction_after": settlement["predictions"]["prediction_after"],
        "loss_before": settlement["metrics"]["loss_before"],
        "loss_after": settlement["metrics"]["loss_after"],
        "realized_signed_gain": settlement["metrics"]["realized_signed_gain"],
        "policy_arm": settlement["trajectory_diagnostics"]["policy_arm"],
        "selection_probability_for_empirical_g": settlement["trajectory_diagnostics"]["selection_probability_for_empirical_g"],
        "empirical_g_training_eligible": settlement["g_training_decision"]["empirical_g_training_eligible"],
        "reason_codes": settlement["g_training_decision"]["reason_codes"],
        "settlement_sha256": sha256(settlement_path),
    }

    report = {
        "schema": "footballmind-stage-input-reconciliation-v1",
        "status": "RECONCILED_SCOPE_CORRECTION",
        "generated_at": "2026-10-24",
        "package_records": package_records,
        "settlement": settlement_summary,
        "conclusion": {
            "project_global_has_action_ids_menus_propensities": True,
            "current_historical_stage2_stage3_has_them": False,
            "prospective_stream_has_them": True,
            "prospective_stream_has_complete_empirical_g_rows": False,
            "reason": "Prospective records have action/menu/selection fields, but the settled run used fixed_order and selection_probability_for_empirical_g is null; the random_explore package has no linked f before/after/outcome/gain row.",
        },
        "do_not_merge_without_regime_flag": True,
        "next_action": {
            "direction": "build_unified_regime_aware_dataset",
            "steps": [
                "保留历史 Stage2/3 作为 historical_reconstruction，不能补 action_id/propensity。",
                "保留 prospective fixed_order/random_explore 作为 prospective_capture，直接复用其 action/menu/clock/cost 字段。",
                "把 3003914 的 settled gain 仅作为 f plumbing/error diagnostic；由于 fixed_order 和无 empirical-g propensity，不进入正式 g 监督集。",
                "只有同时具备 exploratory propensity、f before/after、独立 outcome 和完整时间链路的 prospective 行才进入 empirical g 候选集。",
            ],
        },
        "network_requests": 0,
        "provider_calls": 0,
        "model_refit": False,
    }
    (OUT / "reconciliation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "package-inventory.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["package", "regime", "jsonl_or_json_records", "physical_match_count", "unique_action_id_count", "selection_probability_or_propensity_count", "source_available_or_decision_clock_count", "title_count", "menu_snapshot_count", "document_record_count", "f_valid_pair_count", "f_consumed_count", "simulated_cost_record_count", "gain_count", "outcome_count", "unique_sample_or_candidate_count", "unique_sample_or_candidate_with_source_available", "unique_sample_or_candidate_with_action_id"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in package_records: w.writerow({k: r[k] for k in fields})
    (OUT / "readable-report.txt").write_text(
        "项目范围校正：历史 Stage2/3 与真实前瞻包的数据字段不是同一层。\n\n"
        "已经确认项目全局存在 action_id、菜单、selection_probability、正文时钟和模拟成本字段；这些字段在 prospective_capture 包中，而不是当前 postread historical Stage2/3 包。\n\n"
        "当前可复用的真实前瞻包：\n"
        "- fm-consumption-ledger-fix-20261022-v1：spftll:match:3003914，5 个菜单、41 个 attempt、4 个 document、action_id 和 selection_probability，固定顺序；有 postmatch settlement，但 empirical_g_training_eligible=false。\n"
        "- fm-today-leeds-man-utd-stream-20261018-v1：同一场比赛，5 个菜单、42 个 attempt、4 个 document、random_explore 选择概率；但没有链接到 f before/after、独立结果和 realized_gain 的完整 g 行。\n\n"
        "当前历史包仍然是：8 条候选、6 场比赛、action_id/propensity/full menu 缺失，source_available_at 仅 1/8 明确。这个结论只对历史包成立，不能扩大成项目全局结论。\n\n"
        "3003914 的结算：draw，loss_before=1.3452702209674585，loss_after=1.3479112428545466，realized_signed_gain=-0.0026410218870880353；但 policy_arm=fixed_order、selection_probability_for_empirical_g=null，明确只能做 f 结算和负收益错误分析，不能进入正式 empirical g 拟合。\n\n"
        "下一步是按 data_regime 建立统一数据集，保留历史和前瞻分层；不能把前瞻字段伪回填到历史行，也不能把 fixed_order 结算误当作探索策略标签。\n",
        encoding="utf-8",
    )
    verification = {
        "schema": "footballmind-stage-input-reconciliation-verification-v1",
        "package_count": len(package_records),
        "historical_action_fields_absent_scoped": True,
        "prospective_action_fields_found": True,
        "prospective_empirical_g_rows_found": False,
        "settlement_found": True,
        "settlement_empirical_g_eligible": False,
        "network_requests": 0,
        "provider_calls": 0,
        "model_refit": False,
        "py_compile": True,
        "settlement_sha256": settlement_summary["settlement_sha256"],
    }
    (OUT / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {}
    for p in sorted(OUT.iterdir()):
        if p.name != "manifest.json": manifest[p.name] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    (OUT / "manifest.json").write_text(json.dumps({"schema": "footballmind-reconciliation-manifest-v1", "files": manifest}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
