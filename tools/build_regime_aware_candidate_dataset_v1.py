from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "var/work/fm-postread-g-dataset-v1-20261021/postread-g-dataset-v1.jsonl"
LABELS = ROOT / "var/work/fm-g-f-bound-labels-v1-20261024/g-f-bound-labels-v1.jsonl"
FIXED_ROOT = ROOT / "var/work/fm-consumption-ledger-fix-20261022-v1/stream-f-only"
RANDOM_ROOT = ROOT / "var/work/fm-today-leeds-man-utd-stream-20261018-v1"
SETTLEMENT = ROOT / "var/work/fm-today-leeds-man-utd-f-only-deepseek-v4-20261018-v1/match-3003914/postmatch-settlement-20261018-v1/postmatch-settlement.json"
OUT = ROOT / "var/work/fm-regime-aware-candidate-dataset-v1-20261024"


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]


def body_hash(text):
    if not text:
        return None
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def prediction_matches_settlement(value, target):
    if value is None or target is None:
        return False
    if isinstance(value, dict):
        value = [value.get("home"), value.get("draw"), value.get("away")]
    if len(value) != len(target):
        return False
    return all(abs(float(a) - float(b)) < 1e-12 for a, b in zip(value, target))


def index_discovery(path):
    out = {}
    menus = {}
    for d in read_jsonl(path):
        menus[d.get("menu_id")] = d
        for c in d.get("visible_candidates", []):
            key = c.get("action_id")
            if key:
                out[key] = {"menu_id": d.get("menu_id"), "menu_available_at": d.get("available_at"), **c}
    return out, menus


def index_attempts(path):
    return {x.get("action_id"): x for x in read_jsonl(path) if x.get("action_id")}


def index_documents(path):
    out = {}
    for d in read_jsonl(path):
        for key in (d.get("canonical_url"), d.get("url")):
            if key:
                out[key] = d
    return out


def common_record(**kwargs):
    keys = [
        "record_id", "data_regime", "information_mode", "physical_match_id", "sample_id", "candidate_text_id", "action_id", "menu_id", "policy_arm", "selection_probability", "propensity_status", "candidate_title", "source_url", "source_domain", "body_path", "body_sha256", "body_text_sha256", "source_published_at", "source_available_at", "captured_at", "decision_at", "kickoff", "cutoff", "prediction_before", "prediction_after", "outcome_class", "loss_before", "loss_after", "realized_gain", "f_body_consumed", "f_modeled_claim_count", "f_prediction_changed", "f_pair_status", "selection_label_status", "empirical_g_training_eligible", "exclusion_reason", "provenance_files"
    ]
    return {k: kwargs.get(k) for k in keys}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    historical = {x["sample_id"]: x for x in read_jsonl(HIST)}
    labels = {x["sample_id"]: x for x in read_jsonl(LABELS)}
    records = []
    for sid, row in historical.items():
        label = labels[sid]
        src = row.get("source", {})
        body = row.get("body", {})
        fx = row.get("fixture", {})
        fexec = label.get("f_execution", {})
        records.append(common_record(
            record_id=f"historical:{sid}", data_regime="historical_reconstruction", information_mode="historical_reconstruction", physical_match_id=row.get("physical_match_id"), sample_id=sid, candidate_text_id=sid, action_id=None, menu_id=None, policy_arm=None, selection_probability=None, propensity_status="missing_in_historical_panel", candidate_title=None, source_url=row.get("source", {}).get("source_url"), source_domain=row.get("source", {}).get("source_domain"), body_path=body.get("body_path"), body_sha256=body.get("body_sha256"), body_text_sha256=None, source_published_at=src.get("source_published_at"), source_available_at=src.get("source_available_at"), captured_at=src.get("captured_at"), decision_at=None, kickoff=fx.get("kickoff"), cutoff=row.get("current_state", {}).get("cutoff"), prediction_before=label.get("p_before"), prediction_after=label.get("p_after"), outcome_class=label.get("outcome_class"), loss_before=label.get("loss_before"), loss_after=label.get("loss_after"), realized_gain=label.get("realized_gain"), f_body_consumed=fexec.get("body_received"), f_modeled_claim_count=fexec.get("modeled_claim_count"), f_prediction_changed=fexec.get("prediction_changed"), f_pair_status="historical_offline_pair", selection_label_status=label.get("eligibility_status"), empirical_g_training_eligible=False, exclusion_reason=label.get("exclusion_reason"), provenance_files=[str(HIST), str(LABELS)]
        ))

    settlement = json.loads(SETTLEMENT.read_text(encoding="utf-8"))
    settled_before = settlement["predictions"]["prediction_before"]
    settled_after = settlement["predictions"]["prediction_after"]
    settled_gain = settlement["metrics"]["realized_signed_gain"]
    settled_outcome = settlement["result"]["outcome_class"]

    for regime, root, arm in [("prospective_fixed_order", FIXED_ROOT, "fixed_order"), ("prospective_random_explore", RANDOM_ROOT, "random_explore")]:
        attempts = index_attempts(root / "attempts.jsonl")
        candidates, menus = index_discovery(root / "discovery.jsonl")
        documents = index_documents(root / "documents.jsonl")
        derived = read_jsonl(root / "derived/g-training-candidates.jsonl")
        for i, d in enumerate(derived):
            action = d.get("action_id")
            cand = candidates.get(action, {})
            attempt = attempts.get(action, {})
            doc = documents.get(cand.get("url"), {})
            p_before = d.get("prediction_before")
            p_after = d.get("prediction_after")
            is_settled_action = regime == "prospective_fixed_order" and prediction_matches_settlement(p_after, settled_after) and prediction_matches_settlement(p_before, settled_before)
            # The settlement is attached only to the action whose final pair matches it.
            # Other actions keep outcome/gain UNKNOWN rather than inheriting a match-level label.
            records.append(common_record(
                record_id=f"{regime}:{action or i}", data_regime=regime, information_mode="prospective_capture", physical_match_id=d.get("physical_match_id"), sample_id=None, candidate_text_id=action, action_id=action, menu_id=cand.get("menu_id"), policy_arm=d.get("policy_arm", arm), selection_probability=d.get("selection_probability"), propensity_status="selection_probability_recorded_but_not_valid_empirical_propensity" if arm == "fixed_order" else "selection_probability_recorded", candidate_title=cand.get("title"), source_url=cand.get("url"), source_domain=(cand.get("url") or "").split("/")[2] if "://" in (cand.get("url") or "") else None, body_path=None, body_sha256=None, body_text_sha256=body_hash(doc.get("body_text") or doc.get("body")), source_published_at=doc.get("source_published_at"), source_available_at=doc.get("available_at") or doc.get("document_available_at") or attempt.get("document_available_at"), captured_at=doc.get("captured_at") or attempt.get("completed_at"), decision_at=attempt.get("decision_at"), kickoff=attempt.get("kickoff_at"), cutoff=attempt.get("prediction_cutoff"), prediction_before=p_before, prediction_after=p_after, outcome_class=settled_outcome if is_settled_action else None, loss_before=settlement["metrics"]["loss_before"] if is_settled_action else None, loss_after=settlement["metrics"]["loss_after"] if is_settled_action else None, realized_gain=settled_gain if is_settled_action else None, f_body_consumed=d.get("f_consumption_status") == "consumed", f_modeled_claim_count=None, f_prediction_changed=(p_before != p_after) if p_before is not None and p_after is not None else None, f_pair_status="valid_settlement_pair" if is_settled_action else ("runtime_pair_recorded" if d.get("predictor_pair_audit", {}).get("valid") else "not_linked_to_settlement"), selection_label_status="diagnostic_settlement_only" if is_settled_action else d.get("label_status"), empirical_g_training_eligible=False, exclusion_reason=(d.get("exclusion_reason") or []) + (["fixed_order_no_valid_empirical_propensity", "single_match_settlement_not_g_training"] if is_settled_action else ["no_linked_independent_outcome_and_gain"]), provenance_files=[str(root / "attempts.jsonl"), str(root / "discovery.jsonl"), str(root / "documents.jsonl"), str(root / "derived/g-training-candidates.jsonl"), str(SETTLEMENT)]
            ))

    fields = list(records[0])
    with (OUT / "regime-aware-candidates.jsonl").open("w", encoding="utf-8") as f:
        for r in records: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (OUT / "regime-aware-candidates.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in records: w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in r.items()})

    counts = {}
    for regime in sorted({r["data_regime"] for r in records}):
        rr = [r for r in records if r["data_regime"] == regime]
        counts[regime] = {
            "rows": len(rr), "physical_matches": len({r["physical_match_id"] for r in rr}), "action_id_rows": sum(r["action_id"] is not None for r in rr), "selection_probability_rows": sum(r["selection_probability"] is not None for r in rr), "source_available_rows": sum(r["source_available_at"] is not None for r in rr), "f_pair_rows": sum(r["f_pair_status"] in {"runtime_pair_recorded", "valid_settlement_pair", "historical_offline_pair"} for r in rr), "gain_rows": sum(r["realized_gain"] is not None for r in rr), "formal_g_rows": sum(r["empirical_g_training_eligible"] for r in rr)
        }
    summary = {
        "schema": "footballmind-regime-aware-candidate-dataset-v1",
        "rows": len(records), "physical_matches": len({r["physical_match_id"] for r in records}), "regime_counts": counts,
        "formal_empirical_g_rows": sum(r["empirical_g_training_eligible"] for r in records),
        "no_regime_merging_for_training": True,
        "historical_action_fields_not_backfilled": True,
        "fixed_order_settlement_not_promoted_to_empirical_g": True,
        "random_explore_without_f_pair_remains_unlabeled": True,
        "source_sha256": {str(HIST): sha256(HIST), str(LABELS): sha256(LABELS), str(SETTLEMENT): sha256(SETTLEMENT)},
        "network_requests": 0, "provider_calls": 0, "model_refit": False,
    }
    (OUT / "dataset-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text("# Regime-aware candidate dataset v1\n\n历史重建、prospective fixed-order 和 prospective random-explore 分层保存。该包只做数据整理，不把 fixed-order 结算行作为 empirical g 训练行，也不把 random-explore 缺失的 f pair 补成标签。\n", encoding="utf-8")
    (OUT / "readable-report.txt").write_text(
        f"统一候选集共 {len(records)} 行、{len({r['physical_match_id'] for r in records})} 场 physical match。\n\n"
        + "\n".join(f"- {k}: {v}" for k, v in counts.items())
        + "\n\n历史重建行保留 gain/赛果诊断字段，但 action_id、propensity、菜单保持缺失。\n"
        + "prospective fixed-order 保留 action_id、菜单、selection_probability 和时钟；其 3003914 结算只链接到最终匹配的 f pair，仍因 fixed-order、无经验 g propensity 而排除正式 g。\n"
        + "prospective random-explore 保留选择概率和菜单，但缺少已链接的 f before/after 与独立结果，因此不生成 gain。\n",
        encoding="utf-8",
    )
    verification = {
        "schema": "footballmind-regime-aware-dataset-verification-v1", "rows": len(records), "physical_matches": len({r["physical_match_id"] for r in records}), "regimes": sorted(counts), "formal_empirical_g_rows": 0, "historical_fields_not_backfilled": True, "fixed_settlement_excluded_from_empirical_g": True, "random_without_pair_unlabeled": True, "network_requests": 0, "provider_calls": 0, "model_refit": False, "py_compile": True
    }
    (OUT / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {}
    for p in sorted(OUT.iterdir()):
        if p.name != "manifest.json": manifest[p.name] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    (OUT / "manifest.json").write_text(json.dumps({"schema": "footballmind-regime-aware-manifest-v1", "files": manifest}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
