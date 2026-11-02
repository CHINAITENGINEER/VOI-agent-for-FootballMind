"""Write an independent, evidence-linked second review of priority annotations.

This is a local review artifact only. It does not write Argilla responses and
does not promote model_completed labels to human gold.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "var/audits/football-system-data-audit-20261021/argilla_temporal_validity_v1/preannotation_v3"
OUT = ROOT / "var/repairs/preannotation-v3-second-review-20261024-v1"


def load_jsonl(path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def load_answers(folder):
    result = {}
    for path in folder.glob("part*.json"):
        result.update(json.loads(path.read_text(encoding="utf-8")).get("records", {}))
    return result


def val(answer, key):
    item = answer.get(key)
    return item.get("value") if isinstance(item, dict) else None


MAIN_DECISIONS = {
    "tv1-claim-s1-bs-C01": ("hold_no_body", "needs_expert_review", "标题提示队讯/预测，但 has_body=false，无法核对 claim、时间和逐字证据；系统可用时间虽有记录，也不能替代正文。"),
    "tv1-claim-s1-bs-C02": ("hold_no_body", "needs_expert_review", "标题提示队讯/预测，但 has_body=false，不能确认伤停、首发或赔率内容。"),
    "tv1-claim-s1-bs-C03": ("hold_no_body", "needs_expert_review", "标题提示队讯/预测，但没有正文和 quote，不能判定是否有本场可建模事实。"),
    "tv1-claim-s1-bs-C04": ("hold_no_body", "needs_expert_review", "标题不足以证明正文中的 claim，正文缺失，保持待人工复核。"),
    "tv1-claim-s1-bs-C05": ("hold_no_body", "needs_expert_review", "正文缺失；不得把标题中的 team news/lineup 词直接映射为 f 特征。"),
}

UNLINKED_DECISIONS = {
    "ur1-009e564cb8bdc7d8d14778df": ("relevant_match_resolution", "yes", "明确是 Aston Villa–Liverpool 赛程变更，建议 match_id 需再用 canonical 赛程核验；可保留为 schedule 候选，不能直接进 f。"),
    "ur1-01f6027bd9c1c7a0d5c9e16e": ("relevant_no_body_claim", "yes", "明确是 Liverpool–RB Leipzig 赛前发布会，但正文是视频/发布会入口，没有可审阅的 claim，保留召回不生成 f 特征。"),
    "ur1-032ed808e47b341eda903a24": ("relevant_no_body_claim", "yes", "明确是 Liverpool–Newcastle 联赛杯决赛赛前发布会，但没有正文事实或确定 match_id。"),
    "ur1-05b5a68d66b323348aee6de9": ("relevant_no_body_claim", "yes", "明确是 Liverpool–Villarreal 欧冠赛前发布会，但只有视频入口，不能从标题生成 claim。"),
    "ur1-0dc4b6ad844e3a62fba2be69": ("relevant_match_unknown", "yes", "含 Bajcetic/Endo 训练回归和 Bradley/Ekitike/Leoni 恢复状态，但属于季前集训，未绑定具体比赛；可作为球队状态研究池。"),
    "ur1-11584b7c229abc5a78711fe6": ("relevant_match_unknown", "yes", "正文明确 Liverpool–Newcastle 联赛杯决赛，但当前未给出稳定 canonical match_id；属于比赛背景/动机材料。"),
    "ur1-11a6bd5da775607f27a5ae2b": ("relevant_no_body_claim", "yes", "明确 Liverpool–Barnsley 足总杯赛前发布会，但只有视频入口，不能提取比赛级事实。"),
    "ur1-131a086d9c991a9c2ff9b906": ("relevant_match_unknown", "yes", "明确 Chelsea–Real Madrid 欧冠次回合赛前发布会，缺少稳定比赛身份与正文 claim。"),
    "ur1-227f75b03fc41434da1f5678": ("relevant_no_body_claim", "yes", "明确 Liverpool–Girona 欧冠赛前发布会，但当前记录只有视频说明。"),
    "ur1-22ebea3e194ac334481c8aed": ("relevant_match_unknown", "yes", "Harvey Elliott 参加 U21 康复训练是状态背景，训练不等于本场可出场，且没有目标比赛。"),
    "ur1-2390d22a1c11f2d58a560bd5": ("relevant_match_unknown", "yes", "Mac Allister 季前回归训练属于球队状态背景，没有具体目标比赛和赛前窗口。"),
    "ur1-259372608445ed31f8ea18e4": ("relevant_no_body_claim", "yes", "明确 Liverpool–Atalanta 欧联赛前发布会，但缺少正文 claim 和 canonical match_id。"),
    "ur1-271a9c65304a0f61573628d1": ("relevant_match_unknown", "yes", "明确 Liverpool–Villarreal 欧冠赛前 Q&A，并提到伤停更新，但记录未给稳定 match_id；可保留为待关联候选。"),
    "ur1-278d83c5ab06bc9e067ff65d": ("relevant_match_resolution", "yes", "明确 Robertson 停赛及下场 Tottenham 可出战，建议 Tottenham–Liverpool match_id 方向合理，但仍需 canonical 赛程核验。"),
    "ur1-30ed95e68efde468b44075ea": ("accept_candidate_pending_identity", "yes", "正文明确 Arsenal 客战 Nottingham Forest，列出首发、替补和伤停，属于高价值正文候选；建议 ID 需 canonical 化后才能进入 f/g 数据集。"),
    "ur1-4cd71f9f33133a22c7448145": ("accept_candidate_pending_identity", "yes", "正文明确 Palace 客战 Brighton，并讨论 Sarr 的出场状态，属于本场相关候选；分钟数与可用性仍需按 claim 细化。"),
    "ur1-573c6e94abb19d9273315a64": ("historical_or_unlinked", "no", "正文主要回顾 Welbeck 对 Ostersunds 的上一场表现，未形成确定的当前目标比赛 claim。"),
    "ur1-ab6f614e1ddf96063629826d": ("relevant_match_unknown", "yes", "明确 Arsenal–Napoli 欧联赛前比赛，但记录未给稳定 match_id 和可复核时间绑定。"),
    "ur1-af65da775b2de0dd9396dd4d": ("historical_or_unlinked", "no", "主要是 Baku 欧联杯决赛历史回顾和未来 FA Cup 表态，缺少稳定目标比赛身份，不进入当前 f 候选。"),
    "ur1-efe647982f41ba4a345be862": ("accept_candidate_pending_identity", "yes", "正文明确 Palace–Brighton 2026-10-18、Amex 和 Sarr 状态，适合作为本场 context 候选；需 canonical ID 与正文 claim 审核。"),
}


def make_row(record, answer, decision, reviewed_value, rationale, dataset):
    return {"dataset": dataset, "record_id": record["record_id"],
            "article_title": record.get("article_title"), "source_domain": record.get("source_domain"),
            "source_url": record.get("source_url") or record.get("article_url"),
            "body": record.get("article_body_text"), "claim_quote": record.get("claim_quote"),
            "match": record.get("match"), "times": record.get("times"), "has_body": record.get("has_body"),
            "original_model_values": {key: val(answer, key) for key in ("entity_match_status", "claim_type",
                "event_time_relation", "validity_at_kickoff", "final_disposition", "contains_relevant_claim",
                "suggested_match_id") if val(answer, key) is not None},
            "second_review": {"decision": decision, "reviewed_value": reviewed_value,
                              "rationale": rationale, "reviewer": "codex_total_control_second_review",
                              "human_gold": False, "production_ready": False}}


def build():
    main = {r["record_id"]: r for r in load_jsonl(SRC / "bundles/main_backlog.jsonl")}
    unlinked = {r["record_id"]: r for r in load_jsonl(SRC / "bundles/unlinked.jsonl")}
    ma = load_answers(SRC / "answers/main_backlog")
    ua = load_answers(SRC / "answers/unlinked")
    reviewed = []
    for rid, (decision, reviewed_value, rationale) in MAIN_DECISIONS.items():
        reviewed.append(make_row(main[rid], ma.get(rid, {}), decision, reviewed_value, rationale, "main"))
    for rid, (decision, reviewed_value, rationale) in UNLINKED_DECISIONS.items():
        reviewed.append(make_row(unlinked[rid], ua.get(rid, {}), decision, reviewed_value, rationale, "unlinked"))
    counts = Counter(r["second_review"]["decision"] for r in reviewed)
    summary = {"schema": "footballmind-preannotation-v3-second-review-v1", "status": "research_review_only",
               "reviewed_records": len(reviewed), "main_records": sum(r["dataset"] == "main" for r in reviewed),
               "unlinked_records": sum(r["dataset"] == "unlinked" for r in reviewed),
               "decision_counts": dict(counts),
               "original_source": str(SRC),
               "limitations": ["independent Codex review is not an independent human gold label",
                               "no Argilla responses were modified", "no f/g training or runtime activation",
                               "match suggestions remain candidates until canonical identity verification"]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "second-review.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in reviewed), encoding="utf-8")
    (OUT / "second-review-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "README.md").write_text("# preannotation v3 总控二审\n\n"
        "二审只读回看优先级样本的正文、引文、时间和比赛关联。结果是研究复核意见，不是人工金标，不写回 Argilla，不进入 production f/g。\n",
        encoding="utf-8")
    files = []
    for path in sorted(OUT.iterdir()):
        if path.name == "manifest.json" or not path.is_file():
            continue
        files.append({"path": path.name, "bytes": path.stat().st_size,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (OUT / "manifest.json").write_text(json.dumps({"schema": "footballmind-preannotation-v3-second-review-manifest-v1",
        "files": files, "source_manifest_sha256": hashlib.sha256((SRC / "manifest_v3.json").read_bytes()).hexdigest()},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
