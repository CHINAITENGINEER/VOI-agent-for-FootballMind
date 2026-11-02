from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(r"D:\a1-足彩")
INPUT = ROOT / "var" / "work" / "fm-g-historical-replay-expansion-20261025-v1" / "historical-candidate-inventory.jsonl"
OUT = ROOT / "var" / "work" / "fm-current-f-semantic-review-priority-20261025-v1"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def date_gap(row):
    try:
        published = datetime.fromisoformat(row["published_at"].replace("Z", "+00:00")).date()
        match = datetime.fromisoformat(row["match_date"]).date()
        return max(0, (match - published).days)
    except Exception:
        return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in INPUT.read_text(encoding="utf-8").splitlines() if line.strip()]
    domain_counts = Counter(r.get("source_domain") or "UNKNOWN" for r in rows)
    match_counts = Counter(r.get("physical_match_id") for r in rows)
    body_keys = {id(r): (r.get("body_sha256") or r.get("recorded_body_sha256") or r.get("normalized_body_sha256")) for r in rows}
    body_counts = Counter(body_keys[id(r)] for r in rows)
    max_domain = max(domain_counts.values())
    ranked = []
    for row in rows:
        gap = date_gap(row)
        method = row.get("binding_method")
        score = 0
        reasons = []
        if method == "title_opponent_and_club_date_candidate":
            score += 35
            reasons.append("title_opponent_and_club_date_binding")
        elif method == "official_club_opponent_mention":
            score += 30
            reasons.append("official_club_opponent_binding")
        elif method == "club_temporal_context_candidate":
            score += 20
            reasons.append("club_temporal_context_only")
        else:
            reasons.append("binding_method_unknown")
        if gap is not None:
            if gap <= 1:
                score += 20
                reasons.append("published_0_1_day_before_match")
            elif gap <= 3:
                score += 15
                reasons.append("published_2_3_days_before_match")
            elif gap <= 7:
                score += 10
                reasons.append("published_4_7_days_before_match")
            elif gap <= 30:
                score += 5
                reasons.append("published_8_30_days_before_match")
            else:
                reasons.append("published_more_than_30_days_before_match")
        else:
            reasons.append("published_time_unparseable")
        if row.get("body_sha256_verified"):
            score += 20
            reasons.append("body_sha_recomputed_match")
        else:
            reasons.append("body_sha_needs_reconciliation")
        body_path = Path(row["body_path"]) if row.get("body_path") else None
        chars = body_path.stat().st_size if body_path and body_path.exists() else None
        if chars is not None:
            if chars >= 600:
                score += 5
                reasons.append("body_at_least_600_bytes")
            elif chars >= 200:
                score += 3
                reasons.append("body_200_599_bytes")
            else:
                reasons.append("short_body")
        else:
            reasons.append("body_path_unreadable")
        domain = row.get("source_domain") or "UNKNOWN"
        diversity_bonus = round(10.0 * (1.0 - (domain_counts[domain] - 1) / max_domain), 3)
        score += diversity_bonus
        if domain_counts[domain] <= 10:
            reasons.append("underrepresented_source_domain")
        else:
            reasons.append("source_domain_concentrated")
        mcount = match_counts[row.get("physical_match_id")]
        if mcount == 1:
            score += 5
            reasons.append("single_candidate_match")
        elif mcount == 2:
            score += 3
            reasons.append("two_candidate_match")
        else:
            reasons.append("multi_candidate_match")
        body_key = body_keys[id(row)]
        if body_counts[body_key] == 1:
            score += 5
            reasons.append("unique_body_sha_in_inventory")
        else:
            reasons.append("duplicate_body_sha_group")
        if score >= 75:
            tier = "P1"
        elif score >= 55:
            tier = "P2"
        else:
            tier = "P3"
        out = dict(row)
        out.update({
            "review_priority_score": round(float(score), 3),
            "review_priority_tier": tier,
            "review_priority_reasons": reasons,
            "published_to_match_days": gap,
            "body_characters_recomputed_from_path": chars,
            "source_domain_candidate_count": domain_counts[domain],
            "match_candidate_count": mcount,
            "body_sha_group_count": body_counts[body_key],
            "priority_outcome_usage": "not_used",
            "priority_gain_usage": "not_used",
        })
        ranked.append(out)
    ranked.sort(key=lambda x: (-x["review_priority_score"], x.get("physical_match_id") or "", x.get("candidate_text_id") or ""))
    with (OUT / "semantic-review-priority-queue.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row in ranked:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    counts = Counter(r["review_priority_tier"] for r in ranked)
    summary = {
        "schema": "footballmind-current-f-semantic-review-priority-v1",
        "input_sha256": sha(INPUT),
        "candidate_rows": len(ranked),
        "physical_matches": len(match_counts),
        "tier_counts": dict(counts),
        "binding_method_counts": dict(Counter(r.get("binding_method") for r in ranked)),
        "source_domain_counts": dict(domain_counts),
        "body_sha_verified_rows": sum(bool(r.get("body_sha256_verified")) for r in ranked),
        "body_sha_unverified_rows": sum(not bool(r.get("body_sha256_verified")) for r in ranked),
        "priority_does_not_use": ["outcome_class", "result_source_sha256", "realized_gain", "prediction_after"],
        "network_requests": 0,
        "api_calls": 0,
        "claims_generated": 0,
        "formal_g_training_rows": 0,
    }
    dump(OUT / "priority-summary.json", summary)
    (OUT / "readable-report.txt").write_text(
        "当前实体 f 语义审阅优先级队列（只读）\n\n"
        f"候选 {len(ranked)} 条，独立比赛 {len(match_counts)} 场。\n"
        f"优先级分布：{dict(counts)}。\n"
        "评分只使用正文关联方式、发布时间距比赛日期、正文路径可读性、SHA 对账状态、来源域名分散度、同场候选数和重复正文状态。\n"
        "评分没有使用赛果、gain、prediction_after 或任何训练标签。\n\n"
        "P1 先做：目标绑定较强、发布时间接近比赛、正文可读且来源/比赛不拥挤的记录。\n"
        "P2 再做：绑定或时间证据中等、需要先解决 SHA 或来源聚集的记录。\n"
        "P3 暂缓：仅球队时间语境、发布时间距离较远、重复正文或版本摘要未对账的记录。\n\n"
        "本队列不生成 claims，不调用 f/g，不产生 gain，不等同于训练集。\n",
        encoding="utf-8",
    )
    files = sorted(p for p in OUT.iterdir() if p.is_file() and p.name not in {"manifest.json", "verification.json"})
    dump(OUT / "manifest.json", {p.name: sha(p) for p in files})
    dump(OUT / "verification.json", {
        "schema": "footballmind-current-f-semantic-review-priority-verification-v1",
        "manifest_entries": len(files),
        "network_requests": 0,
        "api_calls": 0,
        "claims_generated": 0,
        "formal_models_modified": False,
        "status": "PASS_READ_ONLY_PRIORITY_BUILD",
    })


if __name__ == "__main__":
    main()
