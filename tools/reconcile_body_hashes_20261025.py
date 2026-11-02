from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(r"D:\a1-足彩")
INV = ROOT / "var/work/fm-g-historical-replay-expansion-20261025-v1/historical-candidate-inventory.jsonl"
ART = ROOT / "var/corpora/football-news-pipeline-v1/articles.jsonl"
OUT = ROOT / "var/work/fm-body-sha-reconciliation-20261025-v3"


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def normalized(data: bytes) -> bytes:
    text = data.decode("utf-8")
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def stripped(data: bytes) -> bytes:
    text = normalized(data).decode("utf-8")
    return "\n".join(line.rstrip() for line in text.split("\n")).encode("utf-8")


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    candidates = read_jsonl(INV)
    articles = {row.get("article_id"): row for row in read_jsonl(ART)}
    rows = []
    reason = Counter()
    path_counts = Counter()
    metadata_counts = Counter()
    for candidate in candidates:
        article = articles.get(candidate.get("candidate_text_id"), {})
        body_path = Path(candidate["body_path"]) if candidate.get("body_path") else None
        exists = bool(body_path and body_path.exists())
        raw_sha = lf_sha = strip_sha = None
        raw_bytes = None
        if exists:
            raw_bytes = body_path.read_bytes()
            raw_sha = sha_bytes(raw_bytes)
            lf_sha = sha_bytes(normalized(raw_bytes))
            strip_sha = sha_bytes(stripped(raw_bytes))
        meta_main = candidate.get("body_sha256") or candidate.get("normalized_body_sha256") or candidate.get("recorded_body_sha256")
        meta_norm = article.get("normalized_body_sha256")
        meta_recorded = article.get("recorded_body_sha256")
        # The inventory prefers normalized_body_sha256 when available. Compare all
        # known declarations rather than silently picking one interpretation.
        matches = []
        for label, value in (("inventory", meta_main), ("article_normalized", meta_norm), ("article_recorded", meta_recorded)):
            if value and raw_sha and value == raw_sha:
                matches.append(label + "_raw")
            if value and lf_sha and value == lf_sha:
                matches.append(label + "_lf")
            if value and strip_sha and value == strip_sha:
                matches.append(label + "_strip")
        if not exists:
            status = "missing_file"
            reason[status] += 1
        elif matches:
            status = "metadata_matches_file"
            reason[status] += 1
        else:
            status = "metadata_mismatch_all_tested_normalizations"
            reason[status] += 1
        if body_path:
            path_counts[str(body_path.parent)] += 1
        if meta_main == meta_norm and meta_main:
            metadata_counts["inventory_equals_article_normalized"] += 1
        if meta_main == meta_recorded and meta_main:
            metadata_counts["inventory_equals_article_recorded"] += 1
        rows.append({
            "candidate_text_id": candidate.get("candidate_text_id"),
            "physical_match_id": candidate.get("physical_match_id"),
            "body_path": str(body_path) if body_path else None,
            "body_file_exists": exists,
            "body_bytes": len(raw_bytes) if raw_bytes is not None else None,
            "raw_file_sha256": raw_sha,
            "lf_normalized_sha256": lf_sha,
            "line_trailing_whitespace_stripped_sha256": strip_sha,
            "inventory_body_sha256": meta_main,
            "article_normalized_body_sha256": meta_norm,
            "article_recorded_body_sha256": meta_recorded,
            "matching_declarations": matches,
            "status": status,
            "article_corpus_versions": article.get("corpus_versions"),
            "article_source_path": article.get("data_source"),
            "article_body_status": article.get("body_status"),
            "article_body_characters": article.get("body_characters"),
        })
    with (OUT / "body-hash-reconciliation.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    fields = ["candidate_text_id", "physical_match_id", "body_path", "body_file_exists", "raw_file_sha256", "lf_normalized_sha256", "line_trailing_whitespace_stripped_sha256", "inventory_body_sha256", "article_normalized_body_sha256", "article_recorded_body_sha256", "matching_declarations", "status", "article_corpus_versions", "article_source_path", "article_body_status", "article_body_characters"]
    with (OUT / "body-hash-reconciliation.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row[key], ensure_ascii=False) if isinstance(row[key], (list, dict)) else row[key] for key in fields})
    report = {
        "schema": "footballmind-body-sha-reconciliation-v1",
        "candidate_rows": len(rows),
        "status_counts": dict(reason),
        "metadata_counts": dict(metadata_counts),
        "body_paths_by_parent": dict(path_counts),
        "raw_sha_matches_any_declaration": sum(bool(row["matching_declarations"]) and any(x.endswith("_raw") for x in row["matching_declarations"]) for row in rows),
        "lf_sha_matches_any_declaration": sum(bool(row["matching_declarations"]) and any(x.endswith("_lf") for x in row["matching_declarations"]) for row in rows),
        "stripped_sha_matches_any_declaration": sum(bool(row["matching_declarations"]) and any(x.endswith("_strip") for x in row["matching_declarations"]) for row in rows),
        "inventory_hash_matches_file": sum(any(x.startswith("inventory_") for x in row["matching_declarations"]) for row in rows),
        "article_normalized_hash_matches_file": sum(any(x.startswith("article_normalized_") for x in row["matching_declarations"]) for row in rows),
        "article_recorded_hash_matches_file": sum(any(x.startswith("article_recorded_") for x in row["matching_declarations"]) for row in rows),
        "source_inventory_sha256": digest(INV),
        "source_articles_sha256": digest(ART),
        "network_requests": 0,
        "api_calls": 0,
        "interpretation": "A mismatch means the declared metadata hash does not equal any tested representation of the current body file; it does not by itself prove the body is corrupt. The corpus projection/version rule must be recovered before f consumption.",
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "readable-report.txt").write_text(
        "FM-G 正文 SHA 对账（2026-10-25）\n\n"
        f"候选记录：{len(rows)}\n"
        f"文件存在：{sum(row['body_file_exists'] for row in rows)}\n"
        f"至少一种摘要匹配：{sum(bool(row['matching_declarations']) for row in rows)}\n"
        f"任意元数据声明与原始字节匹配：{report['raw_sha_matches_any_declaration']}\n"
        f"任意元数据声明与 LF 规范化匹配：{report['lf_sha_matches_any_declaration']}\n"
        f"任意元数据声明与去行尾空白匹配：{report['stripped_sha_matches_any_declaration']}\n"
        f"inventory 选用摘要直接匹配文件：{report['inventory_hash_matches_file']}\n"
        f"article normalized 摘要直接匹配文件：{report['article_normalized_hash_matches_file']}\n"
        f"article recorded 摘要直接匹配文件：{report['article_recorded_hash_matches_file']}\n\n"
        "状态统计：\n" + "\n".join(f"- {key}: {value}" for key, value in sorted(reason.items())) + "\n\n"
        "解释：未匹配不直接等于文件损坏，可能是旧 corpus 版本、正文投影或摘要生成规则不同。未恢复出确切规则前，不将这些正文送入当前实体 f。\n",
        encoding="utf-8",
    )
    files = sorted(path for path in OUT.iterdir() if path.is_file() and path.name not in {"manifest.json", "verification.json"})
    manifest = {path.name: digest(path) for path in files}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verification = {
        "schema": "footballmind-body-sha-reconciliation-verification-v1",
        "manifest_entries": len(files),
        "manifest_sha256_match": all(digest(OUT / name) == value for name, value in manifest.items()),
        "candidate_rows": len(rows),
        "source_files_unchanged": True,
        "network_requests": 0,
        "api_calls": 0,
        "formal_models_modified": False,
        "status": "PASS_READ_ONLY_RECONCILIATION",
    }
    (OUT / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
