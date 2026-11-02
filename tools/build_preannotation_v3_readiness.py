"""Build a read-only training-readiness view from Argilla preannotations.

The source labels are model_completed preannotations, not human gold. This
tool only joins bundles with answer files, records provenance, and computes
explicit admission/review flags; it never trains or modifies the source data.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "var/audits/football-system-data-audit-20261021/argilla_temporal_validity_v1/preannotation_v3"
OUT = ROOT / "var/repairs/preannotation-v3-training-readiness-20261024-v1"
F_CLAIM_TYPES = {"availability", "selection", "suspension", "tactical_plan", "schedule", "stakes"}


def rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def value(answer, key):
    item = answer.get(key)
    return item.get("value") if isinstance(item, dict) else None


def load_answers(folder):
    result = {}
    for path in sorted(folder.glob("part*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        result.update(payload.get("records", {}))
    return result


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    main = rows(SOURCE / "bundles/main_backlog.jsonl")
    unlinked = rows(SOURCE / "bundles/unlinked.jsonl")
    main_answers = load_answers(SOURCE / "answers/main_backlog")
    unlinked_answers = load_answers(SOURCE / "answers/unlinked")
    main_out, review_out, unlinked_out = [], [], []
    counts = Counter()
    for record in main:
        rid = record["record_id"]
        answer = main_answers.get(rid, {})
        rule = record.get("rule_outputs", {})
        entity = value(answer, "entity_match_status") or rule.get("Q2_entity_match_status", {}).get("value")
        final = value(answer, "final_disposition")
        validity = value(answer, "validity_at_kickoff")
        claim_type = value(answer, "claim_type")
        quote = value(answer, "evidence_quote") or record.get("claim_quote")
        body = record.get("article_body_text") or ""
        q5 = rule.get("Q5_source_pre_kickoff_status", {}).get("value")
        q6 = rule.get("Q6_system_available_pre_kickoff", {}).get("value")
        quote_ok = isinstance(quote, str) and bool(quote.strip()) and quote in body
        strict = bool(entity == "correct" and final == "accept_current_claim" and validity == "valid"
                      and claim_type in F_CLAIM_TYPES and record.get("has_body") is True
                      and quote_ok and q5 == "proven_pre_kickoff")
        low = bool(value(answer, "final_disposition") == "needs_expert_review" or not record.get("has_body")
                   or entity == "wrong" or validity in {"unknown", "irrelevant"})
        row = {"record_id": rid, "match_id": (record.get("match") or {}).get("match_id"),
               "home_team": (record.get("match") or {}).get("home_team"),
               "away_team": (record.get("match") or {}).get("away_team"),
               "kickoff_at": (record.get("match") or {}).get("kickoff_at"),
               "article_title": record.get("article_title"), "source_url": record.get("source_url"),
               "source_published_at": (record.get("times") or {}).get("source_published_at"),
               "system_available_at": (record.get("times") or {}).get("system_available_at"),
               "has_body": record.get("has_body"), "claim_quote": record.get("claim_quote"),
               "review_quote": quote, "claim_type": claim_type, "entity_match_status": entity,
               "event_time_relation": value(answer, "event_time_relation"),
               "validity_at_kickoff": validity, "claim_modality": value(answer, "claim_modality"),
               "final_disposition": final, "q5_source_pre_kickoff": q5,
               "q6_system_available_pre_kickoff": q6, "quote_is_body_substring": quote_ok,
               "label_provenance": "model_completed" if rid in main_answers else "missing_answer",
               "strict_f_candidate": strict, "review_priority": low or entity == "ambiguous",
               "training_authority": "research_preannotation_only"}
        main_out.append(row)
        counts["main_rows"] += 1
        counts["strict_f_candidates"] += int(strict)
        counts["review_priority"] += int(row["review_priority"])
        counts["q6_unknown"] += int(q6 == "unknown")
        counts["q6_yes"] += int(q6 == "yes")
        counts["quote_invalid"] += int(not quote_ok)
        if row["review_priority"]:
            review_out.append(row)
    for record in unlinked:
        rid = record["record_id"]
        answer = unlinked_answers.get(rid, {})
        relevant = value(answer, "contains_relevant_claim")
        suggested = value(answer, "suggested_match_id")
        unlinked_out.append({"record_id": rid, "article_title": record.get("article_title"),
            "source_domain": record.get("source_domain"), "source_url": record.get("article_url"),
            "source_published_at": (record.get("times") or {}).get("source_published_at"),
            "has_body": record.get("has_body"), "contains_relevant_claim": relevant,
            "suggested_match_id": suggested or None,
            "suggested_match_basis": value(answer, "suggested_match_basis"),
            "claim_quote": value(answer, "claim_quote"),
            "label_provenance": "model_completed" if rid in unlinked_answers else "missing_answer",
            "review_priority": relevant == "unsure" or bool(suggested),
            "training_authority": "research_preannotation_only"})
    counts["unlinked_rows"] = len(unlinked_out)
    counts["unlinked_review_priority"] = sum(int(r["review_priority"]) for r in unlinked_out)
    counts["unlinked_relevant_yes"] = sum(r["contains_relevant_claim"] == "yes" for r in unlinked_out)
    counts["unlinked_unsure"] = sum(r["contains_relevant_claim"] == "unsure" for r in unlinked_out)
    report = {"schema": "footballmind-preannotation-v3-training-readiness-v1",
              "source": str(SOURCE), "counts": dict(counts),
              "source_rows": {"main_backlog": len(main), "unlinked": len(unlinked),
                              "main_answer_records": len(main_answers), "unlinked_answer_records": len(unlinked_answers)},
              "status": "research_preannotation_only",
              "admission_rule": {"f_candidate": "entity correct + accept_current_claim + validity valid + supported claim type + body + quote substring + Q5 proven_pre_kickoff",
                                 "g_candidate": "not admitted by this artifact; requires fixed-f before/after and independent outcome",
                                 "human_review": "needs_expert_review/low confidence/ambiguous or unlinked unsure/match suggestion"},
              "prohibitions": ["do not treat model_completed as human gold", "do not treat Q6 unknown as prospective availability",
                               "do not generate empirical g labels", "do not enable runtime", "do not use outcome or gain to alter text labels"],
              "limitations": ["main_backlog contains 563 rows; report-level main total is 613 including prior/other rows not in this bundle",
                              "strict candidates are a readiness view, not a fitted dataset", "source availability is unknown for most rows"]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "main-readiness.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in main_out), encoding="utf-8")
    (OUT / "review-priority.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in review_out), encoding="utf-8")
    (OUT / "unlinked-recall-readiness.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in unlinked_out), encoding="utf-8")
    (OUT / "readiness-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "README.md").write_text("# preannotation v3 训练准入视图\n\n"
        "本目录只读连接 Argilla preannotation_v3 的 bundle 与 model_completed 答案。\n"
        "它不是人工金标，不训练 f/g，不生成 empirical gain，不启用 runtime。原始包保持不变。\n",
        encoding="utf-8")
    files = []
    for path in sorted(OUT.iterdir()):
        if path.name == "manifest.json" or not path.is_file():
            continue
        files.append({"path": path.name, "bytes": path.stat().st_size, "sha256": digest(path)})
    (OUT / "manifest.json").write_text(json.dumps({"schema": "footballmind-preannotation-v3-training-readiness-manifest-v1",
        "files": files, "source_manifest": digest(SOURCE / "manifest_v3.json")}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
