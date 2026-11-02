"""Finalize a read-only recovery record for a capped article body.

Parses one already-saved raw HTML response. It does not call a provider,
write back to Argilla, run f/g, or create labels.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

ROOT = Path(r"D:/a1-足彩")
OUT = ROOT / r"var/repairs/preannotation-v3-canonical-candidate-audit-20261024-v1"
RAW = OUT / r"body-recovery/raw/ur1-4cd71f9f33133a22c7448145.html"
RECEIPT = OUT / r"body-recovery/recovery-receipt.json"
REVIEW = ROOT / r"var/repairs/preannotation-v3-second-review-20261024-v1/second-review.jsonl"
RECORD_ID = "ur1-4cd71f9f33133a22c7448145"
SOURCE_URL = "https://www.cpfc.co.uk/news/first-team/sage-sarr-wharton-brighton-preview-palace/"
CANONICAL_NAME_ID = "footballmind:premier-league:2026-27:2026-10-18:brighton:crystal-palace"
KICKOFF = "2026-10-18T13:00:00+00:00"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def compact(text: str) -> str:
    return " ".join(text.split())


def source_row() -> dict[str, Any]:
    for line in REVIEW.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("record_id") == RECORD_ID:
            return row
    raise RuntimeError(f"missing review row: {RECORD_ID}")


def extract_body(html: str) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    sections = soup.find_all("section", class_="overflow-hidden")
    if len(sections) < 2:
        raise RuntimeError("expected article section not found")
    section = sections[1]
    blocks: list[dict[str, Any]] = []
    # The article page has a summary block followed by prose articles and then
    # a Related News block. We stop at Related News and exclude match metadata.
    for child_index, child in enumerate(section.find_all(recursive=False)):
        text = compact(child.get_text(" ", strip=True))
        if text.startswith("Related News"):
            break
        # Only textual content blocks are eligible. Image and footer articles
        # have no p/blockquote and are ignored.
        for elem in child.find_all(["p", "blockquote"], recursive=True):
            # Avoid duplicating the p nested inside a blockquote.
            if elem.name == "p" and elem.find_parent("blockquote") is not None:
                continue
            value = compact(elem.get_text(" ", strip=True))
            if not value:
                continue
            kind = "blockquote" if elem.name == "blockquote" else "paragraph"
            blocks.append({
                "ordinal": len(blocks),
                "kind": kind,
                "text": value,
                "html_selector": f"section.overflow-hidden:nth-of-type(2) > child[{child_index}] {elem.name}",
            })
    if not blocks:
        raise RuntimeError("no textual body blocks extracted")
    body = "\n\n".join(item["text"] for item in blocks) + "\n"
    return body, blocks, {
        "section_selector": "section.overflow-hidden[article_body_index=1]",
        "stop_rule": "stop before first direct child whose normalized text starts with Related News",
        "included_elements": ["p", "blockquote"],
        "excluded_content": ["site navigation", "image-only blocks", "Related News", "match metadata h4/ul"],
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    raw_bytes = RAW.read_bytes()
    raw_sha = sha256_bytes(raw_bytes)
    html = raw_bytes.decode("utf-8", errors="replace")
    body, blocks, parser = extract_body(html)
    row = source_row()
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    recovered_dir = OUT / "body-recovery/recovered"
    recovered_dir.mkdir(parents=True, exist_ok=True)
    body_path = recovered_dir / f"{RECORD_ID}.txt"
    body_path.write_text(body, encoding="utf-8", newline="\n")
    body_sha = sha256_bytes(body.encode("utf-8"))
    recovered_at = receipt.get("retrieved_at_utc")
    old_body = str(row.get("body") or "")
    record = {
        "schema": "preannotation-v3-body-recovery-record-v1",
        "status": "current_retrieval_recovered_for_historical_research_only",
        "record_id": RECORD_ID,
        "source_url": SOURCE_URL,
        "source_domain": "www.cpfc.co.uk",
        "canonical_name_id": CANONICAL_NAME_ID,
        "home_team": "Brighton & Hove Albion",
        "away_team": "Crystal Palace",
        "kickoff": KICKOFF,
        "source_published_at": row.get("times", {}).get("source_published_at"),
        "source_updated_at": row.get("times", {}).get("source_updated_at"),
        "historical_archived_body_chars": len(old_body),
        "historical_archived_body_sha256": hashlib.sha256(old_body.encode("utf-8")).hexdigest(),
        "historical_archived_body_status": "suspected_1600_char_cap_mid_sentence",
        "raw_response_path": str(RAW),
        "raw_response_bytes": len(raw_bytes),
        "raw_response_sha256": raw_sha,
        "retrieved_at_utc": recovered_at,
        "retrieval_kind": "current_retrieval_after_original_archive",
        "source_available_at_at_old_cutoff": "UNKNOWN",
        "current_retrieval_proves_old_availability": False,
        "recovered_body_path": str(body_path),
        "recovered_body_chars": len(body),
        "recovered_body_sha256": body_sha,
        "paragraph_count": len(blocks),
        "blocks": blocks,
        "parser": parser,
        "body_used_for_f_or_g": False,
        "f_admission": "research_only_until_manual_semantic_and_time_review",
        "g_label_eligible": False,
        "exclusion_reason": "No pre-kickoff source_available_at clock; current retrieval cannot establish prospective availability or empirical gain label.",
        "receipt": receipt,
    }
    write_json(recovered_dir / f"{RECORD_ID}.json", record)
    audit = {
        "schema": "preannotation-v3-body-recovery-audit-v1",
        "status": "partial_recovery_one_of_two_capped_bodies",
        "recovery_run_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "recovered": [record],
        "unrecovered": [{
            "record_id": "ur1-efe647982f41ba4a345be862",
            "source_url": "https://www.cpfc.co.uk/news/match-previews/the-opposite-view-hurzeler-on-sage-palace-rivalry/",
            "status": "tls_failed_on_curl_probe",
            "archived_body_status": "suspected_1600_char_cap_mid_sentence",
            "body_used": False,
        }],
        "rules": [
            "Current retrieval is not retroactive evidence of source availability before the historical cutoff.",
            "Recovered body remains outside f/g until semantic review and time eligibility are separately established.",
            "Raw HTML and normalized body are kept with independent SHA-256 values.",
        ],
    }
    write_json(OUT / "body-recovery/recovery-audit.json", audit)
    # Add the latest recovery attempt to the existing network ledger without
    # erasing the earlier probe observations.
    probe_path = OUT / "network-probe.json"
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    probe["status"] = "probe_and_partial_raw_recovery_no_model_call"
    probe.setdefault("attempts", []).append({
        "source_url": SOURCE_URL,
        "method": "GET",
        "observed_status": 200,
        "response_saved": True,
        "response_path": str(RAW),
        "response_bytes": len(raw_bytes),
        "raw_sha256": raw_sha,
        "body_used": False,
        "retrieved_at_utc": recovered_at,
        "note": "Saved after an earlier TLS-failed probe; this is a current retrieval and does not establish old cutoff availability.",
    })
    probe["recovery_summary"] = {
        "recovered_records": 1,
        "unrecovered_records": 1,
        "f_or_g_calls": 0,
        "training_rows": 0,
    }
    write_json(probe_path, probe)
    readme = OUT / "README.md"
    existing = readme.read_text(encoding="utf-8")
    addendum = """

## Body recovery addendum (2026-10-24)

The saved CPFC response for `ur1-4cd71f9f33133a22c7448145` was parsed locally. The archived 1,600-character record was truncated mid-sentence; the current response contains 21 normalized textual blocks (paragraphs and one quote) and is saved under `body-recovery/recovered/`. The current retrieval is a historical reconstruction aid only: its clock is 2026-10-24 and it does not prove that the complete body was available before the 2026-10-18 cutoff. The body has not been sent to f/g and does not create a label. The second Palace article remains unrecovered after a TLS failure.
"""
    if "## Body recovery addendum" not in existing:
        readme.write_text(existing.rstrip() + addendum, encoding="utf-8")
    # Rebuild package manifest after adding recovery files and updated ledger.
    files = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files.append({"path": str(path.relative_to(OUT)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256_bytes(path.read_bytes())})
    write_json(OUT / "manifest.json", {"schema": "preannotation-v3-canonical-candidate-audit-manifest-v1", "files": files})
    print(json.dumps({"record_id": RECORD_ID, "raw_sha256": raw_sha, "body_sha256": body_sha, "body_chars": len(body), "blocks": len(blocks), "manifest_files": len(files)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
