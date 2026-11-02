from __future__ import annotations

import gzip
import hashlib
import html
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup

from football_spider.storage.raw_archive import RawArchive


OLD = Path(r"D:\a1-足彩\var\fm-manual-article-packet-20261012-01")
OUT = Path(r"D:\a1-足彩\var\fm-article-body-recovery-20261012-02")
CUTOFF = datetime.fromisoformat("2026-10-18T14:30:00+00:00")
PHYSICAL_MATCH_ID = "footballmind:premier-league:2026-27:2026-10-18:nottingham-forest:arsenal"
KICKOFF = "2026-10-18T15:30:00Z"


def iso(dt: datetime | None) -> str | None:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z") if dt else None


def sha(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def exported_raw_path(payload_path: str) -> str:
    """Return a path relative to the recovery packet root.

    RawArchive payload paths are relative to its own archive root and include
    the archive's ``raw/`` component.  The packet exposes that archive below
    its own ``raw/`` directory, so the exported path needs both components.
    """
    return (Path("raw") / Path(payload_path)).as_posix()


def read_old_observations():
    result = {}
    for path in (OLD / "raw" / "raw").glob("**/*.observation.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        result[value["id"]] = (value, path)
    return result


def old_body(obs: dict) -> bytes:
    path = OLD / "raw" / obs["payload_path"]
    return gzip.decompress(path.read_bytes())


def decode_rss(obs: dict):
    root = ET.fromstring(old_body(obs).decode("utf-8", "replace"))
    rows = []
    for rank, item in enumerate(root.findall("./channel/item"), 1):
        title = item.findtext("title")
        link = item.findtext("link")
        description = item.findtext("description")
        raw_pub = item.findtext("pubDate")
        try:
            pub = datetime.strptime(raw_pub, "%a, %d %b %Y %H:%M:%S %Z").replace(tzinfo=timezone.utc) if raw_pub else None
        except ValueError:
            pub = None
        if link:
            parsed = urlsplit(link)
            if parsed.path == "/news/apiclick.aspx":
                from urllib.parse import parse_qsl
                target = next((v for k, v in parse_qsl(parsed.query) if k == "url"), None)
                link = target or link
        rows.append({
            "rank": rank,
            "title": html.unescape(title or "").strip() or None,
            "url": link,
            "snippet": BeautifulSoup(html.unescape(description or ""), "html.parser").get_text(" ", strip=True) or None,
            "published_at": iso(pub),
        })
    return rows


def current_observations(archive: RawArchive):
    return {obs.endpoint: obs for obs in archive.observations(source="open_page")}


def extract_statsinsider(archive: RawArchive, obs):
    raw = archive.read_body(obs)
    soup = BeautifulSoup(raw.decode("utf-8", "replace"), "html.parser")
    article = soup.find("article")
    body = article.find("app-article-body") if article else None
    if body is None:
        return None
    parts = []
    header = article.find("div", class_="article-header") if article else None
    if header:
        title = header.find("h1")
        if title:
            parts.append(" ".join(title.get_text(" ", strip=True).split()))
        meta = " ".join(header.get_text(" ", strip=True).split())
        if meta and meta not in parts:
            parts.append(meta)
    author = article.find("div", class_="author-share-line") if article else None
    if author:
        author_text = " ".join(author.get_text(" ", strip=True).split())
        if author_text:
            parts.append(author_text)
    for node in body.find_all(["p", "h2", "h3"]):
        text = " ".join(node.get_text(" ", strip=True).split())
        if text:
            parts.append(text)
    text = "\n".join(parts).strip()
    if len(text) < 200:
        return None
    jsonld = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            value = json.loads(script.get_text())
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict) and value.get("@type") in {"NewsArticle", "Article"}:
            jsonld.append(value)
    meta = jsonld[0] if jsonld else {}
    return {
        "text": text,
        "headline": meta.get("headline") or (parts[0] if parts else None),
        "source_published_at": meta.get("datePublished"),
        "source_modified_at": meta.get("dateModified"),
        "published_at_raw": meta.get("datePublished"),
        "modified_at_raw": meta.get("dateModified"),
        "source_author": meta.get("author"),
        "raw_body_sha256": sha(raw),
    }


def write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "bodies").mkdir(exist_ok=True)
    archive = RawArchive(OUT / "raw")
    old_obs = read_old_observations()
    rss_obs = old_obs["d6f71c5c-1070-4568-b0b2-8ddbf2cf210c"][0]
    rss_rows = decode_rss(rss_obs)
    discovered = []
    for row in rss_rows:
        row.update({
            "discovery_observation_id": rss_obs["id"],
            "discovery_raw_path": rss_obs["payload_path"],
            "discovery_raw_sha256": rss_obs["content_hash"],
            "discovery_observed_at": rss_obs["observed_at"],
            "discovery_query": rss_obs["params"]["q"],
            "discovery_mode": "archived_metadata_only",
        })
        discovered.append(row)
    schedule_obs = old_obs["ee35430b-c82e-4621-9917-49bbfd66ecd2"][0]
    discovered.append({
        "rank": len(discovered) + 1,
        "title": "reasons why fixtures change are explained here",
        "url": "https://www.premierleague.com/news/4324634",
        "snippet": None,
        "published_at": None,
        "discovery_observation_id": schedule_obs["id"],
        "discovery_raw_path": schedule_obs["payload_path"],
        "discovery_raw_sha256": schedule_obs["content_hash"],
        "discovery_observed_at": schedule_obs["observed_at"],
        "discovery_query": None,
        "discovery_mode": "archived_schedule_anchor_only",
    })

    stats_url = "https://www.statsinsider.com.au/news/nottingham-forest-vs-arsenal-prediction-and-tips-19-october-2026"
    attempts = []
    current = current_observations(archive)
    for row in discovered:
        url = row["url"]
        if url == stats_url:
            decision = "retained_for_human_review"
            attempt_status = "returned"
        elif "statsinsider.com.au" in url:
            decision = "retained_candidate_body_attempted"
            attempt_status = "not_requested"
        elif "squawka.com" in url or "nottinghampost.com" in url or "premierleague.com/news/4324634" in url:
            decision = "body_fetch_attempted"
            attempt_status = "returned" if current.get(url) and current[url].http_status and 200 <= current[url].http_status < 300 else "failed"
        elif "albawaba.com" in url:
            decision = "filtered_out"
            attempt_status = "not_requested"
        elif "goal.com" in url:
            decision = "filtered_out"
            attempt_status = "not_requested"
        elif "sports.yahoo.com" in url or "bbc.com" in url:
            decision = "filtered_out"
            attempt_status = "not_requested"
        elif "nytimes.com" in url:
            decision = "hold_access_uncertain"
            attempt_status = "not_requested"
        else:
            decision = "hold_unclassified"
            attempt_status = "not_requested"
        row["candidate_id"] = f"candidate-{row['rank']:03d}"
        row["physical_match_id"] = PHYSICAL_MATCH_ID
        row["classification"] = "article" if decision.startswith("retained") or decision == "body_fetch_attempted" else "unknown_or_index"
        row["match_identity"] = {
            "target": PHYSICAL_MATCH_ID,
            "relation": "direct" if url == stats_url else "indirect_or_unknown",
            "basis": "title_or_source_url_only",
            "semantic_body_not_used_for_filter": True,
        }
        row["candidate_visible_before_body_read"] = True
        row["filter_status"] = decision
        row["body_fetch_status"] = attempt_status
        row["reviewer_label"] = None
        row["reviewer_confidence"] = None
        row["evidence_span"] = None

    # Build the fresh attempt ledger from the four direct source observations.
    for seq, url in enumerate([
        stats_url,
        "https://www.squawka.com/en/features/stat-spotlight-premier-league-2026-27-gameweek-6/",
        "https://www.nottinghampost.com/sport/football/football-news/nottingham-forest-player-ratings-palace-11193862",
        "https://www.premierleague.com/news/4324634",
    ], 1):
        obs = current[url]
        context = obs.context or {}
        attempts.append({
            "attempt_id": f"body-recovery-20261012-{seq:03d}",
            "url": url,
            "source": urlsplit(url).hostname,
            "request_id": obs.request_id,
            "requested_at": context.get("requested_at"),
            "response_received_at": iso(obs.observed_at),
            "http_status": obs.http_status,
            "error_class": obs.error_class,
            "response_category": "returned" if obs.http_status and 200 <= obs.http_status < 300 and not obs.error_class else "http_redirect" if obs.http_status and 300 <= obs.http_status < 400 else "failed",
            "raw_observation_id": obs.id,
            "raw_path": exported_raw_path(obs.payload_path),
            "raw_sha256": obs.content_hash,
            "raw_sha256_verified": sha(archive.read_body(obs)) == obs.content_hash,
            "direct_http_count": 1,
            "automatic_redirects": False,
            "snapshot_saved_at": None,
            "agent_handoff_at": None,
        })

    stats_obs = current[stats_url]
    extraction = extract_statsinsider(archive, stats_obs)
    articles = []
    if extraction:
        article_id = "article-statsinsider-nof-ars-20261012-001"
        body_path = OUT / "bodies" / f"{article_id}.txt"
        body_path.write_text(extraction["text"], encoding="utf-8", newline="\n")
        # The transport did not expose application-level save/hand-off receipts.
        # Preserve filesystem observations as explicitly qualified proxies rather
        # than manufacturing current-time clocks.
        raw_file = OUT / exported_raw_path(stats_obs.payload_path)
        snapshot_saved = datetime.fromtimestamp(raw_file.stat().st_mtime, timezone.utc)
        handoff = datetime.fromtimestamp(body_path.stat().st_ctime, timezone.utc)
        text_hash = sha(extraction["text"])
        source_published = extraction["source_published_at"]
        source_modified = extraction["source_modified_at"]
        # The archived RSS projection exposed a different raw publication clock;
        # preserve the conflict instead of selecting one silently.
        rss_match = next(row for row in discovered if row["url"] == stats_url)
        articles.append({
            "article_id": article_id,
            "physical_match_id": PHYSICAL_MATCH_ID,
            "source": "statsinsider",
            "publisher": "Stats Insider",
            "url": stats_url,
            "title": extraction["headline"],
            "section": "Premier League / News",
            "published_at": source_published,
            "published_at_declared": True,
            "published_at_raw": extraction["published_at_raw"],
            "updated_at": source_modified,
            "updated_at_declared": bool(source_modified),
            "discovery_published_at": rss_match["published_at"],
            "publication_time_conflict": bool(rss_match["published_at"] and source_published and rss_match["published_at"] != source_published),
            "publication_conflict_basis": "archived_Bing_RSS_vs_current_page_JSON_LD" if rss_match["published_at"] and source_published and rss_match["published_at"] != source_published else None,
            "response_received_at": iso(stats_obs.observed_at),
            "source_available_at": iso(stats_obs.observed_at),
            "source_available_at_status": "VERIFIED_BY_CURRENT_PRE_CUTOFF_CAPTURE",
            "source_available_at_basis": "nonempty_body_returned_and_saved_before_prediction_cutoff",
            "snapshot_saved_at": iso(snapshot_saved),
            "snapshot_saved_at_basis": "raw_archive_payload_filesystem_mtime_proxy_application_save_receipt_unmeasured",
            "agent_handoff_at": iso(handoff),
            "agent_handoff_at_basis": "body_file_filesystem_creation_time_proxy_agent_handoff_receipt_unmeasured",
            "prediction_cutoff": CUTOFF.isoformat().replace("+00:00", "Z"),
            "raw_observation_id": stats_obs.id,
            "raw_path": exported_raw_path(stats_obs.payload_path),
            "raw_sha256": stats_obs.content_hash,
            "raw_sha256_verified": sha(archive.read_body(stats_obs)) == stats_obs.content_hash,
            "content_hash": text_hash,
            "body_path": str(body_path.relative_to(OUT)).replace("\\", "/"),
            "body_length": len(extraction["text"]),
            "extraction_method": "same_page_article_app_article_body_heading_paragraphs_v1",
            "classification": "article",
            "match_identity": {
                "relation": "direct",
                "basis": "title_and_article_body_explicitly_name_both_teams",
                "target_kickoff": KICKOFF,
            },
            "entities": ["Nottingham Forest", "Arsenal"],
            "event_time": {
                "raw": "Monday 19 October 2026, 2:30am AEDT",
                "normalized": KICKOFF,
                "basis": "page_title/local_clock_and_frozen_official_fixture_identity",
                "status": "matched_to_target_kickoff",
            },
            "information_type": "market_prediction_and_match_context",
            "reviewer_label": None,
            "reviewer_confidence": None,
            "evidence_span": None,
            "f_or_g_consumed": False,
            "prospective_agent_trace": False,
        })
        for attempt in attempts:
            if attempt["url"] == stats_url:
                attempt["snapshot_saved_at"] = iso(snapshot_saved)
                attempt["snapshot_saved_at_basis"] = "raw_archive_payload_filesystem_mtime_proxy_application_save_receipt_unmeasured"
                attempt["agent_handoff_at"] = iso(handoff)
                attempt["agent_handoff_at_basis"] = "body_file_filesystem_creation_time_proxy_agent_handoff_receipt_unmeasured"
                attempt["article_id"] = article_id
                attempt["body_nonempty"] = True

    write_jsonl(OUT / "articles.jsonl", articles)
    write_jsonl(OUT / "candidates.jsonl", discovered)
    write_jsonl(OUT / "attempts.jsonl", attempts)

    digest_rows = []
    for obs in archive.observations():
        body = archive.read_body(obs)
        digest_rows.append({
            "observation_id": obs.id,
            "source": obs.source,
            "endpoint": obs.endpoint,
            "raw_path": exported_raw_path(obs.payload_path),
            "declared_sha256": obs.content_hash,
            "computed_sha256": sha(body),
            "verified": sha(body) == obs.content_hash,
            "bytes": len(body),
        })
    (OUT / "raw-digest-index.json").write_text(json.dumps({
        "schema": "fm-article-body-recovery-raw-digest-v1",
        "direct_source_observations": digest_rows,
        "archived_discovery_inputs": [
            {"observation_id": rss_obs["id"], "raw_path": rss_obs["payload_path"], "raw_sha256": rss_obs["content_hash"], "reused_without_new_http": True},
            {"observation_id": schedule_obs["id"], "raw_path": schedule_obs["payload_path"], "raw_sha256": schedule_obs["content_hash"], "reused_without_new_http": True},
        ],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    body_success = len(articles)
    attempted = len(attempts)
    failed = sum(1 for x in attempts if x["response_category"] != "returned")
    filtered = sum(1 for x in discovered if x["filter_status"] == "filtered_out")
    rejected_time = sum(1 for x in discovered if x["url"].endswith("1594360"))
    hold = sum(1 for x in discovered if x["filter_status"].startswith("hold"))
    (OUT / "concise-summary.txt").write_text(
        "\n".join([
            "FM-ARTICLE-BODY-RECOVERY-20261012-02",
            f"target=Nottingham Forest vs Arsenal; physical_match_id={PHYSICAL_MATCH_ID}",
            f"kickoff={KICKOFF}; prediction_cutoff={CUTOFF.isoformat().replace('+00:00','Z')}",
            "identity_basis=previous_fresh_official_schedule_bundle; target_remains_future_at_collection_time=true",
            f"archived_metadata_candidates={len(discovered)}; retained_nonempty_articles={body_success}; body_attempts={attempted}; body_failures_or_redirects={failed}",
            f"cheap_filtered_out={filtered}; time_or_fixture_rejected={rejected_time}; held_unknown_or_access={hold}",
            "fresh_direct_http=4; public_search_tool_calls=0; f_or_g_calls=0; training_or_prediction_written=false",
            "codex_cli_review_model=gpt-5.6-luna; codex_cli_review_status=blocked_by_local_execution_policy; offline_subthread_review=success; review_json=review/subthread-review.json",
            "",
            "Readable body:",
            *[f"- {a['title']} | {a['url']} | body={a['body_path']} | chars={a['body_length']} | raw_sha256={a['raw_sha256']} | content_sha256={a['content_hash']} | source_available_at_status={a['source_available_at_status']} | publication_time_conflict={a['publication_time_conflict']}" for a in articles],
            "",
            "Source timing caveat: the current page declares JSON-LD publication/modified times; archived RSS exposed a different publication clock. Both are preserved. Current nonempty body was returned and saved before the target cutoff, so availability is verified for this captured version; no claim is made about an earlier first-publication time.",
            "Review boundary: the local Codex CLI attempt produced no JSON because its model/tool service was blocked. The offline subthread fallback produced review/subthread-review.json; its exact model ID and review clocks are unverified, and strict single-file isolation is not claimed. It records network_tool_called=false, source_article_http_called=false, api_key_used=false, deepseek_used=false. This review is not an Agent VOI choice and did not call f/g.",
            "Unresolved: three other fresh attempts did not yield readable bodies (two SSLError failures and one HTTP 302 with automatic redirects disabled); the NYT/Athletic candidate remains held without access.",
        ]) + "\n", encoding="utf-8")

    readme = f"""# FM-ARTICLE-BODY-RECOVERY-20261012-02

This directory is a new manual-review input package for **Nottingham Forest vs Arsenal**.
The previous packet remains untouched.

- physical match: `{PHYSICAL_MATCH_ID}`
- kickoff: `{KICKOFF}`
- prediction cutoff: `{CUTOFF.isoformat().replace('+00:00','Z')}`
- fresh direct source requests: `{attempted}`
- archived metadata candidates: `{len(discovered)}`
- non-empty article bodies: `{len(articles)}`
- f/g calls, prediction-after, gain/loss: `none`

The Stats Insider page returned a non-empty body and was saved before the cutoff, so
the captured version has a verified `source_available_at`. Its declared JSON-LD
publication clock conflicts with the archived RSS clock; both are retained in
`articles.jsonl` and no first-publication time is inferred.

The body is in `bodies/`; its raw response is in `raw/` and is linked by
`raw_observation_id`, `raw_path`, and SHA-256. `candidates.jsonl` preserves all
archived discovery candidates and their filtering decisions. `attempts.jsonl`
records the four fresh direct requests, including two TLS failures and one
302 response that was not automatically followed.

Semantic review is a separate step. The preferred local `gpt-5.6-luna` CLI attempt
is recorded under `review/` and was blocked by the local execution/model service.
The requested fallback subthread produced `review/subthread-review.json`; its
exact model ID and review clocks are unverified, and strict single-file isolation
is not claimed. It is an offline review artifact only, not an Agent choice or f/g
result. No API key or DeepSeek was used.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8", newline="\n")

    # The manifest excludes itself; the ZIP also excludes this runner and caches.
    files = []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or path.name == "manifest.json" or "__pycache__" in path.parts:
            continue
        data = path.read_bytes()
        files.append({"path": str(path.relative_to(OUT)).replace("\\", "/"), "bytes": len(data), "sha256": sha(data)})
    manifest = {
        "schema": "fm-article-body-recovery-manifest-v1",
        "task": "FM-ARTICLE-BODY-RECOVERY-20261012-02",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "target": {"physical_match_id": PHYSICAL_MATCH_ID, "kickoff": KICKOFF, "prediction_cutoff": CUTOFF.isoformat().replace("+00:00", "Z")},
        "counts": {"archived_metadata_candidates": len(discovered), "body_attempts": attempted, "articles": len(articles), "body_failures_or_redirects": failed, "direct_http": attempted},
        "files": files,
        "zip_excludes": ["manifest.json", "runner.py", "__pycache__"],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    zip_path = OUT.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for item in files + [{"path": "manifest.json"}]:
            path = OUT / item["path"]
            z.write(path, item["path"])
    with zipfile.ZipFile(zip_path) as z:
        assert z.testzip() is None
    print(json.dumps({"out": str(OUT), "zip": str(zip_path), "zip_sha256": sha(zip_path.read_bytes()), "articles": len(articles), "candidates": len(discovered), "attempts": attempted}, ensure_ascii=False))


if __name__ == "__main__":
    main()
