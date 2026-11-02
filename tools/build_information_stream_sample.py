"""Build the first FootballMind information-stream sample package.

The input corpus is an existing public-source archive.  This builder copies
the original response bytes into a new, self-contained package and records
the source capture metadata without changing historical availability into a
prospective timestamp.  It deliberately does not derive probabilities,
VoI labels, or structured injury/odds values from article prose.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SELECTED_SOURCE_INDICES = [0, 1, 3, 4, 6]
LEAGUE = "Premier League"
SEASON = "2025"
SOURCE_PUBLIC_ARTICLE_HOST = "https://www.leedsunited.com"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_time(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if "T" not in text and " " in text:
        text = text.replace(" ", "T", 1)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return text
    if parsed.tzinfo is None:
        return text
    return parsed.isoformat()


def parse_time(value: Any) -> datetime | None:
    normalized = normalize_time(value)
    if not normalized:
        return None
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed


def source_relative_path(source_root: Path, raw_path: str) -> Path:
    """Resolve a corpus path without trusting the mojibake drive prefix."""
    normalized = str(raw_path).replace("\\", "/")
    marker = "/corpus-probe/"
    if marker in normalized:
        relative = normalized.split(marker, 1)[1]
    else:
        relative = normalized
        while relative.startswith("/"):
            relative = relative[1:]
        if relative.startswith("raw/") or relative.startswith("interviews-routed/"):
            pass
        else:
            raise ValueError(f"cannot resolve corpus path: {raw_path}")
    source = (source_root / Path(relative)).resolve()
    source.relative_to(source_root.resolve())
    return source


def slug(text: str, fallback: str = "item") -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return value[:80] or fallback


def copy_bytes(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return sha256(destination)


def read_raw_capture_metadata(source_root: Path, doc: dict[str, Any]) -> dict[str, Any]:
    """Return the real request/response receipt for one archived document."""
    raw_source = source_relative_path(source_root, doc["raw_path"])
    raw_rel = raw_source.relative_to(source_root)
    if str(raw_rel).replace("\\", "/").startswith("interviews-routed/"):
        receipt_path = raw_source.parent / "return.json"
        receipt = read_json(receipt_path)
        observation = receipt.get("observation") or {}
        document = receipt.get("document") or {}
        context = observation.get("context") or {}
        requested_at = context.get("requested_at")
        response_at = observation.get("observed_at") or document.get("observed_at") or doc.get("captured_at")
        status_code = observation.get("http_status") or document.get("http_status")
        error_type = observation.get("error_class") or document.get("error_class")
        latency_ms = receipt.get("latency_seconds")
        if isinstance(latency_ms, (int, float)):
            latency_ms = round(latency_ms * 1000, 3)
        return {
            "requested_at": normalize_time(requested_at),
            "response_at": normalize_time(response_at),
            "status_code": status_code,
            "error_type": error_type,
            "latency_ms": latency_ms,
            "request_count": receipt.get("http_attempts") or 1,
            "capture_metadata_source": str(receipt_path.relative_to(source_root)).replace("\\", "/"),
            "capture_metadata": receipt,
            "available_at_capture": normalize_time(document.get("available_at")),
        }
    sidecar_path = raw_source.with_suffix(".json")
    sidecar = read_json(sidecar_path) if sidecar_path.is_file() else {}
    requested_at = sidecar.get("requested_at")
    response_at = sidecar.get("captured_at") or doc.get("captured_at")
    requested_dt = parse_time(requested_at)
    response_dt = parse_time(response_at)
    latency_ms = None
    if requested_dt and response_dt:
        latency_ms = round((response_dt - requested_dt).total_seconds() * 1000, 3)
    return {
        "requested_at": normalize_time(requested_at),
        "response_at": normalize_time(response_at),
        "status_code": sidecar.get("status"),
        "error_type": sidecar.get("error_type"),
        "latency_ms": latency_ms,
        "request_count": sidecar.get("request_count") or 1,
        "capture_metadata_source": str(sidecar_path.relative_to(source_root)).replace("\\", "/") if sidecar_path.is_file() else None,
        "capture_metadata": sidecar,
        "available_at_capture": None,
    }


def page_from_discovery(document: dict[str, Any]) -> str:
    discovery = document.get("discovery") or {}
    url = discovery.get("parent_url") or discovery.get("discovery_url") or ""
    match = re.search(r"[?&]page=(\d+)", url)
    if not match:
        raise ValueError(f"discovery page missing for {document.get('url')}: {url}")
    return match.group(1)


def result_source_url(source_id: str) -> str:
    return f"https://understat.com/match/{source_id}"


def build_package(workspace: Path, output: Path, selected_indices: list[int] | None = None) -> dict[str, Any]:
    source_root = (workspace / "var/submission/p1-single-match/corpus-probe").resolve()
    cases = read_json(source_root / "cases-with-interviews-routed.json")
    if selected_indices is None:
        selected_indices = list(SELECTED_SOURCE_INDICES)
    selected = [cases[index] for index in selected_indices]
    result_export = workspace / "var/parallel/collectors/round2/exports/understat-2025-historical_reconstruction-a3b3d45811fd5cda/matches.jsonl"
    result_lines: dict[str, tuple[dict[str, Any], bytes]] = {}
    with result_export.open("rb") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            row = json.loads(raw_line.decode("utf-8"))
            source_id = str(row.get("source_id"))
            result_lines[source_id] = (row, raw_line)

    output.mkdir(parents=True, exist_ok=True)
    (output / "raw/documents").mkdir(parents=True, exist_ok=True)
    (output / "raw/discovery").mkdir(parents=True, exist_ok=True)
    (output / "raw/results").mkdir(parents=True, exist_ok=True)
    (output / "labels").mkdir(parents=True, exist_ok=True)

    fixtures: list[dict[str, Any]] = []
    documents: list[dict[str, Any]] = []
    discoveries: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    contexts: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    raw_provenance: list[dict[str, Any]] = []
    doc_ids_by_url: dict[str, str] = {}
    page_entries_cache: dict[str, dict[str, Any]] = {}
    discovery_ids: dict[tuple[str, str], str] = {}
    candidate_ids: dict[tuple[str, str, str], str] = {}
    docs_by_match: dict[str, list[tuple[dict[str, Any], str, dict[str, Any]]]] = {}

    for source_case_index, case in zip(selected_indices, selected):
        fixture = case["fixture"]
        mid = str(fixture["physical_match_id"])
        source_id = mid.split(":", 1)[-1]
        kickoff = normalize_time(fixture.get("kickoff"))
        fixtures.append({
            "physical_match_id": mid,
            "competition": fixture.get("competition_name") or LEAGUE,
            "season": fixture.get("season") or SEASON,
            "home_team": fixture.get("home_name"),
            "away_team": fixture.get("away_name"),
            "kickoff": kickoff,
            "kickoff_timezone": "UTC",
            "identity_source_url": result_source_url(source_id),
            "identity_source_basis": "existing Understat fixture identity in source corpus",
            "source_corpus_index": source_case_index,
        })

        docs_for_match: list[tuple[dict[str, Any], str, dict[str, Any]]] = []
        selected_urls = {doc.get("url") for doc in case.get("documents", [])}
        for doc_index, doc in enumerate(case.get("documents", [])):
            doc_id = f"{mid}:document:{doc_index + 1:02d}"
            raw_source = source_relative_path(source_root, doc["raw_path"])
            destination_name = f"{source_id}_{doc_index + 1:02d}_{slug(doc.get('title') or 'document')}.html"
            destination = output / "raw/documents" / destination_name
            digest = copy_bytes(raw_source, destination)
            if doc.get("raw_sha256") and digest.lower() != str(doc["raw_sha256"]).lower():
                raise ValueError(f"source/raw hash mismatch for {doc_id}: {raw_source}")
            capture = read_raw_capture_metadata(source_root, doc)
            capture_metadata_path = destination.with_suffix(".capture.json")
            write_json(capture_metadata_path, {
                "source_raw_path": str(raw_source),
                "source_metadata_path": capture.get("capture_metadata_source"),
                "source_metadata": capture.get("capture_metadata"),
                "copied_raw_sha256": digest,
            })
            discovery = doc.get("discovery") or {}
            reconstructed_available = discovery.get("available_at")
            timing_basis = (
                f"{doc.get('source_time_basis') or 'source publication metadata'}; "
                "source publication time preserved; historical availability and original menu not proven; "
                "body captured later from public archive"
            )
            fixture_binding_basis = doc.get("fixture_binding_basis")
            if not fixture_binding_basis:
                # The source corpus did not attach a separate binding rationale to
                # these Preview rows.  Preserve the fact that the binding is a
                # derived, title-level identity match instead of leaving the
                # required field silently null.
                fixture_binding_basis = (
                    "official_preview_title_exact_fixture_pair; "
                    "derived_from_saved_title_and_source_fixture_identity"
                )
            document_row = {
                "document_id": doc_id,
                "version_id": f"{doc_id}:v1",
                "physical_match_id": mid,
                "url": doc.get("url"),
                "publisher": "Leeds United",
                "title": doc.get("title"),
                "body": doc.get("text"),
                "language": "en",
                "source_published_at": normalize_time(doc.get("source_published_at")),
                "source_modified_at": normalize_time(doc.get("source_modified_at")),
                "captured_at": normalize_time(doc.get("captured_at")),
                "available_at": None,
                "original_time_text": doc.get("source_published_at"),
                "timing_basis": timing_basis,
                "information_mode": "historical_reconstruction",
                "raw_path": str(destination.relative_to(output)).replace("\\", "/"),
                "raw_sha256": digest,
                "previous_version_id": None,
                "document_kind": doc.get("document_kind"),
                "fixture_binding_basis": fixture_binding_basis,
                "source_time_basis": doc.get("source_time_basis"),
                "published_precision": doc.get("published_precision"),
                "historical_snapshot_proven": False,
                "later_body_revision_possible": doc.get("later_body_revision_possible"),
                "source_availability_text": reconstructed_available,
                "source_availability_basis": (discovery.get("availability_assumption")
                                               or "no historical availability proof; preserved only as source metadata"),
                "source_capture_metadata_path": str(capture_metadata_path.relative_to(output)).replace("\\", "/"),
                "source_corpus_raw_path": doc.get("raw_path"),
                "source_corpus_index": source_case_index,
            }
            documents.append(document_row)
            doc_ids_by_url[doc["url"]] = doc_id
            docs_for_match.append((doc, doc_id, capture))
            raw_provenance.append({
                "kind": "document_body",
                "document_id": doc_id,
                "source_path": doc.get("raw_path"),
                "package_path": document_row["raw_path"],
                "sha256": digest,
            })
        docs_by_match[mid] = docs_for_match

        pages = sorted({page_from_discovery(doc) for doc, _, _ in docs_for_match}, key=int)
        for page in pages:
            listing_path = source_root / f"listing-{page}.json"
            listing = read_json(listing_path)
            page_entries_cache[page] = listing
            capture = listing.get("capture") or {}
            source_listing_raw = source_relative_path(source_root, capture["raw_path"])
            destination_listing = output / "raw/discovery" / f"listing-{page}.html"
            listing_digest = copy_bytes(source_listing_raw, destination_listing)
            if capture.get("raw_sha256") and listing_digest.lower() != str(capture["raw_sha256"]).lower():
                raise ValueError(f"listing/raw hash mismatch for page {page}")
            package_listing_json = output / "raw/discovery" / f"listing-{page}.json"
            copy_bytes(listing_path, package_listing_json)
            discovery_id = f"{mid}:discovery:mens-page-{page}"
            discovery_ids[(mid, page)] = discovery_id
            visible_candidates: list[dict[str, Any]] = []
            selected_urls_for_page = {doc.get("url") for doc, _, _ in docs_for_match if page_from_discovery(doc) == page}
            for rank, entry in enumerate(listing.get("entries") or [], start=1):
                candidate_id = f"{mid}:candidate:page-{page}:{rank:03d}"
                candidate_ids[(mid, page, entry.get("url"))] = candidate_id
                visible_candidates.append({
                    "candidate_id": candidate_id,
                    "url": entry.get("url"),
                    "title": entry.get("title"),
                    "snippet": entry.get("date_label") or "",
                    "rank": rank,
                    "selected_by_collector": entry.get("url") in selected_urls_for_page,
                    "snippet_basis": "directory date_label; no search-excerpt text captured",
                })
            discovery_row = {
                "discovery_id": discovery_id,
                "physical_match_id": mid,
                "observed_at": normalize_time(capture.get("captured_at")),
                "query_or_parent": capture.get("requested_url"),
                "provider": "Leeds United official men's news directory",
                "raw_path": str(destination_listing.relative_to(output)).replace("\\", "/"),
                "raw_sha256": listing_digest,
                "visible_candidates": visible_candidates,
                "status": "reconstructed_menu",
                "menu_status": "historical_reconstruction_from_posthoc_official_listing",
                "historical_snapshot_proven": False,
                "original_agent_menu_proven": False,
                "candidate_snippet_basis": "The source archive retained directory cards and date labels; it did not retain a contemporaneous agent/search response.",
                "source_listing_json_raw_path": str(package_listing_json.relative_to(output)).replace("\\", "/"),
                "source_listing_url": capture.get("requested_url"),
                "source_listing_requested_at": normalize_time(capture.get("requested_at")),
                "source_listing_response_url": capture.get("response_url"),
                "source_listing_status": capture.get("status"),
            }
            discoveries.append(discovery_row)
            raw_provenance.extend([
                {"kind": "discovery_html", "discovery_id": discovery_id, "source_path": capture.get("raw_path"), "package_path": discovery_row["raw_path"], "sha256": listing_digest},
                {"kind": "discovery_json", "discovery_id": discovery_id, "source_path": str(listing_path.relative_to(source_root)).replace("\\", "/"), "package_path": discovery_row["source_listing_json_raw_path"], "sha256": sha256(package_listing_json)},
            ])

        for doc, doc_id, capture in docs_for_match:
            page = page_from_discovery(doc)
            discovery_id = discovery_ids[(mid, page)]
            candidate_id = candidate_ids.get((mid, page, doc.get("url")))
            status_code = capture.get("status_code")
            error_type = capture.get("error_type")
            successful = isinstance(status_code, int) and 200 <= status_code < 300 and not error_type
            attempt_row = {
                "attempt_id": f"{mid}:attempt:{len(attempts) + 1:03d}",
                "physical_match_id": mid,
                "discovery_id": discovery_id,
                "candidate_id": candidate_id,
                "requested_at": capture.get("requested_at"),
                "response_at": capture.get("response_at"),
                "status_code": status_code,
                "error_type": error_type,
                "returned_document_id": doc_id if successful else None,
                "latency_ms": capture.get("latency_ms"),
                "request_count": capture.get("request_count"),
                "token_cost": None,
                "monetary_cost": None,
                "status": "returned" if successful else ("error" if error_type else "no_result"),
                "action_type": "open_candidate",
                "request_url": doc.get("url"),
                "response_source": "public official article capture",
                "selection_basis": "candidate URL equals returned document URL; no usefulness/VoI judgment recorded",
                "capture_metadata_path": capture.get("capture_metadata_source"),
                "historical_capture": False,
            }
            attempts.append(attempt_row)

        actual_captures = [parse_time(capture.get("response_at")) for _, _, capture in docs_for_match]
        actual_captures = [value for value in actual_captures if value]
        context_observed_at = min(actual_captures).isoformat() if actual_captures else None
        source_doc_ids = [doc_id for _, doc_id, _ in docs_for_match]
        source_urls = [doc.get("url") for doc, _, _ in docs_for_match]
        contexts.append({
            "physical_match_id": mid,
            "snapshot_id": f"{mid}:context:sample",
            "snapshot_observed_at": context_observed_at,
            "snapshot_timing_basis": "earliest successful article retrieval in this reconstructed package; not a historical live-state snapshot",
            "schedule": {
                "value": {"home_team": fixture.get("home_name"), "away_team": fixture.get("away_name"), "kickoff": kickoff},
                "observed_at": context_observed_at,
                "source_document_ids": source_doc_ids,
                "source_urls": source_urls,
                "status": "fixture_identity_only",
            },
            "ranking": {"value": None, "observed_at": context_observed_at, "status": "not_structured_in_sample", "source_document_ids": source_doc_ids},
            "points": {"value": None, "observed_at": context_observed_at, "status": "not_structured_in_sample", "source_document_ids": source_doc_ids},
            "qualification_status": {"value": None, "observed_at": context_observed_at, "status": "not_structured_in_sample", "source_document_ids": source_doc_ids},
            "injuries": {"value": None, "observed_at": context_observed_at, "status": "raw_mentions_preserved_in_documents_only", "source_document_ids": source_doc_ids},
            "lineup_status": {"value": None, "observed_at": context_observed_at, "status": "no_official_lineup_in_selected_documents", "source_document_ids": source_doc_ids},
            # Keep the field present while making the absence of a quote explicit.
            # quote_time/captured_at are only meaningful when an odds quote exists;
            # the separate *_observed_at/status fields record the null observation.
            "odds": None,
            "odds_observed_at": context_observed_at,
            "odds_quote_time": None,
            "odds_captured_at": None,
            "odds_status": "not_collected_in_source_corpus",
            "weather": {"value": None, "observed_at": context_observed_at, "status": "not_collected_in_source_corpus", "source_document_ids": source_doc_ids},
            "raw_source_urls": source_urls,
            "information_mode": "historical_reconstruction",
        })

        result, raw_line = result_lines.get(source_id, ({}, b""))
        if not result:
            raise ValueError(f"result row missing from export for {mid}")
        result_raw_path = output / "raw/results" / f"{mid.replace(':', '_')}.json"
        result_raw_path.parent.mkdir(parents=True, exist_ok=True)
        result_raw_path.write_bytes(raw_line if raw_line.endswith(b"\n") else raw_line + b"\n")
        result_digest = sha256(result_raw_path)
        results.append({
            "physical_match_id": mid,
            "regulation_home_goals": result.get("home_goals"),
            "regulation_away_goals": result.get("away_goals"),
            "result_status": result.get("status"),
            "result_source_url": result_source_url(source_id),
            "result_published_at": None,
            "result_observed_at": normalize_time(result.get("observed_at")),
            "raw_path": str(result_raw_path.relative_to(output)).replace("\\", "/"),
            "raw_sha256": result_digest,
            "timing_basis": "source export observed_at retained as the actual result observation; result_available_at is an existing kickoff-plus-three-hours reconstruction and is not used as observed_at",
            "source_provider": result.get("source"),
            "source_id": result.get("source_id"),
            "score_period": result.get("score_period"),
            "source_result_available_at": normalize_time(result.get("result_available_at")),
            "source_observed_at_raw_text": result.get("observed_at"),
            "result_source_url_basis": "canonical provider URL from source=understat and source_id",
        })
        raw_provenance.append({
            "kind": "result_export_row",
            "physical_match_id": mid,
            "source_path": str(result_export),
            "package_path": results[-1]["raw_path"],
            "sha256": result_digest,
        })

    write_jsonl(output / "fixtures.jsonl", fixtures)
    write_jsonl(output / "documents.jsonl", documents)
    write_jsonl(output / "discovery.jsonl", discoveries)
    write_jsonl(output / "attempts.jsonl", attempts)
    write_jsonl(output / "context_snapshots.jsonl", contexts)
    write_jsonl(output / "labels/results.jsonl", results)
    write_json(output / "raw/raw-provenance.json", raw_provenance)

    publication_times = [parse_time(doc.get("source_published_at")) for doc in documents]
    capture_times = [parse_time(doc.get("captured_at")) for doc in documents]
    menu_times = [parse_time(row.get("observed_at")) for row in discoveries]
    result_times = [parse_time(row.get("result_observed_at")) for row in results]
    actual_capture_times = [value for value in capture_times + menu_times + result_times if value]
    clubs = sorted({row["home_team"] for row in fixtures} | {row["away_team"] for row in fixtures})
    kickoff_by_match = {row["physical_match_id"]: parse_time(row["kickoff"]) for row in fixtures}
    pre_match_window_counts = {"within_72h": 0, "between_72h_and_7d": 0, "older_than_7d": 0, "unknown": 0}
    for document in documents:
        published = parse_time(document.get("source_published_at"))
        kickoff_time = kickoff_by_match.get(document.get("physical_match_id"))
        if not published or not kickoff_time:
            pre_match_window_counts["unknown"] += 1
            continue
        hours_before = (kickoff_time - published).total_seconds() / 3600
        if 0 <= hours_before <= 72:
            pre_match_window_counts["within_72h"] += 1
        elif 72 < hours_before <= 24 * 7:
            pre_match_window_counts["between_72h_and_7d"] += 1
        else:
            pre_match_window_counts["older_than_7d"] += 1

    manifest = {
        "schema_version": "footballmind-information-stream/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Raw pre-match information stream sample for autonomous information-value learning data collection.",
        "scope": {
            "competition": LEAGUE,
            "season_code": SEASON,
            "sample_size": len(fixtures),
            "source_corpus": "var/submission/p1-single-match/corpus-probe/cases-with-interviews-routed.json",
            "source_public_article_host": SOURCE_PUBLIC_ARTICLE_HOST,
            "result_source": "Understat export source=understat, score_period=regulation",
        },
        "selection_rule": {
            "universe": "Pre-existing chronological source-corpus cases with explicit fixture binding and official prematch documents.",
            "filter": "Eligible historical-reconstruction cases with at least one explicitly bound official prematch document; no filtering by upset status, forecast correctness, or later usefulness.",
            "coverage_constraint": "The expanded package includes every eligible case in the pre-existing chronological source corpus for the selected date range.",
            "selected_source_indices": selected_indices,
            "source_universe_size": len(cases),
            "outcome_fields_used_for_inclusion": False,
        },
        "coverage": {
            "physical_match_ids": [row["physical_match_id"] for row in fixtures],
            "clubs": clubs,
            "home_clubs": sorted({row["home_team"] for row in fixtures}),
            "away_clubs": sorted({row["away_team"] for row in fixtures}),
            "documents": len(documents),
            "discoveries": len(discoveries),
            "attempts": len(attempts),
            "contexts": len(contexts),
            "results": len(results),
        },
        "time_window": {
            "requested_pre_match_window": "T-72h through kickoff, with wider source publication coverage when available",
            "source_publication_start": min((x for x in publication_times if x), default=None).isoformat() if any(publication_times) else None,
            "source_publication_end": max((x for x in publication_times if x), default=None).isoformat() if any(publication_times) else None,
            "actual_capture_timestamp_start": min(actual_capture_times).isoformat() if actual_capture_times else None,
            "actual_capture_timestamp_end": max(actual_capture_times).isoformat() if actual_capture_times else None,
            "document_publication_relative_to_kickoff": pre_match_window_counts,
        },
        "information_modes": {
            "prospective_capture": 0,
            "archived_snapshot": 0,
            "historical_reconstruction": len(documents),
            "discovery_menu_mode": "historical_reconstruction",
        },
        "historical_reconstruction_disclosure": {
            "documents": "The article publication timestamps come from source metadata, while body captures were made later in the public archive. available_at remains null because contemporaneous historical availability is not proven.",
            "discoveries": "The visible candidate menus are reconstructed from later official Leeds United directory captures. They are not claimed to be the menu originally shown to an historical Agent.",
            "attempts": "Attempts record the real later HTTP request and response receipts used to obtain the archived bodies and directories. They are not prospective pre-match requests.",
            "results": "The result observation timestamp is the actual timestamp in the Understat export; the export's kickoff-plus-three-hours result_available_at is retained as source metadata only and is not used as result_observed_at.",
        },
        "absence_accounting": {
            "failed_requests_in_selected_receipts": "none recorded",
            "empty_results_in_selected_receipts": "none recorded",
            "duplicate_document_attempts_in_selected_receipts": "none recorded",
            "contradictory_results_in_selected_receipts": "none recorded",
            "policy": "No failure, empty, duplicate, or contradiction rows were synthesized. If a source receipt did not contain such an event, the count remains zero; unknown cost fields remain null.",
        },
        "raw_provenance_file": "raw/raw-provenance.json",
        "generated_by": "tools/build_information_stream_sample.py",
        "integrity": {
            "hash_algorithm": "SHA-256",
            "manifest_self_hash_omitted": True,
            "file_hash_scope": "All package JSONL files and raw copied bytes except manifest.json and quality-report.txt.",
        },
        "restrictions": [
            "No VoI labels",
            "No probabilities",
            "No training of g",
            "No result-based document selection",
            "No fabricated timestamps, menus, request costs, or body text",
        ],
    }
    # Hash declarations must be produced after every non-manifest package file is written.
    file_hashes: dict[str, str] = {}
    for path in sorted(output.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(output).as_posix()
        if rel in {"manifest.json", "quality-report.txt"}:
            continue
        file_hashes[rel] = sha256(path)
    manifest["file_hashes"] = file_hashes
    write_json(output / "manifest.json", manifest)
    return {
        "fixtures": fixtures,
        "documents": documents,
        "discoveries": discoveries,
        "attempts": attempts,
        "contexts": contexts,
        "results": results,
        "manifest": manifest,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("footballmind-information-stream"))
    parser.add_argument("--all-cases", action="store_true", help="include every eligible case in the source corpus")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    output = args.output if args.output.is_absolute() else workspace / args.output
    selected_indices = None
    if args.all_cases:
        source_path = workspace / "var/submission/p1-single-match/corpus-probe/cases-with-interviews-routed.json"
        selected_indices = list(range(len(read_json(source_path))))
    summary = build_package(workspace, output.resolve(), selected_indices=selected_indices)
    print(json.dumps({
        "output": str(output.resolve()),
        "fixtures": len(summary["fixtures"]),
        "documents": len(summary["documents"]),
        "discoveries": len(summary["discoveries"]),
        "attempts": len(summary["attempts"]),
        "contexts": len(summary["contexts"]),
        "results": len(summary["results"]),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
