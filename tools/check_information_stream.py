"""Validate a FootballMind pre-match information stream and write a quality report.

The collector is intentionally kept separate from the model/evaluation code.  This
script only checks provenance, shape, timestamps, append-only document versions,
and capture coverage; it never derives probabilities or a value-of-information
label.  It accepts the JSONL layout described in
``docs/research/footballmind-g-data-collection-prompt.txt`` and the narrower
schema in the collection task (for example ``kickoff`` and ``body``).

Usage::

    python tools/check_information_stream.py path/to/footballmind-information-stream

The quality report is written to ``quality-report.txt`` unless ``--report`` is
provided.  A non-zero exit status is returned when hard schema/provenance errors
are found; warnings (for example a missing optional context field) are preserved
in the report but do not by themselves fail the check.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable


MODES = {"prospective_capture", "archived_snapshot", "historical_reconstruction"}
REQUIRED_FILES = (
    "manifest.json",
    "fixtures.jsonl",
    "documents.jsonl",
    "discovery.jsonl",
    "attempts.jsonl",
    "context_snapshots.jsonl",
    "labels/results.jsonl",
)
FIXTURE_FIELDS = (
    "physical_match_id", "competition", "season", "home_team", "away_team",
    "kickoff", "kickoff_timezone", "identity_source_url",
)
DOCUMENT_FIELDS = (
    "document_id", "physical_match_id", "url", "publisher", "title", "body",
    "language", "source_published_at", "source_modified_at", "captured_at",
    "available_at", "original_time_text", "timing_basis", "information_mode",
    "raw_path", "raw_sha256", "previous_version_id", "document_kind",
    "fixture_binding_basis",
)
DISCOVERY_FIELDS = (
    "discovery_id", "physical_match_id", "observed_at", "query_or_parent",
    "provider", "raw_path", "raw_sha256", "visible_candidates",
)
CANDIDATE_FIELDS = ("candidate_id", "url", "title", "snippet", "rank")
ATTEMPT_FIELDS = (
    "attempt_id", "physical_match_id", "discovery_id", "candidate_id", "requested_at",
    "response_at", "status_code", "error_type", "returned_document_id", "latency_ms",
    "request_count", "token_cost", "monetary_cost",
)
RESULT_FIELDS = (
    "physical_match_id", "regulation_home_goals", "regulation_away_goals",
    "result_status", "result_source_url", "result_published_at", "result_observed_at",
    "raw_path", "raw_sha256", "timing_basis",
)
CONTEXT_FIELDS = (
    "ranking", "points", "qualification_status", "schedule", "injuries",
    "lineup_status", "odds", "weather",
)
HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass
class RowSet:
    rows: list[dict[str, Any]] = field(default_factory=list)
    malformed: list[str] = field(default_factory=list)
    missing: Counter[str] = field(default_factory=Counter)


@dataclass
class Audit:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    missing: Counter[str] = field(default_factory=Counter)
    hash_failures: list[str] = field(default_factory=list)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)

    def miss(self, file_name: str, field_name: str, count: int = 1) -> None:
        self.missing[f"{file_name}.{field_name}"] += count


def _load_json(path: Path, audit: Audit) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        audit.error(f"{path.name}: invalid JSON ({type(exc).__name__}: {exc})")
        return {}
    if not isinstance(value, dict):
        audit.error(f"{path.name}: expected a JSON object")
        return {}
    return value


def _load_jsonl(root: Path, relative: str, audit: Audit) -> RowSet:
    result = RowSet()
    path = root / relative
    if not path.is_file():
        return result
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        audit.error(f"{relative}: cannot read ({type(exc).__name__}: {exc})")
        return result
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except Exception as exc:
            result.malformed.append(f"{relative}:{number}: invalid JSON ({exc})")
            continue
        if not isinstance(value, dict):
            result.malformed.append(f"{relative}:{number}: expected object")
            continue
        result.rows.append(value)
    for item in result.malformed:
        audit.error(item)
    return result


def _required(rows: Iterable[dict[str, Any]], required: Iterable[str], file_name: str,
              audit: Audit) -> None:
    fields = tuple(required)
    for index, row in enumerate(rows, 1):
        for name in fields:
            if name not in row:
                audit.miss(file_name, name)
                audit.error(f"{file_name} row {index}: missing required field {name}")


def _parse_time(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    # A time without an offset cannot be safely compared with a kickoff.  The
    # original string remains in the raw field and is reported as imprecise.
    return parsed if parsed.tzinfo else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_raw(root: Path, row: dict[str, Any], file_name: str, index: int, audit: Audit) -> None:
    raw_path = row.get("raw_path")
    expected = row.get("raw_sha256")
    if raw_path in (None, ""):
        audit.miss(file_name, "raw_path")
        return
    if not isinstance(raw_path, str):
        audit.error(f"{file_name} row {index}: raw_path must be a relative string")
        return
    path = (root / raw_path).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        audit.error(f"{file_name} row {index}: raw_path escapes stream root: {raw_path}")
        return
    if not path.is_file():
        audit.error(f"{file_name} row {index}: raw_path does not exist: {raw_path}")
        return
    if not isinstance(expected, str) or not HEX64.fullmatch(expected):
        audit.error(f"{file_name} row {index}: raw_sha256 must be a 64-character hex digest")
        return
    actual = _sha256(path)
    if actual.lower() != expected.lower():
        detail = f"{file_name} row {index}: raw hash mismatch ({raw_path})"
        audit.hash_failures.append(detail)
        audit.error(detail)


def _check_manifest_hashes(root: Path, manifest: dict[str, Any], audit: Audit) -> None:
    """Verify optional manifest file hash declarations.

    Collectors in the wild used both ``file_hashes`` and ``files`` while the
    schema was being piloted.  Supporting both keeps the validator useful while
    still rejecting a declared hash that does not match bytes on disk.
    """
    declarations: list[tuple[str, Any]] = []
    for key in ("file_hashes", "files", "artifacts"):
        value = manifest.get(key)
        if isinstance(value, dict):
            declarations.extend(value.items())
        elif isinstance(value, list):
            for item in value:
                if not isinstance(item, dict):
                    continue
                path = item.get("path") or item.get("file") or item.get("name")
                digest = item.get("sha256") or item.get("hash") or item.get("raw_sha256")
                if path:
                    declarations.append((path, digest))
    if not declarations:
        audit.warning("manifest.json: no file hash declarations found (file integrity cannot be checked from manifest)")
        return
    seen: set[str] = set()
    for name, declaration in declarations:
        if not isinstance(name, str) or not name:
            audit.error("manifest.json: file hash declaration has no path")
            continue
        if name in seen:
            audit.warning(f"manifest.json: duplicate hash declaration for {name}")
            continue
        seen.add(name)
        digest = declaration
        if isinstance(declaration, dict):
            digest = declaration.get("sha256") or declaration.get("hash") or declaration.get("raw_sha256")
        if not isinstance(digest, str) or not HEX64.fullmatch(digest):
            audit.error(f"manifest.json: invalid SHA-256 declaration for {name}")
            continue
        path = (root / name).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError:
            audit.error(f"manifest.json: hash path escapes stream root: {name}")
            continue
        if not path.is_file():
            audit.error(f"manifest.json: hash path does not exist: {name}")
            continue
        actual = _sha256(path)
        if actual.lower() != digest.lower():
            audit.error(f"manifest.json: hash mismatch for {name}")


def _id(row: dict[str, Any], *names: str) -> str | None:
    for name in names:
        value = row.get(name)
        if isinstance(value, str) and value:
            return value
    return None


def _status_ok(status: Any) -> bool:
    return isinstance(status, str) and status.lower() in {"ok", "success", "returned", "completed"}


def _is_empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _context_observed_at(snapshot: dict[str, Any], name: str) -> Any:
    """Return a field's own observation time for common JSON encodings."""
    value = snapshot.get(name)
    if isinstance(value, dict):
        return value.get("observed_at") or value.get("as_of") or value.get("quote_time")
    for key in (f"{name}_observed_at", f"{name}_as_of"):
        if key in snapshot:
            return snapshot[key]
    return None


def _precision(row: dict[str, Any]) -> str:
    basis = str(row.get("timing_basis") or row.get("timezone_basis") or "unknown").lower()
    if any(token in basis for token in ("unknown", "unspecified", "date_only", "date-only")):
        return "unknown_or_date_only"
    captured = _parse_time(row.get("captured_at"))
    if captured is None:
        return "unknown"
    text = str(row.get("original_time_text") or row.get("source_time_raw") or "")
    if re.search(r"\d{1,2}:\d{2}", text) and ("+" in text or "UTC" in text.upper() or "GMT" in text.upper() or text.upper().rstrip().endswith("Z")):
        return "minute_with_timezone"
    if re.search(r"\d{1,2}:\d{2}", text):
        return "minute_without_explicit_timezone"
    if re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", text):
        return "date_only_or_unknown_time"
    return "captured_timestamp_only"


def _validate_rows(root: Path, fixtures: RowSet, documents: RowSet, discoveries: RowSet,
                   attempts: RowSet, contexts: RowSet, results: RowSet, audit: Audit) -> dict[str, Any]:
    # Schema checks. Keep these explicit so a typo in one JSONL line cannot make
    # a seemingly complete package pass by being silently ignored.
    _required(fixtures.rows, FIXTURE_FIELDS, "fixtures.jsonl", audit)
    _required(documents.rows, DOCUMENT_FIELDS, "documents.jsonl", audit)
    _required(discoveries.rows, DISCOVERY_FIELDS, "discovery.jsonl", audit)
    _required(attempts.rows, ATTEMPT_FIELDS, "attempts.jsonl", audit)
    _required(results.rows, RESULT_FIELDS, "labels/results.jsonl", audit)

    fixture_by_id: dict[str, dict[str, Any]] = {}
    fixture_ids: set[str] = set()
    clubs: set[str] = set()
    home_teams: set[str] = set()
    away_teams: set[str] = set()
    kickoffs: dict[str, datetime | None] = {}
    for index, row in enumerate(fixtures.rows, 1):
        mid = row.get("physical_match_id")
        if not isinstance(mid, str) or not mid:
            audit.error(f"fixtures.jsonl row {index}: invalid physical_match_id")
            continue
        if mid in fixture_ids:
            audit.error(f"fixtures.jsonl row {index}: duplicate physical_match_id {mid}")
        fixture_ids.add(mid)
        fixture_by_id[mid] = row
        for key, collection in (("home_team", home_teams), ("away_team", away_teams)):
            value = row.get(key)
            if isinstance(value, str) and value:
                collection.add(value)
                clubs.add(value)
        kickoff = _parse_time(row.get("kickoff"))
        kickoffs[mid] = kickoff
        if kickoff is None:
            audit.error(f"fixtures.jsonl row {index}: kickoff must be ISO-8601 with timezone")

    document_ids: set[str] = set()
    document_versions_seen: set[str] = set()
    # Collect every version identity before checking predecessor links.  A
    # document stream may be ordered by capture time while a late correction
    # can point to a version written later in the file.
    document_versions: set[str] = {
        value for row in documents.rows
        for value in [_id(row, "version_id", "document_id")]
        if value
    }
    document_hashes: Counter[str] = Counter()
    document_match_ids: Counter[str] = Counter()
    document_modes: Counter[str] = Counter()
    time_precisions: Counter[str] = Counter()
    docs_by_match: defaultdict[str, int] = defaultdict(int)
    for index, row in enumerate(documents.rows, 1):
        did = row.get("document_id")
        if not isinstance(did, str) or not did:
            audit.error(f"documents.jsonl row {index}: invalid document_id")
        elif did in document_ids:
            audit.error(f"documents.jsonl row {index}: duplicate document_id {did}")
        document_ids.add(did)
        version_id = _id(row, "version_id", "document_id")
        if version_id:
            if version_id in document_versions_seen:
                audit.warning(f"documents.jsonl row {index}: repeated version_id {version_id}")
            document_versions_seen.add(version_id)
        match_value = row.get("physical_match_id")
        match_ids = row.get("physical_match_ids", match_value)
        if isinstance(match_ids, str):
            match_ids = [match_ids]
        if not isinstance(match_ids, list):
            match_ids = []
        if not match_ids:
            audit.warning(f"documents.jsonl row {index}: no fixture binding (uncertain/ambiguous document)")
        for mid in match_ids:
            if mid not in fixture_ids:
                audit.error(f"documents.jsonl row {index}: unknown physical_match_id {mid}")
            else:
                document_match_ids[mid] += 1
                docs_by_match[mid] += 1
        if _is_empty(row.get("body")):
            audit.warning(f"documents.jsonl row {index}: empty body")
        digest = row.get("raw_sha256")
        if isinstance(digest, str) and HEX64.fullmatch(digest):
            document_hashes[digest.lower()] += 1
        mode = row.get("information_mode")
        if mode not in MODES:
            audit.error(f"documents.jsonl row {index}: invalid information_mode {mode!r}")
        else:
            document_modes[mode] += 1
            if mode == "prospective_capture" and _parse_time(row.get("captured_at")) is None:
                audit.error(f"documents.jsonl row {index}: prospective_capture requires captured_at")
            if mode == "archived_snapshot" and not (row.get("archive_snapshot_at") or row.get("archive_snapshot_url") or row.get("historical_snapshot_proven")):
                audit.warning(f"documents.jsonl row {index}: archived_snapshot lacks archive evidence fields")
            if mode == "historical_reconstruction" and str(row.get("timing_basis") or "").lower() in {"", "unknown", "unspecified"}:
                audit.warning(f"documents.jsonl row {index}: historical_reconstruction has no timing_basis")
        time_precisions[_precision(row)] += 1
        for key in ("source_published_at", "source_modified_at", "captured_at", "available_at"):
            raw = row.get(key)
            if raw not in (None, "") and _parse_time(raw) is None:
                audit.error(f"documents.jsonl row {index}: {key} is not timezone-aware ISO-8601: {raw!r}")
        captured, available = _parse_time(row.get("captured_at")), _parse_time(row.get("available_at"))
        if captured and available and available < captured:
            audit.warning(f"documents.jsonl row {index}: available_at precedes captured_at")
        for key in ("source_published_at", "source_modified_at"):
            published = _parse_time(row.get(key))
            if published and captured and published > captured:
                audit.error(f"documents.jsonl row {index}: {key} is after captured_at")
        previous = row.get("previous_version_id")
        if previous not in (None, "", "null") and previous == version_id:
            audit.error(f"documents.jsonl row {index}: previous_version_id points to the current version {previous}")
        if previous not in (None, "", "null") and previous not in document_versions:
            audit.warning(f"documents.jsonl row {index}: previous_version_id not found in supplied versions: {previous}")
        _check_raw(root, row, "documents.jsonl", index, audit)

    discovery_ids: set[str] = set()
    discovery_by_match: defaultdict[str, int] = defaultdict(int)
    reconstruction_discoveries = 0
    for index, row in enumerate(discoveries.rows, 1):
        ident = row.get("discovery_id")
        if ident in discovery_ids:
            audit.error(f"discovery.jsonl row {index}: duplicate discovery_id {ident}")
        discovery_ids.add(ident)
        mid = row.get("physical_match_id")
        if mid not in fixture_ids:
            audit.error(f"discovery.jsonl row {index}: unknown physical_match_id {mid}")
        else:
            discovery_by_match[mid] += 1
        observed = _parse_time(row.get("observed_at"))
        if observed is None:
            audit.error(f"discovery.jsonl row {index}: observed_at must be timezone-aware ISO-8601")
        candidates = row.get("visible_candidates")
        if not isinstance(candidates, list):
            audit.error(f"discovery.jsonl row {index}: visible_candidates must be a list")
            candidates = []
        ranks: list[int] = []
        candidate_ids: set[str] = set()
        for cindex, candidate in enumerate(candidates, 1):
            if not isinstance(candidate, dict):
                audit.error(f"discovery.jsonl row {index} candidate {cindex}: expected object")
                continue
            for key in CANDIDATE_FIELDS:
                if key not in candidate:
                    audit.error(f"discovery.jsonl row {index} candidate {cindex}: missing {key}")
            cid = candidate.get("candidate_id")
            if cid in candidate_ids:
                audit.error(f"discovery.jsonl row {index}: duplicate candidate_id {cid}")
            candidate_ids.add(cid)
            if isinstance(candidate.get("rank"), int):
                ranks.append(candidate["rank"])
        if ranks and ranks != sorted(ranks):
            audit.warning(f"discovery.jsonl row {index}: visible candidate ranks are not sorted")
        status = str(row.get("status") or row.get("menu_status") or "").lower()
        if any(token in status for token in ("reconstruct", "historical")):
            reconstruction_discoveries += 1
        _check_raw(root, row, "discovery.jsonl", index, audit)

    attempt_ids: set[str] = set()
    attempt_statuses: Counter[str] = Counter()
    attempt_by_match: defaultdict[str, int] = defaultdict(int)
    empty_results = 0
    failed_attempts = 0
    contradictory_attempts = 0
    for index, row in enumerate(attempts.rows, 1):
        ident = row.get("attempt_id")
        if ident in attempt_ids:
            audit.error(f"attempts.jsonl row {index}: duplicate attempt_id {ident}")
        attempt_ids.add(ident)
        mid = row.get("physical_match_id")
        if mid not in fixture_ids:
            audit.error(f"attempts.jsonl row {index}: unknown physical_match_id {mid}")
        else:
            attempt_by_match[mid] += 1
        requested, response = _parse_time(row.get("requested_at")), _parse_time(row.get("response_at"))
        if requested is None:
            audit.error(f"attempts.jsonl row {index}: requested_at must be timezone-aware ISO-8601")
        if row.get("response_at") not in (None, "") and response is None:
            audit.error(f"attempts.jsonl row {index}: response_at is not timezone-aware ISO-8601")
        if requested and response and response < requested:
            audit.error(f"attempts.jsonl row {index}: response_at precedes requested_at")
        latency = row.get("latency_ms")
        if latency is not None and (not isinstance(latency, (int, float)) or isinstance(latency, bool) or latency < 0):
            audit.error(f"attempts.jsonl row {index}: latency_ms must be non-negative or null")
        status = str(row.get("status_code") or row.get("error_type") or "unknown").lower()
        attempt_statuses[status] += 1
        if any(token in status for token in ("conflict", "contradict")):
            contradictory_attempts += 1
        explicit_error = row.get("error_type") not in (None, "", "none", "null")
        code = row.get("status_code")
        http_error = isinstance(code, int) and not 200 <= code < 300
        status_failure = any(token in status for token in ("fail", "error", "timeout", "no_result", "empty", "parse"))
        if explicit_error or http_error or status_failure:
            failed_attempts += 1
        if any(token in status for token in ("no_result", "empty", "no-data", "no_data")) or _is_empty(row.get("returned_document_id")):
            empty_results += 1
        discovery_id = row.get("discovery_id")
        if discovery_id not in discovery_ids and discovery_id not in (None, ""):
            audit.warning(f"attempts.jsonl row {index}: discovery_id not found: {discovery_id}")
        for key in ("request_count", "latency_ms"):
            value = row.get(key)
            if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0):
                audit.error(f"attempts.jsonl row {index}: {key} must be non-negative or null")

    context_by_match: defaultdict[str, int] = defaultdict(int)
    context_field_observations = Counter()
    context_null_fields = Counter()
    for index, row in enumerate(contexts.rows, 1):
        mid = row.get("physical_match_id")
        if mid not in fixture_ids:
            audit.error(f"context_snapshots.jsonl row {index}: unknown physical_match_id {mid}")
        else:
            context_by_match[mid] += 1
        for field_name in CONTEXT_FIELDS:
            if field_name not in row:
                continue
            field_value = row[field_name]
            if field_value is None or (isinstance(field_value, dict) and field_value.get("value") is None):
                context_null_fields[field_name] += 1
            observed = _context_observed_at(row, field_name)
            if observed is None:
                audit.warning(f"context_snapshots.jsonl row {index}: {field_name} has no own observed_at/as_of")
            else:
                context_field_observations[field_name] += 1
                if _parse_time(observed) is None:
                    audit.error(f"context_snapshots.jsonl row {index}: {field_name} observation time is invalid: {observed!r}")
            if field_name == "odds":
                odds = row[field_name]
                odds_items = odds if isinstance(odds, list) else [odds]
                for quote in odds_items:
                    if isinstance(quote, dict):
                        if not quote.get("quote_time"):
                            audit.warning(f"context_snapshots.jsonl row {index}: odds quote missing quote_time")
                        if not quote.get("captured_at"):
                            audit.warning(f"context_snapshots.jsonl row {index}: odds quote missing captured_at")
    result_ids: set[str] = set()
    results_by_match: defaultdict[str, int] = defaultdict(int)
    for index, row in enumerate(results.rows, 1):
        mid = row.get("physical_match_id")
        if mid not in fixture_ids:
            audit.error(f"labels/results.jsonl row {index}: unknown physical_match_id {mid}")
        elif mid in result_ids:
            audit.warning(f"labels/results.jsonl row {index}: repeated result for {mid}")
        else:
            result_ids.add(mid)
            results_by_match[mid] += 1
        for key in ("result_published_at", "result_observed_at"):
            if row.get(key) not in (None, "") and _parse_time(row.get(key)) is None:
                audit.error(f"labels/results.jsonl row {index}: {key} is not timezone-aware ISO-8601")
        for key in ("regulation_home_goals", "regulation_away_goals"):
            value = row.get(key)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
                audit.error(f"labels/results.jsonl row {index}: {key} must be a non-negative integer or null")
        _check_raw(root, row, "labels/results.jsonl", index, audit)

    # A match is considered timeline-complete only if the package has at least a
    # candidate menu, an acquisition attempt, one document and a result label.
    complete_matches = [mid for mid in fixture_ids if docs_by_match[mid] and discovery_by_match[mid] and attempt_by_match[mid] and mid in result_ids]
    missing_by_match = {
        mid: [name for name, seen in (("documents", docs_by_match[mid]), ("discovery", discovery_by_match[mid]), ("attempts", attempt_by_match[mid]), ("result", mid in result_ids)) if not seen]
        for mid in fixture_ids
        if mid not in complete_matches
    }
    duplicate_docs = sum(count - 1 for count in document_hashes.values() if count > 1)
    docs_count = len(documents.rows)
    return {
        "fixtures": len(fixture_ids), "clubs": len(clubs), "home_teams": len(home_teams),
        "away_teams": len(away_teams), "documents": docs_count,
        "documents_by_mode": dict(document_modes), "document_hash_clusters": len(document_hashes),
        "duplicate_documents": duplicate_docs,
        "duplicate_ratio": duplicate_docs / docs_count if docs_count else 0.0,
        "time_precision": dict(time_precisions), "discoveries": len(discoveries.rows),
        "reconstruction_discoveries": reconstruction_discoveries,
        "attempts": len(attempts.rows), "attempt_statuses": dict(attempt_statuses),
        "failed_attempts": failed_attempts, "empty_or_no_result_attempts": empty_results,
        "contradictory_attempts": contradictory_attempts,
        "contexts": len(contexts.rows), "context_field_observations": dict(context_field_observations),
        "context_null_fields": dict(context_null_fields),
        "results": len(results.rows), "result_matches": len(result_ids),
        "timeline_complete_matches": len(complete_matches), "timeline_incomplete_matches": len(missing_by_match),
        "timeline_missing_by_match": missing_by_match,
        "time_window": {
            "earliest_kickoff": min((x for x in kickoffs.values() if x), default=None).isoformat() if any(kickoffs.values()) else None,
            "latest_kickoff": max((x for x in kickoffs.values() if x), default=None).isoformat() if any(kickoffs.values()) else None,
        },
    }


def _render_report(root: Path, manifest: dict[str, Any], metrics: dict[str, Any], audit: Audit,
                   rows: dict[str, RowSet]) -> str:
    mode_counts = metrics["documents_by_mode"]
    lines = [
        "FootballMind information stream quality report",
        f"stream_root: {root}",
        f"generated_at: {datetime.now(timezone.utc).isoformat()}",
        f"status: {'failed' if audit.errors else 'passed_with_warnings' if audit.warnings else 'passed'}",
        "",
        "Coverage",
        f"- independent_matches: {metrics['fixtures']}",
        f"- clubs: {metrics['clubs']} (home clubs: {metrics['home_teams']}; away clubs: {metrics['away_teams']})",
        f"- documents: {metrics['documents']}",
        f"- discoveries: {metrics['discoveries']} (reconstruction menus: {metrics['reconstruction_discoveries']})",
        f"- attempts: {metrics['attempts']}",
        f"- context_snapshots: {metrics['contexts']}",
        f"- result_labels: {metrics['results']} ({metrics['result_matches']} distinct matches)",
        "",
        "Information modes",
        f"- prospective_capture: {mode_counts.get('prospective_capture', 0)}",
        f"- archived_snapshot: {mode_counts.get('archived_snapshot', 0)}",
        f"- historical_reconstruction: {mode_counts.get('historical_reconstruction', 0)}",
        "",
        "Document provenance",
        f"- duplicate_documents: {metrics['duplicate_documents']}",
        f"- duplicate_ratio: {metrics['duplicate_ratio']:.4f} (duplicates beyond first / all document rows)",
        f"- time_precision: {json.dumps(metrics['time_precision'], ensure_ascii=False, sort_keys=True)}",
        f"- raw_hash_failures: {len(audit.hash_failures)}",
        "",
        "Attempts and responses",
        f"- failed_or_error_attempts: {metrics['failed_attempts']}",
        f"- empty_or_no_result_attempts: {metrics['empty_or_no_result_attempts']}",
        f"- contradictory_or_conflict_attempts: {metrics['contradictory_attempts']}",
        f"- status_breakdown: {json.dumps(metrics['attempt_statuses'], ensure_ascii=False, sort_keys=True)}",
        f"- context_fields_with_null_value: {json.dumps(metrics['context_null_fields'], ensure_ascii=False, sort_keys=True)} (null means no structured value was captured; raw article text remains in documents)",
        "",
        "Timeline completeness",
        f"- complete_matches (document + discovery + attempt + result): {metrics['timeline_complete_matches']}",
        f"- incomplete_matches: {metrics['timeline_incomplete_matches']}",
        f"- missing_by_match: {json.dumps(metrics['timeline_missing_by_match'], ensure_ascii=False, sort_keys=True)}",
        "",
        "Missing fields (recorded per file.field)",
        f"- {json.dumps(dict(sorted(audit.missing.items())), ensure_ascii=False, sort_keys=True)}",
        "",
        "Manifest",
        f"- keys: {', '.join(sorted(manifest)) if manifest else '(manifest unavailable)'}",
        "",
        f"Errors: {len(audit.errors)}",
    ]
    lines.extend(f"  - {item}" for item in audit.errors[:200])
    if len(audit.errors) > 200:
        lines.append(f"  - ... {len(audit.errors) - 200} additional errors omitted")
    lines.append("")
    lines.append(f"Warnings: {len(audit.warnings)}")
    lines.extend(f"  - {item}" for item in audit.warnings[:200])
    if len(audit.warnings) > 200:
        lines.append(f"  - ... {len(audit.warnings) - 200} additional warnings omitted")
    lines.append("")
    lines.append("Interpretation: this report checks capture completeness and provenance only. It does not score VoI, estimate win probabilities, or select documents by the final result.")
    return "\n".join(lines) + "\n"


def run(root: Path, report: Path | None = None) -> tuple[int, dict[str, Any]]:
    root = root.resolve()
    audit = Audit()
    for required in REQUIRED_FILES:
        if not (root / required).is_file():
            audit.error(f"missing required file: {required}")
    manifest = _load_json(root / "manifest.json", audit) if (root / "manifest.json").is_file() else {}
    if manifest:
        _check_manifest_hashes(root, manifest, audit)
    rows = {
        "fixtures": _load_jsonl(root, "fixtures.jsonl", audit),
        "documents": _load_jsonl(root, "documents.jsonl", audit),
        "discoveries": _load_jsonl(root, "discovery.jsonl", audit),
        "attempts": _load_jsonl(root, "attempts.jsonl", audit),
        "contexts": _load_jsonl(root, "context_snapshots.jsonl", audit),
        "results": _load_jsonl(root, "labels/results.jsonl", audit),
    }
    metrics = _validate_rows(root, rows["fixtures"], rows["documents"], rows["discoveries"], rows["attempts"], rows["contexts"], rows["results"], audit)
    report_path = report.resolve() if report else root / "quality-report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(root, manifest, metrics, audit, rows), encoding="utf-8")
    output = {"status": "failed" if audit.errors else "passed_with_warnings" if audit.warnings else "passed", "metrics": metrics, "errors": audit.errors, "warnings": audit.warnings, "report": str(report_path)}
    return (1 if audit.errors else 0), output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="footballmind-information-stream directory")
    parser.add_argument("--report", type=Path, help="quality-report.txt destination (default: root/quality-report.txt)")
    parser.add_argument("--json", action="store_true", help="also print the structured summary")
    args = parser.parse_args(argv)
    code, summary = run(args.root, args.report)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"{summary['status']}: report={summary['report']}; errors={len(summary['errors'])}; warnings={len(summary['warnings'])}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
