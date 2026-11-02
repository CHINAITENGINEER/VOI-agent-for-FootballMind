"""Validation contract for FootballMind's timestamped information stream.

The collector owns the raw files; this module only verifies that the package
can be used without silently inventing clocks or dangling cross-references.
It deliberately reports warnings for incomplete historical reconstruction so
that a partial delivery remains inspectable instead of being rewritten.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
from pathlib import Path
from typing import Any


FILES = (
    "fixtures.jsonl",
    "documents.jsonl",
    "discovery.jsonl",
    "attempts.jsonl",
    "context_snapshots.jsonl",
    "labels/results.jsonl",
)
MODES = {"prospective_capture", "archived_snapshot", "historical_reconstruction"}


def _parse(value: Any, *, field: str, row_no: int, errors: list[str]) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{field}: row {row_no} must be an ISO-8601 string or null")
        return None
    raw = value.strip().replace("Z", "+00:00")
    try:
        stamp = datetime.fromisoformat(raw)
    except ValueError:
        errors.append(f"{field}: row {row_no} is not valid ISO-8601: {value!r}")
        return None
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        errors.append(f"{field}: row {row_no} must include a timezone offset")
        return None
    return stamp


def _rows(path: Path, *, errors: list[str]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    result: list[dict[str, Any]] = []
    for row_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path.name}: row {row_no} is invalid JSON: {exc.msg}")
            continue
        if not isinstance(value, dict):
            errors.append(f"{path.name}: row {row_no} must be a JSON object")
            continue
        result.append(value)
    return result


def _required(row: dict[str, Any], fields: tuple[str, ...], *, source: str,
              row_no: int, errors: list[str]) -> None:
    for field in fields:
        if field not in row or row[field] in (None, ""):
            errors.append(f"{source}: row {row_no} missing {field}")


def _unique(rows: list[dict[str, Any]], field: str, *, source: str,
            errors: list[str]) -> set[str]:
    seen: set[str] = set()
    for row_no, row in enumerate(rows, 1):
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            continue
        if value in seen:
            errors.append(f"{source}: duplicate {field}={value!r}")
        seen.add(value)
    return seen


def _order(left: datetime | None, right: datetime | None, *, message: str,
           errors: list[str]) -> None:
    if left is not None and right is not None and left > right:
        errors.append(message)


def validate_stream(root: str | Path) -> dict[str, Any]:
    """Return a deterministic validation report for one raw information package.

    ``ready`` means the package is structurally usable for the next adapter
    stage. It does not claim that the sample is large enough to fit or validate
    a useful ``g`` model. Unknown times remain unknown and are surfaced as
    warnings rather than replaced by estimates.
    """
    root = Path(root)
    errors: list[str] = []
    warnings: list[str] = []
    if not root.is_dir():
        return {"schema": "footballmind-information-stream-validation-v1",
                "root": str(root), "ready": False,
                "errors": [f"package directory does not exist: {root}"],
                "warnings": [], "counts": {}}

    loaded = {name: _rows(root / name, errors=errors) for name in FILES}
    fixtures = loaded["fixtures.jsonl"]
    documents = loaded["documents.jsonl"]
    discoveries = loaded["discovery.jsonl"]
    attempts = loaded["attempts.jsonl"]
    snapshots = loaded["context_snapshots.jsonl"]
    results = loaded["labels/results.jsonl"]

    fixture_ids = _unique(fixtures, "physical_match_id", source="fixtures.jsonl", errors=errors)
    kickoff: dict[str, datetime] = {}
    for no, row in enumerate(fixtures, 1):
        _required(row, ("physical_match_id", "home_team", "away_team", "kickoff"),
                  source="fixtures.jsonl", row_no=no, errors=errors)
        if row.get("home_team") == row.get("away_team") and row.get("home_team"):
            errors.append(f"fixtures.jsonl: row {no} has identical home and away teams")
        stamp = _parse(row.get("kickoff"), field="kickoff", row_no=no, errors=errors)
        if stamp is not None and isinstance(row.get("physical_match_id"), str):
            kickoff[row["physical_match_id"]] = stamp

    document_ids = _unique(documents, "document_id", source="documents.jsonl", errors=errors)
    for no, row in enumerate(documents, 1):
        _required(row, ("document_id", "url", "captured_at"), source="documents.jsonl",
                  row_no=no, errors=errors)
        mid = row.get("physical_match_id")
        if mid is not None and mid not in fixture_ids:
            errors.append(f"documents.jsonl: row {no} references unknown physical_match_id={mid!r}")
        mode = row.get("information_mode")
        if mode not in MODES:
            errors.append(f"documents.jsonl: row {no} has unknown information_mode={mode!r}")
        captured = _parse(row.get("captured_at"), field="captured_at", row_no=no, errors=errors)
        published = _parse(row.get("source_published_at"), field="source_published_at", row_no=no, errors=errors)
        modified = _parse(row.get("source_modified_at"), field="source_modified_at", row_no=no, errors=errors)
        available = _parse(row.get("available_at"), field="available_at", row_no=no, errors=errors)
        _order(published, captured, message=f"documents.jsonl: row {no} publication follows capture", errors=errors)
        _order(modified, captured, message=f"documents.jsonl: row {no} modification follows capture", errors=errors)
        _order(available, captured, message=f"documents.jsonl: row {no} availability follows capture", errors=errors)
        if published is None:
            warnings.append(f"documents.jsonl: row {no} has unknown source_published_at")
        if mode == "historical_reconstruction":
            warnings.append(f"documents.jsonl: row {no} is historical reconstruction, not a real capture clock")

    discovery_ids = _unique(discoveries, "discovery_id", source="discovery.jsonl", errors=errors)
    for no, row in enumerate(discoveries, 1):
        _required(row, ("discovery_id", "physical_match_id", "observed_at", "visible_candidates"),
                  source="discovery.jsonl", row_no=no, errors=errors)
        mid = row.get("physical_match_id")
        if mid not in fixture_ids:
            errors.append(f"discovery.jsonl: row {no} references unknown physical_match_id={mid!r}")
        observed = _parse(row.get("observed_at"), field="observed_at", row_no=no, errors=errors)
        if not isinstance(row.get("visible_candidates"), list):
            continue
        candidate_ids: set[str] = set()
        for candidate in row["visible_candidates"]:
            if not isinstance(candidate, dict):
                errors.append(f"discovery.jsonl: row {no} contains non-object candidate")
                continue
            cid = candidate.get("candidate_id")
            if not isinstance(cid, str) or not cid.strip():
                errors.append(f"discovery.jsonl: row {no} candidate lacks candidate_id")
            elif cid in candidate_ids:
                errors.append(f"discovery.jsonl: row {no} duplicates candidate_id={cid!r}")
            candidate_ids.add(cid)
            if not candidate.get("url"):
                errors.append(f"discovery.jsonl: row {no} candidate lacks url")
        if mid in kickoff and observed is not None and observed > kickoff[mid]:
            warnings.append(f"discovery.jsonl: row {no} was observed after kickoff")

    attempt_ids = _unique(attempts, "attempt_id", source="attempts.jsonl", errors=errors)
    for no, row in enumerate(attempts, 1):
        _required(row, ("attempt_id", "physical_match_id", "requested_at"), source="attempts.jsonl",
                  row_no=no, errors=errors)
        mid = row.get("physical_match_id")
        if mid not in fixture_ids:
            errors.append(f"attempts.jsonl: row {no} references unknown physical_match_id={mid!r}")
        requested = _parse(row.get("requested_at"), field="requested_at", row_no=no, errors=errors)
        response = _parse(row.get("response_at"), field="response_at", row_no=no, errors=errors)
        _order(requested, response, message=f"attempts.jsonl: row {no} response precedes request", errors=errors)
        if mid in kickoff and requested is not None and requested > kickoff[mid]:
            warnings.append(f"attempts.jsonl: row {no} was requested after kickoff")
        discovery_id = row.get("discovery_id")
        if discovery_id is not None and discovery_id not in discovery_ids:
            errors.append(f"attempts.jsonl: row {no} references unknown discovery_id={discovery_id!r}")
        document_id = row.get("returned_document_id")
        if document_id is not None and document_id not in document_ids:
            errors.append(f"attempts.jsonl: row {no} references unknown returned_document_id={document_id!r}")

    snapshot_ids = _unique(snapshots, "snapshot_id", source="context_snapshots.jsonl", errors=errors)
    for no, row in enumerate(snapshots, 1):
        # v1 collectors used the more explicit ``snapshot_observed_at`` name
        # for the top-level clock.  Accept it as an alias while retaining the
        # canonical ``observed_at`` contract for newer packages.  The raw
        # package is never rewritten and the alias is not treated as proof of
        # historical availability.
        _required(row, ("snapshot_id", "physical_match_id"), source="context_snapshots.jsonl",
                  row_no=no, errors=errors)
        observed_key = "observed_at" if row.get("observed_at") not in (None, "") else "snapshot_observed_at"
        if row.get(observed_key) in (None, ""):
            errors.append(f"context_snapshots.jsonl: row {no} missing observed_at (or snapshot_observed_at)")
        if row.get("physical_match_id") not in fixture_ids:
            errors.append(f"context_snapshots.jsonl: row {no} references unknown physical_match_id={row.get('physical_match_id')!r}")
        observed = _parse(row.get(observed_key), field=observed_key, row_no=no, errors=errors)
        mid = row.get("physical_match_id")
        if mid in kickoff and observed is not None and observed > kickoff[mid]:
            warnings.append(f"context_snapshots.jsonl: row {no} was observed after kickoff")

    result_ids = _unique(results, "physical_match_id", source="labels/results.jsonl", errors=errors)
    for no, row in enumerate(results, 1):
        _required(row, ("physical_match_id", "result_status"), source="labels/results.jsonl", row_no=no, errors=errors)
        mid = row.get("physical_match_id")
        if mid not in fixture_ids:
            errors.append(f"labels/results.jsonl: row {no} references unknown physical_match_id={mid!r}")
        observed = _parse(row.get("result_observed_at"), field="result_observed_at", row_no=no, errors=errors)
        if mid in kickoff and observed is not None and observed < kickoff[mid]:
            errors.append(f"labels/results.jsonl: row {no} result observed before kickoff")

    modes = Counter(row.get("information_mode") for row in documents)
    report = {
        "schema": "footballmind-information-stream-validation-v1",
        "root": str(root.resolve()),
        "ready": not errors and bool(fixtures) and bool(documents),
        "errors": errors,
        "warnings": warnings,
        "counts": {"fixtures": len(fixtures), "documents": len(documents),
                   "discoveries": len(discoveries), "attempts": len(attempts),
                   "context_snapshots": len(snapshots), "results": len(results),
                   "document_modes": dict(modes)},
        "unique_ids": {"fixtures": len(fixture_ids), "documents": len(document_ids),
                       "discoveries": len(discovery_ids), "attempts": len(attempt_ids),
                       "context_snapshots": len(snapshot_ids), "result_fixtures": len(result_ids)},
        "limitations": [
            "ready is a structural gate, not a sample-size or g-performance claim",
            "historical_reconstruction records retain an assumed availability basis",
            "unobserved counterfactual returns are not inferred from this package",
        ],
    }
    return report


__all__ = ["FILES", "MODES", "validate_stream"]
