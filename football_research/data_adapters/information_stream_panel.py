"""Convert a validated raw information stream into the existing historical panel.

This is intentionally a conservative bridge. It accepts only explicit fixture
identities, regulation-time result labels, and source-byte hashes. It does not
invent priors, availability times, article usefulness, or missing result
provenance. Prospective records remain in the raw package and are not silently
turned into historical reconstruction cases.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from .information_stream_contract import validate_stream


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _time(value: Any) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("an explicit timezone-aware ISO-8601 time is required")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("a timezone offset is required")
    return parsed


def _raw_path(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("raw_path is required")
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _result_archive(root: Path, result_rows: list[dict[str, Any]]) -> tuple[Path, str]:
    path = root / "labels" / "results.jsonl"
    if not path.exists():
        raise ValueError("labels/results.jsonl is required")
    return path.resolve(), _sha256(path)


def _derived_identity(kind: str, value: Any) -> str:
    """Stable adapter identity for streams that only provide display names."""
    normalized = " ".join(str(value or "").casefold().split())
    if not normalized:
        raise ValueError(f"cannot derive an identity for empty {kind}")
    return f"derived:{kind}:{hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:24]}"


def build_historical_cases(root: str | Path, *, maximum_documents: int = 3, result_clock: str = "source_reconstruction") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build cases and a report without mutating the input package."""
    root = Path(root).resolve()
    if type(maximum_documents) is not int or not 1 <= maximum_documents <= 3:
        raise ValueError("maximum_documents must be an integer from 1 through 3")
    if result_clock not in {"observed", "source_reconstruction"}:
        raise ValueError("result_clock must be observed or source_reconstruction")
    validation = validate_stream(root)
    report: dict[str, Any] = {
        "schema": "footballmind-information-stream-panel-v1",
        "root": str(root), "status": "not_ready", "validation": validation,
        "reasons": [], "excluded": [], "cases": 0,
        "scope": "historical_reconstruction_only",
        "result_clock": result_clock,
    }
    if not validation["ready"]:
        report["reasons"].append("raw_stream_validation_failed")
        return [], report
    fixtures = {row["physical_match_id"]: row for row in _read_jsonl(root / "fixtures.jsonl")}
    documents = _read_jsonl(root / "documents.jsonl")
    results = _read_jsonl(root / "labels" / "results.jsonl")
    result_archive, result_hash = _result_archive(root, results)
    docs_by_match: dict[str, list[dict[str, Any]]] = {}
    for document in documents:
        mid = document.get("physical_match_id")
        if not mid or mid not in fixtures:
            continue
        if document.get("information_mode") == "prospective_capture":
            report["excluded"].append({"document_id": document.get("document_id"),
                                       "reason": "prospective_capture_not_imported_as_historical_panel"})
            continue
        try:
            kickoff = _time(fixtures[mid]["kickoff"])
            publication = _time(document.get("source_modified_at") or document.get("source_published_at"))
            if publication >= kickoff:
                report["excluded"].append({"document_id": document.get("document_id"),
                                           "reason": "document_not_prematch"})
                continue
            if not isinstance(document.get("title"), str) or not document["title"].strip():
                raise ValueError("nonempty title is required")
            document_text = document.get("text", document.get("body"))
            if not isinstance(document_text, str) or not document_text.strip():
                raise ValueError("nonempty body is required")
            raw = _raw_path(root, document.get("raw_path"))
            if not raw.exists() or _sha256(raw) != document.get("raw_sha256"):
                raise ValueError("raw source bytes do not match raw_sha256")
            docs_by_match.setdefault(mid, []).append(deepcopy(document))
        except (KeyError, TypeError, ValueError) as exc:
            report["excluded"].append({"document_id": document.get("document_id"), "reason": str(exc)})

    cases: list[dict[str, Any]] = []
    for mid, fixture in sorted(fixtures.items(), key=lambda item: _time(item[1]["kickoff"])):
        row = next((result for result in results if result.get("physical_match_id") == mid), None)
        docs = sorted(docs_by_match.get(mid, []), key=lambda d: (_time(d.get("source_modified_at") or d["source_published_at"]), d["url"]))
        if row is None:
            report["excluded"].append({"physical_match_id": mid, "reason": "missing_result"})
            continue
        try:
            kickoff = _time(fixture["kickoff"])
            actual_result_at = _time(row["result_observed_at"])
            if actual_result_at < kickoff:
                raise ValueError("result_observed_at precedes kickoff")
            proxy_raw = row.get("source_result_available_at")
            proxy_result_at = _time(proxy_raw) if proxy_raw else actual_result_at
            if proxy_result_at < kickoff:
                proxy_result_at = actual_result_at
            case_result_at = actual_result_at if result_clock == "observed" else proxy_result_at
            home_goals = row.get("regulation_home_goals", row.get("home_goals"))
            away_goals = row.get("regulation_away_goals", row.get("away_goals"))
            if type(home_goals) is not int or type(away_goals) is not int or min(home_goals, away_goals) < 0:
                raise ValueError("regulation goals are required")
            outcome = 0 if home_goals > away_goals else 1 if home_goals == away_goals else 2
            if len(docs) < 1:
                raise ValueError("at least one verified prematch document is required")
            if len(docs) > maximum_documents:
                report["excluded"].append({"physical_match_id": mid,
                                           "reason": "panel_exceeds_declared_maximum_documents",
                                           "available_documents": len(docs)})
                continue
            label_source = row.get("source") or row.get("source_provider")
            label_id = row.get("source_id")
            if not isinstance(label_source, str) or not isinstance(label_id, str):
                raise ValueError("result source and source_id are required for historical verification")
            home_name = fixture.get("home_team")
            away_name = fixture.get("away_team")
            competition = fixture.get("competition")
            if not all(isinstance(value, str) and value.strip()
                       for value in (home_name, away_name, competition)):
                raise ValueError("fixture names and competition are required")
            home_team_id = fixture.get("home_team_id") or _derived_identity("team", home_name)
            away_team_id = fixture.get("away_team_id") or _derived_identity("team", away_name)
            competition_id = fixture.get("competition_id") or _derived_identity("competition", competition)
            if home_team_id == away_team_id:
                raise ValueError("home and away identities must differ")
            converted_fixture = {
                "physical_match_id": mid,
                "home_team_id": home_team_id, "away_team_id": away_team_id,
                "home_name": home_name, "away_name": away_name,
                "competition": competition,
                "competition_id": competition_id, "kickoff": fixture["kickoff"],
                "home_aliases": fixture.get("home_aliases", []),
                "away_aliases": fixture.get("away_aliases", []),
            }
            # Keep the two archive bindings explicit.  The historical kernel
            # must be able to prove that a case still points at the exact
            # submitted fixture/result ledgers; it must not trust a copied
            # projection of either file.
            result_raw = _raw_path(root, row.get("raw_path"))
            if not result_raw.exists() or _sha256(result_raw) != row.get("raw_sha256"):
                raise ValueError("raw result bytes do not match raw_sha256")
            converted_docs = []
            for document in docs:
                converted = deepcopy(document)
                # The raw stream calls the canonical article field ``body``;
                # the historical kernel consumes ``text``.  This is a derived
                # view only and leaves the submitted JSONL unchanged.
                if not converted.get("text") and isinstance(converted.get("body"), str):
                    converted["text"] = converted["body"]
                converted["raw_path"] = str(_raw_path(root, converted["raw_path"]))
                converted_docs.append(converted)
            cases.append({
                "fixture": converted_fixture,
                "outcome": outcome,
                # The case clock is the real observation clock by default.
                # The source's reconstructed availability remains provenance
                # metadata and can only be selected explicitly downstream.
                "result_available_at": case_result_at.isoformat(),
                "label_provenance": {"path": str(result_archive), "sha256": result_hash,
                                     "source": label_source, "source_id": label_id,
                                     "source_schema": "footballmind-information-stream/v1",
                                     "fixtures_path": str((root / "fixtures.jsonl").resolve()),
                                     "fixtures_sha256": _sha256(root / "fixtures.jsonl"),
                                     "raw_result_path": str(result_raw),
                                     "raw_result_sha256": row.get("raw_sha256"),
                                     "result_observed_at": actual_result_at.isoformat(),
                                     "actual_result_observed_at": actual_result_at.isoformat(),
                                     "source_result_available_at": proxy_result_at.isoformat() if proxy_raw else None,
                                     "source_result_available_at_basis": "source_reconstruction" if proxy_raw else None,
                                     "result_available_at_basis": "actual_result_observed_at" if result_clock == "observed" else "source_result_available_at_reconstruction",
                                     "result_clock": result_clock,
                                     "result_clock_assumption": result_clock == "source_reconstruction"},
                "documents": converted_docs,
                "background_links": [],
                "prior_probabilities": None,
                "information_mode": "historical_reconstruction",
            })
        except (KeyError, TypeError, ValueError) as exc:
            report["excluded"].append({"physical_match_id": mid, "reason": str(exc)})
    report["cases"] = len(cases)
    report["status"] = "ready" if cases else "not_ready"
    if not cases:
        report["reasons"].append("no_case_has_explicit_result_and_verified_prematch_body")
    report["limitations"] = [
        "This bridge imports historical reconstruction only; prospective records stay in the raw stream.",
        "Discovery menus and attempts are retained in the source package but are not yet counterfactual propensities.",
        "At most three documents per case are admitted by the existing bounded historical kernel.",
    ]
    return cases, report


__all__ = ["build_historical_cases"]
