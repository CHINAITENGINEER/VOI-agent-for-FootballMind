"""Read-only validation for a ``footballmind-information-stream`` package.

The information stream is an ingestion artifact.  This module deliberately
does not construct features, labels, or policy inputs and does not rewrite a
package while checking it.  It validates the small interchange schema from
``docs/research/footballmind-g-data-collection-prompt.txt`` and the temporal
relationships that prevent a later observation from being presented as a
pre-match observation.

``validate_stream`` returns a JSON-serialisable report so callers can inspect
all failures at once.  A package can contain empty JSONL files (for example a
package without labels yet), but the files themselves must be present.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


DATA_FILES = (
    "fixtures.jsonl",
    "documents.jsonl",
    "discovery.jsonl",
    "attempts.jsonl",
    "context_snapshots.jsonl",
    "labels/results.jsonl",
)
INFORMATION_MODES = {"prospective_capture", "archived_snapshot", "historical_reconstruction"}
ATTEMPT_STATUSES = {
    "ok", "no_results", "timeout", "http_error", "parse_error", "empty", "duplicate",
    "low_relevance", "cancelled", "unknown", "returned", "failed", "blocked",
    "not_requested", "not_configured", "contradictory", "unchanged", "late_return",
}
PROSPECTIVE_SCHEMA = "footballmind-prospective-stream-v2"
POLICY_ARMS = {"voi", "fixed_order", "random_explore", "full_probe"}
CONTEXT_FIELDS = ("ranking", "points", "qualification_status", "schedule",
                  "injuries", "lineup_status", "odds", "weather")
HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass(frozen=True)
class Issue:
    code: str
    message: str
    path: str
    severity: str = "error"

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message, "path": self.path,
                "severity": self.severity}


def _pre_read_metadata(candidate, menu, path, issues):
    """Optional, backwards-compatible labels; never authorize body backfill."""
    if "pre_read_metadata" not in candidate:
        return
    data = candidate["pre_read_metadata"]
    if not isinstance(data, dict) or data.get("schema_version") != "observed-pre-read-metadata-v1":
        issues.append(Issue("pre_read_schema", "unsupported pre-read metadata schema", path))
        return
    if data.get("target_body_used") is not False:
        issues.append(Issue("pre_read_body_leakage", "pre-read metadata must not use the target's revealed body", path))
    if data.get("status") not in {"recorded", "not_recorded"}:
        issues.append(Issue("pre_read_status", "pre-read status must retain recorded versus missing", path))
    if data.get("f_consumed") is not None and type(data.get("f_consumed")) is not bool:
        issues.append(Issue("pre_read_consumption", "f_consumed must be boolean or null", path))
    labels = candidate.get("observed_labels")
    if not isinstance(labels, list) or any(not isinstance(x, dict) or not isinstance(x.get("text"), str)
                                         or not x.get("basis") for x in labels):
        issues.append(Issue("pre_read_labels", "observed labels require literal text and provenance", path))
    visible = _parse_time(data.get("visible_by_decision_at"), path=path + ".visible_by_decision_at", issues=issues)
    observed = _parse_time(menu.get("observed_at"), path=path + ".menu_observed_at", issues=issues)
    if visible is not None and observed is not None and visible != observed:
        issues.append(Issue("pre_read_clock", "label visibility refers to this recorded decision only", path))


def _parse_time(value: Any, *, path: str, issues: list[Issue], required: bool = False) -> datetime | None:
    if value is None:
        if required:
            issues.append(Issue("missing_time", "required timestamp is missing", path))
        return None
    if not isinstance(value, str) or not value.strip():
        issues.append(Issue("invalid_time", "timestamp must be an ISO-8601 string or null", path))
        return None
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        issues.append(Issue("invalid_time", f"invalid ISO-8601 timestamp: {value!r}", path))
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        issues.append(Issue("naive_time", "timestamp must carry an explicit timezone", path))
        return None
    return parsed


def _required(row: Mapping[str, Any], fields: Iterable[str], path: str, issues: list[Issue]) -> None:
    for field in fields:
        if field not in row:
            issues.append(Issue("missing_field", f"required field {field!r} is missing", f"{path}.{field}"))


def _string(value: Any, path: str, issues: list[Issue], *, allow_empty: bool = False) -> bool:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        issues.append(Issue("invalid_type", "expected a non-empty string", path))
        return False
    return True


def _rows(package: Path, relative: str, issues: list[Issue]) -> list[dict[str, Any]]:
    path = package / relative
    if not path.is_file():
        issues.append(Issue("missing_file", "required JSONL file is missing", relative))
        return []
    result: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row_path = f"{relative}:{number}"
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    issues.append(Issue("invalid_json", f"invalid JSONL row: {exc.msg}", row_path))
                    continue
                if not isinstance(value, dict):
                    issues.append(Issue("invalid_row", "JSONL row must be an object", row_path))
                    continue
                result.append(value)
    except (OSError, UnicodeError) as exc:
        issues.append(Issue("read_error", str(exc), relative))
    return result


def _unique(rows: list[dict[str, Any]], key: str, source: str, issues: list[Issue]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, 1):
        value = row.get(key)
        path = f"{source}:{index}.{key}"
        if not _string(value, path, issues=issues):
            continue
        if value in found:
            issues.append(Issue("duplicate_id", f"duplicate {key}: {value!r}", path))
        else:
            found[value] = row
    return found


def _before(left: datetime | None, right: datetime | None, code: str, message: str,
            path: str, issues: list[Issue], *, equal: bool = True) -> None:
    if left is not None and right is not None and (left > right if equal else left >= right):
        issues.append(Issue(code, message, path))


def _visibility_extension(row: Mapping[str, Any], path: str, issues: list[Issue],
                          kickoff: datetime | None = None) -> None:
    """Optional response/admission clocks; old rows remain readable."""
    clocks = {key: _parse_time(row.get(key), path=f"{path}.{key}", issues=issues)
              for key in ("result_observed_at", "receipt_available_at", "document_available_at")}
    if "result_observed" in row:
        if type(row.get("result_observed")) is not bool:
            issues.append(Issue("invalid_type", "result_observed must be boolean", path))
        elif row["result_observed"] != (clocks["result_observed_at"] is not None):
            issues.append(Issue("result_observation_unproven", "result observation flag must match its actual recorded clock", path))
    _before(clocks["result_observed_at"], clocks["receipt_available_at"],
            "receipt_time_order", "receipt was available before its runtime observation", path, issues)
    _before(clocks["receipt_available_at"], clocks["document_available_at"],
            "document_admission_time_order", "document was available before the returned receipt", path, issues)
    if "document_admitted" in row:
        admitted = row.get("document_admitted")
        if admitted is not None and type(admitted) is not bool:
            issues.append(Issue("invalid_type", "document_admitted must be boolean or null", path))
        if admitted is not True and clocks["document_available_at"] is not None:
            issues.append(Issue("document_availability_without_admission", "a returned receipt does not prove document admission", path))
        if admitted is True and row.get("document_admission_basis") != "verified_raw_context_document_membership":
            issues.append(Issue("document_admission_unproven", "document admission requires raw context membership evidence", path))
    for key, value in clocks.items():
        _before(value, kickoff, "post_kickoff_visibility", "prospective visibility must precede kickoff", f"{path}.{key}", issues, equal=False)


def _safe_raw(package: Path, value: Any, expected: Any, path: str, issues: list[Issue]) -> None:
    """Check a raw reference without ever following a path outside package."""
    if value is None:
        issues.append(Issue("missing_raw_reference", "raw_path is required for retained source material", path))
        return
    if not _string(value, path, issues=issues):
        return
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        issues.append(Issue("unsafe_raw_path", "raw_path must be relative to the package", path))
        return
    full = (package / candidate).resolve()
    root = package.resolve()
    if full != root and root not in full.parents:
        issues.append(Issue("unsafe_raw_path", "raw_path escapes the package", path))
        return
    if not full.is_file():
        issues.append(Issue("missing_raw_file", "raw_path does not refer to a file", path))
        return
    if not isinstance(expected, str) or not HEX64.fullmatch(expected):
        issues.append(Issue("invalid_hash", "raw_sha256 must be a 64-character hexadecimal digest", path.replace("raw_path", "raw_sha256")))
        return
    try:
        payload = full.read_bytes()
        # RawArchive hashes response bytes, while its disk representation is gzip.
        # Manifest hashes below always describe the physical files instead.
        if full.name.endswith(".body.gz"):
            payload = gzip.decompress(payload)
        actual = hashlib.sha256(payload).hexdigest()
    except (OSError, EOFError) as exc:
        issues.append(Issue("raw_read_error", str(exc), path))
        return
    if actual.lower() != expected.lower():
        issues.append(Issue("raw_hash_mismatch", f"expected {expected}, got {actual}", path))


def _validate_manifest(package: Path, manifest: Any, issues: list[Issue]) -> None:
    if not isinstance(manifest, dict):
        issues.append(Issue("invalid_manifest", "manifest.json must contain an object", "manifest.json"))
        return
    if "files" not in manifest:
        issues.append(Issue("manifest_files_missing", "manifest should declare file hashes in files", "manifest.json", "warning"))
        return
    files = manifest["files"]
    if not isinstance(files, dict):
        issues.append(Issue("invalid_manifest_files", "manifest.files must be an object", "manifest.json.files"))
        return
    for declared, spec in files.items():
        if not isinstance(declared, str) or not declared:
            issues.append(Issue("invalid_manifest_path", "manifest file key must be a path", "manifest.json.files"))
            continue
        # Support both {"file.jsonl": {"sha256": ..., "row_count": ...}}
        # and the common list-item form {"path": ..., "sha256": ...}.
        relative = declared
        if isinstance(spec, dict):
            relative = spec.get("path", declared)
            digest = spec.get("sha256")
            row_count = spec.get("row_count")
        else:
            digest, row_count = spec, None
        if not _string(relative, "manifest.json.files.path", issues=issues):
            continue
        candidate = (package / relative).resolve()
        if candidate.is_absolute() and (candidate != package.resolve() and package.resolve() not in candidate.parents):
            issues.append(Issue("unsafe_manifest_path", "manifest file path escapes package", f"manifest.json.files.{declared}"))
            continue
        if not candidate.is_file():
            issues.append(Issue("missing_manifest_file", "manifest references a missing file", f"manifest.json.files.{declared}"))
            continue
        if not isinstance(digest, str) or not HEX64.fullmatch(digest):
            issues.append(Issue("invalid_hash", "manifest sha256 must be a 64-character hexadecimal digest", f"manifest.json.files.{declared}"))
        elif hashlib.sha256(candidate.read_bytes()).hexdigest().lower() != digest.lower():
            issues.append(Issue("manifest_hash_mismatch", "manifest sha256 does not match file", f"manifest.json.files.{declared}"))
        if row_count is not None:
            if type(row_count) is not int or row_count < 0:
                issues.append(Issue("invalid_row_count", "manifest row_count must be a nonnegative integer", f"manifest.json.files.{declared}"))
            elif relative.endswith(".jsonl"):
                actual_count = sum(1 for line in candidate.read_text(encoding="utf-8").splitlines() if line.strip())
                if actual_count != row_count:
                    issues.append(Issue("row_count_mismatch", f"manifest declares {row_count}, found {actual_count}", f"manifest.json.files.{declared}"))


def _aliases(row: Mapping[str, Any], kind: str) -> dict[str, Any]:
    """Resolve published legacy spellings in memory; never rewrite source rows."""
    result = dict(row)
    aliases = {
        "fixtures.jsonl": {"kickoff_at": ("kickoff",), "fixture_source_url": ("identity_source_url", "source_url")},
        "documents.jsonl": {"canonical_url": ("url",), "body_text": ("body",),
            "source_time_raw": ("original_time_text",), "timezone_basis": ("timing_basis",),
            "time_precision": ("published_precision",), "time_evidence": ("timing_basis",),
            "document_type": ("document_kind",), "match_binding_basis": ("fixture_binding_basis",)},
        "attempts.jsonl": {"started_at": ("request_started_at", "requested_at"),
            "completed_at": ("response_observed_at", "response_at"), "http_status": ("status_code",),
            "error_code": ("error_class", "error_type"), "token_usage_if_available": ("token_cost",),
            "monetary_cost_if_known": ("monetary_cost",)},
        "context_snapshots.jsonl": {"observed_at": ("snapshot_observed_at",), "quote_at": ("quote_time", "odds_quote_time")},
        "labels/results.jsonl": {"regulation_home_goals": ("home_score",), "regulation_away_goals": ("away_score",),
            "result_source_url": ("source_url",)},
    }
    for key, alternatives in aliases.get(kind, {}).items():
        if key not in result:
            for alternative in alternatives:
                if alternative in row:
                    result[key] = row[alternative]
                    break
    if kind == "documents.jsonl" and "physical_match_ids" not in result and "physical_match_id" in row:
        result["physical_match_ids"] = [row["physical_match_id"]]
    if kind == "attempts.jsonl":
        if "tool" not in result and "action_type" in row:
            result["tool"] = row["action_type"]
        cost = row.get("cost")
        if isinstance(cost, dict):
            for target, source in (("request_count", "requests"), ("latency_ms", "latency_ms"), ("monetary_cost_if_known", "money")):
                if target not in result and source in cost:
                    result[target] = cost[source]
    return result


def _finite_nonnegative(value: Any) -> bool:
    return type(value) in {int, float} and math.isfinite(value) and value >= 0


def _prediction(value: Any, path: str, issues: list[Issue]) -> None:
    if value is None:
        return
    if (not isinstance(value, dict) or set(value) != {"home", "draw", "away"}
            or any(not _finite_nonnegative(x) or x > 1 for x in value.values())
            or not math.isclose(sum(value.values()), 1.0, abs_tol=1e-8)):
        issues.append(Issue("invalid_prediction", "prediction must be null or normalized home/draw/away probabilities", path))


def _availability(row: Mapping[str, Any], path: str, issues: list[Issue]) -> datetime | None:
    _required(row, ("available_at",), path, issues)
    available = _parse_time(row.get("available_at"), path=f"{path}.available_at", issues=issues)
    if available is None and row.get("availability_status") != "unknown":
        issues.append(Issue("availability_unknown", "null available_at requires availability_status=unknown", path))
    if available is not None:
        if row.get("availability_status") == "unknown":
            issues.append(Issue("availability_conflict", "unknown availability must retain null available_at", path))
        if not isinstance(row.get("availability_basis"), str) or not row["availability_basis"].strip():
            severity = "warning" if row.get("information_mode") != "prospective_capture" else "error"
            issues.append(Issue("availability_evidence_missing", "available_at requires an explicit Agent availability basis", path, severity))
    return available


def _policy(row: Mapping[str, Any], path: str, issues: list[Issue]) -> None:
    _required(row, ("policy_arm", "policy_version", "selection_basis", "selection_probability", "selection_probability_basis"), path, issues)
    if row.get("policy_arm") not in POLICY_ARMS | {"not_configured"}:
        issues.append(Issue("invalid_policy_arm", "unsupported policy arm", f"{path}.policy_arm"))
    if row.get("policy_version") is not None:
        _string(row.get("policy_version"), f"{path}.policy_version", issues)
    elif row.get("policy_arm") not in {"not_configured"}:
        issues.append(Issue("policy_version_missing", "configured policy must retain policy_version", f"{path}.policy_version", "warning"))
    probability = row.get("selection_probability")
    basis = row.get("selection_probability_basis")
    if probability is not None:
        if not _finite_nonnegative(probability) or probability > 1:
            issues.append(Issue("invalid_propensity", "selection probability must be finite and in [0,1]", path))
        if not isinstance(basis, str) or not basis.strip() or basis == "not_recorded":
            issues.append(Issue("propensity_basis_missing", "recorded probability requires an actual policy probability basis", path))
        if row.get("selection_basis") in {"manual_predeclared", "not_configured"}:
            issues.append(Issue("fabricated_propensity", "manual or unconfigured choices cannot claim a propensity", path))
        if row.get("selected_by_policy") is True and probability == 0:
            issues.append(Issue("zero_selected_propensity", "a selected action cannot have zero selection probability", path))
    elif row.get("selection_basis") == "configured_numeric_policy" and basis not in {"not_recorded", "not_configured"}:
        issues.append(Issue("propensity_missing", "configured policy must record a probability or explicitly mark it unavailable", path))
    if row.get("selection_basis") == "configured_numeric_policy":
        if probability is None:
            issues.append(Issue("numeric_policy_evidence_missing", "numeric policy requires its computed selection probability", path))
        for field in ("expected_loss_reduction", "acquisition_cost", "net_voi"):
            value = row.get(field)
            if type(value) not in {int, float} or not math.isfinite(value):
                issues.append(Issue("numeric_policy_evidence_missing", f"numeric policy requires actual {field}", f"{path}.{field}"))
    if probability is None and row.get("selection_basis") in {"uniform_exploration", "random_explore"}:
        issues.append(Issue("propensity_missing", "random exploration must retain the actual sampling probability", path))
    if row.get("policy_arm") == "not_configured" and probability is not None:
        # Historical exports may carry a runtime sampler's probability even
        # though no configured arm was persisted.  Preserve this as a warning;
        # it is ineligible for empirical-g training until repaired.
        issues.append(Issue("fabricated_propensity", "an unconfigured policy arm cannot claim a propensity", path, "warning"))


def validate_stream(package: str | Path) -> dict[str, Any]:
    """Validate one package without modifying any file.

    The result has ``valid``, ``errors``, ``warnings`` and compact row counts.
    Missing or malformed fields are reported together rather than raising on
    the first row.  The optional ``raise_on_error`` behaviour belongs to the
    CLI; this function is intentionally convenient for tests and callers.
    """
    root = Path(package).expanduser().resolve()
    issues: list[Issue] = []
    if not root.is_dir():
        return {"valid": False, "package": str(root), "errors": [Issue("missing_package", "package directory is missing", str(root)).as_dict()], "warnings": [], "counts": {}}
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        issues.append(Issue("missing_file", "manifest.json is missing", "manifest.json"))
        manifest = None
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            issues.append(Issue("invalid_manifest", str(exc), "manifest.json"))
            manifest = None
    if manifest is not None:
        _validate_manifest(root, manifest, issues=issues)

    declared_schemas = {manifest.get(name) for name in ("schema_version", "schema")
                        if isinstance(manifest, dict) and isinstance(manifest.get(name), str)}
    v2 = PROSPECTIVE_SCHEMA in declared_schemas
    if v2 and len(declared_schemas) > 1:
        issues.append(Issue("schema_conflict", "schema and schema_version must agree", "manifest.json"))
    originals = {relative: _rows(root, relative, issues=issues) for relative in DATA_FILES}
    rows = {relative: [_aliases(row, relative) for row in values] for relative, values in originals.items()}
    fixtures = _unique(rows["fixtures.jsonl"], "physical_match_id", "fixtures.jsonl", issues=issues)
    documents = _unique(rows["documents.jsonl"], "document_id", "documents.jsonl", issues=issues)
    discoveries = _unique(rows["discovery.jsonl"], "discovery_id", "discovery.jsonl", issues=issues)
    attempts = _unique(rows["attempts.jsonl"], "attempt_id", "attempts.jsonl", issues=issues)
    versions = {row["version_id"] for row in rows["documents.jsonl"] if isinstance(row.get("version_id"), str)}
    document_versions = {key: row.get("version_id") for key, row in documents.items()}
    for row in rows["attempts.jsonl"]:
        if "returned_version_ids" not in row and "returned_document_id" in row:
            document_id = row["returned_document_id"]
            version_id = None if document_id is None else document_versions.get(document_id)
            row["returned_version_ids"] = [] if version_id is None else [version_id]

    fixture_times: dict[str, datetime] = {}
    for index, row in enumerate(rows["fixtures.jsonl"], 1):
        path = f"fixtures.jsonl:{index}"
        _required(row, ("physical_match_id", "competition", "season", "home_team", "away_team", "kickoff_at", "fixture_source_url"), path, issues=issues)
        kickoff = _parse_time(row.get("kickoff_at"), path=f"{path}.kickoff_at", issues=issues, required=True)
        if kickoff is not None and isinstance(row.get("physical_match_id"), str):
            fixture_times[row["physical_match_id"]] = kickoff
        _string(row.get("competition"), f"{path}.competition", issues=issues)
        _string(row.get("season"), f"{path}.season", issues=issues)
        _string(row.get("home_team"), f"{path}.home_team", issues=issues)
        _string(row.get("away_team"), f"{path}.away_team", issues=issues)
        _string(row.get("fixture_source_url"), f"{path}.fixture_source_url", issues=issues)
        if v2:
            prospective_row = row.get("information_mode") == "prospective_capture"
            required_v2 = ["fixture_id", "information_mode", "policy_arm", "policy_version", "random_seed", "propensity_logging"]
            if prospective_row:
                required_v2 += ["registered_at", "registration_source", "pre_kickoff_frozen", "arm_assigned_at", "raw_path", "raw_sha256"]
            else:
                # Reconstructed/archive rows retain null/unknown timing and
                # provenance where the historical source cannot prove it.
                required_v2 += ["registered_at", "registration_source", "pre_kickoff_frozen", "arm_assigned_at"]
            _required(row, required_v2, path, issues)
            prospective = row.get("information_mode") == "prospective_capture"
            if row.get("information_mode") not in INFORMATION_MODES:
                issues.append(Issue("invalid_registration_mode", "invalid fixture information_mode", path))
            for field in ("registered_at", "arm_assigned_at"):
                stamp = _parse_time(row.get(field), path=f"{path}.{field}", issues=issues, required=prospective)
                if prospective:
                    _before(stamp, kickoff, "post_kickoff_registration", f"{field} must precede kickoff", f"{path}.{field}", issues, equal=False)
            if prospective and row.get("pre_kickoff_frozen") is not True:
                issues.append(Issue("invalid_registration_mode", "prospective fixture registration must be frozen before kickoff", path))
            if not prospective:
                issues.append(Issue("historical_fixture", "non-prospective fixture cannot establish a prospective registration", path, "warning"))
            if row.get("policy_arm") not in POLICY_ARMS | ({"not_configured"} if not prospective else set()):
                issues.append(Issue("invalid_policy_arm", "unsupported policy arm", f"{path}.policy_arm"))
            if row.get("policy_arm") == "random_explore" and type(row.get("random_seed")) is not int:
                issues.append(Issue("missing_random_seed", "random exploration requires its actual integer seed", path))
            if type(row.get("propensity_logging")) is not bool:
                issues.append(Issue("invalid_type", "propensity_logging must be boolean", f"{path}.propensity_logging"))
            if prospective or row.get("raw_path") is not None or row.get("raw_sha256") is not None:
                _safe_raw(root, row.get("raw_path"), row.get("raw_sha256"), f"{path}.raw_path", issues)
            else:
                issues.append(Issue("historical_registration_raw_missing", "historical fixture registration has no retained original response", path, "warning"))

    # Every document's timestamp and provenance fields are checked independently
    # so a later row cannot hide a malformed clock behind a missing reference.
    for index, row in enumerate(rows["documents.jsonl"], 1):
        path = f"documents.jsonl:{index}"
        legacy_document = not v2 and "body" in originals["documents.jsonl"][index - 1] and "body_text" not in originals["documents.jsonl"][index - 1]
        if v2:
            required = ("document_id", "physical_match_ids", "canonical_url", "publisher", "title", "body_text", "language",
                        "source_published_at", "source_modified_at", "captured_at", "source_time_raw", "timezone_basis",
                        "information_mode", "available_at", "raw_path", "raw_sha256", "previous_version_id",
                        "document_type", "match_binding_basis")
        else:
            required = ("document_id", "version_id", "physical_match_ids", "canonical_url", "publisher", "title", "body_text", "language", "source_published_at", "source_modified_at", "captured_at", "source_time_raw", "timezone_basis", "time_precision", "time_evidence", "information_mode", "archive_snapshot_at", "archive_snapshot_url", "historical_snapshot_proven", "raw_path", "raw_sha256", "previous_version_id", "document_type", "match_binding_basis", "duplicate_cluster_id")
            if legacy_document:
                required = tuple(field for field in required if field not in {"archive_snapshot_at", "archive_snapshot_url", "duplicate_cluster_id"})
        _required(row, required, path, issues=issues)
        text_fields = ("document_id", "canonical_url", "publisher", "title", "body_text", "language", "source_time_raw", "timezone_basis", "document_type", "match_binding_basis")
        if not v2:
            text_fields += ("version_id", "time_precision", "time_evidence")
            if not legacy_document:
                text_fields += ("duplicate_cluster_id",)
        for key in text_fields:
            _string(row.get(key), f"{path}.{key}", issues, allow_empty=True)
        match_ids = row.get("physical_match_ids")
        if not isinstance(match_ids, list) or any(not isinstance(item, str) for item in match_ids):
            issues.append(Issue("invalid_type", "physical_match_ids must be a list of strings", f"{path}.physical_match_ids"))
        else:
            for match_id in match_ids:
                if match_id not in fixtures:
                    issues.append(Issue("unknown_match", f"document references unknown match {match_id!r}", f"{path}.physical_match_ids"))
        mode = row.get("information_mode")
        if mode not in INFORMATION_MODES:
            issues.append(Issue("invalid_enum", f"information_mode must be one of {sorted(INFORMATION_MODES)}", f"{path}.information_mode"))
        published = _parse_time(row.get("source_published_at"), path=f"{path}.source_published_at", issues=issues)
        modified = _parse_time(row.get("source_modified_at"), path=f"{path}.source_modified_at", issues=issues)
        explicit_unknown_capture = (v2 and row.get("captured_at") is None
                                    and row.get("capture_status") == "unknown"
                                    and row.get("capture_timing_basis") == "unknown")
        captured = _parse_time(row.get("captured_at"), path=f"{path}.captured_at", issues=issues,
                               required=not explicit_unknown_capture)
        if explicit_unknown_capture:
            issues.append(Issue("document_capture_time_unknown",
                "archive capture time was not independently measured; response observation and availability are not capture proof",
                f"{path}.captured_at", "warning"))
        observed = _parse_time(row.get("observed_at"), path=f"{path}.observed_at", issues=issues) if v2 else None
        archive_at = _parse_time(row.get("archive_snapshot_at"), path=f"{path}.archive_snapshot_at", issues=issues) if not v2 else None
        available = _availability(row, path, issues) if v2 else None
        if v2:
            _visibility_extension(row, path, issues)
            if "document_admitted" in row:
                document_available = _parse_time(row.get("document_available_at"), path=f"{path}.document_available_at", issues=issues)
                if row.get("document_admitted") is not True and available is not None:
                    issues.append(Issue("document_availability_without_admission", "unadmitted document cannot have Agent available_at", path))
                if available != document_available:
                    issues.append(Issue("document_availability_mismatch", "available_at must equal the proven document availability clock", path))
            if row.get("capture_timing_basis") == "measured_local_utc_archive_save_return":
                save_started = _parse_time(row.get("archive_save_started_at"), path=f"{path}.archive_save_started_at", issues=issues, required=True)
                save_returned = _parse_time(row.get("archive_save_returned_at"), path=f"{path}.archive_save_returned_at", issues=issues, required=True)
                _before(observed, save_started, "archive_save_time_order", "archive save started before response observation", path, issues)
                _before(save_started, save_returned, "archive_save_time_order", "archive save return preceded save invocation", path, issues)
                if captured != save_returned or row.get("archive_return_timing_basis") != "measured_local_utc_archive_save_return":
                    issues.append(Issue("archive_return_unproven", "captured_at must use the measured archive return bound to this response", path))
                if row.get("persistence_boundary") != "return_observed_fsync_not_instrumented":
                    issues.append(Issue("archive_persistence_overclaim", "archive return does not prove an instrumented fsync event", path))
        _before(published, modified, "source_time_order", "source_modified_at precedes source_published_at", f"{path}.source_modified_at", issues=issues)
        _before(published, captured, "source_time_order", "source_published_at is after captured_at", f"{path}.captured_at", issues=issues)
        _before(modified, captured, "source_time_order", "source_modified_at is after captured_at", f"{path}.captured_at", issues=issues)
        _before(archive_at, captured, "snapshot_time_order", "archive_snapshot_at is after captured_at", f"{path}.archive_snapshot_at", issues=issues)
        previous = row.get("previous_version_id")
        if previous is not None and not isinstance(previous, str):
            issues.append(Issue("invalid_type", "previous_version_id must be a string or null", f"{path}.previous_version_id"))
        elif isinstance(previous, str) and previous not in {item.get("version_id") for item in rows["documents.jsonl"]}:
            issues.append(Issue("unknown_version", f"previous_version_id {previous!r} is not present", f"{path}.previous_version_id"))
        for match_id in match_ids if isinstance(match_ids, list) else []:
            kickoff = fixture_times.get(match_id)
            if mode == "prospective_capture":
                _before(observed, kickoff, "post_kickoff_document_observation", "prospective response was observed at or after kickoff", f"{path}.observed_at", issues, equal=False)
                _before(captured, kickoff, "post_kickoff_document", "prospective document was captured at or after kickoff", f"{path}.captured_at", issues, equal=False)
                _before(available, kickoff, "post_kickoff_document_availability", "prospective document became available at or after kickoff", f"{path}.available_at", issues, equal=False)
        if v2 and row.get("information_mode") == "prospective_capture" and row.get("available_at") is None:
            # A live capture cannot silently substitute source publication/capture time.
            if row.get("availability_status") != "unknown":
                issues.append(Issue("unknown_availability", "prospective document with unknown availability must say so explicitly", path))
        _safe_raw(root, row.get("raw_path"), row.get("raw_sha256"), f"{path}.raw_path", issues=issues)

    candidate_ids: set[str] = set()
    candidate_locations: dict[str, tuple[str, str]] = {}
    for index, row in enumerate(rows["discovery.jsonl"], 1):
        path = f"discovery.jsonl:{index}"
        required = ("discovery_id", "physical_match_id", "observed_at", "provider", "visible_candidates", "status", "truncation_or_cap")
        legacy_reconstructed_menu = not v2 and (row.get("information_mode") == "historical_reconstruction"
            or row.get("original_agent_menu_proven") is False)
        if legacy_reconstructed_menu:
            required = tuple(field for field in required if field != "truncation_or_cap")
        if v2:
            required += ("menu_id", "policy_version", "original_agent_menu_proven", "information_mode")
            # Historical reconstruction may not have an immutable discovery
            # response.  Keep the fields present when available, but do not
            # turn an unknown archive provenance into a false hard failure.
            required += ("raw_path", "raw_sha256")
        _required(row, required, path, issues=issues)
        if row.get("physical_match_id") not in fixtures:
            issues.append(Issue("unknown_match", "discovery references unknown match", f"{path}.physical_match_id"))
        observed = _parse_time(row.get("observed_at"), path=f"{path}.observed_at", issues=issues, required=True)
        kickoff = fixture_times.get(row.get("physical_match_id"))
        if not legacy_reconstructed_menu:
            _before(observed, kickoff, "post_kickoff_discovery", "discovery was observed at or after kickoff", f"{path}.observed_at", issues, equal=False)
        else:
            issues.append(Issue("historical_menu", "later reconstructed menu is not an original prospective Agent menu", path, "warning"))
        _string(row.get("provider"), f"{path}.provider", issues=issues)
        if v2:
            _string(row.get("menu_id"), f"{path}.menu_id", issues=issues)
            if row.get("policy_version") is not None:
                _string(row.get("policy_version"), f"{path}.policy_version", issues=issues)
            elif row.get("information_mode") == "prospective_capture":
                issues.append(Issue("policy_version_missing", "prospective menu must retain policy_version", f"{path}.policy_version"))
            if type(row.get("original_agent_menu_proven")) is not bool:
                issues.append(Issue("invalid_type", "original_agent_menu_proven must be boolean", f"{path}.original_agent_menu_proven"))
            if row.get("information_mode") not in INFORMATION_MODES:
                issues.append(Issue("invalid_enum", "invalid discovery information_mode", f"{path}.information_mode"))
            elif row.get("information_mode") == "prospective_capture" and row.get("original_agent_menu_proven") is not True:
                issues.append(Issue("menu_not_proven", "prospective capture menus must prove the original Agent-visible menu", path))
            if row.get("information_mode") == "prospective_capture" or row.get("raw_path") is not None or row.get("raw_sha256") is not None:
                _safe_raw(root, row.get("raw_path"), row.get("raw_sha256"), f"{path}.raw_path", issues=issues)
            else:
                issues.append(Issue("historical_menu_raw_missing", "historical menu has no retained discovery response", path, "warning"))
        if not isinstance(row.get("visible_candidates"), list):
            issues.append(Issue("invalid_type", "visible_candidates must be a list", f"{path}.visible_candidates"))
        else:
            for candidate_index, candidate in enumerate(row["visible_candidates"], 1):
                cpath = f"{path}.visible_candidates[{candidate_index}]"
                if not isinstance(candidate, dict):
                    issues.append(Issue("invalid_row", "candidate must be an object", cpath))
                    continue
                _required(candidate, ("candidate_id", "url", "title", "snippet", "rank"), cpath, issues=issues)
                _pre_read_metadata(candidate, row, cpath + ".pre_read_metadata", issues)
                cid = candidate.get("candidate_id")
                if isinstance(cid, str):
                    if cid in candidate_ids:
                        issues.append(Issue("duplicate_id", f"duplicate candidate_id {cid!r}", f"{cpath}.candidate_id"))
                    candidate_ids.add(cid)
                    candidate_locations[cid] = (str(row.get("discovery_id")), str(row.get("physical_match_id")))
                if type(candidate.get("rank")) is not int or candidate.get("rank") < 0:
                    issues.append(Issue("invalid_type", "candidate rank must be a nonnegative integer", f"{cpath}.rank"))
                if v2 and (not isinstance(candidate.get("action_type"), str) or not candidate.get("action_type")):
                    issues.append(Issue("missing_field", "v2 candidate action_type is required", f"{cpath}.action_type"))

    for index, row in enumerate(rows["attempts.jsonl"], 1):
        path = f"attempts.jsonl:{index}"
        if v2:
            _required(row, ("attempt_id", "physical_match_id", "discovery_id", "candidate_id", "action_id", "selected_by_policy",
                            "selection_basis", "policy_arm", "policy_version", "selection_probability", "selection_probability_basis",
                            "decision_at", "requested_at", "response_observed_at", "status", "source", "request_count", "latency_ms",
                            "raw_path", "raw_sha256"), path, issues=issues)
        else:
            _required(row, ("attempt_id", "physical_match_id", "discovery_id", "candidate_id", "tool", "started_at", "completed_at", "status", "http_status", "error_code", "returned_version_ids", "request_count", "latency_ms", "token_usage_if_available", "monetary_cost_if_known"), path, issues=issues)
        if row.get("physical_match_id") not in fixtures:
            issues.append(Issue("unknown_match", "attempt references unknown match", f"{path}.physical_match_id"))
        if row.get("discovery_id") is not None and row.get("discovery_id") not in discoveries:
            issues.append(Issue("unknown_discovery", "attempt references unknown discovery", f"{path}.discovery_id"))
        if row.get("candidate_id") is not None and row.get("candidate_id") not in candidate_ids:
            issues.append(Issue("unknown_candidate", "attempt references unknown candidate", f"{path}.candidate_id"))
        location = candidate_locations.get(row.get("candidate_id"))
        if location is not None:
            if location[0] != str(row.get("discovery_id")):
                issues.append(Issue("candidate_menu_mismatch", "attempt candidate belongs to a different discovery menu", f"{path}.candidate_id"))
            if location[1] != str(row.get("physical_match_id")):
                issues.append(Issue("candidate_fixture_mismatch", "attempt candidate belongs to a different fixture", f"{path}.candidate_id"))
        if v2:
            _policy(row, path, issues)
            _visibility_extension(row, path, issues,
                fixture_times.get(row.get("physical_match_id")) if row.get("information_mode") == "prospective_capture" else None)
            if type(row.get("selected_by_policy")) is not bool:
                issues.append(Issue("invalid_type", "selected_by_policy must be boolean", f"{path}.selected_by_policy"))
            decision = _parse_time(row.get("decision_at"), path=f"{path}.decision_at", issues=issues, required=True)
            kickoff = fixture_times.get(row.get("physical_match_id"))
            linked_discovery = discoveries.get(row.get("discovery_id"))
            historical_attempt = isinstance(linked_discovery, Mapping) and linked_discovery.get("information_mode") != "prospective_capture"
            if not historical_attempt:
                _before(decision, kickoff, "post_kickoff_decision", "decision must precede kickoff", f"{path}.decision_at", issues, equal=False)
            status = row.get("status")
            if row.get("selected_by_policy") is False:
                if status not in {"not_requested", "not_configured"}:
                    issues.append(Issue("unselected_status", "unselected actions must use status=not_requested or not_configured", f"{path}.status"))
                if not isinstance(row.get("not_selected_reason"), str) or not row["not_selected_reason"].strip():
                    issues.append(Issue("missing_not_selected_reason", "unselected actions require not_selected_reason", path))
                if row.get("request_count") != 0:
                    issues.append(Issue("unselected_request_count", "not_requested actions must have request_count=0", f"{path}.request_count"))
                if row.get("latency_ms") is not None:
                    severity = "warning" if historical_attempt else "error"
                    issues.append(Issue("unselected_latency", "not_requested actions must have null latency_ms", f"{path}.latency_ms", severity))
            elif status in {"not_requested", "not_configured"}:
                if row.get("request_count") != 0:
                    issues.append(Issue("unconfigured_request_count", "not_requested/not_configured actions must have request_count=0", f"{path}.request_count"))
                if row.get("latency_ms") is not None:
                    issues.append(Issue("unconfigured_latency", "not_requested/not_configured actions must have null latency_ms", f"{path}.latency_ms"))
            for field in ("requested_at", "response_observed_at"):
                required_time = row.get("selected_by_policy") is True and not (field == "response_observed_at" and status in {"failed", "timeout", "blocked", "not_requested", "not_configured"})
                stamp = _parse_time(row.get(field), path=f"{path}.{field}", issues=issues, required=required_time)
                _before(decision, stamp, "attempt_time_order", "attempt occurred before its decision", f"{path}.{field}", issues)
                if not historical_attempt:
                    _before(stamp, kickoff, "post_kickoff_attempt", "attempt occurred at or after kickoff", f"{path}.{field}", issues, equal=False)
            started = _parse_time(row.get("request_started_at", row.get("requested_at")), path=f"{path}.request_started_at", issues=issues)
            completed = _parse_time(row.get("response_observed_at"), path=f"{path}.response_observed_at", issues=issues)
        else:
            tool_clock_unknown = (row.get("information_mode") == "historical_reconstruction"
                and row.get("tool") == "current_web_search" and row.get("request_clock_status") == "not_measured"
                and row.get("actual_historical_agent_action") is False and row.get("raw_path") is not None)
            started = _parse_time(row.get("started_at"), path=f"{path}.started_at", issues=issues, required=not tool_clock_unknown)
            if tool_clock_unknown and started is None:
                issues.append(Issue("historical_tool_request_clock_unknown", "external discovery request start was not measured; no prospective time proof", path, "warning"))
            completed = _parse_time(row.get("completed_at"), path=f"{path}.completed_at", issues=issues, required=True)
        _before(started, completed, "attempt_time_order", "attempt completed before it started", f"{path}.completed_at", issues=issues)
        kickoff = fixture_times.get(row.get("physical_match_id"))
        returned_doc = documents.get(row.get("returned_document_id")) if isinstance(row.get("returned_document_id"), str) else None
        legacy_reconstructed_attempt = (v2 and isinstance(discoveries.get(row.get("discovery_id")), Mapping)
            and discoveries.get(row.get("discovery_id"), {}).get("information_mode") != "prospective_capture") or (not v2 and (row.get("historical_capture") is True
            or row.get("information_mode") == "historical_reconstruction"
            or "response_at" in originals["attempts.jsonl"][index - 1] and returned_doc is not None
            and returned_doc.get("information_mode") == "historical_reconstruction"))
        if not legacy_reconstructed_attempt:
            _before(completed, kickoff, "post_kickoff_attempt", "attempt completed at or after kickoff", f"{path}.completed_at", issues, equal=False)
        else:
            issues.append(Issue("historical_attempt", "later request returned historical reconstruction, not prospective evidence", path, "warning"))
        status = row.get("status")
        if not isinstance(status, str) or not status.strip():
            issues.append(Issue("invalid_type", "attempt status must be a non-empty string", f"{path}.status"))
        elif status not in ATTEMPT_STATUSES:
            issues.append(Issue("unknown_status", f"unsupported attempt status {status!r}", f"{path}.status", "warning"))
        returned = row.get("returned_version_ids", [])
        if not isinstance(returned, list) or any(not isinstance(item, str) for item in returned):
            issues.append(Issue("invalid_type", "returned_version_ids must be a list of strings", f"{path}.returned_version_ids"))
        elif not v2:
            version_ids = {item.get("version_id") for item in rows["documents.jsonl"]}
            for version_id in returned:
                if version_id not in version_ids:
                    issues.append(Issue("unknown_version", f"returned version {version_id!r} is not present", f"{path}.returned_version_ids"))
        external_http_count_unknown = (row.get("tool") == "current_web_search"
            and row.get("information_mode") == "historical_reconstruction"
            and row.get("actual_historical_agent_action") is False
            and row.get("provider_internal_http_count") is None and row.get("tool_call_count") == 1
            and row.get("raw_path") is not None)
        if (v2 or external_http_count_unknown) and row.get("selected_by_policy") is True and row.get("request_count") is None:
            issues.append(Issue("unknown_request_count", "selected action request cost was not recorded", f"{path}.request_count", "warning"))
        elif type(row.get("request_count")) is not int or row.get("request_count") < 0:
            issues.append(Issue("invalid_type", "request_count must be a nonnegative integer", f"{path}.request_count"))
        if row.get("latency_ms") is not None and (type(row.get("latency_ms")) not in {int, float} or row.get("latency_ms") < 0):
            issues.append(Issue("invalid_type", "latency_ms must be nonnegative", f"{path}.latency_ms"))
        if v2 and row.get("selected_by_policy") is True and status not in {"not_requested", "not_configured"} and type(row.get("request_count")) is int and row["request_count"] < 1:
            issues.append(Issue("selected_request_count", "selected actions require at least one request", f"{path}.request_count"))
        if v2:
            if row.get("source") is not None or row.get("selected_by_policy") is True:
                _string(row.get("source"), f"{path}.source", issues=issues)
            returned_document = row.get("returned_document_id")
            if returned_document is not None and returned_document not in documents:
                issues.append(Issue("unknown_document", "attempt references unknown returned_document_id", f"{path}.returned_document_id"))
            if row.get("selected_by_policy") is False:
                if row.get("raw_path") is not None or row.get("raw_sha256") is not None:
                    _safe_raw(root, row.get("raw_path"), row.get("raw_sha256"), f"{path}.raw_path", issues=issues)
            elif row.get("raw_path") is not None:
                _safe_raw(root, row.get("raw_path"), row.get("raw_sha256"), f"{path}.raw_path", issues=issues)

    for index, row in enumerate(rows["context_snapshots.jsonl"], 1):
        path = f"context_snapshots.jsonl:{index}"
        if not isinstance(row, dict):
            continue
        match_id = row.get("physical_match_id")
        if match_id is not None and match_id not in fixtures:
            issues.append(Issue("unknown_match", "context snapshot references unknown match", f"{path}.physical_match_id"))
        if v2:
            _required(row, ("snapshot_id", "sequence", "snapshot_kind", "observed_at", "available_at", "captured_at",
                            "prediction_cutoff", "known_information", "unknown_information", "prediction_before",
                            "prediction_after", "information_mode"), path, issues)
            if type(row.get("sequence")) is not int or row.get("sequence") < 0:
                issues.append(Issue("invalid_sequence", "sequence must be a nonnegative integer", f"{path}.sequence"))
            if row.get("snapshot_kind") not in {"initial", "before_action", "after_action", "final"}:
                issues.append(Issue("invalid_snapshot_kind", "unsupported snapshot_kind", f"{path}.snapshot_kind"))
            if row.get("information_mode") not in INFORMATION_MODES:
                issues.append(Issue("invalid_enum", "invalid context information_mode", f"{path}.information_mode"))
            observed = _parse_time(row.get("observed_at"), path=f"{path}.observed_at", issues=issues)
            captured = _parse_time(row.get("captured_at"), path=f"{path}.captured_at", issues=issues)
            for clock, name in ((observed, "observed_at"), (captured, "captured_at")):
                if clock is None and row.get(name) is None:
                    issues.append(Issue("context_clock_unknown", "unknown context clock is not proof of pre-kickoff capture", f"{path}.{name}", "warning"))
            context_prospective = row.get("information_mode") == "prospective_capture"
            cutoff = _parse_time(row.get("prediction_cutoff"), path=f"{path}.prediction_cutoff", issues=issues, required=context_prospective)
            available = _availability(row, path, issues)
            kickoff = fixture_times.get(match_id)
            if "prediction_cutoff_basis" in row:
                _before(cutoff, observed, "context_cutoff_after_snapshot", "snapshot prediction cutoff follows its observation", f"{path}.prediction_cutoff", issues)
            if context_prospective:
                for value, name in ((observed, "observed_at"), (captured, "captured_at"), (cutoff, "prediction_cutoff"), (available, "available_at")):
                    _before(value, kickoff, "post_kickoff_context", f"{name} must precede kickoff", f"{path}.{name}", issues, equal=False)
            else:
                issues.append(Issue("historical_context", "context snapshot is retrospective or reconstructed", path, "warning"))
            if not isinstance(row.get("known_information"), list) or any(not isinstance(item, str) for item in row["known_information"]):
                issues.append(Issue("invalid_type", "known_information must be a list of strings", f"{path}.known_information"))
            if not isinstance(row.get("unknown_information"), list) or any(not isinstance(item, str) for item in row["unknown_information"]):
                issues.append(Issue("invalid_type", "unknown_information must be a list of strings", f"{path}.unknown_information"))
            # ``known_document_ids`` was added after the original v2 export.
            # Keep it optional for old packages, but validate the extension
            # whenever it is present so a malformed document menu cannot pass
            # silently as context state.
            if "known_document_ids" in row:
                if (not isinstance(row.get("known_document_ids"), list)
                        or any(not isinstance(item, str) or not item.strip()
                               for item in row["known_document_ids"])):
                    issues.append(Issue("invalid_type", "known_document_ids must be a list of non-empty strings", f"{path}.known_document_ids"))
            _prediction(row.get("prediction_before"), f"{path}.prediction_before", issues)
            _prediction(row.get("prediction_after"), f"{path}.prediction_after", issues)
            if row.get("prediction_status") in {"market_prior_only", "model_not_configured"}:
                if row.get("prediction_before") is not None or row.get("prediction_after") is not None:
                    issues.append(Issue("unconfigured_prediction", "an unfitted market prior is not a model prediction", path))
                if row.get("calibration_status") != "not_established":
                    issues.append(Issue("unconfigured_calibration", "unconfigured predictors cannot claim established calibration", f"{path}.calibration_status"))
            if "market_prior" in row:
                _prediction(row.get("market_prior"), f"{path}.market_prior", issues)
            for field in CONTEXT_FIELDS:
                if field not in row:  # Older v2 exports remain readable.
                    continue
                detail = row[field]
                field_path = f"{path}.{field}"
                if not isinstance(detail, dict):
                    issues.append(Issue("invalid_context_field", "context field requires a value and its own observed_at", field_path))
                    continue
                _required(detail, ("value", "observed_at"), field_path, issues)
                field_times = {name: _parse_time(detail.get(name), path=f"{field_path}.{name}", issues=issues)
                               for name in ("observed_at", "captured_at", "available_at")}
                if field == "odds":
                    _required(detail, ("quote_time", "captured_at"), field_path, issues)
                    field_times["quote_time"] = _parse_time(detail.get("quote_time"), path=f"{field_path}.quote_time", issues=issues)
                if context_prospective:
                    for name, stamp in field_times.items():
                        _before(stamp, kickoff, "post_kickoff_context_field", f"{field} {name} must precede kickoff", f"{field_path}.{name}", issues, equal=False)
                _before(field_times["observed_at"], observed, "context_field_observed_after_snapshot", "field observation follows the snapshot", f"{field_path}.observed_at", issues)
                _before(field_times["observed_at"], field_times["captured_at"], "context_field_time_order", "field observation follows capture", f"{field_path}.captured_at", issues)
            if row.get("raw_path") is not None:
                _safe_raw(root, row.get("raw_path"), row.get("raw_sha256"), f"{path}.raw_path", issues=issues)
            parent = row.get("parent_attempt_id")
            if row.get("snapshot_kind") == "after_action":
                _required(row, ("parent_attempt_id",), path, issues)
                if parent not in attempts:
                    issues.append(Issue("unknown_parent_attempt", "after_action snapshot must reference a known attempt", f"{path}.parent_attempt_id"))
                else:
                    parent_row = attempts[parent]
                    if parent_row.get("physical_match_id") != match_id:
                        issues.append(Issue("parent_fixture_mismatch", "after_action parent belongs to a different fixture", f"{path}.parent_attempt_id"))
                    if parent_row.get("selected_by_policy") is False:
                        issues.append(Issue("parent_not_selected", "after_action parent must be an executed selected action", f"{path}.parent_attempt_id"))
            elif parent is not None and parent not in attempts:
                issues.append(Issue("unknown_parent_attempt", "context parent_attempt_id is not a known attempt", f"{path}.parent_attempt_id"))
            if row.get("snapshot_kind") == "final":
                if context_prospective:
                    _required(row, ("stop_reason",), path, issues)
                    _string(row.get("stop_reason"), f"{path}.stop_reason", issues=issues)
                elif row.get("stop_reason") is None:
                    issues.append(Issue("historical_stop_reason_unknown", "historical final context has no observed stop reason", path, "warning"))
            continue
        as_of = _parse_time(row.get("as_of"), path=f"{path}.as_of", issues=issues) if "as_of" in row else None
        captured = _parse_time(row.get("captured_at"), path=f"{path}.captured_at", issues=issues) if "captured_at" in row else None
        _before(as_of, captured, "context_time_order", "context as_of is after captured_at", f"{path}.captured_at", issues=issues)
        quote = _parse_time(row.get("quote_at"), path=f"{path}.quote_at", issues=issues) if "quote_at" in row else None
        _before(quote, captured, "context_time_order", "quote_at is after captured_at", f"{path}.captured_at", issues=issues)

    for index, row in enumerate(rows["labels/results.jsonl"], 1):
        path = f"labels/results.jsonl:{index}"
        _required(row, ("physical_match_id", "regulation_home_goals", "regulation_away_goals", "result_status", "result_source_url", "result_published_at", "result_observed_at", "raw_path", "raw_sha256", "timing_basis"), path, issues=issues)
        match_id = row.get("physical_match_id")
        if match_id not in fixtures:
            issues.append(Issue("unknown_match", "result references unknown match", f"{path}.physical_match_id"))
            kickoff = None
        else:
            kickoff = fixture_times.get(match_id)
        observed = _parse_time(row.get("result_observed_at"), path=f"{path}.result_observed_at", issues=issues, required=True)
        published = _parse_time(row.get("result_published_at"), path=f"{path}.result_published_at", issues=issues)
        _before(kickoff, observed, "result_time_order", "result was observed before kickoff", f"{path}.result_observed_at", issues=issues)
        _before(published, observed, "result_time_order", "result publication is after result observation", f"{path}.result_observed_at", issues=issues)
        for key in ("regulation_home_goals", "regulation_away_goals"):
            if type(row.get(key)) is not int or row.get(key) < 0:
                issues.append(Issue("invalid_score", "regulation score must be a nonnegative integer", f"{path}.{key}"))
        _string(row.get("result_status"), f"{path}.result_status", issues=issues)
        _string(row.get("result_source_url"), f"{path}.result_source_url", issues=issues)
        _string(row.get("timing_basis"), f"{path}.timing_basis", issues=issues)
        _safe_raw(root, row.get("raw_path"), row.get("raw_sha256"), f"{path}.raw_path", issues=issues)

    errors = [issue.as_dict() for issue in issues if issue.severity == "error"]
    warnings = [issue.as_dict() for issue in issues if issue.severity == "warning"]
    return {"valid": not errors, "package": str(root), "errors": errors, "warnings": warnings,
            "counts": {relative: len(values) for relative, values in rows.items()}}


# Explicit aliases make the validator easy to discover without coupling callers
# to one spelling used by the command-line wrapper.
validate_information_stream = validate_stream
validate_package = validate_stream


__all__ = ["DATA_FILES", "Issue", "validate_stream", "validate_information_stream", "validate_package"]
