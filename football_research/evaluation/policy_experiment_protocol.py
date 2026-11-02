"""Schema and holdout gates for an independent autonomous-policy experiment.

Round85 showed that a historical archive can be perfectly paired while still
being unsuitable for an independent policy claim.  This module makes the
missing contract executable before a new run starts:

* every episode carries explicit identity, source, lead-time and policy
  version metadata;
* realized costs come from a nested receipt;
* public context is separated from evaluator labels;
* train, validation and test match IDs are disjoint, and test IDs are absent
  from every prior calibration/training manifest;
* a threshold/parameter freeze is recorded before the first test prediction.

The functions are read-only validators.  They do not turn an old archive into
an independent experiment by inference or by filling fields from neighboring
rows.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from typing import Any, Iterable, Mapping
import re


VERSION = "policy-experiment-protocol-v2"

REQUIRED_EPISODE_FIELDS = (
    "dataset_version",
    "fold",
    "match_id",
    "competition_id",
    "league",
    "kickoff",
    "prediction_cutoff",
    "lead_hours",
    "fixture_available_at",
    "strategy",
    "strategy_version",
    "candidate_set_version",
    "budget_money",
    "seed",
    "source_id",
    "source_version",
    "feature_manifest_hash",
    "policy_context_hash",
    "policy_parameter_hash",
    "evaluator_sidecar_ref",
    "final_probabilities",
    "cost",
)

FORBIDDEN_PUBLIC_KEYS = frozenset({
    "result", "outcome", "score", "home_score", "away_score", "home_goals",
    "away_goals", "final_score", "label", "target", "realized_loss",
    "sampled_result", "actual_result", "actual_outcome",
})


@dataclass(frozen=True)
class ProtocolIssue:
    code: str
    severity: str
    path: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "severity": self.severity,
            "path": self.path,
            "detail": self.detail,
        }


def _utc(value: Any, field: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    else:
        raise ValueError(f"{field} must be an ISO-8601 timestamp")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return float(value)


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value.strip()


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _public_times(value: Any, cutoff: datetime, path: str) -> list[ProtocolIssue]:
    """Check explicit availability timestamps; never infer missing provenance."""
    issues: list[ProtocolIssue] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if str(key) in {"available_at", "published_at", "observed_at", "retrieved_at"}:
                try:
                    if _utc(child, child_path) > cutoff:
                        _issue(issues, "PUBLIC_EVIDENCE_AFTER_CUTOFF", "error", child_path,
                               "public evidence timestamp follows prediction cutoff")
                except (TypeError, ValueError) as exc:
                    _issue(issues, "INVALID_PUBLIC_EVIDENCE_TIME", "error", child_path, str(exc))
            issues.extend(_public_times(child, cutoff, child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            issues.extend(_public_times(child, cutoff, f"{path}[{index}]"))
    return issues


def _public_forbidden(value: Any, path: str = "policy_visible_context") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            if key_text.lower() in FORBIDDEN_PUBLIC_KEYS:
                found.append(child_path)
            found.extend(_public_forbidden(child, child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(_public_forbidden(child, f"{path}[{index}]"))
    return found


def _issue(issues: list[ProtocolIssue], code: str, severity: str, path: str, detail: str) -> None:
    issues.append(ProtocolIssue(code=code, severity=severity, path=path, detail=detail))


def _valid_probabilities(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return False
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item)
           or item < 0 or item > 1 for item in value):
        return False
    return math.isclose(sum(value), 1.0, abs_tol=1e-8, rel_tol=0.0)


def audit_episode_schema(
    rows: Iterable[Mapping[str, Any]],
    *,
    require_public_context: bool = True,
) -> dict[str, Any]:
    """Audit future-run episode fields without deriving missing metadata."""
    issues: list[ProtocolIssue] = []
    rows = tuple(rows)
    missing_counts: Counter[str] = Counter()
    duplicate_keys: Counter[str] = Counter()
    seen_keys: Counter[tuple[str, str, float, str, Any]] = Counter()
    lead_mismatch = 0
    bad_cost_receipts = 0
    bad_probabilities = 0
    public_leak_paths = 0
    source_versions: set[tuple[str, str]] = set()
    dataset_versions: set[str] = set()
    match_ids: set[str] = set()
    if not rows:
        _issue(issues, "EMPTY_EPISODES", "error", "episodes", "at least one episode is required")

    for index, row in enumerate(rows):
        path = f"episodes[{index}]"
        if not isinstance(row, Mapping):
            _issue(issues, "ROW_NOT_MAPPING", "error", path, "episode must be a mapping")
            continue
        raw_match_id = row.get("match_id")
        if isinstance(raw_match_id, str) and raw_match_id.strip():
            match_ids.add(raw_match_id.strip())
        for field in REQUIRED_EPISODE_FIELDS:
            if field not in row or row.get(field) is None:
                missing_counts[field] += 1
                _issue(issues, "MISSING_REQUIRED_FIELD", "error", f"{path}.{field}", "required field is absent")
        for field in ("dataset_version", "match_id", "competition_id", "league", "strategy",
                      "strategy_version", "candidate_set_version", "source_id", "source_version",
                      "evaluator_sidecar_ref"):
            if field in row and row[field] is not None:
                try:
                    _text(row[field], field)
                except ValueError as exc:
                    _issue(issues, "INVALID_TEXT_FIELD", "error", f"{path}.{field}", str(exc))
        for field in ("fold", "seed"):
            try:
                _integer(row.get(field), field)
            except ValueError as exc:
                _issue(issues, "INVALID_INTEGER_FIELD", "error", f"{path}.{field}", str(exc))
        for field in ("feature_manifest_hash", "policy_context_hash", "policy_parameter_hash"):
            if not _sha256(row.get(field)):
                _issue(issues, "INVALID_SHA256", "error", f"{path}.{field}", "lowercase SHA-256 digest is required")
        # Duplicate detection must still run when unrelated metadata is absent.
        try:
            key = (_text(row.get("match_id"), "match_id"),
                   _utc(row.get("prediction_cutoff"), "prediction_cutoff").isoformat(),
                   _finite(row.get("budget_money"), "budget_money"),
                   _text(row.get("strategy"), "strategy"), _integer(row.get("seed"), "seed"))
            seen_keys[key] += 1
        except (TypeError, ValueError):
            pass  # The field validators report these invalid identities.
        try:
            match_id = _text(row.get("match_id"), f"{path}.match_id")
            cutoff = _utc(row.get("prediction_cutoff"), f"{path}.prediction_cutoff")
            kickoff = _utc(row.get("kickoff"), f"{path}.kickoff")
            lead = _finite(row.get("lead_hours"), f"{path}.lead_hours")
            if lead <= 0 or kickoff <= cutoff:
                raise ValueError("lead_hours and kickoff/cutoff ordering are invalid")
            derived_lead = (kickoff - cutoff).total_seconds() / 3600.0
            if not math.isclose(lead, derived_lead, abs_tol=1e-6, rel_tol=0.0):
                lead_mismatch += 1
                _issue(issues, "LEAD_HOURS_MISMATCH", "error", f"{path}.lead_hours", "lead_hours differs from kickoff minus prediction_cutoff")
            strategy = _text(row.get("strategy"), f"{path}.strategy")
            budget = _finite(row.get("budget_money"), f"{path}.budget_money")
            if budget < 0:
                raise ValueError("budget_money must be nonnegative")
            dataset_versions.add(_text(row.get("dataset_version"), f"{path}.dataset_version"))
            source_versions.add((
                _text(row.get("source_id"), f"{path}.source_id"),
                _text(row.get("source_version"), f"{path}.source_version"),
            ))
        except (TypeError, ValueError) as exc:
            _issue(issues, "INVALID_IDENTITY_OR_TIME", "error", path, str(exc))

        fixture_available_at = row.get("fixture_available_at")
        if fixture_available_at is not None:
            try:
                fixture_time = _utc(fixture_available_at, f"{path}.fixture_available_at")
                cutoff_value = _utc(row.get("prediction_cutoff"), f"{path}.prediction_cutoff")
                if fixture_time > cutoff_value:
                    _issue(issues, "FIXTURE_AFTER_CUTOFF", "error", f"{path}.fixture_available_at", "fixture was available after prediction cutoff")
            except (TypeError, ValueError) as exc:
                _issue(issues, "INVALID_FIXTURE_TIME", "error", f"{path}.fixture_available_at", str(exc))

        probabilities = row.get("final_probabilities")
        if not _valid_probabilities(probabilities):
            bad_probabilities += 1
            _issue(issues, "INVALID_PROBABILITIES", "error", f"{path}.final_probabilities", "three finite probabilities summing to one are required")

        cost = row.get("cost")
        if not isinstance(cost, Mapping):
            bad_cost_receipts += 1
            _issue(issues, "MISSING_COST_RECEIPT", "error", f"{path}.cost", "nested cost receipt is required")
        else:
            try:
                money = _finite(cost.get("money"), f"{path}.cost.money")
                requests = _finite(cost.get("requests"), f"{path}.cost.requests")
                if money < 0 or requests < 0:
                    raise ValueError("realized cost values must be nonnegative")
                if not requests.is_integer():
                    raise ValueError("cost.requests must count whole requests")
                budget = _finite(row.get("budget_money"), "budget_money")
                if money > budget + 1e-12:
                    raise ValueError("cost.money exceeds the declared budget")
            except (TypeError, ValueError) as exc:
                bad_cost_receipts += 1
                _issue(issues, "INVALID_COST_RECEIPT", "error", f"{path}.cost", str(exc))

        if require_public_context:
            context = row.get("policy_visible_context")
            if not isinstance(context, Mapping):
                _issue(issues, "MISSING_PUBLIC_CONTEXT", "error", f"{path}.policy_visible_context", "public context must be an explicit mapping")
            else:
                paths = _public_forbidden(context)
                public_leak_paths += len(paths)
                for leak_path in paths:
                    _issue(issues, "PUBLIC_CONTEXT_LABEL_FIELD", "error", leak_path, "evaluator-only field appears in public context")
                try:
                    if row.get("policy_context_hash") != manifest_hash(context):
                        _issue(issues, "PUBLIC_CONTEXT_HASH_MISMATCH", "error", path,
                               "public context does not match its recorded digest")
                except (TypeError, ValueError):
                    _issue(issues, "INVALID_PUBLIC_CONTEXT_JSON", "error", path,
                           "public context must contain finite JSON values")
                try:
                    context_cutoff = _utc(row.get("prediction_cutoff"), "prediction_cutoff")
                    issues.extend(_public_times(context, context_cutoff, f"{path}.policy_visible_context"))
                except (TypeError, ValueError):
                    pass  # Invalid cutoff is already reported above.

    for key, count in seen_keys.items():
        if count > 1:
            duplicate_keys["|".join(map(str, key))] = count
            _issue(issues, "DUPLICATE_POLICY_IDENTITY", "error", "episodes", f"identity occurs {count} times")

    severity = Counter(issue.severity for issue in issues)
    return {
        "schema_version": VERSION,
        "status": "PASS" if severity["error"] == 0 else "SCHEMA_REVIEW_REQUIRED",
        "rows": len(rows),
        "matches": len(match_ids),
        "dataset_versions": sorted(dataset_versions),
        "source_versions": [list(item) for item in sorted(source_versions)],
        "missing_field_counts": dict(sorted(missing_counts.items())),
        "duplicate_identity_count": sum(1 for count in seen_keys.values() if count > 1),
        "duplicate_identity_samples": dict(list(duplicate_keys.items())[:10]),
        "lead_hours_mismatch_rows": lead_mismatch,
        "invalid_probability_rows": bad_probabilities,
        "invalid_cost_receipt_rows": bad_cost_receipts,
        "public_context_label_paths": public_leak_paths,
        "issue_counts": dict(sorted(Counter(issue.code for issue in issues).items())),
        "errors": severity["error"],
        "warnings": severity["warning"],
        "issues": [issue.as_dict() for issue in issues[:100]],
    }


def audit_holdout_manifest(
    partitions: Mapping[str, Iterable[Any]],
    *,
    prior_calibration_match_ids: Iterable[Any] | None = None,
    prior_policy_search_match_ids: Iterable[Any] | None = None,
    episode_match_ids: Iterable[Any] = (),
    declared_independent: bool = False,
) -> dict[str, Any]:
    """Check disjoint partitions and prior-manifest exclusion for test IDs."""
    issues: list[ProtocolIssue] = []
    def ids(values: Any, path: str, *, nonempty: bool = False) -> set[str]:
        if values is None or isinstance(values, (str, bytes, Mapping)):
            _issue(issues, "INVALID_ID_INVENTORY", "error", path, "an explicit ID collection is required, including when empty")
            return set()
        try:
            listed = list(values)
        except TypeError:
            _issue(issues, "INVALID_ID_INVENTORY", "error", path, "ID inventory must be iterable")
            return set()
        clean: set[str] = set()
        for value in listed:
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                _issue(issues, "INVALID_MATCH_ID", "error", path, "IDs must be nonblank canonical strings")
            elif value in clean:
                _issue(issues, "DUPLICATE_MANIFEST_ID", "error", path, "manifest contains a duplicate ID")
            else:
                clean.add(value)
        if nonempty and not clean:
            _issue(issues, "EMPTY_PARTITION", "error", path, "partition must contain match IDs")
        return clean
    if not isinstance(partitions, Mapping):
        _issue(issues, "INVALID_PARTITIONS", "error", "partitions", "partitions must be a mapping")
        partitions = {}
    normalized = {name: ids(partitions.get(name), f"partitions.{name}", nonempty=True)
                  for name in ("train", "validation", "test")}
    prior = ids(prior_calibration_match_ids, "prior_calibration_match_ids")
    policy_prior = ids(prior_policy_search_match_ids, "prior_policy_search_match_ids")
    episode_ids = ids(episode_match_ids, "episode_match_ids")
    names = tuple(normalized)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1:]:
            overlap = normalized[left_name] & normalized[right_name]
            if overlap:
                _issue(issues, "PARTITION_OVERLAP", "error", f"partitions.{left_name}/{right_name}", f"{len(overlap)} match IDs occur in both partitions")
    prior_overlap = normalized["test"] & prior
    if prior_overlap:
        _issue(issues, "TEST_OVERLAPS_PRIOR_CALIBRATION", "error", "partitions.test", f"{len(prior_overlap)} test match IDs occur in a prior calibration/training manifest")
    search_overlap = normalized["test"] & policy_prior
    if search_overlap:
        _issue(issues, "TEST_OVERLAPS_PRIOR_POLICY_SEARCH", "error", "partitions.test",
               f"{len(search_overlap)} test match IDs were already used for policy exploration")
    outside = episode_ids - set().union(*normalized.values())
    if outside:
        _issue(issues, "EPISODE_OUTSIDE_MANIFEST", "error", "episodes", f"{len(outside)} episode match IDs are absent from the split manifest")
    if declared_independent is not True:
        _issue(issues, "INDEPENDENCE_NOT_DECLARED", "error", "manifest.declared_independent", "the run does not declare an independent test holdout")
    severity = Counter(issue.severity for issue in issues)
    return {
        "status": "PASS" if severity["error"] == 0 else "HOLDOUT_REVIEW_REQUIRED",
        "declared_independent": declared_independent is True,
        "verification_scope": "supplied ID inventories only; omitted or undisclosed prior exposure is not certified",
        "partition_counts": {name: len(values) for name, values in normalized.items()},
        "partition_overlaps": {
            f"{left}/{right}": len(normalized[left] & normalized[right])
            for index, left in enumerate(names) for right in names[index + 1:]
        },
        "test_overlap_prior_count": len(prior_overlap),
        "test_overlap_prior_policy_search_count": len(search_overlap),
        "episode_outside_manifest_count": len(outside),
        "issues": [issue.as_dict() for issue in issues],
        "errors": severity["error"],
        "warnings": severity["warning"],
    }


def audit_parameter_freeze(
    manifest: Mapping[str, Any],
    *,
    earliest_test_prediction_cutoff: Any | None = None,
) -> dict[str, Any]:
    """Verify that policy thresholds were selected before test evaluation."""
    required = (
        "fit_partition", "selection_partition", "evaluation_partition",
        "parameter_hash", "parameters", "thresholds", "frozen_at", "frozen",
    )
    issues: list[ProtocolIssue] = []
    if not isinstance(manifest, Mapping):
        manifest = {}
    for field in required:
        if field not in manifest or manifest.get(field) in (None, ""):
            _issue(issues, "MISSING_FREEZE_FIELD", "error", f"manifest.{field}", "parameter freeze field is required")
    if manifest.get("fit_partition") != "train":
        _issue(issues, "INVALID_FIT_PARTITION", "error", "manifest.fit_partition", "parameters must be fitted on train")
    if manifest.get("selection_partition") != "validation":
        _issue(issues, "INVALID_SELECTION_PARTITION", "error", "manifest.selection_partition", "thresholds must be selected on validation")
    if manifest.get("evaluation_partition") != "test":
        _issue(issues, "INVALID_EVALUATION_PARTITION", "error", "manifest.evaluation_partition", "test is the evaluation partition")
    if manifest.get("frozen") is not True:
        _issue(issues, "PARAMETERS_NOT_FROZEN", "error", "manifest.frozen", "parameter freeze must be explicit")
    for field in ("parameters", "thresholds"):
        if not isinstance(manifest.get(field), Mapping) or not manifest[field]:
            _issue(issues, "INVALID_FREEZE_PAYLOAD", "error", f"manifest.{field}", "a nonempty mapping is required")
    thresholds = manifest.get("thresholds")
    if isinstance(thresholds, Mapping):
        for key, value in thresholds.items():
            try:
                _text(key, "threshold name")
                _finite(value, f"thresholds.{key}")
            except ValueError as exc:
                _issue(issues, "INVALID_THRESHOLD", "error", "manifest.thresholds", str(exc))
    if not _sha256(manifest.get("parameter_hash")):
        _issue(issues, "INVALID_PARAMETER_HASH", "error", "manifest.parameter_hash", "a SHA-256 digest is required")
    try:
        computed = manifest_hash({"parameters": manifest.get("parameters"), "thresholds": thresholds})
        if computed != manifest.get("parameter_hash"):
            _issue(issues, "PARAMETER_HASH_MISMATCH", "error", "manifest.parameter_hash", "parameters or thresholds differ from their frozen digest")
    except (TypeError, ValueError):
        _issue(issues, "INVALID_FREEZE_JSON", "error", "manifest", "freeze payload must contain finite JSON values")
    try:
        frozen_at = _utc(manifest.get("frozen_at"), "manifest.frozen_at")
    except (TypeError, ValueError) as exc:
        frozen_at = None
        _issue(issues, "INVALID_FREEZE_TIME", "error", "manifest.frozen_at", str(exc))
    try:
        boundary = _utc(earliest_test_prediction_cutoff, "earliest_test_prediction_cutoff")
        if frozen_at is not None and frozen_at >= boundary:
            _issue(issues, "FREEZE_NOT_BEFORE_TEST_START", "error", "manifest.frozen_at", "freeze must precede the first test prediction cutoff")
    except (TypeError, ValueError) as exc:
        _issue(issues, "INVALID_TEST_BOUNDARY", "error", "earliest_test_prediction_cutoff", str(exc))
    severity = Counter(issue.severity for issue in issues)
    return {
        "status": "PASS" if severity["error"] == 0 else "FREEZE_REVIEW_REQUIRED",
        "verification_scope": "manifest consistency only; historical timestamp authenticity requires an external immutable receipt",
        "required_fields": list(required),
        "errors": severity["error"],
        "warnings": severity["warning"],
        "issues": [issue.as_dict() for issue in issues],
    }


def manifest_hash(manifest: Mapping[str, Any]) -> str:
    """Hash a canonical manifest for a run-level freeze reference."""
    payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def audit_policy_experiment(
    episodes: Iterable[Mapping[str, Any]], manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Check one test fold against a hash-bound run contract without labels.

    This checks declared manifests, not timestamp authenticity or policy benefit.
    Each rolling fold needs its own manifest and invocation.
    """
    rows = tuple(episodes)
    schema = audit_episode_schema(rows)
    issues: list[ProtocolIssue] = []
    if not isinstance(manifest, Mapping):
        _issue(issues, "INVALID_RUN_MANIFEST", "error", "manifest", "run manifest must be a mapping")
        manifest = {}
    episode_ids = sorted({row["match_id"] for row in rows if isinstance(row, Mapping)
                          and isinstance(row.get("match_id"), str) and row["match_id"].strip()})
    holdout = audit_holdout_manifest(
        manifest.get("partitions", {}),
        prior_calibration_match_ids=manifest.get("prior_calibration_match_ids"),
        prior_policy_search_match_ids=manifest.get("prior_policy_search_match_ids"),
        episode_match_ids=episode_ids,
        declared_independent=manifest.get("declared_independent", False),
    )
    cutoffs: list[datetime] = []
    for row in rows:
        if isinstance(row, Mapping):
            try:
                cutoffs.append(_utc(row.get("prediction_cutoff"), "prediction_cutoff"))
            except (TypeError, ValueError):
                pass
    policy_freeze = manifest.get("policy_freeze", {})
    freeze = audit_parameter_freeze(policy_freeze, earliest_test_prediction_cutoff=min(cutoffs) if cutoffs else None)
    parameters = policy_freeze.get("parameters", {}) if isinstance(policy_freeze, Mapping) else {}
    if not isinstance(parameters, Mapping):
        parameters = {}
    feature_manifest = manifest.get("feature_manifest")
    feature_hash = None
    try:
        if not isinstance(feature_manifest, Mapping) or not feature_manifest:
            raise ValueError("a nonempty feature manifest is required")
        feature_hash = manifest_hash(feature_manifest)
        _integer(manifest.get("fold"), "fold")
        for name in ("dataset_version", "candidate_set_version", "cost_interpretation"):
            _text(manifest.get(name), name)
    except (TypeError, ValueError) as exc:
        _issue(issues, "INVALID_RUN_METADATA", "error", "manifest", str(exc))
    versions = manifest.get("strategy_versions")
    if not isinstance(versions, Mapping) or not versions or any(
        not isinstance(k, str) or not k.strip() or not isinstance(v, str) or not v.strip()
        for k, v in versions.items()
    ):
        _issue(issues, "INVALID_STRATEGY_VERSIONS", "error", "manifest.strategy_versions", "explicit strategy versions are required")
        versions = {}
    budgets = manifest.get("budgets")
    if not isinstance(budgets, list) or not budgets or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
        for v in budgets
    ):
        _issue(issues, "INVALID_RUN_BUDGETS", "error", "manifest.budgets", "explicit finite nonnegative budgets are required")
        budgets = []
    bindings = {name: manifest.get(name) for name in (
        "dataset_version", "fold", "candidate_set_version", "strategy_versions", "budgets", "cost_interpretation",
    )}
    bindings["feature_manifest_hash"] = feature_hash
    try:
        bindings["holdout_manifest_hash"] = manifest_hash({name: manifest.get(name) for name in (
            "partitions", "prior_calibration_match_ids", "prior_policy_search_match_ids", "declared_independent",
        )})
    except (TypeError, ValueError):
        bindings["holdout_manifest_hash"] = None
        _issue(issues, "INVALID_HOLDOUT_JSON", "error", "manifest.partitions", "holdout inventory must contain finite JSON values")
    for name, value in bindings.items():
        if name not in parameters or parameters[name] != value:
            _issue(issues, "FROZEN_RUN_BINDING_MISMATCH", "error", f"parameters.{name}", "run metadata differs from the frozen parameter payload")
    partitions = manifest.get("partitions")
    test_ids_raw = partitions.get("test") if isinstance(partitions, Mapping) else None
    test_ids = {v for v in test_ids_raw if isinstance(v, str)} if isinstance(test_ids_raw, list) else set()
    if set(episode_ids) != test_ids:
        _issue(issues, "TEST_EPISODE_COVERAGE_MISMATCH", "error", "episodes", "test episodes must cover exactly the test match inventory for this fold")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        expected = {name: manifest.get(name) for name in ("dataset_version", "fold", "candidate_set_version")}
        expected["feature_manifest_hash"] = feature_hash
        expected["policy_parameter_hash"] = policy_freeze.get("parameter_hash") if isinstance(policy_freeze, Mapping) else None
        strategy = row.get("strategy")
        expected["strategy_version"] = versions.get(strategy) if isinstance(strategy, str) else None
        for name, value in expected.items():
            if value is None or row.get(name) != value:
                _issue(issues, "EPISODE_RUN_BINDING_MISMATCH", "error", f"episodes[{index}].{name}", "episode is not bound to this run manifest")
        if row.get("budget_money") not in budgets:
            _issue(issues, "UNDECLARED_EPISODE_BUDGET", "error", f"episodes[{index}].budget_money", "episode budget was not frozen for this run")
    passed = all(part["status"] == "PASS" for part in (schema, holdout, freeze)) and not issues
    return {
        "schema_version": VERSION,
        "status": "PROTOCOL_CONSISTENT" if passed else "PROTOCOL_REVIEW_REQUIRED",
        "protocol_consistent": passed,
        "independent_confirmation": False,
        "verification_scope": "one-fold declared-manifest consistency; no labels, timestamp attestation, or measured policy benefit",
        "schema": schema, "holdout": holdout, "parameter_freeze": freeze,
        "binding_issues": [issue.as_dict() for issue in issues[:100]],
        "errors": sum(part["errors"] for part in (schema, holdout, freeze)) + len(issues),
        "earliest_test_prediction_cutoff": min(cutoffs).isoformat() if cutoffs else None,
        "cost_interpretation": manifest.get("cost_interpretation"),
    }


__all__ = [
    "VERSION", "REQUIRED_EPISODE_FIELDS", "ProtocolIssue", "audit_episode_schema",
    "audit_holdout_manifest", "audit_parameter_freeze", "manifest_hash", "audit_policy_experiment",
]
