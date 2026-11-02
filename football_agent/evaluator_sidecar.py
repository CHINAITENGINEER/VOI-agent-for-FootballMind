"""Offline evaluator sidecar for Round133-A.

This module is deliberately outside the Agent runtime.  It consumes the
public ``value_evaluation_contract`` episodes produced by Round132 and writes
an independent, deterministic evaluator manifest.  The evaluator may be a
synthetic fixture in this round; the same envelope can later be populated by
an independently maintained labeler.

The sidecar never reads a run directory, calls a provider, calls a model, or
mutates an episode's public pre-action snapshot.  It stores digests and
candidate-level losses instead of returning raw source payloads.  Runtime code
should treat the files emitted here as an offline analysis product.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .value_evaluation_contract import (
    ACTOR_PROVISIONAL,
    BLOCKED,
    INDEPENDENT_GOLD,
    VERSION as EVALUATION_CONTRACT_VERSION,
    validate_evaluation_episode,
)


VERSION = "evaluator-sidecar-v1"
FIXTURE_VERSION = "synthetic-evaluator-fixture-v1"
DEFAULT_EVALUATOR_ID = "synthetic-independent-evaluator-v1"
DEFAULT_LOSS_UNIT = "decision_loss"
_ALLOWED_LOSS_UNITS = {"log_loss", "brier", "decision_loss"}
EVIDENCE_ORIGIN = "synthetic_fixture"
SYNTHETIC_CLOCK = "deterministic_synthetic_clock"


class SidecarValidationError(ValueError):
    """Raised when a sidecar would weaken the evaluation boundary."""


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _parse_time(value: Any) -> datetime:
    if not isinstance(value, str):
        raise SidecarValidationError("SIDECAR_TIME_REQUIRED")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise SidecarValidationError("SIDECAR_TIME_INVALID") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SidecarValidationError("SIDECAR_TIME_MUST_BE_TIMEZONE_AWARE")
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _finite_nonnegative(value: Any) -> bool:
    return (type(value) in (int, float) and value >= 0 and value == value
            and value not in (float("inf"), float("-inf")))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    """Return the stable SHA256 used for sidecar bindings."""
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _episode_identity(episode: Mapping[str, Any]) -> tuple[str, str]:
    if not isinstance(episode, Mapping):
        raise SidecarValidationError("SIDECAR_EPISODE_OBJECT_REQUIRED")
    episode_id = episode.get("episode_id")
    target_key = episode.get("target_key")
    if not _nonempty(episode_id) or not _nonempty(target_key):
        raise SidecarValidationError("SIDECAR_EPISODE_IDENTITY_REQUIRED")
    return str(episode_id), str(target_key)


def _pre_action(episode: Mapping[str, Any]) -> Mapping[str, Any]:
    pre = episode.get("pre_action")
    if not isinstance(pre, Mapping):
        raise SidecarValidationError("SIDECAR_PRE_ACTION_REQUIRED")
    return pre


def _ensure_episode_is_safe(episode: Mapping[str, Any]) -> dict[str, Any]:
    """Validate only the public episode before constructing any gold rows."""
    # Protect existing labels before the general contract validator sees a
    # possibly incomplete actor envelope.  The sidecar must never replace an
    # actor judgement or an already attached independent label.
    existing_outcome = episode.get("evaluation_outcome")
    if isinstance(existing_outcome, Mapping):
        status = existing_outcome.get("label_status")
        if status == ACTOR_PROVISIONAL:
            raise SidecarValidationError("ACTOR_PROVISIONAL_OUTCOME_PROTECTED")
        if status == INDEPENDENT_GOLD:
            raise SidecarValidationError("SIDECAR_EXISTING_OUTCOME_PROTECTED")
    report = validate_evaluation_episode(episode)
    if report["status"] == BLOCKED:
        raise SidecarValidationError("SIDECAR_EPISODE_BLOCKED")
    pre = _pre_action(episode)
    ids = pre.get("candidate_ids")
    if not isinstance(ids, list) or not ids or any(not _nonempty(item) for item in ids):
        raise SidecarValidationError("SIDECAR_CANDIDATE_MENU_REQUIRED")
    # An actor row is evidence from the acting policy, not an evaluator label.
    # Rejecting the join is intentional: an independent sidecar must never
    # silently replace or upgrade that row.
    rows = episode.get("candidate_evaluations")
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            if row.get("label_status") == ACTOR_PROVISIONAL:
                raise SidecarValidationError("ACTOR_PROVISIONAL_CANNOT_BE_UPGRADED")
            if row.get("label_status") == INDEPENDENT_GOLD:
                raise SidecarValidationError("SIDECAR_EXISTING_LABELS_PROTECTED")
    return report


def _synthetic_label(target_key: str) -> str:
    labels = ("home_win", "draw", "away_win")
    value = int(sha256(target_key.encode("utf-8")).hexdigest()[:8], 16)
    return labels[value % len(labels)]


def _synthetic_after_loss(target_key: str, candidate_id: str) -> float:
    """Deterministic fixture loss, kept independent of actor estimates."""
    value = int(sha256(f"{target_key}\0{candidate_id}".encode("utf-8")).hexdigest()[:8], 16)
    return round(0.25 + (value % 500) / 1000.0, 6)


def _loss_pair(value: Any, *, target_key: str, candidate_id: str) -> tuple[float, float]:
    if value is None:
        before = 1.0
        after = _synthetic_after_loss(target_key, candidate_id)
    elif isinstance(value, Mapping):
        before = value.get("decision_loss_before", value.get("before", 1.0))
        after = value.get("decision_loss_after", value.get("after"))
        if after is None:
            raise SidecarValidationError("SIDECAR_CANDIDATE_LOSS_AFTER_REQUIRED")
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        before, after = value
    else:
        before, after = 1.0, value
    if not _finite_nonnegative(before) or not _finite_nonnegative(after):
        raise SidecarValidationError("SIDECAR_CANDIDATE_LOSS_INVALID")
    return float(before), float(after)


def _candidate_cost(episode: Mapping[str, Any], candidate_id: str) -> Any:
    """Copy declared cost metadata only; no actor loss/value is copied."""
    pre = _pre_action(episode)
    assessments = pre.get("candidate_value_assessments")
    if not isinstance(assessments, list):
        # Round132 normally keeps assessments in ledger_binding only.  Missing
        # cost is valid for an independent evaluator and remains explicit.
        return None
    for row in assessments:
        if isinstance(row, Mapping) and row.get("candidate_id") == candidate_id:
            cost = row.get("declared_cost")
            if cost is None:
                cost = row.get("cost")
            return deepcopy(cost)
    return None


def build_evaluator_record(
    episode: Mapping[str, Any], *,
    evaluator_id: str = DEFAULT_EVALUATOR_ID,
    candidate_losses: Mapping[str, Any] | None = None,
    outcome_label: str | None = None,
    loss_unit: str = DEFAULT_LOSS_UNIT,
    fixture_version: str = FIXTURE_VERSION,
) -> dict[str, Any]:
    """Create one independent evaluator result for a Round132 episode.

    ``candidate_losses`` accepts ``candidate_id -> after_loss``, a two-item
    ``(before, after)`` sequence, or a mapping with ``before``/``after`` keys.
    Omitting it selects a deterministic synthetic fixture.  The output has no
    raw return body: each return is represented by a digest and observation
    time, which makes the sidecar auditable without leaking source payloads.
    """
    if not _nonempty(evaluator_id):
        raise SidecarValidationError("SIDECAR_EVALUATOR_ID_REQUIRED")
    if loss_unit not in _ALLOWED_LOSS_UNITS:
        raise SidecarValidationError("SIDECAR_LOSS_UNIT_INVALID")
    _ensure_episode_is_safe(episode)
    episode_id, target_key = _episode_identity(episode)
    pre = _pre_action(episode)
    candidate_ids = list(pre["candidate_ids"])
    losses_supplied = candidate_losses is not None
    if losses_supplied and not isinstance(candidate_losses, Mapping):
        raise SidecarValidationError("SIDECAR_CANDIDATE_LOSSES_MAPPING_REQUIRED")
    supplied = dict(candidate_losses or {})
    unknown = sorted(set(supplied) - set(candidate_ids))
    if unknown:
        raise SidecarValidationError("SIDECAR_CANDIDATE_OUTSIDE_MENU")
    selected_at = _parse_time(pre.get("selected_at"))
    label = outcome_label or _synthetic_label(target_key)
    if not _nonempty(label):
        raise SidecarValidationError("SIDECAR_OUTCOME_LABEL_REQUIRED")
    rows: list[dict[str, Any]] = []
    for index, candidate_id in enumerate(candidate_ids):
        # An explicitly supplied mapping is an observation manifest.  Missing
        # candidates stay unobserved; synthetic hash losses are used only when
        # the whole mapping is omitted (the fixture mode).
        if losses_supplied and candidate_id not in supplied:
            continue
        before, after = _loss_pair(supplied.get(candidate_id),
                                   target_key=target_key, candidate_id=candidate_id)
        observed_at = selected_at + timedelta(seconds=index + 1)
        return_descriptor = {
            "fixture_version": fixture_version,
            "episode_id": episode_id,
            "target_key": target_key,
            "candidate_id": candidate_id,
            "outcome_label": label,
            "decision_loss_before": before,
            "decision_loss_after": after,
            "return_observed_at": _iso(observed_at),
        }
        row = {
            "candidate_id": candidate_id,
            "label_status": INDEPENDENT_GOLD,
            "evaluator_id": evaluator_id,
            "return_digest": digest(return_descriptor),
            "return_observed_at": _iso(observed_at),
            # ``available_at`` is the public sidecar vocabulary for a
            # candidate return.  Keep the more precise observed-at field too
            # so downstream audits can distinguish event time from ingestion.
            "available_at": _iso(observed_at),
            "loss_unit": loss_unit,
            "decision_loss_before": before,
            "decision_loss_after": after,
            "candidate_outcome_label": label,
        }
        cost = _candidate_cost(episode, candidate_id)
        if cost is not None:
            row["declared_cost"] = cost
        rows.append(row)
    available_at = selected_at + timedelta(seconds=len(candidate_ids) + 1)
    outcome_descriptor = {
        "fixture_version": fixture_version,
        "episode_id": episode_id,
        "target_key": target_key,
        "outcome_label": label,
        "available_at": _iso(available_at),
    }
    outcome = {
        "label_status": INDEPENDENT_GOLD,
        "evaluator_id": evaluator_id,
        "available_at": _iso(available_at),
        "outcome_digest": digest(outcome_descriptor),
        "outcome_label": label,
    }
    merged = deepcopy(dict(episode))
    merged["candidate_evaluations"] = rows
    merged["evaluation_outcome"] = outcome
    validation = validate_evaluation_episode(merged)
    if validation["status"] == BLOCKED:
        raise SidecarValidationError("SIDECAR_GENERATED_RECORD_BLOCKED")
    record = {
        "schema_version": VERSION,
        "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
        "status": validation["status"],
        "episode_id": episode_id,
        "target_key": target_key,
        "episode_digest": digest(episode),
        "pre_action_snapshot_digest": pre.get("snapshot_digest"),
        "evaluator": {
            "evaluator_id": evaluator_id,
            "independent_from_actor": True,
            "fixture_version": fixture_version,
        },
        "candidate_evaluations": rows,
        "evaluation_outcome": outcome,
        "validation": validation,
        "numeric_voi_training_eligible": False,
        "actor_assessment_is_gold": False,
        "raw_return_payload_included": False,
        "evidence_origin": EVIDENCE_ORIGIN,
        "world_truth_verified": False,
        "for_mechanism_tests_only": True,
    }
    record["record_digest"] = digest(record)
    return record


def _blocked_record(episode: Any, error: str) -> dict[str, Any]:
    episode_id = episode.get("episode_id") if isinstance(episode, Mapping) else None
    target_key = episode.get("target_key") if isinstance(episode, Mapping) else None
    return {
        "schema_version": VERSION,
        "episode_id": episode_id,
        "target_key": target_key,
        "status": BLOCKED,
        "error": error,
        "numeric_voi_training_eligible": False,
        "actor_assessment_is_gold": False,
        "raw_return_payload_included": False,
        "evidence_origin": EVIDENCE_ORIGIN,
        "world_truth_verified": False,
        "for_mechanism_tests_only": True,
    }


def _expected_row_digest(record: Mapping[str, Any], row: Mapping[str, Any]) -> str:
    outcome = record.get("evaluation_outcome")
    evaluator = record.get("evaluator")
    return digest({
        "fixture_version": evaluator.get("fixture_version") if isinstance(evaluator, Mapping) else None,
        "episode_id": record.get("episode_id"),
        "target_key": record.get("target_key"),
        "candidate_id": row.get("candidate_id"),
        "outcome_label": row.get("candidate_outcome_label"),
        "decision_loss_before": row.get("decision_loss_before"),
        "decision_loss_after": row.get("decision_loss_after"),
        "return_observed_at": row.get("return_observed_at"),
    })


def _expected_outcome_digest(record: Mapping[str, Any]) -> str:
    outcome = record.get("evaluation_outcome")
    evaluator = record.get("evaluator")
    return digest({
        "fixture_version": evaluator.get("fixture_version") if isinstance(evaluator, Mapping) else None,
        "episode_id": record.get("episode_id"),
        "target_key": record.get("target_key"),
        "outcome_label": outcome.get("outcome_label") if isinstance(outcome, Mapping) else None,
        "available_at": outcome.get("available_at") if isinstance(outcome, Mapping) else None,
    })


def verify_evaluator_record(record: Mapping[str, Any], episode: Mapping[str, Any]) -> dict[str, Any]:
    """Re-bind a sidecar record to its unchanged source episode.

    The content digest is checked before the contract is re-evaluated.  The
    identity, evaluator, outcome and per-return digests are then checked again
    so a caller cannot alter a loss or label and simply recompute one outer
    digest.
    """
    if not isinstance(record, Mapping) or not isinstance(episode, Mapping):
        raise SidecarValidationError("SIDECAR_VERIFY_OBJECT_REQUIRED")
    if record.get("schema_version") != VERSION:
        raise SidecarValidationError("SIDECAR_RECORD_SCHEMA_MISMATCH")
    if record.get("episode_digest") != digest(episode):
        raise SidecarValidationError("SIDECAR_EPISODE_DIGEST_MISMATCH")
    episode_id, target_key = _episode_identity(episode)
    if record.get("episode_id") != episode_id or record.get("target_key") != target_key:
        raise SidecarValidationError("SIDECAR_RECORD_IDENTITY_MISMATCH")
    if record.get("evidence_origin") != EVIDENCE_ORIGIN:
        raise SidecarValidationError("SIDECAR_EVIDENCE_ORIGIN_MISMATCH")
    if record.get("world_truth_verified") is not False or record.get("for_mechanism_tests_only") is not True:
        raise SidecarValidationError("SIDECAR_SYNTHETIC_BOUNDARY_MISMATCH")
    record_without_digest = {key: value for key, value in record.items()
                             if key != "record_digest"}
    if record.get("record_digest") != digest(record_without_digest):
        raise SidecarValidationError("SIDECAR_RECORD_DIGEST_MISMATCH")
    evaluator = record.get("evaluator")
    outcome = record.get("evaluation_outcome")
    if not isinstance(evaluator, Mapping) or not _nonempty(evaluator.get("evaluator_id")):
        raise SidecarValidationError("SIDECAR_RECORD_EVALUATOR_MISSING")
    if not isinstance(outcome, Mapping):
        raise SidecarValidationError("SIDECAR_RECORD_OUTCOME_MISSING")
    evaluator_id = evaluator.get("evaluator_id")
    if outcome.get("evaluator_id") != evaluator_id or outcome.get("label_status") != INDEPENDENT_GOLD:
        raise SidecarValidationError("SIDECAR_RECORD_OUTCOME_IDENTITY_MISMATCH")
    if outcome.get("outcome_digest") != _expected_outcome_digest(record):
        raise SidecarValidationError("SIDECAR_OUTCOME_DIGEST_MISMATCH")
    rows = record.get("candidate_evaluations")
    if not isinstance(rows, list):
        raise SidecarValidationError("SIDECAR_RECORD_ROWS_REQUIRED")
    for row in rows:
        if not isinstance(row, Mapping):
            raise SidecarValidationError("SIDECAR_RECORD_ROW_INVALID")
        if row.get("label_status") != INDEPENDENT_GOLD or row.get("evaluator_id") != evaluator_id:
            raise SidecarValidationError("SIDECAR_RECORD_ROW_IDENTITY_MISMATCH")
        if row.get("return_digest") != _expected_row_digest(record, row):
            raise SidecarValidationError("SIDECAR_RETURN_DIGEST_MISMATCH")
        if row.get("candidate_outcome_label") != outcome.get("outcome_label"):
            raise SidecarValidationError("SIDECAR_CANDIDATE_OUTCOME_MISMATCH")
    outcome = record.get("evaluation_outcome")
    merged = deepcopy(dict(episode))
    merged["candidate_evaluations"] = deepcopy(rows)
    merged["evaluation_outcome"] = deepcopy(outcome)
    report = validate_evaluation_episode(merged)
    if report["status"] == BLOCKED:
        raise SidecarValidationError("SIDECAR_RECORD_BLOCKED")
    if "status" in record and record.get("status") != report["status"]:
        raise SidecarValidationError("SIDECAR_RECORD_STATUS_MISMATCH")
    return report


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(_canonical_json(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def _derive_generated_at(source: list[Mapping[str, Any]]) -> datetime:
    """Choose a deterministic synthetic clock after every valid episode."""
    latest: datetime | None = None
    latest_span = 1
    for episode in source:
        try:
            selected = _parse_time(_pre_action(episode).get("selected_at"))
        except SidecarValidationError:
            continue
        candidate_ids = _pre_action(episode).get("candidate_ids")
        span = len(candidate_ids) + 1 if isinstance(candidate_ids, list) else 1
        if latest is None or selected > latest:
            latest = selected
            latest_span = span
        elif selected == latest:
            latest_span = max(latest_span, span)
    if latest is None:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)
    return latest + timedelta(seconds=latest_span)


def run_evaluator_sidecar(
    episodes: Iterable[Mapping[str, Any]], output_dir: str | Path, *,
    evaluator_id: str = DEFAULT_EVALUATOR_ID,
    candidate_losses: Mapping[str, Mapping[str, Any]] | None = None,
    outcome_labels: Mapping[str, str] | None = None,
    generated_at: str | None = None,
    fixture_version: str = FIXTURE_VERSION,
) -> dict[str, Any]:
    """Evaluate a batch and write an independent manifest/results pair.

    ``candidate_losses`` and ``outcome_labels`` are keyed by episode ID.  A
    blocked episode is recorded as blocked and does not yield a gold row.  No
    network, model, run archive, or runtime Experience object is accessed.
    """
    source = [deepcopy(dict(item)) for item in episodes]
    ids = [item.get("episode_id") if isinstance(item, Mapping) else None for item in source]
    if any(not _nonempty(item) for item in ids) or len(ids) != len(set(ids)):
        raise SidecarValidationError("SIDECAR_EPISODE_IDS_MUST_BE_UNIQUE")
    if candidate_losses is not None and not isinstance(candidate_losses, Mapping):
        raise SidecarValidationError("SIDECAR_EPISODE_LOSSES_MAPPING_REQUIRED")
    if outcome_labels is not None and not isinstance(outcome_labels, Mapping):
        raise SidecarValidationError("SIDECAR_EPISODE_OUTCOMES_MAPPING_REQUIRED")
    generated = _derive_generated_at(source) if generated_at is None else _parse_time(generated_at)
    for episode in source:
        try:
            selected = _parse_time(_pre_action(episode).get("selected_at"))
        except SidecarValidationError:
            continue
        if generated < selected:
            raise SidecarValidationError("SIDECAR_GENERATED_AT_BEFORE_SELECTION")
    source_digest = digest(source)
    results: list[dict[str, Any]] = []
    for episode in source:
        episode_id = str(episode.get("episode_id"))
        try:
            result = build_evaluator_record(
                episode,
                evaluator_id=evaluator_id,
                candidate_losses=(None if candidate_losses is None
                                  else candidate_losses.get(episode_id, {})),
                outcome_label=(outcome_labels or {}).get(episode_id),
                fixture_version=fixture_version,
            )
        except SidecarValidationError as exc:
            result = _blocked_record(episode, str(exc))
        results.append(result)
    results_digest = digest(results)
    manifest = {
        "schema_version": VERSION,
        "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
        "evaluator_id": evaluator_id,
        "evaluator_kind": "independent_synthetic_fixture",
        "fixture_version": fixture_version,
        "generated_at": _iso(generated),
        "synthetic_clock": SYNTHETIC_CLOCK,
        "evidence_origin": EVIDENCE_ORIGIN,
        "world_truth_verified": False,
        "for_mechanism_tests_only": True,
        "episode_count": len(source),
        "source_episode_digest": source_digest,
        "results_digest": results_digest,
        "numeric_voi_training_eligible": False,
        "runtime_reads_sidecar": False,
        "real_source_requests": 0,
        "real_model_calls": 0,
    }
    destination = Path(output_dir)
    manifest_path = destination / "manifest.json"
    results_path = destination / "results.json"
    _write_json(manifest_path, manifest)
    _write_json(results_path, {"schema_version": VERSION, "results": results,
                               "results_digest": results_digest})
    status_counts: dict[str, int] = {}
    for item in results:
        status = str(item.get("status", BLOCKED))
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "manifest": manifest,
        "results": results,
        "status_counts": status_counts,
        "manifest_path": str(manifest_path),
        "results_path": str(results_path),
    }


__all__ = [
    "VERSION", "FIXTURE_VERSION", "DEFAULT_EVALUATOR_ID", "DEFAULT_LOSS_UNIT",
    "EVIDENCE_ORIGIN", "SYNTHETIC_CLOCK", "SidecarValidationError", "digest",
    "build_evaluator_record", "verify_evaluator_record", "run_evaluator_sidecar",
]
