"""Historical, as-of evaluation protocol for autonomous information acquisition.

The normal model backtest answers whether a probability vector scored well.  It
does not by itself establish that an acquisition policy only saw information
that had arrived at the decision cutoff.  This module is a small, dependency
free audit layer for that boundary.

It deliberately separates three things:

* ``match_rows`` describe the event and its label availability;
* ``feature_rows`` describe a feature snapshot and its as-of timestamp;
* ``trajectory_rows`` describe the public state seen by a policy.  Evaluator
  metrics may live beside a trajectory, but they are never inspected as policy
  input.

The audit is conservative.  Missing timing metadata produces an issue instead
of silently treating the value as available.  A retrospective reconstruction
can be used for research when its assumptions are declared, but the report
marks it separately from an operational replay.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


VERSION = "historical-autonomous-protocol-v1"

# These names are checked recursively only inside the policy-visible state.  A
# final score can remain in an evaluator-only sidecar used for scoring, while a
# score field in ``visible_state`` or a tool return is a future leak.
FORBIDDEN_PUBLIC_KEYS = frozenset({
    "result", "outcome", "score", "home_score", "away_score", "home_goals",
    "away_goals", "final_score", "label", "target", "realized_loss",
    "sampled_result", "hidden_result", "actual_result", "actual_outcome",
})

PUBLIC_SNAPSHOT_FIELDS = frozenset({
    "match_id", "canonical_match_id", "competition_id", "competition_name",
    "home_team_id", "away_team_id", "home_team_name", "away_team_name",
    "kickoff", "prediction_cutoff", "lead_hours", "fixture_available_at",
    "information_mode", "source", "source_id", "version_id", "data_version",
    "input_manifest_hash", "entity_quality", "time_basis",
})

EVALUATOR_LABEL_FIELDS = frozenset({
    "home_goals", "away_goals", "result_available_at", "label_available_at",
    "score_period", "status", "settlement_status", "result", "outcome",
})


def _utc(value: Any, field: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    else:
        raise ValueError(f"{field} must be a timezone-aware timestamp")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _optional_utc(value: Any, field: str) -> datetime | None:
    if value is None:
        return None
    return _utc(value, field)


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value.strip()


def _json_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, bool, int)) or (
        isinstance(value, float) and math.isfinite(value)
    )


def _find_forbidden(value: Any, path: str = "payload") -> list[tuple[str, str]]:
    """Return ``(path, key)`` for evaluator-only keys in a public payload."""
    issues: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            if key.lower() in FORBIDDEN_PUBLIC_KEYS:
                issues.append((f"{path}.{key}", key))
            issues.extend(_find_forbidden(child, f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            issues.extend(_find_forbidden(child, f"{path}[{index}]"))
    return issues


def public_snapshot_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    """Project a snapshot into fields safe for the Agent's initial context.

    ``build_historical_snapshots`` intentionally retains labels so the
    evaluator can score a replay.  Callers must pass this projection (or an
    equivalent explicit whitelist) into policy-visible state.  The projection
    omits labels, label clocks, settled status and score-period metadata.
    """
    if not isinstance(row, Mapping):
        raise ValueError("snapshot row must be a mapping")
    return {key: row[key] for key in sorted(PUBLIC_SNAPSHOT_FIELDS) if key in row}


def evaluator_label_sidecar(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return result fields for scoring/training outside the policy context."""
    if not isinstance(row, Mapping):
        raise ValueError("snapshot row must be a mapping")
    return {key: row[key] for key in sorted(EVALUATOR_LABEL_FIELDS) if key in row}


def _find_future_public_times(
    value: Any, cutoff: datetime, path: str = "payload",
) -> list[tuple[str, str, str]]:
    """Find freshness timestamps after cutoff in policy-visible state.

    ``visible_lineage`` is checked separately because it has stronger evidence
    requirements; skipping it here avoids duplicate diagnostics.
    """
    found: list[tuple[str, str, str]] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            child_path = f"{path}.{key}"
            if "visible_lineage" in path:
                continue
            if key.lower() in {"available_at", "observed_at", "operational_available_at", "captured_at", "features_available_at"} and child is not None:
                try:
                    parsed = _utc(child, child_path)
                except ValueError:
                    found.append((child_path, key, "invalid"))
                else:
                    if parsed > cutoff:
                        found.append((child_path, key, "future"))
            found.extend(_find_future_public_times(child, cutoff, child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(_find_future_public_times(child, cutoff, f"{path}[{index}]"))
    return found


@dataclass(frozen=True)
class ProtocolIssue:
    code: str
    severity: str
    match_id: str | None
    path: str
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "match_id": self.match_id,
            "path": self.path,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class EpisodeFold:
    """One expanding-window fold grouped by canonical match ID."""

    index: int
    train_start: datetime | None
    validation_start: datetime
    test_start: datetime
    test_end: datetime
    train_match_ids: tuple[str, ...]
    validation_match_ids: tuple[str, ...]
    test_match_ids: tuple[str, ...]
    label_cutoff: datetime | None

    def manifest(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "train_start": None if self.train_start is None else self.train_start.isoformat(),
            "validation_start": self.validation_start.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
            "label_cutoff": None if self.label_cutoff is None else self.label_cutoff.isoformat(),
            "training_label_cutoff": None if self.label_cutoff is None else self.label_cutoff.isoformat(),
            "train_matches": len(self.train_match_ids),
            "validation_matches": len(self.validation_match_ids),
            "test_matches": len(self.test_match_ids),
            "partition_basis": "canonical match kickoff; all trajectories for a match stay together",
            "label_rule": "label_available_at <= label_cutoff for training calibration only",
            "label_cutoff_rule": "min(validation_start, earliest validation/test prediction cutoff)",
        }


def _record_match_id(row: Mapping[str, Any], path: str) -> str:
    value = row.get("match_id") or row.get("canonical_match_id")
    return _text(value, f"{path}.match_id")


def _record_kickoff(row: Mapping[str, Any], path: str) -> datetime:
    return _utc(row.get("kickoff"), f"{path}.kickoff")


def _record_cutoff(row: Mapping[str, Any], path: str) -> datetime:
    return _utc(row.get("prediction_cutoff"), f"{path}.prediction_cutoff")


def _label_time(row: Mapping[str, Any], path: str) -> datetime | None:
    return _optional_utc(
        row.get("label_available_at", row.get("result_available_at")),
        f"{path}.label_available_at",
    )


def build_historical_snapshots(
    matches: Iterable[Mapping[str, Any]], *,
    lead_hours: Iterable[float] = (24.0, 1.0),
) -> tuple[tuple[dict[str, Any], ...], dict[str, int]]:
    """Materialize prematch snapshots from event rows with explicit exclusions.

    The collector's ``matches.jsonl`` is an event table and therefore has no
    per-prediction cutoff.  This helper creates one row per requested lead and
    carries the source's event-time availability fields forward.  It never
    creates a snapshot when the fixture itself was not known at that cutoff.
    """
    leads = tuple(float(value) for value in lead_hours)
    if not leads or len(set(leads)) != len(leads) or any(value <= 0 for value in leads):
        raise ValueError("lead_hours must contain distinct positive values")
    snapshots: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    for index, match in enumerate(matches):
        path = f"matches[{index}]"
        if not isinstance(match, Mapping):
            exclusions["row_not_mapping"] += 1
            continue
        try:
            match_id = _record_match_id(match, path)
            kickoff = _record_kickoff(match, path)
        except ValueError:
            exclusions["missing_match_time"] += 1
            continue
        if match.get("score_period") not in {None, "regulation"}:
            exclusions["unknown_or_non_regulation_period"] += 1
            continue
        if match.get("status") in {"cancelled", "postponed", "suspended"}:
            exclusions["inactive_fixture"] += 1
            continue
        if match.get("home_goals") is None or match.get("away_goals") is None:
            exclusions["no_final_label"] += 1
            continue
        if _label_time(match, path) is None:
            exclusions["unknown_label_availability"] += 1
            continue
        fixture_available = _optional_utc(match.get("fixture_available_at"), f"{path}.fixture_available_at")
        if fixture_available is None:
            exclusions["missing_fixture_availability"] += 1
            continue
        for lead in leads:
            cutoff = kickoff - timedelta(hours=lead)
            if fixture_available > cutoff:
                exclusions["fixture_not_known_at_requested_cutoff"] += 1
                continue
            snapshot = dict(match)
            snapshot.update({
                "match_id": match_id,
                "kickoff": kickoff.isoformat(),
                "prediction_cutoff": cutoff.isoformat(),
                "lead_hours": lead,
            })
            snapshots.append(snapshot)
    return tuple(snapshots), dict(exclusions)


def rolling_episode_folds(
    records: Iterable[Mapping[str, Any]],
    *,
    initial_validation_start: datetime | str,
    validation_window: timedelta,
    test_window: timedelta,
    step: timedelta | None = None,
    train_start: datetime | str | None = None,
    max_folds: int | None = None,
) -> tuple[EpisodeFold, ...]:
    """Create expanding temporal folds without splitting a match's episodes.

    Rows can contain several prediction cutoffs and several policies for the
    same match.  Assignment is by kickoff, never by row order.  This prevents a
    policy trajectory for a match from landing in train while another policy
    trajectory for that match lands in test.
    """
    materialized = tuple(records)
    if validation_window <= timedelta(0) or test_window <= timedelta(0):
        raise ValueError("validation_window and test_window must be positive")
    step = test_window if step is None else step
    if step < test_window:
        raise ValueError("step must be at least test_window")
    if max_folds is not None and (type(max_folds) is not int or max_folds <= 0):
        raise ValueError("max_folds must be a positive integer")
    validation_start = _utc(initial_validation_start, "initial_validation_start")
    train_start_value = None if train_start is None else _utc(train_start, "train_start")
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    kickoff_by_match: dict[str, datetime] = {}
    for index, row in enumerate(materialized):
        if not isinstance(row, Mapping):
            raise ValueError(f"records[{index}] must be a mapping")
        match_id = _record_match_id(row, f"records[{index}]")
        kickoff = _record_kickoff(row, f"records[{index}]")
        cutoff = _record_cutoff(row, f"records[{index}]")
        if cutoff > kickoff:
            raise ValueError(f"records[{index}].prediction_cutoff must not follow kickoff")
        prior = kickoff_by_match.get(match_id)
        if prior is not None and prior != kickoff:
            raise ValueError(f"match {match_id} has conflicting kickoff timestamps")
        kickoff_by_match[match_id] = kickoff
        groups[match_id].append(row)
    if not kickoff_by_match:
        return ()
    latest = max(kickoff_by_match.values())
    folds: list[EpisodeFold] = []
    cursor = validation_start
    index = 0
    while cursor + validation_window <= latest:
        test_start = cursor + validation_window
        test_end = test_start + test_window
        train_ids, validation_ids, test_ids = [], [], []
        for match_id, kickoff in sorted(kickoff_by_match.items(), key=lambda item: (item[1], item[0])):
            if train_start_value is not None and kickoff < train_start_value:
                continue
            if kickoff < cursor:
                train_ids.append(match_id)
            elif kickoff < test_start:
                validation_ids.append(match_id)
            elif kickoff < test_end:
                test_ids.append(match_id)
        # This is the timestamp at which the fold's training/calibration set
        # is frozen.  The fold boundary is the primary clock; the earliest
        # evaluation cutoff is an additional guard for long-lead predictions
        # whose cutoff can precede that boundary.
        evaluation_cutoffs = [
            _record_cutoff(row, f"match:{match_id}")
            for match_id in validation_ids + test_ids
            for row in groups[match_id]
        ]
        earliest_evaluation_cutoff = min(evaluation_cutoffs, default=cursor)
        label_cutoff = min(cursor, earliest_evaluation_cutoff)
        folds.append(EpisodeFold(
            index=index, train_start=train_start_value, validation_start=cursor,
            test_start=test_start, test_end=test_end,
            train_match_ids=tuple(train_ids), validation_match_ids=tuple(validation_ids),
            test_match_ids=tuple(test_ids), label_cutoff=label_cutoff,
        ))
        index += 1
        if max_folds is not None and len(folds) >= max_folds:
            break
        cursor += step
    return tuple(folds)


def _issue(
    issues: list[ProtocolIssue], code: str, severity: str, match_id: str | None,
    path: str, detail: str,
) -> None:
    issues.append(ProtocolIssue(code, severity, match_id, path, detail))


def audit_match_rows(
    rows: Iterable[Mapping[str, Any]], *,
    expected_information_mode: str | None = None,
) -> tuple[ProtocolIssue, ...]:
    """Audit event snapshots and their as-of fields."""
    issues: list[ProtocolIssue] = []
    seen: set[tuple[str, datetime]] = set()
    labels: dict[str, tuple[Any, Any]] = {}
    for index, row in enumerate(rows):
        path = f"matches[{index}]"
        if not isinstance(row, Mapping):
            _issue(issues, "ROW_NOT_MAPPING", "error", None, path, "row must be a mapping")
            continue
        try:
            match_id = _record_match_id(row, path)
            kickoff = _record_kickoff(row, path)
            cutoff = _record_cutoff(row, path)
        except ValueError as exc:
            _issue(issues, "MISSING_TIME_KEY", "error", None, path, str(exc))
            continue
        if cutoff > kickoff:
            _issue(issues, "CUTOFF_AFTER_KICKOFF", "error", match_id, path, "prediction cutoff follows kickoff")
        key = (match_id, cutoff)
        if key in seen:
            _issue(issues, "DUPLICATE_SNAPSHOT", "error", match_id, path, "duplicate match_id/prediction_cutoff")
        seen.add(key)
        label = (row.get("home_goals"), row.get("away_goals"))
        if label != (None, None):
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in label):
                _issue(issues, "INVALID_LABEL", "error", match_id, path, "goals must be nonnegative integers")
            previous = labels.get(match_id)
            if previous is not None and previous != label:
                _issue(issues, "CONFLICTING_LABEL", "error", match_id, path, "match has multiple final labels")
            labels[match_id] = label
        fixture = _optional_utc(row.get("fixture_available_at"), f"{path}.fixture_available_at")
        if fixture is None:
            _issue(issues, "MISSING_FIXTURE_TIME", "error", match_id, path, "fixture availability is required")
        elif fixture > cutoff:
            _issue(issues, "FIXTURE_AFTER_CUTOFF", "error", match_id, path, "fixture was not known at prediction cutoff")
        result_time = _label_time(row, path)
        if label != (None, None) and result_time is None:
            _issue(issues, "MISSING_LABEL_TIME", "error", match_id, path, "final label has no availability timestamp")
        if result_time is not None and result_time < kickoff:
            _issue(issues, "LABEL_BEFORE_KICKOFF", "error", match_id, path, "label availability precedes kickoff")
        mode = row.get("information_mode")
        if expected_information_mode is not None and mode != expected_information_mode:
            _issue(issues, "INFORMATION_MODE_MISMATCH", "warning", match_id, path,
                   f"expected {expected_information_mode!r}, received {mode!r}")
        observed = _optional_utc(row.get("observed_at"), f"{path}.observed_at")
        if observed is not None and observed > cutoff and mode == "operational":
            # The row may still be useful as a final label sidecar, but it must
            # never be treated as a public pre-match fact.
            _issue(issues, "RETROACTIVE_OPERATIONAL_OBSERVATION", "error", match_id, path,
                   "operational observation arrived after prediction cutoff")
    return tuple(issues)


def audit_feature_rows(
    rows: Iterable[Mapping[str, Any]], *,
    match_cutoffs: Mapping[tuple[str, str], datetime] | None = None,
) -> tuple[ProtocolIssue, ...]:
    """Check feature snapshot freshness and identity against match cutoffs."""
    issues: list[ProtocolIssue] = []
    seen: set[tuple[str, datetime]] = set()
    for index, row in enumerate(rows):
        path = f"features[{index}]"
        if not isinstance(row, Mapping):
            _issue(issues, "ROW_NOT_MAPPING", "error", None, path, "row must be a mapping")
            continue
        try:
            match_id = _record_match_id(row, path)
            cutoff = _record_cutoff(row, path)
            kickoff = _record_kickoff(row, path)
        except ValueError as exc:
            _issue(issues, "MISSING_FEATURE_TIME_KEY", "error", None, path, str(exc))
            continue
        if cutoff > kickoff:
            _issue(issues, "CUTOFF_AFTER_KICKOFF", "error", match_id, path, "prediction cutoff follows kickoff")
        key = (match_id, cutoff)
        if key in seen:
            _issue(issues, "DUPLICATE_FEATURE_SNAPSHOT", "error", match_id, path, "duplicate feature snapshot")
        seen.add(key)
        available = _optional_utc(row.get("features_available_at"), f"{path}.features_available_at")
        if available is None:
            _issue(issues, "MISSING_FEATURE_TIME", "error", match_id, path, "features_available_at is required")
        elif available > cutoff:
            _issue(issues, "FEATURE_AFTER_CUTOFF", "error", match_id, path,
                   "feature snapshot was produced after prediction cutoff")
        if match_cutoffs is not None:
            expected = match_cutoffs.get((match_id, cutoff.isoformat()))
            if expected is None:
                _issue(issues, "FEATURE_SNAPSHOT_UNBOUND", "error", match_id, path,
                       "feature snapshot has no audited match_id/prediction_cutoff pair")
            elif available is not None and available > expected:
                _issue(issues, "FEATURE_AFTER_BOUND_CONTEXT", "error", match_id, path,
                       "feature is newer than the bound match snapshot cutoff")
        if not row.get("feature_version"):
            _issue(issues, "MISSING_FEATURE_VERSION", "warning", match_id, path, "feature_version is absent")
        if not row.get("input_manifest_hash") and not row.get("manifest_hash"):
            _issue(issues, "MISSING_FEATURE_PROVENANCE", "warning", match_id, path,
                   "feature input manifest hash is absent")
    return tuple(issues)


def audit_trajectory_rows(
    rows: Iterable[Mapping[str, Any]], *,
    require_historical_mode: bool = True,
) -> tuple[ProtocolIssue, ...]:
    """Audit policy-visible trajectories for future facts and stale evidence."""
    issues: list[ProtocolIssue] = []
    for index, row in enumerate(rows):
        path = f"trajectories[{index}]"
        if not isinstance(row, Mapping):
            _issue(issues, "ROW_NOT_MAPPING", "error", None, path, "row must be a mapping")
            continue
        match_id = row.get("match_id")
        match_id = str(match_id) if match_id is not None else None
        try:
            cutoff = _utc(row.get("events", [{}])[0].get("prediction", {}).get("prediction_cutoff"),
                          f"{path}.events[0].prediction.prediction_cutoff")
        except (ValueError, IndexError, AttributeError, TypeError):
            try:
                cutoff = _utc(row.get("prediction_cutoff"), f"{path}.prediction_cutoff")
            except ValueError as exc:
                _issue(issues, "MISSING_TRAJECTORY_CUTOFF", "error", match_id, path, str(exc))
                continue
        events = row.get("events")
        if not isinstance(events, list) or not events:
            _issue(issues, "MISSING_TRAJECTORY_EVENTS", "error", match_id, path, "trajectory needs a nonempty events list")
            continue
        mode_seen = set()
        for event_index, event in enumerate(events):
            event_path = f"{path}.events[{event_index}]"
            if not isinstance(event, Mapping):
                _issue(issues, "EVENT_NOT_MAPPING", "error", match_id, event_path, "event must be a mapping")
                continue
            prediction = event.get("prediction", {})
            visible_state = event.get("visible_state", {})
            # Only these subtrees are policy-visible.  Evaluator metrics beside
            # the trajectory remain outside this audit boundary.
            public_payload = {"prediction": prediction, "visible_state": visible_state,
                              "decision": event.get("decision", {})}
            for forbidden_path, key in _find_forbidden(public_payload, event_path):
                _issue(issues, "HIDDEN_FIELD_IN_PUBLIC_STATE", "error", match_id, forbidden_path,
                       f"evaluator-only key {key!r} appears in policy-visible state")
            for time_path, time_key, status in _find_future_public_times(public_payload, cutoff, event_path):
                if status == "invalid":
                    _issue(issues, "INVALID_PUBLIC_TIME", "error", match_id, time_path,
                           f"{time_key} must be a timezone-aware timestamp")
                else:
                    _issue(issues, "EVIDENCE_AFTER_CUTOFF", "error", match_id, time_path,
                           f"{time_key} is later than prediction cutoff")
            for public_name, payload in (("prediction", prediction), ("visible_state", visible_state)):
                if not isinstance(payload, Mapping):
                    _issue(issues, "PUBLIC_PAYLOAD_NOT_MAPPING", "error", match_id,
                           f"{event_path}.{public_name}", "policy-visible payload must be a mapping")
                    continue
                payload_cutoff = payload.get("prediction_cutoff")
                if payload_cutoff is not None:
                    try:
                        if _utc(payload_cutoff, f"{event_path}.{public_name}.prediction_cutoff") != cutoff:
                            _issue(issues, "CUTOFF_DRIFT", "error", match_id,
                                   f"{event_path}.{public_name}.prediction_cutoff",
                                   "event cutoff differs from episode cutoff")
                    except ValueError as exc:
                        _issue(issues, "INVALID_PUBLIC_CUTOFF", "error", match_id,
                               f"{event_path}.{public_name}.prediction_cutoff", str(exc))
                mode = payload.get("information_mode")
                if mode is not None:
                    mode_seen.add(mode)
                lineage = payload.get("visible_lineage", {})
                if lineage is not None:
                    if not isinstance(lineage, Mapping):
                        _issue(issues, "INVALID_LINEAGE", "error", match_id,
                               f"{event_path}.{public_name}.visible_lineage", "lineage must be a mapping")
                    else:
                        for group, evidence in lineage.items():
                            if not isinstance(evidence, Mapping):
                                _issue(issues, "INVALID_LINEAGE_ENTRY", "error", match_id,
                                       f"{event_path}.{public_name}.visible_lineage.{group}",
                                       "lineage entry must be a mapping")
                                continue
                            available = evidence.get("available_at")
                            if available is None:
                                _issue(issues, "MISSING_EVIDENCE_TIME", "error", match_id,
                                       f"{event_path}.{public_name}.visible_lineage.{group}",
                                       "observed lineage needs available_at")
                            else:
                                try:
                                    if _utc(available, "available_at") > cutoff:
                                        _issue(issues, "EVIDENCE_AFTER_CUTOFF", "error", match_id,
                                               f"{event_path}.{public_name}.visible_lineage.{group}.available_at",
                                               "evidence arrived after prediction cutoff")
                                except ValueError as exc:
                                    _issue(issues, "INVALID_EVIDENCE_TIME", "error", match_id,
                                           f"{event_path}.{public_name}.visible_lineage.{group}.available_at", str(exc))
                            if evidence.get("status") in {"observed", "available"} and not evidence.get("evidence_refs"):
                                _issue(issues, "MISSING_EVIDENCE_REF", "warning", match_id,
                                       f"{event_path}.{public_name}.visible_lineage.{group}",
                                       "observed evidence has no immutable reference")
            decision = event.get("decision", {})
            if isinstance(decision, Mapping):
                cost = event.get("charged_cost")
                if cost is not None and isinstance(cost, Mapping):
                    money = cost.get("money")
                    if money is not None and (isinstance(money, bool) or not isinstance(money, (int, float)) or money < 0):
                        _issue(issues, "INVALID_ACQUISITION_COST", "error", match_id,
                               f"{event_path}.charged_cost.money", "acquisition cost must be nonnegative")
        if require_historical_mode and mode_seen and mode_seen - {"historical_reconstruction"}:
            _issue(issues, "NON_HISTORICAL_TRAJECTORY_MODE", "warning", match_id, path,
                   f"trajectory modes include {sorted(mode_seen)}")
    return tuple(issues)


def protocol_report(
    *,
    match_rows: Iterable[Mapping[str, Any]] = (),
    feature_rows: Iterable[Mapping[str, Any]] = (),
    trajectory_rows: Iterable[Mapping[str, Any]] = (),
    folds: Sequence[EpisodeFold] = (),
    expected_information_mode: str | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable audit report for a historical run."""
    matches = tuple(match_rows)
    features = tuple(feature_rows)
    trajectories = tuple(trajectory_rows)
    issues = list(audit_match_rows(matches, expected_information_mode=expected_information_mode))
    match_cutoffs: dict[tuple[str, str], datetime] = {}
    for row in matches:
        if not isinstance(row, Mapping):
            continue
        try:
            match_id = _record_match_id(row, "match")
            cutoff = _record_cutoff(row, "match")
        except ValueError:
            continue
        match_cutoffs[(match_id, cutoff.isoformat())] = cutoff
    issues.extend(audit_feature_rows(features, match_cutoffs=match_cutoffs))
    issues.extend(audit_trajectory_rows(trajectories))
    counts = Counter(issue.code for issue in issues)
    severity = Counter(issue.severity for issue in issues)
    match_ids = {_record_match_id(row, "match") for row in matches if isinstance(row, Mapping) and (row.get("match_id") or row.get("canonical_match_id"))}
    trajectory_match_ids = {str(row.get("match_id")) for row in trajectories if isinstance(row, Mapping) and row.get("match_id")}
    return {
        "schema_version": VERSION,
        "status": "PASS" if severity["error"] == 0 else "FAIL",
        "counts": {
            "match_rows": len(matches), "unique_matches": len(match_ids),
            "feature_rows": len(features), "trajectory_rows": len(trajectories),
            "trajectory_matches": len(trajectory_match_ids),
            "issues": len(issues), "errors": severity["error"], "warnings": severity["warning"],
        },
        "cutoff_contract": {
            "public_context_rule": "every visible fact and evidence available_at <= prediction_cutoff",
            "fixture_rule": "fixture_available_at <= prediction_cutoff",
            "label_rule": "labels are evaluator-only and enter calibration/training only after label_available_at",
            "split_rule": "assign by canonical kickoff and keep all trajectories for one match in one partition",
            "unknown_timing_policy": "issue; never impute availability silently",
            "label_boundary": "snapshot labels are evaluator-only sidecars; policy input uses public_snapshot_projection",
            "feature_binding": "feature match_id and prediction_cutoff must match an audited match snapshot",
        },
        "folds": [fold.manifest() for fold in folds],
        "issue_counts": dict(sorted(counts.items())),
        "issues": [issue.as_dict() for issue in issues],
        "evidence_grade": "historical_asof_audit" if severity["error"] == 0 else "blocked_by_protocol_issues",
    }


def summarize_policy_metrics(episodes: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize evaluator-side policy cost/loss without exposing it to policy.

    This is intentionally a secondary report.  It never chooses a model or
    tunes a threshold; it only returns paired descriptive statistics grouped by
    the declared ``strategy`` field.
    """
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for index, row in enumerate(episodes):
        if not isinstance(row, Mapping):
            raise ValueError(f"episodes[{index}] must be a mapping")
        strategy = _text(row.get("strategy"), f"episodes[{index}].strategy")
        groups[strategy].append(row)
    result: dict[str, Any] = {}
    for strategy, rows in sorted(groups.items()):
        def mean_number(name: str) -> float | None:
            values = [row.get(name) for row in rows if isinstance(row.get(name), (int, float)) and not isinstance(row.get(name), bool)]
            return None if not values else float(sum(values) / len(values))

        # Older replay exports put the charged amount in ``cost.money`` while
        # ``budget_money`` is only the policy budget.  Prefer the nested receipt
        # so the report does not mistake a budget ceiling for consumed cost.
        nested_costs = [row.get("cost") for row in rows if isinstance(row.get("cost"), Mapping)]
        realized_money = [item.get("money") for item in nested_costs
                          if isinstance(item.get("money"), (int, float)) and not isinstance(item.get("money"), bool)]
        realized_requests = [item.get("requests") for item in nested_costs
                             if isinstance(item.get("requests"), (int, float)) and not isinstance(item.get("requests"), bool)]
        fallback_requests = [row.get("actual_external_requests") for row in rows
                             if isinstance(row.get("actual_external_requests"), (int, float)) and not isinstance(row.get("actual_external_requests"), bool)]
        result[strategy] = {
            "episodes": len(rows),
            "matches": len({str(row.get("match_id")) for row in rows if row.get("match_id")}),
            "mean_brier_score": mean_number("brier_score"),
            "mean_log_loss": mean_number("log_loss"),
            "mean_declared_budget": mean_number("budget_money"),
            "mean_acquisition_cost": None if not realized_money else float(sum(realized_money) / len(realized_money)),
            "mean_external_requests": (
                None if not realized_requests else float(sum(realized_requests) / len(realized_requests))
            ),
            "request_metric_source": "cost.requests" if realized_requests else "actual_external_requests",
            "fallback_external_requests_mean": None if not fallback_requests else float(sum(fallback_requests) / len(fallback_requests)),
            "descriptive_only": True,
        }
    return {"schema_version": VERSION, "policies": result, "descriptive_only": True}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL while preserving one malformed-line issue for the caller."""
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(dict(value))
    return rows


__all__ = [
    "VERSION", "FORBIDDEN_PUBLIC_KEYS", "ProtocolIssue", "EpisodeFold",
    "PUBLIC_SNAPSHOT_FIELDS", "EVALUATOR_LABEL_FIELDS",
    "public_snapshot_projection", "evaluator_label_sidecar",
    "build_historical_snapshots",
    "rolling_episode_folds", "audit_match_rows", "audit_feature_rows",
    "audit_trajectory_rows", "protocol_report", "summarize_policy_metrics",
    "load_jsonl",
]
