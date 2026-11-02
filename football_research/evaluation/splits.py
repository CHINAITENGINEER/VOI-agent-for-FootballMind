"""Expanding-window splits with match grouping and explicit information cutoffs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping

Row = Mapping[str, Any]


def as_utc(value: Any, field: str = "timestamp") -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class TemporalFold:
    train: tuple[Row, ...]
    validation: tuple[Row, ...]
    test: tuple[Row, ...]
    train_cutoff: datetime
    validation_label_cutoff: datetime
    validation_start: datetime
    test_start: datetime
    test_end: datetime
    purged_train_matches: int
    purged_validation_matches: int

    def manifest(self) -> dict[str, Any]:
        return {
            "train_cutoff": self.train_cutoff.isoformat(),
            "validation_label_cutoff": self.validation_label_cutoff.isoformat(),
            "validation_start": self.validation_start.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
            "train_matches": len({str(row["match_id"]) for row in self.train}),
            "validation_matches": len({str(row["match_id"]) for row in self.validation}),
            "test_matches": len({str(row["match_id"]) for row in self.test}),
            "train_snapshots": len(self.train), "validation_snapshots": len(self.validation),
            "test_snapshots": len(self.test),
            "purged_train_matches": self.purged_train_matches,
            "purged_validation_matches": self.purged_validation_matches,
            "partition_basis": "kickoff, with complete match snapshot groups",
            "availability_rule": "result_available_at <= training cutoff and strictly before every prediction cutoff",
        }


def _group_matches(records: Iterable[Row]) -> list[tuple[datetime, str, list[Row]]]:
    groups: dict[str, list[Row]] = {}
    for row in records:
        if row.get("match_id") is None or str(row["match_id"]) == "":
            raise ValueError("match_id must be present")
        kickoff = as_utc(row["kickoff"], "kickoff")
        cutoff = as_utc(row["prediction_cutoff"], "prediction_cutoff")
        if cutoff > kickoff:
            raise ValueError("prediction_cutoff must not follow kickoff")
        if row.get("result_available_at") is not None:
            if as_utc(row["result_available_at"], "result_available_at") < kickoff:
                raise ValueError("result_available_at must not precede kickoff")
        groups.setdefault(str(row["match_id"]), []).append(row)
    result = []
    for match_id, rows in groups.items():
        kickoffs = {as_utc(row["kickoff"], "kickoff") for row in rows}
        if len(kickoffs) != 1:
            raise ValueError(f"match snapshots have conflicting kickoff; resolve postponements first: {match_id}")
        labels = {(row.get("home_goals"), row.get("away_goals")) for row in rows
                  if row.get("home_goals") is not None and row.get("away_goals") is not None}
        if len(labels) > 1:
            raise ValueError(f"match snapshots have conflicting final labels: {match_id}")
        cutoffs = [as_utc(row["prediction_cutoff"], "prediction_cutoff") for row in rows]
        if len(set(cutoffs)) != len(cutoffs):
            raise ValueError(f"duplicate match_id/prediction_cutoff: {match_id}")
        rows.sort(key=lambda row: as_utc(row["prediction_cutoff"]))
        result.append((next(iter(kickoffs)), match_id, rows))
    return sorted(result, key=lambda item: (item[0], item[1]))


def _labels_available(rows: list[Row], cutoff: datetime, strict_before: datetime) -> bool:
    # A snapshot group is retained or purged as a whole, including missing labels.
    for row in rows:
        goals = row.get("home_goals"), row.get("away_goals")
        if any(isinstance(goal, bool) or not isinstance(goal, int) or goal < 0 for goal in goals):
            return False
        if row.get("result_available_at") is None:
            return False
        available = as_utc(row["result_available_at"], "result_available_at")
        if available > cutoff or available >= strict_before:
            return False
    return True


def make_temporal_fold(
    records: Iterable[Row], *, validation_start: datetime | str,
    test_start: datetime | str, test_end: datetime | str,
    train_start: datetime | str | None = None,
) -> TemporalFold:
    """Split [train_start, validation_start), [validation_start, test_start), test.

    Snapshot groups use canonical kickoff for assignment. To protect long-lead
    predictions, training labels are purged at the earliest evaluation cutoff,
    even when it precedes the nominal validation boundary. Validation labels
    are separately purged before the earliest test prediction for tuning or
    calibration. Every snapshot of a purged match is removed together.

    Training predictions may have old cutoffs: they identify archived snapshot
    groups, not permission to use the eventual outcome at those old cutoffs.
    Callers fitting supervised features must additionally obtain each feature
    vector with its own as-of semantics. No preprocessing is fitted here.
    """
    validation_start = as_utc(validation_start, "validation_start")
    test_start, test_end = as_utc(test_start, "test_start"), as_utc(test_end, "test_end")
    train_start = as_utc(train_start, "train_start") if train_start is not None else None
    if not validation_start < test_start < test_end:
        raise ValueError("require validation_start < test_start < test_end")
    if train_start is not None and train_start >= validation_start:
        raise ValueError("train_start must precede validation_start")
    groups = _group_matches(records)
    train_groups, validation_groups, test_groups = [], [], []
    for kickoff, _, rows in groups:
        if train_start is not None and kickoff < train_start:
            continue
        if kickoff < validation_start:
            train_groups.append(rows)
        elif kickoff < test_start:
            validation_groups.append(rows)
        elif kickoff < test_end:
            test_groups.append(rows)
    evaluation_cutoffs = [as_utc(row["prediction_cutoff"]) for group in validation_groups + test_groups for row in group]
    earliest_prediction = min(evaluation_cutoffs, default=validation_start)
    train_cutoff = min(validation_start, earliest_prediction)
    test_cutoffs = [as_utc(row["prediction_cutoff"]) for group in test_groups for row in group]
    earliest_test_prediction = min(test_cutoffs, default=test_start)
    validation_label_cutoff = min(test_start, earliest_test_prediction)
    eligible_train = [rows for rows in train_groups if _labels_available(rows, train_cutoff, earliest_prediction)]
    eligible_validation = [rows for rows in validation_groups if _labels_available(rows, validation_label_cutoff, earliest_test_prediction)]
    return TemporalFold(
        train=tuple(row for group in eligible_train for row in group),
        validation=tuple(row for group in eligible_validation for row in group),
        test=tuple(row for group in test_groups for row in group),
        train_cutoff=train_cutoff, validation_label_cutoff=validation_label_cutoff,
        validation_start=validation_start, test_start=test_start, test_end=test_end,
        purged_train_matches=len(train_groups) - len(eligible_train),
        purged_validation_matches=len(validation_groups) - len(eligible_validation),
    )


def rolling_origin_splits(
    records: Iterable[Row], *, initial_validation_start: datetime | str,
    validation_window: timedelta, test_window: timedelta, step: timedelta | None = None,
    train_start: datetime | str | None = None, max_folds: int | None = None,
) -> list[TemporalFold]:
    """Expanding training history and non-overlapping test windows by default."""
    if validation_window <= timedelta(0) or test_window <= timedelta(0):
        raise ValueError("validation and test windows must be positive")
    step = test_window if step is None else step
    if step < test_window:
        raise ValueError("step must be at least test_window to avoid repeated test matches")
    if max_folds is not None and (not isinstance(max_folds, int) or max_folds <= 0):
        raise ValueError("max_folds must be a positive integer")
    records = list(records)
    groups = _group_matches(records)
    if not groups:
        return []
    latest_kickoff = max(group[0] for group in groups)
    validation_start = as_utc(initial_validation_start, "initial_validation_start")
    folds = []
    while validation_start + validation_window <= latest_kickoff:
        test_start = validation_start + validation_window
        folds.append(make_temporal_fold(
            records, validation_start=validation_start, test_start=test_start,
            test_end=test_start + test_window, train_start=train_start,
        ))
        if max_folds is not None and len(folds) >= max_folds:
            break
        validation_start += step
    return folds
