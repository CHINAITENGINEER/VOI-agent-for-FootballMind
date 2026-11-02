from __future__ import annotations

import pytest

from football_agent.evidence_ledger import EvidenceLedgerEntry
from football_agent.source_health_calibration import calibrate_source_health


TRAIN_END = "2026-09-10T23:59:59Z"
EVAL_START = "2026-09-12T00:00:00Z"
EVAL_END = "2026-09-20T23:59:59Z"
PREDICTION_CUTOFF = "2026-10-01T00:00:00Z"


def _entry(
    dependency: str,
    captured_at: str,
    availability: str,
    *,
    source_id: str = "api-football",
    source_version: str = "v1",
    sequence: int = 0,
) -> EvidenceLedgerEntry:
    if availability in {"available", "late"}:
        observed_at = captured_at
        operational_available_at = captured_at
    else:
        observed_at = None
        operational_available_at = None
    quality = {
        "available": "VALIDATED",
        "late": "PARTIAL",
        "failed": "PROVIDER_ERROR",
        "duplicate": "DUPLICATE",
    }[availability]
    row = {
        "candidate_id": f"candidate:{dependency}",
        "source_id": source_id,
        "dependency_key": dependency,
        "quality": quality,
        "availability": availability,
        "observed_at": observed_at,
        "operational_available_at": operational_available_at,
        "evidence_refs": [f"raw:{dependency}:{sequence}"],
        "content_hash": f"hash:{dependency}:{sequence}",
    }
    return EvidenceLedgerEntry.from_row(
        row,
        episode_id="episode:calibration",
        target_key="match:synthetic",
        action_id="query:source-health",
        request_id=f"request:{source_id}:{source_version}:{dependency}:{sequence}",
        sequence=sequence,
        prediction_cutoff=PREDICTION_CUTOFF,
        captured_at=captured_at,
        requested_at=captured_at,
        source_version=source_version,
    )


def _report(reports, source_id="api-football", source_version="v1"):
    return next(row for row in reports if row.source_id == source_id and row.source_version == source_version)


def test_rates_use_latest_independent_dependency_units_and_keep_attempts_visible():
    entries = [
        _entry("dep-1", "2026-09-02T00:00:00Z", "duplicate", sequence=0),
        _entry("dep-1", "2026-09-03T00:00:00Z", "available", sequence=1),
        *[_entry(f"dep-{index}", f"2026-09-0{index}T00:00:00Z", "available", sequence=index)
          for index in range(2, 6)],
        _entry("dep-6", "2026-09-07T00:00:00Z", "failed", sequence=6),
        _entry("eval-1", "2026-09-13T00:00:00Z", "available", sequence=10),
        _entry("eval-2", "2026-09-14T00:00:00Z", "available", sequence=11),
        *[_entry(f"eval-{index}", f"2026-09-{13 + index:02d}T00:00:00Z", "failed", sequence=11 + index)
          for index in range(3, 7)],
    ]

    row = _report(calibrate_source_health(
        entries, train_end=TRAIN_END, eval_start=EVAL_START, eval_end=EVAL_END,
    ))

    assert row.train_attempts == 7
    assert row.train_independent_dependency_count == 6
    assert row.train_duplicate_attempts == 1
    assert row.train_attempt_status_counts == {"available": 5, "late": 0, "failed": 1, "duplicate": 1}
    assert row.train_status_counts == {"available": 5, "late": 0, "failed": 1, "duplicate": 0}
    assert row.train_available_rate == pytest.approx(5 / 6)
    assert row.eval_available_rate == pytest.approx(2 / 6)
    assert row.availability_rate_delta == pytest.approx(-.5)
    assert row.drift_status == "DEGRADED"
    assert row.operational_support_status == "SUPPORTED_OPERATIONAL_DIAGNOSTIC"
    assert row.train_available_lower_bound < row.train_available_rate


def test_later_retry_status_is_the_independent_status_for_a_dependency():
    entries = [
        _entry("same", "2026-09-02T00:00:00Z", "available", sequence=0),
        _entry("same", "2026-09-03T00:00:00Z", "failed", sequence=1),
        *[_entry(f"train-{index}", f"2026-09-{index + 2:02d}T00:00:00Z", "available", sequence=index + 2)
          for index in range(5)],
        *[_entry(f"eval-{index}", f"2026-09-{13 + index:02d}T00:00:00Z", "available", sequence=20 + index)
          for index in range(6)],
    ]
    row = _report(calibrate_source_health(
        entries, train_end=TRAIN_END, eval_start=EVAL_START, eval_end=EVAL_END,
    ))

    assert row.train_attempts == 7
    assert row.train_independent_dependency_count == 6
    assert row.train_status_counts["available"] == 5
    assert row.train_status_counts["failed"] == 1
    assert row.train_available_rate == pytest.approx(5 / 6)


def test_source_version_is_a_separate_calibration_channel():
    entries = [
        *[_entry(f"v1-train-{i}", f"2026-09-{2 + i:02d}T00:00:00Z", "available", source_version="v1", sequence=i)
          for i in range(5)],
        *[_entry(f"v1-eval-{i}", f"2026-09-{13 + i:02d}T00:00:00Z", "available", source_version="v1", sequence=20 + i)
          for i in range(5)],
        _entry("v2-train", "2026-09-05T00:00:00Z", "available", source_version="v2"),
        _entry("v2-eval", "2026-09-13T00:00:00Z", "failed", source_version="v2"),
    ]
    reports = calibrate_source_health(
        entries, train_end=TRAIN_END, eval_start=EVAL_START, eval_end=EVAL_END,
    )
    v1, v2 = _report(reports, source_version="v1"), _report(reports, source_version="v2")
    assert v1.drift_status == "STABLE"
    assert v1.operational_support_status == "SUPPORTED_OPERATIONAL_DIAGNOSTIC"
    assert v2.drift_status == "INSUFFICIENT_SUPPORT"
    assert v2.operational_support_status == "INSUFFICIENT_INDEPENDENT_SUPPORT"


def test_split_boundaries_are_inclusive_and_gap_or_future_records_are_ignored():
    entries = [
        _entry("at-train-end", TRAIN_END, "available"),
        _entry("in-gap", "2026-09-11T12:00:00Z", "failed", source_id="gap-only"),
        _entry("at-eval-start", EVAL_START, "available"),
        _entry("at-eval-end", EVAL_END, "available"),
        _entry("after-eval", "2026-09-21T00:00:00Z", "failed"),
    ]
    reports = calibrate_source_health(
        entries, train_end=TRAIN_END, eval_start=EVAL_START, eval_end=EVAL_END,
        min_independent_dependencies=1,
    )
    row = _report(reports)
    assert row.train_attempts == 1
    assert row.eval_attempts == 2
    assert row.train_status_counts["available"] == 1
    assert row.eval_status_counts["available"] == 2
    assert all(item.source_id != "gap-only" for item in reports)


def test_report_keeps_semantic_accuracy_and_numeric_voi_unmeasured():
    row = _report(calibrate_source_health(
        [_entry("train", "2026-09-02T00:00:00Z", "available"),
         _entry("eval", "2026-09-13T00:00:00Z", "available")],
        train_end=TRAIN_END, eval_start=EVAL_START, eval_end=EVAL_END,
        min_independent_dependencies=1,
    ))
    encoded = row.as_dict()
    assert encoded["schema_version"] == "source-health-calibration-v1"
    assert encoded["semantic_accuracy_status"] == "UNMEASURED"
    assert encoded["numeric_voi_status"] == "UNMEASURED"
    assert "match-result probability" in " ".join(encoded["limitations"])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"train_end": EVAL_START, "eval_start": EVAL_START, "eval_end": EVAL_END}, "train_end"),
        ({"train_end": TRAIN_END, "eval_start": "2026-09-11T00:00:00Z", "eval_end": "2026-09-10T00:00:00Z"}, "train_end"),
        ({"train_end": TRAIN_END, "eval_start": EVAL_START, "eval_end": EVAL_END, "min_independent_dependencies": 0}, "positive"),
        ({"train_end": TRAIN_END, "eval_start": EVAL_START, "eval_end": EVAL_END, "confidence": 1.0}, "between"),
    ],
)
def test_rejects_invalid_windows_and_parameters(kwargs, message):
    base = {"train_end": TRAIN_END, "eval_start": EVAL_START, "eval_end": EVAL_END}
    base.update(kwargs)
    with pytest.raises(ValueError, match=message):
        calibrate_source_health([], **base)


def test_rejects_non_ledger_entries():
    with pytest.raises(ValueError, match="EvidenceLedgerEntry"):
        calibrate_source_health([{"source_id": "api"}], train_end=TRAIN_END,
                                eval_start=EVAL_START, eval_end=EVAL_END)
