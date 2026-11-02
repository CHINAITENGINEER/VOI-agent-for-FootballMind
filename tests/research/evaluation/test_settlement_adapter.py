"""Normalized fixture tests, kept separate from real result exports."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from football_research.data_adapters.settlement import export_results
from football_research.data_adapters.types import ResearchDataset, ResearchMatch
from football_research.evaluation.ledger import ForecastLedger


KICKOFF = datetime(2025, 1, 1, 15, tzinfo=timezone.utc)
OBSERVED = KICKOFF + timedelta(hours=3)


def match(**changes):
    return ResearchMatch(match_id="test:match:1", competition_id="test:league:1", season="2024",
                         home_team_id="test:team:1", away_team_id="test:team:2", kickoff=KICKOFF,
                         **{"home_goals": 2, "away_goals": 1, "status": "finished", "score_period": "regulation",
                            "result_available_at": OBSERVED, "fixture_available_at": KICKOFF - timedelta(days=2),
                            "input_manifest": (f"sha256:{'a' * 64}|observation:test-observation|at:{OBSERVED.isoformat()}",),
                            "source": "test-source", **changes})


def dataset(*matches, audit=None):
    return ResearchDataset(tuple(matches or (match(),)), (), "test-dataset-v1", ("test-only",), audit or {})


def test_export_final_and_ledger_roundtrip_idempotence(tmp_path):
    first, first_report = export_results(dataset())
    second, second_report = export_results(dataset())
    assert (first, first_report) == (second, second_report)
    assert first[0]["status"] == "final"
    assert first[0]["available_at"] == OBSERVED.isoformat()
    assert first[0]["observed_at"] == OBSERVED.isoformat()
    ledger = ForecastLedger(tmp_path)
    assert ledger.settle_result(first[0])["status"] == "appended"
    assert ledger.settle_result(second[0])["status"] == "duplicate"


@pytest.mark.parametrize("status", ["cancelled", "postponed", "disputed", "suspended"])
def test_status_changes_use_observation_not_fixture_available_at(tmp_path, status):
    rows, report = export_results(dataset(match(status=status, result_available_at=None, home_goals=None, away_goals=None)))
    row = rows[0]
    assert row["available_at"] == OBSERVED.isoformat()
    assert row["home_goals"] is None and row["away_goals"] is None
    assert row["status"] == ("disputed" if status == "suspended" else status)
    assert ForecastLedger(tmp_path).settle_result(row)["status"] == "appended"


@pytest.mark.parametrize("changes,reason", [
    ({"input_manifest": ()}, "input_manifest_missing"),
    ({"input_manifest": ("opaque-evidence-id",)}, "observation_time_missing"),
    ({"input_manifest": ("opaque|at:bad-date",)}, "invalid_observation_evidence"),
    ({"input_manifest": ("opaque|at:2999-01-01T00:00:00Z",)}, "future_observation"),
    ({"source": "unknown"}, "source_missing"),
    ({"result_available_at": None}, "final_label_or_availability_missing"),
    ({"result_available_at": OBSERVED + timedelta(hours=1)}, "availability_after_observation"),
    ({"information_mode": "synthetic_test"}, "unsupported_information_mode"),
    ({"information_mode": "historical_reconstruction"}, "reconstruction_assumption_missing"),
    ({"status": "scheduled", "result_available_at": None}, "no_settleable_status"),
    ({"status": "live", "result_available_at": None}, "no_settleable_status"),
    ({"score_period": "extra_time", "result_available_at": None}, "regulation_semantics_missing"),
])
def test_missing_or_invalid_evidence_is_reported(changes, reason):
    rows, report = export_results(dataset(match(**changes)))
    assert rows == []
    assert report["skip_reason_counts"] == {reason: 1}


def test_source_export_observation_and_reconstruction_assumptions(tmp_path):
    history = dataset(match(information_mode="historical_reconstruction", input_manifest=("platform-manifest-sha256:abc",),
                            result_available_at=KICKOFF + timedelta(hours=2)),
                      audit={"source_observed_at": OBSERVED.isoformat(), "assumptions": {"result_delay_hours": 2},
                             "source_revision_bias": "revised test snapshot"})
    rows, report = export_results(history)
    assert len(rows) == 1
    assert rows[0]["information_mode"] == "historical_reconstruction"
    assert "revised test snapshot" in rows[0]["availability_assumption"]
    assert rows[0]["observation_time_basis"] == ["dataset.audit.source_observed_at"]
    ForecastLedger(tmp_path).settle_result(rows[0])


def test_source_filter_missing_ids_and_empty_request():
    rows, report = export_results(dataset(), match_ids=["test:match:1", "another-source:match:1"])
    assert len(rows) == 1
    assert report["missing_match_ids"] == ["another-source:match:1"]
    assert export_results(dataset(), match_ids=[])[0] == []
    with pytest.raises(ValueError):
        export_results(dataset(), match_ids="test:match:1")


def test_dataset_version_and_changed_result_have_explicit_new_revisions(tmp_path):
    rows, _ = export_results(dataset())
    revised, _ = export_results(dataset(match(home_goals=0)))
    assert rows[0]["revision_id"] != revised[0]["revision_id"]
    assert revised[0]["supersedes"] == []
    ledger = ForecastLedger(tmp_path)
    ledger.settle_result(rows[0])
    ledger.settle_result(revised[0])
    assert len(ledger.results()) == 2


def test_latest_normalized_lineage_time_selected():
    earlier = OBSERVED - timedelta(minutes=10)
    rows, _ = export_results(dataset(match(input_manifest=(f"ref|at:{earlier.isoformat()}", f"ref|at:{OBSERVED.isoformat()}"))))
    assert rows[0]["observed_at"] == OBSERVED.isoformat()
