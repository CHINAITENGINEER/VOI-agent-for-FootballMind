from football_agent.evidence_ledger import EvidenceLedgerEntry
from football_agent.source_return_calibration import calibrate_source_returns, calibrate_voi_links
from football_agent.voi_evidence_bridge import VoIEvidenceLink
import pytest


def entry(candidate, *, source="source-v1", source_id=None, availability="available", dependency=None, sequence=1):
    if availability == "failed":
        observed = available = None
        quality = "PROVIDER_ERROR"
    elif availability == "late":
        observed, available = "2026-09-27T10:15:00Z", "2026-09-27T10:20:00Z"
        quality = "VALIDATED"
    else:
        observed, available = "2026-09-27T08:00:00Z", "2026-09-27T08:10:00Z"
        quality = "VALIDATED"
    return EvidenceLedgerEntry.from_row(
        {"candidate_id": candidate, "source_id": source_id or source, "dependency_key": dependency or f"dep-{candidate}",
         "quality": quality, "availability": availability, "observed_at": observed,
         "operational_available_at": available, "evidence_refs": [f"archive:{candidate}"],
         "content_hash": f"hash:{candidate}"},
        episode_id="episode-1", target_key="match-1", action_id=candidate,
        request_id=f"request-{candidate}-{sequence}", sequence=sequence,
        prediction_cutoff="2026-09-27T10:00:00Z", captured_at="2026-09-27T08:30:00Z",
        requested_at="2026-09-27T08:20:00Z", source_version=source,
    )


def link(source, *, status="linked_available", error=.01, dependency=None):
    return VoIEvidenceLink(
        link_id=f"link-{source}", trace_id=f"trace-{source}", episode_id="episode-1",
        target_key="match-1", request_id=f"request-{source}", action_id=source,
        candidate_id=source, evidence_entry_id=f"entry-{source}",
        evidence_availability="available", evidence_quality="VALIDATED",
        requested_at="2026-09-27T08:20:00Z", source_version=source,
        source_id="fixture-source", dependency_key=dependency or f"dep-{source}",
        prediction_cutoff="2026-09-27T10:00:00Z", expected_information_value=.1,
        observed_risk_reduction=.1 - error, calibration_error=error, status=status,
    )


def test_source_return_report_counts_statuses_and_dependence():
    rows = [entry("a", dependency="shared"), entry("b", availability="late", dependency="shared", sequence=2),
            entry("c", availability="failed", sequence=3)]
    report = calibrate_source_returns(rows, time_bucket="pre_match", min_attempts=2)[0]
    assert report.attempts == 3
    assert report.independent_dependency_count == 2
    assert report.duplicate_dependency_count == 1
    assert report.status_counts == {"available": 1, "late": 1, "failed": 1, "duplicate": 0}
    assert report.available_rate == 1 / 3
    assert report.support_status == "SUPPORTED_DIAGNOSTIC"


def test_source_versions_are_partitioned_and_low_support_is_explicit():
    reports = calibrate_source_returns([entry("a", source="v1", source_id="source-a"), entry("b", source="v1", source_id="source-b")],
                                       time_bucket="pre_match", min_attempts=2)
    assert [report.source_id for report in reports] == ["source-a", "source-b"]
    assert [report.source_version for report in reports] == ["v1", "v1"]
    assert all(report.support_status == "INSUFFICIENT_SUPPORT" for report in reports)


def test_voi_calibration_compares_expected_and_observed_reduction():
    reports = calibrate_voi_links([link("v1", dependency="d1"), link("v1", error=.03, dependency="d2")], min_available_links=2)
    report = reports[0]
    assert report.available_link_count == 2
    assert report.mean_expected_information_value == .1
    assert report.mean_observed_risk_reduction == pytest.approx(.08)
    assert report.mean_calibration_error == pytest.approx(.02)
    assert report.support_status == "SUPPORTED_DIAGNOSTIC"


def test_voi_calibration_excludes_stop_and_unusable_links():
    stop = VoIEvidenceLink(
        link_id="stop", trace_id="trace-stop", episode_id="e", target_key="m", request_id="r",
        action_id="STOP", candidate_id="STOP", evidence_entry_id=None, evidence_availability=None,
        evidence_quality=None, requested_at=None, source_version=None,
        prediction_cutoff="2026-09-27T10:00:00Z", expected_information_value=None,
        observed_risk_reduction=None, calibration_error=None, status="stop_recorded",
    )
    unusable = link("v1-late", status="linked_unusable")
    report = calibrate_voi_links([stop, unusable], min_available_links=1)[0]
    assert report.link_count == 1
    assert report.available_link_count == 0
    assert report.mean_calibration_error is None
    assert report.support_status == "INSUFFICIENT_SUPPORT"
