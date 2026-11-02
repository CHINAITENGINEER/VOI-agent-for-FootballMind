"""Constructed comparisons and failed acquisitions; no real policy-gain claims."""
from datetime import datetime, timedelta, timezone
import json

import pytest

from football_platform.information.acquisition_journal import AcquisitionDecisionJournal
from football_platform.information.fixture_corroboration import audit_failed_official_capture, compare_assertion
from football_spider.storage.raw_archive import RawArchive, json_bytes


FIXTURE = {"match_id": "source:match:1", "home_team_id": "canonical:team:1", "away_team_id": "canonical:team:2",
           "competition_id": "canonical:competition:1", "kickoff": "2026-10-01T18:45:00Z"}
FIELDS = {key: value for key, value in FIXTURE.items() if key != "match_id"}


def compare(**changes):
    args = dict(source_event_id="official:match:999", explicitly_mapped_match_id=FIXTURE["match_id"],
                fields=FIELDS, evidence_basis="acquired_structured", available_at="2026-10-01T12:00:00Z",
                cutoff="2026-10-01T12:01:00Z")
    args.update(changes)
    return compare_assertion(FIXTURE, **args)


def test_explicit_identity_mapping_and_equal_instants_agree():
    result = compare(fields={**FIELDS, "kickoff": "2026-10-01T19:45:00+01:00"})
    assert result["status"] == "CONFIRMED"
    assert result["semantics"] == "input_comparison_not_semantic_source_verification"
    assert result["disputed_fields"] == result["unknown_fields"] == []


@pytest.mark.parametrize("field,value", [("home_team_id", "canonical:team:9"), ("away_team_id", "canonical:team:8"),
                                        ("competition_id", "canonical:competition:8"), ("kickoff", "2026-10-01T19:45:00Z")])
def test_same_mapped_event_with_identity_or_timezone_discrepancy_is_disputed(field, value):
    result = compare(fields={**FIELDS, field: value})
    assert result["status"] == "DISPUTED" and result["disputed_fields"] == [field]


@pytest.mark.parametrize("changes", [{"source_event_id": None}, {"explicitly_mapped_match_id": None},
                                     {"explicitly_mapped_match_id": "different-match"}])
def test_similar_names_or_dates_never_establish_event_identity(changes):
    assert compare(**changes)["reason"] == "explicit_source_event_identity_mapping_required"


@pytest.mark.parametrize("basis", ["documentary_only", "unavailable"])
def test_unpurchased_or_web_document_reading_does_not_confirm_or_read_payload(basis):
    result = compare(evidence_basis=basis, fields={**FIELDS, "kickoff": "not-a-date"})
    assert result["status"] == "UNCONFIRMED" and result["matched_fields"] == []


@pytest.mark.parametrize("changes", [{"available_at": None}, {"cutoff": None},
                                     {"available_at": "2026-10-01T12:02:00Z"},
                                     {"cutoff": "2026-10-01T18:45:00Z"}])
def test_actual_availability_is_required_and_future_or_postmatch_is_excluded(changes):
    assert compare(**changes)["status"] == "UNCONFIRMED"


def test_partial_fields_are_unknown_not_false_or_confirmed():
    result = compare(fields={"home_team_id": FIELDS["home_team_id"]})
    assert result["status"] == "UNCONFIRMED"
    assert result["unknown_fields"] == ["away_team_id", "competition_id", "kickoff"]


def test_naive_time_and_fuzzy_names_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        compare(fields={**FIELDS, "kickoff": "2026-10-01T18:45:00"})
    with pytest.raises(ValueError, match="inert identifiers"):
        compare(fields={**FIELDS, "home_team_id": "Country Team"})


def test_fixture_comparison_does_not_accept_qualification_or_tactical_inference():
    with pytest.raises(ValueError, match="Unexpected fixture fields"):
        compare(fields={**FIELDS, "qualification": "qualified"})


def failed_capture(root):
    clock = [datetime(2026, 10, 1, 12, tzinfo=timezone.utc)]
    journal = AcquisitionDecisionJournal(root / "journal", RawArchive(root / "archive"), clock=lambda: clock[0])
    run = journal.start(request_limit=2, fixture=FIXTURE)
    run_id = run["run_id"]
    (root / "run-id.txt").write_text(run_id, encoding="utf-8")
    for number, status in enumerate([403, None]):
        request = f"request-{number}"
        endpoint = "https://official.example/match"
        journal.begin(run_id, request_id=request, action_id="official.fixture", candidates=["official.fixture"],
                      input_parameters={"endpoint": endpoint, "params": {}, "source": "constructed_official"})
        requested = clock[0]
        clock[0] += timedelta(seconds=1)
        obs = journal.archive.save("constructed_official", endpoint, {}, b"blocked" if status else b"", status=status,
                                   error_class=None if status else "ConnectionError", observed_at=clock[0],
                                   context={"requested_at": requested.isoformat(), "journal_run_id": run_id,
                                            "journal_request_id": request})
        journal.complete(run_id, request_id=request, observation=obs)
    journal.stop(run_id, "RAW_UNAVAILABLE")
    (root / "documentary-review.json").write_bytes(json_bytes({
        "evidence_class": "human_review_of_web_tool_output_not_provider_raw", "operational_evidence_ids": [],
        "conclusion": "CONSTRUCTED_SUPPORT_ONLY", "sources": []}))
    return journal


def test_failed_http_and_connection_attempts_still_cost_and_publish_no_claim(tmp_path):
    failed_capture(tmp_path)
    result = audit_failed_official_capture(tmp_path)
    assert result["charged_requests"] == result["real_http_attempts"] == 2
    assert result["machine_corroboration_status"] == "UNCONFIRMED"
    assert result["structured_context_claims_added"] == result["real_AI_policy_gain_samples"] == 0
    assert result["prediction"] is None and result["candidate_expected_gains"] is None


def test_audit_rejects_browser_note_relabelled_as_acquired_evidence(tmp_path):
    failed_capture(tmp_path)
    path = tmp_path / "documentary-review.json"
    note = json.loads(path.read_bytes())
    note["operational_evidence_ids"] = ["manufactured-official-claim"]
    path.write_bytes(json_bytes(note))
    with pytest.raises(ValueError, match="outside operational"):
        audit_failed_official_capture(tmp_path)
