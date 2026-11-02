from __future__ import annotations

import pytest

from football_agent.source_reliability import (
    REPORT_VERSION,
    SourceManifestError,
    SourceObservation,
    evaluate_source_manifest,
    validate_source_report,
)


CUTOFF = "2026-10-01T10:00:00Z"


def observation(source_id="injury-a", *, observed="2026-10-01T09:00:00Z",
                available="2026-10-01T09:02:00Z", valid_until="2026-10-01T11:00:00Z",
                freshness="fresh", availability="available", conflict_group=None,
                digest="digest-a", observation_id=None):
    return {
        "source_id": source_id,
        "observed_at": observed,
        "available_at": available,
        "valid_until": valid_until,
        "freshness": freshness,
        "availability": availability,
        "conflict_group": conflict_group,
        "digest": digest,
        "observation_id": observation_id,
    }


def test_fresh_observation_is_visible_without_probability_claim():
    report = evaluate_source_manifest(
        [observation()], cutoff=CUTOFF,
        candidates=[{"candidate_id": "injury", "source_ids": ["injury-a"]}],
    )

    source = report["sources"]["injury-a"]
    assert source["state"] == "available"
    assert source["usable_at_cutoff"] is True
    assert report["candidate_filter"][0]["eligibility"] == "eligible"
    assert report["unknown_is_not_gold"] is True
    assert report["probability_gold"] is False
    assert report["numeric_reliability_estimate"] is None


def test_observation_after_cutoff_is_future_and_is_not_visible():
    row = observation(observed="2026-10-01T10:01:00Z", available="2026-10-01T10:02:00Z")
    report = evaluate_source_manifest([row], cutoff=CUTOFF,
                                      candidates=[{"candidate_id": "injury", "source_ids": ["injury-a"]}])

    assert report["sources"]["injury-a"]["state"] == "future"
    assert report["sources"]["injury-a"]["usable_at_cutoff"] is False
    assert "SOURCE_FUTURE_NOT_VISIBLE_AT_CUTOFF" in report["candidate_filter"][0]["reasons"]
    assert report["candidate_filter"][0]["eligibility"] == "defer_unknown"


def test_valid_until_at_cutoff_is_stale_and_requests_refresh():
    row = observation(valid_until=CUTOFF)
    report = evaluate_source_manifest([row], cutoff=CUTOFF)

    source = report["sources"]["injury-a"]
    assert source["state"] == "stale"
    assert source["action"] == "refresh_source"
    assert "VALID_UNTIL_REACHED" in source["reasons"]


def test_missing_timestamps_or_digest_remain_unknown():
    row = observation(observed=None, available=None, valid_until=None,
                      freshness="unknown", availability="available", digest=None)
    report = evaluate_source_manifest([row], cutoff=CUTOFF)

    source = report["sources"]["injury-a"]
    assert source["state"] == "unknown"
    assert source["action"] == "defer_unknown"
    assert source["usable_at_cutoff"] is False


def test_duplicate_identical_observations_are_reported_without_double_counting():
    row = observation(observation_id="o1")
    duplicate = observation(observation_id="o2")
    report = evaluate_source_manifest(
        [row, duplicate], cutoff=CUTOFF,
        candidates=[{"candidate_id": "injury", "source_ids": ["injury-a"]}],
    )

    source = report["sources"]["injury-a"]
    assert report["duplicate_source_ids"] == ["injury-a"]
    assert source["state"] == "duplicate"
    assert source["usable_at_cutoff"] is True
    assert report["candidate_filter"][0]["eligibility"] == "eligible_with_warnings"
    assert "DUPLICATE_SOURCE_DEDUP_REQUIRED" in report["candidate_filter"][0]["reasons"]


def test_overlapping_digests_in_same_source_are_conflict_and_defer():
    first = observation(digest="digest-a", valid_until="2026-10-01T12:00:00Z")
    second = observation(digest="digest-b", observed="2026-10-01T09:30:00Z",
                         available="2026-10-01T09:32:00Z", valid_until="2026-10-01T12:30:00Z")
    report = evaluate_source_manifest(
        [first, second], cutoff=CUTOFF,
        candidates=[{"candidate_id": "injury", "source_ids": ["injury-a"]}],
    )

    assert report["conflict_source_ids"] == ["injury-a"]
    assert report["sources"]["injury-a"]["state"] == "conflict"
    assert report["candidate_filter"][0]["eligibility"] == "defer_unknown"
    assert "SOURCE_CONFLICT_REQUIRES_RESOLUTION" in report["candidate_filter"][0]["reasons"]


def test_non_overlapping_revision_is_not_conflict():
    old = observation(digest="digest-a", valid_until="2026-10-01T09:30:00Z")
    new = observation(digest="digest-b", observed="2026-10-01T09:30:00Z",
                      available="2026-10-01T09:31:00Z", valid_until="2026-10-01T12:00:00Z")
    report = evaluate_source_manifest([old, new], cutoff=CUTOFF)

    assert report["conflict_source_ids"] == []
    assert report["sources"]["injury-a"]["state"] == "available"


def test_explicit_conflict_group_detects_conflicting_providers():
    first = observation("provider-a", conflict_group="lineup", digest="digest-a")
    second = observation("provider-b", conflict_group="lineup", digest="digest-b")
    report = evaluate_source_manifest([first, second], cutoff=CUTOFF)

    assert report["conflict_source_ids"] == ["provider-a", "provider-b"]
    assert report["sources"]["provider-a"]["state"] == "conflict"
    assert report["sources"]["provider-b"]["state"] == "conflict"


def test_candidate_with_one_fresh_alternative_can_proceed_with_warning():
    report = evaluate_source_manifest(
        [observation("fresh"), observation("stale", valid_until=CUTOFF)],
        cutoff=CUTOFF,
        candidates=[{"candidate_id": "weather", "source_ids": ["stale", "fresh"]}],
    )

    result = report["candidate_filter"][0]
    assert result["eligibility"] == "eligible_with_warnings"
    assert "SOURCE_STALE_REQUIRES_REFRESH" in result["reasons"]


def test_unknown_candidate_source_and_empty_candidate_are_deferred():
    report = evaluate_source_manifest(
        [observation()], cutoff=CUTOFF,
        candidates=[
            {"candidate_id": "missing", "source_ids": ["not-listed"]},
            {"candidate_id": "empty", "source_ids": []},
        ],
    )

    assert all(row["eligibility"] == "defer_unknown" for row in report["candidate_filter"])
    assert "SOURCE_ID_UNKNOWN:not-listed" in report["candidate_filter"][0]["reasons"]
    assert report["candidate_filter"][1]["reasons"] == ["NO_SOURCE_IDS"]


def test_report_digest_and_gold_boundary_validate_and_tamper_is_rejected():
    report = evaluate_source_manifest([observation()], cutoff=CUTOFF)
    result = validate_source_report(report)
    assert result["valid"] is True
    assert report["schema_version"] == REPORT_VERSION

    tampered = dict(report)
    tampered["sources"] = dict(report["sources"])
    tampered["sources"]["injury-a"] = dict(report["sources"]["injury-a"])
    tampered["sources"]["injury-a"]["state"] = "available"
    tampered["sources"]["injury-a"]["probability_gold"] = True
    with pytest.raises(SourceManifestError, match="DIGEST_MISMATCH"):
        validate_source_report(tampered)


def test_manifest_rejects_invalid_order_and_cutoff():
    with pytest.raises(SourceManifestError, match="available_at_BEFORE_observed_at"):
        SourceObservation.from_mapping(observation(observed="2026-10-01T09:10:00Z",
                                                   available="2026-10-01T09:00:00Z"))
    with pytest.raises(SourceManifestError, match="cutoff_INVALID_TIME"):
        evaluate_source_manifest([observation()], cutoff="not-a-time")

