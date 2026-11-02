from copy import deepcopy

import pytest

from football_agent.collector_ledger_adapter import observation_event
from football_agent.source_health_observations import source_health_observations


AS_OF = "2026-09-28T10:00:00Z"


def event():
    return observation_event({
        "id": "obs-1", "source": "spftll", "request_id": "request-1",
        "endpoint": "https://example.test/fixture", "http_status": 200,
        "observed_at": "2026-09-28T08:01:00Z", "content_hash": "shared-content",
        "payload_path": "raw/fixture.body.gz", "error_class": None,
        "context": {"requested_at": "2026-09-28T08:00:00Z"},
    }, result_status="OK", source_version="v1", prediction_cutoff=AS_OF,
        entity_mapping_state="mapped")


def test_successful_received_content_is_not_a_fresh_or_accurate_health_label():
    report = source_health_observations([event()], as_of=AS_OF)
    row = report["observations"][0]
    assert row["transport_state"] == "SUCCESS"
    assert row["quality"] == "VALIDATED"
    assert row["latency_seconds"] == 60
    assert row["receipt_age_seconds"] == 7140
    assert row["content_timestamp_state"] == "UNKNOWN"
    assert row["semantic_accuracy"] is row["source_health_label"] is row["acquisition_value"] is None
    assert report["likelihood_fitted"] is report["numeric_voi_supported"] is False


@pytest.mark.parametrize("field", ["captured_at", "observed_at", "operational_available_at", "requested_at"])
def test_future_status_and_identity_are_absent_from_policy_projection(field):
    late = event()
    late.update(request_id="future-request", http_status=503, error_class="future-error")
    late[field] = "2026-09-28T11:00:00Z"
    baseline = source_health_observations([], as_of=AS_OF)
    assert source_health_observations([late], as_of=AS_OF) == baseline


def test_request_repeats_do_not_inflate_support_or_equate_unchanged_content_with_staleness():
    first = event()
    second = {**first, "request_id": "request-2", "dependency_key": "spftll:request:request-2"}
    report = source_health_observations([first, deepcopy(first), second], as_of=AS_OF)
    assert report["visible_request_count"] == 2
    assert len(report["repeated_content_groups"]) == 1
    assert all(row["source_health_label"] is None for row in report["observations"])


def test_failure_and_missing_latency_preserve_unknowns():
    failed = event()
    failed.update(http_status=None, error_class="SSLError", quality="PROVIDER_ERROR",
                  operational_available_at=None, requested_at=None, source_version="")
    row = source_health_observations([failed], as_of=AS_OF)["observations"][0]
    assert row["transport_state"] == "ERROR"
    assert row["latency_seconds"] is None
    assert row["source_version"] == "UNVERSIONED"
    assert row["semantic_accuracy"] is None


def test_published_timestamp_is_distinct_from_retrieval_age_and_accuracy():
    raw = event()
    raw["source_published_at"] = "2026-09-27T10:00:00Z"
    row = source_health_observations([raw], as_of=AS_OF)["observations"][0]
    assert row["declared_content_age_seconds"] == 86400
    assert row["receipt_age_seconds"] != row["declared_content_age_seconds"]
    assert row["source_health_label"] is None


def test_future_publication_date_is_not_freshness():
    raw = event()
    raw["source_published_at"] = "2026-10-01T10:00:00Z"
    row = source_health_observations([raw], as_of=AS_OF)["observations"][0]
    assert row["content_timestamp_state"] == "FUTURE_TIMESTAMP"
    assert row["declared_content_age_seconds"] is None


def test_rejects_conflicting_metadata_for_one_visible_request():
    row = event()
    with pytest.raises(ValueError, match="conflicting"):
        source_health_observations([row, {**row, "quality": "CONFLICT"}], as_of=AS_OF)
