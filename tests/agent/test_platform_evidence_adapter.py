import pytest

from football_agent.platform_evidence_adapter import (
    acquisition_result_to_evidence_rows,
    acquisition_results_to_evidence_rows,
)
from football_platform.information import AcquisitionResult, CostVector, EvidenceGroup, EvidenceTimes


CUTOFF = "2026-09-27T10:00:00Z"


def group(group_id="lineup", *, quality="VALIDATED", dependency="spftll:lineup"):
    return EvidenceGroup(
        group_id=group_id,
        values={"status": "redacted"},
        source="spftll",
        times=EvidenceTimes(
            observed_at="2026-09-27T08:00:00Z",
            operational_available_at="2026-09-27T08:10:00Z",
            feature_available_at="2026-09-27T08:10:00Z",
            basis="actual_observation_and_ingestion",
        ),
        quality=quality,
        evidence_refs=[f"archive:{group_id}"],
        version="group-v1",
        source_id="spftll:fixture",
        observed_at="2026-09-27T08:00:00Z",
        available_at="2026-09-27T08:10:00Z",
        content_hash=f"sha:{group_id}",
        dependency_key=dependency,
    )


def acquired(action_id="lineup", groups=None, status="ACQUIRED", reason=None):
    return AcquisitionResult(
        episode_id="episode-1",
        action_id=action_id,
        request_id=f"request-{action_id}",
        status=status,
        revealed_groups=groups or {},
        charged_cost=CostVector(),
        sequence=1,
        reason=reason,
        resulting_cutoff=CUTOFF,
    )


def test_public_acquisition_result_and_evidence_group_map_to_one_candidate_row():
    result = acquired(groups={"lineup": group()})
    rows = acquisition_result_to_evidence_rows(result, action_to_candidate={"lineup": "lineup-candidate"})
    assert len(rows) == 1
    row = rows[0]
    assert row["candidate_id"] == "lineup-candidate"
    assert row["quality"] == "VALIDATED"
    assert row["dependency_key"] == "spftll:lineup"
    assert row["source_id"] == "spftll:fixture"
    assert row["group_ids"] == ["lineup"]
    assert row["evidence_refs"] == ["archive:lineup"]


def test_failure_result_without_groups_preserves_provider_status_for_binding():
    result = acquired(action_id="weather", status="TIMEOUT", reason="upstream_timeout")
    row = acquisition_result_to_evidence_rows(result, action_to_candidate={"weather": "weather-candidate"})[0]
    assert row["candidate_id"] == "weather-candidate"
    assert row["quality"] == "PROVIDER_ERROR"
    assert row["status"] == "TIMEOUT"
    assert row["error_class"] == "upstream_timeout"


def test_multiple_revealed_groups_are_merged_without_claiming_independent_sources():
    result = acquired(groups={
        "lineup": group("lineup", dependency="dep-a"),
        "form": group("form", quality="PARTIAL", dependency="dep-b"),
    })
    row = acquisition_result_to_evidence_rows(result, action_to_candidate={"lineup": "combined"})[0]
    assert row["candidate_id"] == "combined"
    assert row["dependency_key"] == "action:lineup"
    assert row["quality"] == "PARTIAL"
    assert row["group_ids"] == ["lineup", "form"]


def test_batch_adapter_rejects_two_versions_for_one_candidate():
    first = acquired(groups={"lineup": group()})
    second = acquired(groups={"lineup": group("lineup-2")})
    with pytest.raises(ValueError, match="Multiple acquisition results"):
        acquisition_results_to_evidence_rows(
            [first, second], action_to_candidate={"lineup": "lineup-candidate"}
        )


def test_action_map_is_required_to_preserve_candidate_identity():
    with pytest.raises(ValueError, match="absent from candidate map"):
        acquisition_result_to_evidence_rows(
            acquired(groups={"lineup": group()}), action_to_candidate={}
        )


@pytest.mark.parametrize("status", ["NO_DATA", "UNAVAILABLE", "BUDGET_EXCEEDED"])
def test_non_acquisition_statuses_keep_an_explicit_failure_row(status):
    row = acquisition_result_to_evidence_rows(
        acquired(action_id="lineup", status=status),
        action_to_candidate={"lineup": "lineup-candidate"},
    )[0]
    assert row["quality"] == "PROVIDER_ERROR"
    assert row["status"] == status
