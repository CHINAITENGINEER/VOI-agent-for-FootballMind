import copy
import pytest

from football_agent.platform_acquisition_context import (
    HISTORICAL, OPERATIONAL, PlatformAcquisitionContextError,
    build_platform_acquisition_context,
)

CUTOFF = "2026-09-30T14:00:00Z"

def base_match(**extra):
    value = {
        "match_id": "m-136", "kickoff": "2026-09-30T18:00:00Z",
        "prediction_cutoff": CUTOFF, "competition_id": "c1",
        "home_team_id": "h", "away_team_id": "a",
        "observed_at": "2026-09-30T10:00:00Z",
        "ingested_at": "2026-09-30T11:00:00Z",
        "available_at": "2026-09-30T12:00:00Z",
        "home_score": 3, "away_score": 0,
        "label_status": "INDEPENDENT_EVALUATOR_GOLD",
    }
    value.update(extra)
    return value

def row(source_id="injury", **extra):
    value = {"source_id": source_id, "kind": "injury",
             "observed_at": "2026-09-30T09:00:00Z",
             "ingested_at": "2026-09-30T09:30:00Z",
             "available_at": "2026-09-30T09:15:00Z",
             "freshness": "fresh", "valid_until": "2026-09-30T17:00:00Z",
             "digest": "d-1", "values": {"player_id": "p1", "status": "active"}}
    value.update(extra)
    return value

def test_operational_uses_max_clock_and_keeps_observed():
    result = build_platform_acquisition_context(
        base_match(operational_available_at="2026-09-30T13:45:00Z"),
        [row(operational_available_at="2026-09-30T13:50:00Z")], information_mode=OPERATIONAL)
    obs = next(x for x in result["actor_context"]["source_observations"] if x["source_id"] == "injury")
    assert obs["available_at"] == "2026-09-30T13:50:00Z"
    assert obs["observed_at"] == "2026-09-30T09:00:00Z"
    assert obs["freshness"] == "fresh"
    assert obs["valid_until"] == "2026-09-30T17:00:00Z"
    assert "status" not in obs["values"]

def test_missing_operational_clock_is_excluded_and_cannot_bypass_fixture():
    with pytest.raises(PlatformAcquisitionContextError, match="TARGET_FIXTURE"):
        build_platform_acquisition_context(base_match(ingested_at=None), [row()])

def test_late_fixture_max_is_excluded_even_when_ingested_looks_early():
    with pytest.raises(PlatformAcquisitionContextError, match="TARGET_FIXTURE"):
        build_platform_acquisition_context(base_match(operational_available_at="2026-09-30T15:00:00Z"), [row()])

def test_historical_unknown_evidence_is_audit_only_and_actual_clocks_are_not_actor_data():
    result = build_platform_acquisition_context(
        base_match(observed_at="2026-10-01T08:00:00Z", ingested_at="2026-10-01T08:05:00Z"),
        [row("unknown", observed_at="2026-10-01T08:00:00Z", ingested_at="2026-10-01T08:05:00Z")],
        information_mode=HISTORICAL, reconstruction_assumptions={"fixture_lead_hours": 24})
    assert [x["source_id"] for x in result["actor_context"]["source_observations"]] == ["fixture:m-136"]
    assert "unknown" in {x["source_id"] for x in result["audit"]["excluded_source_observations"]}
    fixture = result["actor_context"]["source_observations"][0]
    assert fixture["observed_clock_status"] == "unknown_in_reconstructed_view"
    assert "observed_at" not in fixture

def test_foreign_fixture_does_not_authorize_target_identity():
    with pytest.raises(PlatformAcquisitionContextError, match="TARGET_FIXTURE"):
        build_platform_acquisition_context(base_match(ingested_at=None), [
            {"source_id": "foreign", "kind": "fixture_identity", "match_id": "other",
             "observed_at": "2026-09-30T10:00:00Z", "ingested_at": "2026-09-30T11:00:00Z",
             "available_at": "2026-09-30T12:00:00Z"}
        ])

def test_result_changes_do_not_change_actor_context_or_embed_candidates():
    first = build_platform_acquisition_context(base_match(), [row()], candidate_catalog=[{"candidate_id": "query"}])
    changed = base_match(); changed["home_score"] = 0; changed["away_score"] = 4; changed["status"] = "finished"
    second = build_platform_acquisition_context(changed, [row()])
    assert first["actor_context"] == second["actor_context"]
    assert "candidate_catalog" not in first["actor_context"]
    assert first["evaluator_sidecar"]["home_goals"] == 3

def test_explicit_historical_evidence_clock_is_usable():
    result = build_platform_acquisition_context(
        base_match(), [row("xg", kind="xg", reconstructed_available_at="2026-09-30T08:00:00Z")],
        information_mode=HISTORICAL, reconstruction_assumptions={"fixture_lead_hours": 24})
    ids = {x["source_id"] for x in result["actor_context"]["source_observations"]}
    assert {"fixture:m-136", "xg"}.issubset(ids)


def test_missing_clock_evidence_never_falls_back_to_ingestion():
    result = build_platform_acquisition_context(base_match(), [row(observed_at=None, available_at=None)])
    assert "injury" not in {x["source_id"] for x in result["actor_context"]["source_observations"]}
    assert result["audit"]["excluded_source_observations"][0]["reason"] == "missing_required_operational_clock"


def test_explicit_availability_and_claim_metadata_are_preserved():
    obs = build_platform_acquisition_context(base_match(), [row(availability="available", claim_id="lineup:p1")])["actor_context"]["source_observations"][0]
    assert obs["availability"] == "available"
    assert obs["claim_id"] == "lineup:p1"


def test_freshness_and_digest_are_never_invented():
    no_quality = row()
    for key in ("freshness", "valid_until", "digest"):
        no_quality.pop(key)
    observations = build_platform_acquisition_context(base_match(), [no_quality])["actor_context"]["source_observations"]
    assert all("freshness" not in x and "digest" not in x for x in observations)


def test_clockless_known_and_candidate_catalog_are_not_actor_inputs():
    ordinary = build_platform_acquisition_context(base_match(), [row()])["actor_context"]
    forged = build_platform_acquisition_context(base_match(known={"strength": 100, "home_score": 3}), [row()], candidate_catalog=[{"candidate_id": "query"}])["actor_context"]
    assert ordinary == forged


def test_future_duplicate_id_and_future_changes_leave_actor_exactly_unchanged():
    current = row(values={"player_id": "p1", "fit": True})
    reference = build_platform_acquisition_context(base_match(), [current])["actor_context"]
    future = row(available_at="2026-09-30T16:00:00Z", values={"future_result": 999}, digest="future-hash")
    result = build_platform_acquisition_context(base_match(), [future, current])["actor_context"]
    assert result == reference
    future.update(source_id="future-new-id", values={"unseen": "different"})
    assert build_platform_acquisition_context(base_match(), [current, future])["actor_context"] == reference


def test_source_result_changes_and_raw_digest_changes_are_audit_only():
    first = row(values={"rank": 3, "nested": {"home_score": 1, "away_score": 0}}, digest="before")
    second = row(values={"rank": 3, "nested": {"home_score": 0, "away_score": 5}}, digest="after")
    a = build_platform_acquisition_context(base_match(content_hash="match-label-a"), [first])
    b = build_platform_acquisition_context(base_match(content_hash="match-label-b", home_score=0, away_score=5), [second])
    assert a["actor_context"] == b["actor_context"]
    assert a["audit"]["actual_clocks"][1]["original_content_digest"] != b["audit"]["actual_clocks"][1]["original_content_digest"]


def test_clean_payload_digest_is_preserved():
    item = row(values={"rank": 3})
    observed = build_platform_acquisition_context(base_match(), [item])["actor_context"]["source_observations"][0]
    assert observed["digest"] == "d-1"


@pytest.mark.parametrize("values", [
    {},
    {"match_id": "m-136"},
    {"match_id": "m-136", "kickoff": "2026-09-30T18:00:00Z", "home_team_id": "other", "away_team_id": "a"},
    {"match_id": "foreign", "kickoff": "2026-09-30T18:00:00Z", "home_team_id": "h", "away_team_id": "a"},
])
def test_incomplete_or_other_fixture_identity_never_authorizes_late_target(values):
    with pytest.raises(PlatformAcquisitionContextError, match="TARGET_FIXTURE"):
        build_platform_acquisition_context(base_match(operational_available_at="2026-09-30T15:00:00Z"), [
            row("other-fixture", kind="fixture_identity", values=values)
        ])


def test_conflicting_fixture_team_name_does_not_authorize_identity():
    visible = row("earlier-fixture", kind="fixture_identity", values={
        "match_id": "m-136", "kickoff": "2026-09-30T18:00:00Z", "home_team_id": "h", "away_team_id": "a", "home_team_name": "Visible Home"})
    with pytest.raises(PlatformAcquisitionContextError, match="TARGET_FIXTURE"):
        build_platform_acquisition_context(base_match(home_team_name="Late altered name", operational_available_at="2026-09-30T15:00:00Z"), [visible])


def test_historical_actual_clock_changes_never_change_actor_projection():
    kwargs = {"information_mode": HISTORICAL, "reconstruction_assumptions": {"fixture_lead_hours": 24}}
    first = build_platform_acquisition_context(base_match(observed_at=None, ingested_at=None), [], **kwargs)
    second = build_platform_acquisition_context(base_match(observed_at="2026-10-01T08:00:00Z", ingested_at="2026-10-01T08:05:00Z"), [], **kwargs)
    assert first["actor_context"] == second["actor_context"]
    assert first["audit"]["actual_clocks"] != second["audit"]["actual_clocks"]


def test_synthetic_mode_is_labeled_and_preserves_its_explicit_observed_clock():
    context = build_platform_acquisition_context(base_match(), [row()], information_mode="synthetic")["actor_context"]
    assert context["information_mode"] == "synthetic"
    assert context["source_observations"][0]["observed_at"] == "2026-09-30T09:00:00Z"


@pytest.mark.parametrize("lead", [float("inf"), float("nan"), -1, True])
def test_historical_assumption_requires_finite_nonnegative_hours(lead):
    with pytest.raises(PlatformAcquisitionContextError, match="FIXTURE_RECONSTRUCTION"):
        build_platform_acquisition_context(base_match(), information_mode=HISTORICAL, reconstruction_assumptions={"fixture_lead_hours": lead})
