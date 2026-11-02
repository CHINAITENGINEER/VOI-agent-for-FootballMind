import pytest

from football_agent.round135_platform_adapter import (
    HISTORICAL,
    OPERATIONAL,
    build_platform_asof_episode,
    platform_match_to_episode_row,
)


CUTOFF = "2026-09-30T14:00:00Z"


def match(**overrides):
    value = {
        "match_id": "m-135",
        "kickoff": "2026-09-30T18:00:00Z",
        "prediction_cutoff": CUTOFF,
        "competition_id": "c-1",
        "competition_name": "Synthetic League",
        "home_team_id": "home",
        "away_team_id": "away",
        "home_team_name": "Home",
        "away_team_name": "Away",
        "source": "spftll",
        "source_id": "fixture-row",
        "observation_id": "obs-match",
        "observed_at": "2026-09-30T12:00:00Z",
        "ingested_at": "2026-09-30T13:00:00Z",
        "available_at": "2026-09-30T12:30:00Z",
        "status": "finished",
        "score_period": "regulation",
        "home_goals": 2,
        "away_goals": 1,
        "label_status": "INDEPENDENT_EVALUATOR_GOLD",
        "result_available_at": "2026-09-30T20:00:00Z",
    }
    value.update(overrides)
    return value


def evidence(source_id="table", **overrides):
    value = {
        "source_id": source_id,
        "kind": "standings",
        "observed_at": "2026-09-30T10:00:00Z",
        "ingested_at": "2026-09-30T11:00:00Z",
        "available_at": "2026-09-30T10:30:00Z",
        "content_hash": "hash-" + source_id,
        "record": {"rank": 2, "home_score": 99},
    }
    value.update(overrides)
    return value


def test_operational_clock_is_max_of_all_platform_clocks():
    result = platform_match_to_episode_row(
        match(available_at="2026-09-30T12:30:00Z",
              operational_available_at="2026-09-30T13:45:00Z"),
        [evidence(available_at="2026-09-30T10:30:00Z",
                  operational_available_at="2026-09-30T13:50:00Z")],
        information_mode=OPERATIONAL,
    )
    row = result["match"]
    assert row["available_at"] == "2026-09-30T13:45:00Z"
    obs = next(item for item in row["source_observations"] if item["source_id"] == "table")
    assert obs["available_at"] == "2026-09-30T13:50:00Z"
    assert obs["availability_basis"] == "max(observed_at,ingested_at,available_at)"
    assert result["audit"]["today_ingest_clock_used_as_historical_visibility"] is False


def test_operational_missing_clock_stays_unknown_and_builder_blocks_fixture():
    result = platform_match_to_episode_row(
        match(ingested_at=None), [evidence()], information_mode=OPERATIONAL)
    assert result["match"]["available_at"] is None
    assert "match.observed_at" not in result["audit"]["unknown_clock_rows"]
    assert "match.ingested_at" in result["audit"]["unknown_clock_rows"]
    assert result["audit"]["target_fixture_identity_cutoff_visible"] is False
    with pytest.raises(ValueError, match="fixture identity"):
        build_platform_asof_episode(match(ingested_at=None), [evidence()], information_mode=OPERATIONAL)


def test_historical_reconstruction_keeps_actual_clocks_and_requires_explicit_evidence_clock():
    result = platform_match_to_episode_row(
        match(observed_at="2026-10-01T08:00:00Z", ingested_at="2026-10-01T08:05:00Z"),
        [evidence(observed_at="2026-10-01T08:00:00Z", ingested_at="2026-10-01T08:05:00Z")],
        information_mode=HISTORICAL,
        reconstruction_assumptions={"fixture_lead_hours": 24},
    )
    row = result["match"]
    assert row["information_mode"] == HISTORICAL
    assert row["observed_at"] == "2026-10-01T08:00:00Z"
    assert row["ingested_at"] == "2026-10-01T08:05:00Z"
    fixture = next(item for item in row["source_observations"] if item["kind"] == "fixture_identity")
    table = next(item for item in row["source_observations"] if item["source_id"] == "table")
    assert fixture["available_at"] == "2026-09-29T18:00:00Z"
    assert table["available_at"] is None  # no implicit backfill for standings
    assert table["reconstruction_clock_is_assumed"] is True
    assert result["audit"]["actual_clocks_preserved"] is True


def test_non_fixture_historical_evidence_does_not_inherit_fixture_lead():
    result = platform_match_to_episode_row(
        match(),
        [evidence("injury", kind="injury")],
        information_mode=HISTORICAL,
        reconstruction_assumptions={"fixture_lead_hours": 24},
    )
    injury = next(item for item in result["match"]["source_observations"] if item["source_id"] == "injury")
    assert injury["available_at"] is None
    assert injury["availability_basis"] == "unknown_missing_assumption"
    assert "injury" in result["audit"]["unknown_clock_rows"]


def test_future_operational_evidence_never_enters_actor_snapshot():
    with pytest.raises(ValueError, match="fixture identity"):
        build_platform_asof_episode(
            match(available_at="2026-09-30T15:00:00Z"),
            [evidence(available_at="2026-09-30T15:00:00Z")],
            information_mode=OPERATIONAL,
        )


def test_result_is_only_in_evaluator_sidecar_after_build():
    result = build_platform_asof_episode(
        match(), [evidence()], information_mode=OPERATIONAL)
    snapshot = result["episode"]["pre_action"]["snapshot"]
    assert "home_goals" not in snapshot
    assert "away_goals" not in snapshot
    assert result["evaluator_sidecar"]["home_goals"] == 2
    assert result["evaluator_sidecar"]["outcome"] == 0
    assert result["platform_adapter"]["result_fields_passed_for_sidecar_only"] is True

