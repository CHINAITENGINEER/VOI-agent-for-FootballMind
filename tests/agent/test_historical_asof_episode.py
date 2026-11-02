from dataclasses import dataclass

import pytest

from football_agent.historical_asof_episode import (
    UNKNOWN_MENU_ID,
    build_historical_asof_episode,
    build_historical_asof_episodes,
    validate_historical_asof_envelope,
)


def row(**overrides):
    value = {
        "match_id": "m-001",
        "kickoff": "2026-09-30T18:00:00Z",
        "prediction_cutoff": "2026-09-30T14:00:00Z",
        "competition_id": "league-1",
        "competition_name": "Synthetic League",
        "home_team_id": "home-1",
        "away_team_id": "away-1",
        "home_team_name": "Home FC",
        "away_team_name": "Away FC",
        "source": "spftll",
        "source_observations": [
            {
                "source_id": "table-v1",
                "kind": "standings",
                "observed_at": "2026-09-30T12:00:00Z",
                "available_at": "2026-09-30T12:05:00Z",
                "content_digest": "a" * 64,
                "values": {"home_rank": 2, "away_rank": 7, "score": "future-field-dropped"},
            },
            {
                "source_id": "lineup-future",
                "kind": "lineup",
                "observed_at": "2026-09-30T15:30:00Z",
                "available_at": "2026-09-30T15:30:00Z",
                "values": {"home_confirmed": True},
            },
        ],
        "candidate_menu": [
            {"candidate_id": "standings", "kind": "competition_context",
             "source_ids": ["table-v1"], "declared_cost": {"requests": 1}},
            {"candidate_id": "lineup", "kind": "lineup",
             "source_ids": ["lineup-future"], "declared_cost": {"requests": 1}},
        ],
        "candidate_generation_complete": True,
        "known": {"baseline": "balanced", "outcome": "must-not-leak"},
        "home_goals": 2,
        "away_goals": 1,
        "status": "finished",
        "score_period": "regulation",
        "result_available_at": "2026-09-30T20:00:00Z",
        "label_status": "INDEPENDENT_EVALUATOR_GOLD",
    }
    value.update(overrides)
    return value


def test_only_available_before_cutoff_enters_public_snapshot_and_result_is_sidecar():
    envelope = build_historical_asof_episode(row())
    episode = envelope["episode"]
    snapshot = episode["pre_action"]["snapshot"]
    source_ids = [item["source_id"] for item in snapshot["source_observations"]]
    assert source_ids == ["table-v1"]
    assert "outcome" not in snapshot
    assert "home_goals" not in snapshot
    assert "away_goals" not in snapshot
    assert "score" not in snapshot["source_observations"][0]["values"]
    assert envelope["evaluator_sidecar"]["home_goals"] == 2
    assert envelope["evaluator_sidecar"]["outcome"] == 0
    assert validate_historical_asof_envelope(envelope)["valid"] is True
    assert envelope["audit"]["excluded_source_count"] == 1
    assert envelope["episode"]["pre_action"]["candidate_generation_complete"] is True


def test_missing_availability_and_missing_menu_are_explicit_unknowns():
    source = {"source_id": "no-clock", "kind": "injury", "values": {"player_count": 1}}
    envelope = build_historical_asof_episode(row(source_observations=[source], candidate_menu=None))
    assert envelope["episode"]["pre_action"]["candidate_ids"] == [UNKNOWN_MENU_ID]
    assert envelope["episode"]["pre_action"]["candidate_generation_complete"] is False
    assert "candidate_menu" in envelope["audit"]["unknown_fields"]
    assert "no-clock" in envelope["audit"]["unknown_fields"]
    assert envelope["audit"]["included_source_count"] == 0


def test_candidate_source_not_visible_at_cutoff_is_unknown_not_available():
    envelope = build_historical_asof_episode(row())
    menu = envelope["episode"]["pre_action"]["candidate_ids"]
    assert menu == ["standings", "lineup"]
    assert any(item.startswith("lineup:source:lineup-future")
               for item in envelope["audit"]["unknown_fields"])


def test_observed_at_after_cutoff_is_filtered_even_if_availability_clock_is_earlier():
    source = {
        "source_id": "inconsistent-clock", "kind": "odds",
        "observed_at": "2026-09-30T15:00:00Z",
        "available_at": "2026-09-30T13:00:00Z",
        "values": {"home_price": 2.0},
    }
    envelope = build_historical_asof_episode(row(source_observations=[source]))
    assert envelope["audit"]["included_source_count"] == 0
    assert envelope["audit"]["excluded_sources"][0]["reason"] == "observed_after_cutoff"


def test_nested_result_clocks_and_statuses_never_reach_public_snapshot():
    source = {
        "source_id": "nested-leak", "kind": "news",
        "available_at": "2026-09-30T13:00:00Z",
        "observed_at": "2026-09-30T13:00:00Z",
        "status": "finished",
        "values": {
            "status": "finished",
            "score_period": "regulation",
            "label_status": "INDEPENDENT_EVALUATOR_GOLD",
            "label_available_at": "2026-09-30T20:00:00Z",
            "settlement_status": "settled",
            "safe_claim": "lineup uncertainty",
        },
    }
    envelope = build_historical_asof_episode(row(
        source_observations=[source],
        known={
            "status": "finished",
            "result_available_at": "2026-09-30T20:00:00Z",
            "safe_claim": "lineup uncertainty",
        },
    ))
    public = envelope["episode"]["pre_action"]["snapshot"]
    text = str(public)
    for forbidden in ("status", "score_period", "label_status", "label_available_at",
                      "result_available_at", "settlement_status"):
        assert forbidden not in text
    assert public["source_observations"][0]["quality"] == "unknown"
    assert public["source_observations"][0]["values"] == {"safe_claim": "lineup uncertainty"}


def test_validator_detects_tampered_observed_clock_after_cutoff():
    envelope = build_historical_asof_episode(row())
    # The builder already removed the future source; append a tampered source
    # and refresh only the envelope shape to exercise the independent validator.
    snapshot = envelope["episode"]["pre_action"]["snapshot"]
    snapshot["source_observations"].append({
        "source_id": "tampered", "available_at": "2026-09-30T13:00:00Z",
        "observed_at": "2026-09-30T15:00:00Z",
    })
    report = validate_historical_asof_envelope(envelope)
    assert "source_observed_after_cutoff:tampered" in report["reasons"]


def test_cutoff_after_kickoff_is_rejected():
    with pytest.raises(ValueError, match="must not follow kickoff"):
        build_historical_asof_episode(row(prediction_cutoff="2026-09-30T18:00:01Z"))


def test_batch_keeps_exclusions_without_fabricating_rows():
    result = build_historical_asof_episodes([
        row(),
        row(match_id="bad", kickoff=None),
        row(match_id="cancelled", status="cancelled"),
    ])
    assert result["schema_version"].startswith("historical-asof")
    assert result["episode_count"] == 1
    assert len(result["exclusions"]) == 2
    assert all(item["reason"] for item in result["exclusions"])


@dataclass
class MatchRecordFixture:
    match_id: str = "m-dataclass"
    kickoff: str = "2026-09-30T18:00:00Z"
    prediction_cutoff: str = "2026-09-30T14:00:00Z"
    home_team_id: str = "h"
    away_team_id: str = "a"
    source_observations: list = None
    candidate_menu: list = None

    def __post_init__(self):
        self.source_observations = self.source_observations or [{
            "source_id": "source", "observed_at": "2026-09-30T13:00:00Z",
            "values": {"rank": 1},
        }]
        self.candidate_menu = self.candidate_menu or [{
            "candidate_id": "rank", "source_ids": ["source"],
            "declared_cost": {"requests": 1},
        }]


def test_dataclass_rows_are_supported_without_provider_access():
    envelope = build_historical_asof_episode(MatchRecordFixture(), candidate_catalog_complete=True)
    assert envelope["episode"]["target_key"] == "match:m-dataclass"
    assert envelope["episode"]["pre_action"]["candidate_ids"] == ["rank"]
    assert envelope["audit"]["real_source_requests"] == 0
    # An observation clock is not a system-availability clock in strict mode.
    assert envelope["audit"]["included_source_count"] == 0
    assert "source" in envelope["audit"]["unknown_fields"]


def test_observed_at_fallback_requires_explicit_opt_in():
    strict = build_historical_asof_episode(MatchRecordFixture(), candidate_catalog_complete=True)
    legacy = build_historical_asof_episode(
        MatchRecordFixture(), candidate_catalog_complete=True,
        allow_observed_at_fallback=True,
    )
    assert strict["audit"]["included_source_count"] == 0
    assert legacy["audit"]["included_source_count"] == 1
    assert legacy["episode"]["pre_action"]["snapshot"]["source_observations"][0]["availability_basis"] == "observed_at"
    assert legacy["audit"]["allow_observed_at_fallback"] is True


def test_unknown_label_is_preserved_as_unavailable_sidecar():
    envelope = build_historical_asof_episode(row(home_goals=None, away_goals=None,
                                                  label_status="UNAVAILABLE"))
    sidecar = envelope["evaluator_sidecar"]
    assert sidecar["label_status"] == "UNAVAILABLE"
    assert "home_goals" not in sidecar
    assert "outcome" not in sidecar
