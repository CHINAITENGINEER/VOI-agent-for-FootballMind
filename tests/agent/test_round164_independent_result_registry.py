"""Round164 tests for independent result provenance and strict cross-source joins."""
from __future__ import annotations

from copy import deepcopy
import gzip
import json
from pathlib import Path

import pytest

from football_agent.counterfactual_policy_replay import INDEPENDENT_GOLD
from football_agent.round162_policy_replay_adapter import compare_policies
from football_agent.round164_independent_result_registry import (
    INDEPENDENT_GOLD_RECONSTRUCTED,
    build_registry,
    confirmed_match_mappings,
    digest,
    join_result_to_public,
    load_jsonl,
    reconstruct_spftll_fixture_public,
    registry_entry_from_row,
)


def _row(**changes):
    row = {
        "source": "understat",
        "source_id": "28778",
        "match_id": "result-match-28778",
        "kickoff": "2025-08-15T19:00:00Z",
        "home_team_id": "87",
        "away_team_id": "73",
        "home_team_name": "Liverpool",
        "away_team_name": "Bournemouth",
        "competition_id": "EPL-2025",
        "competition_name": "Premier League",
        "observed_at": "2026-09-20T11:59:41Z",
        "result_available_at": "2025-08-15T22:00:00Z",
        "information_mode": "historical_reconstruction",
        "home_goals": 4,
        "away_goals": 2,
        "score_period": "regulation",
        "status": "finished",
        "source_payload_digest": "d" * 64,
    }
    row.update(changes)
    return row


def _entry():
    return registry_entry_from_row(_row(), mapping_evidence_digest="e" * 64)


def _mapping():
    return {
        "mapping_status": "confirmed",
        "actor_source": "spftll",
        "actor_match_id": "2789129",
        "actor_home_team_id": "25",
        "actor_away_team_id": "348",
        "result_source": "understat",
        "result_source_id": "28778",
        "result_home_team_id": "87",
        "result_away_team_id": "73",
        "evidence_digest": "e" * 64,
    }


def _public():
    return {
        "target_key": "spftll:2789129",
        "kickoff": "2025-08-15T19:00:00Z",
        "prediction_cutoff": "2025-08-14T19:00:00Z",
        "match_snapshot": {
            "match_id": "2789129",
            "kickoff": "2025-08-15T19:00:00Z",
            "home_team_id": "25",
            "away_team_id": "348",
            "home_team_name": "利物浦",
            "away_team_name": "伯恩茅斯",
            "competition_id": "EPL-2025",
        },
    }


def _episode_for_replay():
    entry = _entry()
    public = {
        **_public(),
        "initial_context": {"qualification_state": "unknown"},
        "candidate_generation_complete": True,
        "budget": .2,
        "loss_unit": "decision_loss",
        "candidate_menu": [
            {"candidate_id": "lineup", "cost": .1, "source_id": "spftll:lineup"},
        ],
    }
    return {
        "episode_id": "round164-reconstructed",
        "public": public,
        "evaluation_sidecar": {
            "label_status": INDEPENDENT_GOLD_RECONSTRUCTED,
            "baseline_loss": .7,
            "cost_to_loss_rate": 1.0,
            "sequence_losses": {"": .7, "lineup": .4},
            "action_returns": {
                "lineup": {
                    "available": True,
                    "context_update": {"lineup_state": "known"},
                    "observed_at": "2025-08-14T18:00:00Z",
                    "available_at": "2025-08-14T18:01:00Z",
                }
            },
        },
    }


def test_historical_row_is_reconstructed_gold_and_keeps_two_clocks():
    entry = _entry()
    assert entry["label_status"] == INDEPENDENT_GOLD_RECONSTRUCTED
    assert entry["timing_basis"] == "historical_reconstructed_result_delay"
    assert entry["observed_at"] > entry["available_at"]
    assert len(entry["record_digest"]) == 64


def test_operational_row_requires_system_availability_after_observation():
    with pytest.raises(ValueError, match="follow observation"):
        registry_entry_from_row(_row(
            information_mode="operational",
            observed_at="2025-08-15T22:30:00Z",
            result_available_at="2025-08-15T22:00:00Z",
        ))
    entry = registry_entry_from_row(_row(
        information_mode="operational",
        source_published_at="2025-08-15T21:00:00Z",
        observed_at="2025-08-15T21:30:00Z",
        result_available_at="2025-08-15T22:00:00Z",
    ))
    assert entry["label_status"] == INDEPENDENT_GOLD
    assert entry["source_published_at"] < entry["observed_at"] < entry["available_at"]


def test_registry_keeps_rejections_and_counts_without_silent_drop():
    report = build_registry([_row(), _row(source_id="28779", match_id="result-match-28779", status="scheduled")], strict=False)
    assert report["stats"]["accepted_rows"] == 1
    assert report["stats"]["rejected_rows"] == 1
    assert report["rejected"][0]["row_index"] == 1
    with pytest.raises(ValueError, match="rejected 1 rows"):
        build_registry([_row(), _row(source_id="28779", match_id="result-match-28779", status="scheduled")])


def test_confirmed_mappings_require_review_basis_and_preserve_ordered_teams():
    row = {
        "basis": "both_reviewed_team_ids_same_home_away_competition_kickoff_regulation_score",
        "understat_observation_id": "obs-1",
        "evidence": {
            "spftll_id": "2789129",
            "spftll_observation_id": "obs-actor",
            "understat_id": "28778",
            "kickoff": "2025-08-15T19:00:00Z",
            "spftll_teams": [{"source_id": "25"}, {"source_id": "348"}],
            "understat_teams": [{"id": "87"}, {"id": "73"}],
        },
    }
    mapping = confirmed_match_mappings([row])[0]
    assert mapping["actor_home_team_id"] == "25"
    assert mapping["actor_away_team_id"] == "348"
    assert mapping["result_home_team_id"] == "87"
    assert mapping["result_away_team_id"] == "73"
    assert mapping["actor_observation_id"] == "obs-actor"
    assert len(mapping["evidence_digest"]) == 64


def test_join_accepts_explicit_mapping_and_keeps_result_out_of_public():
    public = _public()
    joined = join_result_to_public(
        public, _entry(), _mapping(),
        actor_source_ids=["spftll:2789129"], actor_content_digests=["a" * 64],
    )
    assert joined["accepted"] is True
    assert joined["status"] == INDEPENDENT_GOLD_RECONSTRUCTED
    assert joined["online_availability_claim"] is False
    assert joined["retrospective_result_binding"] is True
    assert joined["retrospective_utility_claim"] is False
    assert joined["counterfactual_replay_ready"] is False
    assert "home_goals" not in public["match_snapshot"]
    assert joined["independent_result"]["home_goals"] == 4


def test_join_accepts_canonical_entity_ids_when_source_ids_differ():
    entry = _entry()
    entry.update({
        "home_team_id": "canonical-liverpool",
        "away_team_id": "canonical-bournemouth",
        "match_id": "canonical-match",
        "competition_id": "canonical-competition",
    })
    entry["record_digest"] = digest({
        key: value for key, value in entry.items() if key != "record_digest"
    })
    bindings = {
        "result_home_canonical_team_id": {
            "entity_type": "team", "source": "understat", "source_id": "87",
            "status": "confirmed", "valid_to": None, "entity_id": "canonical-liverpool",
        },
        "result_away_canonical_team_id": {
            "entity_type": "team", "source": "understat", "source_id": "73",
            "status": "confirmed", "valid_to": None, "entity_id": "canonical-bournemouth",
        },
        "result_canonical_match_id": {
            "entity_type": "match", "source": "understat", "source_id": "28778",
            "status": "confirmed", "valid_to": None, "entity_id": "canonical-match",
        },
        "result_canonical_competition_id": {
            "entity_type": "competition", "source": "understat", "source_id": "EPL",
            "status": "confirmed", "valid_to": None, "entity_id": "canonical-competition",
        },
    }
    mapping = _mapping() | {
        "result_home_canonical_team_id": "canonical-liverpool",
        "result_away_canonical_team_id": "canonical-bournemouth",
        "result_canonical_match_id": "canonical-match",
        "result_canonical_competition_id": "canonical-competition",
        "entity_mapping_evidence": bindings,
    }
    mapping["entity_mapping_digest"] = digest(bindings)
    mapping["mapping_digest"] = digest(mapping)
    joined = join_result_to_public(
        _public(), entry, mapping,
        actor_source_ids=["spftll:2789129"], actor_content_digests=["a" * 64],
    )
    assert joined["accepted"] is True


def test_join_rejects_swapped_teams_and_wrong_cutoff():
    swapped = _mapping() | {"actor_home_team_id": "348", "actor_away_team_id": "25"}
    report = join_result_to_public(_public(), _entry(), swapped)
    assert report["accepted"] is False
    assert "actor_home_team_id" in report["failed_checks"]
    late_public = _public() | {"prediction_cutoff": "2025-08-15T23:00:00Z"}
    report = join_result_to_public(late_public, _entry(), _mapping())
    assert report["accepted"] is False
    assert "result_after_cutoff" in report["failed_checks"]


def test_join_rejects_source_overlap_and_unconfirmed_mapping():
    report = join_result_to_public(_public(), _entry(), _mapping(), actor_source_ids=["understat:28778"])
    assert report["accepted"] is False
    assert "source_separate" in report["failed_checks"]
    report = join_result_to_public(_public(), _entry(), _mapping() | {"mapping_status": "candidate"})
    assert report["accepted"] is False
    assert report["reason"] == "mapping_not_confirmed"


def test_reconstructed_replay_requires_explicit_opt_in():
    episode = _episode_for_replay()
    with pytest.raises(ValueError, match="allow_reconstructed"):
        compare_policies([episode])
    def intervals(state):
        return {
            cid: {"state_digest": state["state_digest"], "lower_bound": 0.0, "upper_bound": .01}
            for cid in state["remaining_candidate_ids"]
        }
    report = compare_policies([episode], interval_provider=intervals, allow_reconstructed=True)
    assert report["evaluation_scope"] == "retrospective_reconstructed_utility"
    assert report["operational_availability_claim"] is False
    assert report["reconstructed_result_labels"] == 1


def test_real_understat_archive_has_380_strict_result_rows():
    path = Path("var/parallel/collectors/round2/exports/understat-2025-historical_reconstruction-9b0135105553a7a7/matches.jsonl")
    rows = load_jsonl(path)
    report = build_registry(rows, strict=True)
    assert report["stats"]["accepted_rows"] == 380
    assert report["stats"]["rejected_rows"] == 0
    assert report["stats"]["label_counts"][INDEPENDENT_GOLD_RECONSTRUCTED] == 380
    assert report["claims"]["operational_availability_claim"] is False


def test_public_snapshot_digest_does_not_include_registry_result():
    public = _public()
    before = digest(public)
    joined = join_result_to_public(public, _entry(), _mapping())
    assert joined["accepted"] is True
    assert digest(public) == before
    assert "outcome" not in json.dumps(public, ensure_ascii=False).lower()


def test_reconstructed_spftll_projection_keeps_observation_clock_separate_from_cutoff():
    metadata_path = Path(
        "football_spider/platform_data/raw/spftll/2026-09-20/"
        "112858_455319_4f89e8b4-4ae1-4731-b1b0-7dd065dddbc6.observation.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload = gzip.decompress(
        metadata_path.with_name(
            metadata_path.name.replace(".observation.json", ".body.gz")
        ).read_bytes()
    )
    public = reconstruct_spftll_fixture_public(
        payload, metadata, match_id="2789129", prediction_lead_hours=1
    )
    observation = public["match_snapshot"]["source_observations"][0]
    assert observation["observed_at"] == "2026-09-20T11:28:58.455319Z"
    assert observation["available_at"] == observation["observed_at"]
    assert public["kickoff"] == "2025-08-15T19:00:00Z"
    assert public["prediction_cutoff"] == "2025-08-15T18:00:00Z"
    assert public["online_availability_claim"] is False
    assert "home_goals" not in json.dumps(public, ensure_ascii=False)
