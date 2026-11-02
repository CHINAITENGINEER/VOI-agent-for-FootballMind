"""Round172 acceptance tests for the first real historical VoI slice."""
from __future__ import annotations

import json
from pathlib import Path

from football_agent.round163_historical_episode_bridge import (
    public_evaluator_fields,
    validate_round163_episode,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "var/parallel/agent/round172"


def _load(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def test_round172_binds_independent_reconstructed_result_without_public_leak():
    episode = _load("episode.json")
    report = validate_round163_episode(episode)
    assert report["valid"] is True
    assert report["independent_gold"] is False
    assert report["reconstructed_result"] is True
    assert episode["evaluation_sidecar"]["independent_result_bound"] is True
    assert episode["evaluation_sidecar"]["outcome"] == 1
    assert public_evaluator_fields(episode["public"]) == []
    encoded = json.dumps(episode["public"], ensure_ascii=False).casefold()
    assert "home_goals" not in encoded
    assert "away_goals" not in encoded
    assert "outcome" not in encoded


def test_round172_voi_trace_selects_one_candidate_but_is_not_training_data():
    trace = _load("voi-trace.json")
    assert trace["selection"]["decision"] == "acquire"
    assert trace["selection"]["candidate_id"] == "lineup_context"
    assert trace["candidate_source_overlap_with_actor"] == []
    assert trace["response_model_scope"] == "development_fixture"
    assert trace["numeric_voi_training_eligible"] is False
    assert trace["hidden_result_included"] is False


def test_round172_candidate_menu_contains_only_unobserved_next_actions():
    public = _load("public.json")
    actor_sources = set(public["historical_provenance"]["actor_source_ids"])
    menu = public["candidate_menu"]
    assert [item["candidate_id"] for item in menu] == [
        "lineup_context", "qualification_context", "tactical_context",
    ]
    assert all(item["action_kind"] == "query_unobserved" for item in menu)
    assert actor_sources.isdisjoint({item["source_id"] for item in menu})
    assert all(item["candidate_id"] not in {"fixture_context", "odds_page_1", "odds_page_2"} for item in menu)


def test_round172_report_keeps_replay_and_performance_claims_separate():
    report = _load("report.json")
    source = report["independent_result_source"]
    assert report["independent_result_label_status"] == "INDEPENDENT_GOLD_RECONSTRUCTED"
    assert report["counterfactual_replay_ready"] is False
    assert report["performance_claim"] is False
    assert report["numeric_voi_training_eligible"] is False
    assert report["candidate_menu_scope"] == "unobserved_next_actions"
    assert report["candidate_source_overlap_with_actor"] == []
    assert source["source_url"].startswith("https://www.scottishfa.co.uk/")
    assert len(source["content_sha256"]) == 64


def test_round173_corrections_preserve_original_clocks_and_align_costs():
    original = json.loads((ROOT / "var/parallel/agent/round163/historical-envelope.json").read_text(encoding="utf-8"))
    public, episode, trace, source = _load("public.json"), _load("episode.json"), _load("voi-trace.json"), _load("report.json")["independent_result_source"]
    assert public["public_snapshot_digest"] == original["episode"]["pre_action"]["snapshot_digest"]
    assert all(item["availability_basis"] == "observed_at" for item in public["match_snapshot"]["source_observations"])
    from football_research.baselines.api import utc
    assert utc(episode["evaluation_sidecar"]["result_observed_at"]) == utc(source["retrieved_at"])
    declared = {item["candidate_id"]: item["cost"] for item in public["candidate_menu"]}
    assert declared == {item["candidate_id"]: item["cost"] for item in trace["candidate_values"]}
