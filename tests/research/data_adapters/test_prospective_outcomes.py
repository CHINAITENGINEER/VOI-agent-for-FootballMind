from datetime import datetime, timezone
import json

from football_research.data_adapters.prospective_outcomes import derive_g_candidates, project_result_rows
from football_spider.storage.raw_archive import RawArchive


def test_project_verified_settlement_copies_declared_raw_observation(tmp_path):
    archive = RawArchive(tmp_path / "result-archive")
    observed = datetime(2026, 10, 12, 21, tzinfo=timezone.utc)
    obs = archive.save("official", "https://example.test/result/1", {}, b'{"finished":true}', status=200, observed_at=observed)
    rows, audit = project_result_rows([{
        "match_id": "source-1", "home_goals": 2, "away_goals": 1,
        "status": "final", "observed_at": observed.isoformat(),
        "score_period": "regulation", "independent_outcome_verified": True,
        "kickoff": "2026-10-12T19:00:00+00:00",
        "source": "https://example.test/result/1",
        "input_manifest": [f"observation:{obs.id}|sha256:{obs.content_hash}|at:{observed.isoformat()}"],
    }], physical_match_ids={"source-1": "physical-1"}, result_archive=archive.root, output_root=tmp_path / "package")
    assert audit["accepted"] == 1 and not audit["rejected"]
    assert rows[0]["physical_match_id"] == "physical-1"
    assert (tmp_path / "package" / rows[0]["raw_path"]).is_file()
    assert rows[0]["result"] == "home"


def test_derived_historical_without_prediction_is_kept_ineligible():
    attempt = {"attempt_id": "a1", "physical_match_id": "m1", "discovery_id": "d1",
               "action_id": "news", "selected_by_policy": True, "decision_at": "2026-10-10T10:00:00+00:00",
               "selection_probability": None, "raw_path": "raw/a.body.gz", "raw_sha256": "a" * 64}
    rows = derive_g_candidates(
        fixtures=[{"physical_match_id": "m1", "information_mode": "historical_reconstruction"}],
        discoveries=[{"discovery_id": "d1", "original_agent_menu_proven": False}],
        attempts=[attempt], contexts=[],
        results=[{"physical_match_id": "m1", "regulation_home_goals": 1, "regulation_away_goals": 0, "result": "home"}],
    )
    assert len(rows) == 1
    assert rows[0]["label_status"] == "ineligible"
    assert "historical_reconstruction" in rows[0]["exclusion_reason"]
    assert rows[0]["realized_gain"] is None


def test_project_result_rejects_corrupted_rawarchive_body(tmp_path):
    archive = RawArchive(tmp_path / "result-archive")
    observed = datetime(2026, 10, 12, 21, tzinfo=timezone.utc)
    obs = archive.save("official", "https://example.test/result/1", {}, b'{"finished":true}', status=200, observed_at=observed)
    body_path = archive.root / obs.payload_path
    body_path.write_bytes(b"corrupt")
    rows, audit = project_result_rows([{
        "match_id": "source-1", "home_goals": 2, "away_goals": 1,
        "status": "final", "score_period": "regulation", "independent_outcome_verified": True,
        "kickoff": "2026-10-12T19:00:00+00:00", "observed_at": observed.isoformat(),
        "input_manifest": [f"observation:{obs.id}|sha256:{obs.content_hash}|at:{observed.isoformat()}"],
    }], result_archive=archive.root, output_root=tmp_path / "package")
    assert rows == []
    assert audit["rejected"][0]["reason"] == "result_raw_digest_mismatch"


def test_project_accepts_verified_settlement_gateway_marker(tmp_path):
    archive = RawArchive(tmp_path / "result-archive")
    observed = datetime(2026, 10, 12, 21, tzinfo=timezone.utc)
    obs = archive.save("official", "https://example.test/result/1", {}, b'{"finished":true}', status=200, observed_at=observed)
    rows, audit = project_result_rows([{
        "match_id": "source-1", "home_goals": 2, "away_goals": 1,
        "status": "final", "score_period": "regulation", "information_mode": "operational",
        "verification_schema": "verified-settlement-v1", "identity_binding_sha256": "a" * 64,
        "regulation_rule_version": "regulation-v1", "kickoff": "2026-10-12T19:00:00+00:00",
        "observed_at": observed.isoformat(),
        "input_manifest": [f"observation:{obs.id}|sha256:{obs.content_hash}|at:{observed.isoformat()}"],
    }], result_archive=archive.root)
    assert audit["accepted"] == 1 and rows[0]["information_mode"] == "post_match_outcome"


def test_derived_prospective_requires_verified_bytes_and_strict_clocks(tmp_path):
    archive = RawArchive(tmp_path / "stream")
    kickoff = "2026-10-10T12:00:00+00:00"
    def raw(label):
        return archive.save("fixture", f"https://example.test/{label}", {}, label.encode(), status=200,
                            observed_at=datetime(2026, 10, 9, 8, tzinfo=timezone.utc))
    fixture_obs, menu_obs, action_obs, result_obs = [raw(x) for x in ("fixture", "menu", "action", "result")]
    fixture = {"physical_match_id": "m1", "information_mode": "prospective_capture", "kickoff_at": kickoff,
               "registered_at": "2026-10-09T07:00:00+00:00", "arm_assigned_at": "2026-10-09T07:01:00+00:00",
               "pre_kickoff_frozen": True, "policy_arm": "fixed_order", "policy_version": "v1",
               "raw_path": fixture_obs.payload_path, "raw_sha256": fixture_obs.content_hash}
    menu = {"discovery_id": "d1", "physical_match_id": "m1", "information_mode": "prospective_capture",
            "observed_at": "2026-10-09T08:00:00+00:00", "original_agent_menu_proven": True,
            "policy_version": "v1", "raw_path": menu_obs.payload_path, "raw_sha256": menu_obs.content_hash,
            "visible_candidates": [{"candidate_id": "c1", "action_id": "news"}]}
    attempt = {"attempt_id": "a1", "physical_match_id": "m1", "discovery_id": "d1", "candidate_id": "c1",
               "action_id": "news", "selected_by_policy": True, "information_mode": "prospective_capture",
               "policy_arm": "fixed_order", "policy_version": "v1", "selection_basis": "fixed_order",
               "selection_probability": 1.0, "selection_probability_basis": "fixed_order_next_action",
               "decision_at": "2026-10-09T09:00:00+00:00", "requested_at": "2026-10-09T09:00:01+00:00",
               "request_started_at": "2026-10-09T09:00:01+00:00", "response_observed_at": "2026-10-09T09:00:02+00:00",
               "status": "returned", "raw_path": action_obs.payload_path, "raw_sha256": action_obs.content_hash,
               "request_count": 1, "acquisition_cost": 0.1}
    before = {"physical_match_id": "m1", "parent_attempt_id": "a1", "snapshot_kind": "before_action",
              "observed_at": "2026-10-09T08:59:00+00:00", "captured_at": "2026-10-09T08:59:01+00:00",
              "available_at": "2026-10-09T08:59:00+00:00", "prediction_cutoff": "2026-10-09T08:59:00+00:00",
              "availability_basis": "agent_state", "availability_status": "known", "information_mode": "prospective_capture",
              "prediction_before": {"home": .5, "draw": .3, "away": .2}}
    after = {"physical_match_id": "m1", "parent_attempt_id": "a1", "snapshot_kind": "after_action",
             "observed_at": "2026-10-09T09:00:03+00:00", "captured_at": "2026-10-09T09:00:04+00:00",
             "available_at": "2026-10-09T09:00:03+00:00", "prediction_cutoff": "2026-10-09T09:00:03+00:00",
             "availability_basis": "response_observed", "availability_status": "known", "information_mode": "prospective_capture",
             "prediction_after": {"home": .55, "draw": .25, "away": .2}}
    result = {"physical_match_id": "m1", "kickoff_at": kickoff, "regulation_home_goals": 1,
              "regulation_away_goals": 0, "result": "home", "result_status": "final", "score_period": "regulation",
              "result_source_url": "https://example.test/result", "result_observed_at": "2026-10-10T15:00:00+00:00",
              "information_mode": "post_match_outcome", "independent_outcome_verified": True,
              "raw_path": result_obs.payload_path, "raw_sha256": result_obs.content_hash}
    rows = derive_g_candidates(fixtures=[fixture], discoveries=[menu], attempts=[attempt], contexts=[before, after],
                               results=[result], archive_root=archive.root)
    assert rows[0]["label_status"] == "eligible", rows[0]
