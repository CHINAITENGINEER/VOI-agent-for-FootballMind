"""Constructed controlled-reveal tests, separate from real evaluation evidence."""
from copy import deepcopy
import hashlib
import json

import pytest

from football_research.acquisition.data import ControlledEpisode, load_episodes
from football_research.acquisition.state import GROUP_FIELDS


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def episode_row():
    return {"match_id": "s:match:1", "competition_id": "s:league:1", "home_team_id": "s:team:1",
            "away_team_id": "s:team:2", "kickoff": "2025-01-03T12:00:00Z",
            "prediction_cutoff": "2025-01-02T12:00:00Z", "information_mode": "historical_reconstruction",
            "features_available_at": "2025-01-02T11:00:00Z", "feature_version": "fixture-v1",
            "home_goals": 1, "away_goals": 0, "input_manifest": ["D:/secret-full-feature-file.json"],
            **{field: float(i) for i, field in enumerate(f for fs in GROUP_FIELDS.values() for f in fs)}}


def actions():
    return [{"action_id": "a-strength", "group_ids": ["elo", "recent_form"], "cost": {"requests": 1}, "available": True},
            {"action_id": "a-xg", "group_ids": ["xg"], "cost": {"requests": 1}, "available": True}]


def create_export(tmp_path, *, changes=None, feature_changes=None):
    row = episode_row()
    row.update(status="finished", score_period="regulation", season="2025")
    row.update(changes or {})
    keys = ("match_id", "prediction_cutoff", "information_mode", "feature_version")
    lineage = {k: row[k] for k in keys}
    lineage["history_match_ids"] = ["s:match:previous"]
    feature = {**{k: row[k] for k in keys}, "features_available_at": row["features_available_at"],
               "input_manifest": lineage, "input_manifest_hash": digest(lineage),
               "features": {f: row[f] for fs in GROUP_FIELDS.values() for f in fs}}
    feature.update(feature_changes or {})
    manifest = {"schema": "footballmind.platform.history-export.v1", "dataset_version": "constructed-v1",
                "source": "unit-test", "source_observed_at": "2026-09-20T12:00:00Z",
                "information_mode": "historical_reconstruction",
                "source_evidence": [{"observation_id": "unit-test", "sha256": "a" * 64}],
                "assumptions": {"result_delay_hours": 3, "xg_delay_hours": 24, "fixture_lead_hours": 168,
                                "kickoff_timezone_policy": "synthetic_UTC"}, "files": {}}
    for name, item in (("matches", row), ("features", feature)):
        payload = (json.dumps(item) + "\n").encode()
        (tmp_path / (name + ".jsonl")).write_bytes(payload)
        manifest["files"][name] = {"path": name + ".jsonl", "sha256": hashlib.sha256(payload).hexdigest(), "row_count": 1}
    payload = json.dumps(manifest).encode()
    (tmp_path / "manifest.json").write_bytes(payload)
    (tmp_path / "COMPLETE.json").write_text(json.dumps({"manifest_sha256": hashlib.sha256(payload).hexdigest()}))
    return tmp_path


def test_real_shape_join_is_exact_and_target_fields_remain_offline(tmp_path):
    rows, audit = load_episodes(create_export(tmp_path))
    assert len(rows) == audit["unique_matches"] == 1
    env = ControlledEpisode(rows[0], actions())
    text = json.dumps(env.visible_state())
    assert "home_goals" not in text and "input_manifest" not in text and "secret" not in text
    assert env.visible_state()["mask"] == []
    assert all(g["status"] == "not_acquired" for g in env.visible_state()["groups"].values())


@pytest.mark.parametrize("feature_changes", [
    {"features_available_at": "2025-01-02T12:00:01Z"},
    {"home_team_id": "s:team:WRONG"},
    {"match_id": "s:match:WRONG"},
    {"information_mode": "operational"},
])
def test_future_and_wrong_identity_rows_are_excluded(tmp_path, feature_changes):
    rows, audit = load_episodes(create_export(tmp_path, feature_changes=feature_changes))
    assert not rows and audit["excluded_count"] == 1


def test_unfinished_match_and_wrong_lead_are_excluded(tmp_path):
    rows, audit = load_episodes(create_export(tmp_path, changes={"status": "scheduled", "home_goals": None, "away_goals": None}))
    assert not rows and audit["exclusions"][0]["reason"] == "missing_finished_regulation_label"
    create_export(tmp_path)
    rows, audit = load_episodes(tmp_path, lead_hours=1)
    assert not rows and audit["exclusions"][0]["reason"] == "missing_exact_feature_snapshot"


def test_hidden_feature_and_label_poisoning_preserves_state_catalog_and_first_reveal():
    original, changed = episode_row(), episode_row()
    changed.update(home_goals=88, away_goals=77, input_manifest=["other hidden path"])
    for field in GROUP_FIELDS["xg"]:
        changed[field] = 999.
    left, right = ControlledEpisode(original, actions()), ControlledEpisode(changed, actions())
    assert left.visible_state() == right.visible_state()
    assert left.legal_actions() == right.legal_actions()
    assert left.reveal("a-strength") == right.reveal("a-strength")
    assert left.visible_state()["mask"] == ["elo", "recent_form"]
    assert left.performed_actions == ("a-strength",)
    assert [a["action_id"] for a in left.legal_actions()] == ["a-xg"]
    assert left.reveal("a-xg") != right.reveal("a-xg")


def test_null_group_is_unknown_until_reveal_and_partial_null_is_observed():
    row = episode_row()
    for field in GROUP_FIELDS["xg"]:
        row[field] = None
    row["home_elo"] = None
    env = ControlledEpisode(row, actions())
    assert env.visible_state()["groups"]["xg"]["status"] == "not_acquired"
    assert len(env.legal_actions()) == 2
    assert env.reveal("a-xg")["groups"]["xg"]["status"] == "source_missing"
    state = env.reveal("a-strength")
    assert state["groups"]["elo"]["status"] == "observed"
    assert state["groups"]["elo"]["values"]["home_elo"] is None
    with pytest.raises(ValueError, match="already executed"):
        env.reveal("a-xg")


def test_unavailable_action_and_future_episode_fail_before_reveal():
    catalog = actions()
    catalog[0]["available"] = False
    env = ControlledEpisode(episode_row(), catalog)
    with pytest.raises(ValueError, match="unavailable"):
        env.reveal("a-strength")
    row = episode_row()
    row["features_available_at"] = "2030-01-01T00:00:00Z"
    with pytest.raises(ValueError, match="unavailable"):
        ControlledEpisode(row, actions())


def test_visible_mutation_isolated_and_invalid_joint_reveal_is_atomic():
    row = episode_row()
    row["home_rolling_points"] = "bad numeric"
    env = ControlledEpisode(row, actions())
    before = env.visible_state()
    supplied = env.visible_state()
    supplied["groups"]["elo"]["status"] = "observed"
    assert env.visible_state() == before
    with pytest.raises(ValueError, match="numeric"):
        env.reveal("a-strength")
    assert env.visible_state() == before
    assert not env.performed_actions
