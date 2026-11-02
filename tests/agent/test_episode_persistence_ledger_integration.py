from football_agent.action_value_ledger import build_action_value_ledger
from football_agent.episode_persistence import EpisodePersistence


def _row():
    return {
        "episode_id": "episode-ledger", "target_key": "match-ledger", "sequence": 0,
        "action_id": "lineup", "candidate_id": "lineup",
        "selection_operation_id": "selection-0", "selection_digest": "c" * 64,
        "acquisition_id": "acquisition-0", "acquisition_digest": "a" * 64,
        "admission_status": "admitted", "admission_digest": "e" * 64,
        "state_before_digest": "0" * 64, "state_after_digest": "1" * 64,
        "prediction_before": None, "prediction_after": None,
        "prediction_cutoff": "2026-10-01T12:00:00Z",
    }


def test_legacy_ledger_discovers_optional_episode_snapshot(tmp_path, monkeypatch):
    EpisodePersistence(tmp_path).append(_row())
    base = {"schema_version": "action-value-ledger-v1", "state_digest": "state",
            "entries": [], "ledger_digest": "legacy"}

    import football_agent.acquisition_replay as replay
    monkeypatch.setattr(replay, "verify_run", lambda *args, **kwargs:
                        {"action_value_ledger": dict(base)})
    ledger = build_action_value_ledger(tmp_path)
    assert ledger["episode_persistence_status"] == "PARTIAL"
    assert ledger["episode_persistence_eligible_for_replay"] is False
    assert ledger["episode_persistence"]["records"][0]["prediction_before"] is None
    assert ledger.get("numeric_voi_calibrated", False) is False
    assert ledger["ledger_digest"] != "legacy"

