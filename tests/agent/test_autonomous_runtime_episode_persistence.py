from football_agent.autonomous_runtime_loop import run_information_loop
from football_agent.episode_persistence import EpisodePersistence

from test_autonomous_runtime_budget import Provider, setup_run


def test_runtime_loop_optionally_persists_selection_boundaries_without_claiming_predictions(tmp_path):
    args = setup_run(budget=1)
    args.pop("max_steps", None)
    store = EpisodePersistence(tmp_path)
    result = run_information_loop(**args, execute_fn=Provider(), max_steps=2,
                                  episode_store=store, episode_id="runtime-episode")
    rows = store.records()
    assert rows
    assert len(rows) == result["steps"]
    assert rows[0]["episode_id"] == "runtime-episode"
    assert rows[0]["selected_at"] == "2026-10-01T10:00:00Z"
    assert rows[0]["prediction_before"] is None
    assert rows[0]["prediction_after"] is None
    assert store.validate()["status"] == "PARTIAL"
    assert store.validate()["eligible_for_replay"] is False


def test_runtime_loop_legacy_call_does_not_create_episode_files(tmp_path):
    args = setup_run(budget=1)
    args.pop("max_steps", None)
    run_information_loop(**args, execute_fn=Provider(), max_steps=1)
    assert not list(tmp_path.iterdir())

