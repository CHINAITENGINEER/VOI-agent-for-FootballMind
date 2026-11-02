"""Versioned read-model contract tests for the FootballMind browser."""

from copy import deepcopy

from demo.server import DemoApplication
from demo.ui_projection import RUN_VIEW_SCHEMA, build_matches_view, build_run_view


def prediction(home=0.45, draw=0.28, away=0.27, *, model="market-v1", cutoff="2026-09-20T11:10:00Z", info="operational"):
    return {"match_id": "m-1", "prediction_cutoff": cutoff,
            "home_win_probability": home, "draw_probability": draw,
            "away_win_probability": away, "model_version": model,
            "dataset_version": "dataset-v1", "information_mode": info,
            "input_manifest": {"feature_input": [{"evidence_id": "e-1"}]}}


def base_run(**overrides):
    run = {"schema_version": "agent-run-v2", "run_id": "run-1", "match_id": "m-1",
           "mode": "real", "test_mode": False, "information_mode": "operational",
           "time_mode": "archive_replay", "prediction_cutoff": "2026-09-20T11:10:00Z",
           "state": "PREDICTED", "reason": "MODEL_RETURNED_PROBABILITIES", "match": None,
           "evidence": [], "features": None, "prediction": None, "last_prediction": None,
           "events": [], "quality": [], "budget": {}, "policy_version": "fixed-order-v1",
           "policy_configuration": {"topics": ["odds", "lineups"]}, "tool_configuration": {},
           "queried_topics": []}
    run.update(overrides)
    return run


def test_initial_prediction_has_no_after_or_delta():
    first = prediction()
    run = base_run(prediction=first, events=[{"type": "prediction_updated", "data": {
        "before": None, "after": first, "update_reason": "INITIAL_PREDICTION"}}])
    view = build_run_view(run)
    assert view["schema_version"] == RUN_VIEW_SCHEMA
    assert view["prediction"]["before"] == first
    assert view["prediction"]["after"] is None
    assert view["prediction"]["delta"] is None
    assert view["prediction"]["change_status"] == "not_applicable"


def test_evidence_saved_without_consumption_cannot_claim_after():
    first = prediction()
    second = prediction(0.47, 0.28, 0.25)
    run = base_run(
        evidence=[{"evidence_id": "e-1", "source": "archive", "kind": "odds",
                   "observed_at": "2026-09-20T10:00:00Z", "available_at": "2026-09-20T10:00:00Z",
                   "content_hash": "hash", "content": {}, "raw_ref": "archive://e-1"}],
        prediction=second, last_prediction=first,
        events=[{"type": "prediction_updated", "data": {"before": None, "after": first, "update_reason": "INITIAL_PREDICTION"}},
                {"type": "prediction_updated", "data": {"before": first, "after": second, "update_reason": "NEW_VALID_EVIDENCE"}}])
    view = build_run_view(run)
    assert view["prediction"]["after"] is None
    assert view["f"]["status"] == "not_run"


def test_f_consumed_zero_change_is_executed_no_change():
    first = prediction()
    second = deepcopy(first)
    run = base_run(
        evidence=[{"evidence_id": "e-1", "source": "archive", "kind": "odds",
                   "observed_at": "2026-09-20T10:00:00Z", "available_at": "2026-09-20T10:00:00Z",
                   "content_hash": "hash", "content": {}, "raw_ref": "archive://e-1"}],
        prediction=second, last_prediction=first,
        features={"feature_version": "f-v1", "feature_snapshot_hash": "snap",
                  "input_manifest": [{"evidence_id": "e-1"}]},
        events=[{"type": "prediction_updated", "data": {"before": None, "after": first, "update_reason": "INITIAL_PREDICTION"}},
                {"type": "features_ready", "snapshot": {"features": {"feature_version": "f-v1", "input_manifest": [{"evidence_id": "e-1"}]}}, "data": {}},
                {"type": "prediction_updated", "data": {"before": first, "after": second, "update_reason": "NEW_VALID_EVIDENCE"},
                 "snapshot": {"features": {"feature_version": "f-v1", "input_manifest": [{"evidence_id": "e-1"}]}}}])
    view = build_run_view(run)
    assert view["prediction"]["after"] == second
    assert view["prediction"]["delta"]["home"] == 0
    assert view["prediction"]["change_status"] == "unchanged_after_f"
    assert view["f"]["status"] == "executed_no_change"
    assert view["f"]["consumed_evidence_ids"] == ["e-1"]


def test_g_unknown_and_no_default_candidate_topics():
    view = build_run_view(base_run())
    assert view["g"]["status"] == "unknown"
    assert view["g"]["candidate_availability"] == "not_returned"
    assert view["g"]["candidate_actions"] == []


def test_synthetic_prior_has_no_selection_authority():
    run = base_run(mode="test", test_mode=True, information_mode="synthetic_test")
    view = build_run_view(run)
    assert view["g"]["status"] == "synthetic_diagnostic_only"
    assert view["g"]["selection_authority"] == "none"


def test_archive_replay_and_demo_markers_are_explicit():
    archive = build_run_view(base_run())
    assert archive["time_mode"] == "archive_replay"
    assert archive["is_live"] is False
    assert archive["source_label"] == "真实档案回放 / 本地逐批披露"
    demo = build_run_view(base_run(mode="demo", is_demo=True))
    assert demo["is_demo"] is True
    assert demo["source_label"] == "演示数据 · 接口无可用比赛"


def test_matches_projection_has_metadata_and_nullable_eligibility():
    result = build_matches_view({"status": "OK", "data": [{"match_id": "m-1", "home": "A", "away": "B"}]}, cutoff="2026-09-20T11:10:00Z")
    assert result["schema_version"] == "footballmind-ui-matches-v1"
    assert result["meta"]["is_live"] is False
    assert result["meta"]["total"] == 1
    assert result["data"][0]["eligibility"] == {"prematch_at_cutoff": None,
        "fixture_available_at_cutoff": None, "identity_verified": None,
        "odds_available_at_cutoff": None, "run_allowed": None, "reason": None}


def test_legacy_run_endpoint_remains_available(tmp_path):
    app = DemoApplication(tmp_path)
    record = app.post("/api/runs", {"mode": "test", "scenario": "normal", "match_id": "test:match-001",
                                    "prediction_cutoff": "2026-09-20T11:30:00Z"})
    legacy = app.get(f"/api/runs/{record['run_id']}", {})
    view = app.get(f"/api/runs/{record['run_id']}/view", {})
    assert legacy["run_id"] == record["run_id"]
    assert view["schema_version"] == RUN_VIEW_SCHEMA

