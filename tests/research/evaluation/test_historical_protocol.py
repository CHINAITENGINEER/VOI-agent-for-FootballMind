from datetime import timedelta

from football_research.evaluation.historical_protocol import (
    audit_feature_rows,
    audit_match_rows,
    audit_trajectory_rows,
    build_historical_snapshots,
    evaluator_label_sidecar,
    public_snapshot_projection,
    protocol_report,
    rolling_episode_folds,
    summarize_policy_metrics,
)


def match(match_id="m1", kickoff="2025-01-10T12:00:00Z", *, fixture="2025-01-09T12:00:00Z"):
    return {
        "match_id": match_id,
        "kickoff": kickoff,
        "fixture_available_at": fixture,
        "home_goals": 1,
        "away_goals": 0,
        "result_available_at": "2025-01-10T15:00:00Z",
        "score_period": "regulation",
        "status": "finished",
        "information_mode": "historical_reconstruction",
    }


def test_materialize_snapshots_and_keep_match_group_together():
    rows = [match(f"m{i}", f"2025-01-{10 + i:02d}T12:00:00Z") for i in range(8)]
    snapshots, exclusions = build_historical_snapshots(rows, lead_hours=(24, 1))
    assert not exclusions
    assert len(snapshots) == 16
    folds = rolling_episode_folds(
        snapshots,
        initial_validation_start="2025-01-12T00:00:00Z",
        validation_window=timedelta(days=2),
        test_window=timedelta(days=2),
        max_folds=1,
    )
    assert len(folds) == 1
    fold = folds[0]
    assert set(fold.train_match_ids).isdisjoint(fold.validation_match_ids)
    assert set(fold.validation_match_ids).isdisjoint(fold.test_match_ids)
    assert fold.label_cutoff.isoformat() == "2025-01-11T12:00:00+00:00"


def test_long_lead_uses_earliest_evaluation_cutoff_as_training_guard():
    rows = [match(f"m{i}", f"2025-01-{10 + i:02d}T12:00:00Z") for i in range(10)]
    snapshots, _ = build_historical_snapshots(rows, lead_hours=(100.0,))
    folds = rolling_episode_folds(
        snapshots,
        initial_validation_start="2025-01-12T00:00:00Z",
        validation_window=timedelta(days=2),
        test_window=timedelta(days=2),
        max_folds=1,
    )
    assert folds[0].label_cutoff < folds[0].validation_start


def test_snapshot_label_sidecar_is_separate_from_public_projection():
    row = match()
    row["input_manifest"] = ["observation:x|at:2026-09-20T00:00:00Z"]
    row.update({"prediction_cutoff": "2025-01-10T11:00:00Z", "lead_hours": 1.0})
    public = public_snapshot_projection(row)
    sidecar = evaluator_label_sidecar(row)
    assert "home_goals" not in public and "result_available_at" not in public
    assert "input_manifest" not in public
    assert sidecar["home_goals"] == 1
    assert sidecar["away_goals"] == 0


def test_match_audit_rejects_fixture_after_cutoff():
    row = match(fixture="2025-01-10T11:30:00Z")
    row["prediction_cutoff"] = "2025-01-10T11:00:00Z"
    issues = audit_match_rows([row])
    assert any(issue.code == "FIXTURE_AFTER_CUTOFF" for issue in issues)


def test_feature_audit_rejects_future_feature():
    issues = audit_feature_rows([{
        "match_id": "m1",
        "kickoff": "2025-01-10T12:00:00Z",
        "prediction_cutoff": "2025-01-10T11:00:00Z",
        "features_available_at": "2025-01-10T11:01:00Z",
        "feature_version": "v1",
        "input_manifest_hash": "sha256:x",
    }])
    assert any(issue.code == "FEATURE_AFTER_CUTOFF" for issue in issues)


def test_report_rejects_feature_without_audited_match_cutoff():
    report = protocol_report(
        match_rows=[{
            **match(), "prediction_cutoff": "2025-01-10T11:00:00Z",
        }],
        feature_rows=[{
            "match_id": "other", "kickoff": "2025-01-10T12:00:00Z",
            "prediction_cutoff": "2025-01-10T11:00:00Z",
            "features_available_at": "2025-01-10T10:00:00Z",
            "feature_version": "v1", "input_manifest_hash": "sha256:x",
        }],
    )
    assert report["status"] == "FAIL"
    assert report["issue_counts"]["FEATURE_SNAPSHOT_UNBOUND"] == 1


def trajectory(*, hidden=False, late=False):
    cutoff = "2025-01-10T11:00:00Z"
    state = {
        "match_id": "m1",
        "prediction_cutoff": cutoff,
        "information_mode": "historical_reconstruction",
        "base": {"kickoff": "2025-01-10T12:00:00Z"},
        "groups": {},
    }
    if hidden:
        state["outcome"] = "home_win"
    if late:
        state["groups"]["lineup"] = {
            "status": "observed",
            "available_at": "2025-01-10T11:01:00Z",
            "evidence_refs": ["sha256:x"],
        }
    event = {
        "step": 0,
        "decision": {"decision": "stop", "reason": "test"},
        "prediction": {
            "prediction_cutoff": cutoff,
            "information_mode": "historical_reconstruction",
            "probabilities": [0.5, 0.2, 0.3],
            "visible_lineage": {},
        },
        "visible_state": state,
    }
    return {"match_id": "m1", "strategy": "voi_gate", "events": [event]}


def test_trajectory_audit_catches_hidden_and_late_public_facts():
    hidden = audit_trajectory_rows([trajectory(hidden=True)])
    late = audit_trajectory_rows([trajectory(late=True)])
    assert any(issue.code == "HIDDEN_FIELD_IN_PUBLIC_STATE" for issue in hidden)
    assert any(issue.code == "EVIDENCE_AFTER_CUTOFF" for issue in late)


def test_report_and_policy_summary_are_explicitly_descriptive():
    report = protocol_report(trajectory_rows=[trajectory()])
    assert report["status"] == "PASS"
    summary = summarize_policy_metrics([{
        "strategy": "voi_gate", "match_id": "m1", "brier_score": 0.2,
        "log_loss": 0.4, "budget_money": 0.01, "actual_external_requests": 1,
    }])
    assert summary["descriptive_only"] is True
    assert summary["policies"]["voi_gate"]["mean_brier_score"] == 0.2
