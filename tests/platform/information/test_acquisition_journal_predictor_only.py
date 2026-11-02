"""Constructed f-only journal checks: no HTTP and no empirical training data."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from football_platform.information.acquisition_journal import AcquisitionDecisionJournal
from football_spider.storage.raw_archive import RawArchive


P = {"home": .5, "draw": .3, "away": .2}
ASSESSMENT = {"status": "PREDICTOR_ONLY", "prediction": P,
              "candidate_expected_gains": None, "model_versions": ["f-frozen-v1"]}


def create(tmp_path, **overrides):
    now = [datetime(2030, 1, 1, 9, tzinfo=timezone.utc)]
    journal = AcquisitionDecisionJournal(tmp_path / "journal", RawArchive(tmp_path / "raw"), clock=lambda: now[0])
    options = dict(fixture={"match_id": "constructed-f", "competition_id": "constructed-league",
        "home_team_id": "constructed-home", "away_team_id": "constructed-away",
        "kickoff": "2030-01-01T12:00:00Z"}, request_limit=1,
        policy_arm="random_explore", policy_version="f-only-policy-v1", random_seed=42,
        propensity_logging=True, initial_assessment=deepcopy(ASSESSMENT))
    options.update(overrides)
    return journal, journal.start(**options), now


def begin(journal, run, **overrides):
    args = dict(request_id="request-1", action_id="read-article", input_parameters={"endpoint": "https://example.test/a"},
        candidates=["read-article", "read-other"], assessment=deepcopy(ASSESSMENT),
        selection_basis="uniform_exploration", selection_probability=.5, policy_version="f-only-policy-v1")
    args.update(overrides)
    return journal.begin(run["run_id"], **args)


def test_zero_request_initial_and_final_record_actual_f_without_g(tmp_path):
    journal, run, now = create(tmp_path, request_limit=0)
    assert run["schema"] == "prospective-competition-journal-v2"
    assert run["assessment_schema"] == "prospective-predictor-only-assessment-v1"
    for kind in ("initial", "final"):
        snapshot = journal.record_context(run["run_id"], snapshot_kind=kind,
            observed_at=now[0], available_at=now[0], availability_basis="runtime_state_handoff",
            prediction_after=P, model_versions=["f-frozen-v1"])
        assert snapshot["prediction_after"] == P
        assert snapshot["prediction_before"] is None
        assert snapshot["candidate_expected_gains"] is None
        assert snapshot["model_status"] == "PREDICTOR_ONLY"
    final = journal.stop(run["run_id"], "no_feasible_candidate")
    assert final["budget"]["charged_requests"] == 0
    assert final["candidate_expected_gains"] is None
    assert journal.store.verify(run["run_id"])["events_verified"] == 4


@pytest.mark.parametrize("value", [{}, {"read-article": .2}])
def test_f_only_never_accepts_even_empty_fake_gain_map(tmp_path, value):
    with pytest.raises(ValueError, match="cannot supply expected gains"):
        create(tmp_path, initial_assessment={**ASSESSMENT, "candidate_expected_gains": value})


@pytest.mark.parametrize("change", [
    {"prediction": {"home": float("nan"), "draw": .3, "away": .2}},
    {"prediction": {"home": .4, "draw": .3, "away": .2}},
    {"model_versions": []}, {"model_versions": "f-frozen-v1"},
])
def test_initial_f_assessment_is_validated_before_run(tmp_path, change):
    with pytest.raises(ValueError):
        create(tmp_path, initial_assessment={**ASSESSMENT, **change})


def test_f_only_requires_nonnumeric_versioned_policy_at_start(tmp_path):
    with pytest.raises(ValueError, match="nonnumeric"):
        create(tmp_path, policy_arm="voi")


@pytest.mark.parametrize("change,match", [
    ({"assessment": None}, "assessment mode"),
    ({"assessment": {**ASSESSMENT, "model_versions": ["f-other"]}}, "model versions"),
    ({"policy_version": "policy-other"}, "policy version"),
    ({"selection_basis": "configured_numeric_policy"}, "numeric policy"),
    ({"selection_basis": "manual_predeclared", "selection_probability": None}, "nonnumeric exploration"),
    ({"selection_basis": "unknown_value_exploration", "selection_probability": None}, "frozen collection arm"),
])
def test_begin_cannot_change_f_or_mode_or_claim_numeric_authority(tmp_path, change, match):
    journal, run, _ = create(tmp_path)
    with pytest.raises(ValueError, match=match):
        begin(journal, run, **change)
    assert journal.read(run["run_id"])["budget"]["reserved_requests"] == 0


def test_predictor_only_cannot_be_silently_added_to_legacy_run(tmp_path):
    journal, run, _ = create(tmp_path, initial_assessment=None)
    assert run["schema"] == "prospective-competition-journal-v1"
    assert "assessment_schema" not in run
    with pytest.raises(ValueError, match="opted in"):
        begin(journal, run)


def test_configured_numeric_assessment_still_requires_all_candidates(tmp_path):
    journal, run, _ = create(tmp_path, initial_assessment=None)
    with pytest.raises(ValueError, match="Every public candidate"):
        begin(journal, run, assessment={**ASSESSMENT, "status": "CONFIGURED",
            "candidate_expected_gains": {"read-article": .1}}, selection_basis="configured_numeric_policy")


def test_f_context_can_be_unchanged_or_missing_but_cannot_change_version_or_clock(tmp_path):
    journal, run, now = create(tmp_path)
    initial = journal.record_context(run["run_id"], snapshot_kind="initial", prediction_after=P,
        model_versions=["f-frozen-v1"])
    for kind in ("before_action", "after_action"):
        now[0] += timedelta(seconds=1)
        snap = journal.record_context(run["run_id"], snapshot_kind=kind, prediction_before=P,
            prediction_after=P, model_versions=["f-frozen-v1"])
        assert snap["prediction_before"] == snap["prediction_after"] == initial["prediction_after"]
    missing = journal.record_context(run["run_id"], snapshot_kind="final", model_versions=["f-frozen-v1"])
    assert missing["prediction_before"] is None and missing["prediction_after"] is None
    with pytest.raises(ValueError, match="frozen model versions"):
        journal.record_context(run["run_id"], snapshot_kind="final", prediction_after=P, model_versions=["other"])
    with pytest.raises(ValueError, match="after captured_at"):
        journal.record_context(run["run_id"], snapshot_kind="final", model_versions=["f-frozen-v1"],
            prediction_cutoff=now[0] + timedelta(hours=1))


def test_begin_records_exact_f_prediction_and_no_g(tmp_path):
    journal, run, _ = create(tmp_path)
    before = {"home": .6, "draw": .25, "away": .15}
    result = begin(journal, run, assessment={**ASSESSMENT, "prediction": before})
    assert result["intent"]["assessment"]["prediction"] == before
    assert result["intent"]["assessment"]["candidate_expected_gains"] is None
    assert result["intent"]["selection_probability"] == .5


def test_initial_context_cannot_replace_initial_f_assessment(tmp_path):
    journal, run, _ = create(tmp_path)
    with pytest.raises(ValueError, match="frozen initial f assessment"):
        journal.record_context(run["run_id"], snapshot_kind="initial", model_versions=["f-frozen-v1"],
            prediction_after={"home": .6, "draw": .2, "away": .2})


def test_late_audit_snapshot_retains_real_clocks_and_frozen_forecast(tmp_path):
    journal, run, now = create(tmp_path)
    forecast_time = now[0]
    now[0] = datetime(2030, 1, 1, 12, 0, 1, tzinfo=timezone.utc)
    snap = journal.record_context(run["run_id"], snapshot_kind="final", observed_at=now[0],
        available_at=now[0], availability_basis="runtime_state_handoff", prediction_after=P,
        prediction_cutoff=forecast_time, forecast_as_of=forecast_time,
        acquisition_deadline="2030-01-01T11:59:00Z", model_versions=["f-frozen-v1"])
    assert snap["observed_at"] == snap["captured_at"] == snap["available_at"] == now[0].isoformat()
    assert snap["forecast_as_of"] == forecast_time.isoformat()
    assert snap["recording_scope"] == "audit_only" and not snap["forecast_eligible"]
    assert journal.stop(run["run_id"], "kickoff_elapsed")["state"] == "STOPPED"
    assert journal.store.verify(run["run_id"])["events_verified"] == 3


@pytest.mark.parametrize("fields", [
    {"forecast_as_of": "2030-01-01T12:00:00Z", "prediction_cutoff": "2030-01-01T12:00:00Z"},
    {"forecast_as_of": "2030-01-01T11:59:01Z", "prediction_cutoff": "2030-01-01T11:59:01Z"},
    {"prediction_cutoff": "2030-01-01T09:00:01Z"},
    {"acquisition_deadline": None},
])
def test_audit_mode_cannot_backdate_late_forecasts_or_omit_deadline(tmp_path, fields):
    journal, run, now = create(tmp_path)
    args = {"prediction_cutoff": now[0], "forecast_as_of": now[0],
            "acquisition_deadline": "2030-01-01T11:59:00Z"}
    args.update(fields)
    now[0] = datetime(2030, 1, 1, 12, 1, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        journal.record_context(run["run_id"], snapshot_kind="final", prediction_after=P,
            model_versions=["f-frozen-v1"], **args)


def test_legacy_context_still_rejects_postkickoff_capture(tmp_path):
    journal, run, now = create(tmp_path)
    now[0] = datetime(2030, 1, 1, 12, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="must precede kickoff"):
        journal.record_context(run["run_id"], snapshot_kind="final", prediction_after=P,
                               model_versions=["f-frozen-v1"])
