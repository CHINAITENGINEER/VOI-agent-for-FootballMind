"""Constructed journal tests; zero network and zero real policy-gain samples."""
from datetime import datetime, timedelta
import hashlib
from types import SimpleNamespace

import pytest

from football_platform.information.acquisition_journal import AcquisitionDecisionJournal
from football_platform.information.evidence_registry import CompetitionEvidenceRegistry
from football_platform.information.prospective_capture import capture_once
from football_spider.storage.raw_archive import RawArchive

FIXTURE = dict(match_id="constructed-fixture", competition_id="constructed-cup", home_team_id="home-a", away_team_id="away-b",
               kickoff="2026-10-01T12:00:00Z")
ENDPOINT = "https://example.test/context"
BODY = b'{"qualification":"qualified"}'


def make(tmp_path, *, limit=4, fixture=FIXTURE):
    clock = [datetime.fromisoformat("2026-10-01T08:00:00+00:00")]
    archive = RawArchive(tmp_path / "archive")
    journal = AcquisitionDecisionJournal(tmp_path / "journal", archive, clock=lambda: clock[0])
    run = journal.start(request_limit=limit, fixture=fixture)
    return journal, run["run_id"], clock


def begin(journal, run_id, **changes):
    parameters = dict(request_id="request-1", action_id="query.qualification", input_parameters={"endpoint": ENDPOINT, "params": {}},
                      candidates=["query.qualification"], policy_version="constructed-manual-v1")
    parameters.update(changes)
    return journal.begin(run_id, **parameters)


def archive_response(journal, run_id, clock, *, request_id="request-1", status=200, error=None, endpoint=ENDPOINT,
                     requested_at=None):
    requested = requested_at or clock[0]
    clock[0] += timedelta(seconds=1)
    return journal.archive.save("constructed_source", endpoint, {}, BODY, status=status, observed_at=clock[0],
                                error_class=error, context={"journal_request_id": request_id, "journal_run_id": run_id,
                                                           "requested_at": requested.isoformat()})


def test_durable_intent_precedes_return_and_records_no_fabricated_model_outputs(tmp_path):
    journal, run_id, clock = make(tmp_path)
    decision = begin(journal, run_id)
    assert decision["execute"]
    before = journal.read(run_id)
    assert before["budget"]["reserved_requests"] == 1 and before["budget"]["charged_requests"] == 0
    assert before["prediction"] is None and before["candidate_expected_gains"] is None
    assert before["model_status"] == "NOT_CONFIGURED"
    assert decision["intent"]["selection_probability"] is None
    assert decision["intent"]["current_evidence_ids"] == []
    obs = archive_response(journal, run_id, clock)
    receipt = journal.complete(run_id, request_id="request-1", observation=obs)
    after = journal.read(run_id)
    assert receipt["observed_at"] > decision["intent"]["decision_at"]
    assert after["budget"]["reserved_requests"] == 0 and after["budget"]["charged_requests"] == 1
    assert after["budget"]["provider_money"] is None
    assert after["acquired_observation_ids"] == [obs.id]
    assert journal.store.verify(run_id)["events_verified"] == 3


def test_repeated_intent_and_receipt_are_idempotent_and_do_not_repeat_tool(tmp_path):
    journal, run_id, clock = make(tmp_path)
    begin(journal, run_id)
    assert begin(journal, run_id)["execute"] is False
    assert begin(journal, run_id)["recovery_status"] == "NEEDS_RECONCILIATION"
    obs = archive_response(journal, run_id, clock)
    receipt = journal.complete(run_id, request_id="request-1", observation=obs)
    assert journal.complete(run_id, request_id="request-1", observation=obs) == receipt
    assert begin(journal, run_id)["receipt"] == receipt
    assert journal.read(run_id)["budget"]["charged_requests"] == 1


def test_same_request_identity_cannot_be_reused_for_different_inputs(tmp_path):
    journal, run_id, _ = make(tmp_path)
    begin(journal, run_id)
    with pytest.raises(ValueError, match="different intent"):
        begin(journal, run_id, input_parameters={"endpoint": ENDPOINT, "params": {"team": "different"}})


def test_crash_after_archive_is_recovered_without_another_network_request(tmp_path):
    journal, run_id, clock = make(tmp_path)
    begin(journal, run_id)
    obs = archive_response(journal, run_id, clock)
    restarted = AcquisitionDecisionJournal(journal.store.root, journal.archive, clock=lambda: clock[0])
    recovery = restarted.recover_pending(run_id)
    assert recovery["status"] == "RECOVERED_FROM_RAW_ARCHIVE"
    assert recovery["receipt"]["observation_id"] == obs.id
    assert restarted.read(run_id)["budget"]["charged_requests"] == 1


def test_missing_return_keeps_reservation_and_never_assumes_zero_cost_or_retries(tmp_path):
    journal, run_id, _ = make(tmp_path)
    begin(journal, run_id)
    recovery = journal.recover_pending(run_id)
    assert recovery["status"] == "NEEDS_RECONCILIATION"
    assert recovery["network_retry_performed"] is False and recovery["reservation_retained"] is True
    with pytest.raises(ValueError, match="pending"):
        journal.stop(run_id, "NOT_CONFIGURED")
    assert journal.read(run_id)["budget"]["reserved_requests"] == 1


def test_unknown_receipt_rejected_before_reading_raw_payload(tmp_path, monkeypatch):
    journal, run_id, clock = make(tmp_path)
    obs = archive_response(journal, run_id, clock)
    monkeypatch.setattr(journal.archive, "read_body", lambda *_: pytest.fail("Unpurchased payload was read"))
    with pytest.raises(ValueError, match="durable pending intent"):
        journal.complete(run_id, request_id="request-1", observation=obs)


def test_pre_intent_request_is_rejected_even_if_response_arrives_later(tmp_path):
    journal, run_id, clock = make(tmp_path)
    earlier = clock[0] - timedelta(seconds=1)
    begin(journal, run_id)
    obs = archive_response(journal, run_id, clock, requested_at=earlier)
    with pytest.raises(ValueError, match="after intent"):
        journal.complete(run_id, request_id="request-1", observation=obs)


def test_request_between_decision_and_durable_event_is_rejected(tmp_path, monkeypatch):
    journal, run_id, clock = make(tmp_path)
    decision_time = clock[0]
    original_append = journal._append

    def delayed_commit(run, kind, data):
        clock[0] += timedelta(seconds=2)
        return original_append(run, kind, data)

    monkeypatch.setattr(journal, "_append", delayed_commit)
    begin(journal, run_id)
    obs = archive_response(journal, run_id, clock, requested_at=decision_time + timedelta(seconds=1))
    with pytest.raises(ValueError, match="after intent"):
        journal.complete(run_id, request_id="request-1", observation=obs)


def test_return_metadata_cannot_backdate_committed_observation(tmp_path):
    journal, run_id, clock = make(tmp_path)
    begin(journal, run_id)
    obs = archive_response(journal, run_id, clock)
    forged = obs.model_copy(update={"observed_at": obs.observed_at - timedelta(microseconds=1)})
    with pytest.raises(ValueError, match="metadata differs"):
        journal.complete(run_id, request_id="request-1", observation=forged)


def test_fixture_binding_requires_purchased_source_and_future_kickoff(tmp_path):
    journal, run_id, clock = make(tmp_path, fixture=None)
    with pytest.raises(ValueError, match="purchased evidence"):
        journal.bind_fixture(run_id, fixture=FIXTURE, observation_id="unbought", source_json_path="$.fixture")
    begin(journal, run_id)
    obs = archive_response(journal, run_id, clock)
    journal.complete(run_id, request_id="request-1", observation=obs)
    clock[0] = datetime.fromisoformat(FIXTURE["kickoff"].replace("Z", "+00:00"))
    with pytest.raises(ValueError, match="future fixture"):
        journal.bind_fixture(run_id, fixture=FIXTURE, observation_id=obs.id, source_json_path="$.fixture")


def test_wrong_source_endpoint_is_not_accepted_as_this_tool_return(tmp_path):
    journal, run_id, clock = make(tmp_path)
    begin(journal, run_id)
    obs = archive_response(journal, run_id, clock, endpoint="https://example.test/other-tool")
    with pytest.raises(ValueError, match="differ from the durable intent"):
        journal.complete(run_id, request_id="request-1", observation=obs)


def test_archived_failure_is_charged_and_does_not_publish_observation(tmp_path):
    journal, run_id, clock = make(tmp_path)
    begin(journal, run_id)
    obs = archive_response(journal, run_id, clock, status=500)
    receipt = journal.complete(run_id, request_id="request-1", observation=obs)
    assert receipt["status"] == "FAILED"
    run = journal.read(run_id)
    assert run["budget"]["charged_requests"] == 1 and run["acquired_observation_ids"] == []


def test_return_after_kickoff_is_charged_but_cannot_enter_prematch_evidence(tmp_path):
    journal, run_id, clock = make(tmp_path)
    begin(journal, run_id)
    requested = clock[0]
    clock[0] = datetime.fromisoformat(FIXTURE["kickoff"].replace("Z", "+00:00"))
    obs = archive_response(journal, run_id, clock, requested_at=requested)
    assert journal.complete(run_id, request_id="request-1", observation=obs)["status"] == "LATE_RETURN"
    run = journal.read(run_id)
    assert run["state"] == "STOPPED" and run["acquired_observation_ids"] == []
    assert run["budget"]["charged_requests"] == 1


def test_actual_request_overrun_is_not_hidden_by_the_original_reservation(tmp_path):
    journal, run_id, clock = make(tmp_path, limit=1)
    begin(journal, run_id)
    obs = archive_response(journal, run_id, clock)
    journal.complete(run_id, request_id="request-1", observation=obs, charged_requests=2)
    run = journal.read(run_id)
    assert run["budget"]["charged_requests"] == 2
    assert run["stop_reason"] == "ACTUAL_REQUEST_BUDGET_OVERRUN"


def test_budget_prevents_another_intent_after_budget_is_spent(tmp_path):
    journal, run_id, clock = make(tmp_path, limit=1)
    begin(journal, run_id)
    journal.complete(run_id, request_id="request-1", observation=archive_response(journal, run_id, clock))
    with pytest.raises(ValueError, match="budget exhausted"):
        begin(journal, run_id, request_id="request-2")


def test_unconfigured_assessment_rejects_fake_forecast_or_zero_expected_gain(tmp_path):
    journal, run_id, _ = make(tmp_path)
    with pytest.raises(ValueError, match="Unconfigured models"):
        begin(journal, run_id, assessment={"status": "NOT_CONFIGURED", "prediction": None,
                                          "candidate_expected_gains": {"query.qualification": 0.0}, "model_versions": []})


def test_configured_model_records_versions_probability_and_expected_gain(tmp_path):
    journal, run_id, _ = make(tmp_path)
    assessment = {"status": "CONFIGURED", "prediction": {"home": .4, "draw": .3, "away": .3},
                  "candidate_expected_gains": {"query.qualification": .02}, "model_versions": ["constructed-model-v1"]}
    result = begin(journal, run_id, assessment=assessment, selection_basis="configured_numeric_policy",
                   policy_version="constructed-policy-v1", selection_probability=.8)
    assert result["intent"]["selection_probability"] == .8
    assert result["intent"]["assessment"]["model_versions"] == ["constructed-model-v1"]


def test_manual_selection_does_not_manufacture_a_propensity(tmp_path):
    journal, run_id, _ = make(tmp_path)
    with pytest.raises(ValueError, match="Manual collection"):
        begin(journal, run_id, selection_probability=1.0)


def test_published_context_must_be_from_a_successfully_purchased_return(tmp_path):
    journal, run_id, clock = make(tmp_path)
    begin(journal, run_id)
    obs = archive_response(journal, run_id, clock)
    journal.complete(run_id, request_id="request-1", observation=obs)
    registry = CompetitionEvidenceRegistry(tmp_path / "registry", journal.archive, clock=lambda: clock[0])
    record = registry.register(evidence_id="candidate-1", fixture=FIXTURE, observation=obs, document_scope="pre_match_preview",
                               source_kind="synthetic", claim={"topic": "qualification", "value": "qualified", "team_id": "home-a"},
                               anchor={"byte_start": 0, "byte_end": len(BODY), "sha256": hashlib.sha256(BODY).hexdigest()})
    projection = journal.publish_evidence(run_id, request_id="request-1", evidence_ids=[record.evidence_id], registry=registry)
    assert projection.context.teams["home-a"]["qualification"].status == "unverified"
    assert journal.read(run_id)["current_evidence_ids"] == ["candidate-1"]
    next_intent = begin(journal, run_id, request_id="request-2")
    assert next_intent["intent"]["current_evidence_ids"] == ["candidate-1"]


def test_unpurchased_registry_records_are_not_read(tmp_path, monkeypatch):
    journal, run_id, clock = make(tmp_path)
    registry = CompetitionEvidenceRegistry(tmp_path / "registry", journal.archive, clock=lambda: clock[0])
    monkeypatch.setattr(registry, "read", lambda *_: pytest.fail("Unpurchased registry record was read"))
    with pytest.raises(ValueError, match="successfully purchased"):
        journal.publish_evidence(run_id, request_id="never-selected", evidence_ids=["hidden"], registry=registry)


def test_immutable_event_blobs_detect_mutation_and_stopping_is_idempotent(tmp_path):
    journal, run_id, _ = make(tmp_path)
    stopped = journal.stop(run_id, "NOT_CONFIGURED")
    assert journal.stop(run_id, "NOT_CONFIGURED") == stopped
    path = journal.store._folder(run_id) / stopped["event_blob_refs"][0]
    path.write_bytes(b"{}")
    with pytest.raises(ValueError, match="checksum"):
        journal.read(run_id)


def test_capture_once_commits_intent_before_fake_http_and_replay_has_no_request(tmp_path):
    journal, run_id, _ = make(tmp_path)
    calls = []
    class Session:
        def get(self, *args, **kwargs):
            before = journal.read(run_id)
            assert before["state"] == "PENDING_RETURN" and before["budget"]["reserved_requests"] == 1
            calls.append("get")
            return SimpleNamespace(content=BODY, status_code=200, headers={"content-type": "application/json"}, close=lambda: None)
        def close(self):
            pass
    args = dict(request_id="captured", action_id="query.qualification", source="constructed_source", endpoint=ENDPOINT, session=Session())
    first = capture_once(journal, run_id, **args)
    second = capture_once(journal, run_id, **args)
    assert first["new_http_attempts"] == 1 and second["new_http_attempts"] == 0
    assert calls == ["get"] and journal.read(run_id)["budget"]["charged_requests"] == 1


def test_http_success_with_no_data_remains_charged_but_is_not_usable_evidence(tmp_path, monkeypatch):
    journal, run_id, clock = make(tmp_path)
    begin(journal, run_id)
    obs = archive_response(journal, run_id, clock)
    journal.complete(run_id, request_id="request-1", observation=obs)
    record = journal.record_processing(run_id, request_id="request-1", status="NO_DATA", extraction_sha256="0" * 64,
                                       missing_fields=["lineup"])
    assert journal.record_processing(run_id, request_id="request-1", status="NO_DATA", extraction_sha256="0" * 64,
                                      missing_fields=["lineup"]) == record
    with pytest.raises(ValueError, match="append-only"):
        journal.record_processing(run_id, request_id="request-1", status="DATA_AVAILABLE", extraction_sha256="0" * 64)
    registry = CompetitionEvidenceRegistry(tmp_path / "registry", journal.archive, clock=lambda: clock[0])
    monkeypatch.setattr(registry, "read", lambda *_: pytest.fail("No-data response cannot publish payload"))
    with pytest.raises(ValueError, match="usable semantic data"):
        journal.publish_evidence(run_id, request_id="request-1", evidence_ids=["hidden"], registry=registry)
    assert journal.read(run_id)["budget"]["charged_requests"] == 1


def test_start_records_predeclared_arm_seed_and_assignment_clock(tmp_path):
    clock = [datetime.fromisoformat("2026-10-01T08:00:00+00:00")]
    journal = AcquisitionDecisionJournal(tmp_path / "journal", RawArchive(tmp_path / "archive"), clock=lambda: clock[0])
    run = journal.start(request_limit=2, fixture=FIXTURE, policy_arm="random_explore",
                        policy_version="policy-v2", random_seed=1234, propensity_logging=True)
    assert run["policy_arm"] == "random_explore"
    assert run["arm_assigned_at"] == "2026-10-01T08:00:00+00:00"
    assert run["random_seed"] == 1234 and run["propensity_logging"] is True
    event = run["events"][0]
    assert event["type"] == "journal_started" and event["at"] == run["arm_assigned_at"]
    assert event["data"]["policy_version"] == "policy-v2"


def test_random_arm_requires_seed_and_propensity_logging(tmp_path):
    journal, _, _ = make(tmp_path)
    with pytest.raises(ValueError, match="random_explore requires a random seed"):
        journal.start(policy_arm="random_explore")
    with pytest.raises(ValueError, match="propensity logging"):
        journal.start(policy_arm="random_explore", random_seed=1, propensity_logging=False)


def test_begin_preserves_ordered_candidate_metadata_and_non_numeric_bases(tmp_path):
    journal, run_id, _ = make(tmp_path)
    menu = [{"candidate_id": "query.qualification", "action_type": "competition_context", "rank": 0},
            {"candidate_id": "query.weather", "action_type": "weather", "rank": 1}]
    result = begin(journal, run_id, candidates=[item["candidate_id"] for item in menu],
                   action_id="query.qualification", candidate_metadata=menu,
                   selection_basis="unknown_value_exploration")
    assert result["intent"]["candidate_metadata"] == menu
    assert result["intent"]["policy_kind"] == "unknown_value_exploration"
    assert result["intent"]["selection_probability"] is None


def test_uniform_exploration_requires_real_design_probability(tmp_path):
    journal, run_id, _ = make(tmp_path)
    with pytest.raises(ValueError, match="design probability"):
        begin(journal, run_id, selection_basis="uniform_exploration")
    result = begin(journal, run_id, selection_basis="uniform_exploration", selection_probability=.5)
    assert result["intent"]["selection_probability"] == .5


def test_capture_once_forwards_selection_basis_probability_and_menu(tmp_path):
    journal, run_id, _ = make(tmp_path)
    menu = [{"candidate_id": "query.qualification", "title": "Qualification", "rank": 0}]
    calls = []

    class Session:
        def get(self, *args, **kwargs):
            calls.append("get")
            return SimpleNamespace(content=BODY, status_code=200,
                                   headers={"content-type": "application/json"}, close=lambda: None)

        def close(self):
            pass

    result = capture_once(journal, run_id, request_id="forwarded", action_id="query.qualification",
                          source="constructed_source", endpoint=ENDPOINT, session=Session(),
                          candidates=["query.qualification"], candidate_metadata=menu,
                          selection_basis="uniform_exploration", selection_probability=1.0,
                          policy_version="uniform-v1")
    assert result["receipt"]["status"] == "RETURNED"
    intent = journal.read(run_id)["intents"]["forwarded"]
    assert intent["candidate_metadata"] == menu
    assert intent["selection_basis"] == "uniform_exploration"
    assert intent["selection_probability"] == 1.0 and calls == ["get"]


def test_record_context_keeps_available_unknown_and_null_unconfigured_predictions(tmp_path):
    journal, run_id, clock = make(tmp_path)
    initial = journal.record_context(run_id, snapshot_kind="initial", known_information=["fixture"],
                                     unknown_information=["lineup"])
    assert initial["captured_at"] == "2026-10-01T08:00:00+00:00"
    assert initial["available_at"] is None and initial["availability_status"] == "unknown"
    assert initial["prediction_before"] is None and initial["prediction_after"] is None
    clock[0] += timedelta(seconds=1)
    before = journal.record_context(run_id, snapshot_kind="before_action", observed_at=clock[0],
                                    parent_request_id="request-1")
    assert before["parent_request_id"] == "request-1" and before["sequence"] == 1
    clock[0] += timedelta(seconds=1)
    after = journal.record_context(run_id, snapshot_kind="after_action", observed_at=clock[0],
                                   parent_request_id="request-1", parent_attempt_id="attempt-1")
    assert after["parent_attempt_id"] == "attempt-1" and after["sequence"] == 2
    clock[0] += timedelta(seconds=1)
    final = journal.record_context(run_id, snapshot_kind="final", stop_reason="NO_POSITIVE_VALUE")
    assert final["sequence"] == 3 and final["stop_reason"] == "NO_POSITIVE_VALUE"
    assert len(journal.read(run_id)["context_snapshots"]) == 4


def test_stop_preserves_real_stop_clock(tmp_path):
    journal, run_id, clock = make(tmp_path)
    clock[0] += timedelta(seconds=4)
    stopped = journal.stop(run_id, "NO_POSITIVE_VALUE")
    assert stopped["stopped_at"] == "2026-10-01T08:00:04+00:00"
    event = stopped["events"][-1]
    assert event["type"] == "stopped" and event["at"] == stopped["stopped_at"]


def test_candidate_menu_order_changes_cannot_reuse_intent_identity(tmp_path):
    journal, run_id, _ = make(tmp_path)
    menu = [{"candidate_id": "query.qualification", "title": "Qualification"}]
    begin(journal, run_id, candidate_metadata=menu)
    with pytest.raises(ValueError, match="different intent"):
        begin(journal, run_id, candidate_metadata=[{"candidate_id": "query.qualification", "title": "Changed"}])
    with pytest.raises(ValueError, match="preserve candidate order"):
        begin(journal, run_id, candidate_metadata=[{"candidate_id": "query.weather"}])


def test_legacy_intent_signature_replays_without_rewriting_history(tmp_path):
    from football_agent.audit import digest

    journal, run_id, _ = make(tmp_path)
    decision = begin(journal, run_id)
    # Construct the exact original v1 shape through the existing store helper;
    # the test deliberately creates no HTTP calls or historical timestamps.
    run = journal.read(run_id)
    intent = run["intents"]["request-1"]
    legacy_fields = {"action_id": intent["action_id"],
                     "input_parameters": {"endpoint": ENDPOINT, "params": {}},
                     "candidates": intent["candidates"], "reserve_requests": 1,
                     "assessment": intent["assessment"], "selection_basis": "manual_predeclared",
                     "policy_version": "constructed-manual-v1", "selection_probability": None}
    intent["signature"] = digest(legacy_fields)
    intent.pop("candidate_metadata")
    # Rebuild this constructed fixture's last immutable event with the v1 intent.
    event = run["events"][-1]
    event["data"].pop("candidate_metadata")
    event["data"]["signature"] = intent["signature"]
    event["snapshot"]["intents"]["request-1"] = dict(intent)
    event.pop("event_hash")
    event["event_hash"] = digest(event)
    run["event_blob_refs"][-1] = journal.store.put_blob(run_id, event)
    journal.store.save(run)
    before = journal.read(run_id)
    replay = begin(journal, run_id)
    assert replay["execute"] is False and replay["recovery_status"] == "NEEDS_RECONCILIATION"
    assert journal.read(run_id) == before
    with pytest.raises(ValueError, match="different intent"):
        begin(journal, run_id, candidate_metadata=[{"candidate_id": "query.qualification"}])


def test_context_rejects_future_clocks_and_fabricated_availability_or_predictions(tmp_path):
    journal, run_id, clock = make(tmp_path)
    with pytest.raises(ValueError, match="availability_basis"):
        journal.record_context(run_id, snapshot_kind="initial", available_at=clock[0])
    known = journal.record_context(run_id, snapshot_kind="initial", available_at=clock[0],
                                   availability_basis="agent_state_was_published")
    assert known["available_at"] == known["observed_at"]
    with pytest.raises(ValueError, match="Unconfigured models"):
        journal.record_context(run_id, snapshot_kind="after_action",
                               prediction_after={"home": .4, "draw": .3, "away": .3})
    with pytest.raises(ValueError, match="prediction_cutoff cannot be after captured_at"):
        journal.record_context(run_id, snapshot_kind="initial", prediction_cutoff=clock[0] + timedelta(seconds=1))
    clock[0] = datetime.fromisoformat(FIXTURE["kickoff"].replace("Z", "+00:00"))
    with pytest.raises(ValueError, match="must precede kickoff"):
        journal.record_context(run_id, snapshot_kind="final")
