from __future__ import annotations

from copy import deepcopy
import json

from football_platform.information.stream_validator import validate_stream
from football_research.data_adapters.prospective_stream import (
    CONTEXT_FIELDS, _context_documents, _context_fields, _context_cutoff_fields, _context_information,
    _forecast_fields, export_batch,
)


OBSERVED = "2026-10-09T08:00:00+00:00"
CAPTURED = "2026-10-09T08:00:01+00:00"
AVAILABLE = "2026-10-09T08:00:02+00:00"


def _market_config():
    return {"policy_manifest": {"schema": "prospective-market-prior-policy-v1",
        "market_initialization": {"probabilities": [.5, .3, .2], "quote": {
            "home_price": 2., "draw_price": 3., "away_price": 4., "provider": "spftll",
            "observed_at": OBSERVED, "source_timestamp": "2026-10-09T07:59:00+00:00"}}}}


def test_context_fields_have_individual_clocks_and_unknowns_remain_unknown():
    state = {"observed_at": AVAILABLE, "state": {"groups": {
        "ranking": {"status": "observed", "values": {"home": 4, "away": 7},
                    "observed_at": OBSERVED, "captured_at": CAPTURED},
        "points": {"status": "not_acquired", "values": {"hidden": 99}, "available_at": None},
        "weather": {"status": "observed", "values": {"rain": True}},
    }}, "documents": [{"text": "Injury and lineup news; clear weather."}]}
    original = deepcopy(state)
    fields = _context_fields(state, {})
    assert tuple(fields) == CONTEXT_FIELDS
    assert fields["ranking"]["value"] == {"home": 4, "away": 7}
    assert fields["ranking"]["observed_at"] == OBSERVED
    assert fields["ranking"]["captured_at"] == CAPTURED
    assert fields["ranking"]["available_at"] is None
    assert fields["points"]["value"] is None
    assert fields["weather"]["observed_at"] is None
    for field in ("qualification_status", "schedule", "injuries", "lineup_status"):
        assert fields[field]["value"] is None
        assert fields[field]["observed_at"] is None
        assert fields[field]["availability_status"] == "unknown"
    assert state == original


def test_odds_keep_source_quote_observation_and_capture_times_separate():
    fields = _context_fields({}, _market_config())
    odds = fields["odds"]
    assert odds["value"]["home_price"] == 2.
    assert odds["observed_at"] == OBSERVED
    assert odds["quote_time"] == "2026-10-09T07:59:00+00:00"
    assert odds["captured_at"] is None
    assert odds["available_at"] is None
    assert odds["availability_status"] == "unknown"
    assert odds["quote_time_basis"] == "recorded_provider_source_timestamp"


def test_market_prior_is_not_exported_as_trained_f_prediction():
    config = _market_config()
    before = {"home": .5, "draw": .3, "away": .2}
    after = {"home": .4, "draw": .3, "away": .3}
    values = _forecast_fields(config, before, after)
    assert values["prediction_before"] is None
    assert values["prediction_after"] is None
    assert values["market_prior"] == before
    assert values["prediction_status"] == "market_prior_only"
    assert values["calibration_status"] == "not_established"
    unknown = _forecast_fields({}, before, after)
    assert unknown["prediction_before"] is None
    assert unknown["prediction_status"] == "model_not_configured"
    assert unknown["market_prior"] is None


def test_recorded_fitted_predictor_keeps_its_real_predictions():
    config = _market_config()
    config["policy_manifest"] = {"schema": "visible-context-empirical-query-policy-v1",
        "predictor_version": "f:123", "predictor_sha256": "a" * 64,
        "base_policy": config["policy_manifest"]}
    before = {"home": .5, "draw": .3, "away": .2}
    after = {"home": .4, "draw": .3, "away": .3}
    values = _forecast_fields(config, before, after)
    assert values["prediction_before"] == before
    assert values["prediction_after"] == after
    assert values["prediction_status"] == "trained_f_configured"
    assert values["calibration_status"] == "not_established"


def test_information_projection_lists_only_recorded_document_ids_and_explicit_state():
    state = {"match_id": "m-1", "base": {"kickoff": "2026-10-09T19:00:00+00:00",
        "competition_id": "test-league"}}
    fields = _context_fields(state, {})
    values = _context_information(state, fields, [{"document_id": "doc-1"},
        {"document_id": "doc-2"}, {"document_id": "doc-1"}, {"title": "no ID"}])
    assert values["known_document_ids"] == ["doc-1", "doc-2"]
    assert values["known_information"] == ["fixture", "competition", "document:doc-1", "document:doc-2"]
    assert all(name in values["unknown_information"] for name in CONTEXT_FIELDS)
    assert "export_projection_of_explicit_state" in values["unknown_information_basis"]
    empty = _context_information({}, fields, [])
    assert empty["known_document_ids"] == []
    assert empty["known_information"] == []
    nested = {"base": {"match_id": "m-nested", "kickoff": "2026-10-09T19:00:00+00:00",
                        "competition_id": "nested-league"}}
    nested_values = _context_information(nested, _context_fields(nested, {}), [])
    assert nested_values["known_information"] == ["fixture", "competition"]


def test_context_documents_preserve_raw_empty_and_fallback_only_missing_key():
    public = {"documents": [{"document_id": "public-1"}]}
    assert _context_documents({"documents": []}, public) == []
    assert _context_documents({}, public) == public["documents"]
    assert _context_documents({"documents": None}, public) == []


def test_export_after_and_final_document_state_comes_from_their_own_archive(tmp_path):
    from datetime import datetime
    from football_spider.storage.raw_archive import RawArchive

    folder = _batch(tmp_path)
    archive = RawArchive(folder / "source-archive")
    timings = {}
    for kind, state, documents in (
        ("after_action", {"match_id": "m-1", "base": {"competition_id": "actual"}},
         [{"document_id": "actual-1"}, {"document_id": "actual-2"}]),
        ("final", {}, []),
    ):
        timing = {"observed_at": OBSERVED, "captured_at": CAPTURED,
            "available_at": OBSERVED, "availability_status": "known",
            "prediction_cutoff": OBSERVED, "availability_basis": "runtime_state_handoff"}
        body = json.dumps({"schema": "runtime-context-v1", "state": state,
            "documents": documents, "run_id": "run-1", "snapshot_kind": kind, **timing}).encode()
        observation = archive.save("agent_trajectory", "agent://state", {}, body,
            observed_at=datetime.fromisoformat(CAPTURED), status=200)
        timings[kind] = {**timing, "raw_path": observation.payload_path,
            "raw_sha256": observation.content_hash, "raw_observation_id": str(observation.id)}
    run = {"cycles": [{"decision_at": OBSERVED, "returned_at": OBSERVED,
        "generated_actions": [], "public_before": {"documents": []},
        "public_state_after": {"documents": [{"document_id": "unproven-public-copy"}]},
        "after_state_timing": timings["after_action"]}],
        "final_state_timing": timings["final"],
        "final_state": {"base": {"competition_id": "incorrect-fallback"}},
        "documents": [{"document_id": "incorrect-fallback"}],
        "terminal_selection": {"reason": "no_candidates"}}
    (folder / "run.json").write_text(json.dumps(run), encoding="utf-8")
    output = tmp_path / "output"
    export_batch(folder.parent, output)
    contexts = [json.loads(line) for line in (output / "context_snapshots.jsonl").read_text(encoding="utf-8").splitlines()]
    after = next(row for row in contexts if row["snapshot_kind"] == "after_action")
    assert after["known_document_ids"] == ["actual-1", "actual-2"]
    assert "document:unproven-public-copy" not in after["known_information"]
    assert "competition" in after["known_information"]
    final = next(row for row in contexts if row["snapshot_kind"] == "final")
    assert final["known_document_ids"] == []
    assert "competition" not in final["known_information"]
    assert not any(name.startswith("document:") for name in final["known_information"])


def test_snapshot_cutoff_uses_recorded_context_clock_and_preserves_run_deadline():
    deadline = "2026-10-09T18:59:00+00:00"
    config = {"initial_state": {"prediction_cutoff": deadline}}
    observed = _context_cutoff_fields(config, {"prediction_cutoff": OBSERVED}, OBSERVED)
    assert observed["prediction_cutoff"] == OBSERVED
    assert observed["acquisition_deadline"] == deadline
    missing = _context_cutoff_fields(config, {}, OBSERVED)
    assert missing["prediction_cutoff"] is None
    assert missing["prediction_cutoff_basis"] == "not_recorded"
    assert missing["acquisition_deadline"] == deadline


def _batch(tmp_path):
    folder = tmp_path / "batch" / "match-1"
    folder.mkdir(parents=True)
    config = {**_market_config(), "run_id": "run-1", "created_at": "2026-10-09T09:00:00+00:00",
        "initial_state": {"match_id": "m-1", "base": {"kickoff": "2026-10-09T19:00:00+00:00"}},
        "evidence_projection": {"match": {"match_id": "m-1", "competition": "Test League",
            "season": "2026", "home_team": "A", "away_team": "B",
            "kickoff_at": "2026-10-09T19:00:00+00:00",
            "fixture_available_at": "2026-10-09T08:30:00+00:00",
            "prediction_cutoff": "2026-10-09T18:59:00+00:00"}},
        "supervisor_metadata": {"fixture": {"source_url": "https://example.test/match-1"}}}
    (folder / "configuration.json").write_text(json.dumps(config), encoding="utf-8")
    (folder / "run.json").write_text(json.dumps({"cycles": [], "terminal_selection": {"reason": "no_candidates"}}), encoding="utf-8")
    return folder


def test_export_keeps_unknown_context_capture_clocks_and_initial_and_final_fields(tmp_path):
    folder = _batch(tmp_path)
    config_before = (folder / "configuration.json").read_bytes()
    output = tmp_path / "output"
    report = export_batch(folder.parent, output)
    contexts = [json.loads(line) for line in (output / "context_snapshots.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["snapshot_kind"] for row in contexts] == ["initial", "final"]
    assert contexts[0]["observed_at"] == "2026-10-09T09:00:00+00:00"
    assert contexts[1]["observed_at"] is None
    for row in contexts:
        assert row["captured_at"] is None
        assert row["available_at"] is None
        assert row["prediction_before"] is row["prediction_after"] is None
        assert row["prediction_status"] == "market_prior_only"
        assert row["prediction_cutoff"] is None
        assert all(field in row and "observed_at" in row[field] for field in CONTEXT_FIELDS)
    assert report["contexts_with_proven_availability_before_kickoff"] == 0
    assert validate_stream(output)["valid"] is True
    assert (folder / "configuration.json").read_bytes() == config_before


def test_context_archive_is_copied_without_altering_response_bytes(tmp_path):
    from datetime import datetime
    from football_spider.storage.raw_archive import RawArchive

    folder = _batch(tmp_path)
    archive = RawArchive(folder / "source-archive")
    timing = {"observed_at": OBSERVED, "captured_at": CAPTURED,
        "available_at": OBSERVED, "availability_status": "known",
        "prediction_cutoff": OBSERVED, "availability_basis": "runtime_state_handoff"}
    timings = {}
    for kind in ("initial", "final"):
        body = json.dumps({"schema": "runtime-context-v1", "state": {}, "run_id": "run-1",
            "snapshot_kind": kind, **timing}).encode()
        observation = archive.save("agent_trajectory", "agent://state", {}, body,
            observed_at=datetime.fromisoformat(CAPTURED), status=200)
        timings[kind] = {**timing, "raw_path": observation.payload_path,
            "raw_sha256": observation.content_hash, "raw_observation_id": str(observation.id)}
    config = json.loads((folder / "configuration.json").read_text(encoding="utf-8"))
    config["initial_state_timing"] = timings["initial"]
    (folder / "configuration.json").write_text(json.dumps(config), encoding="utf-8")
    run = {"cycles": [], "final_state_timing": timings["final"], "terminal_selection": {"reason": "no_candidates"}}
    (folder / "run.json").write_text(json.dumps(run), encoding="utf-8")
    output = tmp_path / "output"
    export_batch(folder.parent, output)
    contexts = [json.loads(line) for line in (output / "context_snapshots.jsonl").read_text(encoding="utf-8").splitlines()]
    for row in contexts:
        assert row["observed_at"] == OBSERVED
        assert row["captured_at"] is None
        assert row["source_context_captured_at"] == CAPTURED
        assert row["context_capture_completion_proven"] is False
        assert row["raw_digest_verified"] is True
        assert row["raw_context_timing_proven"] is True
        assert row["prediction_cutoff"] == OBSERVED
        source = timings[row["snapshot_kind"]]["raw_path"]
        assert (output / row["raw_path"]).read_bytes() == (folder / "source-archive" / source).read_bytes()
    config["initial_state_timing"]["captured_at"] = AVAILABLE
    (folder / "configuration.json").write_text(json.dumps(config), encoding="utf-8")
    second = tmp_path / "tampered-timing-export"
    export_batch(folder.parent, second)
    changed = json.loads((second / "context_snapshots.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert changed["raw_digest_verified"] is True
    assert changed["raw_context_timing_proven"] is True
    assert changed["context_capture_completion_proven"] is False
    assert changed["captured_at"] is None
    assert changed["source_context_captured_at"] == AVAILABLE


def test_context_save_return_receipt_proves_capture_after_raw_start_without_rewriting_archive(tmp_path):
    from datetime import datetime
    from football_spider.storage.raw_archive import RawArchive

    folder = _batch(tmp_path)
    archive = RawArchive(folder / "source-archive")
    timing = {"observed_at": OBSERVED, "available_at": OBSERVED, "availability_status": "known",
              "prediction_cutoff": OBSERVED, "availability_basis": "runtime_state_handoff"}
    body = json.dumps({"schema": "runtime-context-v1", "state": {}, "documents": [], "run_id": "run-1",
                       "snapshot_kind": "initial", **timing}).encode()
    observation = archive.save("agent_trajectory", "agent://state", {}, body,
        observed_at=datetime.fromisoformat(CAPTURED), status=200)
    callback = {**timing, "captured_at": AVAILABLE, "archive_save_started_at": CAPTURED,
        "archive_save_returned_at": AVAILABLE, "capture_timing_basis": "measured_local_utc_archive_save_return",
        "persistence_boundary": "return_observed_fsync_not_instrumented",
        "raw_path": observation.payload_path, "raw_sha256": observation.content_hash,
        "raw_observation_id": str(observation.id)}
    config = json.loads((folder / "configuration.json").read_text(encoding="utf-8"))
    config["initial_state_timing"] = callback
    (folder / "configuration.json").write_text(json.dumps(config), encoding="utf-8")
    original = (folder / "source-archive" / observation.payload_path).read_bytes()
    output = tmp_path / "output"
    export_batch(folder.parent, output)
    row = json.loads((output / "context_snapshots.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["captured_at"] == AVAILABLE
    assert row["raw_context_timing_proven"] is True
    assert row["context_capture_completion_proven"] is True
    assert row["raw_context_timing_basis"] == "local_callback_return_receipt_bound_to_raw_observation_and_payload"
    assert (output / row["raw_path"]).read_bytes() == original
    assert (folder / "source-archive" / observation.payload_path).read_bytes() == original
    callback["archive_save_returned_at"] = OBSERVED
    (folder / "configuration.json").write_text(json.dumps(config), encoding="utf-8")
    second = tmp_path / "bad-return"
    export_batch(folder.parent, second)
    changed = json.loads((second / "context_snapshots.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert changed["context_capture_completion_proven"] is False
    assert changed["raw_context_timing_proven"] is True
    assert changed["captured_at"] is None


def test_raw_handoff_and_document_membership_are_proven_without_callback_completion_fields(tmp_path):
    from datetime import datetime
    from football_spider.storage.raw_archive import RawArchive
    from football_research.data_adapters.prospective_stream import _document_timing_fields

    folder = _batch(tmp_path)
    archive = RawArchive(folder / "source-archive")
    timing = {"observed_at": OBSERVED, "available_at": OBSERVED, "availability_status": "known",
              "prediction_cutoff": OBSERVED, "availability_basis": "runtime_state_handoff"}
    body = json.dumps({"schema": "runtime-context-v1", "state": {}, "documents": [{"document_id": "admitted-doc"}],
        "run_id": "run-1", "snapshot_kind": "after_action", **timing}).encode()
    obs = archive.save("agent_trajectory", "agent://state", {}, body,
                       observed_at=datetime.fromisoformat(CAPTURED), status=200)
    timing.update(captured_at=AVAILABLE, raw_path=obs.payload_path, raw_sha256=obs.content_hash,
                  raw_observation_id=str(obs.id))
    run = {"cycles": [{"decision_at": OBSERVED, "returned_at": OBSERVED, "generated_actions": [],
        "after_state_timing": timing, "public_before": {"documents": []}}]}
    (folder / "run.json").write_text(json.dumps(run), encoding="utf-8")
    output = tmp_path / "output"
    export_batch(folder.parent, output)
    contexts = [json.loads(line) for line in (output / "context_snapshots.jsonl").read_text(encoding="utf-8").splitlines()]
    handoff = next(row for row in contexts if row["snapshot_kind"] == "after_action")
    assert handoff["captured_at"] is None
    assert handoff["context_capture_completion_proven"] is False
    assert handoff["raw_context_timing_proven"] is True
    assert handoff["raw_document_membership_proven"] is True
    assert handoff["source_context_captured_at"] == AVAILABLE
    admitted = _document_timing_fields({"document_id": "admitted-doc"}, {}, handoff)
    assert admitted["document_admitted"] is True
    assert admitted["document_available_at"] == OBSERVED
    absent = _document_timing_fields({"document_id": "no-data-doc"}, {}, handoff)
    assert absent["document_admitted"] is False
    assert absent["document_available_at"] is None
    # Changing the state handoff clock still destroys its independent proof.
    timing["available_at"] = AVAILABLE
    (folder / "run.json").write_text(json.dumps(run), encoding="utf-8")
    second = tmp_path / "bad-handoff"
    export_batch(folder.parent, second)
    changed = next(json.loads(line) for line in (second / "context_snapshots.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line)["snapshot_kind"] == "after_action")
    assert changed["raw_context_timing_proven"] is False
    assert _document_timing_fields({"document_id": "admitted-doc"}, {}, changed)["document_admitted"] is None
