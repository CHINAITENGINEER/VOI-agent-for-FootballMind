"""Frozen f-only snapshots must survive export without inventing observations."""
from datetime import datetime
import json

import pytest

from football_research.data_adapters.prospective_stream import (
    _context_payload, _context_raw_fields, _document_timing_fields,
    _predictor_context_fields, export_batch,
)
from football_spider.storage.raw_archive import RawArchive
from tests.research.data_adapters.test_prospective_stream_contexts import _batch


FORECAST = "2026-10-09T09:00:00+00:00"
HANDOFF = "2026-10-09T09:00:01+00:00"
SAVE_START = "2026-10-09T09:00:02+00:00"
SAVE_RETURN = "2026-10-09T09:00:03+00:00"
DEADLINE = "2026-10-09T18:59:00+00:00"
KICKOFF = "2026-10-09T19:00:00+00:00"
BEFORE = {"home": .57, "draw": .22, "away": .21}
AFTER = {"home": .50, "draw": .29, "away": .21}


def archive_context(folder, *, kind, schema, late=False):
    handed, started, returned = (HANDOFF, SAVE_START, SAVE_RETURN) if not late else (
        "2026-10-09T19:00:01+00:00", "2026-10-09T19:00:02+00:00", "2026-10-09T19:00:03+00:00")
    payload = {"schema": schema, "run_id": "run-1", "snapshot_kind": kind,
        "state": {"match_id": "m-1", "base": {"kickoff": KICKOFF}},
        "documents": [] if kind == "initial" else [{"document_id": "body-1"}],
        "observed_at": handed, "available_at": handed,
        "availability_status": "known", "availability_basis": "runtime_state_handoff",
        "prediction_cutoff": FORECAST, "acquisition_deadline": DEADLINE,
        "parent_attempt_id": None,
        "prediction_before": BEFORE if kind == "after_action" else None,
        "prediction_after": BEFORE if kind == "initial" else AFTER,
        "predictor_model_versions": ["f:frozen"]}
    if schema == "runtime-predictor-context-v3":
        payload["forecast_as_of"] = FORECAST
    obs = RawArchive(folder / "source-archive").save("agent_trajectory", "agent://state", {},
        json.dumps(payload).encode(), status=200, observed_at=datetime.fromisoformat(started))
    timing = {k: payload[k] for k in ("observed_at", "available_at", "availability_status",
        "availability_basis", "prediction_cutoff", "parent_attempt_id")}
    timing.update(captured_at=returned, archive_save_started_at=started, archive_save_returned_at=returned,
        capture_timing_basis="measured_local_utc_archive_save_return",
        persistence_boundary="return_observed_fsync_not_instrumented",
        raw_path=obs.payload_path, raw_sha256=obs.content_hash, raw_observation_id=str(obs.id))
    if schema == "runtime-predictor-context-v3":
        timing.update(forecast_as_of=FORECAST, forecast_eligible=not late,
                      recording_scope="audit_only" if late else "pre_match_context")
    return timing


@pytest.mark.parametrize("schema", ["runtime-predictor-context-v2", "runtime-predictor-context-v3"])
def test_export_preserves_f_predictions_membership_and_market_separation(tmp_path, schema):
    folder = _batch(tmp_path)
    timings = {kind: archive_context(folder, kind=kind, schema=schema)
               for kind in ("initial", "after_action", "final")}
    config = json.loads((folder / "configuration.json").read_text("utf8"))
    config["policy_manifest"] = {"predictor_version": "f:frozen", "predictor_sha256": "a" * 64,
                                 "base_policy": config["policy_manifest"]}
    config["initial_state_timing"] = timings["initial"]
    (folder / "configuration.json").write_text(json.dumps(config), encoding="utf8")
    run = {"cycles": [{"decision_at": HANDOFF, "returned_at": HANDOFF, "generated_actions": [],
        "public_before": {"documents": []}, "after_state_timing": timings["after_action"]}],
        "final_state_timing": timings["final"], "terminal_selection": {"reason": "no_candidates"}}
    (folder / "run.json").write_text(json.dumps(run), encoding="utf8")
    originals = {p: p.read_bytes() for p in folder.rglob("*") if p.is_file()}
    target = tmp_path / "export"
    export_batch(folder.parent, target)
    rows = [json.loads(line) for line in (target / "context_snapshots.jsonl").read_text("utf8").splitlines()]
    for kind in timings:
        row = next(r for r in rows if r["snapshot_kind"] == kind)
        assert row["raw_context_timing_proven"] and row["raw_document_membership_proven"]
        assert row["context_capture_completion_proven"]
        assert row["prediction_before"] == (BEFORE if kind == "after_action" else None)
        assert row["prediction_after"] == (BEFORE if kind == "initial" else AFTER)
        assert row["prediction_after"] != row["market_prior"]
        assert row["prediction_basis"] == "immutable_predictor_context_payload"
        assert row["predictor_model_versions"] == ["f:frozen"]
        assert row["known_document_ids"] == ([] if kind == "initial" else ["body-1"])
        assert row["captured_at"] == SAVE_RETURN and row["available_at"] == HANDOFF
        if schema.endswith("v3"):
            assert row["forecast_as_of"] == row["prediction_cutoff"] == FORECAST
            assert row["forecast_eligible"] and row["recording_scope"] == "pre_match_context"
    assert all(path.read_bytes() == content for path, content in originals.items())


@pytest.mark.parametrize("tamper", ["run", "kind", "forecast", "digest"])
def test_unbound_v3_payload_cannot_prove_admission_or_supply_f(tmp_path, tamper):
    folder = _batch(tmp_path)
    timing = archive_context(folder, kind="after_action", schema="runtime-predictor-context-v3")
    run, kind = "run-1", "after_action"
    if tamper == "run": run = "other-run"
    if tamper == "kind": kind = "before_action"
    if tamper == "forecast": timing["forecast_as_of"] = HANDOFF
    if tamper == "digest": timing["raw_sha256"] = "0" * 64
    target = tmp_path / "export"
    target.mkdir()
    raw = _context_raw_fields(folder.parent, folder, target, timing, snapshot_kind=kind, run_id=run)
    assert not raw["raw_context_timing_proven"]
    assert _predictor_context_fields({}, _context_payload(target, raw), raw) == {}
    assert _document_timing_fields({"document_id": "body-1"}, {}, raw)["document_admitted"] is None


def test_v3_late_snapshot_remains_audit_only_with_earlier_forecast(tmp_path):
    folder = _batch(tmp_path)
    timing = archive_context(folder, kind="final", schema="runtime-predictor-context-v3", late=True)
    # The derived eligibility cannot be promoted by editing callback annotations.
    timing.update(recording_scope="pre_match_context", forecast_eligible=True)
    target = tmp_path / "export"
    target.mkdir()
    raw = _context_raw_fields(folder.parent, folder, target, timing, snapshot_kind="final", run_id="run-1")
    assert raw["raw_context_timing_proven"] and raw["context_capture_completion_proven"]
    assert raw["forecast_as_of"] == FORECAST
    assert raw["captured_at"] == "2026-10-09T19:00:03+00:00"
    assert raw["forecast_eligible"] is False and raw["recording_scope"] == "audit_only"


def test_unknown_schema_or_invalid_probabilities_do_not_supply_prediction():
    config = {"policy_manifest": {"predictor_version": "f:1", "predictor_sha256": "a" * 64}}
    raw = {"raw_context_timing_proven": True}
    payload = {"schema": "runtime-predictor-context-v3", "prediction_before": {"home": 5, "draw": 3, "away": 2}}
    assert _predictor_context_fields(config, payload, raw)["prediction_before"] is None
    payload.update(schema="runtime-predictor-context-v99", prediction_before=BEFORE)
    assert _predictor_context_fields(config, payload, raw) == {}
