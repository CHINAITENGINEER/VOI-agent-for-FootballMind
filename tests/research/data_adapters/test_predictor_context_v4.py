"""Constructed v4 processing clocks; zero HTTP and no empirical evidence."""
from copy import deepcopy
from datetime import datetime
import gzip
import json

import pytest

from football_research.data_adapters.predictor_context_pairs import verify_predictor_pair
from football_research.data_adapters.prospective_stream import (
    _context_raw_fields, _context_payload, _predictor_context_fields, export_batch,
)
from football_spider.storage.raw_archive import RawArchive
from tests.research.data_adapters.test_predictor_context_pairs import pair, audit, derive, stamp
from tests.research.data_adapters.test_prospective_stream_contexts import _batch


SCHEMA = "runtime-predictor-context-v4"
PIPELINE = "entity-market-f:constructed-frozen-pipeline"


def processing(input_at, available_at, *, ready=True):
    return {"schema": "entity-semantic-prediction-processing-v1",
        "status": "ready" if ready else "not_updated", "reason": None if ready else "semantic_extraction_failed",
        "input_at": input_at, "available_at": available_at,
        "effective_at": max(input_at, available_at) if ready else None}


def rewrite(p, name, payload):
    p[name].update(payload)
    p[name].update(p["raw"](payload, p[name]["archive_save_started_at"]))
    p["payloads"][name] = deepcopy(payload)


@pytest.fixture
def v4_pair(pair):
    for name in ("before", "after"):
        p = deepcopy(pair["payloads"][name])
        p.update(schema=SCHEMA, prediction_pipeline_version=PIPELINE,
            prediction_processing=processing(stamp(0 if name == "before" else 10), stamp(0 if name == "before" else 12)))
        pair[name]["raw_context_schema"] = SCHEMA
        if name == "after":
            p.update(forecast_as_of=stamp(12), prediction_cutoff=stamp(12), observed_at=stamp(13), available_at=stamp(13))
            pair[name].update(captured_at=stamp(15), archive_save_started_at=stamp(14), archive_save_returned_at=stamp(15))
        rewrite(pair, name, p)
    menu = json.loads(gzip.decompress((pair["root"] / pair["menu"]["raw_path"]).read_bytes()))
    menu["public_before"]["prediction"]["prediction_pipeline_version"] = PIPELINE
    menu["public_before"]["prediction_processing"] = deepcopy(pair["payloads"]["before"]["prediction_processing"])
    pair["menu"].update(pair["raw"](menu, stamp(0)))
    pair["menu_payload"] = menu
    pair["document"]["available_at"] = pair["after"]["available_at"]
    return pair


def test_v4_accepts_real_processing_delay_but_v3_equal_receipt_rule_remains(pair, v4_pair):
    result = audit(v4_pair)
    assert result["valid"], result
    assert result["schema"] == "raw-bound-predictor-context-pair-v2"
    assert result["prediction_pipeline_version"] == PIPELINE
    derived = derive(v4_pair)
    assert derived["label_status"] == "eligible", derived.get("exclusion_reason")
    for name in ("before", "after"):
        payload = deepcopy(v4_pair["payloads"][name])
        payload["schema"] = "runtime-predictor-context-v3"
        v4_pair[name]["raw_context_schema"] = payload["schema"]
        rewrite(v4_pair, name, payload)
    assert not audit(v4_pair)["valid"]
    assert "after_predictor_not_this_return_state" in audit(v4_pair)["reasons"]


@pytest.mark.parametrize("which", ["before", "after"])
def test_v4_cached_processing_can_complete_before_current_input(v4_pair, which):
    p = deepcopy(v4_pair["payloads"][which])
    p["prediction_processing"] = processing(stamp(0 if which == "before" else 10),
        "2026-10-09T08:59:59+00:00" if which == "before" else stamp(8))
    p["forecast_as_of"] = p["prediction_cutoff"] = p["prediction_processing"]["effective_at"]
    rewrite(v4_pair, which, p)
    if which == "before":
        menu = deepcopy(v4_pair["menu_payload"])
        menu["public_before"]["prediction_processing"] = deepcopy(p["prediction_processing"])
        v4_pair["menu"].update(v4_pair["raw"](menu, stamp(0)))
    assert audit(v4_pair)["valid"], audit(v4_pair)


@pytest.mark.parametrize("case", ["input_not_receipt", "effective_not_completion", "future_completion", "missing_processing",
    "extra_processing_field", "wrong_processing_schema", "not_updated", "false_ready_reason", "pipeline_changed",
    "outer_pipeline", "outer_processing", "menu_pipeline", "menu_processing", "mixed_schema", "late_capture"])
def test_v4_failures_never_generate_gain(v4_pair, case):
    row = v4_pair["after"]
    p = deepcopy(v4_pair["payloads"]["after"])
    if case == "input_not_receipt": p["prediction_processing"]["input_at"] = stamp(9)
    elif case == "effective_not_completion": p["prediction_processing"]["available_at"] = stamp(11)
    elif case == "future_completion": p["prediction_processing"]["available_at"] = stamp(14)
    elif case == "missing_processing": p["prediction_processing"] = None
    elif case == "extra_processing_field": p["prediction_processing"]["guessed"] = True
    elif case == "wrong_processing_schema": p["prediction_processing"]["schema"] = "unknown-v1"
    elif case == "not_updated":
        p["prediction_processing"] = processing(stamp(10), stamp(12), ready=False)
        p["forecast_as_of"] = p["prediction_cutoff"] = stamp(0)
        p["prediction_after"] = deepcopy(p["prediction_before"])
        row.update(forecast_eligible=False, recording_scope="audit_only")
    elif case == "false_ready_reason": p["prediction_processing"]["reason"] = "error"
    elif case == "pipeline_changed": p["prediction_pipeline_version"] = "entity-market-f:different"
    elif case == "outer_pipeline": row["prediction_pipeline_version"] = "forged"
    elif case == "outer_processing": row["prediction_processing"] = processing(stamp(10), stamp(11))
    elif case in {"menu_pipeline", "menu_processing"}:
        menu = deepcopy(v4_pair["menu_payload"])
        if case == "menu_pipeline": menu["public_before"]["prediction"]["prediction_pipeline_version"] = "other"
        else: menu["public_before"]["prediction_processing"]["available_at"] = stamp(1)
        v4_pair["menu"].update(v4_pair["raw"](menu, stamp(0)))
    elif case == "mixed_schema":
        p["schema"] = "runtime-predictor-context-v3"
        row["raw_context_schema"] = p["schema"]
    elif case == "late_capture":
        row["captured_at"] = row["archive_save_returned_at"] = v4_pair["fixture"]["kickoff_at"]
    if p != v4_pair["payloads"]["after"]:
        rewrite(v4_pair, "after", p)
    result = derive(v4_pair)
    assert not result["predictor_pair_audit"]["valid"]
    assert result["label_status"] == "ineligible"
    assert result["realized_gain"] is None and result["net_gain"] is None


def save_context(folder, *, kind="after_action", ready=True, late=False):
    deadline = "2026-10-09T18:59:00+00:00"
    kickoff = "2026-10-09T19:00:00+00:00"
    forecast = stamp(12) if ready else stamp(0)
    handed = stamp(13) if not late else "2026-10-09T19:00:01+00:00"
    started = stamp(14) if not late else "2026-10-09T19:00:02+00:00"
    saved = stamp(15) if not late else "2026-10-09T19:00:03+00:00"
    p = {"schema": SCHEMA, "run_id": "run-1", "snapshot_kind": kind,
        "state": {"match_id": "m-1", "base": {"kickoff": kickoff}}, "documents": [],
        "observed_at": handed, "available_at": handed, "availability_status": "known",
        "availability_basis": "runtime_state_handoff", "prediction_cutoff": forecast, "forecast_as_of": forecast,
        "acquisition_deadline": deadline, "parent_attempt_id": None,
        "prediction_before": {"home": .5, "draw": .3, "away": .2},
        "prediction_after": {"home": .5, "draw": .3, "away": .2},
        "predictor_model_versions": ["f:one"], "prediction_pipeline_version": PIPELINE,
        "prediction_processing": processing(stamp(10), stamp(12), ready=ready)}
    observation = RawArchive(folder / "source-archive").save("agent_trajectory", "agent://constructed-state", {},
        json.dumps(p).encode(), status=200, observed_at=datetime.fromisoformat(started))
    timing = {key: deepcopy(p[key]) for key in ("observed_at", "available_at", "availability_status", "availability_basis",
        "prediction_cutoff", "parent_attempt_id", "forecast_as_of", "prediction_pipeline_version", "prediction_processing")}
    timing.update(captured_at=saved, archive_save_started_at=started, archive_save_returned_at=saved,
        capture_timing_basis="measured_local_utc_archive_save_return", persistence_boundary="return_observed_fsync_not_instrumented",
        raw_path=observation.payload_path, raw_sha256=observation.content_hash, raw_observation_id=str(observation.id))
    return timing


@pytest.mark.parametrize("ready,late", [(True, False), (False, False), (True, True)])
def test_v4_export_preserves_processing_and_failure_lateness_are_audit_only(tmp_path, ready, late):
    folder = _batch(tmp_path)
    config = json.loads((folder / "configuration.json").read_text("utf8"))
    config["policy_manifest"] = {"predictor_version": "f:one", "predictor_sha256": "a" * 64,
        "base_policy": config["policy_manifest"]}
    timings = {kind: save_context(folder, kind=kind, ready=ready, late=late) for kind in ("initial", "after_action", "final")}
    config["initial_state_timing"] = timings["initial"]
    (folder / "configuration.json").write_text(json.dumps(config), encoding="utf8")
    run = {"cycles": [{"decision_at": stamp(0), "returned_at": stamp(10), "generated_actions": [],
        "public_before": {"documents": []}, "after_state_timing": timings["after_action"]}],
        "final_state_timing": timings["final"], "terminal_selection": {"reason": "no_candidates"}}
    (folder / "run.json").write_text(json.dumps(run), encoding="utf8")
    original = {p: p.read_bytes() for p in folder.rglob("*") if p.is_file()}
    target = tmp_path / "export"
    export_batch(folder.parent, target)
    rows = [json.loads(line) for line in (target / "context_snapshots.jsonl").read_text("utf8").splitlines()]
    for kind in timings:
        row = next(row for row in rows if row["snapshot_kind"] == kind)
        assert row["raw_context_timing_proven"] and row["raw_context_schema"] == SCHEMA
        assert row["prediction_pipeline_version"] == PIPELINE
        assert row["prediction_processing"] == timings[kind]["prediction_processing"]
        assert row["forecast_eligible"] is (ready and not late)
        assert row["recording_scope"] == ("pre_match_context" if ready and not late else "audit_only")
    assert all(path.read_bytes() == content for path, content in original.items())


@pytest.mark.parametrize("tamper", ["pipeline", "processing"])
def test_unbound_v4_processing_cannot_supply_predictions(tmp_path, tamper):
    folder = _batch(tmp_path)
    timing = save_context(folder)
    if tamper == "pipeline": timing["prediction_pipeline_version"] = "other"
    else: timing["prediction_processing"]["available_at"] = stamp(11)
    target = tmp_path / "export"
    target.mkdir()
    raw = _context_raw_fields(folder.parent, folder, target, timing, snapshot_kind="after_action", run_id="run-1")
    assert not raw["raw_context_timing_proven"]
    assert _predictor_context_fields({}, _context_payload(target, raw), raw) == {}
