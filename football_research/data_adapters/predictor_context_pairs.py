"""Verify v3 forecast pairs using immutable payloads and distinct event clocks.

A forecast's information time precedes its archive handoff. A before snapshot
can therefore be saved after the decision clock, but must be saved before I/O
and match that decision's archived menu. Old snapshot formats keep their own
rules; this adapter never manufactures a clock, propensity, cost, or outcome.
"""
from collections.abc import Mapping
from datetime import datetime
import json

SCHEMA = "runtime-predictor-context-v3"
PROCESSING_SCHEMA = "runtime-predictor-context-v4"
VERSION = "raw-bound-predictor-context-pair-v1"


def predictor_processing_valid(payload):
    """One v4 interpretation shared by raw projection and pair admission.

    A cached forecast can precede the current input clock, but cannot become
    effective until that input. A failed update has no effective forecast.
    """
    from football_platform.information.acquisition_journal import AcquisitionDecisionJournal
    pipeline = payload.get("prediction_pipeline_version")
    if not isinstance(pipeline, str) or not pipeline.strip():
        return False
    try:
        AcquisitionDecisionJournal._prediction_processing(payload.get("prediction_processing"),
            observed_at=payload.get("observed_at"), forecast_as_of=payload.get("forecast_as_of"))
        return True
    except (TypeError, ValueError, AttributeError):
        return False


def _time(value):
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return stamp if stamp.tzinfo and stamp.utcoffset() is not None else None
    except (TypeError, ValueError, AttributeError):
        return None


def _ordered(*values):
    stamps = [_time(value) for value in values]
    return all(x is not None for x in stamps) and all(a <= b for a, b in zip(stamps, stamps[1:]))


def _payload(root, row):
    # Reuse the existing RawArchive reader, including sidecar/path/hash checks.
    from .prospective_stream import _verified_archive_body
    if root is None:
        return {}
    try:
        body = _verified_archive_body(root, row.get("raw_path"), row.get("raw_sha256"))
        result = json.loads(body) if body is not None else {}
        return result if isinstance(result, dict) else {}
    except (OSError, EOFError, TypeError, ValueError):
        return {}


def is_predictor_context(row):
    return (row.get("raw_context_schema") in {SCHEMA, PROCESSING_SCHEMA, "runtime-predictor-context-v2"}
            or row.get("prediction_basis") == "immutable_predictor_context_payload")


def verify_predictor_pair(*, before, after, menu, attempt, fixture, root):
    """Audit a selected pair; callers must still enforce outcomes/costs/etc.

    Outer available_at retains the later *context handoff*. The distinct
    document projector available_at is read only from the hashed f input and
    is never backfilled into that outer field.
    """
    reasons = []
    raw_menu = _payload(root, menu)
    payloads = {"before": _payload(root, before), "after": _payload(root, after)}
    for name, payload in tuple(payloads.items()):
        if (not isinstance(payload.get("state"), Mapping)
                or not isinstance(payload["state"].get("base"), Mapping)
                or not isinstance(payload.get("documents"), list)
                or any(not isinstance(document, Mapping) for document in payload.get("documents", []))):
            reasons.append(name + "_predictor_payload_malformed")
            payloads[name] = {}
    decision = attempt.get("decision_at")
    run_id, step = raw_menu.get("run_id"), raw_menu.get("step")
    public = raw_menu.get("public_before") or {}
    public = public if isinstance(public, Mapping) else {}
    if (not isinstance(public.get("state"), Mapping) or not isinstance(public["state"].get("base"), Mapping)
            or not isinstance(public.get("prediction"), Mapping) or not isinstance(public.get("documents"), list)):
        public = {}
        reasons.append("predictor_pair_menu_public_state_malformed")
    if (raw_menu.get("schema") != "prospective-menu-archive-v1" or not isinstance(run_id, str)
            or type(step) is not int or step < 0
            or attempt.get("attempt_id") != f"{run_id}:attempt:{step}"
            or _time(raw_menu.get("observed_at")) != _time(decision)
            or _time(public.get("decision_at")) != _time(decision)):
        reasons.append("predictor_pair_menu_binding_invalid")
    menu_rows = raw_menu.get("menu") if isinstance(raw_menu.get("menu"), list) else []
    chosen = [row for row in menu_rows if isinstance(row, Mapping) and row.get("selected") is True]
    if len(chosen) != 1 or chosen[0].get("action_id") != attempt.get("action_id"):
        reasons.append("predictor_pair_selected_action_not_raw_bound")
    elif (chosen[0].get("selection_probability") != attempt.get("selection_probability")
            or (chosen[0].get("selection_probability") is not None
                and chosen[0].get("selection_probability_basis") != attempt.get("selection_probability_basis"))
            or raw_menu.get("policy_arm") != attempt.get("policy_arm")):
        reasons.append("predictor_pair_propensity_not_raw_bound")
    context_time_proven = {}
    for name, row in (("before", before), ("after", after)):
        p = payloads[name]
        prefix = name + "_predictor_"
        if p.get("schema") not in {SCHEMA, PROCESSING_SCHEMA} or row.get("raw_context_schema") != p.get("schema"):
            reasons.append(prefix + "schema_or_raw_unverified")
        if p.get("schema") == PROCESSING_SCHEMA:
            if not predictor_processing_valid(p):
                reasons.append(prefix + "processing_invalid")
            for key in ("prediction_pipeline_version", "prediction_processing"):
                if row.get(key) != p.get(key):
                    reasons.append(prefix + "processing_or_pipeline_not_raw_bound")
            if not isinstance(p.get("prediction_processing"), Mapping) or p["prediction_processing"].get("status") != "ready":
                reasons.append(prefix + "processing_not_ready")
        if (p.get("run_id") != run_id or p.get("snapshot_kind") != name + "_action"
                or p.get("state", {}).get("match_id") != public.get("state", {}).get("match_id")
                or p.get("state", {}).get("base") != public.get("state", {}).get("base")
                or _time(p.get("state", {}).get("base", {}).get("kickoff")) != _time(fixture.get("kickoff_at"))
                or p.get("parent_attempt_id") != (attempt.get("attempt_id") if name == "after" else None)):
            reasons.append(prefix + "run_or_attempt_binding_invalid")
        fields = ("observed_at", "available_at", "prediction_cutoff", "forecast_as_of", "acquisition_deadline")
        if any(_time(p.get(key)) is None or _time(p.get(key)) != _time(row.get(key)) for key in fields):
            reasons.append(prefix + "clock_not_raw_bound")
        if (p.get("availability_basis") != "runtime_state_handoff"
                or p.get("availability_status") != "known"
                or row.get("availability_basis") != p.get("availability_basis")
                or row.get("availability_status") != p.get("availability_status")
                or _time(p.get("observed_at")) != _time(p.get("available_at"))
                or _time(p.get("forecast_as_of")) != _time(p.get("prediction_cutoff"))):
            reasons.append(prefix + "clock_semantics_invalid")
        capture = row.get("captured_at")
        valid_time = (_ordered(p.get("forecast_as_of"), p.get("observed_at"),
                              row.get("archive_save_started_at"), row.get("archive_save_returned_at"),
                              p.get("acquisition_deadline"))
                      and _time(capture) == _time(row.get("archive_save_returned_at"))
                      and _time(capture) is not None and _time(fixture.get("kickoff_at")) is not None
                      and _time(capture) < _time(fixture["kickoff_at"])
                      and row.get("capture_timing_basis") == "measured_local_utc_archive_save_return"
                      and row.get("context_capture_completion_proven") is True
                      and row.get("forecast_eligible") is True and row.get("recording_scope") == "pre_match_context")
        # The sidecar binds the archive save start. The save return remains an
        # explicitly recorded local callback measurement, not a remote stamp.
        if root is not None and isinstance(row.get("raw_path"), str):
            path = (root / row["raw_path"]).resolve()
            sidecar = path.with_name(path.name.replace(".body.gz", ".observation.json"))
            try:
                observation = json.loads(sidecar.read_bytes()) if root.resolve() in sidecar.parents else {}
            except (OSError, ValueError):
                observation = {}
            valid_time = valid_time and _time(observation.get("observed_at")) == _time(row.get("archive_save_started_at"))
        else:
            valid_time = False
        context_time_proven[name] = bool(valid_time)
        if not valid_time:
            reasons.append(prefix + "save_time_or_deadline_unproven")
        for key in ("prediction_before", "prediction_after", "predictor_model_versions"):
            if row.get(key) != p.get(key):
                reasons.append(prefix + "prediction_or_model_not_raw_bound")
        if name == "before":
            forecast = (public.get("prediction") or {}).get("probabilities")
            if (p.get("state") != public.get("state") or p.get("documents") != public.get("documents")
                    or p.get("prediction_before") != (dict(zip(("home", "draw", "away"), forecast))
                                                       if isinstance(forecast, list) and len(forecast) == 3 else None)
                    or _time(p.get("forecast_as_of")) != _time(decision)
                    or not _ordered(capture, attempt.get("requested_at"))):
                reasons.append("before_predictor_not_this_pre_io_menu_state")
            if p.get("schema") == PROCESSING_SCHEMA and (
                    p.get("prediction_processing") != public.get("prediction_processing")
                    or p.get("prediction_pipeline_version") != (public.get("prediction") or {}).get("prediction_pipeline_version")):
                reasons.append("before_predictor_processing_or_pipeline_not_this_menu")
        elif (not _ordered(attempt.get("response_observed_at"), attempt.get("receipt_available_at"),
                           p.get("forecast_as_of"), p.get("observed_at"))
              or (p.get("schema") != PROCESSING_SCHEMA
                  and _time(attempt.get("receipt_available_at")) != _time(p.get("forecast_as_of")))
              or (p.get("schema") == PROCESSING_SCHEMA and (not isinstance(p.get("prediction_processing"), Mapping)
                  or _time(attempt.get("receipt_available_at")) != _time(p["prediction_processing"].get("input_at"))))):
            reasons.append("after_predictor_not_this_return_state")
    b, a = payloads["before"], payloads["after"]
    processing_pair = any(p.get("schema") == PROCESSING_SCHEMA for p in payloads.values())
    if processing_pair and (b.get("schema") != PROCESSING_SCHEMA or a.get("schema") != PROCESSING_SCHEMA
            or b.get("prediction_pipeline_version") != a.get("prediction_pipeline_version")):
        reasons.append("predictor_pair_pipeline_or_processing_schema_changed")
    versions = b.get("predictor_model_versions")
    if (not isinstance(versions, list) or len(versions) != 1 or not isinstance(versions[0], str) or not versions[0]
            or versions != a.get("predictor_model_versions") or b.get("prediction_before") != a.get("prediction_before")
            or (public.get("prediction") or {}).get("model_version") != versions[0]):
        reasons.append("predictor_pair_model_or_before_prediction_changed")
    doc_id = attempt.get("returned_document_id")
    inputs = [doc for doc in a.get("documents", []) if isinstance(doc, Mapping) and doc.get("document_id") == doc_id]
    # A zero/failed return is allowed to have no document. A claimed document
    # needs one exact member with its original projection clock in the f input.
    document_input = inputs[0] if doc_id is not None and len(inputs) == 1 else None
    if doc_id is not None and document_input is None:
        reasons.append("predictor_returned_document_not_unique_in_f_input")
    if document_input is not None and not _ordered(document_input.get("observed_at"),
            document_input.get("available_at"), attempt.get("receipt_available_at"), a.get("forecast_as_of")):
        reasons.append("predictor_document_input_after_forecast_or_unknown")
    retained = [doc for doc in a.get("documents", []) if doc is not document_input]
    if retained != b.get("documents"):
        reasons.append("predictor_pair_unexplained_document_change")
    for name, payload in payloads.items():
        docs = payload.get("documents", [])
        ids = [doc.get("document_id") for doc in docs]
        if (any(not isinstance(did, str) or not did for did in ids) or len(set(ids)) != len(ids)
                or any(not _ordered(doc.get("observed_at"), doc.get("available_at"), payload.get("forecast_as_of"))
                       for doc in docs)):
            reasons.append(name + "_predictor_input_visibility_invalid")
    return {"schema": "raw-bound-predictor-context-pair-v2" if processing_pair else VERSION,
        **({"prediction_pipeline_version": b.get("prediction_pipeline_version")} if processing_pair else {}),
        "valid": not reasons, "reasons": sorted(set(reasons)),
        "context_time_proven": context_time_proven, "predictor_model_versions": versions,
        "before_snapshot_id": before.get("snapshot_id"), "after_snapshot_id": after.get("snapshot_id"),
        "document_input": document_input,
        "scope": "recorded_same_f_pair_only; outcomes_costs_propensities_require_separate_checks"}
