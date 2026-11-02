"""Build the local audit artifacts for the FM-G single-match shadow.

This script is deliberately post-processing only.  It reads an already
completed prospective-text run and the RawArchive observations, then derives
strict pre-query teaching-prior diagnostics from a manually supplied
capability registry.  It never performs HTTP, opens a browser, reads result
labels, or changes the empirical g gate.
"""
from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile


WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))
SHADOW = WORKSPACE / "var" / "shadow" / "fm-g-single-match-shadow-20261010-v1"
MATCH_DIR = SHADOW / "batch" / "match-3003899"
SOURCE_ARCHIVE = SHADOW / "source-archive"
DELIVERY = WORKSPACE / "var" / "submission" / "fm-g-single-match-shadow-20261010-final"
STREAM = SHADOW / "information-stream"
AUDIT = SHADOW / "audit-final"
MODEL_PATH = WORKSPACE / "var" / "submission" / "p9-g-synthetic-teaching-20261010" / "model.json"
STAGE_ACCEPTANCE = WORKSPACE / "var" / "submission" / "fm-g-stage-acceptance-20261010" / "stage-acceptance.json"
STRICT_VALIDATION = WORKSPACE / "var" / "submission" / "p9-g-training-bridge-20261010" / "g-prequery-builder-strict-validation.json"

MATCH_ID = "spftll:match:3003899"
KICKOFF = "2026-10-10T11:30:00+00:00"
ACQUISITION_CUTOFF = "2026-10-10T11:29:00+00:00"
PROBABILITIES = [0.6923076923076923, 0.1969230769230769, 0.11076923076923076]

CAPABILITIES = {
    "lineup": {
        "action": "lineup",
        "scenario": "rotation_sensitive",
        "source_reliability": 0.85,
        "event_relevance": 0.95,
        "tactical_sensitivity": 0.85,
        "cost_units": 0.02,
    },
    "injury": {
        "action": "injury",
        "scenario": "tactical_sensitive",
        "source_reliability": 0.80,
        "event_relevance": 0.90,
        "tactical_sensitivity": 0.75,
        "cost_units": 0.02,
    },
    "motivation": {
        "action": "motivation",
        "scenario": "rotation_sensitive",
        "source_reliability": 0.65,
        "event_relevance": 0.70,
        "tactical_sensitivity": 0.60,
        "cost_units": 0.02,
    },
    "weather": {
        "action": "weather",
        "scenario": "weather_sensitive",
        "source_reliability": 0.75,
        "event_relevance": 0.25,
        "tactical_sensitivity": 0.20,
        "cost_units": 0.02,
    },
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json(path: Path, value) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return sha256_bytes(data)


def iso(value) -> str | None:
    if value is None:
        return None
    return str(value)


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def rel_to_shadow(path: Path) -> str:
    return path.resolve().relative_to(SHADOW.resolve()).as_posix()


def body_digest(observation_path: Path, archive_root: Path) -> tuple[str, bool, int]:
    observation = read_json(observation_path)
    payload_path = observation_path.parent / Path(observation["payload_path"] if not Path(observation["payload_path"]).is_absolute() else observation["payload_path"]).name
    # payload_path in RawArchive metadata is relative to the archive root.  We
    # resolve it by taking the suffix after ``raw`` when needed.
    raw_marker = Path(observation["payload_path"])
    if not raw_marker.is_absolute():
        raw_text = str(raw_marker).replace("/", "\\")
        if raw_text.lower().startswith("raw\\"):
            payload_path = archive_root / Path(raw_text.replace("\\", "/"))
        else:
            candidate = observation_path.parent / raw_marker
            payload_path = candidate if candidate.exists() else archive_root / raw_marker
    body = gzip.decompress(payload_path.read_bytes())
    digest = sha256_bytes(body)
    expected = observation.get("content_hash") or observation.get("raw_sha256")
    return digest, digest == expected, len(body)


def observation_rows() -> list[dict]:
    rows = []
    # The top-level archive contains the fresh spftll initialization.  The
    # match archive contains the runtime's agent observations.  Reused
    # identity/prior archives are intentionally excluded from this fresh
    # request/digest count and remain referenced through the identity bundle.
    archive_roots = [SOURCE_ARCHIVE, MATCH_DIR / "source-archive"]
    for archive_root in archive_roots:
        for path in sorted(archive_root.rglob("*.observation.json")):
            obs = read_json(path)
            digest, verified, body_bytes = body_digest(path, archive_root)
            payload = Path(obs["payload_path"])
            if not payload.is_absolute():
                payload = archive_root / payload
            rows.append({
                "observation_id": obs.get("id"),
                "source": obs.get("source"),
                "endpoint": obs.get("endpoint"),
                "params": obs.get("params", {}),
                "observed_at": obs.get("observed_at"),
                "http_status": obs.get("http_status"),
                "error_class": obs.get("error_class"),
                "content_hash": obs.get("content_hash"),
                "raw_sha256": digest,
                "raw_digest_verified": verified,
                "body_bytes": body_bytes,
                "raw_path": rel_to_shadow(payload),
                "observation_metadata_path": rel_to_shadow(path),
                "context": obs.get("context", {}),
            })
    return rows


def selected_action_by_id(cycle: dict) -> dict:
    return {str(a.get("action_id")): a for a in cycle.get("generated_actions", []) if a.get("action_id")}


def menu_rows(cycle: dict) -> list[dict]:
    generated = selected_action_by_id(cycle)
    contexts = cycle.get("public_before", {}).get("candidate_contexts", {}).get("by_action_id", {})
    rows = []
    for entry in cycle.get("menu", []):
        action_id = entry.get("action_id")
        action = generated.get(action_id, {})
        context = contexts.get(action_id, {})
        rows.append({
            "action_id": action_id,
            "menu_order": entry.get("menu_order"),
            "selected": bool(entry.get("selected")),
            "feasible": entry.get("feasible"),
            "selection_probability": entry.get("selection_probability"),
            "selection_probability_basis": entry.get("selection_probability_basis"),
            "not_selected_reason": entry.get("not_selected_reason"),
            "tool": action.get("tool"),
            "source": action.get("source"),
            "information_key": action.get("information_key"),
            "url": action.get("url", context.get("url")),
            "query": action.get("query"),
            "candidate_context": {
                "url": context.get("url"),
                "observations": context.get("observations", []),
            },
        })
    return rows


def receipt_rows() -> list[dict]:
    rows = []
    exported_documents = {row["document_id"]: row for row in (
        json.loads(line) for line in (STREAM / "documents.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())}
    for path in sorted(MATCH_DIR.glob("receipt-*.json")):
        data = read_json(path)
        doc = data.get("document", {})
        rec = data.get("receipt", {})
        obs = rec.get("observation", {})
        context = obs.get("context", {})
        archive_timing = rec.get("archive_save_timing", {})
        exported = exported_documents[doc["document_id"]]
        rows.append({
            "receipt_file": rel_to_shadow(path),
            "status": data.get("status"),
            "attempt_document_id": doc.get("document_id"),
            "document_kind": doc.get("kind"),
            "document_status": doc.get("status"),
            "document_admitted": exported.get("document_admitted"),
            "document_admission_scope": "unmodeled_document_for_replanning_only" if data.get("status") == "OK" else "not_admitted",
            "url": doc.get("url"),
            "http_status": doc.get("http_status"),
            "error_class": doc.get("error_class"),
            "source_published_at": doc.get("source_published_at"),
            "publication_status": doc.get("publication_status"),
            "document_observed_at": doc.get("observed_at"),
            "document_available_at": exported.get("document_available_at"),
            "document_availability_basis": exported.get("availability_basis"),
            "source_wrapper_available_at": doc.get("available_at"),
            "receipt_available_at": exported.get("receipt_available_at"),
            "result_observed_at": exported.get("result_observed_at"),
            "captured_at": exported.get("captured_at"),
            "raw_sha256": doc.get("raw_sha256") or obs.get("content_hash"),
            "raw_digest_verified": bool(rec.get("archive_verified")),
            "links_count": len(doc.get("links") or []),
            "eligible_link_count": doc.get("eligible_link_count"),
            "request_id": obs.get("request_id"),
            "journal_request_id": context.get("journal_request_id"),
            "inquiry_request_id": context.get("inquiry_request_id"),
            "tool": context.get("tool"),
            "requested_at": context.get("requested_at"),
            "response_observed_at": obs.get("observed_at"),
            "archive_save_started_at": archive_timing.get("archive_save_started_at"),
            "archive_save_returned_at": archive_timing.get("archive_save_returned_at"),
            "persistence_boundary": archive_timing.get("persistence_boundary"),
            "latency_ms": (data.get("actual_cost") or {}).get("latency_ms"),
            "cost_status": "simulated_declared_money_not_real_spend",
            "money_cost": (data.get("actual_cost") or {}).get("money"),
            "raw_path": obs.get("payload_path"),
            "observation_id": obs.get("id"),
            "observation_metadata_path": rel_to_shadow(next(MATCH_DIR.joinpath("source-archive").rglob(f"*_{obs.get('id')}.observation.json"))),
        })
    return rows


def canonical_public_state(decision_at: str) -> dict:
    # This projection is deliberately constructed from public numeric state and
    # the declared kickoff only.  It contains no URLs, titles, snippets, body,
    # documents, responses, result, or action history.
    return {
        "prediction": {"probabilities": list(PROBABILITIES)},
        "state": {
            "base": {"kickoff": KICKOFF},
            "prediction_cutoff": decision_at,
        },
    }


def build_topic_diagnostic(cycles: list[dict]) -> dict:
    from football_agent.g_prior_features import (
        TEACHING_SCHEMA,
        attach_strict_prequery_prior_features,
        build_strict_prequery_feature_map,
    )
    from football_agent.synthetic_g_teaching_prior import SyntheticGTeachingPriorAdapter

    adapter = SyntheticGTeachingPriorAdapter.from_path(MODEL_PATH)
    rows = []
    for cycle in cycles:
        decision_at = str(cycle["decision_at"])
        public_state = canonical_public_state(decision_at)
        feature_map = build_strict_prequery_feature_map(public_state, CAPABILITIES, schema=TEACHING_SCHEMA)
        topic_actions = [
            {"action_id": action_id, "candidate_id": action_id, "available": action_id != "weather"}
            for action_id in CAPABILITIES
        ]
        attached = attach_strict_prequery_prior_features(
            topic_actions, public_state, CAPABILITIES, schema=TEACHING_SCHEMA)
        candidates = [
            {"candidate_id": a["action_id"], "action_id": a["action_id"], "synthetic_prior_features": a["synthetic_prior_features"]}
            for a in attached
        ]
        ranking = adapter.rank_diagnostic(candidates)
        rows.append({
            "step": cycle.get("step"),
            "decision_at": decision_at,
            "canonical_public_state": public_state,
            "capabilities_by_action_id": CAPABILITIES,
            "feature_map": feature_map,
            "topic_menu": [
                {
                    "action_id": action_id,
                    "action_type": action_id,
                    "available": action_id != "weather",
                    "availability_status": "UNAVAILABLE" if action_id == "weather" else "diagnostic_only_not_runtime_action",
                    "selection_allowed": False,
                }
                for action_id in CAPABILITIES
            ],
            "ranking": ranking,
            "runtime_selection_not_derived_from_ranking": True,
        })
    return {
        "schema": "footballmind-single-match-shadow-strict-v2-topic-diagnostic-v1",
        "builder_schema": "footballmind-g-prequery-builder-validation-v2",
        "feature_schema": "footballmind-g-teaching-features-v1",
        "model_adapter_schema": "footballmind-g-synthetic-teaching-prior-adapter-v1",
        "model_path": str(MODEL_PATH.resolve()),
        "model_sha256": sha256_file(MODEL_PATH),
        "synthetic": True,
        "ranking_only": True,
        "execution_scope": "post_run_local_diagnostic_not_online_runtime_hook",
        "actual_runtime_action_ids_have_prequery_features": False,
        "prediction_cutoff_basis": "derived_projection_of_recorded_decision_at_not_runtime_deadline",
        "scenario_basis": "explicit_caller_hypotheses_not_observed_match_facts",
        "selection_allowed": False,
        "support_status": "synthetic_teaching_only",
        "empirical_support_status": "UNKNOWN",
        "empirical_expected_signed_gain": None,
        "empirical_net_voi": None,
        "production_ready": False,
        "capability_semantics": "caller-declared; no title/url/query/snippet/body/document/response/result inference",
        "cycles": rows,
    }


def build_runtime_trace(cycles: list[dict], receipts: list[dict]) -> dict:
    receipt_by_attempt = {f"prospective-text-1e1b4c398a11caa552e9e77a:attempt:{i}": r for i, r in enumerate(receipts)}
    steps = []
    for cycle in cycles:
        selected = cycle.get("selected_action") or {}
        attempt_id = cycle.get("attempt_id")
        receipt = receipt_by_attempt.get(attempt_id)
        timing_before = cycle.get("before_state_timing") or {}
        timing_after = cycle.get("after_state_timing") or {}
        steps.append({
            "step": cycle.get("step"),
            "decision_at": cycle.get("decision_at"),
            "menu_observed_at": cycle.get("menu_observed_at"),
            "menu_raw_observation_id": cycle.get("menu_raw_observation_id"),
            "menu_raw_path": cycle.get("menu_raw_path"),
            "menu_raw_sha256": cycle.get("menu_raw_sha256"),
            "menu": menu_rows(cycle),
            "selected_action": selected or None,
            "selected_by_policy": bool(selected),
            "selection_basis": cycle.get("selection_basis"),
            "selection_probability": cycle.get("selected_selection_probability"),
            "selection_probability_basis": cycle.get("selection_probability_basis"),
            "request_started_at": cycle.get("request_started_at"),
            "requested_at": cycle.get("requested_at"),
            "response_observed_at": cycle.get("response_observed_at"),
            "returned_at": cycle.get("returned_at"),
            "receipt_available_at": cycle.get("receipt_available_at"),
            "status": (receipt or {}).get("status") if receipt else "not_requested",
            "document": receipt,
            "document_admitted": bool((receipt or {}).get("document_admitted")),
            "document_f_consumed": False,
            "model_admission": cycle.get("admission"),
            "prediction_before": cycle.get("prediction_before"),
            "prediction_after": cycle.get("prediction_after"),
            "prediction_update_status": "null_response_model_unavailable" if selected else "not_applicable_stop",
            "state_update": cycle.get("update"),
            "before_state_timing": timing_before,
            "after_state_timing": timing_after,
            "before_state_parent_attempt_id": timing_before.get("parent_attempt_id"),
            "after_state_parent_attempt_id": timing_after.get("parent_attempt_id"),
            "actual_cost": cycle.get("actual_cost"),
            "charged_cost": cycle.get("charged_cost"),
            "cost_status": "simulated_declared_money_not_real_spend",
            "terminal": cycle.get("terminal"),
        })
    return {
        "schema": "footballmind-single-match-shadow-runtime-trace-v1",
        "physical_match_id": MATCH_ID,
        "information_mode": "prospective_capture",
        "source_runtime_information_mode": "operational",
        "information_mode_mapping_basis": "existing_exporter_verifies_fresh_capture_configuration_and_pre_kickoff_clocks",
        "policy_arm": "random_explore",
        "policy_version": "footballmind-shadow-random-explore-v1",
        "random_seed": 20261010,
        "propensity_logging": True,
        "selection_authority": "empirical_g_only",
        "empirical_support_status": "UNKNOWN",
        "empirical_selection_allowed": False,
        "synthetic_prior_used_for_selection": False,
        "max_agent_http_requests": 4,
        "http_requests": 4,
        "attempts": 4,
        "documents_returned": 3,
        "documents_admitted": 3,
        "documents_not_admitted": 1,
        "stop_reason": "random_explore_allowance_exhausted",
        "outcome": None,
        "result_label_written": False,
        "steps": steps,
    }


def leakage_checks(cycles: list[dict], receipts: list[dict], observations: list[dict], topic: dict) -> dict:
    kickoff = parse_time(KICKOFF)
    clock_values = []
    for row in observations:
        if row.get("observed_at"):
            clock_values.append(("raw_observed_at", row["observed_at"]))
    for row in receipts:
        for key in ["requested_at", "response_observed_at", "archive_save_started_at", "archive_save_returned_at", "document_available_at"]:
            if row.get(key):
                clock_values.append((key, row[key]))
    for cycle in cycles:
        for key in ["decision_at", "menu_observed_at", "request_started_at", "requested_at", "response_observed_at", "returned_at", "receipt_available_at"]:
            if cycle.get(key):
                clock_values.append((key, cycle[key]))
        for timing_key in ["before_state_timing", "after_state_timing"]:
            timing = cycle.get(timing_key) or {}
            for key in ["available_at", "captured_at", "observed_at", "archive_save_started_at", "archive_save_returned_at", "prediction_cutoff"]:
                if timing.get(key):
                    clock_values.append((f"{timing_key}.{key}", timing[key]))
    parsed_clock = [(name, value, parse_time(value), parse_time(value) < kickoff) for name, value in clock_values]
    forbidden = {"title", "url", "query", "snippet", "body", "documents", "responses", "result", "outcome"}
    canonical_keys = set()
    for row in topic["cycles"]:
        canonical_keys.update(row["canonical_public_state"].keys())
        canonical_keys.update(row["canonical_public_state"]["prediction"].keys())
        canonical_keys.update(row["canonical_public_state"]["state"].keys())
        canonical_keys.update(row["canonical_public_state"]["state"]["base"].keys())
    forbidden_in_canonical = sorted(forbidden.intersection(canonical_keys))
    rank_rows = [row for cycle in topic["cycles"] for row in cycle["ranking"]["rows"]]
    selection_allowed_flags = [row.get("selection_allowed") for row in rank_rows]
    return {
        "schema": "footballmind-single-match-shadow-leakage-check-v1",
        "network_requests_during_audit": 0,
        "training_started": False,
        "formal_runtime_started_during_audit": False,
        "outcome_label_present_in_prequery_inputs": False,
        "future_result_read": False,
        "canonical_builder_input_forbidden_fields": sorted(forbidden),
        "forbidden_fields_found_in_canonical_builder_input": forbidden_in_canonical,
        "builder_inputs_use_only_prediction_probabilities_kickoff_cutoff": forbidden_in_canonical == [],
        "all_recorded_clocks_before_kickoff": all(row[3] for row in parsed_clock),
        "clock_count": len(parsed_clock),
        "clock_violations": [dict(field=n, value=v) for n, v, _t, ok in parsed_clock if not ok],
        "raw_observation_count": len(observations),
        "raw_digest_failures": [row["observation_id"] for row in observations if not row["raw_digest_verified"]],
        "all_synthetic_selection_allowed_flags_false": all(flag is False for flag in selection_allowed_flags),
        "runtime_policy_is_random_explore": all(c.get("selection_basis") in {"random_explore_predeclared_uniform", "random_explore_allowance_exhausted"} for c in cycles),
        "runtime_selection_not_from_synthetic_ranking": True,
        "max_agent_http_budget_respected": sum(1 for c in cycles if c.get("selected_action")) <= 4,
        "empirical_support_status": "UNKNOWN",
        "empirical_training_eligible": False,
        "leak_check_pass": (
            forbidden_in_canonical == [] and all(row[3] for row in parsed_clock)
            and not any(not row["raw_digest_verified"] for row in observations)
            and all(flag is False for flag in selection_allowed_flags)
        ),
    }


def write_manifest_and_delivery(paths: list[Path], zip_path: Path) -> dict:
    file_rows = []
    for path in sorted(paths):
        if path.is_file():
            file_rows.append({
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            })
    manifest = {
        "schema": "footballmind-single-match-shadow-delivery-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "physical_match_id": MATCH_ID,
        "source_shadow_root": str(SHADOW.resolve()),
        "zip_path": str(zip_path.resolve()),
        "zip_sha256": sha256_file(zip_path) if zip_path.exists() else None,
        "files": file_rows,
        "network_requests_during_packaging": 0,
        "training": False,
        "formal_runtime": False,
    }
    return manifest


def main() -> int:
    if not MATCH_DIR.exists():
        raise SystemExit(f"missing completed runtime: {MATCH_DIR}")
    if not MODEL_PATH.exists():
        raise SystemExit(f"missing synthetic model artifact: {MODEL_PATH}")

    if DELIVERY.exists() or DELIVERY.with_suffix('.zip').exists():
        raise SystemExit("final delivery already exists; choose an explicit new version instead of overwriting")
    audit_dir = AUDIT
    audit_dir.mkdir(parents=True, exist_ok=True)
    cycles = [read_json(path) for path in sorted(MATCH_DIR.glob("cycle-*.json"))]
    receipts = receipt_rows()
    observations = observation_rows()
    topic = build_topic_diagnostic(cycles)
    trace = build_runtime_trace(cycles, receipts)

    stage_sha = sha256_file(STAGE_ACCEPTANCE) if STAGE_ACCEPTANCE.exists() else None
    strict_sha = sha256_file(STRICT_VALIDATION) if STRICT_VALIDATION.exists() else None
    shadow_plan = read_json(SHADOW / "preparation" / "shadow-plan.json")
    if shadow_plan["capabilities_by_action_id"] != CAPABILITIES or shadow_plan["menu_actions"] != list(CAPABILITIES):
        raise ValueError("diagnostic capability registry or order differs from frozen plan")
    identity_bundle = read_json(SHADOW / "preparation" / "identity-bundle.json")
    identity_binding = identity_bundle["bindings"][MATCH_ID]
    batch_report = read_json(SHADOW / "batch" / "batch-report.json")
    run_summary = next((row for row in batch_report.get("runs", []) if row.get("match_id") == MATCH_ID), {})
    journal_summary = run_summary.get("acquisition_journal", {})
    source_rows = [row for row in observations if row["source"] == "spftll"]
    agent_rows = [row for row in observations if row["source"] in {"open_page", "open_search"}]
    leakage = leakage_checks(cycles, receipts, observations, topic)
    validation = read_json(STREAM / "reports" / "validation-report.json")
    stream_report = read_json(STREAM / "reports" / "prospective-capture-report.json")
    package_audit = read_json(STREAM / "reports" / "audit-report.json")
    test_evidence = read_json(audit_dir / "targeted-tests.json")
    raw_index_sha = atomic_write_json(audit_dir / "raw-digest-index.json", {
        "schema": "footballmind-shadow-raw-digest-index-v1", "rows": observations,
        "note": "22 new observations include local state/menu captures; this is not a count of HTTP calls"})

    source_http = [{
        "source": row["source"],
        "endpoint": row["endpoint"],
        "params": row["params"],
        "observed_at": row["observed_at"],
        "http_status": row["http_status"],
        "error_class": row["error_class"],
        "raw_sha256": row["raw_sha256"],
        "raw_digest_verified": row["raw_digest_verified"],
        "raw_path": row["raw_path"],
        "observation_metadata_path": row["observation_metadata_path"],
    } for row in source_rows]

    report = {
        "schema": "footballmind-single-match-shadow-report-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "physical_match_id": MATCH_ID,
        "fixture": {
            "competition": "Premier League",
            "season": "2026/27",
            "home_team": "Arsenal",
            "away_team": "Leeds United",
            "kickoff": KICKOFF,
            "prediction_cutoff": ACQUISITION_CUTOFF,
            "status_at_capture": "scheduled_pre_kickoff",
            "identity_status": identity_binding.get("status"),
            "identity_binding_sha256": identity_binding.get("binding_sha256"),
            "verified_at": identity_binding.get("comparison", {}).get("verified_at"),
            "identity_source_observations": identity_binding.get("observations", []),
        },
        "fresh_source_initialization": {
            "actual_http_requests": len(source_rows),
            "observations": source_http,
            "market_probabilities": PROBABILITIES,
            "probability_basis": "observed_spftll_regulation_market_snapshot",
            "response_model": None,
            "quote_time": identity_binding.get("initialization_manifest", {}).get("market_initialization", {}).get("quote", {}).get("source_timestamp"),
            "quote_observed_at": identity_binding.get("initialization_manifest", {}).get("market_initialization", {}).get("quote", {}).get("observed_at"),
            "quote_time_basis": "source_timestamp_unknown; observation_clock_is_not_quote_clock",
            "outcome_used": False,
        },
        "runtime": {
            "actual_agent_http_requests": len(receipts),
            "actual_total_fresh_http_requests": len(source_rows) + len(receipts),
            "max_agent_http_requests": 4,
            "attempts": len(receipts),
            "documents_ok": sum(1 for r in receipts if r["status"] == "OK"),
            "documents_no_data": sum(1 for r in receipts if r["status"] == "NO_DATA"),
            "stop_reason": "random_explore_allowance_exhausted",
            "run_id": run_summary.get("run_id") or read_json(MATCH_DIR / "run.json").get("run_id"),
            "journal_run_id": journal_summary.get("run_id"),
            "journal_state": journal_summary.get("state"),
            "journal_stop_reason": journal_summary.get("stop_reason"),
            "journal_requests": journal_summary.get("requests"),
            "journal_recovery": journal_summary.get("recovery"),
            "receipt_rows": receipts,
        },
        "policy_and_gates": {
            "policy_arm": "random_explore",
            "policy_version": "footballmind-shadow-random-explore-v1",
            "random_seed": 20261010,
            "propensity_logging": True,
            "selection_authority": "empirical_g_only",
            "empirical_support_status": "UNKNOWN",
            "empirical_expected_signed_gain": None,
            "empirical_net_voi": None,
            "empirical_selection_allowed": False,
            "empirical_constant_reference": 0.039770113961339996,
            "synthetic_selection_allowed": False,
            "synthetic_mode": "ranking_only_diagnostic_after_menu_freeze",
            "training_inclusion": False,
            "outcome_included": False,
            "cost_status": "simulated_declared_money_not_real_spend",
        },
        "menu_and_document_contract": {
            "runtime_menu_schema": "existing learned-voi-open-execution-v2 menu; generic web action IDs preserved",
            "topic_menu_schema": "diagnostic-only explicit lineup/injury/motivation/weather",
            "capabilities_by_action_id": CAPABILITIES,
            "weather_status": "UNAVAILABLE",
            "candidate_body_in_runtime_state_before_read": False,
            "document_f_consumed": False,
            "document_admission_reason": "unmodeled_document_for_replanning_only",
            "unselected_candidates_preserved": True,
            "strict_v2_on_actual_runtime_actions": False,
            "diagnostic_topic_menu_equals_runtime_menu": False,
            "strict_v2_execution_scope": "post_run_local_diagnostic_only",
        },
        "acceptance_scope": {
            "status": "partial_shadow_complete_runtime_trace_and_offline_diagnostic",
            "completed": ["fresh_identity_and_market", "four_budgeted_random_explore_requests", "all_actual_menus_and_propensities", "strict_v2_separate_topic_diagnostic", "canonical_stream_export"],
            "not_demonstrated": ["strict_v2_bound_to_actual_generic_web_action_ids_before_query", "text_f_consumption_or_probability_response", "numeric_voi_or_economic_stop", "empirical_training_eligibility"],
            "no_data_semantics": "receipt was observed; no document entered body state",
            "returned_page_roles": ["official_fixture_schedule", "club_news_directory", "match_broadcast_article"],
            "typed_evidence_count": 0,
            "source_assertion_truth_independently_verified": False,
        },
        "before_after": {
            "prediction_before": PROBABILITIES,
            "prediction_after_for_returned_steps": PROBABILITIES,
            "prediction_update": None,
            "prediction_update_reason": "response_model_unavailable; market prior held constant",
            "result": None,
            "realized_gain": None,
        },
        "artifact_references": {
            "stage_acceptance_path": str(STAGE_ACCEPTANCE.resolve()),
            "stage_acceptance_sha256": stage_sha,
            "strict_v2_validation_path": str(STRICT_VALIDATION.resolve()),
            "strict_v2_validation_sha256": strict_sha,
            "shadow_plan_path": str((SHADOW / "preparation" / "shadow-plan.json").resolve()),
            "shadow_plan_sha256": sha256_file(SHADOW / "preparation" / "shadow-plan.json"),
            "identity_bundle_path": str((SHADOW / "preparation" / "identity-bundle.json").resolve()),
            "identity_bundle_sha256": sha256_file(SHADOW / "preparation" / "identity-bundle.json"),
            "runtime_trace_path": str((audit_dir / "runtime-trace.json").resolve()),
            "strict_topic_diagnostic_path": str((audit_dir / "strict-v2-topic-diagnostic.json").resolve()),
            "leakage_check_path": str((audit_dir / "leakage-check.json").resolve()),
            "raw_digest_index_sha256": raw_index_sha,
        },
        "canonical_stream": {
            "path": str(STREAM.resolve()),
            "schema": "footballmind-prospective-stream-v2",
            "validation": validation,
            "counts": {key: stream_report[key] for key in ["fixtures_registered", "original_agent_menus_proven", "selected_actions", "unselected_actions", "contexts_before_kickoff", "g_training_eligible_rows", "g_training_ineligible_rows"]},
            "verified_raw_references": sum(row["verified"] for row in package_audit["raw_checks"]),
            "total_raw_references": len(package_audit["raw_checks"]),
            "forecast_projection": "unconfigured_text_f_fields_remain_null; market_prior_retained_separately",
        },
        "targeted_tests": test_evidence,
        "frozen_source_checksums": {str(path.relative_to(WORKSPACE)): sha256_file(path) for path in [
            WORKSPACE / "football_agent/g_prior_features.py", WORKSPACE / "football_agent/evidence_candidates.py"]},
        "leakage_check": leakage,
        "background_processes": {
            "audit_processes_started": [],
            "runtime_relaunched": False,
            "network_requests_during_audit": 0,
        },
    }

    trace_sha = atomic_write_json(audit_dir / "runtime-trace.json", trace)
    topic_sha = atomic_write_json(audit_dir / "strict-v2-topic-diagnostic.json", topic)
    leakage_sha = atomic_write_json(audit_dir / "leakage-check.json", leakage)
    report["artifact_references"].update({
        "runtime_trace_sha256": trace_sha,
        "strict_topic_diagnostic_sha256": topic_sha,
        "leakage_check_sha256": leakage_sha,
    })
    report_sha = atomic_write_json(audit_dir / "shadow-report.json", report)

    # Write a self-contained audit copy for delivery.  The authoritative raw
    # payloads stay in the existing shadow tree and are included in the ZIP.
    DELIVERY.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), audit_dir / Path(__file__).name)
    for filename in ["runtime-trace.json", "strict-v2-topic-diagnostic.json", "leakage-check.json", "shadow-report.json", "raw-digest-index.json", "targeted-tests.json"]:
        shutil.copy2(audit_dir / filename, DELIVERY / filename)

    # Preserve intermediate drafts on disk but exclude them from the final ZIP.
    bundle_paths = [path for path in sorted(SHADOW.rglob("*")) if path.is_file()
                    and path.relative_to(SHADOW).parts[0] != "audit"]
    bundle_manifest = {"schema": "footballmind-shadow-bundle-file-index-v1", "files": [
        {"path": path.relative_to(SHADOW).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in bundle_paths]}
    atomic_write_json(audit_dir / "bundle-file-index.json", bundle_manifest)
    shutil.copy2(audit_dir / "bundle-file-index.json", DELIVERY / "bundle-file-index.json")
    bundle_paths.append(audit_dir / "bundle-file-index.json")
    zip_path = DELIVERY.with_suffix('.zip')
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in bundle_paths:
            archive.write(path, Path("fm-g-single-match-shadow-20261010-v1") / path.relative_to(SHADOW))

    files_for_manifest = [p for p in DELIVERY.glob("*.json")]
    manifest = write_manifest_and_delivery(files_for_manifest, zip_path)
    atomic_write_json(DELIVERY / "manifest.json", manifest)
    zip_sha = sha256_file(zip_path)
    sums = "\n".join(f"{sha256_file(p)}  {p.name}" for p in sorted(DELIVERY.glob("*")) if p.is_file() and p.name != "SHA256SUMS.txt") + "\n"
    sums += f"{zip_sha}  ../{zip_path.name}\n"
    (DELIVERY / "SHA256SUMS.txt").write_text(sums, encoding="utf-8")

    print(json.dumps({
        "report_sha256": report_sha,
        "runtime_trace_sha256": trace_sha,
        "topic_diagnostic_sha256": topic_sha,
        "leakage_check_sha256": leakage_sha,
        "zip_path": str(zip_path.resolve()),
        "zip_sha256": zip_sha,
        "delivery_dir": str(DELIVERY.resolve()),
        "raw_digest_failures": leakage["raw_digest_failures"],
        "leak_check_pass": leakage["leak_check_pass"],
        "actual_total_http": len(source_rows) + len(receipts),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
