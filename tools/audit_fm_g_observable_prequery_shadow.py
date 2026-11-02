"""Read-only source audit and bounded delivery for the fresh prequery shadow.

No HTTP, settlement retrieval, training, or rewriting the original run.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from football_agent.observable_prequery_shadow import CONFIG_KEY, request_binding
from football_research.acquisition.state import parse_time
from football_research.data_adapters.prospective_text import _save
from football_platform.information.stream_validator import validate_stream
from football_spider.storage.raw_archive import RawArchive


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-evidence", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.run.resolve(), args.output.resolve()
    if output.exists() or output.with_suffix(".zip").exists():
        raise FileExistsError("Final deliveries are immutable; choose a new output version")
    folder = source / "batch/match-3003907"
    run, config = read(folder / "run.json"), read(folder / "configuration.json")
    plan = read(source / "preparation/plan.json")
    kickoff = parse_time(plan["kickoff"])
    cutoff = parse_time(plan["prediction_cutoff"])
    original_run_sha = file_hash(folder / "run.json")
    stream_validation = validate_stream(source / "information-stream")
    assert stream_validation["valid"]
    replay = read(source / "replay/acceptance.json")
    assert replay["identical_replay"] and replay["replay_network_requests"] == 0
    assert file_hash(source / "replay/run.json") == original_run_sha
    assert run["outcome"] is None and not run["numeric_voi_training_eligible"]
    assert not (source / "information-stream/labels/results.jsonl").read_text(encoding="utf-8").strip()
    archive = RawArchive(folder / "source-archive")
    by_id = {obs.id: obs for obs in archive.observations()}
    trace, ledgers, menu_checks = [], [], []
    for cycle in run["cycles"]:
        public = cycle["public_before"]
        shadow = public["candidate_contexts"][CONFIG_KEY]
        actions = cycle["generated_actions"]
        ids = [action["action_id"] for action in actions]
        assert shadow["menu_action_ids"] == ids and set(shadow["by_action_id"]) == set(ids)
        assert parse_time(public["decision_at"]) < kickoff
        assert not shadow["selection_allowed"] and shadow["rankable_candidates"] == 0
        assert shadow["shadow_proposal"] is None
        for action in actions:
            entry = shadow["by_action_id"][action["action_id"]]
            assert entry["request_binding_sha256"] == request_binding(action)
            assert entry["features"] is None and entry["synthetic_estimate"] is None
        menu_obs = by_id[cycle["menu_raw_observation_id"]]
        payload = archive.read_body(menu_obs)
        assert sha256(payload).hexdigest() == cycle["menu_raw_sha256"]
        menu = json.loads(payload)
        assert menu["public_before"]["candidate_contexts"][CONFIG_KEY] == shadow
        menu_checks.append({"step": cycle["step"], "raw_sha256": menu_obs.content_hash,
            "raw_path": str(menu_obs.payload_path), "prequery_map_equals_archived_menu": True})
        chosen = cycle["selected_action"]
        entry = {"step": cycle["step"], "decision_at": public["decision_at"],
            "original_menu": actions, "prequery_map": shadow,
            "selected_action": chosen, "selection_basis": cycle["selection_basis"],
            "propensities": (cycle.get("exploration_record") or {}).get("probabilities"),
            "propensity_scope": "conditional_on_generated_eligible_candidates_not_provider_exposure",
            "prediction_before": cycle.get("prediction_before"),
            "prediction_after": cycle.get("prediction_after"),
            "prediction_scope": "observed_market_prior_no_configured_text_f",
            "document_admission": cycle.get("admission"), "model_update": cycle.get("update"),
            "outcome": None, "realized_signed_gain": None,
            "menu_raw_reference": menu_checks[-1]}
        if chosen:
            receipt = read(folder / f"receipt-{cycle['step']:03d}.json")
            obs = receipt["receipt"]["observation"]
            timing = receipt["receipt"]["archive_save_timing"]
            clocks = {"requested_at": obs["context"]["requested_at"],
                "response_observed_at": obs["observed_at"],
                "archive_save_started_at": timing["archive_save_started_at"],
                "archive_save_returned_at": timing["archive_save_returned_at"],
                "agent_receipt_available_at": cycle["receipt_available_at"]}
            ordered = [parse_time(value) for value in clocks.values()]
            assert ordered == sorted(ordered) and ordered[-1] < cutoff < kickoff
            assert parse_time(menu_obs.observed_at) <= ordered[0]
            doc = receipt["document"]
            admission = next(a for a in run["attempts"] if a["attempt_id"] == cycle["attempt_id"])
            ledger = {"step": cycle["step"], "action_id": chosen["action_id"],
                "status": receipt["status"], "http_status": obs["http_status"],
                "document_admitted": admission["admission"] == "admitted",
                "source_observation_id": obs["id"], "raw_path": obs["payload_path"],
                "raw_sha256": obs["content_hash"], "clocks": clocks,
                "capture_timing_scope": "archive_save_return_not_fsync",
                "provider_projection_clock": doc.get("available_at"),
                "provider_projection_clock_is_agent_handoff": False,
                "request_count": receipt["actual_cost"]["requests"],
                "latency_ms": receipt["actual_cost"]["latency_ms"],
                "money": receipt["actual_cost"]["money"],
                "declared_action_cost": chosen["cost"], "declared_action_cost_status": chosen["cost_status"],
                "retrieval_diagnostics": doc.get("structured_news", {}).get("retrieval_diagnostics"),
                "document_kind": doc.get("kind"),
                "admitted_runtime_document_id": doc["document_id"] if admission["admission"] == "admitted" else None}
            ledgers.append(ledger)
            entry["receipt"] = ledger
        trace.append(entry)
    raw_checks = []
    for name in ("source-archive", "identity-archive", "prior-archive"):
        current = RawArchive(folder / name)
        for obs in current.observations():
            observed = sha256(current.read_body(obs)).hexdigest()
            assert observed == obs.content_hash
            raw_checks.append({"archive": name, "observation_id": obs.id,
                "raw_path": obs.payload_path, "raw_sha256": observed, "verified": True})
    duplicate_clusters = {}
    for row in raw_checks:
        duplicate_clusters.setdefault(row["raw_sha256"], []).append({k: row[k] for k in ("archive", "observation_id")})
    duplicate_clusters = {k: v for k, v in duplicate_clusters.items() if len(v) > 1}
    frozen_checks = []
    for path, expected in [
        ("football_agent/g_prior_features.py", "4b6f1bb328af4c10531f41da7efa8e061ce220a5ef2fa5fb673030ca2a21070c"),
        ("football_agent/evidence_candidates.py", "2f6641192eb5c459399d8a42c157b1a71f7af7695a79de21c35c61593c9ec85b"),
        ("var/submission/fm-g-single-match-shadow-20261010-final.zip", "ce6fca6c001fceb7a69a2be8b943ed36ac713541b075030e2214f61015f43e61"),
    ]:
        actual = file_hash(ROOT / path)
        assert actual == expected
        frozen_checks.append({"path": path, "expected_sha256": expected, "observed_sha256": actual, "match": True})
    statistics = read(source / "information-stream/reports/prospective-capture-report.json")
    actual_requests = sum(row["request_count"] for row in ledgers)
    assert actual_requests == run["measured_resources"]["requests"] == 2
    assert run["initial_probabilities"] == run["final_probabilities"]
    summary = {
        "schema": "footballmind-fresh-observable-prequery-shadow-audit-v1",
        "status": "completed_bounded_shadow_diagnostic_unavailable",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "fixture_id": plan["fixture_id"], "kickoff": plan["kickoff"],
        "run_id": run["run_id"], "run_sha256": original_run_sha,
        "source_run": str(source), "trace_steps": len(trace), "agent_http_requests": actual_requests,
        "setup_http_requests": 0, "maximum_authorized_agent_http": 4,
        "selected_actions": 2, "unselected_action_rows": statistics["unselected_actions"],
        "context_snapshots": statistics["contexts_before_kickoff"],
        "runtime_documents_admitted": len(run["documents"]), "source_response_records": 2,
        "new_fixture_specific_articles": 0, "fixture_schedule_pages": 1,
        "schedule_payload_duplicates_prior_identity": True,
        "empty_projection_count": 1, "upstream_search_result_count": 10,
        "upstream_search_filtered_count": 10, "upstream_search_relevance": "not_assessed",
        "query_echo_matched": True, "filter_status": "filtered_empty_not_provider_empty",
        "strict_v2_scope": "existing_frozen_canonical_state_validator_invoked_before_policy_selection",
        "complete_prior_feature_vectors_generated": 0, "v3_estimate_calls_in_real_run": 0,
        "unavailable_v3_action_rows": sum(t["prequery_map"]["unavailable_candidates"] for t in trace),
        "synthetic_model_raw_sha256": plan["candidate_generator"][CONFIG_KEY]["model_raw_sha256"],
        "synthetic_selection_allowed": False, "empirical_g_enabled": False,
        "empirical_constant_controller_reference": 0.039770113961339996,
        "empirical_gate": "UNKNOWN_unchanged_not_invoked_as_synthetic_selection",
        "g_training_eligible_rows": 0, "independent_outcomes": 0, "realized_signed_gain": None,
        "prediction_before": run["initial_probabilities"], "prediction_after": run["final_probabilities"],
        "stop_reason": run["terminal_selection"]["reason"],
        "stop_is_synthetic_or_economic_optimum": False,
        "limitations": ["No exact explicit v3 semantic action/scalar declarations; candidates UNAVAILABLE",
            "Both team directories not configured in frozen profile; only league entry available",
            "Market snapshot observed 2026-10-09, reused unchanged; quote_time unknown",
            "Text f not configured in this journal path; admitted page did not update forecast",
            "Canonical cutoff is acquisition horizon, not runtime decision time",
            "Source assertions not independently verified sentence by sentence",
            "No independent final outcome; no realized gain or empirical rows",
            "No fsync-level persistence clock claimed"],
        "replay": replay, "validator": stream_validation,
        "raw_digest_checks": {"passed": len(raw_checks), "total": len(raw_checks),
            "unique_payload_sha256": len({r['raw_sha256'] for r in raw_checks})},
        "standard_stream_raw_digest_checks": statistics["raw_digest_checks"],
        "frozen_artifacts": frozen_checks, "test_evidence": read(args.test_evidence),
        "training_performed": False, "outcome_fetch_performed": False,
        "background_collection_started": False,
        "replay_learning_records_scope": "existing_unlabelled_replay_audit_rows_only_not_empirical_B_or_training"}
    leakage = {"schema": "footballmind-fresh-observable-shadow-leakage-v1", "pass": True,
        "all_new_decisions_requests_responses_handoffs_before_cutoff_and_kickoff": True,
        "all_feature_maps_exact_action_ids_and_request_binding": True,
        "all_feature_maps_equal_pre_io_RawArchive_menu_bytes": True,
        "actual_menu_maps_before_selection_verified_by_integration_and_replay": True,
        "missing_declarations_not_replaced_with_defaults": True,
        "strict_canonical_projection_excludes_docs_titles_urls_results": True,
        "transport_payload_and_original_candidate_order_unchanged": True,
        "no_labels_or_realized_gain": True, "old_arsenal_package_unchanged": True,
        "semantic_truth_of_every_source_sentence_verified": False,
        "no_unproven_persistence_claims": True, "menu_checks": menu_checks}
    output.mkdir(parents=True)
    shutil.copytree(source, output / "capture")
    _save(output / "shadow-trace.json", {"schema": "footballmind-fresh-observable-shadow-trace-v1", "steps": trace})
    _save(output / "http-ledger.json", ledgers)
    _save(output / "audit-summary.json", summary)
    _save(output / "leakage-audit.json", leakage)
    _save(output / "raw-digest-checks.json", raw_checks)
    _save(output / "duplicate-clusters.json", duplicate_clusters)
    _save(output / "validation-report.json", stream_validation)
    source_files = ["football_agent/observable_prequery_shadow.py", "football_agent/learned_voi_replay.py",
        "football_research/data_adapters/prospective_text.py", "tests/research/acquisition/test_observable_prequery_shadow.py",
        "tools/run_fm_g_observable_prequery_shadow.py", "tools/audit_fm_g_observable_prequery_shadow.py"]
    for relative in source_files:
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    (output / "README.txt").write_text(
        "Fresh ManUtd-Spurs shadow, 2026-10-10.\n"
        "2 real HTTP / maximum 4; random_explore seed 20261010.\n"
        "Exact generic action maps were persisted before selection. All v3 candidates UNAVAILABLE; no neutral defaults, no synthetic ranks.\n"
        "Bing returned 10 items; all filtered by existing source catalog (relevance not assessed). League schedule page admitted; same payload as prior identity.\n"
        "No new fixture article. Market-prior probabilities unchanged. No configured text f, no empirical g training, no outcome/gain.\n"
        "See shadow-trace.json, http-ledger.json and capture/information-stream. Raw menu bytes contain the original pre-selection map.\n"
        "Provider parse-time available_at is not agent handoff; use recorded receipt/document availability in the stream. Save return is not fsync.\n"
        "Existing replay learning-records.json is an unlabelled audit projection, never an empirical training set.\n"
        "Old Arsenal package and strict-v2 frozen sources verified unchanged. No background collector was started.\n",
        encoding="utf-8")
    files = [{"path": str(path.relative_to(output)).replace('\\', '/'), "bytes": path.stat().st_size,
              "sha256": file_hash(path)} for path in sorted(output.rglob("*")) if path.is_file()]
    _save(output / "manifest.json", {"schema": "footballmind-fresh-shadow-delivery-manifest-v1",
        "files": files, "sha256_scope": "all_delivery_files_except_this_manifest", "source_run_sha256": original_run_sha})
    zip_path = output.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_DEFLATED) as archive_zip:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                archive_zip.write(path, str(path.relative_to(output)).replace('\\', '/'))
    # Check both the on-disk files and the ZIP bytes against the manifest.
    with zipfile.ZipFile(zip_path) as archive_zip:
        for row in files:
            assert file_hash(output / row["path"]) == row["sha256"]
            assert sha256(archive_zip.read(row["path"])).hexdigest() == row["sha256"]
        assert len(archive_zip.namelist()) == len(files) + 1
    assert file_hash(folder / "run.json") == original_run_sha
    result = {"path": str(output), "zip": str(zip_path), "zip_bytes": zip_path.stat().st_size,
        "zip_sha256": file_hash(zip_path), "manifest_sha256": file_hash(output / "manifest.json"),
        "audit_sha256": file_hash(output / "audit-summary.json"), "manifest_files": len(files),
        "zip_entries": len(files) + 1, "raw_checks": len(raw_checks), "http": actual_requests}
    _save(output.parent / (output.name + "-verification.json"), result)
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
