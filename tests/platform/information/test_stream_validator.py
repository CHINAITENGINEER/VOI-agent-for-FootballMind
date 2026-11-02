from __future__ import annotations

import hashlib
import json
import gzip
import pytest
from pathlib import Path

from football_platform.information.stream_validator import validate_stream


def _write_jsonl(root: Path, relative: str, rows: list[dict]) -> tuple[str, str]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows).encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest(), str(relative).replace("\\", "/")


def _package(tmp_path: Path, *, attempt_times=("2026-10-08T08:10:00Z", "2026-10-08T08:11:00Z")) -> Path:
    root = tmp_path / "footballmind-information-stream"
    root.mkdir()
    raw = root / "raw" / "document.html"
    raw.parent.mkdir()
    raw.write_text("<p>lineup pending</p>\n", encoding="utf-8")
    raw_hash = hashlib.sha256(raw.read_bytes()).hexdigest()
    fixture = {"physical_match_id": "m-1", "competition": "Premier League", "season": "2026/27",
               "home_team": "Home", "away_team": "Away", "kickoff_at": "2026-10-08T12:00:00Z",
               "fixture_source_url": "https://example.test/fixture/m-1"}
    document = {"document_id": "d-1", "version_id": "v-1", "physical_match_ids": ["m-1"],
                "canonical_url": "https://example.test/document/1", "publisher": "Example",
                "title": "Lineup", "body_text": "lineup pending", "language": "en",
                "source_published_at": "2026-10-08T07:00:00Z", "source_modified_at": None,
                "captured_at": "2026-10-08T08:00:00Z", "source_time_raw": "07:00 UTC",
                "timezone_basis": "explicit_utc", "time_precision": "minute", "time_evidence": "byline",
                "information_mode": "prospective_capture", "archive_snapshot_at": None,
                "archive_snapshot_url": None, "historical_snapshot_proven": False,
                "raw_path": "raw/document.html", "raw_sha256": raw_hash, "previous_version_id": None,
                "document_type": "news", "match_binding_basis": "canonical fixture link",
                "duplicate_cluster_id": "cluster-1"}
    discovery = {"discovery_id": "disc-1", "physical_match_id": "m-1", "observed_at": "2026-10-08T08:05:00Z",
                 "query": "Home Away lineup", "provider": "example", "raw_path": None, "raw_sha256": None,
                 "visible_candidates": [{"candidate_id": "c-1", "url": document["canonical_url"],
                                         "title": document["title"], "snippet": "lineup pending", "rank": 0}],
                 "status": "ok", "truncation_or_cap": None}
    attempt = {"attempt_id": "a-1", "physical_match_id": "m-1", "discovery_id": "disc-1", "candidate_id": "c-1",
               "tool": "http", "request": {"url": document["canonical_url"]}, "started_at": attempt_times[0],
               "completed_at": attempt_times[1], "status": "ok", "http_status": 200, "error_code": None,
               "returned_version_ids": ["v-1"], "request_count": 1, "latency_ms": 100,
               "token_usage_if_available": None, "monetary_cost_if_known": None}
    context = {"physical_match_id": "m-1", "as_of": "2026-10-08T07:30:00Z", "captured_at": "2026-10-08T08:00:00Z",
               "source": "example"}
    label_raw = root / "raw" / "result.json"
    label_raw.write_text('{"status":"pending"}\n', encoding="utf-8")
    label = {"physical_match_id": "m-1", "regulation_home_goals": 1, "regulation_away_goals": 0,
             "result_status": "final", "result_source_url": "https://example.test/result/m-1",
             "result_published_at": "2026-10-08T14:00:00Z", "result_observed_at": "2026-10-08T14:01:00Z",
             "raw_path": "raw/result.json", "raw_sha256": hashlib.sha256(label_raw.read_bytes()).hexdigest(),
             "timing_basis": "independent_final_source"}
    values = {}
    for relative, rows in (("fixtures.jsonl", [fixture]), ("documents.jsonl", [document]),
                           ("discovery.jsonl", [discovery]), ("attempts.jsonl", [attempt]),
                           ("context_snapshots.jsonl", [context]), ("labels/results.jsonl", [label])):
        digest, path = _write_jsonl(root, relative, rows)
        values[path] = {"path": path, "sha256": digest, "row_count": len(rows)}
    # The manifest is intentionally written after the data files; validating it
    # must not require rewriting any package file.
    (root / "manifest.json").write_text(json.dumps({"schema_version": "footballmind-information-stream-v1",
        "information_modes": ["prospective_capture", "archived_snapshot", "historical_reconstruction"],
        "files": values}, indent=2) + "\n", encoding="utf-8")
    return root


def test_valid_stream_package_and_hash_manifest_are_accepted(tmp_path):
    report = validate_stream(_package(tmp_path))
    assert report["valid"] is True, report
    assert report["counts"]["documents.jsonl"] == 1


def test_historical_external_tool_keeps_unknown_http_count_and_unmeasured_clock(tmp_path):
    root = _package(tmp_path)
    path = root / "attempts.jsonl"
    row = json.loads(path.read_text(encoding="utf-8"))
    row.update(tool="current_web_search", information_mode="historical_reconstruction",
        actual_historical_agent_action=False, request_clock_status="not_measured", started_at=None,
        request_count=None, provider_internal_http_count=None, tool_call_count=1,
        selected_by_policy=True, raw_path="raw/document.html")
    value, relative = _write_jsonl(root, "attempts.jsonl", [row])
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"][relative]["sha256"] = value
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    report = validate_stream(root)
    assert report["valid"], report["errors"]
    assert {x["code"] for x in report["warnings"]} >= {"historical_tool_request_clock_unknown", "unknown_request_count"}
    # Merely calling a prospective request an external tool cannot bypass clocks.
    row["information_mode"] = "prospective_capture"
    value, relative = _write_jsonl(root, "attempts.jsonl", [row])
    manifest["files"][relative]["sha256"] = value
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    report = validate_stream(root)
    assert any(x["code"] == "missing_time" for x in report["errors"])


def test_time_order_error_is_reported_without_mutating_package(tmp_path):
    root = _package(tmp_path, attempt_times=("2026-10-08T08:12:00Z", "2026-10-08T08:11:00Z"))
    before = (root / "attempts.jsonl").read_bytes()
    report = validate_stream(root)
    assert report["valid"] is False
    assert any(item["code"] == "attempt_time_order" for item in report["errors"])
    assert (root / "attempts.jsonl").read_bytes() == before


def test_schema_and_raw_hash_failures_are_reported(tmp_path):
    root = _package(tmp_path)
    path = root / "documents.jsonl"
    row = json.loads(path.read_text(encoding="utf-8"))
    row.pop("body_text")
    row["raw_sha256"] = "0" * 64
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    report = validate_stream(root)
    assert report["valid"] is False
    codes = {item["code"] for item in report["errors"]}
    assert "missing_field" in codes
    assert "raw_hash_mismatch" in codes
    assert "manifest_hash_mismatch" in codes


def _v2_package(tmp_path: Path, *, mutate=None) -> Path:
    root = tmp_path / "footballmind-prospective-stream-v2"
    root.mkdir()
    kickoff = "2026-10-09T12:00:00Z"
    raw = {}
    for name, payload in {
        "fixture.json": b'{"fixture":"m-v2"}\n',
        "menu.json": b'{"menu":"m-v2"}\n',
        "document.json": b'{"document":"d-v2"}\n',
        "attempt.json": b'{"attempt":"a-v2"}\n',
    }.items():
        path = root / "raw" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        raw[name] = hashlib.sha256(payload).hexdigest()
    fixture = {
        "fixture_id": "fixture-v2", "physical_match_id": "m-v2", "competition": "Premier League", "season": "2026",
        "home_team": "Home", "away_team": "Away", "kickoff_at": kickoff, "fixture_source_url": "https://example.test/m-v2",
        "registered_at": "2026-10-09T08:00:00Z", "registration_source": "spftll", "information_mode": "prospective_capture",
        "raw_path": "raw/fixture.json", "raw_sha256": raw["fixture.json"], "pre_kickoff_frozen": True,
        "policy_arm": "fixed_order", "arm_assigned_at": "2026-10-09T07:59:00Z", "policy_version": "policy-v2",
        "random_seed": None, "propensity_logging": True,
    }
    document = {
        "document_id": "d-v2", "physical_match_id": "m-v2", "url": "https://example.test/doc-v2", "publisher": "Example",
        "title": "Preview", "body": "preview body", "language": "en", "source_published_at": "2026-10-09T07:00:00Z",
        "source_modified_at": None, "captured_at": "2026-10-09T10:02:30Z", "available_at": "2026-10-09T10:02:00Z",
        "availability_status": "known", "availability_basis": "response_observed_before_cutoff", "original_time_text": "07:00 UTC",
        "timing_basis": "explicit_utc", "information_mode": "prospective_capture", "raw_path": "raw/document.json",
        "raw_sha256": raw["document.json"], "previous_version_id": None, "document_kind": "official_preview",
        "fixture_binding_basis": "canonical fixture URL",
    }
    discovery = {
        "discovery_id": "disc-v2", "physical_match_id": "m-v2", "observed_at": "2026-10-09T10:00:00Z",
        "query": "Home Away preview", "provider": "NewsDiscoveryTools", "visible_candidates": [
            {"candidate_id": "cand-v2", "action_type": "official_preview", "url": document["url"], "title": document["title"], "snippet": "preview", "rank": 0},
            {"candidate_id": "cand-weather", "action_type": "weather", "url": "https://example.test/weather", "title": "Weather", "snippet": "", "rank": 1},
        ], "menu_id": "menu-v2", "policy_version": "policy-v2", "original_agent_menu_proven": True,
        "information_mode": "prospective_capture", "raw_path": "raw/menu.json", "raw_sha256": raw["menu.json"],
        "status": "returned", "truncation_or_cap": None,
    }
    attempts = [
        {
            "attempt_id": "a-v2", "physical_match_id": "m-v2", "discovery_id": "disc-v2", "candidate_id": "cand-v2",
            "action_id": "official_preview", "selected_by_policy": True, "selection_basis": "manual_predeclared",
            "policy_arm": "fixed_order", "policy_version": "policy-v2", "selection_probability": None,
            "selection_probability_basis": "not_recorded", "decision_at": "2026-10-09T10:01:00Z",
            "requested_at": "2026-10-09T10:01:01Z", "response_observed_at": "2026-10-09T10:02:00Z",
            "status": "returned", "source": "NewsDiscoveryTools", "request_count": 1, "latency_ms": 1000,
            "raw_path": "raw/attempt.json", "raw_sha256": raw["attempt.json"], "returned_document_id": "d-v2",
        },
        {
            "attempt_id": "a-weather-not-requested", "physical_match_id": "m-v2", "discovery_id": "disc-v2", "candidate_id": "cand-weather",
            "action_id": "weather", "selected_by_policy": False, "selection_basis": "manual_predeclared",
            "policy_arm": "fixed_order", "policy_version": "policy-v2", "selection_probability": None,
            "selection_probability_basis": "not_recorded", "decision_at": "2026-10-09T10:01:00Z",
            "requested_at": None, "response_observed_at": None, "status": "not_requested", "source": "weather",
            "request_count": 0, "latency_ms": None, "raw_path": None, "raw_sha256": None,
            "not_selected_reason": "fixed_order_budget_exhausted",
        },
    ]
    contexts = [
        {"snapshot_id": "s0", "physical_match_id": "m-v2", "sequence": 0, "snapshot_kind": "initial",
         "observed_at": "2026-10-09T10:00:00Z", "available_at": "2026-10-09T10:00:00Z", "availability_status": "known",
         "availability_basis": "agent_state", "captured_at": "2026-10-09T10:00:01Z", "prediction_cutoff": "2026-10-09T10:00:00Z",
         "known_information": ["fixture"], "unknown_information": ["weather"], "prediction_before": None, "prediction_after": None,
         "information_mode": "prospective_capture"},
        {"snapshot_id": "s1", "physical_match_id": "m-v2", "sequence": 1, "snapshot_kind": "before_action",
         "parent_attempt_id": "a-v2", "observed_at": "2026-10-09T10:01:00Z", "available_at": "2026-10-09T10:01:00Z",
         "availability_status": "known", "availability_basis": "agent_state", "captured_at": "2026-10-09T10:01:00Z",
         "prediction_cutoff": "2026-10-09T10:01:00Z", "known_information": ["fixture"], "unknown_information": ["preview"],
         "prediction_before": None, "prediction_after": None, "information_mode": "prospective_capture"},
        {"snapshot_id": "s2", "physical_match_id": "m-v2", "sequence": 2, "snapshot_kind": "after_action",
         "parent_attempt_id": "a-v2", "observed_at": "2026-10-09T10:02:00Z", "available_at": "2026-10-09T10:02:00Z",
         "availability_status": "known", "availability_basis": "response_observed", "captured_at": "2026-10-09T10:02:01Z",
         "prediction_cutoff": "2026-10-09T10:02:00Z", "known_information": ["fixture", "preview"], "unknown_information": ["weather"],
         "prediction_before": None, "prediction_after": None, "information_mode": "prospective_capture"},
        {"snapshot_id": "s3", "physical_match_id": "m-v2", "sequence": 3, "snapshot_kind": "final",
         "observed_at": "2026-10-09T10:03:00Z", "available_at": "2026-10-09T10:03:00Z", "availability_status": "known",
         "availability_basis": "agent_state", "captured_at": "2026-10-09T10:03:01Z", "prediction_cutoff": "2026-10-09T10:03:00Z",
         "known_information": ["fixture", "preview"], "unknown_information": ["weather"], "prediction_before": None,
         "prediction_after": None, "information_mode": "prospective_capture", "stop_reason": "estimated_net_voi_nonpositive"},
    ]
    rows = {"fixtures.jsonl": [fixture], "documents.jsonl": [document], "discovery.jsonl": [discovery],
            "attempts.jsonl": attempts, "context_snapshots.jsonl": contexts, "labels/results.jsonl": []}
    if mutate:
        mutate(rows)
    values = {}
    for relative, data in rows.items():
        digest, path = _write_jsonl(root, relative, data)
        values[path] = {"path": path, "sha256": digest, "row_count": len(data)}
    (root / "manifest.json").write_text(json.dumps({"schema_version": "footballmind-prospective-stream-v2", "files": values}) + "\n", encoding="utf-8")
    return root


def test_v2_validates_prospective_menu_actions_and_context_chain(tmp_path):
    report = validate_stream(_v2_package(tmp_path))
    assert report["valid"] is True, report
    assert report["counts"]["attempts.jsonl"] == 2


def test_v2_observable_empty_response_does_not_require_document_admission(tmp_path):
    def mutate(rows):
        rows["documents.jsonl"][0].update(body="", available_at=None, document_available_at=None,
            availability_status="unknown", availability_basis=None, document_admitted=False,
            document_admission_basis="verified_raw_context_document_membership",
            result_observed=True, result_observed_at="2026-10-09T10:02:31Z",
            receipt_available_at="2026-10-09T10:02:31Z")
        rows["attempts.jsonl"][0].update(status="empty", result_observed=True,
            result_observed_at="2026-10-09T10:02:31Z", receipt_available_at="2026-10-09T10:02:31Z",
            document_admitted=False, document_available_at=None)
    report = validate_stream(_v2_package(tmp_path, mutate=mutate))
    assert report["valid"] is True, report


def test_v2_rejects_receipt_visibility_relabelled_as_unadmitted_document_availability(tmp_path):
    def mutate(rows):
        rows["documents.jsonl"][0].update(document_admitted=False,
            document_available_at=rows["documents.jsonl"][0]["available_at"])
    report = validate_stream(_v2_package(tmp_path, mutate=mutate))
    assert "document_availability_without_admission" in {issue["code"] for issue in report["errors"]}


def test_v2_capture_completion_cannot_use_archive_invocation_clock(tmp_path):
    def mutate(rows):
        rows["documents.jsonl"][0].update(capture_timing_basis="measured_local_utc_archive_save_return",
            archive_save_started_at="2026-10-09T10:02:30Z", archive_save_returned_at="2026-10-09T10:02:31Z",
            archive_return_timing_basis="measured_local_utc_archive_save_return",
            persistence_boundary="return_observed_fsync_not_instrumented")
    report = validate_stream(_v2_package(tmp_path, mutate=mutate))
    assert "archive_return_unproven" in {issue["code"] for issue in report["errors"]}


def test_v2_unknown_context_capture_is_readable_but_not_temporal_evidence(tmp_path):
    def mutate(rows):
        rows["context_snapshots.jsonl"][0]["captured_at"] = None
    report = validate_stream(_v2_package(tmp_path, mutate=mutate))
    assert report["valid"] is True
    assert any(issue["code"] == "context_clock_unknown" for issue in report["warnings"])


def test_v2_explicit_unknown_document_capture_is_warning_without_substitute_clock(tmp_path):
    def mutate(rows):
        rows["documents.jsonl"][0].update(captured_at=None, capture_status="unknown",
            capture_timing_basis="unknown", observed_at="2026-10-09T10:01:59Z")
    report = validate_stream(_v2_package(tmp_path, mutate=mutate))
    assert report["valid"] is True, report
    assert any(issue["code"] == "document_capture_time_unknown" for issue in report["warnings"])


def test_v2_unknown_document_capture_does_not_allow_post_kickoff_observation(tmp_path):
    def mutate(rows):
        rows["documents.jsonl"][0].update(captured_at=None, capture_status="unknown",
            capture_timing_basis="unknown", observed_at="2026-10-20T10:01:59Z")
    report = validate_stream(_v2_package(tmp_path, mutate=mutate))
    assert report["valid"] is False
    assert any(issue["code"] == "post_kickoff_document_observation" for issue in report["errors"])


def test_v2_context_field_has_its_own_clock_and_cannot_come_from_future(tmp_path):
    def mutate(rows):
        rows["context_snapshots.jsonl"][0]["weather"] = {
            "value": {"rain": True}, "observed_at": "2026-10-09T10:00:30Z"}
    report = validate_stream(_v2_package(tmp_path, mutate=mutate))
    assert report["valid"] is False
    assert any(issue["code"] == "context_field_observed_after_snapshot" for issue in report["errors"])


def test_v2_market_prior_cannot_masquerade_as_trained_prediction(tmp_path):
    def mutate(rows):
        rows["context_snapshots.jsonl"][0].update(
            prediction_status="market_prior_only", calibration_status="not_established",
            prediction_before={"home": .5, "draw": .3, "away": .2})
    report = validate_stream(_v2_package(tmp_path, mutate=mutate))
    assert report["valid"] is False
    assert any(issue["code"] == "unconfigured_prediction" for issue in report["errors"])


def test_v2_known_document_ids_extension_is_optional_but_type_checked(tmp_path):
    def mutate(rows):
        rows["context_snapshots.jsonl"][0]["known_document_ids"] = ["doc-v2"]
        rows["context_snapshots.jsonl"][1]["known_document_ids"] = [None]
    report = validate_stream(_v2_package(tmp_path, mutate=mutate))
    assert report["valid"] is False
    assert any(issue["code"] == "invalid_type"
               and issue["path"].endswith("known_document_ids")
               for issue in report["errors"])


def test_v2_rejects_unselected_action_with_request_or_missing_reason(tmp_path):
    def mutate(rows):
        row = rows["attempts.jsonl"][1]
        row["request_count"] = 1
        row.pop("not_selected_reason")
    report = validate_stream(_v2_package(tmp_path, mutate=mutate))
    codes = {item["code"] for item in report["errors"]}
    assert {"missing_not_selected_reason", "unselected_request_count"} <= codes


def test_v2_rejects_post_kickoff_registration_and_unknown_after_action_parent(tmp_path):
    def mutate(rows):
        rows["fixtures.jsonl"][0]["arm_assigned_at"] = "2026-10-09T12:00:00Z"
        rows["context_snapshots.jsonl"][2]["parent_attempt_id"] = "missing"
    report = validate_stream(_v2_package(tmp_path, mutate=mutate))
    codes = {item["code"] for item in report["errors"]}
    assert "post_kickoff_registration" in codes
    assert "unknown_parent_attempt" in codes


def test_v2_rawarchive_body_gzip_hash_is_checked_against_decompressed_body(tmp_path):
    root = _v2_package(tmp_path)
    body = b'{"document":"rawarchive-body"}\n'
    compressed = root / "raw" / "document.body.gz"
    compressed.write_bytes(gzip.compress(body, mtime=0))
    docs = json.loads((root / "documents.jsonl").read_text(encoding="utf-8"))
    docs["raw_path"] = "raw/document.body.gz"
    docs["raw_sha256"] = hashlib.sha256(body).hexdigest()
    digest, _ = _write_jsonl(root, "documents.jsonl", [docs])
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"]["documents.jsonl"]["sha256"] = digest
    (root / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    report = validate_stream(root)
    assert report["valid"] is True, report


def test_legacy_unknown_result_fixture_is_reported_without_keyerror(tmp_path):
    root = _package(tmp_path)
    result_path = root / "labels" / "results.jsonl"
    row = json.loads(result_path.read_text(encoding="utf-8"))
    row["physical_match_id"] = "missing-fixture"
    result_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    report = validate_stream(root)
    assert report["valid"] is False
    assert any(item["code"] == "unknown_match" for item in report["errors"])


def _replace_rows(root, filename, rows):
    digest, _ = _write_jsonl(root, filename, rows)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"][filename]["sha256"] = digest
    manifest["files"][filename]["row_count"] = len(rows)
    (root / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")


def test_legacy_confirmation_field_names_and_historical_clocks_remain_readable(tmp_path):
    root = _package(tmp_path)
    fixture = json.loads((root / "fixtures.jsonl").read_text(encoding="utf-8"))
    fixture["kickoff"] = fixture.pop("kickoff_at")
    fixture["identity_source_url"] = fixture.pop("fixture_source_url")
    _replace_rows(root, "fixtures.jsonl", [fixture])
    doc = json.loads((root / "documents.jsonl").read_text(encoding="utf-8"))
    doc["physical_match_id"] = doc.pop("physical_match_ids")[0]
    for old, new in (("canonical_url", "url"), ("body_text", "body"), ("source_time_raw", "original_time_text"),
                     ("time_precision", "published_precision"), ("document_type", "document_kind"),
                     ("match_binding_basis", "fixture_binding_basis")):
        doc[new] = doc.pop(old)
    doc["timing_basis"] = doc.pop("timezone_basis")
    for field in ("archive_snapshot_at", "archive_snapshot_url", "duplicate_cluster_id", "time_evidence"):
        doc.pop(field)
    doc.update(captured_at="2026-10-09T08:00:00Z", information_mode="historical_reconstruction")
    _replace_rows(root, "documents.jsonl", [doc])
    discovery = json.loads((root / "discovery.jsonl").read_text(encoding="utf-8"))
    discovery.update(observed_at="2026-10-09T08:00:00Z", original_agent_menu_proven=False)
    discovery.pop("truncation_or_cap")
    _replace_rows(root, "discovery.jsonl", [discovery])
    attempt = json.loads((root / "attempts.jsonl").read_text(encoding="utf-8"))
    for old, new in (("started_at", "requested_at"), ("completed_at", "response_at"), ("http_status", "status_code"),
                     ("error_code", "error_type"), ("tool", "action_type"), ("token_usage_if_available", "token_cost"),
                     ("monetary_cost_if_known", "monetary_cost")):
        attempt[new] = attempt.pop(old)
    attempt.pop("returned_version_ids")
    attempt.update(returned_document_id="d-1", requested_at="2026-10-09T08:01:00Z", response_at="2026-10-09T08:02:00Z")
    _replace_rows(root, "attempts.jsonl", [attempt])
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    report = validate_stream(root)
    assert report["valid"], report
    assert {row["code"] for row in report["warnings"]} >= {"historical_attempt", "historical_menu"}
    assert all(p.read_bytes() == payload for p, payload in before.items())


@pytest.mark.parametrize("probability", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_v2_rejects_invalid_propensities(tmp_path, probability):
    def mutate(rows):
        row = rows["attempts.jsonl"][0]
        row.update(selection_basis="uniform_exploration", selection_probability=probability,
                   selection_probability_basis="uniform_over_eligible_actions")
    report = validate_stream(_v2_package(tmp_path, mutate=mutate))
    assert any(row["code"] == "invalid_propensity" for row in report["errors"])


def test_v2_manual_policy_cannot_claim_propensity_and_unknown_availability_cannot_claim_time(tmp_path):
    def mutate(rows):
        rows["attempts.jsonl"][0].update(selection_probability=1, selection_probability_basis="assumed")
        rows["context_snapshots.jsonl"][0]["availability_status"] = "unknown"
    report = validate_stream(_v2_package(tmp_path, mutate=mutate))
    assert {row["code"] for row in report["errors"]} >= {"fabricated_propensity", "availability_conflict"}


def test_v2_context_unknown_availability_remains_null_without_inventing_time(tmp_path):
    def mutate(rows):
        rows["context_snapshots.jsonl"][0].update(available_at=None, availability_status="unknown")
    report = validate_stream(_v2_package(tmp_path, mutate=mutate))
    assert report["valid"], report


def test_v2_selected_unknown_request_cost_remains_null_with_warning(tmp_path):
    def mutate(rows):
        rows["attempts.jsonl"][0]["request_count"] = None
    report = validate_stream(_v2_package(tmp_path, mutate=mutate))
    assert report["valid"], report
    assert any(row["code"] == "unknown_request_count" for row in report["warnings"])
