"""Archive visibility, provenance, real baseline and explicit fault fixtures."""
from __future__ import annotations

from datetime import datetime
import json

import pytest

from football_agent.adapters import ArchiveReplayTools, LocalArchiveTools, _platform
from football_agent.testing import SyntheticTools, TEST_CUTOFF, TEST_MATCH_ID
from football_agent.types import ToolResult


def save_fixture(root, *, observed="2026-09-20T11:00:00+00:00", home="Home", price=2.0, context=None):
    RawArchive, _, _ = _platform()
    row = {"matchId": "123", "leagueId": 1, "leagueChs": "英超", "homeId": 10, "awayId": 20,
           "homeChs": home, "awayChs": "Away", "matchTime": "2026-09-20 20:00:00",
           "footMatchState": 0, "homeScore": None, "awayScore": None,
           "spfWinFoot": price, "spfEqualFoot": 3.0, "spfLoseFoot": 4.0}
    payload = json.dumps({"code": 0, "data": [{"list": [row]}]}, ensure_ascii=False).encode()
    return RawArchive(root).save("spftll", "https://fixture.invalid/GetSimpleMatchsAll", {}, payload,
                                 status=200, observed_at=datetime.fromisoformat(observed), context=context)


def base_inputs():
    return {"match_id": "spftll:123", "prediction_cutoff": TEST_CUTOFF}


def test_future_fixture_and_later_versions_are_excluded(tmp_path):
    save_fixture(tmp_path)
    save_fixture(tmp_path, observed="2026-09-20T11:45:00+00:00", home="Future rename", price=1.5)
    tools = LocalArchiveTools(tmp_path / "raw")
    result = tools.execute("query_matches", base_inputs())
    assert result.status == "OK"
    assert result.data[0]["home"] == "Home"
    evidence = tools.execute("read_evidence", base_inputs())
    assert len(evidence.data) == 2  # fixture and complete 1x2; missing prices are not invented


def test_cutoff_before_first_observation_returns_no_data(tmp_path):
    save_fixture(tmp_path)
    tools = LocalArchiveTools(tmp_path / "raw")
    result = tools.execute("query_matches", {"prediction_cutoff": "2026-09-20T10:59:59Z"})
    assert result.status == "NO_DATA" and result.data == []


def test_ingestion_time_is_a_visibility_gate(tmp_path):
    save_fixture(tmp_path, context={"ingested_at": "2026-09-20T11:45:00Z"})
    tools = LocalArchiveTools(tmp_path / "raw")
    assert tools.execute("read_evidence", base_inputs()).status == "NO_DATA"
    later = tools.execute("read_evidence", {**base_inputs(), "prediction_cutoff": "2026-09-20T11:50:00Z"})
    assert later.status == "OK"
    assert all(item["available_at"] == "2026-09-20T11:45:00+00:00" for item in later.data)


def test_real_market_tool_returns_service_probability_and_lineage(tmp_path):
    save_fixture(tmp_path)
    tools = LocalArchiveTools(tmp_path / "raw")
    args = base_inputs()
    match = tools.execute("query_matches", args).data[0]
    evidence = tools.execute("read_evidence", args).data
    features = tools.execute("get_features", {**args, "evidence": evidence})
    assert features.status == "OK"
    result = tools.execute("predict", {**args, "match": match, "features": features.data})
    assert result.status == "OK", result.error
    assert result.data["model_version"] == "market-proportional-1.0"
    assert result.data["home_win_probability"] == pytest.approx((1 / 2) / (1 / 2 + 1 / 3 + 1 / 4))
    assert result.data["information_mode"] == "operational"
    assert result.data["input_manifest"]["feature_input"][0]["raw_ref"].endswith(".observation.json")
    assert features.data["odds_snapshot_id"] in {item["evidence_id"] for item in evidence}


def test_duplicate_observations_keep_raw_references_but_same_content_hash(tmp_path):
    save_fixture(tmp_path)
    save_fixture(tmp_path, observed="2026-09-20T11:10:00+00:00")
    evidence = LocalArchiveTools(tmp_path / "raw").execute("read_evidence", base_inputs()).data
    odds = [item for item in evidence if item["kind"] == "odds"]
    assert len(odds) == 2
    assert odds[0]["content_hash"] == odds[1]["content_hash"]
    assert odds[0]["raw_ref"] != odds[1]["raw_ref"]


def test_missing_model_and_refresh_config_are_explicit(tmp_path):
    save_fixture(tmp_path)
    tools = LocalArchiveTools(tmp_path / "raw", model_artifact=tmp_path / "missing.json")
    args = base_inputs()
    match = tools.execute("query_matches", args).data[0]
    evidence = tools.execute("read_evidence", args).data
    features = tools.execute("get_features", {**args, "evidence": evidence}).data
    assert tools.execute("predict", {**args, "match": match, "features": features}).status == "MODEL_UNAVAILABLE"
    assert tools.execute("refresh_information", {**args, "topic": "odds"}).status == "NEEDS_CONFIGURATION"


def test_service_refresh_preserves_real_rate_limit_and_cost(tmp_path):
    class Service:
        def estimate_cost(self, inputs):
            return 0.3

        def execute(self, inputs):
            return ToolResult(status="RATE_LIMITED", error="429", cost=0.2, retry_after_seconds=15)

    tools = LocalArchiveTools(tmp_path / "raw", refresh_service=Service())
    assert tools.estimate_cost("refresh_information", base_inputs()) == 0.3
    result = tools.execute("refresh_information", base_inputs())
    assert (result.status, result.cost, result.retry_after_seconds) == ("RATE_LIMITED", 0.2, 15)


def test_successful_refresh_does_not_backdate_new_information(tmp_path):
    class Service:
        def estimate_cost(self, inputs):
            return 0.2

        def execute(self, inputs):
            save_fixture(tmp_path, observed="2026-09-20T11:50:00+00:00")
            return ToolResult(data={"requests": 1}, cost=0.2)

    tools = LocalArchiveTools(tmp_path / "raw", refresh_service=Service())
    result = tools.execute("refresh_information", base_inputs())
    assert result.status == "NO_DATA" and result.data["evidence"] == []
    assert result.cost == 0.2


def test_features_reject_future_and_expired_evidence(tmp_path):
    save_fixture(tmp_path)
    tools = LocalArchiveTools(tmp_path / "raw")
    args = base_inputs()
    evidence = tools.execute("read_evidence", args).data
    for item in evidence:
        item["available_at"] = "2026-09-20T11:40:00Z"
    assert tools.execute("get_features", {**args, "evidence": evidence}).status == "NO_DATA"
    for item in evidence:
        item["available_at"] = "2026-09-20T11:00:00Z"
        item["expires_at"] = "2026-09-20T11:20:00Z"
    assert tools.execute("get_features", {**args, "evidence": evidence}).status == "NO_DATA"


def test_archive_replay_reveals_real_arrivals_without_network(tmp_path):
    save_fixture(tmp_path)
    save_fixture(tmp_path, observed="2026-09-20T11:10:00+00:00", price=1.9)
    save_fixture(tmp_path, observed="2026-09-20T11:40:00+00:00", price=1.8)
    tools = ArchiveReplayTools(tmp_path / "raw")
    initial = tools.execute("read_evidence", base_inputs())
    assert len(initial.data) == 2
    query = tools.execute("refresh_information", {**base_inputs(), "topic": "odds"})
    assert query.status == "OK" and len(query.data["evidence"]) == 1
    assert query.data["mechanism"] == "archive_replay"
    assert query.data["network_requests"] == 0
    assert query.data["evidence"][0]["observed_at"] == "2026-09-20T11:10:00+00:00"
    assert tools.capabilities()["external_requests_enabled"] is False


def test_corrupt_archive_reports_failure_and_keeps_reference(tmp_path):
    observation = save_fixture(tmp_path)
    (tmp_path / observation.payload_path).write_bytes(b"corrupt archive body")
    tools = LocalArchiveTools(tmp_path / "raw")
    result = tools.execute("read_evidence", base_inputs())
    assert result.status == "FAILED" and result.data == []
    assert result.error["archive_issues"][0]["raw_ref"].endswith(".observation.json")


@pytest.mark.parametrize("field", ["source_published_at", "source_timestamp"])
def test_future_publication_or_vendor_timestamp_is_not_a_feature(tmp_path, field):
    save_fixture(tmp_path)
    tools = LocalArchiveTools(tmp_path / "raw")
    evidence = tools.execute("read_evidence", base_inputs()).data
    for item in evidence:
        target = item["content"] if field == "source_timestamp" else item
        target[field] = "2026-09-20T11:45:00Z"
    result = tools.execute("get_features", {**base_inputs(), "evidence": evidence})
    assert result.status == "NO_DATA"


@pytest.mark.parametrize("scenario,expected", [("timeout", "TIMEOUT"), ("rate_limited", "RATE_LIMITED"), ("empty", "NO_DATA")])
def test_synthetic_failure_results_are_labelled_and_preserved(scenario, expected):
    tools = SyntheticTools(scenario)
    assert tools.capabilities()["mode"] == "test"
    args = {"match_id": TEST_MATCH_ID, "prediction_cutoff": TEST_CUTOFF, "topic": "odds"}
    assert tools.execute("refresh_information", args).status == expected


def test_synthetic_update_and_duplicate_are_deterministic():
    tools = SyntheticTools()
    args = {"match_id": TEST_MATCH_ID, "prediction_cutoff": TEST_CUTOFF}
    old = tools.execute("read_evidence", args).data
    new = tools.execute("refresh_information", args).data["evidence"]
    again = tools.execute("refresh_information", args).data["evidence"]
    assert old[0]["content_hash"] != new[0]["content_hash"]
    assert new == again
    features = tools.execute("get_features", {**args, "evidence": new}).data
    prediction = tools.execute("predict", {**args, "features": features}).data
    assert prediction["test_mode"] is True
    assert prediction["model_version"] == "synthetic-test-model-v1"
