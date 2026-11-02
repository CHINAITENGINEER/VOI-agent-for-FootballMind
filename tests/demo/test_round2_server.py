"""Model allowlisting and durable configuration at the local HTTP boundary."""
from http.client import HTTPConnection
import json
import threading

import pytest

from demo.server import DemoApplication, create_server
from football_agent.model_catalog import ModelCatalog


@pytest.fixture
def server(tmp_path):
    instance = create_server("127.0.0.1", 0, DemoApplication(tmp_path / "runs", archive_root=tmp_path / "raw"))
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    yield instance
    instance.shutdown()
    instance.server_close()
    thread.join(2)


def call(server, method, path, payload=None):
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
    connection.request(method, path, body=json.dumps(payload) if payload is not None else None,
                       headers={"Content-Type": "application/json"})
    response = connection.getresponse()
    result = response.status, json.loads(response.read())
    connection.close()
    return result


def initial(**kwargs):
    return {"mode": "real", "match_id": "absent", "prediction_cutoff": "2026-09-20T11:30:00Z", **kwargs}


def test_catalog_exposes_model_specific_inputs_and_training_metadata(server):
    status, body = call(server, "GET", "/api/models")
    assert status == 200
    models = {m["model_id"]: m for m in body["models"]}
    assert models["market"]["required_inputs"]["odds_required"] is True
    assert models["frequency-spain"]["required_inputs"]["odds_required"] is False
    assert models["frequency-spain"]["trained_at"]
    assert len(models["frequency-spain"]["artifact_sha256"]) == 64


@pytest.mark.parametrize("payload", [
    {"model_id": "../../var/model.json"}, {"model_id": "C:\\model.json"},
    {"model_id": {"model_name": "market"}}, {"model_id": "unknown"},
    {"model_artifact": "C:\\model.json"}, {"model_catalog": "C:\\models.json"},
    {"platform_export": "C:\\private.json"}, {"service": "os.system"},
    {"information_mode": "future"}, {"time_mode": "future"},
])
def test_client_configuration_never_bypasses_server_allowlist(server, payload):
    status, body = call(server, "POST", "/api/runs", initial(**payload))
    assert status == 400
    assert body["status"] == "FAILED"
    assert not server.application.run_owners


def test_query_rejects_client_model_path(server):
    status, body = call(server, "GET", "/api/config?model_artifact=C%3A%2Fmodel.json")
    assert status == 400
    assert "服务端" in body["error"]


def test_restart_restores_nondefault_model_and_archive_path(tmp_path):
    first = DemoApplication(tmp_path / "runs", archive_root=tmp_path / "original-raw")
    record = first.post("/api/runs", initial(model_id="frequency-spain"))
    restarted = DemoApplication(tmp_path / "runs", archive_root=tmp_path / "different-raw")
    workflow = restarted.run_workflow(record["run_id"])
    capabilities = workflow.tools.capabilities()
    assert capabilities["model_id"] == "frequency-spain"
    assert capabilities["artifact_sha256"] == record["tool_configuration"]["artifact_sha256"]
    assert capabilities["archive_root"] == record["tool_configuration"]["archive_root"]
    assert workflow.get_run(record["run_id"])["tool_configuration"] == record["tool_configuration"]


def test_restart_reports_configuration_change_without_default_fallback(tmp_path):
    first = DemoApplication(tmp_path / "runs", archive_root=tmp_path / "raw")
    record = first.post("/api/runs", initial(model_id="frequency-spain"))

    class ChangedCatalog(ModelCatalog):
        def resolve(self, model_id):
            value = super().resolve(model_id)
            value["model_version"] = "changed-version"
            return value

    restarted = DemoApplication(tmp_path / "runs", model_catalog=ChangedCatalog())
    with pytest.raises(ValueError, match="RUN_CONFIGURATION_CHANGED"):
        restarted.run_workflow(record["run_id"])
    assert record["run_id"] not in restarted.run_owners


def test_actions_reject_configuration_mutation(server):
    _, record = call(server, "POST", "/api/runs", initial())
    status, body = call(server, "POST", f"/api/runs/{record['run_id']}/action", {"action": "next", "model_id": "frequency-spain"})
    assert status == 400
    assert "已存审计记录" in body["error"]


def test_busy_run_returns_http_409(server, monkeypatch):
    from football_agent.audit import RunBusyError

    def busy(run_id):
        raise RunBusyError("RUN_BUSY: owned by another process")

    monkeypatch.setattr(server.application, "run_workflow", busy)
    status, body = call(server, "POST", "/api/runs/busy/action", {"action": "next"})
    assert status == 409
    assert "RUN_BUSY" in body["error"]


def test_restart_restores_cli_query_policy_and_original_budget(tmp_path):
    from football_agent.engine import Workflow
    from football_agent.policy import FixedOrderPolicy
    from football_agent.testing import SyntheticTools, TEST_CUTOFF, TEST_MATCH_ID
    from football_agent.types import Budget

    first = Workflow(SyntheticTools(), tmp_path, FixedOrderPolicy(["injuries", "lineups"]))
    record = first.start(TEST_MATCH_ID, TEST_CUTOFF, Budget(max_calls=9, max_queries=1, max_requests=2))
    restarted = DemoApplication(tmp_path)
    workflow = restarted.run_workflow(record["run_id"])
    assert workflow.policy.configuration == {"topics": ["injuries", "lineups"]}
    actual = restarted.post(f"/api/runs/{record['run_id']}/action", {"action": "next"})
    assert actual["reason"] != "RUN_CONFIGURATION_CHANGED"
    assert actual["budget"]["max_calls"] == 9
    assert actual["budget"]["max_requests"] == 2


def test_audit_content_reads_only_hash_verified_referenced_tool_io(server):
    from football_agent.testing import TEST_CUTOFF, TEST_MATCH_ID

    _, record = call(server, "POST", "/api/runs", {"mode": "test", "match_id": TEST_MATCH_ID, "prediction_cutoff": TEST_CUTOFF})
    _, updated = call(server, "POST", f"/api/runs/{record['run_id']}/action", {"action": "next"})
    output = next(event["data"]["output_ref"] for event in updated["events"] if event["data"].get("output_ref"))
    status, body = call(server, "GET", f"/api/runs/{record['run_id']}/{output[:-5]}")
    assert status == 200
    assert body["status"] == "OK"
    status, _ = call(server, "GET", f"/api/runs/{record['run_id']}/blobs/{'0' * 64}")
    assert status == 404


def test_platform_live_unconfigured_retains_real_status_and_archive_root(tmp_path):
    app = DemoApplication(tmp_path / "runs", archive_root=tmp_path / "missing-raw")
    config = app.get("/api/config", {"time_mode": ["live"]})
    assert config["capabilities"]["tools"]["refresh_information"] == "NEEDS_CONFIGURATION"
    assert config["capabilities"]["archive_root"] == str((tmp_path / "missing-raw").resolve())
    assert config["capabilities"]["time_mode"] == "live"
    record = app.post("/api/runs", initial(time_mode="live"))
    restarted = DemoApplication(tmp_path / "runs")
    workflow = restarted.run_workflow(record["run_id"])
    assert workflow.tools.capabilities()["configuration_hash"] == record["tool_configuration"]["configuration_hash"]


def test_platform_fixed_cutoff_reader_type_survives_restart(tmp_path):
    from football_agent.engine import Workflow
    from football_agent.platform_tools import PlatformTools

    workflow = Workflow(PlatformTools(archive_root=tmp_path / "raw"), tmp_path / "runs")
    record = workflow.start("absent", "2026-09-20T11:30:00Z")
    restored = DemoApplication(tmp_path / "runs").run_workflow(record["run_id"])
    assert type(restored.tools) is PlatformTools
    assert restored.tools.capabilities()["reader_version"] == record["tool_configuration"]["reader_version"]
