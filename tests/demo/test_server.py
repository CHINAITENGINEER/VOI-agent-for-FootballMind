"""HTTP boundary and workflow wiring checks; no network provider requests."""

from contextlib import contextmanager
from http.client import HTTPConnection
import json
from pathlib import Path
import threading

import pytest

from demo.server import DemoApplication, create_server


class StubWorkflow:
    def __init__(self, mode, scenario, store_dir, archive_replay=True):
        self.mode = mode
        self.scenario = scenario
        self.records = {}

    def capabilities(self):
        return {"mode": self.mode, "scenario": self.scenario, "tools": {"predict": "MODEL_UNAVAILABLE"}}

    def list_matches(self, cutoff=None):
        return {"status": "NO_DATA", "data": [], "error": "archive empty"}

    def start(self, match_id, cutoff, budget=None):
        record = {"run_id": "test-run", "match_id": match_id, "prediction_cutoff": cutoff,
                  "mode": self.mode, "test_mode": self.mode == "test", "state": "INITIALIZED", "events": [], "prediction": None}
        self.records["test-run"] = record
        return record

    def get_run(self, run_id):
        return self.records[run_id]

    def step(self, run_id, action=None, topic=None):
        record = self.get_run(run_id)
        record["last_action"] = action
        record["topic"] = topic
        return record

    def run_to_completion(self, run_id):
        record = self.get_run(run_id)
        record.update(state="DEFERRED", reason="MODEL_UNAVAILABLE")
        return record


@contextmanager
def running_server(tmp_path, factory=StubWorkflow):
    server = create_server("127.0.0.1", 0, DemoApplication(tmp_path, factory))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def request(server, method, path, body=None, headers=None):
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
    request_headers = {"Content-Type": "application/json", **(headers or {})}
    connection.request(method, path, body=None if body is None else json.dumps(body), headers=request_headers)
    response = connection.getresponse()
    payload = response.read()
    result = (response.status, dict(response.getheaders()), payload)
    connection.close()
    return result


def test_actual_synthetic_flow_and_report_roundtrip(tmp_path):
    from demo.server import make_workflow
    from football_agent.testing import TEST_CUTOFF, TEST_MATCH_ID

    with running_server(tmp_path, make_workflow) as server:
        status, _, body = request(server, "POST", "/api/runs", {"mode": "test", "scenario": "normal", "match_id": TEST_MATCH_ID, "prediction_cutoff": TEST_CUTOFF})
        assert status == 200
        initial = json.loads(body)
        assert initial["test_mode"] and initial["prediction"] is None
        run_id = initial["run_id"]
        status, _, body = request(server, "POST", f"/api/runs/{run_id}/action", {"action": "run"})
        final = json.loads(body)
        assert status == 200
        assert final["state"] == "STOPPED"
        assert final["prediction"]["test_mode"] is True
        assert final["events"][0]["snapshot"]["prediction"] is None
        status, headers, body = request(server, "GET", f"/api/runs/{run_id}/report")
        assert status == 200
        assert headers["Content-Disposition"].startswith("attachment;")
        assert json.loads(body)["events"] == final["events"]


def test_status_is_not_replaced_with_fake_success(tmp_path):
    with running_server(tmp_path) as server:
        status, _, body = request(server, "GET", "/api/matches")
        assert status == 200
        assert json.loads(body) == {"status": "NO_DATA", "data": [], "error": "archive empty"}
        status, _, body = request(server, "GET", "/api/config")
        assert json.loads(body)["capabilities"]["tools"]["predict"] == "MODEL_UNAVAILABLE"


def test_http_origin_and_static_path_boundaries(tmp_path):
    with running_server(tmp_path) as server:
        assert request(server, "GET", "/", headers={"Host": "attacker.example"})[0] == 403
        assert request(server, "POST", "/api/runs", {}, {"Origin": "https://example.com"})[0] == 403
        assert request(server, "GET", "/../football_agent/types.py")[0] == 404
        assert request(server, "GET", "/%2e%2e/football_agent/types.py")[0] == 404
        status, headers, body = request(server, "GET", "/")
        assert status == 200
        assert b"FootballMind" in body
        assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]


@pytest.mark.parametrize("path, expected_script", [
    ("/", b'/live.js'),
    ("/?refresh=1", b'/live.js'),
    ("/?workspace=defense", b'/defense.js'),
    ("/defense.html", b'/defense.js'),
    ("/live.html", b'/live.js'),
    ("/?workspace=live", b'/live.js'),
    ("/?run=lg-saved-run", b'/live.js'),
    ("/?workspace=research", b'/app.js'),
    ("/?workspace=legacy", b'/app.js'),
    ("/?run=archived-research-run", b'/app.js'),
])
def test_home_defaults_to_online_and_keeps_explicit_workspaces(tmp_path, path, expected_script):
    with running_server(tmp_path) as server:
        status, headers, body = request(server, "GET", path)
        assert status == 200
        assert expected_script in body
        assert headers["Cache-Control"] == "no-store"
        for other in (b'/defense.js', b'/live.js', b'/app.js'):
            if other != expected_script:
                assert other not in body


@pytest.mark.parametrize("payload", [
    [],
    {"match_id": "x", "prediction_cutoff": "2026-09-20T11:00:00"},
    {"match_id": "x", "prediction_cutoff": "2026-09-20T11:00:00Z", "mode": "unknown"},
    {"match_id": "x", "prediction_cutoff": "2026-09-20T11:00:00Z", "budget": {"unexpected": 1}},
    {"match_id": "x", "prediction_cutoff": "2026-09-20T11:00:00Z", "budget": {"max_cost": -1}},
])
def test_invalid_requests_do_not_start_runs(tmp_path, payload):
    with running_server(tmp_path) as server:
        status, _, body = request(server, "POST", "/api/runs", payload)
        assert status == 400
        assert json.loads(body)["status"] == "FAILED"
        assert not server.application.run_owners


def test_remote_bind_is_rejected():
    with pytest.raises(ValueError):
        create_server("0.0.0.0", 8765)


def test_wait_resume_action_is_forwarded(tmp_path):
    with running_server(tmp_path) as server:
        request(server, "POST", "/api/runs", {"match_id": "x", "prediction_cutoff": "2026-09-20T11:00:00Z"})
        for action in ("wait", "resume", "query"):
            status, _, body = request(server, "POST", "/api/runs/test-run/action", {"action": action, "topic": "odds"})
            assert status == 200
            assert json.loads(body)["last_action"] == action


@pytest.mark.parametrize("scenario", ["normal", "model_unavailable", "rate_limited"])
def test_restarted_application_restores_synthetic_scenario(tmp_path, scenario):
    from football_agent.testing import TEST_CUTOFF, TEST_MATCH_ID

    first = DemoApplication(tmp_path)
    record = first.post("/api/runs", {"mode": "test", "scenario": scenario, "match_id": TEST_MATCH_ID, "prediction_cutoff": TEST_CUTOFF})
    restarted = DemoApplication(tmp_path)
    workflow = restarted.run_workflow(record["run_id"])
    assert workflow.tools.scenario == scenario
    final = restarted.post(f"/api/runs/{record['run_id']}/action", {"action": "run"})
    assert final["reason"] != "RUN_CONFIGURATION_CHANGED"
    if scenario == "normal":
        assert final["state"] == "STOPPED"
    elif scenario == "model_unavailable":
        assert final["prediction"] is None
        assert final["state"] == "DEFERRED"
    else:
        assert final["state"] == "WAITING"


@pytest.mark.parametrize("archive_replay", [True, False])
def test_restarted_application_restores_real_adapter_selection(tmp_path, archive_replay):
    from football_agent.adapters import ArchiveReplayTools, LocalArchiveTools

    first = DemoApplication(tmp_path)
    record = first.post("/api/runs", {"mode": "real", "archive_replay": archive_replay, "match_id": "absent", "prediction_cutoff": "2026-09-20T11:10:00Z"})
    restarted = DemoApplication(tmp_path)
    workflow = restarted.run_workflow(record["run_id"])
    assert type(workflow.tools) is (ArchiveReplayTools if archive_replay else LocalArchiveTools)
    assert workflow.get_run(record["run_id"])["tool_configuration"] == record["tool_configuration"]
