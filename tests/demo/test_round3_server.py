"""HTTP boundary and fair comparison tests for acquisition UI."""
from copy import deepcopy
import json
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from demo.acquisition_server import AcquisitionApplication, UnconfiguredWorkflow, make_server


class Workflow:
    def __init__(self):
        self.runs = {}
        self.calls = []

    def start(self, match_id, cutoff, **kwargs):
        self.calls.append((match_id, cutoff, deepcopy(kwargs)))
        run_id = "run-" + str(len(self.calls))
        self.runs[run_id] = {"run_id": run_id, "state": "INITIALIZED", "reason": None,
                             "strategy": kwargs["strategy"], "visible_state": {"mask": {"xg": False}, "groups": {}},
                             "prediction": None, "configuration": {"mode": "synthetic_test"}, "events": [],
                             "budget": kwargs["budget"]}
        return deepcopy(self.runs[run_id])

    def get_run(self, run_id):
        return deepcopy(self.runs[run_id])

    def step(self, run_id):
        self.runs[run_id]["state"] = "STOPPED"
        self.runs[run_id]["reason"] = "VALUE_NONPOSITIVE"
        return self.get_run(run_id)

    run_to_completion = step


@pytest.fixture
def service():
    workflow = Workflow()
    app = AcquisitionApplication(workflow, {"mode": "synthetic_test", "matches": []})
    server = make_server("127.0.0.1", 0, app)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", workflow, app
    server.shutdown()
    server.server_close()
    thread.join()


def request(url, path, data=None, headers=None):
    req = Request(url + path, data=None if data is None else json.dumps(data).encode(),
                  headers={"Content-Type": "application/json", **(headers or {})})
    try:
        response = urlopen(req)
    except HTTPError as exc:
        response = exc
    with response:
        payload = response.read()
        if "application/json" in response.headers.get("Content-Type", ""):
            payload = json.loads(payload)
        return response.status, payload


VALUES = {"match_id": "fixture:1", "cutoff": "2025-01-01T12:00:00Z", "seed": 42,
          "budget": {"max_actions": 3, "max_simulated_cost": 2}}


def test_create_step_restore_and_public_mask(service):
    url, workflow, _ = service
    status, run = request(url, "/api/runs", VALUES)
    assert status == 201
    assert run["visible_state"] == {"mask": {"xg": False}, "groups": {}}
    status, run = request(url, f'/api/runs/{run["run_id"]}/step', {})
    assert status == 200 and run["reason"] == "VALUE_NONPOSITIVE"
    assert request(url, f'/api/runs/{run["run_id"]}')[1] == run
    assert workflow.calls[0][2]["strategy"] == "all_candidates"


def test_comparison_has_identical_match_cutoff_seed_and_budget(service):
    url, workflow, _ = service
    status, result = request(url, "/api/compare", VALUES)
    assert status == 201
    assert [r["strategy"] for r in result["runs"]] == ["fixed", "all_candidates"]
    first, second = deepcopy(workflow.calls)
    assert first[:2] == second[:2]
    first[2].pop("strategy")
    second[2].pop("strategy")
    assert first[2] == second[2]


@pytest.mark.parametrize("field", ["hidden_groups", "features", "target", "model_artifact", "bundle_path", "action_id", "cost_conversion"])
def test_client_cannot_supply_private_state_or_paths(service, field):
    url, workflow, _ = service
    assert request(url, "/api/runs", {**VALUES, field: "injected"})[0] == 400
    assert workflow.calls == []


def test_step_cannot_override_state(service):
    url, _, _ = service
    _, run = request(url, "/api/runs", VALUES)
    assert request(url, f'/api/runs/{run["run_id"]}/step', {"prediction": [1, 0, 0]})[0] == 400


def test_cross_origin_mutation_rejected(service):
    url, workflow, _ = service
    assert request(url, "/api/runs", VALUES, {"Origin": "https://attacker.example"})[0] == 403
    assert workflow.calls == []


def test_unconfigured_is_visible_without_fake_run(service):
    url, _, app = service
    app.workflow = UnconfiguredWorkflow("fitted artifact missing")
    app.capabilities = {"mode": "unconfigured", "reason": "fitted artifact missing"}
    assert request(url, "/api/config")[1]["capabilities"]["mode"] == "unconfigured"
    status, result = request(url, "/api/runs", VALUES)
    assert status == 503 and "BACKEND_UNCONFIGURED" in result["detail"]


def test_static_paths_and_strategy_validation(service):
    url, _, _ = service
    assert request(url, "/")[0] == 200
    assert request(url, "/app.js")[0] == 200
    assert request(url, "/../football_agent/acquisition.py")[0] == 404
    assert request(url, "/api/runs", {**VALUES, "strategy": "custom_plugin"})[0] == 400
    assert request(url, "/api/config")[1]["strategies"] == ["fixed", "pure_llm", "all_candidates", "llm_gate"]


def test_nonloopback_rejected():
    with pytest.raises(ValueError):
        make_server("0.0.0.0", 0, AcquisitionApplication(Workflow()))
