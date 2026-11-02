"""Submission backend: actual frozen models, no provider or future-feature access."""
from copy import deepcopy
import json
import socket
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from demo.research_g import ResearchGApplication, SCHEMA, observed_snapshot
from demo.server import DemoApplication, create_server


@pytest.fixture(scope="module")
def backend(tmp_path_factory):
    return ResearchGApplication(tmp_path_factory.mktemp("research-g"))


def start(backend, **kwargs):
    return backend.start({"match_id": backend.match_ids[0], **kwargs})


def test_catalog_f_binding_and_scope(backend):
    catalog = backend.catalog()
    assert len(catalog["matches"]) == 5
    assert catalog["model_bindings"]["f"] == backend.g["f_labeler_sha256"]
    assert catalog["selection_authority"] == "research_g"
    assert catalog["is_live"] is False
    assert catalog["production_ready"] is False


def test_start_scores_real_g_without_revealing_bodies(backend):
    view = start(backend)
    assert view["schema_version"] == SCHEMA
    assert view["g"]["candidate_actions"]
    assert view["steps"] == []
    assert view["prediction"]["after"] is None
    assert view["f"]["consumed_body_input_ids"] == []
    for candidate in view["g"]["candidate_actions"]:
        assert candidate["expected_net_voi"] == pytest.approx(
            candidate["immediate_gain"] - candidate["current_cost"] + candidate["continuation_net_value"])
        assert candidate["propensity"] is None
        assert "body" not in candidate


def test_selector_only_receives_current_public_menu(backend, monkeypatch):
    from demo.research_g import continuation
    original = continuation.choose
    calls = []
    def checked(model, menu, remaining, cost):
        for action in menu:
            assert set(action) == {"action_id", "query"}
            assert set(action["query"]) == {"schema", "public", "relations"}
            assert set(action["query"]["public"]) == {
                "schema", "probabilities", "acquired_documents", "visible_candidates", "article_age_hours", "title"}
        calls.append(deepcopy(menu))
        return original(model, menu, remaining, cost)
    monkeypatch.setattr(continuation, "choose", checked)
    view = start(backend, cost_loss_units=0)
    backend.action(view["run_id"], {"action": "run", "expected_revision": 0})
    assert len(calls) >= 2
    assert all(c["query"]["public"]["acquired_documents"] == 1 for c in calls[1])


def test_actual_f_called_after_reveal_and_stop(backend, monkeypatch):
    from demo.research_g import predictor
    original = predictor.predict
    seen = []
    def checked(model, row):
        assert set(row) == {"physical_match_id", "decision_at", "prior", "used_input_ids", "evidence_events"}
        seen.append(list(row["used_input_ids"]))
        return original(model, row)
    monkeypatch.setattr(predictor, "predict", checked)
    view = start(backend)
    done = backend.action(view["run_id"], {"action": "run", "expected_revision": 0})
    assert len(seen) == len(done["steps"]) + 1
    assert done["lifecycle"]["is_terminal"]
    assert done["f"]["status"] == "updated"
    assert done["steps"][0]["revealed_input_id"] in seen[1]
    assert done["steps"][0]["revealed_input_id"] not in seen[0]
    assert done["steps"][0]["body"]["text"]
    assert done["prediction"]["change_status"] == "unchanged_after_f"


@pytest.mark.parametrize("strategy", ["g_one_step", "g_two_step", "fixed_order"])
def test_all_five_matches_complete_with_bounded_replanning(backend, strategy):
    for mid in backend.match_ids:
        view = backend.start({"match_id": mid, "strategy": strategy, "cost_loss_units": 0})
        done = backend.action(view["run_id"], {"action": "run", "expected_revision": 0})
        assert done["lifecycle"]["is_terminal"]
        assert len(done["steps"]) <= 3
        ids = [s["action_id"] for s in done["steps"]]
        assert len(set(ids)) == len(ids)
        assert backend.get(done["run_id"], report=True)["evaluation"] is not None
        assert sum(done["prediction"]["current"][k] for k in
                   ("home_win_probability", "draw_probability", "away_win_probability")) == pytest.approx(1)


def test_no_network_and_no_label_access_until_terminal_report(backend, monkeypatch):
    import demo.research_g as module
    original = module.read
    def safe_read(path):
        assert path != backend.files["cases"], "Result labels must not be reread by policy"
        return original(path)
    def forbidden(*args, **kwargs):
        raise AssertionError("Network is forbidden")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(module, "read", safe_read)
    view = start(backend)
    assert backend.get(view["run_id"], report=True)["evaluation"] is None
    backend.action(view["run_id"], {"action": "run", "expected_revision": 0})


def test_failure_does_not_publish_after(backend, monkeypatch):
    from demo.research_g import predictor
    view = start(backend)
    def fail(*args):
        raise ValueError("f inference failed")
    monkeypatch.setattr(predictor, "predict", fail)
    with pytest.raises(ValueError, match="f inference failed"):
        backend.action(view["run_id"], {"action": "next", "expected_revision": 0})
    loaded = backend.get(view["run_id"])
    assert loaded["prediction"]["after"] is None
    assert loaded["revision"] == 0
    assert loaded["steps"] == []


def test_revision_prevents_double_click_advancing_twice(backend):
    view = start(backend, strategy="fixed_order")
    backend.action(view["run_id"], {"action": "next", "expected_revision": 0})
    with pytest.raises(ValueError, match="STALE_REVISION"):
        backend.action(view["run_id"], {"action": "next", "expected_revision": 0})
    assert len(backend.get(view["run_id"])["steps"]) == 1


def test_restart_recovers_same_run(backend):
    view = start(backend, strategy="fixed_order")
    view = backend.action(view["run_id"], {"action": "next", "expected_revision": 0})
    restored = ResearchGApplication(backend.store_dir)
    assert restored.get(view["run_id"]) == view
    done = restored.action(view["run_id"], {"action": "run", "expected_revision": view["revision"]})
    assert len(done["steps"]) == 3


@pytest.mark.parametrize("payload", [
    {"max_reads": True}, {"max_reads": 4}, {"cost_loss_units": float("nan")},
    {"cost_loss_units": -1}, {"strategy": "live"}, {"model_path": "unknown"},
    {"match_id": "spftll:match:3003899"},
])
def test_reject_unsupported_inputs(backend, payload):
    with pytest.raises(ValueError):
        start(backend, **payload)


def test_artifact_change_blocks_resume(backend, monkeypatch):
    import demo.research_g as module
    view = start(backend)
    monkeypatch.setattr(module, "digest", lambda path: "changed")
    with pytest.raises(ValueError, match="CONFIGURATION_CHANGED"):
        backend.get(view["run_id"])


def test_http_frontend_contract(tmp_path, backend):
    app = DemoApplication(store_dir=tmp_path)
    app._research_g = backend
    server = create_server("127.0.0.1", 0, app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    def request(path, payload=None):
        req = Request(base + path, data=json.dumps(payload).encode() if payload is not None else None,
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=30) as response:
            return json.load(response)
    try:
        catalog = request("/api/research-g/catalog")
        view = request("/api/research-g/runs", {"match_id": catalog["matches"][0]["match_id"]})
        prefix = "/api/research-g/runs/" + view["run_id"]
        assert request(prefix + "/view") == view
        done = request(prefix + "/action", {"action": "run", "expected_revision": 0})
        assert done["lifecycle"]["is_terminal"]
        assert request(prefix + "/report")["evaluation"] is not None
        with pytest.raises(HTTPError) as exc:
            request(prefix + "/action", {"action": "run", "expected_revision": 0})
        assert exc.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
