"""Preselection integration and refusal boundaries; constructed inputs, zero HTTP."""
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import socket

import pytest

from football_agent.learned_voi_web import _web_action
from football_agent.observable_prequery_shadow import (
    CONFIG_KEY, DECLARATIONS, REGISTRY_SCHEMA, SCHEMA, ObservablePrequeryShadow,
    build_observable_menu, request_binding,
)
from football_research.acquisition.synthetic_g_observable_v3 import (
    FEATURE_SCHEMA, OBSERVABLE_NAMES, PROVENANCE, ObservableOnlyGModel,
    SCHEMA as MODEL_SCHEMA,
)
from test_verified_collection import verified_inputs


def model_bytes():
    return json.dumps({"schema": MODEL_SCHEMA, "feature_schema": FEATURE_SCHEMA,
        "feature_names": list(OBSERVABLE_NAMES), "provenance": PROVENANCE,
        "stop_margin": 0., "trees": [{"left": [-1], "right": [-1],
            "feature": [-2], "threshold": [-2.], "value": [0.9]}]}).encode()


def public_state():
    return {"prediction": {"probabilities": [.52, .23, .25]},
        "state": {"base": {"kickoff": "2026-10-10T16:30:00Z"},
                  "prediction_cutoff": "2026-10-10T12:00:00Z"}}


def declaration(action):
    return {"action": "lineup", "source_reliability": .8, "freshness": .7,
        "redundancy": .2, "response_probability": .9, "updater_trust": .6,
        "cost_units": .02, "declared_at": "2026-10-10T10:00:00Z",
        "basis": "constructed test declaration, not football evidence",
        "request_binding_sha256": request_binding(action)}


def menu(registry, actions=None, public=None):
    actions = actions or [_web_action("read_url", "https://example.com/a", cost={"requests": 1, "money": .01})]
    return build_observable_menu(public or public_state(), actions,
        registry={"schema": REGISTRY_SCHEMA, "capabilities_by_action_id": registry},
        model=ObservableOnlyGModel(json.loads(model_bytes())), decision_at="2026-10-10T11:00:00Z")


@pytest.mark.parametrize("missing", DECLARATIONS)
def test_missing_explicit_fields_never_become_neutral_or_stop(missing):
    action = _web_action("read_url", "https://example.com/a", cost={"requests": 1, "money": .01})
    entry = declaration(action)
    del entry[missing]
    result = menu({action["action_id"]: entry}, [action])
    row = result["by_action_id"][action["action_id"]]
    assert missing in row["missing_fields"]
    assert row["status"] == "UNAVAILABLE"
    assert row["features"] is row["synthetic_estimate"] is None
    assert result["shadow_proposal"] is result["subset_ranking"] is None
    assert not result["selection_allowed"]


def test_exact_ids_and_request_binding_not_topic_or_title_inference():
    action = _web_action("read_url", "https://example.com/injury-lineup", cost={"requests": 1, "money": .01})
    result = menu({"lineup": declaration(action)}, [action])
    assert result["unavailable_candidates"] == 1
    assert result["registry_entries_outside_current_menu"] == ["lineup"]
    entry = declaration(action)
    changed = {**action, "cost": {"requests": 1, "money": .02}}
    result = menu({action["action_id"]: entry}, [changed])
    assert "bind this exact request" in result["by_action_id"][action["action_id"]]["validation_errors"][0]


@pytest.mark.parametrize("mutation", ["weighted_probabilities", "equal_cutoff", "future_declaration", "hidden_field"])
def test_strict_v2_and_declaration_temporal_boundaries(mutation):
    action = _web_action("read_url", "https://example.com/a", cost={"requests": 1, "money": .01})
    state, entry = public_state(), declaration(action)
    if mutation == "weighted_probabilities":
        state["prediction"]["probabilities"] = [52, 23, 25]
    if mutation == "equal_cutoff":
        state["state"]["prediction_cutoff"] = state["state"]["base"]["kickoff"]
    if mutation == "future_declaration":
        entry["declared_at"] = "2026-10-10T11:00:01Z"
    if mutation == "hidden_field":
        entry["latent_probability"] = [.52, .23, .25]
    result = menu({action["action_id"]: entry}, [action], state)
    assert result["unavailable_candidates"] == 1
    assert result["shadow_proposal"] is None


def test_partial_menu_keeps_order_and_does_not_issue_global_stop_or_choice():
    actions = [_web_action("read_url", "https://example.com/" + path, cost={"requests": 1, "money": .01})
               for path in ("z", "a")]
    result = menu({actions[0]["action_id"]: declaration(actions[0])}, actions)
    assert result["menu_action_ids"] == [a["action_id"] for a in actions]
    assert list(result["by_action_id"]) == result["menu_action_ids"]
    assert result["subset_ranking"]["shadow_proposal"] == actions[0]["action_id"]
    assert result["shadow_proposal"] is None and not result["coverage_complete"]


def test_complete_menu_uses_only_canonical_state_and_explicit_declaration():
    action = _web_action("read_url", "https://example.com/a", cost={"requests": 1, "money": .01})
    state = public_state()
    state["documents"] = [{"body": "leaking result and future facts must be ignored"}]
    result = menu({action["action_id"]: declaration(action)}, [action], state)
    assert result["coverage_complete"]
    assert result["shadow_proposal"] == action["action_id"]
    row = result["by_action_id"][action["action_id"]]
    assert row["features"]["probabilities"] == [.52, .23, .25]
    assert row["features"]["values"][9] == 4.5
    assert row["empirical_net_voi"] is None
    assert not row["selection_allowed"]
    assert "leaking result" not in json.dumps(result)


@pytest.mark.parametrize("diagnostic_declaration", [False, True])
def test_collect_preselection_raw_and_exact_replay(verified_inputs, monkeypatch, diagnostic_declaration):
    from football_agent.news_discovery import NewsDiscoveryTools
    from football_agent.learned_voi_replay import replay_saved_run
    from football_research.data_adapters.prospective_text import collect_prepared_batch, MarketPriorPolicy

    def forbidden(*args, **kwargs):
        pytest.fail("Offline integration must not connect to network")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    base = verified_inputs["root"]
    raw = model_bytes()
    (base / "model.json").write_bytes(raw)
    spec = {"schema": SCHEMA, "model_path": str(base / "model.json"),
        "model_raw_sha256": sha256(raw).hexdigest(),
        "registry": {"schema": REGISTRY_SCHEMA, "capabilities_by_action_id": {}}}
    if diagnostic_declaration:
        declared_action = _web_action("read_url", verified_inputs["binding"]["canonical"]["source_url"],
                                     cost={"requests": 1., "money": .01})
        entry = declaration(declared_action)
        entry["declared_at"] = verified_inputs["now"].isoformat()
        spec["registry"]["capabilities_by_action_id"][declared_action["action_id"]] = entry
    (base / "shadow.json").write_text(json.dumps(spec), encoding="utf-8")
    events = []
    original_contexts = ObservablePrequeryShadow.candidate_contexts
    original_plan = MarketPriorPolicy.plan

    def contexts(self, public, actions):
        result = original_contexts(self, public, actions)
        events.append("sidecar")
        return result

    def plan(self, *args, **kwargs):
        assert events[-1] == "sidecar"
        events.append("policy")
        return original_plan(self, *args, **kwargs)

    def execute(self, prepared, *, run_id, request_id):
        assert events[-1] == "policy"
        pending = json.loads((base / "batch/match-101/cycle-000.json").read_text(encoding="utf-8"))
        sidecar = pending["public_before"]["candidate_contexts"][CONFIG_KEY]
        assert sidecar["rankable_candidates"] == int(diagnostic_declaration)
        assert sidecar["unavailable_candidates"] == len(pending["generated_actions"]) - int(diagnostic_declaration)
        assert sidecar["shadow_proposal"] is None
        if diagnostic_declaration:
            assert sidecar["subset_ranking"]["candidates"][0]["expected_net_voi"] > 0
        assert pending["exploration_record"]["selected_p"] == .5
        now = datetime.now(timezone.utc)
        body = b"<rss><channel></channel></rss>" if prepared["tool"] == "search_web" else verified_inputs["body"]
        obs = self.archive.save(prepared["source"], prepared["endpoint"], prepared["params"], body,
            status=200, observed_at=now, context={"requested_at": now.isoformat()})
        return {"document": self.project(obs, prepared), "observation": obs.model_dump(mode="json"),
                "http_attempts": 1, "money": None, "money_status": "not_measured", "latency_seconds": .01}

    monkeypatch.setattr(ObservablePrequeryShadow, "candidate_contexts", contexts)
    monkeypatch.setattr(MarketPriorPolicy, "plan", plan)
    monkeypatch.setattr(NewsDiscoveryTools, "execute", execute)
    report = collect_prepared_batch(base / "plan.json", output=base / "batch", archive_root=base / "market",
        allowed_hosts=["www.uefa.com"], identity_bundle=base / "bindings.json", identity_archive=base / "identity",
        news_provider="bing_rss", source_profile_version=1, news_projection_version=4,
        evidence_version="match-evidence-extraction-v5", candidate_discovery_version=7,
        requests_per_match=1, seed=20261010, policy_arm="random_explore", journal_enabled=True,
        observable_shadow_path=base / "shadow.json")
    folder = base / "batch/match-101"
    run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    assert report["http_requests"] == 1  # constructed provider receipt; zero actual HTTP
    assert run["initial_probabilities"] == run["final_probabilities"]
    assert run["outcome"] is None
    first = run["cycles"][0]
    assert first["selection_basis"] == "random_explore_predeclared_uniform"
    assert first["selected_action"].get(CONFIG_KEY) is None
    replay = replay_saved_run(folder, output=base / "replay", project_root=base)
    assert replay["identical_replay"] and replay["replay_network_requests"] == 0
    assert (folder / "run.json").read_bytes() == (base / "replay/run.json").read_bytes()
