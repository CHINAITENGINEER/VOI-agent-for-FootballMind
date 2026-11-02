"""Offline full collector -> frozen f -> journal -> exact raw replay."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import socket

import pytest

from tests.research.acquisition.test_verified_collection import verified_inputs
from football_agent.historical_f_only_policy import HistoricalFOnlyPolicy, fixture_from_verified_identity
from football_agent.learned_voi_replay import replay_saved_run
from football_agent.news_discovery import NewsDiscoveryTools
from football_research.acquisition.historical_context_training import HistoricalContextModel
from football_research.data_adapters.prospective_text import MarketPriorPolicy, collect_prepared_batch

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "var/submission/p9-g-timestamped-replay-20261010/model.json"


@pytest.fixture(autouse=True)
def forbid_network_and_g(monkeypatch):
    def forbidden(*a, **kw):
        pytest.fail("No network or bundled g call is allowed in f-only integration")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(HistoricalContextModel, "estimate", forbidden)
    if not MODEL.is_file():
        pytest.skip("Separately retained frozen historical f artifact is absent")


def collect_case(inputs, monkeypatch, status="OK", *, clock_state=None, return_times=(),
                 policy_arm="random_explore", request_limit=2, policy_arm_version="footballmind-policy-arm-v2"):
    base = inputs["root"]
    returns = iter(return_times)
    def execute(self, prepared, *, run_id, request_id):
        requested = clock_state[0] if clock_state is not None else datetime.now(timezone.utc)
        if clock_state is not None:
            clock_state[0] = next(returns, clock_state[0])
        now = clock_state[0] if clock_state is not None else datetime.now(timezone.utc)
        body = (b"<rss><channel></channel></rss>" if prepared["tool"] == "search_web"
                else inputs["body"]) if status == "OK" else b""
        obs = self.archive.save(prepared["source"], prepared["endpoint"], prepared["params"], body,
            status=503 if status == "FAILED" else 200, observed_at=now,
            context={"requested_at": requested.isoformat()})
        return {"document": self.project(obs, prepared), "observation": obs.model_dump(mode="json"),
            "http_attempts": 1, "money": None, "money_status": "not_measured", "latency_seconds": .01}
    monkeypatch.setattr(NewsDiscoveryTools, "execute", execute)
    report = collect_prepared_batch(base / "plan.json", output=base / "batch", archive_root=base / "market",
        allowed_hosts=["www.uefa.com"], identity_bundle=base / "bindings.json", identity_archive=base / "identity",
        news_provider="bing_rss", source_profile_version=1, news_projection_version=4,
        evidence_version="match-evidence-extraction-v5", candidate_discovery_version=7,
        requests_per_match=request_limit, seed=20261010, policy_arm=policy_arm,
        policy_arm_version=policy_arm_version, journal_enabled=True,
        f_only_predictor_path=MODEL)
    folder = base / "batch/match-101"
    run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    journal = json.loads(next((base / "batch/acquisition-journal").glob("*/run.json")).read_text(encoding="utf-8"))
    return folder, run, journal, report


def freeze_collector_clocks(monkeypatch, start):
    """Advance real collector interfaces without sleeping or making HTTP."""
    import importlib
    tick = [start]
    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return tick[0].astimezone(tz) if tz else tick[0].replace(tzinfo=None)
    for name in ("football_agent.learned_voi_runtime", "football_agent.inquiry_tools",
                 "football_agent.news_discovery", "football_research.data_adapters.prospective_text",
                 "football_research.data_adapters.prospective_journal",
                 "football_platform.information.acquisition_journal"):
        monkeypatch.setattr(importlib.import_module(name), "datetime", FrozenDatetime)
    return tick


@pytest.mark.parametrize("status", ["OK", "NO_DATA", "FAILED"])
def test_collector_records_actual_f_and_replays_without_g(verified_inputs, monkeypatch, status):
    folder, run, journal, report = collect_case(verified_inputs, monkeypatch, status)
    config = json.loads((folder / "configuration.json").read_text(encoding="utf-8"))
    assert config["policy_manifest"]["bundled_g_enabled"] is False
    assert "context_value_model" not in config
    assert not (folder / "context-value-model.json").exists()
    model = HistoricalContextModel.load(MODEL)
    assert run["initial_probabilities"] == model.metadata["empirical_base"]
    assert run["initial_probabilities"] != config["policy_manifest"]["base_policy"]["market_initialization"]["probabilities"]
    assert run["outcome"] is None and not run["numeric_voi_training_eligible"]
    assert journal["state"] == "STOPPED" and journal["model_status"] == "PREDICTOR_ONLY"
    assert journal["candidate_expected_gains"] is None
    assert journal["policy_version"] == config["policy_version"]
    for request_id, intent in journal["intents"].items():
        assert intent["assessment"]["status"] == "PREDICTOR_ONLY"
        assert intent["assessment"]["candidate_expected_gains"] is None
        cycle = run["cycles"][int(request_id.rsplit("-", 1)[1])]
        distribution = cycle["selection_distribution"]
        assert sum(distribution.values()) == pytest.approx(1.)
        feasible = [p for p in distribution.values() if p > 0]
        assert intent["selection_probability"] == pytest.approx(1 / len(feasible))
        assert intent["selection_probability"] == distribution[intent["action_id"]]
    transitions = [c for c in run["cycles"] if c["selected_action"] is not None]
    assert len(transitions) == report["http_requests"] == 2  # simulated provider receipts, actual HTTP=0
    snapshots = journal["context_snapshots"]
    assert snapshots[0]["snapshot_kind"] == "initial"
    assert snapshots[-1]["snapshot_kind"] == "final"
    afters = [s for s in snapshots if s["snapshot_kind"] == "after_action"]
    assert len(afters) == len(transitions)
    for c, snapshot in zip(transitions, afters):
        assert c["update"]["model_version"] == model.model_version
        assert list(snapshot["prediction_before"][k] for k in ("home", "draw", "away")) == c["prediction_before"]
        assert list(snapshot["prediction_after"][k] for k in ("home", "draw", "away")) == c["prediction_after"]
        assert snapshot["context"]["raw_sha256"]
    for cycle in run["cycles"]:
        for candidate in cycle["strict_decision"]["candidates"]:
            if candidate["value"] is not None:
                assert candidate["value"]["expected_signed_gain"] is None
                assert candidate["value"]["net_voi"] is None
    if status == "OK":
        assert any(c["public_after"]["prediction"]["modeled_read_documents"] > 0 for c in transitions)
        assert any(c["update"]["probability_changed"] for c in transitions)
    else:
        assert all(c["public_after"]["prediction"]["modeled_read_documents"] == 0 for c in transitions)
        assert run["initial_probabilities"] == run["final_probabilities"]
    replay = replay_saved_run(folder, output=verified_inputs["root"] / "replay", project_root=ROOT)
    assert replay["identical_replay"] and replay["replay_network_requests"] == 0
    assert (folder / "run.json").read_bytes() == (verified_inputs["root"] / "replay/run.json").read_bytes()
    if status == "OK":
        (folder / "context-predictor.json").write_bytes(MODEL.read_bytes() + b"\n")
        with pytest.raises(ValueError, match="SHA binding"):
            replay_saved_run(folder, output=verified_inputs["root"] / "tampered", project_root=ROOT)


def test_f_only_rejects_market_prior_and_expired_start(verified_inputs):
    base = MarketPriorPolicy.from_manifest(verified_inputs["plan"]["selected"][0])
    policy = HistoricalFOnlyPolicy(base, HistoricalContextModel.load(MODEL),
        fixture=fixture_from_verified_identity(base, verified_inputs["binding"]), artifact=MODEL.read_bytes())
    state = deepcopy(base.model.initial)
    state["open_text"] = {"prior_probabilities": base.model.probabilities}
    with pytest.raises(ValueError, match="A-learned base"):
        policy.public_state(state)
    with pytest.raises(ValueError, match="valid acquisition start"):
        policy.check_start(datetime.fromisoformat(base.model.cutoff) + timedelta(seconds=1))


@pytest.mark.parametrize("arm", ["fixed_order", "full_probe"])
def test_deterministic_f_collection_rejects_unimplemented_propensity_version(verified_inputs, monkeypatch, arm):
    with pytest.raises(ValueError, match="footballmind-policy-arm-v2"):
        collect_case(verified_inputs, monkeypatch, policy_arm=arm, policy_arm_version="custom-experiment-label")
    assert not (verified_inputs["root"] / "batch").exists()


@pytest.mark.parametrize("arm", ["fixed_order", "full_probe"])
def test_deterministic_f_v2_records_actual_degenerate_distribution(verified_inputs, monkeypatch, arm):
    _, run, journal, _ = collect_case(verified_inputs, monkeypatch, policy_arm=arm)
    for cycle in run["cycles"]:
        if cycle["selected_action"] is None:
            continue
        probabilities = cycle["selection_distribution"]
        assert sum(probabilities.values()) == 1.
        assert probabilities[cycle["selected_action"]["action_id"]] == 1.
        assert cycle["selection_probability_basis"] == "deterministic_predeclared_order_conditional_on_observed_menu"
    assert all(intent["selection_probability"] == 1. for intent in journal["intents"].values())


@pytest.mark.parametrize("arm", ["voi", None])
def test_f_only_requires_explicit_exploration(verified_inputs, arm):
    from football_agent.learned_voi_runtime import LearnedVOIRuntime
    base = MarketPriorPolicy.from_manifest(verified_inputs["plan"]["selected"][0])
    policy = HistoricalFOnlyPolicy(base, HistoricalContextModel.load(MODEL),
        fixture=fixture_from_verified_identity(base, verified_inputs["binding"]), artifact=MODEL.read_bytes())
    with pytest.raises(ValueError, match="controlled exploration"):
        LearnedVOIRuntime(policy, seed=1, policy_arm=arm, exploration_limits={"requests": 1},
                          max_exploration_actions=1)


def test_first_cycle_crossing_cutoff_keeps_f_not_market(verified_inputs):
    from football_agent.learned_voi_runtime import LearnedVOIRuntime
    base = MarketPriorPolicy.from_manifest(verified_inputs["plan"]["selected"][0])
    policy = HistoricalFOnlyPolicy(base, HistoricalContextModel.load(MODEL),
        fixture=fixture_from_verified_identity(base, verified_inputs["binding"]), artifact=MODEL.read_bytes())
    cutoff = datetime.fromisoformat(base.model.cutoff)
    ticks = iter([verified_inputs["now"], cutoff + timedelta(seconds=1), cutoff + timedelta(seconds=2)])
    def forbidden(*args):
        pytest.fail("An expired first decision must not query or generate candidates")
    result = LearnedVOIRuntime(policy, seed=1, policy_arm="random_explore",
        exploration_limits={"requests": 1, "money": .01}, max_exploration_actions=1).run(
        base.model.initial, run_id="expired-f-only", limits={"requests": 1, "money": .01},
        propose=forbidden, execute=forbidden, output=verified_inputs["root"] / "expired",
        provider_version="offline-test", generator_version="offline-test", clock=lambda: next(ticks),
        evidence_match={"match_id": base.model.fixture["match_id"], "home_team": "Croatia", "away_team": "Spain",
            "kickoff_at": base.model.fixture["kickoff"], "prediction_cutoff": base.model.cutoff})
    assert result["initial_probabilities"] == result["final_probabilities"] == policy.predictor.metadata["empirical_base"]
    assert result["terminal_selection"]["reason"] == "forecast_cutoff_elapsed"
    assert not result["attempts"]


def test_missing_body_publication_is_not_f_consumption(verified_inputs, monkeypatch):
    # The transported body can be present but still ineligible for this f.
    body = verified_inputs["body"].decode()
    import re
    verified_inputs["body"] = re.sub(r'"datePublished": "[^"]+", ', '', body).encode()
    folder, run, journal, _ = collect_case(verified_inputs, monkeypatch)
    predictions = [c["public_after"]["prediction"] for c in run["cycles"] if c.get("public_after")]
    assert predictions and all(p["modeled_read_documents"] == 0 for p in predictions)
    assert run["initial_probabilities"] == run["final_probabilities"]


@pytest.mark.parametrize("bad", ["trained_identity", "model_bytes", "fixture_order"])
def test_f_only_rejects_invalid_binding(verified_inputs, bad):
    base = MarketPriorPolicy.from_manifest(verified_inputs["plan"]["selected"][0])
    model = HistoricalContextModel.load(MODEL)
    fixture = fixture_from_verified_identity(base, verified_inputs["binding"])
    artifact = MODEL.read_bytes()
    if bad == "trained_identity":
        fixture["physical_match_id"] = model.metadata["A_match_ids"][0]
    if bad == "model_bytes":
        payload = json.loads(artifact)
        payload["metadata"]["empirical_base"] = [.2, .3, .5]
        artifact = json.dumps(payload).encode()
    if bad == "fixture_order":
        fixture["home_team_id"], fixture["away_team_id"] = fixture["away_team_id"], fixture["home_team_id"]
    with pytest.raises(ValueError):
        policy = HistoricalFOnlyPolicy(base, model, fixture=fixture, artifact=artifact)
        policy.public_state(base.model.initial)


@pytest.mark.parametrize("boundary,offset,reason", [
    ("cutoff", 1, "forecast_cutoff_elapsed"),
    ("kickoff", 0, "kickoff_elapsed"),
    ("kickoff", 1, "kickoff_elapsed"),
])
def test_late_return_preserves_raw_cost_and_terminal_without_late_f(verified_inputs, monkeypatch,
                                                                    boundary, offset, reason):
    base = MarketPriorPolicy.from_manifest(verified_inputs["plan"]["selected"][0])
    bound = base.model.cutoff if boundary == "cutoff" else base.model.fixture["kickoff"]
    late = datetime.fromisoformat(bound) + timedelta(microseconds=offset)
    tick = freeze_collector_clocks(monkeypatch, verified_inputs["now"] + timedelta(seconds=1))
    predictions = []
    original_predict = HistoricalFOnlyPolicy.predict_public
    def audited_predict(self, public):
        predictions.append(tick[0])
        return original_predict(self, public)
    monkeypatch.setattr(HistoricalFOnlyPolicy, "predict_public", audited_predict)
    folder, run, journal, report = collect_case(verified_inputs, monkeypatch, clock_state=tick, return_times=[late])
    assert predictions and all(t < late for t in predictions)
    assert report["http_requests"] == run["measured_resources"]["requests"] == 1
    assert journal["budget"]["charged_requests"] == 1 and journal["budget"]["reserved_requests"] == 0
    assert journal["state"] == "STOPPED" and journal["pending_request_id"] is None
    assert run["terminal_selection"]["reason"] == reason
    assert journal["stop_reason"] == ("KICKOFF_REACHED" if boundary == "kickoff" else reason)
    assert len(run["attempts"]) == len(run["cycles"]) == 1
    cycle = run["cycles"][0]
    assert cycle["execution_status"] == "returned"
    assert cycle["admission"]["status"] == "not_admitted" and not run["documents"]
    assert cycle["requested_at"] < cycle["returned_at"] == late.isoformat()
    assert cycle["update"]["status"] == "not_updated" and cycle["update"]["prediction_retained"]
    assert cycle["prediction_before"] == cycle["prediction_after"] == run["final_probabilities"]
    assert run["initial_probabilities"] == run["final_probabilities"]
    assert run["final_probabilities"] == HistoricalContextModel.load(MODEL).metadata["empirical_base"]
    receipt = json.loads((folder / "receipt-000.json").read_text(encoding="utf-8"))
    assert receipt["actual_cost"] == cycle["actual_cost"]
    assert receipt["receipt"]["observation"]["content_hash"]
    assert next(iter(journal["receipts"].values()))["status"] == ("LATE_RETURN" if boundary == "kickoff" else "RETURNED")
    for snapshot in journal["context_snapshots"][-2:]:
        assert snapshot["recording_scope"] == "audit_only" and not snapshot["forecast_eligible"]
        assert snapshot["observed_at"] == snapshot["captured_at"] == late.isoformat()
        assert snapshot["forecast_as_of"] < snapshot["acquisition_deadline"]
        assert snapshot["prediction_after"] == dict(zip(("home", "draw", "away"), run["final_probabilities"]))
    assert journal["context_snapshots"][-1]["snapshot_kind"] == "final"
    assert not run["numeric_voi_training_eligible"] and run["outcome"] is None
    assert journal["candidate_expected_gains"] is None
    monkeypatch.setattr(HistoricalFOnlyPolicy, "predict_public", original_predict)
    replay = replay_saved_run(folder, output=verified_inputs["root"] / "replay", project_root=ROOT)
    assert replay["identical_replay"] and replay["replay_network_requests"] == 0
    assert (folder / "run.json").read_bytes() == (verified_inputs["root"] / "replay/run.json").read_bytes()


def test_late_return_retains_previously_text_updated_f(verified_inputs, monkeypatch):
    base = MarketPriorPolicy.from_manifest(verified_inputs["plan"]["selected"][0])
    start = verified_inputs["now"] + timedelta(seconds=1)
    tick = freeze_collector_clocks(monkeypatch, start)
    late = datetime.fromisoformat(base.model.fixture["kickoff"]) + timedelta(seconds=1)
    folder, run, journal, report = collect_case(verified_inputs, monkeypatch, clock_state=tick,
        return_times=[start + timedelta(seconds=1), start + timedelta(seconds=2), late],
        policy_arm="fixed_order", request_limit=3)
    search, first, last = run["cycles"]
    assert search["selected_action"]["tool"] == "search_web"
    assert first["update"]["probability_changed"] and first["public_after"]["prediction"]["modeled_read_documents"] > 0
    assert last["update"]["status"] == "not_updated"
    assert run["initial_probabilities"] != run["final_probabilities"] == first["prediction_after"]
    assert last["prediction_before"] == last["prediction_after"] == run["final_probabilities"]
    assert len(run["documents"]) == 1 and report["http_requests"] == 3
    assert journal["budget"]["charged_requests"] == 3 and journal["state"] == "STOPPED"
    assert replay_saved_run(folder, output=verified_inputs["root"] / "replay", project_root=ROOT)["identical_replay"]


def test_context_archive_finishing_after_kickoff_is_audit_only(verified_inputs, monkeypatch):
    from football_spider.storage.raw_archive import RawArchive
    base = MarketPriorPolicy.from_manifest(verified_inputs["plan"]["selected"][0])
    tick = freeze_collector_clocks(monkeypatch, verified_inputs["now"] + timedelta(seconds=1))
    late = datetime.fromisoformat(base.model.fixture["kickoff"]) + timedelta(seconds=1)
    save = RawArchive.save
    def delayed_save(self, source, endpoint, params, body, **kwargs):
        obs = save(self, source, endpoint, params, body, **kwargs)
        if source == "agent_trajectory" and params.get("snapshot_kind") == "after_action":
            payload = json.loads(body)
            if payload.get("prediction_before") != payload.get("prediction_after"):
                tick[0] = late
        return obs
    monkeypatch.setattr(RawArchive, "save", delayed_save)
    folder, run, journal, report = collect_case(verified_inputs, monkeypatch, clock_state=tick,
                                              policy_arm="fixed_order", request_limit=3)
    assert report["http_requests"] == 2 and journal["state"] == "STOPPED"
    first = run["cycles"][1]
    assert first["update"]["probability_changed"]
    assert first["returned_at"] < base.model.cutoff
    assert first["after_state_timing"]["captured_at"] == late.isoformat()
    assert first["after_state_timing"]["recording_scope"] == "audit_only"
    assert run["final_probabilities"] == first["prediction_after"]
    assert run["final_state_timing"]["observed_at"] == late.isoformat()
    assert replay_saved_run(folder, output=verified_inputs["root"] / "replay", project_root=ROOT)["identical_replay"]


def test_exact_cutoff_return_is_still_admitted(verified_inputs, monkeypatch):
    base = MarketPriorPolicy.from_manifest(verified_inputs["plan"]["selected"][0])
    tick = freeze_collector_clocks(monkeypatch, verified_inputs["now"] + timedelta(seconds=1))
    cutoff = datetime.fromisoformat(base.model.cutoff)
    folder, run, journal, _ = collect_case(verified_inputs, monkeypatch, clock_state=tick,
        return_times=[tick[0], cutoff, cutoff + timedelta(microseconds=1)],
        policy_arm="fixed_order", request_limit=3)
    search, first, last = run["cycles"]
    assert first["admission"]["status"] == "admitted" and first["update"]["probability_changed"]
    assert last["admission"]["status"] == "not_admitted"
    assert run["final_probabilities"] == first["prediction_after"]
    assert journal["context_snapshots"][-1]["recording_scope"] == "audit_only"
    assert replay_saved_run(folder, output=verified_inputs["root"] / "replay", project_root=ROOT)["identical_replay"]


def test_v2_f_only_archive_still_replays_exactly(verified_inputs, monkeypatch):
    from football_agent.learned_voi_runtime import LearnedVOIRuntime
    original_init = LearnedVOIRuntime.__init__
    def legacy_init(self, *args, **kwargs):
        kwargs["legacy_predictor_context"] = True
        original_init(self, *args, **kwargs)
    monkeypatch.setattr(LearnedVOIRuntime, "__init__", legacy_init)
    folder, run, _, _ = collect_case(verified_inputs, monkeypatch)
    config = json.loads((folder / "configuration.json").read_text(encoding="utf-8"))
    assert config["predictor_context_schema"] == "runtime-predictor-context-v2"
    assert "recording_scope" not in run["final_state_timing"]
    monkeypatch.setattr(LearnedVOIRuntime, "__init__", original_init)
    assert replay_saved_run(folder, output=verified_inputs["root"] / "replay", project_root=ROOT)["identical_replay"]
