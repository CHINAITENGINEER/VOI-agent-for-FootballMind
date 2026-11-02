from copy import deepcopy
from datetime import datetime, timezone
import json

import pytest

from football_agent.learned_voi_runtime import LearnedVOIRuntime
from football_agent.learned_voi_web import MatchDocumentCandidates
from football_agent.training_exploration_adapter import TrainingExplorationAdapter
from football_research.acquisition.conservative_joint import ConservativeJointReturnPredictor
from football_research.acquisition.data import ControlledEpisode
from football_research.acquisition.joint_voi import LearnedJointVOIPolicy
from football_spider.storage.raw_archive import RawArchive
from test_joint_returns import ACTIONS, fitted, state


def unknown(aid="news", key=None, cost=.01):
    return {"action_id": aid, "information_key": key or aid, "group_ids": [],
            "available": True, "cost_status": "simulated", "cost": {"money": cost, "requests": 1.}}


def receipt(*, status="OK", late=False):
    return {"status": status, "document": {"document_id": "doc-one", "raw_sha256": "a"*64,
            "status": "OK", "observed_at": "2025-04-30T00:00:00Z",
            "available_at": "2025-05-01T01:00:00Z" if late else "2025-04-30T00:00:01Z",
            "text": "Source content", "links": []},
            "actual_cost": {"money": None, "requests": 1., "latency_ms": 10., "tokens": 0.},
            "receipt": {"supervisor_only_label": "must_not_reach_actor"}}


def runtime(model, *, allowance=2, money=.05, steps=8, seed=17, strength=0.):
    policy = LearnedJointVOIPolicy(ConservativeJointReturnPredictor(model, strength=strength),
                                  lambda_cost={"money": 1.})
    return LearnedVOIRuntime(policy, seed=seed, max_exploration_actions=allowance,
                             exploration_limits={"money": money, "requests": 2.}, max_steps=steps)


def run(rt, folder, propose, execute, *, initial=None, limits=None):
    return rt.run(initial or state(), run_id="test-run", limits=limits or {"money": .1, "requests": 4},
                  propose=propose, execute=execute, output=folder,
                  provider_version="synthetic-test", generator_version="synthetic-test")


def test_admitted_return_generates_next_query_without_fake_probability_change(fitted, tmp_path):
    seen, sent = [], []
    numeric = [{**a, "information_key": a["action_id"]} for a in ACTIONS]

    def propose(public):
        seen.append(deepcopy(public))
        return numeric+[unknown()]+([unknown("followup")] if public["documents"] else [])

    def execute(action, public):
        sent.append(action["action_id"])
        pending = json.loads((tmp_path/"run"/f"cycle-{len(sent)-1:03}.json").read_text())
        assert pending["execution_status"] == "reserved_before_io"
        assert pending["reservation_spent"]["requests"] == len(sent)
        returned = receipt()
        returned["document"]["document_id"] = "doc-"+action["action_id"]
        return returned

    result = run(runtime(fitted[0]), tmp_path/"run", propose, execute)
    assert sent == ["news", "followup"]
    assert len(result["documents"]) == 2
    assert not seen[0]["documents"]
    assert len(seen[1]["documents"]) == 1
    assert "supervisor_only_label" not in json.dumps(seen)
    assert result["initial_probabilities"] == result["final_probabilities"]
    assert all(c["update"] is None for c in result["cycles"])
    assert result["terminal_selection"]["decision"] == "defer"
    assert result["terminal_selection"]["stop_certificate"] is None
    assert result["exploration_actions"] == 2
    assert result["unmeasured_resource_receipts"]["money"] == 2
    assert result["cycles"][0]["exploration_exclusions"]["xg_view"] == "computed_nonpositive_value"


def test_opt_in_training_exploration_overrides_selector_and_logs_positive_propensities(fitted, tmp_path):
    model, _, _ = fitted
    rt = runtime(model, allowance=1, money=.1, steps=2, seed=19)
    adapter = TrainingExplorationAdapter(enabled=True, epsilon=.4, seed=23)
    numeric = [{**a, "information_key": a["action_id"]} for a in ACTIONS] + [unknown("news")]
    result = rt.run(state(), run_id="training-explore-run", limits={"money": .1, "requests": 2},
                    propose=lambda _: numeric,
                    execute=lambda *_: receipt(), output=tmp_path / "run",
                    provider_version="synthetic-test", generator_version="synthetic-test",
                    training_exploration=adapter)
    cycle = result["cycles"][0]
    assert cycle["selection_basis"] == "training_collection_epsilon_greedy"
    trace = cycle["training_exploration"]
    assert trace["selection_authority"] == "data_collection_only"
    assert trace["empirical_g_enabled"] is False
    assert trace["label_status"] == "pending_outcome"
    assert all(item["propensity"] > 0 for item in trace["candidate_menu"])
    assert abs(sum(item["propensity"] for item in trace["candidate_menu"]) - 1.0) < 1e-12
    assert json.loads((tmp_path / "run" / "configuration.json").read_text())["training_exploration"]["enabled"] is True


def test_computed_positive_voi_precedes_exploration_and_update_is_verified(fitted, tmp_path):
    model, rows, _ = fitted
    row = {**deepcopy(rows[5]), **state()["base"], "match_id": "future",
           "prediction_cutoff": state()["prediction_cutoff"], "feature_version": "synthetic"}
    environment = ControlledEpisode(row, ACTIONS)
    numeric = [{**a, "information_key": a["action_id"]} for a in ACTIONS]

    def execute(action, public):
        if action["action_id"] == "news":
            return receipt()
        after = environment.reveal(action["action_id"])
        after["action_history"] = public["state"]["action_history"]+[{"action_id": action["action_id"]}]
        return {"status": "OK", "state_after": after,
                "actual_cost": {"money": 0., "requests": 0., "latency_ms": None, "tokens": 0.}}

    result = run(runtime(model, strength=1.), tmp_path/"run", lambda _: numeric+[unknown()], execute)
    first = result["cycles"][0]
    assert first["selection_basis"] == "computed_positive_voi"
    assert first["selected_action"]["action_id"] in {"results_view", "xg_view"}
    assert first["update"]["prediction_after"] == first["prediction_after"]
    assert first["exploration_record"] is None
    assert result["exploration_actions"] == 1


@pytest.mark.parametrize("status,late", [("FAILED", False), ("NO_DATA", False), ("OK", True)])
def test_failed_empty_or_late_return_consumes_allowance_without_admission(fitted, tmp_path, status, late):
    result = run(runtime(fitted[0], allowance=1), tmp_path/"run", lambda _: [unknown(), unknown("second")],
                 lambda *_: receipt(status=status, late=late))
    assert result["exploration_actions"] == 1
    assert result["reserved_cost"]["requests"] == 1
    assert result["documents"] == []
    assert result["initial_probabilities"] == result["final_probabilities"]
    assert len(result["attempts"]) == 1


def test_crash_keeps_reservation_and_reopening_cannot_execute_again(fitted, tmp_path):
    calls = []

    def crash(*args):
        calls.append(1)
        raise OSError("transport lost")

    rt = runtime(fitted[0])
    result = run(rt, tmp_path/"run", lambda _: [unknown()], crash)
    assert result["terminal_selection"]["reason"] == "execution_reconciliation_required"
    assert result["reserved_cost"]["requests"] == 1
    with pytest.raises(FileExistsError):
        run(rt, tmp_path/"run", lambda _: [unknown()], crash)
    assert calls == [1]


@pytest.mark.parametrize("fault", ["posterior", "early_available", "missing_hash", "missing_requests"])
def test_invalid_receipt_stays_out_of_actor_and_halts(fitted, tmp_path, fault):
    returned = receipt()
    if fault == "posterior":
        returned["state_after"] = state()
    elif fault == "early_available":
        returned["document"]["available_at"] = "2025-04-01T00:00:00Z"
    elif fault == "missing_hash":
        returned["document"]["raw_sha256"] = "invalid"
    else:
        returned["actual_cost"]["requests"] = None
    result = run(runtime(fitted[0]), tmp_path/"run", lambda _: [unknown()], lambda *_: returned)
    assert result["terminal_selection"]["reason"] == "receipt_reconciliation_required"
    assert result["documents"] == []
    assert result["final_state"]["action_history"] == []


@pytest.mark.parametrize("actions,money", [(0, .05), (2, 0.)])
def test_no_exploration_without_explicit_allowance(fitted, tmp_path, actions, money):
    def forbidden(*_):
        pytest.fail("No budget to execute")
    result = run(runtime(fitted[0], allowance=actions, money=money), tmp_path/"run", lambda _: [unknown()], forbidden)
    assert result["exploration_actions"] == 0
    assert result["terminal_selection"]["decision"] == "defer"


def test_actual_request_overrun_stops_further_calls(fitted, tmp_path):
    returned = receipt()
    returned["actual_cost"]["requests"] = 3.
    result = run(runtime(fitted[0]), tmp_path/"run", lambda _: [unknown(), unknown("second")],
                 lambda *_: returned, limits={"money": .05, "requests": 2})
    assert result["terminal_selection"]["reason"] == "actual_resource_budget_overrun"
    assert result["measured_resources"]["requests"] == 3
    assert len(result["attempts"]) == 1


def test_same_information_under_new_id_cannot_reopen_query(fitted, tmp_path):
    def propose(public):
        return [unknown("alias" if public["attempts"] else "first", key="same-information")]
    result = run(runtime(fitted[0]), tmp_path/"run", propose, lambda *_: receipt(status="FAILED"))
    assert len(result["attempts"]) == 1


def test_precommitted_draw_records_conditional_probabilities_and_replays(fitted, tmp_path):
    candidates = [unknown("lineup"), unknown("tactics"), unknown("weather")]
    results = [run(runtime(fitted[0], allowance=1), tmp_path/name, lambda _: candidates, lambda *_: receipt())
               for name in ("first", "replay")]
    a, b = (r["cycles"][0]["exploration_record"] for r in results)
    assert a == b
    assert a["selected_p"] == pytest.approx(1/3)
    assert a["candidate_generation_probability"] is None
    assert a["expected_information_value"] is None
    assert "seed" not in results[0]["cycles"][0]["public_before"]["exploration"]


def test_trajectory_recorder_binds_complete_menu_before_io(fitted, tmp_path):
    """The optional recorder receives every candidate before execute()."""
    calls = []
    model = fitted[0]
    policy = LearnedJointVOIPolicy(ConservativeJointReturnPredictor(model, strength=0.), lambda_cost={"money": 1.})
    runtime = LearnedVOIRuntime(policy, seed=17, random_seed=17, policy_arm="fixed_order",
        arm_assigned_at="2025-04-30T00:00:00Z", exploration_limits={"money": .05, "requests": 2},
        max_exploration_actions=1)
    clock = lambda: datetime(2025, 4, 30, tzinfo=timezone.utc)
    archive = RawArchive(tmp_path / "run" / "source-archive")

    def record(payload):
        calls.append(payload)
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        obs = archive.save("agent_trajectory", "https://footballmind.local/menu", {}, raw,
                           status=200, observed_at=clock())
        return {"menu_raw_path": obs.payload_path, "menu_raw_sha256": obs.content_hash,
                "menu_raw_observation_id": obs.id}

    result = runtime.run(state(), run_id="trajectory-recorder", limits={"money": .05, "requests": 2},
        propose=lambda _: [unknown("weather"), unknown("lineup")],
        execute=lambda *_: receipt(), output=tmp_path / "run", provider_version="synthetic",
        generator_version="synthetic", clock=clock,
        run_metadata={"information_mode": "prospective_capture"}, trajectory_recorder=record)
    assert calls and [a["action_id"] for a in calls[0]["generated_actions"]] == ["weather", "lineup"]
    assert [item["action_id"] for item in calls[0]["menu"]] == ["weather", "lineup"]
    assert result["cycles"][0]["menu_raw_path"].replace("\\", "/").startswith("raw/agent_trajectory/")
    assert archive.observations("agent_trajectory")
    config = json.loads((tmp_path / "run" / "configuration.json").read_text())
    assert config["information_mode"] == "prospective_capture"
    assert config["trajectory_raw_archive"]["schema"] == "prospective-menu-archive-v1"
    assert result["attempts"][0]["requested_at"] is None
    assert result["attempts"][0]["request_started_at"] is None
    assert result["attempts"][0]["request_timing_basis"] == "not_recorded"
    assert result["attempts"][0]["decision_at"] == clock().isoformat()
    assert result["cycles"][0]["after_state_timing"]["captured_at"] is None


def test_context_recorder_tracks_actual_handoff_and_capture(fitted, tmp_path):
    from datetime import timedelta
    policy = LearnedJointVOIPolicy(ConservativeJointReturnPredictor(fitted[0], strength=0.),
                                   lambda_cost={"money": 1.})
    rt = LearnedVOIRuntime(policy, seed=17, policy_arm="fixed_order",
        exploration_limits={"money": .05, "requests": 2}, max_exploration_actions=1)
    now = datetime(2025, 4, 30, tzinfo=timezone.utc)
    events = []

    def clock():
        nonlocal now
        now += timedelta(microseconds=10)
        return now

    def record(payload):
        events.append(payload)
        assert payload["availability_basis"] == "runtime_state_handoff"
        return {"captured_at": clock().isoformat(), "raw_path": "raw/test.body.gz",
                "raw_sha256": "a" * 64}

    def execute(*_):
        assert events[-1]["snapshot_kind"] == "before_action"
        return receipt(status="FAILED")

    result = rt.run(state(), run_id="context-clock", limits={"money": .05, "requests": 2},
        propose=lambda _: [unknown()], execute=execute, output=tmp_path / "context-clock",
        provider_version="synthetic", generator_version="synthetic", clock=clock,
        context_recorder=record)
    config = json.loads((tmp_path / "context-clock" / "configuration.json").read_text())
    timings = [config["initial_state_timing"], result["cycles"][0]["before_state_timing"],
               result["cycles"][0]["after_state_timing"], result["final_state_timing"]]
    assert events[0]["snapshot_kind"] == "initial"
    assert events[-1]["snapshot_kind"] == "final"
    for timing in timings:
        assert timing["observed_at"] == timing["available_at"] < timing["captured_at"]
        assert timing["availability_basis"] == "runtime_state_handoff"
    assert result["cycles"][0]["after_state_timing"]["parent_attempt_id"]


def test_hidden_payload_cannot_reach_generator(fitted, tmp_path):
    class Poison:
        def __deepcopy__(self, memo):
            raise AssertionError("Hidden value was copied")
    initial = state()
    initial["outcome"] = Poison()
    for group in initial["groups"].values():
        group["values"] = Poison()
    result = run(runtime(fitted[0], allowance=0), tmp_path/"run", lambda _: [], lambda *_: None, initial=initial)
    assert "outcome" not in result["cycles"][0]["public_before"]["state"]


@pytest.mark.parametrize("fault", ["hidden_return", "modeled_group", "modeled_key", "duplicate_key"])
def test_candidate_cannot_smuggle_return_or_alias_known_action(fitted, tmp_path, fault):
    candidates = [unknown()]
    if fault == "hidden_return":
        candidates[0]["future_return"] = "secret"
    elif fault == "modeled_group":
        candidates[0]["group_ids"] = ["xg"]
    elif fault == "modeled_key":
        candidates[0]["information_key"] = "xg_view"
    else:
        candidates.append(unknown("different", "news"))
    with pytest.raises(ValueError):
        run(runtime(fitted[0]), tmp_path/"run", lambda _: candidates, lambda *_: None)


def test_match_link_discovery_uses_admitted_content_and_deduplicates_request():
    generator = MatchDocumentCandidates(match_terms=["Arsenal", "Leeds"], seed_urls=["https://understat.com/league/EPL/2026"],
        allowed_hosts=["understat.com"], numeric_actions=ACTIONS, web_cost={"money": .01, "requests": 1})
    public = {"state": state(), "documents": []}
    initial = generator(public)
    public["documents"] = [{"links": [
        {"url": "https://understat.com/team/Arsenal/2026#one", "title": "Arsenal"},
        {"url": "https://understat.com/team/Arsenal/2026#two", "title": "Arsenal"},
        {"url": "https://understat.com/team/Chelsea/2026", "title": "Chelsea"},
        {"url": "https://unapproved.example/team/Arsenal", "title": "Arsenal"}]}]
    later = generator(public)
    assert len(later) == len(initial)+1
    assert any(a.get("url") == "https://understat.com/team/Arsenal/2026" for a in later)
    assert all(a.get("url", "").find("Chelsea") == -1 for a in later)
