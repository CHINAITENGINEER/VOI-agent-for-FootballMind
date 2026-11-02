"""Synthetic core-mechanism integration; never a real football performance claim."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import importlib.util
import json
from pathlib import Path

import pytest
import requests

from football_research.acquisition.text_returns import TextReturnModel
from test_joint_returns import fitted, state as numeric_state


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "var/parallel/agent/round178/evaluation"
_spec = importlib.util.spec_from_file_location("round178_synthetic_text_scenario", EXPERIMENT / "scenario.py")
scenario = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scenario)


@pytest.fixture(scope="module")
def text_model():
    """Load the pre-registered train-only model, which is an intentional test artifact."""
    return TextReturnModel.load(EXPERIMENT / "frozen-text-model.json")


@pytest.fixture(scope="module")
def heldout():
    return scenario.matches(300, seed=17802, split="test")


def _known_context(model, row):
    state = model.start(row["prior_probabilities"], row["match_id"], row["prediction_cutoff"])
    decision = model.enumerate(state, scenario.ACTIONS["D"], cost=0.)
    update = model.update(state, scenario.ACTIONS["D"], row["returns"]["D"], decision=decision)
    assert update["status"] == "updated"
    return update["state"]


def test_same_injury_query_learns_different_value_from_already_read_context(text_model, heldout):
    values = {}
    for context in ("competitive", "resolved"):
        row = next(row for row in heldout if row["context"] == context)
        known = _known_context(text_model, row)
        decision = text_model.enumerate(known, scenario.ACTIONS["A"], cost=scenario.COST)
        assert decision["status"] == "supported"
        assert decision["value"]["coherence_error"] < 1e-12
        values[context] = decision["value"]["net_voi"]
    assert values["competitive"] > 0
    assert values["resolved"] < 0
    assert values["competitive"] > values["resolved"] + .1


def test_correlated_republication_is_conditioned_and_updates_match_frozen_branches(text_model, heldout):
    for context in ("competitive", "resolved"):
        row = next(row for row in heldout if row["context"] == context)
        known = _known_context(text_model, row)
        first = text_model.enumerate(known, scenario.ACTIONS["A"], cost=scenario.COST)
        updated = text_model.update(known, scenario.ACTIONS["A"], row["returns"]["A"], decision=first)
        branch = next(item for item in first["branches"] if item["branch_id"] == updated["observed_token"])
        assert updated["state"]["current_probabilities"] == branch["posterior"]
        duplicate = text_model.enumerate(updated["state"], scenario.ACTIONS["B"], cost=scenario.COST)
        assert duplicate["status"] == "supported"
        assert duplicate["value"]["gross_evsi"] == pytest.approx(0., abs=1e-12)
        repeated = text_model.update(updated["state"], scenario.ACTIONS["B"], row["returns"]["B"], decision=duplicate)
        assert repeated["status"] == "updated"
        assert repeated["state"]["current_probabilities"] == pytest.approx(branch["posterior"], abs=1e-12)
        assert len(repeated["state"]["history"]) == 3


def test_new_source_oov_and_unseen_history_remain_unknown_not_zero_gain(text_model, heldout):
    row = heldout[0]
    known = _known_context(text_model, row)
    source = {"tool": "read_url", "url": "https://new-source.synthetic.example/injury-bulletin"}
    unseen = text_model.enumerate(known, source, cost=scenario.COST)
    assert unseen["status"] == "unknown"
    assert unseen["value"]["gross_evsi"] is None
    assert unseen["value"]["net_voi"] is None
    article = {"status": "OK", "kind": "article", "text": "zzqwxflorp vvvtrang"}
    supported = text_model.enumerate(known, scenario.ACTIONS["A"], cost=scenario.COST)
    after_oov = text_model.update(known, scenario.ACTIONS["A"], article, decision=supported)
    assert after_oov["status"] == "unknown"
    assert after_oov["state"]["current_probabilities"] == known["current_probabilities"]
    assert after_oov["state"]["history"][-1]["unsupported"] is True
    next_query = text_model.enumerate(after_oov["state"], scenario.ACTIONS["C"], cost=scenario.COST)
    assert next_query["status"] == "unknown"
    assert next_query["reason"] == "unsupported_observed_history"
    assert next_query["value"]["net_voi"] is None


def test_train_test_match_split_and_frozen_encoder_survive_heldout_inference(text_model, heldout):
    assert not set(text_model.metadata["training_match_ids"]) & {row["match_id"] for row in heldout}
    before = deepcopy(text_model.text_codebook.payload)
    for row in heldout[:6]:
        known = _known_context(text_model, row)
        decision = text_model.enumerate(known, scenario.ACTIONS["A"], cost=scenario.COST)
        text_model.update(known, scenario.ACTIONS["A"], {"status": "OK", "kind": "article", "text": "uniqueheldoutvocabularytoken"}, decision=decision)
    assert text_model.text_codebook.payload == before
    assert "uniqueheldoutvocabularytoken" not in before["vocabulary"]
    assert text_model.metadata["evidence_origin"] == "constructed_mechanism"
    assert text_model.metadata["production_ready"] is False


def test_saved_comparison_preserves_all_policies_budgets_and_negative_updates():
    results = json.loads((EXPERIMENT / "policy-records.json").read_text(encoding="utf-8"))
    acceptance = json.loads((EXPERIMENT / "acceptance.json").read_text(encoding="utf-8"))
    assert len(results) == 300*4*2
    assert {row["budget"] for row in results} == {.08, .24}
    assert {row["policy"] for row in results} == {"strict_greedy_positive_voi", "no_query", "fixed_A_B_C", "uniform_without_replacement"}
    assert acceptance["all_successful_updates_match_frozen_branch"]
    assert acceptance["all_queries_within_budget"]
    assert acceptance["all_strict_queries_have_positive_computed_voi"]
    assert acceptance["negative_realized_updates_retained"] > 0
    assert any(transition["realized_log_loss_reduction"] < 0 for row in results for transition in row["transitions"])
    # Deliberately no assertion that any policy wins heldout prediction scores.


def _run_text_runtime(base_model, text_model, row, output, *, future_available=False,
                      saved_receipts=None):
    """Execute the production runner over synthetic archived provider bytes."""
    from football_agent.learned_voi_runtime import LearnedVOIRuntime
    from football_agent.text_voi_policy import TextVOIPolicy
    from football_research.acquisition.joint_voi import LearnedJointVOIPolicy
    from football_spider.storage.raw_archive import RawArchive

    initial = numeric_state()
    initial.update(match_id=row["match_id"], prediction_cutoff=row["prediction_cutoff"])
    initial["base"]["kickoff"] = row["kickoff"]
    clock = datetime.fromisoformat(row["prediction_cutoff"]) - timedelta(hours=2)
    policy = TextVOIPolicy(LearnedJointVOIPolicy(base_model, lambda_cost={"money": 1.}), text_model)
    actions = [{**deepcopy(action), "action_id": "synthetic-text-"+key,
                "information_key": "synthetic-text-"+key, "group_ids": [],
                "available": True, "cost_status": "simulated", "cost": {"money": 0. if key == "D" else scenario.COST, "requests": 1.}}
               for key, action in scenario.ACTIONS.items()]
    calls, receipts, proposals = [], [], []

    def propose(public):
        proposals.append(deepcopy(public))
        # The menu exposes URLs only; no outcome or unacquired article body.
        assert "outcome" not in public and "returns" not in public
        return deepcopy(actions)

    def execute(action, public):
        assert (output/"configuration.json").is_file()
        assert (output/"text-response-model.json").is_file()
        calls.append(deepcopy(action))
        if saved_receipts is not None:
            saved = deepcopy(saved_receipts[len(calls)-1])
            assert saved["receipt"]["selected_action"] == action
            receipts.append(saved)
            return saved
        key = action["action_id"].rsplit("-", 1)[1]
        response = row["returns"][key]
        body = response["text"].encode("utf-8")
        archive = RawArchive(output/"source-archive")
        observation = archive.save("synthetic_text_provider", action["url"], {}, body,
                                   status=200, observed_at=clock)
        assert archive.read_body(observation) == body
        document = {**deepcopy(response), "document_id": "synthetic-doc-"+key,
                    "url": action["url"], "source": "synthetic_text_provider",
                    "observed_at": clock.isoformat(),
                    "available_at": (clock+timedelta(hours=1) if future_available else clock).isoformat(),
                    "raw_sha256": sha256(body).hexdigest(), "links": [],
                    "source_published_at": None, "text_truncated": False}
        returned = {"status": "OK", "document": document,
                    "actual_cost": {"money": None, "requests": 1., "latency_ms": 0., "tokens": 0.},
                    "receipt": {"observation": observation.model_dump(mode="json"),
                                "selected_action": deepcopy(action),
                                "scope": "synthetic_provider_with_archived_bytes_no_network"}}
        receipts.append(deepcopy(returned))
        return returned

    runtime = LearnedVOIRuntime(policy, seed=178,
                                exploration_limits={"money": 0., "requests": 0.},
                                max_exploration_actions=0)
    run = runtime.run(initial, run_id="synthetic-text-runtime-178",
                      limits={"money": .24, "requests": 4.}, propose=propose,
                      execute=execute, output=output,
                      provider_version="synthetic-archived-text-provider-v1",
                      generator_version="synthetic-public-request-menu-v1", clock=lambda: clock,
                      run_metadata={"evidence_origin": "constructed_mechanism", "network_requests": 0},
                      evidence_match={"match_id": initial["match_id"], "home_team": "Synthetic North",
                                      "away_team": "Synthetic South", "kickoff_at": row["kickoff"],
                                      "prediction_cutoff": row["prediction_cutoff"]})
    return run, calls, receipts, proposals


def test_full_runtime_selects_positive_text_voi_executes_updates_then_replans_and_replays(
        tmp_path, fitted, text_model, heldout, monkeypatch):
    def no_network(*_args, **_kwargs):
        raise AssertionError("Synthetic replay may not use a network provider")
    monkeypatch.setattr(requests.sessions.Session, "request", no_network)
    row = next(row for row in heldout if row["context"] == "competitive")
    run, calls, receipts, proposals = _run_text_runtime(fitted[0], text_model, row, tmp_path/"original")
    assert run["schema"] == "learned-voi-open-execution-v3"
    assert len(calls) >= 2
    assert calls[0]["action_id"] == "synthetic-text-D"
    # A and its syndicated B are equally informative before either is read.
    assert any(action["action_id"] in {"synthetic-text-A", "synthetic-text-B", "synthetic-text-C"} for action in calls[1:])
    injury_calls = [action for action in calls if action["action_id"] in {"synthetic-text-A", "synthetic-text-B"}]
    assert len(injury_calls) == 1
    unchosen_copy = "synthetic-text-B" if injury_calls[0]["action_id"] == "synthetic-text-A" else "synthetic-text-A"
    after_read = next(item for item in run["cycles"][-1]["strict_decision"]["candidates"]
                      if item["action_id"] == unchosen_copy)
    assert after_read["value"]["gross_evsi"] == pytest.approx(0., abs=1e-12)
    assert after_read["value"]["net_voi"] == pytest.approx(-scenario.COST, abs=1e-12)
    assert run["exploration_actions"] == 0
    assert run["text_response_learning"]["modeled_returns"] == len(calls)
    assert run["initial_probabilities"] != run["final_probabilities"]
    for cycle in run["cycles"]:
        if cycle["selected_action"] is None:
            continue
        selected = next(item for item in cycle["strict_decision"]["candidates"]
                        if item["action_id"] == cycle["selected_action"]["action_id"])
        assert selected["value"]["net_voi"] > 0
        assert cycle["execution_status"] == "returned"
        assert cycle["update"]["status"] == "updated"
        branch = next(item for item in selected["branches"]
                      if item["branch_id"] == cycle["update"]["observed_token"])
        assert cycle["prediction_after"] == branch["posterior"]
    assert len(proposals) > len(calls)  # Includes decision after the latest return.
    assert any(item["state"]["open_text"]["history"] for item in proposals[1:])
    replayed = _run_text_runtime(fitted[0], TextReturnModel.load(tmp_path/"original/text-response-model.json"),
                                row, tmp_path/"replayed", saved_receipts=receipts)[0]
    assert replayed == run


def test_runtime_future_document_consumes_attempt_but_cannot_update_probability(
        tmp_path, fitted, text_model, heldout, monkeypatch):
    monkeypatch.setattr(requests.sessions.Session, "request",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("No HTTP")))
    run, calls, _, _ = _run_text_runtime(fitted[0], text_model, heldout[0], tmp_path/"future",
                                        future_available=True)
    assert len(calls) == 1
    assert run["reserved_cost"]["requests"] == 1
    assert run["initial_probabilities"] == run["final_probabilities"]
    assert run["cycles"][0]["admission"]["reason"] == "after_actual_return_clock"
    assert run["text_response_learning"]["unknown_returns"] == 1
    assert run["text_response_learning"]["modeled_returns"] == 0
    assert run["final_state"]["open_text"]["history"][-1]["unsupported"] is True
    assert not run["documents"]
    assert run["terminal_selection"]["stop_certificate"] is None

