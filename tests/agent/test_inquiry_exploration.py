"""Declared exploration must remain conditional, replayable and return-blind."""
from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest

from football_agent.inquiry_exploration import UniformInquiryExploration


def inputs(explorer, *, run_id="run-01", prior=0):
    run = {"run_id": run_id, "decisions": [{} for _ in range(prior)],
           "configuration": {"inquiry_exploration": explorer.configuration()}}
    response = {"decision": {"kind": "query", "candidate_id": "verify",
                             "reason": "Original proposal", "revisit_at": None},
                "candidates": [{"candidate_id": name} for name in ["verify", "fill", "known_bad"]]}
    return run, response


def test_configuration_is_frozen_roundtrips_and_public_summary_has_no_seed():
    explorer = UniformInquiryExploration(987654321)
    restored = UniformInquiryExploration.from_configuration(explorer.configuration())
    assert restored == explorer
    assert "seed" not in explorer.public_summary()
    assert "987654321" not in repr(explorer.public_summary())
    assert "987654321" not in repr(explorer)
    assert explorer.public_summary()["candidate_generation_probability"] is None
    with pytest.raises(FrozenInstanceError):
        explorer.seed = 42
    changed = explorer.configuration()
    changed["seed"] += 1
    with pytest.raises(ValueError, match="commitment mismatch"):
        UniformInquiryExploration.from_configuration(changed)
    for seed in [True, False, -1, 2**256, 1.5, "1", None]:
        with pytest.raises(ValueError, match="seed must be an integer"):
            UniformInquiryExploration(seed)


def test_replay_and_candidate_order_normalization_preserve_inputs():
    explorer = UniformInquiryExploration(73)
    run, response = inputs(explorer)
    original = deepcopy((run, response))
    chosen = explorer.select(run, response, ["verify", "fill"])
    response_reordered = {**response, "candidates": list(reversed(response["candidates"]))}
    restored = UniformInquiryExploration.from_configuration(explorer.configuration())
    assert restored.select(run, response_reordered, ["fill", "verify"]) == chosen
    assert (run, response) == original
    record = chosen["exploration_record"]
    assert record["eligible_ids"] == ["fill", "verify"]
    assert record["probabilities"] == {"fill": 0.5, "verify": 0.5}
    assert record["selected_p"] == record["probabilities"][chosen["candidate_id"]]
    assert record["scope"] == "conditional_on_generated_eligible_candidates_only"
    assert record["expected_information_value"] is None


def test_private_seed_space_explores_only_eligible_candidates_not_always_original():
    # This is a deterministic test over possible randomization states, not an
    # empirical claim about real task coverage or calibrated return probability.
    selected = set()
    for seed in range(32):
        explorer = UniformInquiryExploration(seed)
        run, response = inputs(explorer)
        chosen = explorer.select(run, response, ["verify", "fill"])
        selected.add(chosen["candidate_id"])
        assert "known_bad" not in chosen["exploration_record"]["probabilities"]
    assert selected == {"fill", "verify"}


def test_three_way_draw_rejects_biased_tail_before_recording_uniform_probability(monkeypatch):
    # For three choices, 2**256 - 1 has no partner in the equal-size modulo
    # classes; consuming the next private stream block keeps the design uniform.
    import football_agent.inquiry_exploration as module
    values = iter([2**256 - 1, 2])
    class Block:
        def __init__(self, value): self.value = value
        def digest(self): return self.value.to_bytes(32, "big")
    monkeypatch.setattr(module.hmac, "new", lambda *args: Block(next(values)))
    explorer = UniformInquiryExploration(1)
    run, response = inputs(explorer)
    response["candidates"][-1]["candidate_id"] = "refresh"
    chosen = explorer.select(run, response, ["verify", "refresh", "fill"])
    assert chosen["candidate_id"] == "verify"
    assert chosen["exploration_record"]["probabilities"] == {
        "fill": 1/3, "refresh": 1/3, "verify": 1/3}
    assert next(values, None) is None


def test_singleton_is_conditional_probability_one_and_run_index_drive_new_draws():
    explorer = UniformInquiryExploration(11)
    run, response = inputs(explorer)
    singleton = explorer.select(run, response, ["fill"])["exploration_record"]
    assert singleton["chosen_id"] == "fill" and singleton["selected_p"] == 1
    assert singleton["candidate_generation_probability"] is None
    across_indices, across_runs = set(), set()
    for index in range(20):
        run, response = inputs(explorer, prior=index)
        record = explorer.select(run, response, ["verify", "fill"])["exploration_record"]
        assert record["decision_index"] == index
        across_indices.add(record["chosen_id"])
        run, response = inputs(explorer, run_id=f"run-{index}")
        across_runs.add(explorer.select(run, response, ["verify", "fill"])["candidate_id"])
    assert across_indices == across_runs == {"fill", "verify"}


def test_selection_reads_only_frozen_config_identity_and_previous_decision_count():
    explorer = UniformInquiryExploration(9)
    run, response = inputs(explorer)
    class ReturnBlind(dict):
        def get(self, key, *args):
            assert key in {"configuration", "run_id", "decisions"}, key
            return super().get(key, *args)
        def __getitem__(self, key):
            assert key in {"configuration", "run_id", "decisions"}, key
            return super().__getitem__(key)
    class NeverRead:
        def __deepcopy__(self, memo):
            raise AssertionError("Return/state content was read")
    sealed = ReturnBlind({**run, "documents": NeverRead(), "receipts": NeverRead(),
                         "future_returns": NeverRead(), "prediction": NeverRead()})
    sealed["decisions"] = [NeverRead(), NeverRead()]
    selected = explorer.select(sealed, response, ["fill", "verify"])
    expected_run, _ = inputs(explorer, prior=2)
    assert selected == explorer.select(expected_run, response, ["fill", "verify"])


def test_nonquery_decisions_remain_unchanged_even_without_candidates_or_run_state():
    explorer = UniformInquiryExploration(9)
    for kind in ["stop", "wait", "reflect", "defer"]:
        decision = {"kind": kind, "candidate_id": None, "reason": "Already decided",
                    "revisit_at": "2026-09-28T12:00:00Z" if kind == "wait" else None}
        actual = explorer.select(None, {"decision": decision}, None)
        assert actual == {**decision, "exploration_record": None}
        assert actual is not decision


def test_query_requires_committed_configuration_unique_generated_eligible_subset():
    explorer = UniformInquiryExploration(9)
    run, response = inputs(explorer)
    for eligible in [None, [], "fill", ["fill", "fill"], ["invented"], [True], [""]]:
        with pytest.raises(ValueError):
            explorer.select(run, response, eligible)
    for bad_run in [{**run, "configuration": {}},
                    {**run, "configuration": {"inquiry_exploration": UniformInquiryExploration(10).configuration()}},
                    {**run, "run_id": ""}, {**run, "decisions": None}]:
        with pytest.raises(ValueError):
            explorer.select(bad_run, response, ["fill"])
    zero = UniformInquiryExploration(0)
    boolean_config, _ = inputs(zero)
    boolean_config["configuration"]["inquiry_exploration"]["seed"] = False
    with pytest.raises(ValueError, match="frozen run configuration"):
        zero.select(boolean_config, response, ["fill"])
    repeated = {**response, "candidates": [{"candidate_id": "fill"}, {"candidate_id": "fill"}]}
    with pytest.raises(ValueError, match="must be unique"):
        explorer.select(run, repeated, ["fill"])
    absent = {**response, "decision": {**response["decision"], "candidate_id": "invented"}}
    with pytest.raises(ValueError, match="original query"):
        explorer.select(run, absent, ["fill"])
