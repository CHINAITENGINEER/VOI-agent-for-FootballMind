"""Controls are conditional replays, not football-effectiveness tests."""
from copy import deepcopy

from football_agent.semantic_scenario_controls import compare_controls
from football_agent.semantic_scenario_model import SemanticScenarioModel


def fixture():
    proposal = {"schema": "semantic-scenario-proposal-v1",
        "hypotheses": [
            {"id": "ready", "description": "constructed", "prior": .5,
             "outcome_probabilities": [.7, .2, .1]},
            {"id": "limited", "description": "constructed", "prior": .5,
             "outcome_probabilities": [.2, .3, .5]}],
        "actions": [
            {"action_id": "a", "question": "constructed A", "observation_group": "ga",
             "likelihoods": [{"hypothesis_id": "ready", "probabilities": [.8, .1, .1]},
                             {"hypothesis_id": "limited", "probabilities": [.1, .8, .1]}]},
            {"action_id": "b", "question": "constructed B", "observation_group": "gb",
             "likelihoods": [{"hypothesis_id": "ready", "probabilities": [.7, .2, .1]},
                             {"hypothesis_id": "limited", "probabilities": [.2, .7, .1]}]}],
        "assumptions": ["constructed test only"]}
    model = SemanticScenarioModel(proposal, [.45, .25, .30], request_loss_cost=0,
                                  contamination_levels=(0., .25, .5))
    bindings = [{"action_id": "a", "request_cost": 1, "tool": "read_url", "url": "https://example/a"},
                {"action_id": "b", "request_cost": 1, "tool": "read_url", "url": "https://example/b"}]
    panel = {"a": {"action_id": "a", "outcome": "supports"},
             "b": {"action_id": "b", "outcome": "refutes"}}
    return model.manifest(), bindings, panel


def test_controls_replay_same_panel_without_result_or_gain():
    manifest, bindings, panel = fixture()
    result = compare_controls(manifest, bindings, panel, request_budget=2, fixed_order=["a", "b"])
    assert result["status"] == "completed"
    assert result["real_match_result_used"] is False
    assert result["empirical_gain"] is None
    assert result["future_returns_used"] == result["counterfactual_returns_used"] == 0
    assert set(result["results"]) == {"adaptive_voi", "fixed_order", "no_belief_update", "no_acquisition"}
    assert result["results"]["no_acquisition"]["actual_read_entries"] == 0
    assert result["results"]["fixed_order"]["actual_read_entries"] == 2
    assert result["terminal_entropy_is_accuracy"] is False


def test_missing_selected_panel_is_incomplete_and_never_fabricates_return():
    manifest, bindings, panel = fixture()
    del panel["b"]
    result = compare_controls(manifest, bindings, panel, request_budget=2, fixed_order=["a", "b"])
    assert result["status"] == "incomplete"
    incomplete = [row for row in result["results"].values() if row["status"] == "incomplete"]
    assert incomplete
    assert all(row["stop_reason"] == "missing_panel_return" for row in incomplete)
    # The arm may only expose returns that were already supplied in ``panel``.
    # In particular, a missing selected cell must not be filled by a future or
    # counterfactual value hidden inside a per-arm result.
    assert result["future_returns_used"] == 0
    assert result["counterfactual_returns_used"] == 0
    assert all("returned_observation" not in event
               for row in incomplete for event in row["trace"]
               if event.get("stop_reason") == "missing_panel_return")


def test_paired_panels_change_only_adaptive_next_action():
    """A different *constructed* answer must reroute the next VoI action.

    This is the smallest control-level witness for the project's central
    claim.  The match, candidate menu, costs and frozen model are identical;
    only the already-supplied answer for action ``a`` changes.  Adaptive VoI
    must use that posterior change, while fixed-order and the frozen-belief
    ablation must not.  The panel supplies all three *constructed* returns up
    front, but each policy may consume only the returns for the actions it
    selected.  It is mechanism evidence, not real-source or football-
    effectiveness evidence.
    """
    proposal = {"schema": "semantic-scenario-proposal-v1",
        "hypotheses": [
            {"id": "h0", "description": "constructed", "prior": .5,
             "outcome_probabilities": [.65, .20, .15]},
            {"id": "h1", "description": "constructed", "prior": .5,
             "outcome_probabilities": [.15, .25, .60]}],
        "actions": [
            {"action_id": "a", "question": "constructed first observation", "observation_group": "ga",
             "likelihoods": [
                 {"hypothesis_id": "h0", "probabilities": [.11133969590884814, .7386603040911518, .15]},
                 {"hypothesis_id": "h1", "probabilities": [.41736425992866133, .43263574007133865, .15]}]},
            {"action_id": "b", "question": "constructed branch B", "observation_group": "gb",
             "likelihoods": [
                 {"hypothesis_id": "h0", "probabilities": [.8150330367420804, .03496696325791959, .15]},
                 {"hypothesis_id": "h1", "probabilities": [.6954310617971666, .15456893820283335, .15]}]},
            {"action_id": "c", "question": "constructed branch C", "observation_group": "gc",
             "likelihoods": [
                 {"hypothesis_id": "h0", "probabilities": [.40619433365095814, .44380566634904184, .15]},
                 {"hypothesis_id": "h1", "probabilities": [.21946299270555336, .6305370072944466, .15]}]}],
        "assumptions": ["constructed paired-panel mechanism test"]}
    model = SemanticScenarioModel(proposal, [.4, .3, .3], request_loss_cost=0,
                                  contamination_levels=(0., .25, .5))
    bindings = [{"action_id": action_id, "request_cost": 1, "tool": "read_url",
                 "url": "https://example/" + action_id} for action_id in ("a", "b", "c")]

    def run(first_outcome):
        panel = {action_id: {"action_id": action_id, "outcome": "no_answer"}
                 for action_id in ("a", "b", "c")}
        panel["a"]["outcome"] = first_outcome
        original_panel = deepcopy(panel)
        result = compare_controls(model.manifest(), bindings, panel, request_budget=2,
                                fixed_order=["a", "b", "c"])
        assert panel == original_panel
        return result

    supporting = run("supports")
    refuting = run("refutes")
    adaptive_support = supporting["results"]["adaptive_voi"]
    adaptive_refute = refuting["results"]["adaptive_voi"]
    assert adaptive_support["selected_actions"] == ["a", "c"]
    assert adaptive_refute["selected_actions"] == ["a", "b"]
    assert adaptive_support["selected_actions"] != adaptive_refute["selected_actions"]
    assert supporting["results"]["fixed_order"]["selected_actions"] == ["a", "b"]
    assert refuting["results"]["fixed_order"]["selected_actions"] == ["a", "b"]
    assert supporting["results"]["no_belief_update"]["selected_actions"] == ["a", "c"]
    assert refuting["results"]["no_belief_update"]["selected_actions"] == ["a", "c"]
    assert adaptive_support["posterior_final"] != adaptive_refute["posterior_final"]
    assert (supporting["results"]["no_belief_update"]["posterior_final"]
            == refuting["results"]["no_belief_update"]["posterior_final"])
    for result in (supporting, refuting):
        assert result["future_returns_used"] == result["counterfactual_returns_used"] == 0
        for policy in ("adaptive_voi", "fixed_order", "no_belief_update"):
            arm = result["results"][policy]
            assert arm["requests_spent"] == 2
            assert arm["empirical_gain"] is None
            selected_steps = [event for event in arm["trace"] if event.get("actual_read_entry")]
            assert len(selected_steps) == 2
            assert all(event["selected_robust_conditional_net_value"] > 0
                       for event in selected_steps)
