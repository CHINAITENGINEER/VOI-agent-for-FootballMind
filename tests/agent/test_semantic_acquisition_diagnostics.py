"""Exercise real mathematical choices under explicit, reproducible perturbations."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from football_agent.semantic_acquisition_diagnostics import (
    GRID, analyze_acquisition, render_acquisition_report, semantic_call_profile, vary_manifest,
)
from football_agent.semantic_scenario_model import SemanticScenarioModel


def make_run(*, budget=2, history=(), horizon=1):
    proposal = {"schema": "semantic-scenario-proposal-v1",
        "hypotheses": [
            {"id": "strong", "description": "constructed strong", "prior": .5,
             "outcome_probabilities": [.8, .1, .1]},
            {"id": "weak", "description": "constructed weak", "prior": .5,
             "outcome_probabilities": [.1, .1, .8]}],
        "actions": [{"action_id": name, "question": "constructed "+name, "observation_group": name,
                     "likelihoods": [{"hypothesis_id": "strong", "probabilities": [signal, .95-signal, .05]},
                                     {"hypothesis_id": "weak", "probabilities": [.95-signal, signal, .05]}]}
                    for name, signal in (("a", .85), ("b", .75))],
        "assumptions": ["Constructed probabilities, no football evidence."]}
    model = SemanticScenarioModel(proposal, [.45, .1, .45], request_loss_cost=.001,
                                  contamination_levels=(0., .25, .5), planning_horizon=horizon)
    observations = [{"action_id": a, "outcome": z} for a, z in history]
    return {"run_id": "constructed-sensitivity", "status": "completed",
            "configuration": {"planning_horizon": horizon, "request_budget": budget+len(history),
                              "match": {"match_id": "m1", "prediction_cutoff": "2026-10-09T00:00:00Z"}},
            "cycles": [{"index": 0, "model": model.manifest(), "observations": observations,
                        "remaining_requests": budget, "requests_spent": len(history),
                        "decision": model.evaluate(observations, budget)}]}


def test_downgrading_one_source_changes_the_choice_without_changing_current_prediction():
    run = make_run()
    before = deepcopy(run)
    report = analyze_acquisition(run)
    state = report["cycles"][0]
    assert state["original"]["selected_action_id"] == "a"
    changed = next(v for v in state["variations"] if v["specification"] ==
                   {"family": "observation_signal", "action_id": "a", "amount": 0.})
    assert changed["selected_action_id"] == "b"
    assert changed["changed_beyond_value_tolerance"]
    assert changed["modeled_shortfall_of_original_action"] > 0
    assert changed["prediction"] == pytest.approx(state["original"]["prediction"], abs=1e-12)
    assert run == before
    assert report["network_calls"] == report["semantic_calls"] == 0
    assert report["used_for_action_selection"] is False


@pytest.mark.parametrize("horizon", [1, 2])
def test_no_target_contrast_makes_all_queries_worthless(horizon):
    state = analyze_acquisition(make_run(horizon=horizon))["cycles"][0]
    point = next(v for v in state["variations"] if v["specification"] ==
                 {"family": "outcome_contrast", "amount": 0.})
    assert point["selected_action_id"] is None
    assert point["stop_reason"] == "no_positive_declared_lower_net_value"
    assert all(v < 0 for v in point["candidate_lower_values"].values())


@pytest.mark.parametrize("family", ["observation_signal", "answer_retention"])
def test_action_variants_preserve_missingness_and_do_not_rewrite_observed_laws(family):
    run = make_run(history=[("a", "supports")])
    state = analyze_acquisition(run)["cycles"][0]
    assert all(v["specification"].get("action_id") != "a" for v in state["variations"])
    for amount in GRID[family]:
        spec = {"family": family, "action_id": "b", "amount": amount}
        manifest = vary_manifest(run["cycles"][0]["model"], spec)
        model = SemanticScenarioModel.from_manifest(manifest)
        no_answers = [p[2] for p in model.likelihoods["b"].values()]
        assert max(no_answers) == min(no_answers)
        assert all(sum(p) == pytest.approx(1.) for p in model.likelihoods["b"].values())
        original = SemanticScenarioModel.from_manifest(run["cycles"][0]["model"])
        assert model.likelihoods["a"] == original.likelihoods["a"]
        prior = model.evaluate(run["cycles"][0]["observations"], 1)["prediction_nominal"]
        after = model.evaluate([*run["cycles"][0]["observations"], {"action_id": "b", "outcome": "no_answer"}], 0)
        assert after["prediction_nominal"] == pytest.approx(prior, abs=1e-12)


def test_prior_variants_change_odds_but_keep_external_baseline():
    manifest = make_run()["cycles"][0]["model"]
    changed = vary_manifest(manifest, {"family": "prior_odds", "hypothesis_id": "strong", "amount": 2.})
    model = SemanticScenarioModel.from_manifest(changed)
    assert model.prior == pytest.approx([2/3, 1/3])
    assert model.evaluate([], 2)["prediction_nominal"] == pytest.approx([.45, .1, .45], abs=1e-12)


def test_terminal_budget_state_is_not_counted_as_evidence_of_stability():
    result = analyze_acquisition(make_run(budget=0))
    state = result["cycles"][0]
    assert state["status"] == "not_actionable" and not state["variations"]
    assert state["reason"] == "request_budget_exhausted"
    assert "不将预算停止算作稳定性证据" in render_acquisition_report(result)


def test_saved_math_must_verify_before_sensitivity_is_computed():
    run = make_run()
    run["cycles"][0]["decision"]["candidates"][0]["lower"] = 123.
    with pytest.raises(ValueError, match="replay mismatch"):
        analyze_acquisition(run)


def test_equal_mathematical_values_do_not_prove_a_unique_preferred_query():
    run = make_run()
    cycle = run["cycles"][0]
    proposal = cycle["model"]["proposal"]
    proposal["actions"][1]["likelihoods"] = deepcopy(proposal["actions"][0]["likelihoods"])
    model = SemanticScenarioModel.from_manifest(cycle["model"])
    cycle["decision"] = model.evaluate([], cycle["remaining_requests"])
    report = analyze_acquisition(run)
    assert report["cycles"][0]["original"]["near_optimal_actions_within_tolerance"] == ["a", "b"]
    assert "不能声称首选唯一优于" in render_acquisition_report(report)


def test_unknown_specifications_and_targets_are_rejected():
    manifest = make_run()["cycles"][0]["model"]
    for spec in ({"family": "bad", "amount": 1}, {"family": "cost_multiplier", "amount": True},
                 {"family": "observation_signal", "amount": .5, "action_id": "missing"},
                 {"family": "prior_odds", "amount": 2., "hypothesis_id": "missing"}):
        with pytest.raises(ValueError):
            vary_manifest(manifest, spec)


def test_no_accepted_state_is_unavailable_and_missing_identity_is_rejected():
    run = make_run()
    run["cycles"] = []
    assert analyze_acquisition(run)["status"] == "not_available"
    run["configuration"]["match"] = {}
    with pytest.raises(ValueError, match="identity"):
        analyze_acquisition(run)


def test_profile_reports_failed_overhead_and_unknown_costs_without_inventing_savings():
    run = {"semantic_calls": [
        {"index": 0, "protocol": "p", "status": "invalid", "receipt": {"latency_seconds": 5.,
            "usage": [{"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}]}},
        {"index": 1, "protocol": "p", "status": "accepted", "receipt": {"latency_seconds": 2.}},
        {"index": 2, "protocol": "other", "status": "failed"}]}
    result = semantic_call_profile(run)
    assert result["unsuccessful_reported_seconds"] == 5.
    assert result["by_protocol"]["p"]["reported_seconds"] == 7.
    assert result["by_protocol"]["p"]["calls_without_token_usage"] == 1
    assert result["by_protocol"]["other"]["calls_without_latency"] == 1
    assert result["by_protocol"]["p"]["total_tokens"] == 110


def test_cli_writes_diagnostics_without_changing_source_or_replacing_existing_output(tmp_path):
    source, output = tmp_path/"run.json", tmp_path/"sensitivity.json"
    before = json.dumps(make_run()).encode()
    source.write_bytes(before)
    command = [sys.executable, str(Path(__file__).resolve().parents[2]/"tools/analyze_semantic_acquisition.py"),
               "--run", str(source), "--output", str(output)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert output.with_suffix(".txt").is_file()
    assert source.read_bytes() == before
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and "immutable" in result.stderr
