"""Controlled mathematical comparisons, without mocked candidate value scores."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from football_agent.semantic_scenario_model import SemanticScenarioModel
from tools.compare_semantic_horizons import _paired, analyze_frozen_run, compare, render_report


def _proposal():
    names = ("00", "01", "10", "11")
    return {"schema": "semantic-scenario-proposal-v1",
            "hypotheses": [{"id": name, "description": name, "prior": .25,
                "outcome_probabilities": [.9, .05, .05] if name[0] == name[1] else [.05, .05, .9]}
                for name in names],
            "actions": [{"action_id": "a"+str(index), "question": "Constructed context "+str(index),
                "observation_group": "g"+str(index),
                "likelihoods": [{"hypothesis_id": name,
                    "probabilities": [.97, .02, .01] if name[index] == "0" else [.02, .97, .01]}
                    for name in names]} for index in (0, 1)],
            "assumptions": ["Constructed complementary-information test, not football evidence."]}


def _run(horizon, status="completed", *, raw=None, cost=.02, budget=2):
    model = SemanticScenarioModel(raw or _proposal(), [.475, .05, .475],
                                  request_loss_cost=cost, contamination_levels=(0.,), planning_horizon=horizon)
    return {"run_id": "constructed", "status": status,
            "configuration": {"planning_horizon": horizon, "request_budget": budget,
                "match": {"match_id": "m1", "target_key": "t1", "match": "A vs B", "competition": "League",
                          "kickoff": "2026-10-10T00:00:00Z", "prediction_cutoff": "2026-10-09T00:00:00Z"}},
            "cycles": [{"index": 0, "model": model.manifest(), "observations": [],
                        "requests_spent": 0, "remaining_requests": budget,
                        "candidate_bindings": {"a0": {"tool": "read_url", "url": "https://example.test/a"}},
                        "decision": model.evaluate([], budget)}]}


def test_incomplete_runs_are_preserved_but_not_compared():
    result = compare(_run(1, status="incomplete"), _run(2))
    assert result["status"] == "not_comparable"
    assert "must complete" in result["reason"]


def test_same_model_reports_conditional_two_step_evidence():
    result = compare(_run(1), _run(2))
    assert result["status"] == "comparable"
    row = result["h2_positive_continuation_evidence"][0]
    assert row["h1"]["selected_action_id"] is None
    assert row["h2"]["planning_lower"] > 0
    assert result["scope"] == "shared_frozen_states_only"


def test_different_fixture_is_rejected():
    first, second = _run(1), _run(2)
    second["configuration"]["match"]["match_id"] = "m2"
    with pytest.raises(ValueError, match="same fixture"):
        compare(first, second)


@pytest.mark.parametrize("kind", ["proposal", "cost", "budget", "bindings"])
def test_same_match_is_insufficient_when_experimental_inputs_differ(kind):
    first, second = _run(1), _run(2)
    if kind == "proposal":
        raw = _proposal()
        raw["hypotheses"][0]["outcome_probabilities"] = [.8, .1, .1]
        second = _run(2, raw=raw)
    elif kind == "cost":
        second = _run(2, cost=.03)
    elif kind == "budget":
        second = _run(2, budget=3)
    else:
        second["cycles"][0]["candidate_bindings"]["a0"]["url"] = "https://example.test/other"
    result = compare(first, second)
    assert result["status"] == "not_comparable"
    assert "differ" in result["reason"]


def test_missing_identity_and_empty_complete_run_cannot_prove_an_effect():
    first, second = _run(1), _run(2)
    first["configuration"]["match"] = second["configuration"]["match"] = {}
    with pytest.raises(ValueError, match="Missing fixture"):
        compare(first, second)
    first, second = _run(1), _run(2)
    second["cycles"] = []
    assert compare(first, second)["status"] == "not_comparable"


def test_frozen_ablation_preserves_input_and_reports_no_real_calls():
    source = _run(1)
    before = deepcopy(source)
    result = analyze_frozen_run(source)
    assert source == before
    assert result["network_calls"] == result["semantic_calls"] == 0
    assert result["changed_choices"] == result["h2_acquisition_where_h1_stops"] == 1
    row = result["comparisons"][0]
    assert row["h1"]["prediction"] == row["h2"]["prediction"]
    assert row["h1"]["selected_action_id"] is None
    assert row["h2"]["immediate_net"] == pytest.approx(-.02)
    assert row["h2"]["planning_lower"] > 0
    assert "not_counterfactual_rollout" in result["scope"]


def test_one_remaining_request_does_not_manufacture_continuation():
    result = analyze_frozen_run(_run(1, budget=1))
    assert result["changed_choices"] == result["h2_acquisition_where_h1_stops"] == 0


def test_equal_valued_action_change_is_explicitly_labeled_as_near_tie():
    # Both action orders in this actual XOR computation have the same h2 value.
    # Choose the other tied action in the comparison input; never replace scores.
    second = _run(2)["cycles"][0]["decision"]
    first = deepcopy(second)
    first["selected_action_id"] = next(c["action_id"] for c in first["candidates"]
                                       if c["action_id"] != second["selected_action_id"])
    row = _paired(0, first, second)
    assert row["choice_changed"] and row["changed_choice_is_near_tie"]
    assert abs(row["h2_value_delta_vs_h1_action_under_same_h2_policy"]) <= row["value_tolerance"]
    assert not row["h2_enables_acquisition_where_h1_stops"]
    assert "不能将顺序变化作为优势证据" in render_report({"status": "completed", "comparisons": [row]})


def test_tampered_saved_value_is_rejected_before_comparison():
    run = _run(1)
    run["cycles"][0]["decision"]["candidates"][0]["lower"] = 999
    with pytest.raises(ValueError, match="replay mismatch"):
        analyze_frozen_run(run)


def test_no_accepted_state_remains_unavailable():
    run = _run(1, status="incomplete")
    run["cycles"] = []
    result = analyze_frozen_run(run)
    assert result["status"] == "not_available"
    assert result["comparisons"] == []


def test_partial_run_can_analyze_only_already_accepted_states():
    result = analyze_frozen_run(_run(1, status="incomplete"))
    assert result["status"] == "completed"
    assert result["source_run_status"] == "incomplete"
    assert len(result["comparisons"]) == 1


def test_cli_generates_readable_artifact_and_refuses_overwrite(tmp_path):
    source, output = tmp_path/"run.json", tmp_path/"ablation.json"
    source.write_text(json.dumps(_run(1)), encoding="utf8")
    command = [sys.executable, str(Path(__file__).resolve().parents[2]/"tools/compare_semantic_horizons.py"),
               "--run", str(source), "--output", str(output)]
    p = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert p.returncode == 0, p.stderr
    assert output.with_suffix(".txt").exists()
    assert json.loads(output.read_text(encoding="utf8"))["h2_acquisition_where_h1_stops"] == 1
    second = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert second.returncode != 0
    assert "immutable" in second.stderr
