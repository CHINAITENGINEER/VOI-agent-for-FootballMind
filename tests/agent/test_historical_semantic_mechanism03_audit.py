"""Consistency audit for the mechanism-03 historical semantic run.

This test checks only persisted provenance and replay invariants.  It does not
interpret the archived match outcome and does not turn the run into an
effectiveness or accuracy evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "var/submission/p7-historical-semantic/mechanism-03"
REPLAY = ROOT / "var/submission/p7-historical-semantic/mechanism-03-replay/replay.json"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_mechanism03_run_counts_and_prediction_are_internally_consistent():
    run = _load(RUN_DIR / "run.json")
    analysis = _load(RUN_DIR / "analysis.json")

    cycles = run["cycles"]
    observations = run["observations"]
    assert run["status"] == "completed"
    assert len(cycles) == 6
    assert len(observations) == 5
    assert len(run["http_receipts"]) == 6
    assert len(run["semantic_calls"]) == 4
    assert run["configuration"]["request_budget"] == 6

    outcomes = [item["outcome"] for item in observations]
    assert outcomes.count("supports") == 1
    assert sum(
        item["outcome"] == "supports" and item["applicability"] == "match_specific"
        for item in observations
    ) == 1
    assert outcomes.count("no_answer") == 4
    assert all(item["semantic_classifier_verified_truth"] is False for item in observations)

    spent = [item["requests_spent_after"] for item in observations]
    assert spent == [2, 3, 4, 5, 6]
    assert cycles[-1]["remaining_requests"] == 0
    assert cycles[-1]["decision"]["stop_reason"] == "request_budget_exhausted"

    assert run["initial_prediction"] == analysis["initial_prediction"]
    assert run["final_prediction"] == analysis["final_prediction"]
    assert run["final_prediction"] == cycles[-1]["decision"]["prediction_nominal"]
    assert run["initial_prediction"] == [
        0.5348837209302325,
        0.3023255813953488,
        0.16279069767441862,
    ]
    assert run["final_prediction"] == [
        0.5073056341887977,
        0.3104133506278784,
        0.1822810151833239,
    ]

    checks = analysis["no_answer_checks"]
    assert len(checks) == 4
    assert all(item["outcome"] == "no_answer" for item in checks)
    assert all(item["unchanged_within_1e_12"] is True for item in checks)


def test_mechanism03_replay_and_controls_do_not_claim_empirical_gain():
    run = _load(RUN_DIR / "run.json")
    controls = _load(RUN_DIR / "controls.json")
    replay = _load(REPLAY)

    assert replay == {
        "cycles": 6,
        "http_requests": 0,
        "scope": "exact_math_replay_of_frozen_proposals_and_saved_observations_not_reclassification",
        "semantic_calls": 0,
        "status": "passed",
    }
    assert controls["status"] == "completed"
    assert controls["different_action_choices"] is True
    assert controls["counterfactual_returns_used"] == 0
    assert controls["future_returns_used"] == 0
    assert controls["real_match_result_used"] is False
    assert controls["empirical_gain"] is None
    assert controls["outcome_accuracy"] is None
    assert all(result["empirical_gain"] is None for result in controls["results"].values())
    assert run["configuration"]["transport"]["network_requests"] == 0
    assert run["configuration"]["study_role"] == "zero_loss_cost_mechanism_exercise_not_policy_success"

    audit = _load(RUN_DIR / "audit.json")
    review = audit["semantic_boundary_review"]
    assert review["source_facts"]
    assert review["not_established_by_text"]
    assert "conditional diagnostics" in review["current_model_risk"]
