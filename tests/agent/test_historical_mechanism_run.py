"""Audit invariants for the historical semantic mechanism run.

These checks deliberately validate the recorded execution contract rather than
claiming forecast accuracy.  The run is a local replay of archived documents,
so the test must catch accidental result leakage, missing observations, and a
report that no longer agrees with the raw trace.
"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "var/submission/p7-historical-semantic/mechanism-03"
REPLAY = ROOT / "var/submission/p7-historical-semantic/mechanism-03-replay/replay.json"


def _load(name):
    return json.loads((RUN_DIR / name).read_text(encoding="utf-8"))


def test_mechanism03_trace_has_one_informative_return_and_four_no_answers():
    run = _load("run.json")
    observations = run["observations"]
    assert run["status"] == "completed"
    assert len(run["cycles"]) == 6
    assert len(run["http_receipts"]) == 6
    assert len(run["semantic_calls"]) == 4
    assert len(observations) == 5

    outcomes = [row["outcome"] for row in observations]
    assert outcomes.count("supports") == 1
    assert outcomes.count("refutes") == 0
    assert outcomes.count("no_answer") == 4

    support = next(row for row in observations if row["outcome"] == "supports")
    assert support["applicability"] == "match_specific"
    assert support["semantic_classifier_verified_truth"] is False
    quotes = {item["quote"] for item in support["evidence"]}
    assert any("Daniel James" in quote for quote in quotes)
    assert any("Bornauw" in quote for quote in quotes)


def test_mechanism03_posterior_change_is_traceable_to_support_observation():
    run = _load("run.json")
    initial = run["initial_prediction"]
    final = run["final_prediction"]
    assert len(initial) == len(final) == 3
    assert abs(sum(initial) - 1.0) < 1e-12
    assert abs(sum(final) - 1.0) < 1e-12
    assert any(abs(left - right) > 1e-9 for left, right in zip(initial, final))

    # The persisted terminal model records exactly which observations update
    # beliefs.  no_answer must not be allowed to move the posterior.
    terminal = run["terminal"]
    assert [row["outcome"] for row in terminal["observations"]] == [
        "no_answer", "supports", "no_answer", "no_answer", "no_answer"
    ]
    assert terminal["observations"][0]["updates_belief"] is False
    assert terminal["observations"][1]["updates_belief"] is True
    assert all(row["updates_belief"] is False for row in terminal["observations"][2:])
    assert terminal["statistically_calibrated"] is False
    assert terminal["global_stop_certificate"] is False


def test_mechanism03_replay_and_controls_do_not_claim_empirical_gain():
    replay = json.loads(REPLAY.read_text(encoding="utf-8"))
    assert replay["status"] == "passed"
    assert replay["http_requests"] == 0
    assert replay["semantic_calls"] == 0

    controls = _load("controls.json")
    assert controls["status"] == "completed"
    assert controls["counterfactual_returns_used"] == 0
    assert controls["future_returns_used"] == 0
    assert controls["real_match_result_used"] is False
    assert controls["empirical_gain"] is None
    assert controls["outcome_accuracy"] is None
    assert controls["different_action_choices"] is True

    for result in controls["results"].values():
        assert result["used_real_match_result"] is False
        assert result["empirical_gain"] is None
        assert result["terminal_entropy_is_accuracy"] is False
