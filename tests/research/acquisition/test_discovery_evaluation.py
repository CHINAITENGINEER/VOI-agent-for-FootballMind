"""Evaluation isolation checks: no hidden URLs, outcome peeking or phantom links."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
EVALUATION = ROOT / "var/parallel/agent/round179/evaluation"


def _load(name):
    path = EVALUATION / (name + ".py")
    spec = importlib.util.spec_from_file_location("round179_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scenario():
    return _load("scenario")


@pytest.fixture(scope="module")
def evaluation(scenario):
    # Script-style sibling import is deliberately local to the artifact runner.
    previous = sys.modules.get("scenario")
    sys.modules["scenario"] = scenario
    try:
        return _load("run_evaluation")
    finally:
        if previous is None:
            sys.modules.pop("scenario", None)
        else:
            sys.modules["scenario"] = previous


@pytest.fixture(scope="module")
def frozen(evaluation):
    text = evaluation.TextReturnModel.load(EVALUATION / "frozen-text-model.json")
    discovery = evaluation.DiscoveryReturnModel.load(EVALUATION / "final-request-class/frozen-discovery-model.json", text_model=text)
    heldout = json.loads((EVALUATION / "heldout-matches.json").read_text(encoding="utf-8"))
    return text, discovery, heldout


def test_simulator_rejects_undiscovered_article_and_exposes_exact_actual_menu(scenario):
    rows = scenario.matches(100, seed=17902, split="test")
    row = next(r for r in rows if r["discovery_status"] == "links")
    environment = scenario.EpisodeEnvironment(row)
    hidden = row["hidden_actions"]["alpha"]
    assert hidden["url"] not in json.dumps(environment.menu())
    with pytest.raises(ValueError, match="before.*exposed"):
        environment.execute(hidden)
    _, discovered = environment.execute(scenario.DISCOVERY)
    assert hidden in discovered
    response, _ = environment.execute(hidden)
    assert response == row["article_response"]
    with pytest.raises(ValueError, match="before.*exposed"):
        environment.execute(hidden)


def test_outcome_label_changes_scores_but_not_actions_or_probabilities(evaluation, frozen):
    text, discovery, heldout = frozen
    original = next(r for r in heldout if r["context"] == "competitive" and r["discovery_status"] == "links")
    altered = deepcopy(original)
    altered["outcome"] = (original["outcome"] + 1) % 3
    initial = evaluation.public_initial_state(text, original)
    left = evaluation.run_policy(text, discovery, original, initial, "strict_two_step", .165, 17903)
    right = evaluation.run_policy(text, discovery, altered, initial, "strict_two_step", .165, 17903)
    assert left["paid_queries"] >= 2
    assert left["final_probabilities"] == right["final_probabilities"]
    assert [t["action"] for t in left["transitions"]] == [t["action"] for t in right["transitions"]]
    assert [t["candidate_values"] for t in left["transitions"]] == [t["candidate_values"] for t in right["transitions"]]
    assert left["outcome"] != right["outcome"]


def test_failed_discovery_cannot_execute_training_prototype_or_hidden_article(evaluation, frozen):
    text, discovery, heldout = frozen
    failed = next(r for r in heldout if r["discovery_status"] == "failure")
    initial = evaluation.public_initial_state(text, failed)
    result = evaluation.run_policy(text, discovery, failed, initial, "fixed_discover_then_alpha_then_beta", .165, 17903)
    assert result["paid_queries"] == 1
    assert result["cost"] == .025
    assert result["final_probabilities"] == pytest.approx(result["initial_probabilities"], abs=1e-12)
    assert result["transitions"][0]["newly_discovered_actions"] == []
    assert result["terminal"] == "observed_menu_exhausted"
