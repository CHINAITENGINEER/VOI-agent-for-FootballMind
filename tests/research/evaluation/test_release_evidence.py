"""Integrity tests for the release evidence boundary, not model fitting tests."""
from copy import deepcopy
import json
import math
from pathlib import Path
import socket

import pytest

from football_research.evaluation import release_evidence as release


def history_row(mid, p, *, seed, coefficient=2.0):
    loss = -math.log(p)
    probabilities = [p, (1-p)/2, (1-p)/2]
    return {"match_id": mid, "final_probabilities": probabilities, "initial_probabilities": [.5, .25, .25],
        "log_loss": loss, "brier_score": sum((q-int(i == 0))**2 for i,q in enumerate(probabilities)),
        "cost": {"money": .02, "requests": 1}, "actions": ["results_view"], "limits": {"money": .03, "requests": 1},
        "fold": 0, "lambda_money": coefficient, "utility_loss": loss + coefficient*.02,
        "model_name": "model", "strategy": "uniform-random", "seed": seed}


def test_seed_averaging_keeps_match_count_and_both_cost_objectives():
    rows = [history_row(mid, p, seed=seed) for mid, probs in [("a", [.8, .6]), ("b", [.4, .2])]
            for seed,p in enumerate(probs)]
    metric, = release.summarize_historical_records(rows, {"a": 0, "b": 0}, fold_lambdas={0: 2.0}, old_report_objective=True)
    expected = sum(-math.log(p) for p in [.8, .6, .4, .2])/4
    assert metric["independent_matches"] == 2
    assert metric["recorded_trajectories"] == 4
    assert metric["mean_log_loss"] == pytest.approx(expected)
    assert metric["mean_reported_objective"] == pytest.approx(expected + .5*.02)
    assert metric["mean_policy_lambda_objective"] == pytest.approx(expected + 2*.02)


@pytest.mark.parametrize("problem", ["duplicate", "missing_seed", "outcome", "budget"])
def test_historical_integrity_rejects_inflated_or_unfair_records(problem):
    rows = [history_row(mid, p, seed=seed) for mid, probs in [("a", [.8, .6]), ("b", [.4, .2])]
            for seed,p in enumerate(probs)]
    if problem == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif problem == "missing_seed":
        rows.pop()
    elif problem == "outcome":
        rows[0]["log_loss"] = .01
    else:
        rows[0]["limits"]["requests"] = 0
    with pytest.raises(release.EvidenceError):
        release.summarize_historical_records(rows, {"a": 0, "b": 0}, fold_lambdas={0: 2.0}, old_report_objective=False)


def synthetic_row():
    before, after = [.5, .25, .25], [.2, .4, .4]
    return {"match_id": "a", "policy": "fixed", "budget": .08, "cost": .08, "paid_queries": 1,
            "initial_probabilities": before, "final_probabilities": after, "outcome": 0,
            "log_loss": -math.log(.2), "objective": -math.log(.2)+.08,
            "transitions": [{"posterior_before": before, "posterior_after": after,
                             "realized_log_loss_reduction": math.log(.2/.5), "update_status": "updated"}]}


def test_synthetic_harmful_update_remains_in_cost_quality_average():
    metric, counts = release.summarize_synthetic_records([synthetic_row()], expected_matches={"a"}, version=178, cost_per_query=.08)
    assert metric[0]["mean_objective"] == pytest.approx(-math.log(.2)+.08)
    assert counts == {"negative_realized_updates": 1, "negative_beyond_roundoff": 1, "unknown_updates": 0}
    changed = synthetic_row()
    changed["transitions"][0]["realized_log_loss_reduction"] *= -1
    with pytest.raises(release.EvidenceError, match="gain changed"):
        release.summarize_synthetic_records([changed], expected_matches={"a"}, version=178, cost_per_query=.08)


def test_declared_sha_rejects_missing_changed_and_external_paths(tmp_path):
    path = tmp_path / "frozen.json"
    path.write_text("{}", encoding="utf-8")
    expected = release.Inputs(tmp_path).bind("frozen.json")
    path.write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(release.EvidenceError, match="SHA mismatch"):
        release.Inputs(tmp_path).bind("frozen.json", expected)
    with pytest.raises(release.EvidenceError, match="Missing evidence"):
        release.Inputs(tmp_path).bind("missing.json")
    with pytest.raises(release.EvidenceError, match="escapes repository"):
        release.Inputs(tmp_path).bind("../outside.json")


def test_saved_claim_cannot_promote_true_after_recomputation(tmp_path, monkeypatch):
    original = {"schema": release.SCHEMA, "file_sha256": {}, "research_flags": {"real_text_numeric_voi_benefit_proven": False}}
    saved = deepcopy(original)
    saved["research_flags"]["real_text_numeric_voi_benefit_proven"] = True
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(saved), encoding="utf-8")
    monkeypatch.setattr(release, "build_evidence_summary", lambda _: original)
    with pytest.raises(release.EvidenceError, match="differs from complete"):
        release.verify_evidence_summary(tmp_path, path)


def test_frozen_release_recomputes_with_network_disabled(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    summary = root / release.DEFAULT_OUTPUT / "evidence-summary.json"
    if not summary.exists():
        pytest.skip("Frozen release artifacts are not present in this source-only checkout")
    def no_network(*args, **kwargs):
        raise AssertionError("Release evidence tried to access the network")
    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    result = release.verify_evidence_summary(root, summary)
    assert result["status"] == "passed"
    assert result["sample_accounting"]["round177_additional_independent_matches"] == 0
    assert not any(result["research_flags"].values())
