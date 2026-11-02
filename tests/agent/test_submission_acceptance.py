"""Submission acceptance checks behavior, not prediction scores or case labels."""
from copy import deepcopy
import json
import socket

import pytest

from tools import check_footballmind_submission as acceptance


def completed(match_id="m1", *, direct_stop=False, unknown=False):
    limits = {"money": .05, "requests": 2}
    cost = {"money": .02, "requests": 1}
    initial = [.4, .3, .3]
    posterior = [.5, .3, .2]
    before = {"match_id": match_id, "base": {"home_team_id": "home", "away_team_id": "away"},
              "groups": {"elo": {"status": "not_acquired", "values": {}},
                         "xg": {"status": "not_acquired", "values": {}}}, "action_history": []}
    after = deepcopy(before)
    after["groups"]["elo"] = {"status": "observed", "values": {"home_elo": 1500, "away_elo": 1480}}
    after["action_history"] = [{"action_id": "results_view"}]
    candidate = {"action_id": "results_view", "feasible": True, "group_ids": ["elo"],
                 "cost": cost, "value": {"status": "COMPUTED", "net_voi": .08}}
    stop = {"decision": "defer", "reason": "candidate_generation_incomplete"}
    cycle = {"step": 0, "state_before": before, "state_after": after,
             "decision": {"prediction": {"probabilities": initial}, "candidates": [candidate],
                          "selection": {"decision": "acquire", "candidate_id": "results_view"},
                          "physical_budget": {"limits": limits, "spent": {}}},
             "update": {"action_id": "results_view", "prediction_before": initial, "prediction_after": posterior}, "charged_cost": cost}
    terminal = {"step": 1, "state_before": after, "state_after": None,
                "decision": {"prediction": {"probabilities": posterior}, "candidates": [], "selection": stop,
                             "physical_budget": {"limits": limits, "spent": cost}},
                "update": None, "charged_cost": None}
    cycles, actions, spent = [cycle, terminal], ["results_view"], cost
    if direct_stop:
        terminal["state_before"] = before
        terminal["decision"]["prediction"]["probabilities"] = initial
        terminal["decision"]["physical_budget"]["spent"] = {}
        posterior, cycles, actions, spent = initial, [terminal], [], {}
        if unknown:
            terminal["decision"]["candidates"] = [{"action_id": "unknown_news", "feasible": True,
                "value": {"status": "UNKNOWN", "net_voi": None, "reason": "insufficient_support"}}]
    trajectory = {"match_id": match_id, "actual_external_requests": 0, "cycles": cycles,
                  "limits": limits, "actions": actions, "cost": spent,
                  "initial_probabilities": initial, "final_probabilities": posterior, "terminal_selection": stop}
    return {"match": {"match_id": match_id}, "configuration": {"limits": limits}, "trajectory": trajectory,
            "cycles": cycles, "actions": actions, "cost": spent, "initial_probabilities": initial,
            "final_probabilities": posterior, "terminal_selection": stop}


def backend(monkeypatch, *, ids=("m1", "m2", "m3"), failure=None):
    seen = []

    def catalog(**_kw):
        return [{"match_id": mid, "home_team": "Home", "away_team": "Away"} for mid in ids]

    def analyze(match, output, **_kw):
        seen.append(match)
        if failure == match:
            raise ValueError("corrupt archive")
        result = completed(match, direct_stop=match == "m2", unknown=match == "m2")
        output.mkdir(parents=True)
        for name, value in (("analysis.json", result), ("trajectory.json", result["trajectory"])):
            (output/name).write_text(json.dumps(value), encoding="utf-8")
        (output/"report.md").write_text("test fixture report", encoding="utf-8")
        return result

    monkeypatch.setattr(acceptance, "_backend", lambda: (catalog, analyze))
    return seen


def test_full_catalog_executes_in_order_and_cases_are_first_path_occurrences(tmp_path, monkeypatch):
    seen = backend(monkeypatch)
    summary = acceptance.run_acceptance(tmp_path/"acceptance")
    assert summary["status"] == "passed"
    assert seen == ["m1", "m2", "m3"]
    assert summary["coverage"] == "full_catalog"
    assert [c["match_id"] for c in summary["first_cases"]] == ["m1", "m2"]
    assert summary["selection_sequences"] == [{"actions": ["results_view"], "matches": 2}, {"actions": [], "matches": 1}]
    assert summary["matches"][1]["unsupported_candidates"] == {"insufficient_support": 1}
    assert summary["forecast_scores"]["status"] == "skipped"
    assert (tmp_path/"acceptance/report.md").is_file()


def test_explicit_limit_is_disclosed_and_existing_output_is_not_overwritten(tmp_path, monkeypatch):
    seen = backend(monkeypatch)
    output = tmp_path/"limited"
    summary = acceptance.run_acceptance(output, limit=1)
    assert seen == ["m1"] and summary["explicit_limit"] == 1 and summary["coverage"] == "explicit_subset"
    frozen = (output/"summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        acceptance.run_acceptance(output)
    assert (output/"summary.json").read_bytes() == frozen


def test_one_corrupt_match_fails_exit_but_remaining_cases_are_checked(tmp_path, monkeypatch):
    seen = backend(monkeypatch, failure="m2")
    assert acceptance.main(["--output", str(tmp_path/"failed")]) == 1
    summary = json.loads((tmp_path/"failed/summary.json").read_text(encoding="utf-8"))
    assert seen == ["m1", "m2", "m3"]
    assert summary["failed_matches"] == 1 and summary["passed_matches"] == 2
    assert summary["failures"][0]["error"] == "corrupt archive"


def test_offline_guard_is_effective_and_restored():
    original = socket.create_connection
    with acceptance.offline_guard() as attempts:
        with pytest.raises(RuntimeError, match="must not access"):
            socket.create_connection(("example.invalid", 80))
    assert len(attempts) == 1 and socket.create_connection is original


@pytest.mark.parametrize("mutation, message", [
    (lambda r: r["cycles"][0]["state_before"].update(outcome="home"), "result label"),
    (lambda r: r["cycles"][0]["state_before"]["groups"]["xg"].update(values={"xg": 2}), "exposed values"),
    (lambda r: r["cycles"][0]["decision"]["prediction"].update(probabilities=[.6, .6, .1]), "invalid probability"),
    (lambda r: r["cycles"][0]["decision"]["candidates"][0]["value"].update(net_voi=-.01), "positive net"),
    (lambda r: r["cycles"][1]["decision"]["physical_budget"].update(spent={"money": .04, "requests": 1}), "spent vector"),
    (lambda r: r["cycles"][0]["state_after"]["groups"]["xg"].update(status="observed", values={"xg": 2}), "unselected evidence"),
    (lambda r: r["terminal_selection"].update(reason=""), "stop reason"),
    (lambda r: r["trajectory"].update(actual_external_requests=1), "external request"),
    (lambda r: r["cycles"][0]["update"].update(prediction_after=[.6, .2, .2]), "post-query probability"),
])
def test_semantic_failures_are_not_counted_as_success(mutation, message):
    result = completed()
    mutation(result)
    with pytest.raises(ValueError, match=message):
        acceptance.validate_analysis(result, {"match_id": "m1"})


def test_catalog_labels_are_rejected_before_any_analysis(tmp_path, monkeypatch):
    def forbidden(*_a, **_kw):
        pytest.fail("labelled catalog reached analysis")
    monkeypatch.setattr(acceptance, "_backend", lambda: (lambda **_kw: [{"match_id": "m1", "label": 0}], forbidden))
    summary = acceptance.run_acceptance(tmp_path/"labels")
    assert summary["status"] == "failed" and summary["executed_matches"] == 0
    assert "result label" in summary["failures"][0]["error"]


def test_network_attempt_cannot_be_hidden_by_backend_catching_exception(tmp_path, monkeypatch):
    backend(monkeypatch, ids=("m1",))
    original_backend = acceptance._backend
    load, analyze = original_backend()

    def tries_network(**kwargs):
        try:
            socket.create_connection(("example.invalid", 80))
        except RuntimeError:
            pass
        return load(**kwargs)

    monkeypatch.setattr(acceptance, "_backend", lambda: (tries_network, analyze))
    summary = acceptance.run_acceptance(tmp_path/"blocked")
    assert summary["network_attempts"] == 1 and summary["status"] == "failed"
