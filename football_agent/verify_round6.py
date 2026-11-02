"""Replay the round-six finite EVSI reference fixtures, all explicitly synthetic."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .decision_value import EVIDENCE_CLASS, FiniteDecisionModel, simulate_episode, synthetic_scenarios


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def replay_trace(model: FiniteDecisionModel, episode: dict) -> dict:
    """Recompute every decision from observed evidence; evaluator labels ignored."""
    _require(episode["evidence_class"] == EVIDENCE_CLASS, "Unexpected evidence class")
    state, budget = model.initial_state(), episode["budget"]
    trace = episode["trace"]
    _require(bool(trace) and len(trace) <= episode["max_actions"] + 1, "Invalid trace length")
    stopped = False
    for index, event in enumerate(trace):
        _require(not stopped, "Events after STOP")
        _require(event["step"] == index, "Nonsequential event index")
        _require(event["belief_before"] == dict(zip(model.hypothesis_names, state.belief)), "Changed posterior")
        _require(event["observed_before"] == [list(pair) for pair in state.observations], "Changed evidence history")
        expected = model.evaluate(state, remaining_budget=budget,
                                  horizon=min(episode["horizon"], episode["max_actions"] - index))
        _require(expected == event["decision"], "Recomputed decision differs")
        action = expected["action_id"]
        if action == "STOP":
            _require("observation" not in event, "STOP must not receive a new observation")
            stopped = True
            continue
        state = model.observe(state, action, event["observation"])
        cost = model.tools[action].cost
        _require(cost <= budget + 1e-12 and event["cost"] == cost, "Acquisition exceeds budget")
        budget = max(0.0, budget - cost)
        _require(event["remaining_budget_after"] == budget, "Changed budget")
        _require(event["belief_after"] == dict(zip(model.hypothesis_names, state.belief)), "Changed acquired posterior")
        _require(event["prediction_after"] == list(model.prediction(state)), "Changed acquired prediction")
    _require(stopped, "Trace has no terminal STOP")
    _require(episode["spent"] == episode["budget"] - budget, "Changed aggregate cost")
    _require(episode["final_prediction"] == list(model.prediction(state)), "Changed final prediction")
    return {"status": "VERIFIED", "actions": len(state.observations), "spent": episode["spent"],
            "final_prediction": list(model.prediction(state)), "policy_input_has_match_result": False}


def competition_context_integration(*, lineup_outcome: str = "reserves") -> dict:
    """Real platform projection plus a declared synthetic observation model.

    The platform does not infer a numerical probability from a qualification or
    player list. This fixture explicitly pairs a returned symbolic observation
    with a constructed lineup; a production measurement model remains research.
    """
    from football_platform.information.competition_context import project_competition_context

    _require(lineup_outcome in ("starters", "reserves"), "Unknown constructed lineup response")
    model = synthetic_scenarios()["qualified_rotation_unknown"]["model"]
    fixture = {"match_id": "synthetic.round6.context", "competition_id": "synthetic.group",
               "home_team_id": "synthetic.home", "away_team_id": "synthetic.away",
               "kickoff": "2025-01-01T18:00:00Z"}
    cutoff = "2025-01-01T16:00:00Z"
    qualification_id, lineup_id = "synthetic.qualification.1", "synthetic.lineup.1"
    common = {"match_id": fixture["match_id"], "competition_id": fixture["competition_id"],
              "team_id": fixture["home_team_id"], "source_id": "synthetic.round6.environment",
              "source_kind": "synthetic", "epistemic_status": "hypothesis",
              "observation_scope": "pre_match", "availability_status": "AVAILABLE"}
    qualification = {**common, "evidence_id": qualification_id, "topic": "qualification", "value": "qualified",
                     "available_at": "2025-01-01T10:00:00Z"}
    lineup = {**common, "evidence_id": lineup_id, "topic": "lineup", "value": "starting_xi",
              "available_at": "2025-01-01T15:00:00Z",
              "player_ids": [f"synthetic.player.{lineup_outcome}.{i:02d}" for i in range(1, 12)]}
    owner_observations = [qualification, lineup]
    before = project_competition_context(fixture=fixture, prediction_cutoff=cutoff,
                                         observations=owner_observations, acquired_evidence_ids=[qualification_id])
    _require(before.teams[fixture["home_team_id"]]["lineup"].status == "unknown", "Unpurchased lineup leaked")
    _require(lineup_id not in before.evidence, "Unpurchased lineup payload leaked")
    state = model.initial_state()
    budget = .015
    decision = model.evaluate(state, remaining_budget=budget, horizon=1)
    _require(decision["action_id"] == "lineup", "Constructed model must select lineup")
    # The owner now returns the selected tool's fixed response and matching
    # structured evidence. Neither was passed to evaluate before selection.
    receipt = {"action_id": "lineup", "observation": lineup_outcome, "evidence_id": lineup_id,
               "cost": model.tools["lineup"].cost, "cost_unit": model.cost_unit,
               "response_class": EVIDENCE_CLASS}
    posterior = model.observe(state, receipt["action_id"], receipt["observation"])
    budget -= receipt["cost"]
    after = project_competition_context(fixture=fixture, prediction_cutoff=cutoff,
                                        observations=owner_observations, acquired_evidence_ids=[qualification_id, lineup_id])
    _require(after.teams[fixture["home_team_id"]]["lineup"].status == "hypothesis", "Synthetic evidence became verified")
    stop = model.evaluate(posterior, remaining_budget=budget, horizon=1)
    _require(stop["action_id"] == "STOP", "The integration fixture exhausted its acquisition budget")
    return {"evidence_class": EVIDENCE_CLASS, "model_sha256": _hash(model.manifest()),
            "fixture": fixture, "prediction_cutoff": cutoff,
            "assumption": "The manually specified model conditions on assumed qualification; no automatic context-to-probability mapping exists.",
            "response_semantics": "The constructed environment pairs a starters/reserves observation with its synthetic player IDs; real rotation measurement is not implemented here.",
            "events": [
                {"event": "before_acquisition", "context": before.model_dump(mode="json"), "decision": decision},
                {"event": "tool_response", "receipt": receipt},
                {"event": "after_acquisition", "context": after.model_dump(mode="json"),
                 "posterior": dict(zip(model.hypothesis_names, posterior.belief)),
                 "prediction": list(model.prediction(posterior)), "decision": stop}],
            "status": "PLATFORM_PROJECTION_AND_FINITE_POSTERIOR_VERIFIED"}


def run_verification(output: Path) -> dict:
    cases = synthetic_scenarios()
    manifests, mechanism_checks, episodes, replay_results = {}, [], [], []
    for name, case in cases.items():
        model = case["model"]
        manifest = model.manifest()
        reconstructed = FiniteDecisionModel.from_manifest(manifest)
        _require(reconstructed.manifest() == manifest, "Manifest round trip differs")
        manifests[name] = {"model": manifest, "model_sha256": _hash(manifest), "context": case["context"],
                           "description": case["description"], "budget": case["budget"]}
        initial = model.initial_state()
        one = model.evaluate(initial, remaining_budget=case["budget"], horizon=1)
        two = model.evaluate(initial, remaining_budget=case["budget"], horizon=2)
        for candidate in one["candidates"]:
            if candidate["feasible"]:
                mi = model.target_mutual_information(initial, candidate["action_id"])
                _require(abs(candidate["immediate_evsi"] - mi) < 1e-10, "Log-loss EVSI must equal target mutual information")
        mechanism_checks.append({"scenario": name, "evidence_class": EVIDENCE_CLASS,
                                 "one_step_action": one["action_id"], "two_step_action": two["action_id"],
                                 "one_step_net_value": one["planning_net_value"],
                                 "two_step_net_value": two["planning_net_value"],
                                 "one_step_candidates": [{k: v for k, v in c.items() if k != "branches"} for c in one["candidates"]]})
        for horizon in (1, 2):
            for seed in (7, 19, 41, 73, 101):
                episode = simulate_episode(model, seed=seed, budget=case["budget"], horizon=horizon, max_actions=2)
                copy = simulate_episode(reconstructed, seed=seed, budget=case["budget"], horizon=horizon, max_actions=2)
                _require(copy == episode, "Seeded replay differs")
                record = {"scenario": name, "model_sha256": _hash(manifest), "episode": episode}
                episodes.append(record)
                replay_results.append({"scenario": name, "seed": seed, "horizon": horizon, **replay_trace(reconstructed, episode)})
        no_budget = model.evaluate(initial, remaining_budget=0, horizon=2)
        _require(no_budget["action_id"] == "STOP", "Positive-cost fixtures must stop at zero budget")

    indexed = {row["scenario"]: row for row in mechanism_checks}
    _require(indexed["qualified_rotation_unknown"]["one_step_action"] == "lineup", "Unknown rotation should query lineup in this assumed model")
    _require(indexed["qualified_reserves_known_strong"]["two_step_action"] == "STOP", "Adequate known reserve strength should stop in this assumed model")
    _require(indexed["draw_sufficient_tactics_unknown"]["one_step_action"] == "tactical_evidence", "Tactics evidence should be valued in this assumed model")
    _require(indexed["irrelevant_high_entropy_news"]["two_step_action"] == "STOP", "Outcome-irrelevant news should not be acquired")
    _require(indexed["complementary_xor_reference"]["one_step_action"] == "STOP", "XOR must defeat myopic acquisition")
    _require(indexed["complementary_xor_reference"]["two_step_action"] != "STOP", "Two-step solver must acquire XOR pair")
    duplicate_model = cases["duplicate_shared_report"]["model"]
    for outcome in ("positive", "negative"):
        state = duplicate_model.observe(duplicate_model.initial_state(), "report_a", outcome)
        decision = duplicate_model.evaluate(state, remaining_budget=.075, horizon=2)
        residual = next(c for c in decision["candidates"] if c["action_id"] == "report_b")
        _require(residual["immediate_evsi"] < 1e-12 and decision["action_id"] == "STOP", "Shared report was counted twice")

    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    implementation_hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                             for name in ("football_agent/decision_value.py", "football_agent/verify_round6.py",
                                          "football_platform/information/competition_context.py")}
    payload = {"schema_version": "round6-exact-evsi-manifest-v1", "evidence_class": EVIDENCE_CLASS,
               "scenarios": manifests,
               "implementation_sha256": implementation_hashes,
               "meaning": "Constructed mechanism/reference-solver checks only; no observed football benefit or learned prior claim.",
               "loss_units": "Natural-log expected predictive loss; costs are not money or percentages.",
               "planning_scope": "Exact for the specified finite model within one or two remaining acquisitions, not unrestricted global optimality."}
    integrations = [competition_context_integration(lineup_outcome=outcome) for outcome in ("starters", "reserves")]
    (output / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (output / "trajectories.jsonl").write_text("".join(_json(e) + "\n" for e in episodes), encoding="utf-8")
    (output / "mechanism-checks.json").write_text(json.dumps(mechanism_checks, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (output / "replay.json").write_text(json.dumps(replay_results, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (output / "context-integration.json").write_text(json.dumps(integrations, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    hashes = {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
              for name in ("manifest.json", "trajectories.jsonl", "mechanism-checks.json", "replay.json", "context-integration.json")}
    acceptance = {"status": "ROUND6_EXACT_EVSI_SYNTHETIC_MECHANISMS_VERIFIED", "evidence_class": EVIDENCE_CLASS,
                  "scenarios": len(cases), "replayed_episodes": len(episodes), "independent_real_matches": 0,
                  "cross_layer_context_trajectories": len(integrations),
                  "http_calls": 0, "llm_calls": 0, "real_provider_money": None,
                  "artifact_sha256": hashes, "mechanism_checks": mechanism_checks,
                  "limitations": payload["meaning"], "planning_scope": payload["planning_scope"]}
    (output / "acceptance.json").write_text(json.dumps(acceptance, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return acceptance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_verification(args.output)
    print(json.dumps({k: result[k] for k in ("status", "scenarios", "replayed_episodes", "independent_real_matches")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
