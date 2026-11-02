"""Check a saved evidence-only VoI loop without sources or semantic calls.

This checks recorded contracts and mathematical transitions, not whether a
source is true or its semantic interpretation is correct. Unexercised branches
remain explicit coverage gaps; unchanged probabilities are not a failure.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from football_agent.directional_evidence import validate_directional_observation, directional_update_for_action
from football_agent.semantic_scenario_protocol import OBSERVATION_INPUT_VERSION, observation_public_projection
from football_agent.semantic_scenario_report import _no_answer_checks
from football_agent.semantic_scenario_runtime import replay_math


def audit_evidence_loop(run):
    failures, classified = [], []

    def require(condition, message):
        if not condition:
            failures.append(message)

    configuration = run["configuration"]
    require(configuration.get("observation_input_contract") == OBSERVATION_INPUT_VERSION,
            "This acceptance requires the evidence-only input contract")
    require(configuration.get("directional_evidence_required") is True,
            "Directional evidence gating must be enabled")
    try:
        replay = replay_math(run)
    except (ValueError, KeyError, TypeError, ArithmeticError) as exc:
        replay = {"status": "failed", "error": str(exc)}
        failures.append("Saved mathematical decisions do not replay")
    observations, cycles, calls = run["observations"], run["cycles"], run["semantic_calls"]
    require(bool(cycles) and len(cycles) == len(observations) + 1,
            "Expected one decision before each observation and one final decision")
    for index, cycle in enumerate(cycles):
        require(cycle["observations"] == observations[:index],
                f"Cycle {index} does not contain the actual observation prefix")
    documents = {d["document_id"]: d for d in run.get("documents", [])}
    by_action = {}
    for index, observation in enumerate(observations):
        if index >= len(cycles):
            break
        cycle = cycles[index]
        action_id = observation["action_id"]
        require(cycle["decision"]["selected_action_id"] == action_id,
                f"Observation {index} is not the action selected by VoI")
        require(action_id not in by_action, "An already attempted action was repeated")
        action = next((a for a in cycle["model"]["proposal"]["actions"] if a["action_id"] == action_id), None)
        require(action is not None, f"Observation {index} has no frozen action contract")
        if action is not None:
            by_action[action_id] = (action, observation)
    for call in calls:
        if call.get("protocol") != "semantic-directional-observation-v1":
            continue
        public = call.get("input", {})
        action_id = public.get("action", {}).get("action_id")
        if action_id not in by_action:
            failures.append(f"Classification {call['index']} has no executed action")
            continue
        action, observation = by_action[action_id]
        returned = observation["returned_document_ids"]
        require(all(identity in documents for identity in returned), "Returned document is absent from the run")
        admitted = [documents[i] for i in returned if i in documents and documents[i].get("status") == "OK"
                    and documents[i].get("kind") == "article" and documents[i].get("text")]
        expected = observation_public_projection(
            match={**configuration["match"], "decision_at": public.get("match", {}).get("decision_at")},
            action=action, documents=admitted)
        actual = {k: v for k, v in public.items() if k != "validation_errors"}
        require(actual == expected, f"Classification {call['index']} leaked expectations or changed acquired evidence")
        if call.get("status") != "accepted":
            continue
        try:
            validated = validate_directional_observation(call["response"], action, expected["documents"])
            require(validated == call["validated"], f"Classification {call['index']} validated payload differs")
            permits = directional_update_for_action(validated, action) is not None
            expected_outcome = validated["outcome"] if permits else "no_answer"
            require(observation["outcome"] == expected_outcome,
                    f"Classification {call['index']} bypassed the directional update gate")
            classified.append({"call_index": call["index"], "action_id": action_id,
                               "reported_outcome": validated["outcome"], "effective_outcome": expected_outcome,
                               "permits_directional_update": permits})
        except (ValueError, KeyError, TypeError) as exc:
            failures.append(f"Classification {call['index']} failed revalidation: {exc}")
    classified_ids = {c["action_id"] for c in classified}
    require(all(o["outcome"] == "no_answer" or o["action_id"] in classified_ids for o in observations),
            "An informative observation has no accepted source classification")
    inert = _no_answer_checks(run)
    for check in inert:
        require(check["unchanged_within_1e_12"] is True,
                f"No-answer transition is changed or unverified: {check['action_id']}")
        following = next((c for c in cycles if c["index"] == check["after_cycle"]), None)
        check["next_selected_action_id"] = None if following is None else following["decision"]["selected_action_id"]
        check["next_stop_reason"] = None if following is None else following["decision"]["stop_reason"]
    require(run["status"] in {"completed", "completed_with_resource_stop"}, "Runtime did not complete")
    if cycles:
        last = cycles[-1]["decision"]
        require(run["final_prediction"] == last["prediction_nominal"], "Final prediction differs from final decision")
        if run["status"] == "completed":
            require(last["selected_action_id"] is None and run["terminal"] == last,
                    "Normal termination differs from the solver's final stop")
        elif run["status"] == "completed_with_resource_stop":
            require(last["selected_action_id"] is not None and run["terminal"] == {**last, "selected_action_id": None,
                    "stop_reason": "semantic_call_budget_exhausted"}
                    and len(calls) == configuration["semantic_call_budget"],
                    "Resource termination is not supported by the saved semantic budget")
    require(len(calls) <= configuration["semantic_call_budget"], "Semantic call budget was exceeded")
    return {"schema": "semantic-evidence-loop-acceptance-v1", "status": "failed" if failures else "passed",
            "runtime_status": run["status"], "run_id": run.get("run_id"),
            "scope": "saved_contracts_and_mathematical_transitions_not_truth_or_forecast_validation",
            "math_replay": replay, "classification_checks": classified, "no_answer_transitions": inert,
            "coverage": {"accepted_body_classifications": len(classified), "no_answer_transitions": len(inert),
                         "continued_after_no_answer": sum(c["next_selected_action_id"] is not None for c in inert),
                         "informative_updates": sum(o["outcome"] in {"supports", "refutes"} for o in observations),
                         "unexercised": [name for name, present in (
                             ("accepted_body_classification", classified), ("no_answer_transition", inert),
                             ("reselection_after_no_answer", any(c["next_selected_action_id"] is not None for c in inert)),
                             ("informative_directional_update", any(o["outcome"] in {"supports", "refutes"}
                                                                   for o in observations))) if not present]},
            "semantic_status_counts": dict(Counter(c.get("status", "pending") for c in calls)),
            "terminal": deepcopy(run["terminal"]), "network_calls": 0, "semantic_calls": 0, "failures": failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output path; existing artifacts are immutable")
    raw = args.run.read_bytes()
    result = audit_evidence_loop(json.loads(raw))
    result["source_file"] = {"path": str(args.run.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}
    if args.run.read_bytes() != raw:
        raise RuntimeError("Source run changed during acceptance")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(result, target, ensure_ascii=False, allow_nan=False, indent=2)
        target.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "coverage", "failures", "network_calls", "semantic_calls")},
                     ensure_ascii=False))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
