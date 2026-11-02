"""The accepted autonomous action cannot be relabelled after dispatch."""
from copy import deepcopy

import pytest

from football_agent.autonomous_acquisition_step import select_acquisition_action
from football_agent.candidate_action_registry import build_execution_request, replan_after_execution_receipt
from football_agent.candidate_menu_compiler import compile_candidate_menu
from football_agent.execution_action_adapter import complete_execution_request
from football_agent.execution_receipt_bridge import accept_action, complete_action
from test_autonomous_runtime_budget import setup_run, evidence


def bound_bundle():
    args = setup_run()
    state = args["state_bundle"]["state"]
    trace = select_acquisition_action(
        state, loss_unit="decision_loss", remaining_budget={"http_calls": 2},
        estimates={"lineup": {"candidate_id": "lineup", "state_digest": state["state_digest"],
            "expected_value": .8, "unit": "decision_loss", "source": "offline",
            "training_cutoff": "2026-09-01T00:00:00Z", "available_at": "2026-09-30T00:00:00Z"}})
    request = build_execution_request(trace, args["state_bundle"]["registry"],
                                      parameters={"match_id": args["match"]["match_id"]})
    accepted = accept_action(trace, provider_id="offline", accepted_at=args["accepted_at"],
                             execution_request=request)
    return args, request, accepted


def observation():
    return dict(result_payload={"ok": True}, evidence_rows=[evidence("lineup", {"lineup_status": "confirmed"})],
                actual_resource_cost={"http_calls": 1}, observed_at="2026-10-01T10:30:00Z",
                operational_available_at="2026-10-01T10:31:00Z", captured_at="2026-10-01T10:31:01Z")


@pytest.mark.parametrize("change", ["action", "sources", "target", "route_version", "request_id", "drop_state"])
def test_request_rebinding_is_rejected_before_completion(change):
    _, request, receipt = bound_bundle()
    changed = deepcopy(request)
    if change == "action":
        changed["action_id"] = changed["route"]["action_id"] = "foreign-action"
    elif change == "sources":
        changed["source_ids"].append("foreign-source")
    elif change == "target":
        changed["parameters"]["match_id"] = "another-match"
    elif change == "route_version":
        changed["route"]["source_version"] = "replacement"
    elif change == "request_id":
        changed["request_id"] = "replacement"
    else:
        changed.pop("state_digest")
    with pytest.raises(ValueError, match="request"):
        complete_execution_request(receipt, changed, **observation())


def test_equivalent_cutoff_serialization_and_detached_request_are_supported():
    _, request, accepted = bound_bundle()
    saved = deepcopy(accepted)
    request["prediction_cutoff"] = "2026-10-01T20:00:00+08:00"
    result = complete_execution_request(accepted, request, **observation())
    assert result["receipt"]["status"] == "completed"
    assert result["contract"]["request_binding_verified"] is True
    request["route"]["source_version"] = "mutated"
    assert accepted == saved
    assert result["receipt"]["request_binding"] == saved["request_binding"]


def test_direct_bridge_cannot_bypass_bound_source_ids():
    _, _, accepted = bound_bundle()
    payload = observation()
    payload["evidence_rows"][0]["source_id"] = "foreign-source"
    with pytest.raises(ValueError, match="source.*binding"):
        complete_action(accepted, **payload)


def test_replan_rejects_relabelled_selected_menu_even_with_same_candidate():
    args, request, accepted = bound_bundle()
    receipt = complete_execution_request(accepted, request, **observation())["receipt"]
    changed = deepcopy(args["catalog"])
    changed[0]["route"]["action_id"] = "foreign-action"
    menu = compile_candidate_menu(changed, context_report={"context": args["context"]},
                                  prediction_cutoff=args["match"]["prediction_cutoff"])
    with pytest.raises(ValueError, match="receipt selected route"):
        replan_after_execution_receipt(args["actor_context"], receipt, args["match"],
                                        menu_report=menu, catalog=changed)


def test_unselected_next_catalog_change_remains_allowed():
    args, request, accepted = bound_bundle()
    receipt = complete_execution_request(accepted, request, **observation())["receipt"]
    changed = deepcopy(args["catalog"])
    changed[1]["route"]["action_id"] = "weather-v2"
    result = replan_after_execution_receipt(args["actor_context"], receipt, args["match"],
                                            menu_report=args["menu_report"], catalog=changed)
    assert result["context_changed"]
    assert result["executed_route"]["action_id"] == request["action_id"]
    assert result["registry"]["by_candidate"]["weather"]["action_id"] == "weather-v2"
