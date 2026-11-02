"""Behavioral regression for real multi-step orchestration, with offline providers."""
from copy import deepcopy
import hashlib
import json

import pytest

from football_agent.autonomous_runtime_loop import run_information_loop
from football_agent.candidate_action_registry import build_state_from_menu
from football_agent.candidate_menu_compiler import compile_candidate_menu
from football_agent.platform_acquisition_context import build_platform_acquisition_context
from football_agent.state_bound_context_value import StateBoundContextValueEstimator
from football_agent.voi_evidence_bridge import VoIEvidenceLink

CUTOFF = "2026-10-01T12:00:00Z"


def evidence(source, values, stamp="2026-10-01T09:00:00Z"):
    return {"source_id": source, "kind": source, "observed_at": stamp,
            "ingested_at": stamp, "availability": "available", "freshness": "fresh",
            "digest": hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest(),
            "values": {"context": values}}


def setup_run(*, budget=2, complete=True, values=None):
    match = {"match_id": "runtime-budget-fixture", "kickoff": "2026-10-01T18:00:00Z",
             "prediction_cutoff": CUTOFF, "observed_at": "2026-10-01T08:00:00Z",
             "ingested_at": "2026-10-01T08:01:00Z", "home_team_id": "home", "away_team_id": "away"}
    context = {"qualification_state": "must_win"}
    actor = build_platform_acquisition_context(match, [evidence("situation", context)])["actor_context"]
    catalog = [{"candidate_id": cid, "kind": "query", "source_ids": [cid],
                "availability": "available", "route": {"source_id": cid, "source_version": "v1",
                "action_id": f"query-{cid}", "cost": .1, "unit": "decision_loss"},
                "declared_cost": {"http_calls": 1},
                "utility_cost": {"amount": .1, "unit": "decision_loss"}}
               for cid in ("lineup", "weather")]
    menu = compile_candidate_menu(catalog, context_report={"context": context},
                                  prediction_cutoff=CUTOFF, candidate_generation_complete=complete)
    state_bundle = build_state_from_menu(actor, menu)
    links, contexts = [], {}
    values = values or {"lineup": .9, "weather": .8}
    for candidate in catalog:
        cid = candidate["candidate_id"]
        for index in range(60):
            identity = f"{cid}-{index}"
            link = VoIEvidenceLink(
                link_id=identity, trace_id=identity, episode_id=f"past-{index}", target_key=f"historical-{index}",
                request_id=identity, action_id=f"query-{cid}", candidate_id=cid,
                evidence_entry_id=identity, evidence_availability="available", evidence_quality="VALIDATED",
                requested_at="2026-09-01T08:00:00Z", source_version="v1", source_id=cid,
                dependency_key=identity, prediction_cutoff="2026-09-01T12:00:00Z",
                expected_information_value=values[cid], observed_risk_reduction=values[cid],
                calibration_error=None, status="linked_available")
            links.append(link)
            contexts[identity] = deepcopy(context)
    estimator = StateBoundContextValueEstimator(
        links, contexts, {item["candidate_id"]: item["route"] for item in catalog},
        context_keys=("qualification_state",), min_support=5, confidence=.9,
        calibration_as_of="2026-09-30T00:00:00Z", family_size=2)
    return dict(actor_context=actor, state_bundle=state_bundle, menu_report=menu,
                catalog=catalog, match=match, estimator=estimator, context=context,
                loss_unit="decision_loss", remaining_budget={"http_calls": budget},
                provider_id="offline-test", accepted_at="2026-10-01T10:00:00Z", max_steps=5)


class Provider:
    def __init__(self, actual=1, status=None):
        self.requests = []
        self.actual = actual
        self.status = status

    def __call__(self, request):
        self.requests.append(deepcopy(request))
        cid = request["candidate_id"]
        stamp = f"2026-10-01T10:{20 + len(self.requests):02d}:00Z"
        values = {"lineup_status": "confirmed"} if cid == "lineup" else {"weather_state": "rain"}
        actual = {"http_calls": self.actual} if not isinstance(self.actual, dict) else self.actual
        result = {"result_payload": {"fetched": True}, "evidence_rows": [evidence(cid, values, stamp)],
                  "actual_resource_cost": actual, "observed_at": stamp,
                  "operational_available_at": stamp, "captured_at": stamp}
        if self.status:
            result.update(status=self.status, evidence_rows=[])
        return result


def test_two_autonomous_queries_preserve_memory_and_then_stop_without_duplicates():
    args, provider = setup_run(), Provider()
    before = deepcopy(args["actor_context"])
    result = run_information_loop(**args, execute_fn=provider)
    assert [req["candidate_id"] for req in provider.requests] == ["lineup", "weather"]
    assert [step["selection"]["decision"] for step in result["history"]] == ["acquire", "acquire", "stop"]
    assert [step["selection"]["remaining_budget"]["http_calls"] for step in result["history"]] == [2, 1, 0]
    assert result["remaining_budget"] == {"http_calls": 0}
    assert result["actual_resource_cost"] == {"http_calls": 2}
    assert result["final_context"]["qualification_state"] == "must_win"
    assert result["final_context"]["weather_state"] == "rain"
    assert args["actor_context"] == before
    assert args["remaining_budget"] == {"http_calls": 2}
    assert all(req["parameters"]["match_id"] == args["match"]["match_id"] for req in provider.requests)
    first, second = result["history"][:2]
    assert first["state_digest"] != second["state_digest"]
    assert second["receipt"]["accepted_at"] >= first["receipt"]["captured_at"]
    assert result["contract"]["provider_invocations"] == 2
    assert result["contract"]["real_source_requests"] is None


def test_one_call_budget_defers_second_useful_query_instead_of_overspending():
    provider = Provider()
    result = run_information_loop(**setup_run(budget=1), execute_fn=provider)
    assert len(provider.requests) == 1
    last = result["history"][-1]["selection"]
    assert last["decision"] == "defer"
    assert last["query_value_decision"]["selection_basis"] == "resource_budget_exhausted"
    assert result["actual_resource_cost"] == {"http_calls": 1}


def test_unused_reservation_is_released_using_actual_not_declared_cost():
    provider = Provider(actual=.5)
    result = run_information_loop(**setup_run(budget=1.5), execute_fn=provider)
    assert len(provider.requests) == 2
    assert result["remaining_budget"] == {"http_calls": .5}
    assert result["history"][1]["selection"]["remaining_budget"] == {"http_calls": 1}


@pytest.mark.parametrize("status", ["failed", "late", "ambiguous"])
def test_empty_noncompleted_receipt_consumes_budget_without_changing_evidence(status):
    provider, args = Provider(status=status), setup_run()
    result = run_information_loop(**args, execute_fn=provider)
    assert len(provider.requests) == 1
    expected_status = "execution_deadline_reached" if status == "late" else "replan_required"
    assert result["terminal_status"] == expected_status
    assert result["remaining_budget"] == {"http_calls": 1}
    assert result["actual_resource_cost"] == {"http_calls": 1}
    assert result["final_actor_context"] == args["actor_context"]


def test_actual_overrun_is_recorded_and_blocks_further_execution():
    provider = Provider(actual=3)
    result = run_information_loop(**setup_run(), execute_fn=provider)
    assert len(provider.requests) == 1
    assert result["terminal_status"] == "budget_overrun"
    assert result["history"][0]["budget"]["overrun"] == {"http_calls": 1}
    assert result["actual_resource_cost"] == {"http_calls": 3}
    assert result["remaining_budget"] == {"http_calls": 0}


def test_missing_actual_cost_holds_reservation_without_inventing_free_query():
    provider = Provider(actual={})
    result = run_information_loop(**setup_run(), execute_fn=provider)
    assert len(provider.requests) == 1
    assert result["terminal_status"] == "cost_reconciliation_required"
    assert result["actual_resource_cost"] == {}
    assert result["held_resource_cost"] == {"http_calls": 1}
    assert result["remaining_budget"] == {"http_calls": 1}


def test_unbudgeted_actual_resource_stops_without_merging_units():
    provider = Provider(actual={"http_calls": 1, "llm_tokens": 50})
    result = run_information_loop(**setup_run(), execute_fn=provider)
    assert len(provider.requests) == 1
    assert result["terminal_status"] == "cost_reconciliation_required"
    assert result["actual_resource_cost"] == {"http_calls": 1, "llm_tokens": 50}
    assert result["history"][0]["budget"]["unbudgeted_resources"] == ["llm_tokens"]


def test_provider_exception_keeps_accepted_receipt_and_hold_without_retry():
    calls = []
    def explode(request):
        calls.append(request)
        raise RuntimeError("arbitrary response body and credentials must not enter audit")
    result = run_information_loop(**setup_run(), execute_fn=explode)
    assert len(calls) == 1
    assert result["terminal_status"] == "execution_unresolved"
    assert result["history"][0]["receipt"]["status"] == "accepted"
    assert result["history"][0]["budget"]["actual_resource_cost"] is None
    assert result["held_resource_cost"] == {"http_calls": 1}
    assert "credentials" not in repr(result)


def test_invalid_provider_response_stops_with_hold_and_no_evaluator_leak():
    provider, args = Provider(), setup_run()
    def invalid(request):
        result = provider(request)
        result["result_payload"] = {"score": "PRIVATE_OUTCOME"}
        return result
    result = run_information_loop(**args, execute_fn=invalid)
    assert result["terminal_reason"] == "provider_response_rejected"
    assert result["held_resource_cost"] == {"http_calls": 1}
    assert "PRIVATE_OUTCOME" not in repr(result)
    assert result["final_actor_context"] == args["actor_context"]


def test_generator_catalog_is_reusable_across_multiple_cycles():
    args, provider = setup_run(), Provider()
    args["catalog"] = iter(args["catalog"])
    result = run_information_loop(**args, execute_fn=provider)
    assert len(provider.requests) == 2
    assert result["history"][-1]["selection"]["decision"] == "stop"


def test_open_candidate_menu_remains_incomplete_after_all_known_queries():
    provider = Provider()
    result = run_information_loop(**setup_run(complete=False), execute_fn=provider)
    assert len(provider.requests) == 2
    assert result["final_state"]["candidate_generation_complete"] is False
    assert result["history"][-1]["selection"]["decision"] == "defer"
    assert result["history"][-1]["selection"]["query_value_decision"]["selection_basis"] == "candidate_menu_incomplete"


def test_known_low_value_source_is_skipped_even_with_budget_left():
    provider = Provider()
    result = run_information_loop(**setup_run(values={"lineup": .9, "weather": -.8}), execute_fn=provider)
    assert [req["candidate_id"] for req in provider.requests] == ["lineup"]
    assert result["remaining_budget"] == {"http_calls": 1}
    assert result["history"][-1]["selection"]["decision"] == "stop"
    assert result["history"][-1]["selection"]["query_value_decision"]["selection_basis"] == "all_feasible_modeled_net_values_nonpositive"


def test_step_cap_is_not_presented_as_information_sufficiency():
    args, provider = setup_run(), Provider()
    args["max_steps"] = 1
    result = run_information_loop(**args, execute_fn=provider)
    assert result["terminal_status"] == "max_steps"
    assert result["terminal_reason"] == "max_steps"
    assert result["remaining_budget"] == {"http_calls": 1}


@pytest.mark.parametrize("budget", [-1, float("nan"), True])
def test_bad_budget_is_rejected_before_provider_invocation(budget):
    provider = Provider()
    with pytest.raises(ValueError, match="finite and nonnegative"):
        run_information_loop(**setup_run(budget=budget), execute_fn=provider)
    assert not provider.requests


def test_replanning_failure_keeps_receipt_and_accounted_cost():
    provider, args = Provider(), setup_run()
    def no_change(request):
        response = provider(request)
        response["evidence_rows"][0]["ingested_at"] = "2026-10-01T13:00:00Z"
        return response
    result = run_information_loop(**args, execute_fn=no_change)
    assert len(provider.requests) == 1
    assert result["terminal_reason"] == "public_state_rebuild_rejected"
    assert result["actual_resource_cost"] == {"http_calls": 1}
    assert result["history"][0]["receipt"]["status"] == "completed"
    assert result["final_actor_context"] == args["actor_context"]


def test_target_identity_mismatch_is_rejected_before_query():
    args, provider = setup_run(), Provider()
    args["target_key"] = "different-match"
    with pytest.raises(ValueError, match="target_key"):
        run_information_loop(**args, execute_fn=provider)
    assert not provider.requests


@pytest.mark.parametrize("field,value", [("home_team_id", "foreign-team"), ("kickoff", "2026-10-02T18:00:00Z")])
def test_changed_fixture_is_rejected_before_query(field, value):
    args, provider = setup_run(), Provider()
    args["match"][field] = value
    with pytest.raises(ValueError, match="runtime match"):
        run_information_loop(**args, execute_fn=provider)
    assert not provider.requests


def test_catalog_state_route_mismatch_is_rejected_before_query():
    args, provider = setup_run(), Provider()
    args["menu_report"]["routes"]["lineup"]["action_id"] = "unbound-action"
    with pytest.raises(ValueError, match="registry differs"):
        run_information_loop(**args, execute_fn=provider)
    assert not provider.requests


@pytest.mark.parametrize("actual", [{"http_calls": 1, " http_calls": 1}, {"": 1}, None])
def test_bad_actual_resource_names_or_shape_preserve_attempt_audit(actual):
    args, provider = setup_run(), Provider()
    def invalid(request):
        result = provider(request)
        result["actual_resource_cost"] = actual
        return result
    result = run_information_loop(**args, execute_fn=invalid)
    assert len(provider.requests) == 1
    assert result["terminal_reason"] == "provider_response_rejected"
    assert result["history"][0]["receipt"]["status"] == "accepted"
    assert result["held_resource_cost"] == {"http_calls": 1}


def test_capture_after_cutoff_stops_loop_without_executing_another_action():
    args, provider = setup_run(), Provider()
    def delayed_capture(request):
        result = provider(request)
        result["captured_at"] = "2026-10-01T12:01:00Z"
        return result
    result = run_information_loop(**args, execute_fn=delayed_capture)
    assert len(provider.requests) == 1
    assert result["terminal_status"] == "execution_deadline_reached"
    assert result["actual_resource_cost"] == {"http_calls": 1}
    assert result["history"][0]["receipt"]["status"] == "late"
    assert result["final_actor_context"] == args["actor_context"]


def test_each_resource_dimension_is_debited_without_conversion_to_utility():
    args, provider = setup_run(), Provider(actual={"http_calls": 1, "local_reads": 2})
    args["remaining_budget"] = {"http_calls": 2, "local_reads": 2}
    for candidate in args["catalog"]:
        candidate["declared_cost"]["local_reads"] = 2
    args["menu_report"] = compile_candidate_menu(args["catalog"], context_report={"context": args["context"]}, prediction_cutoff=CUTOFF)
    args["state_bundle"] = build_state_from_menu(args["actor_context"], args["menu_report"])
    result = run_information_loop(**args, execute_fn=provider)
    assert len(provider.requests) == 1
    assert result["remaining_budget"] == {"http_calls": 1, "local_reads": 0}
    assert result["history"][-1]["selection"]["decision"] == "defer"


def test_deferred_evidence_iterator_failure_keeps_accepted_attempt_and_hold():
    args, provider = setup_run(), Provider()
    def broken_rows():
        yield evidence("lineup", {"lineup_status": "confirmed"})
        raise RuntimeError("remote body interrupted")
    def response(request):
        result = provider(request)
        result["evidence_rows"] = broken_rows()
        return result
    result = run_information_loop(**args, execute_fn=response)
    assert len(provider.requests) == 1
    assert result["terminal_reason"] == "provider_response_rejected"
    assert result["history"][0]["receipt"]["status"] == "accepted"
    assert result["held_resource_cost"] == {"http_calls": 1}
    assert result["final_actor_context"] == args["actor_context"]


def test_budget_overrun_survives_simultaneous_replanning_failure():
    args, provider = setup_run(), Provider(actual=3)
    def excluded(request):
        result = provider(request)
        result["evidence_rows"][0]["ingested_at"] = "2026-10-01T13:00:00Z"
        return result
    result = run_information_loop(**args, execute_fn=excluded)
    assert len(provider.requests) == 1
    assert result["terminal_status"] == "budget_overrun"
    assert result["history"][0]["replan_error"] == "public_state_rebuild_rejected"
    assert result["actual_resource_cost"] == {"http_calls": 3}
    assert result["remaining_budget"] == {"http_calls": 0}


def test_foreign_match_evidence_never_changes_actor_or_triggers_second_query():
    args, provider = setup_run(), Provider()
    def foreign(request):
        result = provider(request)
        result["evidence_rows"][0]["match_id"] = "another-fixture"
        return result
    result = run_information_loop(**args, execute_fn=foreign)
    assert len(provider.requests) == 1
    assert result["terminal_reason"] == "provider_response_rejected"
    assert result["final_actor_context"] == args["actor_context"]
    assert result["held_resource_cost"] == {"http_calls": 1}


def test_conflicting_new_information_defers_instead_of_reusing_old_voi():
    args, provider = setup_run(), Provider()
    def contradict(request):
        result = provider(request)
        result["evidence_rows"] = [evidence("lineup", {"qualification_state": "already_qualified"},
                                              stamp="2026-10-01T10:21:00Z")]
        return result
    result = run_information_loop(**args, execute_fn=contradict)
    assert len(provider.requests) == 1
    assert result["final_context"]["qualification_state"] == "unknown"
    assert result["remaining_budget"] == {"http_calls": 1}
    final = result["history"][-1]
    assert final["selection"]["decision"] == "defer"
    assert final["evaluation"]["gate_estimates"] == {}
    assert final["selection"]["query_value_decision"]["selection_basis"] == "unresolved_action_cost_or_value"


def test_missing_capture_halts_without_new_information_but_accounts_spend():
    args, provider = setup_run(), Provider()
    def incomplete(request):
        response = provider(request)
        response.pop("captured_at")
        return response
    result = run_information_loop(**args, execute_fn=incomplete)
    assert len(provider.requests) == 1
    assert result["history"][0]["receipt"]["status"] == "ambiguous"
    assert result["final_actor_context"] == args["actor_context"]
    assert result["actual_resource_cost"] == {"http_calls": 1}
    assert result["remaining_budget"] == {"http_calls": 1}
    assert not result["contract"]["autonomous_completion"]
    assert not result["contract"]["replay_eligible"]
