"""Cutoff-safe public memory and action contracts across autonomous steps."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json

import pytest

from football_agent.autonomous_acquisition_step import select_acquisition_action
from football_agent.candidate_action_registry import build_state_from_menu, replan_after_execution_receipt
from football_agent.candidate_menu_compiler import compile_candidate_menu, compile_source_backed_menu
from football_agent.execution_receipt_bridge import accept_action, complete_action
from football_agent.platform_acquisition_context import build_platform_acquisition_context
from football_agent.source_context_adapter import build_source_backed_context

CUTOFF = "2026-10-01T12:00:00Z"


def evidence(source, values, *, ingested_at="2026-10-01T09:01:00Z"):
    return {
        "source_id": source, "kind": source,
        "observed_at": "2026-10-01T09:00:00Z", "ingested_at": ingested_at,
        "availability": "available", "freshness": "fresh",
        "digest": hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest(),
        "values": {"context": values},
    }


def catalog():
    return [
        {"candidate_id": cid, "kind": cid, "source_ids": [cid],
         "route": {"source_id": cid, "action_id": f"query-{cid}", "cost": .1, "unit": "decision_loss"},
         "declared_cost": {"http_calls": 1}, "utility_cost": {"amount": .1, "unit": "decision_loss"}}
        for cid in ("lineup", "weather")
    ]


def prepare(*, complete=True, extra_receipt_rows=()):
    match = {
        "match_id": "replan-contract", "kickoff": "2026-10-01T18:00:00Z",
        "prediction_cutoff": CUTOFF, "observed_at": "2026-10-01T08:00:00Z",
        "ingested_at": "2026-10-01T08:01:00Z", "home_team_id": "home", "away_team_id": "away",
    }
    actor = build_platform_acquisition_context(
        match, [evidence("situation", {"qualification_state": "must_win"})]
    )["actor_context"]
    report = compile_candidate_menu(
        catalog(), context_report={"context": {"qualification_state": "must_win"}},
        prediction_cutoff=CUTOFF, candidate_generation_complete=complete,
        source_rows=[{"source_id": cid, "available_at": "2026-10-01T09:00:00Z"}
                     for cid in ("lineup", "weather")],
    )
    state = build_state_from_menu(actor, report)["state"]
    estimate = {"candidate_id": "lineup", "state_digest": state["state_digest"],
                "expected_value": .8, "unit": "decision_loss", "source": "test",
                "training_cutoff": "2026-09-01T00:00:00Z", "available_at": "2026-09-30T00:00:00Z"}
    trace = select_acquisition_action(state, loss_unit="decision_loss", estimates={"lineup": estimate},
                                      remaining_budget={"http_calls": 2})
    accepted = accept_action(trace, provider_id="offline", accepted_at="2026-10-01T10:00:00Z")
    completed = complete_action(
        accepted, result_payload={"fetched": True},
        evidence_rows=[evidence("lineup", {"lineup_status": "confirmed"})] + list(extra_receipt_rows),
        actual_resource_cost={"http_calls": 1}, observed_at="2026-10-01T10:01:00Z",
        operational_available_at="2026-10-01T10:01:00Z",
        captured_at="2026-10-01T10:01:01Z",
    )
    return actor, report, match, completed


def replan(prepared, **kwargs):
    actor, report, match, receipt = prepared
    return replan_after_execution_receipt(actor, receipt, match, menu_report=report,
                                         catalog=kwargs.pop("catalog", catalog()), **kwargs)


def test_acquisition_contract_survives_menu_and_state_projection():
    rows = catalog()
    rows[0].update(action_kind="refresh", requires_source_ids=["situation"], aggregation_mode="any_usable")
    actor, _, _, _ = prepare()
    report = compile_candidate_menu(rows, context_report={"context": {}}, prediction_cutoff=CUTOFF)
    state = build_state_from_menu(actor, report)["state"]
    candidate = next(row for row in state["candidates"] if row["candidate_id"] == "lineup")
    assert candidate["action_kind"] == "refresh"
    assert candidate["requires_source_ids"] == ["situation"]
    assert candidate["aggregation_mode"] == "any_usable"


@pytest.mark.parametrize("update", [
    {"action_kind": "injury"}, {"requires_source_ids": "situation"},
    {"requires_source_ids": ["situation", " situation "]},
    {"requires_source_ids": [""]}, {"aggregation_mode": "majority"},
])
def test_invalid_acquisition_declarations_are_rejected_at_compilation(update):
    rows = catalog()
    rows[0].update(update)
    with pytest.raises(ValueError):
        compile_candidate_menu(rows, context_report={"context": {}}, prediction_cutoff=CUTOFF)


def test_topic_is_not_treated_as_query_or_refresh_operation():
    actor, report, _, _ = prepare()
    state = build_state_from_menu(actor, report)["state"]
    assert {row["action_kind"] for row in state["candidates"]} == {"query"}


def test_unqueried_tool_remains_executable_when_target_evidence_is_missing():
    result = replan(prepare())
    weather = next(row for row in result["state"]["candidates"] if row["candidate_id"] == "weather")
    assert weather["action_availability"] == "available"
    assert result["menu_report"]["source_visibility"].get("weather", "unknown") == "unknown"


def test_explicit_catalog_availability_update_overrides_previous_declaration():
    rows = catalog()
    rows[1]["availability"] = "unavailable"
    result = replan(prepare(), catalog=rows)
    weather = next(row for row in result["state"]["candidates"] if row["candidate_id"] == "weather")
    assert weather["action_availability"] == "unavailable"


@pytest.mark.parametrize("changed", ["action_id", "source_id", "source_ids", "source_version"])
def test_rebound_candidate_does_not_inherit_previous_tool_availability(changed):
    rows = catalog()
    if changed == "source_ids":
        rows[1]["source_ids"].append("new-source")
    elif changed == "source_id":
        rows[1]["source_ids"] = ["new-source"]
        rows[1]["route"]["source_id"] = "new-source"
    else:
        rows[1]["route"][changed] = "changed"
    result = replan(prepare(), catalog=rows)
    weather = next(row for row in result["state"]["candidates"] if row["candidate_id"] == "weather")
    assert weather["action_availability"] == "unknown"


def test_open_world_completeness_is_not_promoted_after_query():
    result = replan(prepare(complete=False))
    assert result["state"]["candidate_generation_complete"] is False
    assert result["menu_report"]["candidate_generation_complete"] is False


def test_source_backed_wrapper_accepts_explicit_incomplete_catalog():
    report = build_source_backed_context([], target_key="replan-contract", prediction_cutoff=CUTOFF)
    menu = compile_source_backed_menu(catalog(), report, candidate_generation_complete=False)
    assert menu["candidate_generation_complete"] is False


def test_actor_memory_and_provider_fields_use_same_cutoff_visible_projection():
    result = replan(prepare())
    context = result["context_report"]["binding"]["context"]
    assert context["qualification_state"] == "must_win"
    assert context["lineup_status"] == "confirmed"
    assert "situation" in {row["source_id"] for row in result["actor_context"]["source_observations"]}


def test_late_ingestion_cannot_reenter_context_through_early_available_at():
    late = evidence("news", {"weather_state": "snow"}, ingested_at="2026-10-01T13:00:00Z")
    late["available_at"] = "2026-10-01T09:00:00Z"
    result = replan(prepare(extra_receipt_rows=[late]))
    assert result["context_report"]["binding"]["context"]["weather_state"] == "unknown"
    assert "news" not in {row["source_id"] for row in result["actor_context"]["source_observations"]}


def test_generator_evidence_and_catalog_are_consumed_once_and_retained():
    consumed = []
    def observations():
        consumed.append("evidence")
        yield evidence("schedule", {"schedule_congestion": "high"})
    def declarations():
        for row in catalog():
            consumed.append(row["candidate_id"])
            yield row
    result = replan(prepare(), existing_evidence_rows=observations(), catalog=declarations())
    assert consumed == ["evidence", "lineup", "weather"]
    assert result["context_report"]["binding"]["context"]["schedule_congestion"] == "high"


def test_datetime_platform_clocks_remain_supported_in_extra_evidence():
    row = evidence("schedule", {"schedule_congestion": "high"})
    row["observed_at"] = datetime(2026, 10, 1, 9, tzinfo=timezone.utc)
    row["ingested_at"] = datetime(2026, 10, 1, 9, 1, tzinfo=timezone.utc)
    result = replan(prepare(), existing_evidence_rows=[row, deepcopy(row)])
    assert result["context_report"]["binding"]["context"]["schedule_congestion"] == "high"


def test_replanning_does_not_mutate_actor_receipt_menu_or_catalog():
    prepared, rows = prepare(), catalog()
    before, before_rows = deepcopy(prepared), deepcopy(rows)
    replan(prepared, catalog=rows)
    assert prepared == before
    assert rows == before_rows


def test_operational_replanner_does_not_discard_historical_assumptions():
    prepared = prepare()
    prepared[0]["information_mode"] = "historical_reconstruction"
    with pytest.raises(ValueError, match="replay adapter"):
        replan(prepared)
