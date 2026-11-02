"""Round148 tests for the explicit candidate menu compiler."""
from __future__ import annotations

from copy import deepcopy

import pytest

from football_agent.candidate_menu_compiler import VERSION, compile_candidate_menu, compile_source_backed_menu
from football_agent.source_context_adapter import build_source_backed_context

BASE = "2026-10-04T08:00:00Z"


def _catalog():
    return [
        {
            "candidate_id": "lineup",
            "kind": "query",
            "source_ids": ["api-football"],
            "route": {"source_id": "api-football", "source_version": "v2", "action_id": "query-lineup", "cost": .10, "unit": "decision_loss"},
            "declared_cost": {"http_calls": 1},
            "utility_cost": {"amount": .10, "unit": "decision_loss"},
            "context_affinity": {"rotation_risk": ["high"], "dominant_uncertainty": ["lineup"]},
        },
        {
            "candidate_id": "tactics",
            "kind": "query",
            "source_ids": ["news-feed"],
            "route": {"source_id": "news-feed", "action_id": "query-tactics", "cost": .05, "unit": "decision_loss"},
            "declared_cost": {"http_calls": 1},
            "utility_cost": {"amount": .05, "unit": "decision_loss"},
            "context_affinity": {"qualification_state": ["draw_sufficient", "must_win"]},
            "declared_priority": 1,
        },
    ]


def _report():
    return {
        "target_key": "match-148",
        "prediction_cutoff": BASE,
        "context": {"qualification_state": "draw_sufficient", "rotation_risk": "high", "dominant_uncertainty": "lineup"},
        "evaluator_fields_used": False,
    }


def test_compiler_orders_candidates_by_public_context_and_keeps_routes():
    result = compile_candidate_menu(_catalog(), context_report=_report(), prediction_cutoff=BASE, source_rows=[
        {"source_id": "api-football", "available_at": "2026-10-04T07:00:00Z"},
        {"source_id": "news-feed", "available_at": "2026-10-04T07:30:00Z"},
    ])
    assert result["schema_version"] == VERSION
    assert result["candidate_order"] == ["lineup", "tactics"]
    assert result["routes"]["lineup"]["action_id"] == "query-lineup"
    assert result["source_visibility"] == {"api-football": "available", "news-feed": "available"}
    assert result["contract"]["numeric_voi_estimated"] is False


def test_unknown_source_visibility_does_not_become_available():
    result = compile_candidate_menu(_catalog(), context_report=_report(), prediction_cutoff=BASE, source_rows=[
        {"source_id": "api-football", "available_at": "2026-10-04T09:00:00Z"},
    ])
    assert result["source_visibility"] == {"api-football": "unknown"}
    tactics = next(row for row in result["candidate_menu"] if row["candidate_id"] == "tactics")
    assert tactics["availability"] == "unknown"


def test_source_backed_wrapper_uses_binding_report():
    context_report = build_source_backed_context([{
        "source_id": "rulebook", "available_at": "2026-10-04T07:00:00Z",
        "values": {"qualification_status": "draw_sufficient"},
    }], target_key="match-148", prediction_cutoff=BASE)
    result = compile_source_backed_menu(_catalog(), context_report, source_rows=[
        {"source_id": "api-football", "available_at": "2026-10-04T07:00:00Z"},
        {"source_id": "news-feed", "available_at": "2026-10-04T07:00:00Z"},
    ])
    assert result["context"]["qualification_state"] == "draw_sufficient"
    assert result["candidate_order"][0] == "tactics"


def test_compiler_does_not_read_evaluator_payload():
    report = _report()
    report["evaluation_sidecar"] = {"sequence_losses": {"": .5}}
    result = compile_candidate_menu(_catalog(), context_report=report, prediction_cutoff=BASE)
    assert result["candidate_order"] == ["lineup", "tactics"]


def test_rejects_duplicate_candidates_and_route_unit_mismatch():
    duplicate = _catalog() + [deepcopy(_catalog()[0])]
    with pytest.raises(ValueError, match="duplicate candidate_id"):
        compile_candidate_menu(duplicate, context_report=_report(), prediction_cutoff=BASE)
    mismatch = _catalog()
    mismatch[0]["route"]["unit"] = "requests"
    with pytest.raises(ValueError, match="units differ"):
        compile_candidate_menu(mismatch, context_report=_report(), prediction_cutoff=BASE)


def test_rejects_missing_cost_or_route_identity():
    missing = _catalog()
    missing[0]["route"].pop("action_id")
    with pytest.raises(ValueError, match="action_id"):
        compile_candidate_menu(missing, context_report=_report(), prediction_cutoff=BASE)
    missing = _catalog()
    missing[0].pop("declared_cost")
    with pytest.raises(ValueError, match="declared_cost"):
        compile_candidate_menu(missing, context_report=_report(), prediction_cutoff=BASE)


def test_compiler_is_proposal_only_and_does_not_call_sources():
    result = compile_candidate_menu(_catalog(), context_report=_report(), prediction_cutoff=BASE)
    assert result["contract"] == {
        "proposal_only": True,
        "numeric_voi_estimated": False,
        "state_bound_selection_required": True,
        "real_source_requests": 0,
        "real_model_calls": 0,
        "evaluator_fields_used": False,
    }


def test_context_report_with_evaluator_usage_is_rejected():
    report = _report()
    report["evaluator_fields_used"] = True
    with pytest.raises(ValueError, match="evaluator fields"):
        compile_candidate_menu(_catalog(), context_report=report, prediction_cutoff=BASE)
