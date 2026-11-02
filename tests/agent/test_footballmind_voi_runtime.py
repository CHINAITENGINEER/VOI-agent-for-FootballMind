import pytest

from football_agent.footballmind_voi_runtime import (
    VERSION,
    build_match_voi_state,
    run_match_voi_step,
)


CUTOFF = "2026-10-09T12:00:00Z"


def _candidate_menu():
    return [{
        "candidate_id": "query.injury_availability",
        "kind": "injury_availability",
        "action_kind": "query",
        "source_ids": ["injury_report"],
        "requires_source_ids": [],
        "aggregation_mode": "any_usable",
        "action_availability": "available",
        "resource_cost": {"requests": 1},
        "utility_cost": {"amount": 0.2, "unit": "decision_loss", "source": "test"},
    }]


def _estimate(expected_value=0.8):
    return {
        "query.injury_availability": {
            "expected_value": expected_value,
            "source": "offline-test-model",
            "unit": "decision_loss",
            "training_cutoff": "2026-01-01T00:00:00Z",
            "available_at": "2026-10-09T11:00:00Z",
        }
    }


def test_match_only_defaults_to_agent_owned_defer_without_value_or_cost():
    result = run_match_voi_step(
        "Santos vs Flamengo",
        {"prediction_cutoff": CUTOFF},
    )
    assert result["schema_version"] == VERSION
    assert result["audit"]["user_scope_question"] is None
    assert result["audit"]["selection_owner"] == "agent"
    assert result["decision_trace"]["decision"] == "defer"
    assert result["audit"]["stop_or_defer_reason"] in {
        "candidate_menu_incomplete", "unresolved_action_cost_or_value",
    }
    assert result["audit"]["network_requests"] == 0
    assert result["audit"]["outcome_fields_consumed"] is False


def test_state_bound_estimate_selects_positive_candidate():
    public = {
        "prediction_cutoff": CUTOFF,
        "candidate_menu": _candidate_menu(),
        "candidate_generation_complete": True,
    }
    result = run_match_voi_step(
        "Santos vs Flamengo",
        public,
        estimates=_estimate(),
        remaining_budget={"requests": 1},
    )
    assert result["decision_trace"]["decision"] == "acquire"
    assert result["decision_trace"]["selected_candidate_id"] == "query.injury_availability"
    row = result["candidate_values"][0]
    assert row["expected_value"] == 0.8
    assert row["cost"] == 0.2
    assert row["net_value"] == pytest.approx(0.6)
    assert result["audit"]["trace_digest"] == result["decision_trace"]["trace_digest"]


def test_public_outcome_fields_are_rejected_before_selection():
    with pytest.raises(ValueError, match="outcome/evaluator"):
        build_match_voi_state(
            "Santos vs Flamengo",
            {"prediction_cutoff": CUTOFF, "score": "3-0"},
        )
