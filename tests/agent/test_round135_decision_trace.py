"""Round135-C tests for the deterministic decision trace."""
import pytest
from football_agent.round135_decision_trace import decide_next_action, VERSION

CUTOFF = "2026-01-01T00:00:00Z"

def cand(cid, *, value=None, cost=.1, status="available", action="executable", kind=None, training="2025-01-01T00:00:00Z", source="cal-v1", unit="loss"):
    d = {"candidate_id": cid, "cost": cost, "cost_unit": unit,
         "provenance_gate": {"status": status, "action_eligibility": action}}
    if kind: d["action_kind"] = kind
    if value is not None:
        d.update(expected_value=value, value_unit=unit,
                 value_estimate_source=source, value_training_cutoff=training)
    return d

def test_numeric_gate_selects_highest_net_and_reserves_without_spend():
    out = decide_next_action(episode_id="e", prediction_cutoff=CUTOFF,
        candidates=[cand("a", value=.4, cost=.1), cand("b", value=.5, cost=.3)],
        candidate_generation_complete=True, budget=1)
    assert out["schema_version"] == VERSION
    assert out["decision"] == "acquire"
    assert out["selected_candidate_id"] == "a"
    assert out["reservation"] == {"candidate_id":"a", "reserved_cost":.1, "actual_spent":0.0, "status":"reserved"}
    assert out["cost_ledger"][-1]["actual_spent"] == 0.0

def test_unknown_value_defers_and_is_not_treated_as_zero():
    out = decide_next_action(episode_id="e", prediction_cutoff=CUTOFF,
        candidates=[cand("weather"), cand("injury", value=0, cost=.1)],
        candidate_generation_complete=True)
    assert out["decision"] == "defer"
    assert out["stop_reason"] == "defer_unknown_estimate"
    assert out["selected_candidate_id"] is None

def test_unknown_does_not_block_a_positive_known_candidate():
    out = decide_next_action(episode_id="e", prediction_cutoff=CUTOFF,
        candidates=[cand("weather"), cand("injury", value=.5, cost=.1)],
        candidate_generation_complete=True)
    assert out["selected_candidate_id"] == "injury"

def test_incomplete_menu_never_claims_stop():
    out = decide_next_action(episode_id="e", prediction_cutoff=CUTOFF,
        candidates=[cand("a", value=0, cost=.1)], candidate_generation_complete=False)
    assert out["decision"] == "defer"
    assert out["stop_reason"] == "defer_incomplete_candidate_menu"

def test_stop_certificate_requires_complete_menu_and_nonpositive_all():
    out = decide_next_action(episode_id="e", prediction_cutoff=CUTOFF,
        candidates=[cand("a", value=.1, cost=.2), cand("b", value=0, cost=.1)],
        candidate_generation_complete=True)
    assert out["decision"] == "stop"
    assert out["stop_reason"] == "stop_all_feasible_net_utility_nonpositive"

def test_stale_or_missing_evidence_remains_queryable_and_auto_refreshes():
    out = decide_next_action(episode_id="e", prediction_cutoff=CUTOFF,
        candidates=[cand("stale", value=.4, status="stale", kind=None)],
        candidate_generation_complete=True)
    assert out["selected_action_kind"] == "refresh"
    assert out["selected_candidate_id"] == "stale"

def test_future_conflict_provenance_defers_even_with_numeric_value():
    out = decide_next_action(episode_id="e", prediction_cutoff=CUTOFF,
        candidates=[cand("future", value=.9, status="future")], candidate_generation_complete=True)
    assert out["decision"] == "defer"
    assert out["stop_reason"] == "provenance_deferred"

def test_action_eligibility_is_separate_from_evidence_eligibility():
    out = decide_next_action(episode_id="e", prediction_cutoff=CUTOFF,
        candidates=[cand("missing", value=.9, status="missing", action="executable")], candidate_generation_complete=True)
    assert out["selected_candidate_id"] == "missing"
    blocked = decide_next_action(episode_id="e", prediction_cutoff=CUTOFF,
        candidates=[cand("blocked", value=.9, status="missing", action="unavailable")], candidate_generation_complete=True)
    assert blocked["selected_candidate_id"] is None
    assert blocked["stop_reason"] == "provenance_deferred"

def test_value_training_after_cutoff_is_rejected():
    with pytest.raises(ValueError, match="after prediction cutoff"):
        decide_next_action(episode_id="e", prediction_cutoff=CUTOFF,
          candidates=[cand("a", value=.2, training="2026-02-01T00:00:00Z")], candidate_generation_complete=True)

def test_mismatched_units_are_rejected():
    with pytest.raises(ValueError, match="units differ"):
        decide_next_action(episode_id="e", prediction_cutoff=CUTOFF,
          candidates=[{"candidate_id":"a", "cost":.1, "cost_unit":"credits",
                       "expected_value":.2, "value_unit":"loss", "value_estimate_source":"s",
                       "value_training_cutoff":"2025-01-01T00:00:00Z"}], candidate_generation_complete=True)

def test_dependency_and_budget_gate_are_explicit():
    out = decide_next_action(episode_id="e", prediction_cutoff=CUTOFF,
       candidates=[dict(cand("a", value=.9), dependencies=["lineup"])], budget=.05, candidate_generation_complete=True)
    assert out["decision"] == "defer"
    assert out["stop_reason"] in {"defer_budget", "provenance_deferred"}

def test_trace_is_deterministic_and_does_not_copy_result_fields():
    c = cand("a", value=.2)
    c["result"] = {"score":"3-0"}
    a = decide_next_action(episode_id="e", prediction_cutoff=CUTOFF, candidates=[c], candidate_generation_complete=True)
    b = decide_next_action(episode_id="e", prediction_cutoff=CUTOFF, candidates=[c], candidate_generation_complete=True)
    assert a == b
    text = str(a)
    assert "3-0" not in text and "result" not in text.lower()
