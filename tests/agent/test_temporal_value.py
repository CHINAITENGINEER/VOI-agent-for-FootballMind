import pytest

from football_agent.decision_value import FiniteDecisionModel, ObservationTool
from football_agent.temporal_inquiry import TemporalInquiryGate
from football_agent.temporal_value import TemporalEVSIEvaluator


def model():
    return FiniteDecisionModel(
        {"starters": (1, 0, 0), "reserves": (0, 0, 1)},
        {"starters": 0.5, "reserves": 0.5},
        {"lineup": ObservationTool(("starters", "reserves"),
                                     {"starters": (1, 0), "reserves": (0, 1)},
                                     0.01, evidence_id="lineup-report")},
    )


def gate(cutoff="2026-09-23T09:30:00Z", *, status=200, scope="pre_match_structured", error=None,
         available="2026-09-23T09:15:00Z"):
    return TemporalInquiryGate(
        prediction_cutoff=cutoff, kickoff="2026-09-23T10:30:00Z", allowed_hosts=["example.com"],
        entries=[{"source_id": "lineup-source", "url": "https://example.com/lineup", "publisher": "Official",
                  "information_role": "lineup", "menu_note": "Frozen source.", "document_scope": scope,
                  "observed_at": "2026-09-23T09:10:00Z", "available_at": available,
                  "http_status": status, "error_class": error}],
    )


def test_temporal_unavailable_is_explicit_and_does_not_become_zero_evsi():
    evaluator = TemporalEVSIEvaluator(model(), {"lineup": "lineup-source"}, gate(cutoff="2026-09-23T09:00:00Z"))
    result = evaluator.evaluate(model().initial_state(), remaining_budget=0.02, horizon=1)
    row = result["candidates"][0]
    assert result["action_id"] == "STOP"
    assert row["reason"] == "temporally_unavailable"
    assert row["availability_status"] == "EXCLUDED_AFTER_CUTOFF"
    assert row["immediate_evsi"] is None and row["planning_net_value"] is None
    assert result["open_world_value"] is None


def test_same_model_gets_exact_evsi_once_source_is_temporally_eligible():
    evaluator = TemporalEVSIEvaluator(model(), {"lineup": "lineup-source"}, gate())
    result = evaluator.evaluate(model().initial_state(), remaining_budget=0.02, horizon=1)
    row = result["candidates"][0]
    assert result["action_id"] == "lineup"
    assert row["availability_status"] == "ELIGIBLE"
    assert row["immediate_evsi"] == pytest.approx(__import__("math").log(2))
    assert row["planning_net_value"] == pytest.approx(__import__("math").log(2) - 0.01)


@pytest.mark.parametrize("kwargs,expected", [
    ({"status": None, "error": "SSLError"}, "FAILED"),
    ({"scope": "unknown"}, "QUARANTINED_SCOPE"),
])
def test_failed_or_quarantined_sources_are_not_assigned_numeric_value(kwargs, expected):
    evaluator = TemporalEVSIEvaluator(model(), {"lineup": "lineup-source"}, gate(**kwargs))
    result = evaluator.evaluate(model().initial_state(), remaining_budget=0.02)
    assert result["action_id"] == "STOP"
    assert result["candidates"][0]["availability_status"] == expected
    assert result["candidates"][0]["immediate_evsi"] is None


def test_already_acquired_observation_remains_valid_when_source_later_expires():
    m = model()
    evaluator = TemporalEVSIEvaluator(m, {"lineup": "lineup-source"}, gate(cutoff="2026-09-23T09:00:00Z"))
    state = m.observe(m.initial_state(), "lineup", "starters")
    result = evaluator.evaluate(state, remaining_budget=0.02)
    assert result["action_id"] == "STOP"
    assert result["candidates"][0]["reason"] == "already_observed"


def test_bindings_require_registered_temporal_sources():
    with pytest.raises(ValueError, match="registered temporal source"):
        TemporalEVSIEvaluator(model(), {"lineup": "missing-source"}, gate())
