from __future__ import annotations

import pytest

from football_agent.decision_value import FiniteDecisionModel, ObservationTool
from football_agent.robust_decision_value import RobustFiniteDecisionModel


HYPOTHESES = {
    "starters_strong": (.75, .17, .08),
    "starters_weak": (.75, .17, .08),
    "reserves_strong": (.70, .20, .10),
    "reserves_weak": (.35, .30, .35),
}
PRIOR = {name: .25 for name in HYPOTHESES}


def _lineup(exact: bool) -> dict[str, tuple[float, ...]]:
    if exact:
        return {
            name: ((.95, .0, .05) if name.startswith("starters") else (.0, .95, .05))
            for name in HYPOTHESES
        }
    return {name: (.45, .45, .10) for name in HYPOTHESES}


def _reserve(exact: bool) -> dict[str, tuple[float, ...]]:
    if exact:
        return {
            name: ((.95, .0, .05) if name.endswith("strong") else (.0, .95, .05))
            for name in HYPOTHESES
        }
    return {name: (.45, .45, .10) for name in HYPOTHESES}


def _model(*, exact_lineup: bool, exact_reserve: bool) -> FiniteDecisionModel:
    return FiniteDecisionModel(
        HYPOTHESES,
        PRIOR,
        {
            "lineup": ObservationTool(
                ("starters", "reserves", "unavailable"), _lineup(exact_lineup), .015,
                evidence_id="lineup-report",
            ),
            "reserve_quality": ObservationTool(
                ("strong", "weak", "unavailable"), _reserve(exact_reserve), .015,
                evidence_id="reserve-report",
            ),
        },
        assumptions=("manually specified source-quality mechanism",),
    )


def test_worst_case_envelope_switches_from_uncertain_lineup_to_reliable_reserve_signal():
    robust = RobustFiniteDecisionModel({
        "fresh_source": _model(exact_lineup=True, exact_reserve=True),
        "stale_lineup_source": _model(exact_lineup=False, exact_reserve=True),
    })

    result = robust.evaluate(remaining_budget=.04, horizon=1)
    by_action = {row["action_id"]: row for row in result["candidates"] if row["feasible"]}

    assert result["action_id"] == "reserve_quality"
    assert by_action["lineup"]["worst_case_planning_net_value"] < 0
    assert by_action["reserve_quality"]["worst_case_planning_net_value"] > 0
    assert result["certificate"]["kind"] == "ROBUST_WORST_CASE_ENVELOPE"
    assert result["certificate"]["global_claim"] is False


def test_stop_when_all_declared_sources_have_a_nonpositive_lower_envelope():
    robust = RobustFiniteDecisionModel({
        "stale_lineup": _model(exact_lineup=False, exact_reserve=False),
        "fresh_lineup_but_stale_reserve": _model(exact_lineup=True, exact_reserve=False),
    })

    result = robust.evaluate(remaining_budget=.04, horizon=1)

    assert result["action_id"] == "STOP"
    assert result["reason"] == "no_positive_worst_case_net_value"
    assert result["worst_case_planning_net_value"] <= 0


def test_history_is_replayed_in_every_declared_model_without_merging_posteriors():
    robust = RobustFiniteDecisionModel({
        "fresh": _model(exact_lineup=True, exact_reserve=True),
        "stale": _model(exact_lineup=False, exact_reserve=True),
    })

    result = robust.evaluate(
        remaining_budget=.025,
        horizon=1,
        observations=(("lineup", "starters"),),
    )

    assert result["observations"] == [["lineup", "starters"]]
    assert result["action_id"] in {"reserve_quality", "STOP"}
    assert set(result["scenario_current_risks"]) == {"fresh", "stale"}


def test_models_must_share_outcome_model_and_action_contract():
    incompatible = FiniteDecisionModel(
        HYPOTHESES,
        PRIOR,
        {"lineup": ObservationTool(("starters", "reserves"), {
            name: ((1.0, 0.0) if name.startswith("starters") else (0.0, 1.0))
            for name in HYPOTHESES
        }, .015, evidence_id="lineup-report"),
         "reserve_quality": ObservationTool(("strong", "weak", "unavailable"), _reserve(True), .015, evidence_id="reserve-report")},
    )

    with pytest.raises(ValueError, match="observation alphabet"):
        RobustFiniteDecisionModel({"a": _model(exact_lineup=True, exact_reserve=True), "b": incompatible})


def test_manifest_declares_lower_envelope_and_no_hindsight_label():
    robust = RobustFiniteDecisionModel(
        {"fresh": _model(exact_lineup=True, exact_reserve=True)},
        assumptions=("source quality is an uncertainty set, not a fitted posterior",),
    )
    manifest = robust.manifest()

    assert manifest["schema_version"] == "robust-finite-voi-envelope-v1"
    assert manifest["interpretation"].startswith("lower envelope")
    assert "sampled_result" not in str(manifest)
    assert "realized_loss" not in str(manifest)
