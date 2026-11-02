from __future__ import annotations

import math

import pytest

from football_agent.decision_value import FiniteDecisionModel
from football_agent.inquiry_returns import OUTCOMES
from football_agent.return_likelihood import (
    CENSORED_AT_CUTOFF,
    OBSERVED,
    NOT_SELECTED,
    ReturnLikelihoodLedger,
    ReturnObservation,
    from_return_dataset,
)


CUTOFF = "2026-09-27T10:00:00Z"


def row(
    row_id: str,
    *,
    state: str | None = "defensive",
    basis: str | None = "constructed_pre_query_state",
    label: str | None = "source_support",
    status: str = OBSERVED,
    origin: str = "constructed",
    design: str = "constructed_randomized_source_probe",
    source: str = "tactical-source",
    bucket: str = "pre_match",
    cluster: str | None = None,
    available: str = "2026-09-27T09:55:00Z",
    observed: str = "2026-09-27T09:56:00Z",
):
    return ReturnObservation(
        row_id=row_id, source_id=source, time_bucket=bucket,
        query_at="2026-09-27T09:50:00Z", prediction_cutoff=CUTOFF,
        status=status, return_label=label if status == OBSERVED else None,
        available_at=available, observed_at=observed, origin=origin,
        latent_state=state, latent_state_basis=basis,
        selection_design=design, cluster_id=cluster or row_id,
    )


def test_cutoff_and_selection_censoring_never_becomes_a_return_count():
    late = row("late", status=CENSORED_AT_CUTOFF, label=None,
               available="2026-09-27T10:01:00Z", observed="2026-09-27T10:02:00Z")
    unselected = row("unselected", status=NOT_SELECTED, label=None,
                     state=None, basis=None, available=None, observed=None)
    ledger = ReturnLikelihoodLedger([row("early"), late, unselected])
    estimate = ledger.fit(source_id="tactical-source", time_bucket="pre_match", states=("defensive",))
    assert estimate.status == "INSUFFICIENT_SUPPORT"
    assert estimate.selected_row_ids == ("early",)
    assert set(estimate.excluded_row_ids) == {"late", "unselected"}
    assert ledger.summary()["status_counts"][CENSORED_AT_CUTOFF] == 1


def test_constructed_point_estimate_is_smoothing_only_and_feeds_observation_tool():
    rows = [
        row("d1", state="defensive", label="source_support"),
        row("d2", state="defensive", label="source_support"),
        row("a1", state="attacking", label="source_challenge"),
        row("a2", state="attacking", label="source_challenge"),
    ]
    estimate = ReturnLikelihoodLedger(rows).fit(
        source_id="tactical-source", time_bucket="pre_match",
        states=("defensive", "attacking"), outcomes=("source_support", "source_challenge"), alpha=0.5,
    )
    assert estimate.status == "SUPPORTED_CONSTRUCTED_POINT_ESTIMATE"
    assert estimate.calibration_status == "NOT_CALIBRATED"
    assert estimate.counts["defensive"] == {"source_support": 2, "source_challenge": 0}
    assert math.fsum(estimate.likelihoods["defensive"]) == pytest.approx(1.0)
    tool = estimate.observation_tool(cost=0.01, evidence_id="tactical-report")
    assert tool.outcomes == ("source_support", "source_challenge")
    assert tuple(tool.likelihoods["attacking"]) == pytest.approx((1 / 6, 5 / 6))


def test_adaptive_operational_agent_rows_remain_descriptive():
    rows = [
        row("o1", state="defensive", basis="pre_query_external_annotation", origin="operational_network",
            design="agent_adaptive"),
        row("o2", state="defensive", basis="pre_query_external_annotation", origin="operational_network",
            design="agent_adaptive", label="source_challenge"),
    ]
    estimate = ReturnLikelihoodLedger(rows).fit(
        source_id="tactical-source", time_bucket="pre_match", states=("defensive",), min_rows_per_state=1,
    )
    assert estimate.status == "UNSUPPORTED_ADAPTIVE_SELECTION"
    assert estimate.likelihoods is None


def test_match_result_cannot_define_theta():
    estimate = ReturnLikelihoodLedger([
        row("leak", basis="match_result"),
    ]).fit(source_id="tactical-source", time_bucket="pre_match", states=("defensive",), min_rows_per_state=1)
    assert estimate.status == "UNSUPPORTED_MATCH_RESULT_LEAKAGE"
    assert estimate.likelihoods is None


def test_source_return_without_prequery_state_has_no_conditional_law():
    estimate = ReturnLikelihoodLedger([
        row("unknown", state=None, basis=None),
        row("unknown-2", state=None, basis=None, label="source_challenge"),
    ]).fit(source_id="tactical-source", time_bucket="pre_match", states=("defensive",), min_rows_per_state=1)
    assert estimate.status == "UNSUPPORTED_NO_LATENT_STATE"
    assert estimate.likelihoods is None


def test_reused_dependence_cluster_is_not_counted_as_iid():
    estimate = ReturnLikelihoodLedger([
        row("same-a", cluster="same"), row("same-b", cluster="same"),
    ]).fit(source_id="tactical-source", time_bucket="pre_match", states=("defensive",), min_rows_per_state=1)
    assert estimate.status == "UNSUPPORTED_DEPENDENCE_CLUSTER_REUSE"


def test_dataset_adapter_requires_explicit_semantic_resolvers_and_preserves_cutoff():
    dataset = {
        "schema_version": "inquiry-return-dataset-v2", "as_of": CUTOFF,
        "rows": [
            {"row_id": "observed", "origin": "constructed", "cluster_id": "c1",
             "pre_query": {"query_decided_at": "2026-09-27T09:50:00Z"},
             "observed_return": {"available_at": "2026-09-27T09:55:00Z"}},
            {"row_id": "late", "origin": "constructed", "cluster_id": "c2",
             "pre_query": {"query_decided_at": "2026-09-27T09:50:00Z"},
             "observed_return": {"available_at": "2026-09-27T10:05:00Z"}},
        ],
    }
    ledger = from_return_dataset(
        dataset,
        source_id_resolver=lambda raw: "tactical-source",
        return_label_resolver=lambda raw: "source_support",
        latent_state_resolver=lambda raw: ("defensive", "constructed_pre_query_state"),
        time_bucket_resolver=lambda raw: "pre_match",
        selection_design_resolver=lambda raw: "constructed_randomized_source_probe",
    )
    assert [item.status for item in ledger.rows] == [OBSERVED, CENSORED_AT_CUTOFF]
    assert ledger.rows[1].return_label is None
    assert all(item in OUTCOMES for item in ("source_support", "source_challenge"))


def test_supported_return_point_estimate_bridges_to_finite_query_value_as_assumptions_only():
    estimate = ReturnLikelihoodLedger([
        row("d1", state="defensive", label="source_support"),
        row("d2", state="defensive", label="source_support"),
        row("a1", state="attacking", label="source_challenge"),
        row("a2", state="attacking", label="source_challenge"),
    ]).fit(source_id="tactical-source", time_bucket="pre_match",
           states=("defensive", "attacking"), outcomes=("source_support", "source_challenge"))
    model = FiniteDecisionModel(
        {"defensive": (.28, .55, .17), "attacking": (.55, .20, .25)},
        {"defensive": .5, "attacking": .5},
        {"tactical": estimate.observation_tool(cost=.01, evidence_id="tactical-source-probe")},
    )
    model_row = next(item for item in model.evaluate(model.initial_state(), remaining_budget=.05)["candidates"]
                     if item["action_id"] == "tactical")
    assessment = estimate.query_value_assessment(candidate_id="coach", model_row=model_row)
    assert assessment.value_status == "MODELED_DECLARED_ASSUMPTIONS"
    assert assessment.calibration_status == "ASSUMPTION_ONLY"
    assert "NOT_CALIBRATED" in assessment.reason


def test_unsupported_return_estimate_stays_unmeasured_at_value_bridge():
    estimate = ReturnLikelihoodLedger([row("only-one")]).fit(
        source_id="tactical-source", time_bucket="pre_match", states=("defensive",), min_rows_per_state=2,
    )
    assessment = estimate.query_value_assessment(
        candidate_id="coach", model_row={"feasible": True, "immediate_evsi": .2, "cost": .01},
    )
    assert estimate.status == "INSUFFICIENT_SUPPORT"
    assert assessment.value_status == "VALUE_UNMEASURED"
    assert assessment.expected_value is None and assessment.net_value is None
