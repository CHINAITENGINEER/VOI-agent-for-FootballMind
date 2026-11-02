from __future__ import annotations

import pytest

from football_agent.prequery_state import PreQueryStateAnnotation
from football_agent.source_probe import DESIGN, ProbeCandidate, SourceProbeDesign


def candidates():
    return [
        ProbeCandidate("b", "source-b", "https://b.example/tactical", "match-1", "pre_match", "2026-09-27T10:00:00Z", "tactical_intent", "pre_match_structured"),
        ProbeCandidate("a", "source-a", "https://a.example/tactical", "match-1", "pre_match", "2026-09-27T10:00:00Z", "tactical_intent", "pre_match_structured"),
    ]


def selection(seed=7, selection_id="probe-1"):
    return SourceProbeDesign.select(
        selection_id=selection_id, task_id="task-1", query_at="2026-09-27T09:00:00Z",
        candidates=candidates(), latent_state="defensive",
        latent_state_basis="constructed_pre_query_state", seed=seed,
    )


def tactical_annotation():
    return PreQueryStateAnnotation(
        annotation_id="tactical-1", target_key="match-1", state_variable="tactical_intent",
        value="defensive", basis="pre_query_external_annotation",
        provenance_kind="pre_match_statement", source_ids=("coach-statement",),
        available_at="2026-09-27T09:00:00Z", observed_at="2026-09-27T09:01:00Z",
        prediction_cutoff="2026-09-27T10:00:00Z", scope="match_plan",
    )


def test_selection_is_canonical_uniform_and_seed_verifiable():
    selected = selection()
    assert [candidate.candidate_id for candidate in selected.candidates] == ["a", "b"]
    assert selected.selection_design == DESIGN
    assert selected.probabilities == {"a": .5, "b": .5}
    assert selected.selected_probability == .5
    assert selected.latent_state_visibility == "private_evaluator_only"
    SourceProbeDesign.verify(selected, seed=7)
    assert selected.as_dict()["candidate_ids"] == ["a", "b"]


def test_same_commitment_replays_same_selection_and_wrong_seed_is_rejected():
    first = selection(seed=11)
    second = selection(seed=11)
    assert first.selected_candidate_id == second.selected_candidate_id
    with pytest.raises(ValueError, match="commitment"):
        SourceProbeDesign.verify(first, seed=12)


def test_selection_binds_prequery_annotation_provenance_without_exposing_value_in_menu():
    selected = SourceProbeDesign.select(
        selection_id="annotated-1", task_id="task-1", query_at="2026-09-27T09:00:00Z",
        candidates=candidates(), latent_state="defensive",
        latent_state_basis="pre_query_external_annotation", seed=7,
        state_annotation=tactical_annotation(),
    )
    record = selected.as_dict()
    assert selected.state_annotation_id == "tactical-1"
    assert selected.state_annotation_variable == "tactical_intent"
    assert selected.state_annotation_source_ids == ("coach-statement",)
    assert record["state_annotation"]["scope"] == "match_plan"
    assert all("defensive" not in candidate["document_scope"] for candidate in record["candidates"])


def test_selection_rejects_annotation_value_or_cutoff_mismatch():
    with pytest.raises(ValueError, match="latent_state"):
        SourceProbeDesign.select(
            selection_id="annotated-bad", task_id="task-1", query_at="2026-09-27T09:00:00Z",
            candidates=candidates(), latent_state="attacking",
            latent_state_basis="pre_query_external_annotation", seed=7,
            state_annotation=tactical_annotation(),
        )
    late = PreQueryStateAnnotation(
        annotation_id="late-cutoff", target_key="match-1", state_variable="tactical_intent",
        value="defensive", basis="pre_query_external_annotation", provenance_kind="pre_match_statement",
        source_ids=("coach-statement",), available_at="2026-09-27T10:05:00Z",
        observed_at="2026-09-27T10:06:00Z", prediction_cutoff="2026-09-27T10:30:00Z", scope="match_plan",
    )
    with pytest.raises(ValueError, match="cutoff"):
        SourceProbeDesign.select(
            selection_id="annotated-late", task_id="task-1", query_at="2026-09-27T09:00:00Z",
            candidates=candidates(), latent_state="defensive",
            latent_state_basis="pre_query_external_annotation", seed=7,
            state_annotation=late,
        )


def test_candidate_menu_requires_distinct_sources_and_shared_scope():
    with pytest.raises(ValueError, match="distinct source"):
        SourceProbeDesign.select(
            selection_id="bad", task_id="task-1", query_at="2026-09-27T09:00:00Z",
            candidates=[candidates()[0], ProbeCandidate("c", "source-b", "https://c.example", "match-1", "pre_match", "2026-09-27T10:00:00Z", "tactical_intent", "pre_match_structured")],
            latent_state="defensive", latent_state_basis="constructed_pre_query_state", seed=1,
        )


def test_bind_return_records_selection_probability_and_private_state():
    selected = selection()
    row = SourceProbeDesign.bind_return(
        selected, seed=7, row_id="return-1", return_label="source_support",
        available_at="2026-09-27T09:05:00Z", observed_at="2026-09-27T09:06:00Z",
        cluster_id="cluster-1",
    )
    assert row.selection_record_id == "probe-1"
    assert row.selection_probability == .5
    assert row.source_id in {"source-a", "source-b"}
    assert row.latent_state == "defensive"


def test_bound_rows_keep_unselected_candidate_unlabeled_and_support_strict_fit():
    rows, selections = [], []
    for index, state in enumerate(("defensive", "defensive", "defensive", "defensive",
                                   "attacking", "attacking", "attacking", "attacking")):
        chosen = SourceProbeDesign.select(
            selection_id=f"p-{index}", task_id=f"task-{index}", query_at="2026-09-27T09:00:00Z",
            candidates=candidates(), latent_state=state,
            latent_state_basis="constructed_pre_query_state", seed=index + 20,
        )
        selections.append(chosen)
        rows.append(SourceProbeDesign.bind_return(
            chosen, seed=index + 20, row_id=f"r-{index}",
            return_label="source_support" if state == "defensive" else "source_challenge",
            available_at="2026-09-27T09:05:00Z", observed_at="2026-09-27T09:06:00Z",
            cluster_id=f"cluster-{index}",
        ))
    binding = SourceProbeDesign.validate_bound_rows(selections, rows)
    assert binding["selection_count"] == 8 and binding["unobserved_selection_count"] == 0
    assert binding["labels_visible_to_agent"] is False
    ledger = SourceProbeDesign.ledger(selections, rows)
    source_counts = {source: sum(row.source_id == source for row in rows) for source in ("source-a", "source-b")}
    assert sum(source_counts.values()) == 8
    for source in source_counts:
        estimate = ledger.fit(source_id=source, time_bucket="pre_match",
                              states=("defensive", "attacking"),
                              outcomes=("source_support", "source_challenge"),
                              min_rows_per_state=1, require_selection_records=True)
        if source_counts[source]:
            assert estimate.status == "SUPPORTED_CONSTRUCTED_POINT_ESTIMATE"


def test_missing_probe_binding_is_rejected():
    selected = selection()
    row = SourceProbeDesign.bind_return(
        selected, seed=7, row_id="return-1", return_label="source_support",
        available_at="2026-09-27T09:05:00Z", observed_at="2026-09-27T09:06:00Z",
        cluster_id="cluster-1",
    )
    with pytest.raises(ValueError, match="unknown probe selection"):
        SourceProbeDesign.validate_bound_rows([selection(selection_id="different")], [row])


def test_duplicate_return_for_one_probe_is_rejected():
    selected = selection()
    first = SourceProbeDesign.bind_return(
        selected, seed=7, row_id="return-1", return_label="source_support",
        available_at="2026-09-27T09:05:00Z", observed_at="2026-09-27T09:06:00Z",
        cluster_id="cluster-1",
    )
    second = SourceProbeDesign.bind_return(
        selected, seed=7, row_id="return-2", return_label="source_challenge",
        available_at="2026-09-27T09:05:00Z", observed_at="2026-09-27T09:06:00Z",
        cluster_id="cluster-2",
    )
    with pytest.raises(ValueError, match="one observed return"):
        SourceProbeDesign.validate_bound_rows([selected], [first, second])
