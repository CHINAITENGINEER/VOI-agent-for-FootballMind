from __future__ import annotations

import pytest

from football_agent.prequery_state import PreQueryStateAnnotation, PreQueryStateRegistry


TARGET = "match-1"
CUTOFF = "2026-09-27T10:00:00Z"


def external(*, annotation_id="qualification-1", state_variable="qualification_status",
             value="already_qualified", source_ids=("official-standings",),
             provenance_kind="official_pre_match_rule", scope="group_stage"):
    return PreQueryStateAnnotation(
        annotation_id=annotation_id, target_key=TARGET, state_variable=state_variable,
        value=value, basis="pre_query_external_annotation", provenance_kind=provenance_kind,
        source_ids=source_ids, available_at="2026-09-27T09:00:00Z",
        observed_at="2026-09-27T09:01:00Z", prediction_cutoff=CUTOFF, scope=scope,
    )


def test_annotation_keeps_evidence_time_scope_and_private_visibility():
    annotation = external()
    manifest = annotation.manifest()
    assert manifest["state_variable"] == "qualification_status"
    assert manifest["source_ids"] == ["official-standings"]
    assert manifest["available_at"] < manifest["prediction_cutoff"]
    assert manifest["visibility"] == "private_evaluator_only"


def test_registry_preserves_qualification_and_tactical_intent_as_independent_labels():
    registry = PreQueryStateRegistry([
        external(),
        external(annotation_id="tactical-1", state_variable="tactical_intent",
                 value="defensive", source_ids=("coach-statement",),
                 provenance_kind="pre_match_statement", scope="match_plan"),
    ])
    eligible = registry.eligible(target_key=TARGET, prediction_cutoff=CUTOFF)
    assert [item.state_variable for item in eligible] == ["qualification_status", "tactical_intent"]
    assert registry.manifest()["state_variables_are_independent"] is True
    assert registry.manifest()["post_match_result_derivation"] is False


def test_registry_excludes_annotation_for_other_target_or_cutoff():
    registry = PreQueryStateRegistry([external()])
    assert registry.eligible(target_key="other-match", prediction_cutoff=CUTOFF) == ()
    assert registry.eligible(target_key=TARGET, prediction_cutoff="2026-09-27T10:30:00Z") == ()


def test_external_annotation_requires_source_provenance():
    with pytest.raises(ValueError, match="source ID"):
        external(source_ids=())


def test_constructed_annotation_has_explicit_constructed_provenance():
    annotation = PreQueryStateAnnotation(
        annotation_id="fixture-1", target_key=TARGET, state_variable="tactical_intent",
        value="defensive", basis="constructed_pre_query_state", provenance_kind="constructed_fixture",
        source_ids=(), available_at="2026-09-27T09:00:00Z",
        observed_at="2026-09-27T09:00:00Z", prediction_cutoff=CUTOFF, scope="fixture",
    )
    assert annotation.source_ids == ()
    assert annotation.manifest()["provenance_kind"] == "constructed_fixture"


def test_constructed_basis_cannot_claim_external_provenance():
    with pytest.raises(ValueError, match="constructed_fixture"):
        PreQueryStateAnnotation(
            annotation_id="bad-1", target_key=TARGET, state_variable="tactical_intent",
            value="defensive", basis="constructed_pre_query_state", provenance_kind="pre_match_statement",
            source_ids=("coach-statement",), available_at="2026-09-27T09:00:00Z",
            observed_at="2026-09-27T09:01:00Z", prediction_cutoff=CUTOFF, scope="fixture",
        )


def test_post_cutoff_observation_is_rejected_at_annotation_boundary():
    with pytest.raises(ValueError, match="available_at <= observed_at <= cutoff"):
        PreQueryStateAnnotation(
            annotation_id="late-1", target_key=TARGET, state_variable="lineup_status",
            value="starting_xi", basis="pre_query_external_annotation", provenance_kind="pre_match_lineup",
            source_ids=("lineup-feed",), available_at="2026-09-27T10:05:00Z",
            observed_at="2026-09-27T10:06:00Z", prediction_cutoff=CUTOFF, scope="match",
        )


def test_validate_for_requires_the_frozen_cutoff():
    annotation = external()
    with pytest.raises(ValueError, match="cutoff"):
        annotation.validate_for(target_key=TARGET, prediction_cutoff="2026-09-27T10:30:00Z")
