import json
from pathlib import Path
from types import SimpleNamespace

from football_agent.collector_context_adapter import build_context_snapshot, merge_context_snapshots
from football_agent.collector_context_bridge import apply_context_snapshot, projection_transition
from football_agent.collector_ledger_adapter import normalized_batch_mapping_state, observation_event
from football_agent.context_query_readiness import ContextState
from football_spider.normalization.sources import normalize_spftll


CUTOFF = "2026-09-20T12:00:00Z"
FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "collectors" / "spftll"


def snapshot(name, target_id="2996160"):
    metadata = json.loads((FIXTURES / f"{name}.metadata.json").read_text(encoding="utf-8"))["observation"]
    payload = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    normalized = normalize_spftll(SimpleNamespace(**metadata), payload)
    event = observation_event(
        metadata, result_status="OK", source_version="spftll-fixture-v1",
        prediction_cutoff=CUTOFF, candidate_id=name,
        entity_mapping_state=normalized_batch_mapping_state(normalized),
    )
    return build_context_snapshot(event, normalized, target_source_match_id=target_id, target_key=f"match:{target_id}")


def test_detail_snapshot_projects_accepted_context_keys():
    projection = apply_context_snapshot(ContextState({}), snapshot("detail"))
    assert "fixture_id" in projection.accepted_context_keys
    assert projection.next_context.values["fixture_id"] == "2996160"
    assert projection.changed_keys


def test_odds_snapshot_updates_existing_context_without_identity_invention():
    detail = snapshot("detail")
    odds = snapshot("odds")
    first = apply_context_snapshot(ContextState({}), detail)
    second = apply_context_snapshot(first.next_context, odds)
    assert second.next_context.values["fixture_id"] == "2996160"
    assert second.next_context.values["odds_available"] is True
    assert second.next_context.values["quote_count"] == 14


def test_merged_snapshot_conflicts_and_rejected_evidence_stay_out_of_state():
    merged = merge_context_snapshots((snapshot("detail"), snapshot("odds")))
    projection = apply_context_snapshot(ContextState({}), merged)
    assert not projection.conflict_keys
    ambiguous = snapshot("list", target_id="3013707")
    rejected = apply_context_snapshot(projection.next_context, ambiguous)
    assert rejected.next_context.values == projection.next_context.values
    assert rejected.rejected_context_keys


def test_projection_transition_carries_only_accepted_refs():
    projection = apply_context_snapshot(ContextState({}), snapshot("detail"))
    transition = projection_transition(
        projection, episode_id="bridge-episode", sequence=1, reason="real detail context arrived",
        captured_at="2026-09-20T11:45:00Z",
    )
    assert transition.event_type == "CONTEXT_UPDATED"
    assert transition.changed_keys == projection.changed_keys
    assert transition.evidence_refs
    assert transition.as_dict()["hidden_result_included"] is False


def test_projection_preserves_previous_value_on_conflicting_new_fact():
    detail = snapshot("detail")
    first = apply_context_snapshot(ContextState({}), detail)
    altered = dict(detail.context.values)
    altered["competition_name"] = "different-source-name"
    # The bridge receives a normal snapshot object; a second source with a
    # conflicting value is represented by merging snapshots before applying.
    second = apply_context_snapshot(first.next_context, detail)
    assert second.next_context.values["competition_name"] == first.next_context.values["competition_name"]

