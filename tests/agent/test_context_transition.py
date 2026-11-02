from pathlib import Path

import pytest

from football_agent.context_transition import ContextTransition, ContextTransitionLedger


CUTOFF = "2026-09-29T12:00:00Z"


def event(sequence, event_type, previous, following, *, candidate=None, action=None, reason="state update"):
    return ContextTransition.from_transition(
        episode_id="episode-transition", target_key="match-transition", sequence=sequence,
        event_type=event_type, previous_context=previous, next_context=following,
        trigger="test", reason=reason, prediction_cutoff=CUTOFF,
        captured_at=f"2026-09-29T0{sequence}:00:00Z", selected_candidate_id=candidate,
        action=action, source_id="fixture-source", evidence_refs=[f"raw:{sequence}"],
    )


def test_transition_identity_records_exact_context_diff():
    item = event(1, "INITIALIZED", {}, {"qualification": None, "reserve_strength": None})
    assert item.changed_keys == ("qualification", "reserve_strength")
    assert item.as_dict()["hidden_result_included"] is False


def test_query_selected_event_is_audit_only_and_keeps_context():
    item = event(1, "QUERY_SELECTED", {"qualification": None}, {"qualification": None},
                 candidate="qualification-refresh", action="QUERY")
    assert item.changed_keys == ()
    assert item.selected_candidate_id == "qualification-refresh"


def test_context_transition_rejects_evaluator_fields():
    with pytest.raises(ValueError, match="evaluator-only"):
        event(1, "INITIALIZED", {}, {"result": "home_win"})


def test_hash_chained_ledger_replays_context_updates(tmp_path: Path):
    ledger = ContextTransitionLedger(tmp_path)
    first = event(1, "INITIALIZED", {}, {"qualification": None, "reserve_strength": None})
    second = event(2, "QUERY_SELECTED", first.next_context, first.next_context,
                   candidate="qualification-refresh", action="QUERY")
    third = event(3, "CONTEXT_UPDATED", second.next_context,
                  {"qualification": "already_qualified", "reserve_strength": None}, reason="fresh standings evidence")
    fourth = event(4, "QUERY_SELECTED", third.next_context, third.next_context,
                   candidate="reserve-strength", action="QUERY")
    fifth = event(5, "CONTEXT_UPDATED", fourth.next_context,
                  {"qualification": "already_qualified", "reserve_strength": "known_strong"}, reason="squad context evidence")
    ledger.append([first, second, third, fourth, fifth])
    assert ledger.verify()["events_verified"] == 5
    replay = ledger.replay_episode("episode-transition", target_key="match-transition")
    assert replay["final_context"]["reserve_strength"] == "known_strong"
    assert [item["event_type"] for item in replay["events"]] == [
        "INITIALIZED", "QUERY_SELECTED", "CONTEXT_UPDATED", "QUERY_SELECTED", "CONTEXT_UPDATED"
    ]


def test_ledger_rejects_duplicate_event_identity(tmp_path: Path):
    ledger = ContextTransitionLedger(tmp_path)
    item = event(1, "INITIALIZED", {}, {"qualification": None})
    ledger.append([item])
    with pytest.raises(ValueError, match="append-only"):
        ledger.append([item])


def test_replay_detects_context_snapshot_gap(tmp_path: Path):
    ledger = ContextTransitionLedger(tmp_path)
    first = event(1, "INITIALIZED", {}, {"qualification": None})
    wrong = event(2, "CONTEXT_UPDATED", {"qualification": "must_win"}, {"qualification": "already_qualified"})
    ledger.append([first, wrong])
    with pytest.raises(ValueError, match="previous snapshot"):
        ledger.replay_episode("episode-transition")

