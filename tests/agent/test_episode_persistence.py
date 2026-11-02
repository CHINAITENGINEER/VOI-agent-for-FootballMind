import hashlib
import json

import pytest

from football_agent.episode_persistence import (
    STATUS_COMPLETE, STATUS_PARTIAL, VERSION, EpisodePersistence,
)


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


S0, S1, S2 = "0" * 64, "1" * 64, "2" * 64
A0, A1 = "a" * 64, "b" * 64
SEL0, SEL1 = "c" * 64, "d" * 64
ADM0, ADM1 = "e" * 64, "f" * 64


def _row(sequence=0, *, before=None, after=None, action="lineup", candidate="lineup",
         selection="sel-0", acquisition="acq-0", action_digest=A0,
         admission_digest=ADM0, status="admitted"):
    return {
        "episode_id": "episode-1", "target_key": "match-1", "sequence": sequence,
        "action_id": action, "candidate_id": candidate,
        "selection_operation_id": selection, "selection_digest": action_digest,
        "acquisition_id": acquisition, "acquisition_digest": action_digest,
        "admission_status": status, "admission_digest": admission_digest,
        "state_before_digest": S0 if sequence == 0 else S1,
        "state_after_digest": S1 if sequence == 0 else S2,
        "prediction_before": [.5, .3, .2] if before is None else before,
        "prediction_after": [.6, .25, .15] if after is None else after,
        "prediction_cutoff": "2026-10-01T12:00:00Z",
        "captured_at": "2026-10-01T10:00:00Z",
    }


def test_unknown_prediction_is_preserved_and_reported_partial(tmp_path):
    store = EpisodePersistence(tmp_path)
    row = _row(before=None, after=None)
    row["prediction_before"] = None
    row["prediction_after"] = None
    saved = store.append(row)
    assert saved["prediction_before"] is None
    assert saved["prediction_after"] is None
    report = store.validate()
    assert report["status"] == STATUS_PARTIAL
    assert report["eligible_for_replay"] is False
    assert "record_0_prediction_before_unknown" in report["reasons"]
    assert "record_0_prediction_after_unknown" in report["reasons"]
    assert "prediction_before" not in repr([0])


def test_complete_episode_requires_no_inferred_values(tmp_path):
    store = EpisodePersistence(tmp_path)
    store.append(_row())
    second = _row(1, before=[.6, .25, .15], after=[.55, .3, .15], selection="sel-1",
                  acquisition="acq-1", action_digest=A1, admission_digest=ADM1,
                  candidate="weather", action="weather")
    store.append(second)
    # A terminal action may have no acquisition/admission binding, but still
    # requires its own immutable selection boundary.
    terminal = _row(2, before=[.55, .3, .15], after=None, selection="sel-2",
                    acquisition=None, action_digest="9" * 64, admission_digest=None,
                    candidate="STOP", action="STOP")
    terminal.pop("acquisition_digest")
    terminal.pop("admission_status")
    terminal.pop("admission_digest")
    terminal["state_after_digest"] = None
    terminal["prediction_after"] = None
    store.append(terminal)
    report = store.validate()
    assert report["status"] == STATUS_PARTIAL  # terminal posterior is explicitly unknown
    assert "record_2_prediction_after_unknown" in report["reasons"]


def test_selection_and_acquisition_records_are_append_only(tmp_path):
    store = EpisodePersistence(tmp_path)
    store.append(_row())
    with pytest.raises(ValueError, match="SELECTION_IMMUTABLE"):
        store.append(_row(1, selection="sel-0", acquisition="acq-1", action_digest=A1))
    with pytest.raises(ValueError, match="ACQUISITION_IMMUTABLE"):
        store.append(_row(1, selection="sel-1", acquisition="acq-0", action_digest=A1))


def test_tampered_digest_and_cross_action_binding_are_rejected(tmp_path):
    store = EpisodePersistence(tmp_path)
    store.append(_row())
    store.append(_row(1, before=[.6, .25, .15], selection="sel-1", acquisition="acq-1",
                      action_digest=A1, admission_digest=ADM1, candidate="weather", action="weather"))
    path = tmp_path / "episode-actions.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    envelope = json.loads(lines[1])
    envelope["record"]["selection_operation_id"] = "sel-foreign"
    path.write_text(lines[0] + "\n" + json.dumps(envelope, ensure_ascii=False, sort_keys=True,
                                                  separators=(",", ":")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="RECORD_TAMPERED"):
        store.records()


def test_future_result_and_invalid_prediction_are_rejected(tmp_path):
    store = EpisodePersistence(tmp_path)
    with pytest.raises(ValueError, match="evaluator fields"):
        store.append({**_row(), "future_result": {"score": "1-0"}})
    with pytest.raises(ValueError, match="sum to one"):
        store.append({**_row(), "prediction_before": [.4, .4, .1]})


def test_conflicting_nested_binding_cannot_be_normalized_away(tmp_path):
    store = EpisodePersistence(tmp_path)
    with pytest.raises(ValueError, match="selection binding identity conflict"):
        store.append({**_row(), "selection_binding": {"operation_id": "foreign-selection"}})


def test_snapshot_has_stable_digest_and_binding_fields(tmp_path):
    store = EpisodePersistence(tmp_path)
    store.append(_row())
    snapshot = store.snapshot()
    assert snapshot["schema_version"] == VERSION
    assert snapshot["records"][0]["selection_digest"] == A0
    assert snapshot["records"][0]["acquisition_digest"] == A0
    assert snapshot["records"][0]["admission_digest"] == ADM0
    assert len(snapshot["snapshot_digest"]) == 64


def test_compact_state_digest_alias_is_expanded_without_fabrication(tmp_path):
    row = _row()
    row.pop("state_before_digest")
    row["state_digest"] = S0
    store = EpisodePersistence(tmp_path)
    store.append(row)
    saved = store.records()[0]
    assert saved["state_digest"] == saved["state_before_digest"] == S0

