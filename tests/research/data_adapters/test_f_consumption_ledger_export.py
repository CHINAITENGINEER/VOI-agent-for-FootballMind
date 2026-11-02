"""Targeted regressions for the f consumption ledger export projection.

The runtime records the explicit semantic-consumption ledger inside the
cycle's public prediction projection (and, on newer runtimes, on the action
cycle itself).  ``public_state_after`` is a state slice that usually carries no
``prediction`` at all; it must never hide a recorded projection, and a
same-cycle runtime record must remain readable.  These tests pin the projection
order, the no-merge rule and the unknown semantics without touching any
prediction value, clock, fixture binding or label gate.
"""
from __future__ import annotations

import json

import pytest

from football_research.data_adapters.prospective_stream import (
    _consumption_projection, _cycle_consumption_ledger, _projection_ledger, export_batch,
)
from tests.research.data_adapters.test_predictor_context_export import (
    BEFORE, HANDOFF, archive_context,
)
from tests.research.data_adapters.test_prospective_stream_contexts import _batch

IDS = [f"{index:064x}" for index in (1, 2, 3, 4, 5)]
LEDGER_KEYS = ("f_consumed_input_ids", "f_consumption_status", "f_consumption_basis",
               "f_modeled_read_documents")
BASIS = "prediction.used_semantic_input_ids"


def _projection(ids=None, *, modeled=None):
    projection = {"probabilities": {"home": .4, "draw": .3, "away": .3}}
    if ids is not None:
        projection["used_semantic_input_ids"] = list(ids)
    if modeled is not None:
        projection["modeled_read_documents"] = modeled
    return projection


def _cycle_ledger(ids, *, status="consumed", basis=BASIS, modeled=1, **extra):
    return {"f_consumed_input_ids": list(ids), "f_consumption_status": status,
            "f_consumption_basis": basis, "f_modeled_read_documents": modeled, **extra}


def _write_run(folder, cycle):
    (folder / "run.json").write_text(json.dumps({
        "cycles": [cycle], "terminal_selection": {"reason": "no_candidates"}}), encoding="utf-8")


def _batch_with_predictor(root, *, public_after=None, state_after=None, cycle_ledger=None,
                          before_prediction=None, silent=False):
    folder = _batch(root)
    timings = {kind: archive_context(folder, kind=kind, schema="runtime-predictor-context-v3")
               for kind in ("before_action", "after_action")}
    config = json.loads((folder / "configuration.json").read_text(encoding="utf-8"))
    config["policy_manifest"] = {"predictor_version": "f:frozen", "predictor_sha256": "a" * 64,
                                 "base_policy": config["policy_manifest"]}
    (folder / "configuration.json").write_text(json.dumps(config), encoding="utf-8")
    cycle = {"decision_at": HANDOFF, "returned_at": HANDOFF,
        "generated_actions": [{"action_id": "news", "tool": "read_url", "query": "team news"}],
        "selected_action": {"action_id": "news"}, "selection_distribution": {"news": 1.0},
        "selection_probability_basis": "conditional_on_observed_menu",
        "public_before": {"documents": []}, "before_state_timing": timings["before_action"],
        "after_state_timing": timings["after_action"], **(cycle_ledger or {})}
    if before_prediction is not None:
        cycle["public_before"]["prediction"] = before_prediction
    if public_after is not None:
        cycle["public_after"] = public_after
    if state_after is not None:
        cycle["public_state_after"] = state_after
    if not silent:
        _write_run(folder, cycle)
    return folder, cycle


def _export(fixture, root, name="export"):
    folder, _cycle = fixture
    target = root / name
    export_batch(folder.parent, target)

    def rows(relative):
        return [json.loads(line) for line in (target / relative).read_text(encoding="utf-8").splitlines()
                if line.strip()]
    return {"contexts": rows("context_snapshots.jsonl"), "attempts": rows("attempts.jsonl"),
            "discovery": rows("discovery.jsonl"), "fixtures": rows("fixtures.jsonl"),
            "derived": rows("derived/g-training-candidates.jsonl")}


def _after(result):
    return next(row for row in result["contexts"] if row["snapshot_kind"] == "after_action")


def _before(result):
    return next(row for row in result["contexts"] if row["snapshot_kind"] == "before_action")


def _selected(result):
    return next(row for row in result["attempts"] if row.get("selected_by_policy") is True)


def _ledger(row):
    return {key: row.get(key) for key in LEDGER_KEYS}


def test_public_after_projection_supplies_the_consumption_ledger(tmp_path):
    result = _export(_batch_with_predictor(tmp_path,
        public_after={"prediction": _projection(IDS, modeled=1),
                      "documents": [{"document_id": "unrelated-document-id"}]}), tmp_path)
    after, attempt = _after(result), _selected(result)
    assert _ledger(after) == {"f_consumed_input_ids": IDS, "f_consumption_status": "consumed",
                              "f_consumption_basis": BASIS, "f_modeled_read_documents": 1}
    # The attempt handoff carries the same record, including the recorded order.
    assert _ledger(attempt) == _ledger(after)
    # Document IDs are never promoted into the predictor's own input ledger.
    assert "unrelated-document-id" not in after["f_consumed_input_ids"]
    assert all(isinstance(value, str) and len(value) == 64 for value in after["f_consumed_input_ids"])


def test_state_slice_without_prediction_does_not_hide_the_recorded_projection(tmp_path):
    # This is the recorded real-run layout: ``public_state_after`` exists but
    # carries no ``prediction``; the ledger lives on ``public_after``.
    result = _export(_batch_with_predictor(tmp_path,
        public_after={"prediction": _projection(IDS, modeled=1)},
        state_after={"documents": [{"document_id": "state-only"}]}), tmp_path)
    after, attempt = _after(result), _selected(result)
    assert after["f_consumed_input_ids"] == IDS
    assert after["f_consumption_status"] == "consumed"
    assert after["f_consumption_basis"] == BASIS
    assert after["f_modeled_read_documents"] == 1
    assert _ledger(attempt) == _ledger(after)
    # The state slice still supplies only state, never a fabricated ledger.
    assert after["known_document_ids"] == ["body-1"]
    assert "state-only" not in after["f_consumed_input_ids"]


def test_cycle_level_ledger_is_exported_when_no_projection_records_one(tmp_path):
    result = _export(_batch_with_predictor(tmp_path,
        public_after={"prediction": _projection()},
        cycle_ledger=_cycle_ledger(IDS[:2], modeled=2)), tmp_path)
    after, attempt = _after(result), _selected(result)
    assert _ledger(after) == {"f_consumed_input_ids": IDS[:2], "f_consumption_status": "consumed",
                              "f_consumption_basis": BASIS, "f_modeled_read_documents": 2}
    assert _ledger(attempt) == _ledger(after)
    # The same-cycle record describes this action only; the pre-decision state
    # keeps its own (absent) record.
    assert _ledger(_before(result)) == {"f_consumed_input_ids": None,
        "f_consumption_status": "unknown", "f_consumption_basis": "not_recorded",
        "f_modeled_read_documents": None}


def test_explicit_empty_cycle_ledger_is_not_consumed(tmp_path):
    result = _export(_batch_with_predictor(tmp_path,
        public_after={"prediction": _projection()},
        cycle_ledger=_cycle_ledger([], status="not_consumed", modeled=0)), tmp_path)
    after = _after(result)
    assert _ledger(after) == {"f_consumed_input_ids": [], "f_consumption_status": "not_consumed",
                              "f_consumption_basis": BASIS, "f_modeled_read_documents": 0}
    assert _ledger(_selected(result)) == _ledger(after)


def test_conflicting_sources_keep_the_highest_priority_record_without_merging(tmp_path):
    first = _export(_batch_with_predictor(tmp_path / "case-one",
        public_after={"prediction": _projection([IDS[0]], modeled=1)},
        state_after={"prediction": _projection([IDS[1]], modeled=2)},
        cycle_ledger=_cycle_ledger([IDS[2]], modeled=3)), tmp_path / "case-one")
    after = _after(first)
    assert after["f_consumed_input_ids"] == [IDS[0]]
    assert after["f_modeled_read_documents"] == 1
    assert after["f_consumption_basis"] == BASIS
    # No silent union with the state slice or the cycle record.
    assert IDS[1] not in after["f_consumed_input_ids"]
    assert IDS[2] not in after["f_consumed_input_ids"]
    second = _export(_batch_with_predictor(tmp_path / "case-two",
        public_after={"prediction": _projection()},
        state_after={"prediction": _projection([IDS[1]], modeled=2)},
        cycle_ledger=_cycle_ledger([IDS[2]], modeled=3)), tmp_path / "case-two")
    recorded = _after(second)
    assert recorded["f_consumed_input_ids"] == [IDS[1]]
    assert recorded["f_modeled_read_documents"] == 2
    assert IDS[2] not in recorded["f_consumed_input_ids"]


def test_missing_ledger_sources_keep_unknown_semantics(tmp_path):
    result = _export(_batch_with_predictor(tmp_path,
        public_after={"prediction": _projection()}, state_after={"documents": []}), tmp_path)
    expected = {"f_consumed_input_ids": None, "f_consumption_status": "unknown",
                "f_consumption_basis": "not_recorded", "f_modeled_read_documents": None}
    assert _ledger(_after(result)) == expected
    assert _ledger(_selected(result)) == expected
    assert _ledger(_before(result)) == expected


@pytest.mark.parametrize("record", [
    {"f_consumed_input_ids": [IDS[0]]},  # status and basis missing
    {"f_consumed_input_ids": [IDS[0], IDS[0]], "f_consumption_status": "consumed",
     "f_consumption_basis": BASIS, "f_modeled_read_documents": 1},  # duplicate IDs
    {"f_consumed_input_ids": ["  "], "f_consumption_status": "consumed",
     "f_consumption_basis": BASIS, "f_modeled_read_documents": 1},  # blank ID
    {"f_consumed_input_ids": [7], "f_consumption_status": "consumed",
     "f_consumption_basis": BASIS, "f_modeled_read_documents": 1},  # non-string ID
    {"f_consumed_input_ids": IDS[0], "f_consumption_status": "consumed",
     "f_consumption_basis": BASIS, "f_modeled_read_documents": 1},  # not a list
    {"f_consumed_input_ids": [], "f_consumption_status": "consumed",
     "f_consumption_basis": BASIS, "f_modeled_read_documents": 1},  # empty but consumed
    {"f_consumed_input_ids": [IDS[0]], "f_consumption_status": "not_consumed",
     "f_consumption_basis": BASIS, "f_modeled_read_documents": 1},  # non-empty but not consumed
    {"f_consumed_input_ids": [IDS[0]], "f_consumption_status": "consumed",
     "f_modeled_read_documents": 1},  # basis missing
    {"f_consumed_input_ids": [IDS[0]], "f_consumption_status": "consumed",
     "f_consumption_basis": "not_recorded", "f_modeled_read_documents": 1},  # basis not a record
    {"f_consumed_input_ids": [IDS[0]], "f_consumption_status": "consumed",
     "f_consumption_basis": BASIS, "f_modeled_read_documents": -1},  # negative count
    {"f_consumed_input_ids": [IDS[0]], "f_consumption_status": "consumed",
     "f_consumption_basis": BASIS, "f_modeled_read_documents": True},  # boolean count
])
def test_malformed_cycle_records_never_become_consumed(tmp_path, record):
    result = _export(_batch_with_predictor(tmp_path,
        public_after={"prediction": _projection()}, cycle_ledger=record), tmp_path)
    expected = {"f_consumed_input_ids": None, "f_consumption_status": "unknown",
                "f_consumption_basis": "not_recorded", "f_modeled_read_documents": None}
    assert _ledger(_after(result)) == expected
    assert _ledger(_selected(result)) == expected


def test_ledger_projection_changes_no_predictions_clocks_bindings_or_labels(tmp_path):
    fixture = _batch_with_predictor(tmp_path, public_after={"prediction": _projection()},
                                    before_prediction={"probabilities": [.5, .3, .2]})
    folder, cycle = fixture
    without = _export(fixture, tmp_path, "without-ledger")
    cycle["public_after"]["prediction"]["used_semantic_input_ids"] = list(IDS)
    cycle["public_after"]["prediction"]["modeled_read_documents"] = 1
    _write_run(folder, cycle)
    with_ledger = _export(fixture, tmp_path, "with-ledger")
    assert _ledger(_after(without))["f_consumed_input_ids"] is None
    assert _after(with_ledger)["f_consumed_input_ids"] == IDS
    for name in ("contexts", "attempts", "discovery", "fixtures", "derived"):
        assert len(with_ledger[name]) == len(without[name])
        for changed, original in zip(with_ledger[name], without[name]):
            assert ({key: value for key, value in changed.items() if key not in LEDGER_KEYS}
                    == {key: value for key, value in original.items() if key not in LEDGER_KEYS})
    # The label row itself is unchanged: still gated, still no invented gain.
    assert [(row["label_status"], row["realized_gain"], row["net_gain"])
            for row in with_ledger["derived"]] == [
            (row["label_status"], row["realized_gain"], row["net_gain"]) for row in without["derived"]]
    assert with_ledger["derived"][0]["f_consumed_input_ids"] == IDS
    assert with_ledger["derived"][0]["f_consumption_status"] == "consumed"


def test_ledger_helpers_reject_unproven_inputs_without_repair():
    assert _projection_ledger(None) is None
    assert _projection_ledger({}) is None
    assert _projection_ledger({"used_semantic_input_ids": []})["f_consumption_status"] == "not_consumed"
    assert _projection_ledger({"projection": {"used_input_ids": [IDS[0]]}}) == {
        "f_consumed_input_ids": [IDS[0]], "f_consumption_status": "consumed",
        "f_consumption_basis": "prediction.projection.used_input_ids"}
    assert _consumption_projection(({}, {"prediction": {}})) == {}
    assert _cycle_consumption_ledger({}) is None
    assert _cycle_consumption_ledger({"f_consumed_input_ids": None}) is None
    assert _cycle_consumption_ledger({"f_consumed_input_ids": [],
        "f_consumption_status": "not_consumed", "f_consumption_basis": BASIS,
        "f_modeled_read_documents": 0})["f_consumed_input_ids"] == []
