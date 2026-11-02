"""Offline constructed raw chains; no synthetic fixture is a real-data result."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import shutil
import socket

import pytest

from football_research.acquisition.visible_state import encode_visible_state
from football_research.data_adapters.context_training import build_context_training_from_runs
from football_spider.storage.raw_archive import RawArchive
from test_prospective_training_integration import archived_training_cohort, _source_inputs, _score


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        pytest.fail("The context bridge may only read archived original bytes")
    for obj, name in ((socket, "create_connection"), (socket, "getaddrinfo"),
                      (socket.socket, "connect"), (socket.socket, "connect_ex")):
        monkeypatch.setattr(obj, name, denied)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _convert(data, output, *, runs=None, result_archive=None, previous=None):
    return build_context_training_from_runs(data["runs"] if runs is None else runs,
        [data["exposure"]] if previous is None else previous,
        data["result_archive"] if result_archive is None else result_archive,
        output, as_of=data["as_of"])


def test_raw_actor_prefixes_have_exact_after_states_and_match_level_supervision(archived_training_cohort, tmp_path):
    data, output = archived_training_cohort, tmp_path/"all"
    report = _convert(data, output)
    records, pairs = _read(output/"prefix-records.json"), _read(output/"actual-transitions.json")
    assert report["input_runs"] == 13 and report["physical_matches"] == 12
    assert report["supervised_prefixes"] == 39 and len(pairs) == 26
    assert report["readiness"]["ready_to_fit"] and report["readiness"]["physical_matches"] == 12
    assert report["readiness"]["class_counts"] == {"0": 4, "1": 4, "2": 4}
    assert not report["old_enumeration_or_unknown_role_gate_applied"]
    assert report["model"] is None and not report["model_training_performed"]
    assert not list(output.glob("*model*"))
    index = {row["prefix_id"]: row for row in records}
    original_runs = {_read(folder/"run.json")["run_id"]: _read(folder/"run.json") for folder in data["runs"]}
    for row in records:
        cycle = original_runs[row["run_id"]]["cycles"][row["prefix_index"]]
        assert row["visible_state"] == encode_visible_state(cycle["public_before"], cycle["generated_actions"])
        assert row["match_id"] == row["visible_state"]["before"]["match_id"]
        assert "outcome" not in json.dumps(row["visible_state"])
        assert "result_available_at" not in json.dumps(row["visible_state"])
    for pair in pairs:
        before, after = index[pair["before_prefix_id"]], index[pair["after_prefix_id"]]
        assert after["prefix_index"] == before["prefix_index"]+1
        assert after["physical_match_id"] == before["physical_match_id"]
        assert pair["action"]["action_id"] in before["visible_state"]["candidates"]
        assert after["visible_state"]["before"]["history"]["action_history"][-1]["action_id"] == pair["action"]["action_id"]
    alias_group = next(g for g in _read(output/"physical-groups.json") if len(g["provider_match_ids"]) == 2)
    assert len(alias_group["run_ids"]) == 2 and alias_group["prefixes"] == 6


def test_same_physical_alias_and_duplicate_input_do_not_manufacture_readiness(archived_training_cohort, tmp_path):
    data = archived_training_cohort
    report = _convert(data, tmp_path/"one", runs=[data["single"], data["single"], data["alias"]])
    assert report["input_runs"] == 2 and report["supervised_prefixes"] == 6
    assert report["physical_matches"] == report["readiness"]["physical_matches"] == 1
    assert not report["readiness"]["ready_to_fit"]
    assert report["readiness"]["class_counts"] == {"0": 1, "1": 0, "2": 0}
    assert report["model"] is None


def test_missing_raw_labels_preserve_pending_states_instead_of_inventing_supervision(archived_training_cohort, tmp_path):
    data, output = archived_training_cohort, tmp_path/"pending"
    empty = RawArchive(tmp_path/"results")
    report = _convert(data, output, runs=[data["single"]], result_archive=empty.root)
    assert report["supervised_prefixes"] == 0 and report["pending_prefixes"] == 3
    assert report["actual_transitions"] == 2 and not report["readiness"]["ready_to_fit"]
    assert _read(output/"prefix-records.json") == []
    for row in _read(output/"pending-prefixes.json"):
        assert "outcome" not in row and "result_available_at" not in row
        assert row["training_exclusion_reasons"] == ["independent_settlement_missing"]


def test_modified_cycle_cannot_be_used_as_a_shortened_valid_prefix(archived_training_cohort, tmp_path):
    data = archived_training_cohort
    folder = tmp_path/"r"
    shutil.copytree(data["single"], folder)
    path = folder/"cycle-001.json"
    cycle = _read(path)
    cycle["public_before"]["documents"].append({"document_id": "unexecuted-future-article"})
    path.write_text(json.dumps(cycle), encoding="utf-8")
    with pytest.raises(ValueError, match="rejected|snapshot|differs"):
        _convert(data, tmp_path/"bad", runs=[folder])
    assert not (tmp_path/"bad").exists()


def test_records_and_unregistered_identity_cannot_bypass_raw_gateway(archived_training_cohort, tmp_path):
    data = archived_training_cohort
    with pytest.raises(TypeError):
        build_context_training_from_runs([], [], data["result_archive"], tmp_path/"injected",
            records=[{"outcome": 0, "physical_match_id": "fake-real"}])
    with pytest.raises(ValueError, match="inventory"):
        _convert(data, tmp_path/"unregistered", runs=[data["single"]], previous=[])
    assert not (tmp_path/"injected").exists() and not (tmp_path/"unregistered").exists()


def test_raw_unknown_response_is_a_visible_mask_not_an_old_model_training_exclusion(tmp_path, monkeypatch):
    import football_agent.inquiry_tools as web_tools
    import football_agent.learned_voi_runtime as runtime
    import football_research.data_adapters.prospective_text as collector
    import test_verified_collection as fixture_module
    from football_research.data_adapters.prospective_learning import import_training_exposure
    from test_verified_collection import _collect
    now = datetime.now(timezone.utc).replace(microsecond=0)
    at = now-timedelta(days=40)
    clock = [at+timedelta(milliseconds=100)]
    class FixtureClock(datetime):
        @classmethod
        def now(cls, tz=None):
            clock[0] += timedelta(milliseconds=10)
            return clock[0].astimezone(tz) if tz is not None else clock[0].replace(tzinfo=None)
    inputs = _source_inputs(tmp_path/"s", at=at, local_id=101, event_id=2046001)
    inputs["body"] = b"<html><body><p>Unknown page navigation text must not become article content.</p></body></html>"
    with monkeypatch.context() as mp:
        for module in (web_tools, runtime, collector, fixture_module):
            mp.setattr(module, "datetime", FixtureClock)
        folder, _, _ = _collect(inputs, mp)
    archive = RawArchive(tmp_path/"results")
    _score(archive, inputs, 1)
    exposure = import_training_exposure([folder], created_at=now-timedelta(days=2))
    output = tmp_path/"context"
    report = build_context_training_from_runs([folder], [exposure], archive.root, output,
                                               as_of=now-timedelta(days=1))
    assert report["supervised_prefixes"] == 3 and report["readiness"]["physical_matches"] == 1
    dataset = _read(output/"gateway-dataset.json")["dataset"]
    assert any("response_kind_unknown" in r["training_exclusion_reasons"] for r in dataset["episodes"])
    records = _read(output/"prefix-records.json")
    assert any(d["mask"]["unknown_role"] for r in records for d in r["visible_state"]["before"]["documents"])
    assert all("Unknown page navigation" not in json.dumps(r["visible_state"]) for r in records)
