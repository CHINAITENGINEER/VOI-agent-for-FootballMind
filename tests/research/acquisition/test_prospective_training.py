"""Training gateway boundaries; fixtures are constructed, never effect data."""
from copy import deepcopy
import json
import socket

import pytest

from football_research.data_adapters import prospective_training as module
from football_research.data_adapters.open_text_episodes import empirical_risk_records, _digest
from test_open_text_episodes import archived
from test_text_risk_records import _two_steps


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("Offline training must not use network"))


def _rows(count, *, outcomes=(0, 1, 2)):
    rows = [{"match_id": "provider-"+str(i), "outcome": outcomes[i % len(outcomes)],
             "history": [{"action": {"url": "https://club.example/old"}}]} for i in range(count)]
    identities = {row["match_id"]: {"source_event_id": "official-"+str(i),
        "representative_match_id": row["match_id"]} for i, row in enumerate(rows)}
    return rows, identities


@pytest.mark.parametrize("count,outcomes,ready", [
    (0, (0, 1, 2), False), (11, (0, 1, 2), False),
    (12, (0, 1), False), (12, (0, 1, 2), True),
])
def test_fixed_global_gate_needs_physical_count_and_each_outcome(count, outcomes, ready):
    rows, mapping = _rows(count, outcomes=outcomes)
    fitted, gate = module._canonical_records(rows, mapping)
    assert gate["ready_to_fit"] is ready
    assert gate["qualified_physical_matches"] == count
    assert sum(gate["class_counts"].values()) == count
    assert len(fitted) == count
    assert "not_conditional_support" in gate["gate_scope"]


def test_aliases_duplicate_runs_and_versions_do_not_inflate_independent_matches():
    rows, mapping = _rows(2, outcomes=(0,))
    mapping["provider-1"] = {"source_event_id": "official-0", "representative_match_id": "provider-0"}
    records = rows*20
    before = deepcopy(records)
    fitted, gate = module._canonical_records(records, mapping)
    assert gate["qualified_physical_matches"] == 1
    assert gate["class_counts"] == {"0": 1, "1": 0, "2": 0}
    assert {row["match_id"] for row in fitted} == {"provider-0"}
    assert not gate["ready_to_fit"]
    fitted[0]["history"][0]["action"]["url"] = "changed"
    assert records == before


def test_cross_provider_conflicting_label_for_same_fixture_is_rejected():
    rows, mapping = _rows(2)
    mapping["provider-1"] = {"source_event_id": "official-0", "representative_match_id": "provider-0"}
    with pytest.raises(ValueError, match="same physical fixture"):
        module._canonical_records(rows, mapping)


@pytest.mark.parametrize("first_kind", ["article", "failure", "index"])
def test_all_actual_neutral_and_article_responses_survive_complete_chains(archived, first_kind):
    dataset = _two_steps(archived, first_kind)
    before = deepcopy(dataset)
    run_id = dataset["episodes"][0]["run_id"]
    records, audit = module._complete_run_records(dataset, [{"run_id": run_id, "selected_web_steps": [0, 1]}])
    assert records == empirical_risk_records(dataset)["records"]
    assert len(records) == 2 and len(records[1]["history"]) == 1
    assert audit["complete_chain_excluded_runs"] == []
    assert dataset == before


def test_missing_earlier_web_episode_cannot_make_later_history_appear_complete(archived):
    dataset = _two_steps(archived)
    # Reproduce the old gateway's partial-read behavior without modifying any
    # archived run: rejected step zero, then a falsely shortened empty history.
    later = deepcopy(dataset["episodes"][1])
    later["history"] = []
    dataset["episodes"] = [later]
    dataset["audit"]["episodes_sha256"] = _digest(dataset["episodes"])
    assert len(empirical_risk_records(dataset)["records"]) == 1
    records, audit = module._complete_run_records(dataset, [{"run_id": later["run_id"], "selected_web_steps": [0, 1]}])
    assert records == []
    assert audit["complete_chain_excluded_runs"] == [{"run_id": later["run_id"],
        "reason": "incomplete_actual_web_episode_chain", "expected_steps": [0, 1], "reconstructed_steps": [1]}]


def test_unclassified_success_is_not_imputed_as_zero_gain(archived):
    dataset = _two_steps(archived)
    later = dataset["episodes"][1]
    later["response"]["kind"] = "unknown"
    later["numeric_voi_training_eligible"] = False
    later["training_exclusion_reasons"] = ["response_kind_unknown"]
    dataset["audit"]["episodes_sha256"] = _digest(dataset["episodes"])
    records, audit = module._complete_run_records(dataset, [{"run_id": later["run_id"], "selected_web_steps": [0, 1]}])
    assert len(records) == 1
    assert audit["exclusions"][0]["gateway_exclusion_reasons"] == ["response_kind_unknown"]
    assert audit["complete_chain_excluded_runs"] == []


def test_insufficient_data_writes_diagnosis_and_never_calls_fit(archived, tmp_path, monkeypatch):
    dataset = _two_steps(archived, "failure")
    row = dataset["episodes"][0]
    runs = [{"folder": "constructed-boundary-fixture", "run_id": row["run_id"], "selected_web_steps": [0, 1]}]
    identities = {row["match_id"]: {"original_match_id": row["match_id"],
        "representative_match_id": row["match_id"], "source_event_id": "official-fixture", "purpose": "f_train"}}
    monkeypatch.setattr(module, "_registered_runs", lambda *a, **k: (runs, {}, identities, []))
    monkeypatch.setattr(module, "join_settlements", lambda *a, **k: {"dataset": dataset})
    monkeypatch.setattr(module.TextReturnModel, "fit", lambda *a, **k: pytest.fail("An insufficient sample must not fit"))
    output = tmp_path/"not-ready"
    report = module.frozen_f_from_runs([tmp_path/"original"], [], None, output, as_of=dataset["audit"]["as_of"])
    assert report["status"] == "not_ready" and report["model"] is None
    assert report["qualified_physical_matches"] == 1 and report["qualified_records"] == 2
    assert report["class_counts"] == {"0": 1, "1": 0, "2": 0}
    assert report["production_ready"] is False and not report["model_training_performed"]
    assert report["response_statuses"] == {"FAILED": 1, "OK": 1}
    assert not (output/"frozen-text-model.json").exists()
    assert json.loads((output/"original-records.json").read_text(encoding="utf-8")) == empirical_risk_records(dataset)["records"]
    saved = (output/"training-report.json").read_bytes()
    with pytest.raises(FileExistsError, match="immutable"):
        module.frozen_f_from_runs([tmp_path/"original"], [], None, output)
    assert (output/"training-report.json").read_bytes() == saved


@pytest.mark.parametrize("extra", [["--records", "external.json"], ["--min-matches", "1"], ["--as-of", "2020-01-01T00:00:00Z"]])
def test_cli_has_no_arbitrary_record_import_threshold_override_or_backdated_clock(tmp_path, extra):
    with pytest.raises(SystemExit) as raised:
        module.main(["fit", "--run", "original", "--previous-plan", "plan.json", "--output", str(tmp_path/"out")]+extra)
    assert raised.value.code == 2 and not (tmp_path/"out").exists()


def test_unknown_model_choice_and_non_folder_inputs_fail_before_any_write(tmp_path):
    with pytest.raises(ValueError, match="model kind"):
        module.frozen_f_from_runs([tmp_path], [], None, tmp_path/"bad", model_kind="auto-tune")
    with pytest.raises(ValueError, match="never arbitrary training records"):
        module.frozen_f_from_runs({"records": []}, [], None, tmp_path/"bad")
    assert not (tmp_path/"bad").exists()
