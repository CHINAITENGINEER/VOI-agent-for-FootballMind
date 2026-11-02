"""Raw fixture checks the empirical projection, not real football outcomes."""
from copy import deepcopy
from datetime import datetime, timezone

import pytest

from football_agent.inquiry_tools import OpenWebTools
from football_agent.learned_voi_runtime import _save
from football_research.data_adapters.open_text_episodes import (
    _digest, empirical_risk_records, training_records,
)
from football_spider.storage.raw_archive import RawArchive
from test_open_text_episodes import archived, build


def _two_steps(archived, first_kind="article"):
    """Build two archived executed steps through the existing gateway."""
    folder, run, _, receipt, _, _, write = archived
    first = run["cycles"][0]
    first_action = first["selected_action"]
    archive = RawArchive(folder / "source-archive")
    if first_kind != "article":
        at = datetime(2026, 9, 1, 12, 0, 1, tzinfo=timezone.utc)
        tools = OpenWebTools(archive, ["club.example"], extraction_version=3, clock=lambda: at)
        if first_kind == "index":
            first_action["url"] = "https://club.example/"
        prepared = {"tool": "read_url", "source": "open_page", "endpoint": first_action["url"], "params": {}}
        body = b'<html><a href="/news/current-update">News</a></html>' if first_kind == "index" else b""
        observation = archive.save("open_page", prepared["endpoint"], {}, body,
            status=500 if first_kind == "failure" else 200, observed_at=at,
            context={"requested_at": "2026-09-01T12:00:00Z"})
        receipt.update(status="FAILED" if first_kind == "failure" else "OK",
            document=tools.project(observation, prepared),
            receipt={"observation": observation.model_dump(mode="json"), "prepared": prepared})
    second = deepcopy(first)
    action = deepcopy(first_action)
    action.update(action_id="open-b", information_key="page-b", url="https://club.example/news/second-update")
    second.update(step=1, selected_action=action, generated_actions=[action], exploration_record=None)
    second["public_before"].update(decision_at="2026-09-01T12:00:02Z",
        documents=[deepcopy(receipt["document"])] if receipt["status"] == "OK" else [],
        attempts=[{"action_id": "open-a", "status": receipt["status"]}])
    at = datetime(2026, 9, 1, 12, 0, 3, tzinfo=timezone.utc)
    tools = OpenWebTools(archive, ["club.example"], extraction_version=3, clock=lambda: at)
    prepared = {"tool": "read_url", "source": "open_page", "endpoint": action["url"], "params": {}}
    observation = archive.save("open_page", action["url"], {}, b"<html>Another squad update.</html>",
                              status=200, observed_at=at)
    _save(folder / "receipt-001.json", {"status": "OK", "document": tools.project(observation, prepared),
        "receipt": {"observation": observation.model_dump(mode="json"), "prepared": prepared}})
    run["cycles"].append(second)
    write()
    dataset = build(archived)
    assert dataset["audit"]["numeric_voi_training_eligible"] == 2
    return dataset


def _project_mutated(dataset):
    # Mutations below simulate bad complete gateway contracts; hash remains
    # internally consistent so per-record guards, not the envelope, are tested.
    dataset["audit"]["episodes_sha256"] = _digest(dataset["episodes"])
    return empirical_risk_records(dataset)


def test_actual_same_run_history_clocks_and_home_label_zero_survive(archived):
    dataset = _two_steps(archived)
    before = deepcopy(dataset)
    legacy = training_records(dataset)
    projected = empirical_risk_records(dataset)
    assert len(projected["records"]) == 2
    second = projected["records"][1]
    assert second["outcome"] == 0
    assert second["kickoff"] == dataset["episodes"][1]["kickoff_at"]
    assert second["decision_at"] == "2026-09-01T12:00:02+00:00"
    assert second["history"][0]["decision_at"] == "2026-09-01T12:00:00+00:00"
    assert second["history"][0]["observed_at"] == "2026-09-01T12:00:01+00:00"
    assert second["history"][0]["available_at"] == "2026-09-01T12:00:01+00:00"
    assert projected["audit"]["exclusions"] == []
    assert projected["audit"]["label_scope"] == "trust_existing_raw_settlement_gateway_not_integer_only_verification"
    assert projected["audit"]["network_requests"] == 0
    assert dataset == before and training_records(dataset) == legacy


@pytest.mark.parametrize("key", ["decision_at", "observed_at", "available_at"])
def test_any_missing_actual_clock_is_not_reconstructed_from_other_steps(archived, key):
    dataset = _two_steps(archived)
    del dataset["episodes"][0][key]
    projected = _project_mutated(dataset)
    assert projected["records"] == []
    assert projected["audit"]["exclusion_counts"] == {"actual_step_clock_missing": 2}


def test_history_cannot_be_borrowed_from_other_run_even_for_same_match(archived):
    dataset = _two_steps(archived)
    dataset["episodes"][0]["run_id"] = "different-run"
    projected = _project_mutated(dataset)
    assert [row["action"]["action_id"] for row in projected["records"]] == ["open-a"]
    assert projected["audit"]["exclusion_counts"] == {"history_episode_missing_or_ambiguous_in_same_run": 1}


def test_available_before_match_cutoff_but_after_next_decision_is_excluded(archived):
    dataset = _two_steps(archived)
    prior = dataset["episodes"][0]
    prior["available_at"] = prior["response"]["available_at"] = "2026-09-01T12:00:03+00:00"
    projected = _project_mutated(dataset)
    assert len(projected["records"]) == 1
    assert projected["audit"]["exclusion_counts"] == {"history_response_unavailable_at_next_decision": 1}


@pytest.mark.parametrize("target,key,value,reason", [
    ("prior", "step", 2, "history_order_or_match_context_conflict"),
    ("prior", "match_id", "different-match", "history_order_or_match_context_conflict"),
    ("prior", "numeric_voi_training_eligible", False, "history_episode_not_training_eligible"),
    ("history_response", "text", "Not the observed body", "history_differs_from_actual_executed_prefix"),
    ("history_response", "status", "FAILED", "history_differs_from_actual_executed_prefix"),
    ("history_response", "kind", "index", "history_differs_from_actual_executed_prefix"),
])
def test_history_inherits_qualification_and_exact_executed_semantics(archived, target, key, value, reason):
    dataset = _two_steps(archived)
    row = dataset["episodes"][0] if target == "prior" else dataset["episodes"][1]["history"][0]["response"]
    row[key] = value
    projected = _project_mutated(dataset)
    assert all(row["action"]["action_id"] != "open-b" for row in projected["records"])
    assert reason in projected["audit"]["exclusion_counts"]


@pytest.mark.parametrize("first_kind,status,kind", [("failure", "FAILED", "unknown"), ("index", "OK", "index")])
def test_neutral_failure_and_index_history_keep_actual_clocks(archived, first_kind, status, kind):
    projected = empirical_risk_records(_two_steps(archived, first_kind))
    assert len(projected["records"]) == 2
    history = projected["records"][1]["history"][0]
    assert history["response"]["status"] == status
    assert history["response"]["kind"] == kind
    assert history["decision_at"] == "2026-09-01T12:00:00+00:00"
    assert history["available_at"] == "2026-09-01T12:00:01+00:00"


def test_missing_independent_label_produces_empty_record_set(archived):
    projected = empirical_risk_records(build(archived, settlements=[]))
    assert projected["records"] == []
    assert projected["audit"]["output_records"] == 0
    assert projected["audit"]["exclusions"][0]["gateway_exclusion_reasons"] == ["independent_settlement_missing"]


def test_changed_gateway_episode_payload_is_rejected(archived):
    dataset = build(archived)
    dataset["episodes"][0]["label"]["outcome"] = 2
    with pytest.raises(ValueError, match="complete unchanged gateway"):
        empirical_risk_records(dataset)
