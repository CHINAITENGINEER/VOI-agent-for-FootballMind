"""Archived synthetic I/O checks data eligibility, not football performance."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from football_agent.inquiry_tools import OpenWebTools
from football_agent.learned_voi_runtime import _save, digest
from football_research.data_adapters.open_text_episodes import (
    build_open_text_episodes, document_role, response_for_model, training_records,
)
from football_spider.storage.raw_archive import RawArchive


@pytest.fixture
def archived(tmp_path):
    folder = tmp_path / "original"
    archive = RawArchive(folder / "source-archive")
    clock = datetime(2026, 9, 1, 12, 0, 1, tzinfo=timezone.utc)
    tools = OpenWebTools(archive, ["club.example"], extraction_version=3, clock=lambda: clock)
    action = {"action_id": "open-a", "information_key": "page-a", "source": "open_web",
              "tool": "read_url", "url": "https://club.example/news/current-update"}
    prepared = {"tool": "read_url", "source": "open_page", "endpoint": action["url"], "params": {}}
    obs = archive.save("open_page", action["url"], {},
                       b"<html><p>The squad's preparations continue before Saturday.</p></html>",
                       status=200, observed_at=clock,
                       context={"requested_at": "2026-09-01T12:00:00Z"})
    document = tools.project(obs, prepared)
    receipt = {"status": "OK", "document": document,
               "receipt": {"observation": obs.model_dump(mode="json"), "prepared": prepared}}
    public = {"decision_at": "2026-09-01T12:00:00Z", "documents": [], "attempts": [],
              "prediction": {"probabilities": [.5, .3, .2], "model_version": "frozen-model"}}
    cycle = {"step": 0, "selected_action": action, "generated_actions": [action],
             "public_before": public, "public_digest": digest(public),
             "prediction_before": [.5, .3, .2], "selection_basis": "logged_uniform",
             "exploration_record": {"selected_p": 1., "chosen_id": "open-a", "probabilities": {"open-a": 1.}}}
    config = {"initial_state": {"match_id": "m1", "prediction_cutoff": "2026-09-02T12:00:00Z",
                                "base": {"kickoff": "2026-09-03T12:00:00Z"}, "information_mode": "operational"},
              "supervisor_metadata": {"transport": tools.configuration()}}
    run = {"schema": "learned-voi-open-execution-v2", "run_id": "synthetic-fixture",
           "configuration_digest": digest(config), "initial_probabilities": [.5, .3, .2],
           "cycles": [cycle]}
    label = {"match_id": "m1", "kickoff": "2026-09-03T12:00:00Z", "status": "final",
             "score_period": "regulation", "home_goals": 1, "away_goals": 0,
             "available_at": "2026-09-03T15:00:00Z", "observed_at": "2026-09-03T15:00:00Z",
             "source": "independent-result-fixture", "source_reference": "fixture-result:1",
             "input_manifest": ["result-artifact-fixture"], "revision_id": "result-revision-1",
             "information_mode": "operational"}
    proof = {"model_version": "frozen-model", "model_sha256": "a" * 64,
             "training_as_of": "2026-08-01T00:00:00Z", "training_match_ids": ["previous-match"],
             "probability_basis": "frozen_pre_match"}

    def write():
        run["configuration_digest"] = digest(config)
        for row in run["cycles"]:
            row["public_digest"] = digest(row["public_before"])
            _save(folder / f"cycle-{row['step']:03}.json", row)
        _save(folder / "configuration.json", config)
        _save(folder / "receipt-000.json", receipt)
        _save(folder / "run.json", run)
    write()
    return folder, run, config, receipt, label, proof, write


def build(archived, **kwargs):
    folder, _, _, _, label, proof, _ = archived
    return build_open_text_episodes([folder], as_of="2026-09-04T00:00:00Z",
                                   settlements=kwargs.pop("settlements", [label]),
                                   prior_bindings=kwargs.pop("prior_bindings", {"frozen-model": proof}), **kwargs)


def test_qualified_record_has_separate_label_and_fixed_initial_prior(archived):
    _, run, _, _, _, _, write = archived
    run["cycles"][0]["prediction_before"] = [.6, .2, .2]
    run["cycles"][0]["public_before"]["prediction"]["probabilities"] = [.6, .2, .2]
    write()
    result = build(archived)
    assert result["audit"]["numeric_voi_training_eligible"] == 1
    row = result["episodes"][0]
    assert row["prior_probabilities"] == [.5, .3, .2]
    assert row["decision_probabilities"] == [.6, .2, .2]
    assert row["label"]["outcome"] == 0 and "outcome" not in row["response"]
    assert training_records(result)[0]["result_available_at"] == "2026-09-03T15:00:00+00:00"


def test_missing_labels_and_training_provenance_produce_no_fitted_rows(archived):
    result = build(archived, settlements=[], prior_bindings={})
    assert result["audit"]["response_observation_eligible"] == 1
    assert training_records(result) == []
    assert set(result["episodes"][0]["training_exclusion_reasons"]) == {
        "independent_settlement_missing", "forward_prior_training_binding_missing"}


@pytest.mark.parametrize("mutation,reason", [
    (lambda label, proof: proof["training_match_ids"].append("m1"), "forward_prior_training_binding_invalid"),
    (lambda label, proof: proof.update(training_as_of="2026-09-04T00:00:00Z"), "forward_prior_training_binding_invalid"),
    (lambda label, proof: label.update(available_at="2026-09-05T00:00:00Z"), "independent_settlement_unavailable_or_invalid"),
    (lambda label, proof: label.update(score_period="extra_time"), "independent_settlement_unavailable_or_invalid"),
])
def test_label_and_forward_training_barriers(archived, mutation, reason):
    mutation(archived[4], archived[5])
    result = build(archived)
    assert training_records(result) == []
    assert reason in result["episodes"][0]["training_exclusion_reasons"]


def test_label_conflicts_do_not_choose_the_latest_score(archived):
    other = deepcopy(archived[4])
    other.update(home_goals=0, away_goals=1, revision_id="different-result")
    result = build(archived, settlements=[archived[4], other])
    assert "independent_settlement_conflict" in result["episodes"][0]["training_exclusion_reasons"]


def test_receipt_text_cannot_diverge_from_raw_archive(archived):
    archived[3]["document"]["text"] = "Invented injury revelation."
    archived[6]()
    result = build(archived)
    assert result["episodes"] == []
    assert result["audit"]["rejected"][0]["reason"] == "document_does_not_reproduce_raw_archive"


def test_history_does_not_read_final_documents_or_labels(archived):
    folder, run, _, receipt, _, _, write = archived
    run["documents"] = [dict(receipt["document"], text="Future secret final outcome")]
    run["outcome"] = 2
    write()
    assert build(archived)["episodes"][0]["history"] == []
    assert build(archived)["episodes"][0]["label"]["outcome"] == 0


def test_absent_selection_probability_is_not_invented(archived):
    archived[1]["cycles"][0]["exploration_record"] = None
    archived[6]()
    row = build(archived)["episodes"][0]
    assert row["numeric_voi_training_eligible"]
    assert row["selection_probability"] is None and not row["off_policy_eligible"]


def test_response_after_cutoff_remains_auditable_but_not_training_eligible(archived):
    archived[2]["initial_state"]["prediction_cutoff"] = "2026-09-01T12:00:00Z"
    archived[6]()
    row = build(archived)["episodes"][0]
    assert not row["response_observation_eligible"]
    assert "response_not_operationally_available_before_cutoff" in row["training_exclusion_reasons"]


@pytest.mark.parametrize("case,reason", [
    ("late_at_return", "response_not_available_at_actual_return"),
    ("missing_clock", "actual_return_clock_missing"),
    ("rejected_body", "successful_response_not_admitted"),
    ("missing_admission", "successful_response_not_admitted"),
])
def test_text_runtime_rejected_body_never_becomes_training_when_audited_later(archived, case, reason):
    _, run, config, _, _, _, write = archived
    run["schema"] = config["schema"] = "learned-voi-open-execution-v3"
    cycle = run["cycles"][0]
    cycle.update(returned_at="2026-09-01T12:00:02Z", execution_status="returned",
                 admission={"status": "admitted", "reason": "learned_text_return_verified"})
    if case == "late_at_return":
        cycle["returned_at"] = "2026-09-01T12:00:00.500000Z"
        cycle["admission"] = {"status": "not_admitted", "reason": "after_actual_return_clock"}
    elif case == "missing_clock":
        cycle.pop("returned_at")
    elif case == "rejected_body":
        cycle["admission"] = {"status": "not_admitted", "reason": "document_not_usable"}
    else:
        cycle.pop("admission")
    write()
    result = build(archived)
    row = result["episodes"][0]
    assert reason in row["training_exclusion_reasons"]
    assert not row["response_observation_eligible"]
    assert not row["numeric_voi_training_eligible"]
    assert row["response"]["text"]  # Keep the rejected body for audit only.
    assert training_records(result) == []


def test_admitted_text_at_actual_return_remains_training_eligible(archived):
    _, run, config, _, _, _, write = archived
    run["schema"] = config["schema"] = "learned-voi-open-execution-v3"
    run["cycles"][0].update(returned_at="2026-09-01T12:00:02Z", execution_status="returned",
                            admission={"status": "admitted", "reason": "learned_text_return_verified"})
    write()
    assert len(training_records(build(archived))) == 1


@pytest.mark.parametrize("mode,provider_status", [
    ("http_failure", "FAILED"), ("empty", "NO_DATA"), ("invalid_content", "NO_DATA"),
])
def test_unadmitted_failure_or_empty_return_is_a_neutral_training_observation(archived, mode, provider_status):
    folder, run, config, receipt, _, _, write = archived
    run["schema"] = config["schema"] = "learned-voi-open-execution-v3"
    cycle = run["cycles"][0]
    cycle.update(returned_at="2026-09-01T12:00:02Z", execution_status="returned",
                 admission={"status": "not_admitted", "reason": "provider_"+provider_status.lower()})
    observation_time = datetime(2026, 9, 1, 12, 0, 1, tzinfo=timezone.utc)
    archive = RawArchive(folder / "source-archive")
    tools = OpenWebTools(archive, ["club.example"], extraction_version=3, clock=lambda: observation_time)
    prepared = deepcopy(receipt["receipt"]["prepared"])
    if mode == "invalid_content":
        prepared["tool"] = "search_web"
        cycle["selected_action"]["tool"] = "search_web"
        cycle["selected_action"]["query"] = "match preview"
    body = b"<html>not an RSS response</html>" if mode == "invalid_content" else b""
    observation = archive.save("open_page", prepared["endpoint"], {}, body,
                               status=500 if mode == "http_failure" else 200,
                               observed_at=observation_time,
                               context={"requested_at": "2026-09-01T12:00:00Z"})
    receipt.update(status=provider_status, document=tools.project(observation, prepared),
                   receipt={"observation": observation.model_dump(mode="json"), "prepared": prepared})
    write()
    result = build(archived)
    assert result["audit"]["numeric_voi_training_eligible"] == 1
    record = training_records(result)[0]
    assert record["response"]["status"] == provider_status
    from football_research.acquisition.text_returns import _response
    assert _response(record["response"])[0] == ("failure" if provider_status == "FAILED" else "no_data")


def test_later_successful_body_cannot_hide_an_unadmitted_history_return(archived):
    folder, run, config, receipt, _, _, write = archived
    run["schema"] = config["schema"] = "learned-voi-open-execution-v3"
    first = run["cycles"][0]
    first.update(returned_at="2026-09-01T12:00:00.500000Z", execution_status="returned",
                 admission={"status": "not_admitted", "reason": "after_actual_return_clock"})
    second = deepcopy(first)
    second.update(step=1, returned_at="2026-09-01T12:00:04Z",
                  admission={"status": "admitted", "reason": "learned_text_return_verified"})
    action = deepcopy(first["selected_action"])
    action.update(action_id="open-b", information_key="page-b", url="https://club.example/news/second-update")
    second.update(selected_action=action, generated_actions=[action], exploration_record=None)
    second["public_before"].update(decision_at="2026-09-01T12:00:02Z", documents=[],
        attempts=[{"action_id": "open-a", "status": "OK", "admission": "not_admitted"}])
    observation_time = datetime(2026, 9, 1, 12, 0, 3, tzinfo=timezone.utc)
    archive = RawArchive(folder / "source-archive")
    tools = OpenWebTools(archive, ["club.example"], extraction_version=3, clock=lambda: observation_time)
    prepared = {"tool": "read_url", "source": "open_page", "endpoint": action["url"], "params": {}}
    observation = archive.save("open_page", action["url"], {}, b"<html>Another squad update.</html>",
                               status=200, observed_at=observation_time)
    _save(folder / "receipt-001.json", {"status": "OK", "document": tools.project(observation, prepared),
        "receipt": {"observation": observation.model_dump(mode="json"), "prepared": prepared}})
    run["cycles"].append(second)
    write()
    result = build(archived)
    assert result["audit"]["episodes"] == 2
    later = result["episodes"][1]
    assert later["response_observation_eligible"]
    assert not later["numeric_voi_training_eligible"]
    assert "history_response_kind_unknown" in later["training_exclusion_reasons"]
    assert "history_response_not_operationally_eligible" in later["training_exclusion_reasons"]
    assert later["history"][0]["response"]["text"] == ""
    assert training_records(result) == []


def test_structural_page_roles_do_not_use_news_keywords_as_value_labels():
    def role(url, **extra):
        return document_role({"url": url, "text": "Injury tactics lineup", "status": "OK", **extra})
    assert role("https://club.test/en?article=26856") == "index_page"
    assert role("https://club.test/en/news/squad-update") == "article_candidate_not_semantically_verified"
    assert role("https://search.test/?q=a", kind="discovery") == "search_results"
    assert role("https://club.test/mysterious") == "unknown_page_role"
    assert response_for_model({"status": "OK", "text": "A verified tool article.",
                               "kind": "article", "outcome": 2}) == {
        "status": "OK", "text": "A verified tool article.", "kind": "article"}
    assert response_for_model({"status": "OK", "text": "injury lineup", "kind": "page",
                               "url": "https://club.test/unknown"})["kind"] == "unknown"


def test_structured_article_requires_an_actual_anchored_body_not_event_metadata():
    body = "The manager discussed the squad before the match."
    document = {"status": "OK", "kind": "page", "url": "https://club.test/match/42",
                "text": body, "structured_news": {"version": "structured-news-discovery-v1",
                    "role": "article", "segments": [{"start": 0, "end": len(body), "text": body}],
                    "identity_metadata": [{"@type": "SportsEvent", "name": "Home vs Away"}]}}
    assert document_role(document) == "article_candidate_not_semantically_verified"
    assert response_for_model(document)["kind"] == "article"
    document["structured_news"]["segments"][0]["text"] = "Not present in the acquired body"
    assert document_role(document) == "unknown_page_role"
    document["structured_news"].update(role="unknown", segments=[])
    assert document_role(document) == "unknown_page_role"
    assert response_for_model(document)["kind"] == "unknown"


def test_unrecognised_search_transport_cannot_fall_back_to_legacy_projection(archived):
    archived[2]["supervisor_metadata"]["transport"]["search_provider"] = "unknown-search-provider"
    archived[6]()
    result = build(archived)
    assert result["episodes"] == []
    assert result["audit"]["rejected"][0]["reason"] == "unsupported_recorded_search_provider"


@pytest.mark.parametrize("article", [True, False])
def test_recorded_news_projector_reproduces_body_and_keeps_event_metadata_separate(archived, article):
    from football_agent.news_discovery import NewsDiscoveryTools
    folder, run, config, receipt, _, _, write = archived
    run["schema"] = config["schema"] = "learned-voi-open-execution-v3"
    cycle = run["cycles"][0]
    cycle.update(returned_at="2026-09-01T12:00:02Z", execution_status="returned",
                 admission={"status": "admitted", "reason": "learned_text_return_verified"})
    action = cycle["selected_action"]
    action["url"] = "https://club.example/match/42"
    clock = datetime(2026, 9, 1, 12, 0, 1, tzinfo=timezone.utc)
    archive = RawArchive(folder / "source-archive")
    tools = NewsDiscoveryTools(archive, ["club.example"], provider="bing_news_rss", clock=lambda: clock)
    config["supervisor_metadata"]["transport"] = tools.configuration()
    event = {"@type": "SportsEvent", "name": "Home vs Away", "startDate": "2026-09-03T12:00:00Z"}
    graph = [event]
    if article:
        graph.append({"@type": "NewsArticle", "datePublished": "2026-09-01T10:00:00Z",
                      "articleBody": "The manager reported that a player might miss the next match."})
    body = ('<html><p>Fixture centre.</p><script type="application/ld+json">'
            + json.dumps({"@graph": graph}) + '</script></html>').encode()
    prepared = tools.prepare(action, {"task": {"seed_urls": [action["url"]]}, "documents": []})
    observation = archive.save(prepared["source"], prepared["endpoint"], prepared["params"],
                               body, status=200, observed_at=clock)
    document = tools.project(observation, prepared)
    receipt.update(status="OK", document=document,
        receipt={"observation": observation.model_dump(mode="json"), "prepared": prepared})
    write()
    result = build(archived)
    assert result["audit"]["rejected"] == []
    row = result["episodes"][0]
    assert row["lineage"]["raw_response_reprojected"]
    assert row["numeric_voi_training_eligible"] is article
    assert row["response"]["kind"] == ("article" if article else "unknown")
    if not article:
        assert "response_kind_unknown" in row["training_exclusion_reasons"]
        assert training_records(result) == []
