from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from football_agent.audit import digest
from football_agent.decision_risk_experiment import proposal
from football_agent.inquiry import InquiryWorkflow
from football_agent.inquiry_choice import InformationChoiceProtocol, validate_catalog
from football_agent.inquiry_continuation import ContinuationPolicy
from football_agent.inquiry_tools import OpenWebTools
from football_agent.inquiry_verify import verify
from football_spider.storage.raw_archive import RawArchive


NOW = datetime(2026, 9, 27, tzinfo=timezone.utc)
URLS = ["https://example.com/report_a", "https://example.com/report_b"]
CATALOG = [{"source_id": "entry-"+str(i), "url": url, "publisher": "Constructed source",
            "discovery_topics": ["Personnel reporting"], "limitations": "Entry hint only; no acquired statement."}
           for i, url in enumerate(URLS)]
TASK = {"task_id": "constructed-information-choice", "objective": "Investigate a source answer.",
        "deadline": (NOW+timedelta(hours=1)).isoformat(), "initial_facts": [], "seed_urls": URLS}


def choice(actions=("report_a", "report_b"), selected="report_b", kind="query"):
    result = proposal(list(actions), chosen=selected, kind=kind)
    selected = result["decision"]["candidate_id"]
    result["choice_review"] = {"alternatives": [{"candidate_id": c["candidate_id"],
        "information_advantage": "This source could address the declared personnel unknown.",
        "source_limitations": "No guarantee it covers the match or supplies an answer.",
        "evidence_document_ids": [],
        "not_selected_reason": None if c["candidate_id"] == selected else "Leave unqueried while the other path is examined."}
        for c in result["candidates"]], "tradeoff": "Compare the same question under this budget.",
        "missing_routes": "The catalog is not exhaustive and values are unknown."}
    return result


def answer(public):
    result = choice((), kind="stop")
    doc = public["documents"][-1]
    result["claim_updates"] = [{"claim_id": "answer", "question_id": "q1", "status": "source_claim",
        "text": "This source states that Player K starts.",
        "citations": [{"document_id": doc["document_id"], "quote": doc["text"]}],
        "evidence": {"kind": "official_lineup", "subject": "Team-A", "scope": "match", "scope_key": "fixture-M"},
        "hypothesis_links": [{"hypothesis_id": "h1", "relation": "supports"}]}]
    return result


def setup(tmp_path, responses, *, catalog=CATALOG, body=None):
    calls, seen = [], []
    class Session:
        def get(self, url, **kwargs):
            calls.append(url)
            status, text = body(url) if body else (200, "For fixture M, Team A reports that Player K will start.")
            return SimpleNamespace(content=text.encode(), status_code=status, headers={}, close=lambda: None)
        def close(self): pass
    class Tools(OpenWebTools):
        def configuration(self):
            return {**super().configuration(), "execution_mode": "constructed_transport_no_network"}
    class Provider:
        def configuration(self): return {"provider": "constructed-choice-test"}
        def decide(self, public):
            seen.append(deepcopy(public))
            response = responses[len(seen)-1]
            return {"response": response(public) if callable(response) else deepcopy(response),
                    "receipt": {"real_provider_call": False}}
    tools = Tools(RawArchive(tmp_path/"evidence"), ["example.com"], session_factory=Session, clock=lambda: NOW)
    workflow = InquiryWorkflow(tmp_path/"runs", Provider(), tools, clock=lambda: NOW,
        protocol=InformationChoiceProtocol(), continuation=ContinuationPolicy(), source_catalog=catalog)
    return workflow, calls, seen


def test_two_routes_choose_one_without_acquiring_or_labelling_the_other(tmp_path):
    workflow, calls, seen = setup(tmp_path, [choice(), answer])
    run = workflow.run(workflow.start(TASK, max_requests=1, max_llm_calls=2)["run_id"])
    assert calls == [URLS[1]] and run["state"] == "STOPPED"
    comparison = run["decisions"][0]["hypothesis_assessment"]["information_choice"]
    assert comparison["distinct_prepared_actions"] == comparison["count_feasible_candidates"] == 2
    assert comparison["remaining_after_proposal"] == {"source_requests": 1, "llm_calls": 1}
    assert all(row["current_return"] == "UNOBSERVED_AT_CHOICE" and row["expected_gain"] is None
               and row["selection_probability"] is None for row in comparison["candidate_routes"])
    assert comparison["candidate_routes"][0]["catalog_entry_ids"] == ["entry-0"]
    assert seen[0]["claims"] == seen[0]["documents"] == []
    assert len(run["receipts"]) == len(run["query_reviews"]) == 1
    assert verify(tmp_path, run["run_id"])["decisions_verified"] == 2
    report = tmp_path/"report.md"
    workflow.export(run["run_id"], report)
    assert "候选信息路径比较" in report.read_text(encoding="utf-8")


def test_failed_first_route_can_change_selection_without_refuting_the_hypothesis(tmp_path):
    def reconsider(public):
        response = choice(("report_a",), selected="report_a")
        response["choice_review"]["alternatives"][0]["evidence_document_ids"] = [public["documents"][-1]["document_id"]]
        return response
    workflow, calls, seen = setup(tmp_path, [choice(), reconsider, answer],
        body=lambda url: (503, "Temporary failure") if url == URLS[1] else (200, "Player K starts in fixture M."))
    run = workflow.run(workflow.start(TASK, max_requests=2, max_llm_calls=3)["run_id"])
    assert calls == [URLS[1], URLS[0]] and len(run["claims"]) == 1
    assert seen[1]["claims"] == [] and seen[1]["documents"][0]["status"] == "FAILED"
    assert [r["status"] for r in run["receipts"]] == ["FAILED", "OK"]
    verify(tmp_path, run["run_id"])


def test_catalog_binding_uses_executed_url_identity_across_redaction(tmp_path):
    response = choice()
    response["candidates"][1]["url"] += "#reference"
    workflow, calls, _ = setup(tmp_path, [response, answer])
    run = workflow.run(workflow.start(TASK, max_requests=1, max_llm_calls=2)["run_id"])
    assert calls == [URLS[1]]
    rows = run["decisions"][0]["hypothesis_assessment"]["information_choice"]["candidate_routes"]
    assert rows[1]["catalog_entry_ids"] == ["entry-1"]
    verify(tmp_path, run["run_id"])


def test_search_and_read_costs_are_derived_at_the_same_post_proposal_budget(tmp_path):
    response = choice()
    response["candidates"][0].update(tool="search_web", query="site:example.com source answer", url=None)
    workflow, calls, _ = setup(tmp_path, [response, answer])
    run = workflow.run(workflow.start(TASK, max_requests=1, max_llm_calls=2)["run_id"])
    comparison = run["decisions"][0]["hypothesis_assessment"]["information_choice"]
    rows = comparison["candidate_routes"]
    assert comparison["count_feasible_candidates"] == 1
    assert rows[0]["minimum_counts"] == {"source_requests_including_current": 2, "llm_calls_after_current": 2}
    assert not rows[0]["within_count_budget"] and rows[1]["within_count_budget"]
    assert rows[0]["expected_total_cost"] is None and calls == [URLS[1]]
    verify(tmp_path, run["run_id"])


@pytest.mark.parametrize("actions", [(), ("report_a",)])
def test_zero_or_one_route_is_valid_without_a_candidate_quota(tmp_path, actions):
    response = choice(actions, kind="defer")
    workflow, calls, _ = setup(tmp_path, [response])
    run = workflow.run(workflow.start(TASK)["run_id"])
    assert run["state"] == "DEFERRED" and not calls
    assert len(run["decisions"][0]["candidates"]) == len(actions)
    verify(tmp_path, run["run_id"])


@pytest.mark.parametrize("mutation", ["missing", "repeated", "unknown_document", "unselected_null", "selected_reason", "same_action"])
def test_invalid_comparisons_are_rejected_before_acquisition(tmp_path, mutation):
    response = choice()
    rows = response["choice_review"]["alternatives"]
    if mutation == "missing": rows.pop()
    elif mutation == "repeated": rows[1] = deepcopy(rows[0])
    elif mutation == "unknown_document": rows[0]["evidence_document_ids"] = ["unacquired-page"]
    elif mutation == "unselected_null": rows[0]["not_selected_reason"] = None
    elif mutation == "selected_reason": rows[1]["not_selected_reason"] = "Not selected."
    else: response["candidates"][1]["url"] = response["candidates"][0]["url"]
    workflow, calls, _ = setup(tmp_path, [response])
    run = workflow.run(workflow.start(TASK)["run_id"])
    assert run["reason"] == "INVALID_INQUIRY_RESPONSE" and not calls and not run["decisions"]


@pytest.mark.parametrize("mutation", ["unknown_host", "duplicate_id", "duplicate_url"])
def test_catalog_metadata_does_not_expand_the_source_scope(mutation):
    catalog = deepcopy(CATALOG)
    if mutation == "unknown_host": catalog[0]["url"] = "https://elsewhere.example/"
    elif mutation == "duplicate_id": catalog[1]["source_id"] = catalog[0]["source_id"]
    else: catalog[1]["url"] = catalog[0]["url"]
    with pytest.raises(ValueError): validate_catalog(catalog, ["example.com"])


def test_catalog_entry_must_be_a_seed_and_requires_continuation(tmp_path):
    workflow, _, _ = setup(tmp_path, [])
    with pytest.raises(ValueError, match="seed"):
        workflow.start({**TASK, "seed_urls": URLS[:1]})
    with pytest.raises(ValueError, match="continuation"):
        InquiryWorkflow(tmp_path, None, None, protocol=InformationChoiceProtocol())


@pytest.mark.parametrize("field,value", [("within_count_budget", False), ("minimum_counts", {}), ("current_return", "FAILED")])
def test_rehashed_comparison_still_binds_computed_costs_and_unobserved_return(tmp_path, field, value):
    workflow, _, _ = setup(tmp_path, [choice(), answer])
    run = workflow.run(workflow.start(TASK, max_requests=1, max_llm_calls=2)["run_id"])
    for event in run["events"]:
        decisions = event["snapshot"]["decisions"]
        if decisions:
            decisions[0]["hypothesis_assessment"]["information_choice"]["candidate_routes"][0][field] = value
        if event["type"] == "inquiry_decision" and event["data"]["candidate_id"] == "report_b":
            event["data"]["hypothesis_assessment"]["information_choice"]["candidate_routes"][0][field] = value
    previous = None
    for event in run["events"]:
        event["previous_hash"] = previous
        event["event_hash"] = digest({k:v for k,v in event.items() if k != "event_hash"})
        previous = event["event_hash"]
    run.update(deepcopy(run["events"][-1]["snapshot"]))
    workflow.store.save(run)
    with pytest.raises(ValueError): verify(tmp_path, run["run_id"])
