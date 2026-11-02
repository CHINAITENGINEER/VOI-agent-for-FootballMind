"""Interventions on prospective plans, not measured match-prediction utility."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from jsonschema import ValidationError

from football_agent.inquiry import InquiryWorkflow
from football_agent.inquiry_returns import OUTCOMES, ReturnForecastProtocol, classify_return, summarize_model
from football_agent.inquiry_tools import OpenWebTools
from football_agent.inquiry_verify import verify
from football_spider.storage.raw_archive import RawArchive


NOW = datetime(2026, 9, 26, 15, tzinfo=timezone.utc)
TASK = {"task_id": "constructed-return-intervention", "objective": "Choose a continuation from the actual return.",
        "deadline": (NOW + timedelta(hours=1)).isoformat(), "initial_facts": [],
        "seed_urls": ["https://example.com/statement", "https://example.com/confirmation"]}
H = {"hypothesis_id": "h", "question_id": "q", "statement": "Team A intends to rotate in match one.",
     "hypothesis_type": "rotation_intent", "subject": "Team-A", "scope": "match", "scope_key": "match-one",
     "required_evidence": ["coach_statement"]}
SUPPORT = "Our coach intends to rotate in match one."
CHALLENGE = "Our coach intends to keep the same team in match one."


def model(*, challenge_query=True):
    return {"branches": [{"outcome": outcome,
        "missing_after": [{"hypothesis_id": "h", "types": [] if outcome == "source_support" else ["coach_statement"]}],
        "next_action": {"kind": "query" if outcome == "source_challenge" and challenge_query else "defer",
                        "target_hypothesis_ids": ["h"] if outcome == "source_challenge" and challenge_query else [],
                        "evidence_types": ["coach_statement"] if outcome == "source_challenge" and challenge_query else [],
                        "reason": "Seek independent confirmation of opposition; otherwise defer."}}
        for outcome in OUTCOMES]}


def answer(kind="query", *, endpoint="statement", claims=None):
    return {"questions": [{"question_id": "q", "question": "Has the coach stated a rotation intention?",
                "decision_relevance": "Identify whether another source needs checking.",
                "answer_could_change": "Choose confirmation or defer.", "based_on_documents": []}],
            "hypotheses": [deepcopy(H)], "claim_updates": claims or [], "context_bindings": [],
            "candidates": [{"candidate_id": "c", "question_id": "q", "tool": "read_url",
                "url": "https://example.com/" + endpoint, "query": None,
                "rationale": "Acquire a coach statement.", "answer_could_change": "Confirm opposition or defer.",
                "inquiry_intent": {"hypothesis_ids": ["h"], "purpose": "resolve", "expected_evidence": ["coach_statement"],
                    "subject": "Team-A", "scope": "match", "scope_key": "match-one",
                    "supports_change": "Defer further acquisition.", "challenges_change": "Seek another source.",
                    "no_answer_change": "Retain the gap."},
                "return_model": model(challenge_query=endpoint == "statement")}] if kind == "query" else [],
            "decision": {"kind": kind, "candidate_id": "c" if kind == "query" else None,
                         "reason": "Respond to the acquired evidence.", "revisit_at": None}}


def claim(document, relation):
    return {"claim_id": "claim-one", "question_id": "q", "text": document["text"], "status": "source_claim",
            "citations": [{"document_id": document["document_id"], "quote": document["text"]}],
            "evidence": {"kind": "coach_statement", "subject": "Team-A", "scope": "match", "scope_key": "match-one"},
            "hypothesis_links": [{"hypothesis_id": "h", "relation": relation}]}


def build(root, body, *, deviate=False, open_plan=False, wrong_gap=False, support_query=False):
    calls = []
    class Session:
        def get(self, url, **kwargs):
            calls.append(url)
            return SimpleNamespace(content=("<main>"+body+"</main>").encode(), status_code=200,
                                   headers={"content-type": "text/html"}, close=lambda: None)
        def close(self):
            pass
    class Provider:
        def __init__(self): self.seen = []
        def configuration(self): return {"provider": "constructed", "configured": True}
        def decide(self, state):
            self.seen.append(deepcopy(state))
            if len(self.seen) == 1:
                response = answer()
                if open_plan:
                    response["candidates"][0]["return_model"]["branches"][0]["next_action"]["kind"] = "reassess"
                if wrong_gap:
                    response["candidates"][0]["return_model"]["branches"][0]["missing_after"][0]["types"] = ["coach_statement"]
                if support_query:
                    response["candidates"][0]["return_model"]["branches"][0]["next_action"].update(
                        kind="query", target_hypothesis_ids=["h"], evidence_types=["coach_statement"])
            elif len(self.seen) == 2:
                support = SUPPORT in state["documents"][-1]["text"]
                response = answer("query" if deviate or not support else "defer", endpoint="confirmation",
                                  claims=[claim(state["documents"][-1], "supports" if support else "challenges")])
            else:
                response = answer("defer")
            return {"response": response, "receipt": {"real_provider_call": False}}
    provider = Provider()
    tools = OpenWebTools(RawArchive(root/"evidence"), ["example.com"], clock=lambda: NOW, session_factory=Session)
    workflow = InquiryWorkflow(root/"runs", provider, tools, clock=lambda: NOW, protocol=ReturnForecastProtocol())
    return workflow, provider, calls


@pytest.mark.parametrize("body,outcome,requests", [(SUPPORT, "source_support", 1), (CHALLENGE, "source_challenge", 2)])
def test_return_changes_continuation_without_rewriting_the_prior_plan(tmp_path, body, outcome, requests):
    workflow, provider, calls = build(tmp_path, body)
    run = workflow.run(workflow.start(TASK, max_requests=2, max_llm_calls=3)["run_id"])
    first = run["return_reviews"][0]
    assert first["outcome"] == outcome and first["status"] == "matched_declared_plan"
    assert first["gap_forecast_match"] and first["continuation_match"]
    assert len(calls) == requests and provider.seen[0]["documents"] == []
    intent = next(e for e in run["events"] if e["type"] == "inquiry_acquisition_intent")
    assert intent["snapshot"]["documents"] == []
    assert intent["data"]["inquiry_contract"]["return_model"] == model()
    assert first["expected_gain"] is None and first["realized_decision_utility"] is None
    proof = verify(tmp_path, run["run_id"])
    assert proof["raw_observations_verified"] == requests and proof["new_network_requests"] == 0
    assert len(calls) == requests


def test_changed_plan_is_retained_as_a_deviation_and_not_forced_to_match(tmp_path):
    workflow, _, calls = build(tmp_path, SUPPORT, deviate=True)
    run = workflow.run(workflow.start(TASK, max_requests=2, max_llm_calls=3)["run_id"])
    first = run["return_reviews"][0]
    assert first["gap_forecast_match"] and first["continuation_match"] is False
    assert first["status"] == "deviation_from_declared_plan"
    assert first["predicted_continuation"]["kind"] == "defer" and first["proposed_continuation"]["kind"] == "query"
    assert len(calls) == 2 and verify(tmp_path, run["run_id"])["decisions_verified"] == 3


def test_reassess_is_open_and_is_not_counted_as_a_successful_action_prediction(tmp_path):
    workflow, _, _ = build(tmp_path, SUPPORT, open_plan=True)
    run = workflow.run(workflow.start(TASK)["run_id"])
    result = run["return_reviews"][0]
    assert result["status"] == "open_continuation_not_scored" and result["continuation_match"] is None


def test_wrong_gap_forecast_is_visible_even_if_the_next_action_matches(tmp_path):
    workflow, _, _ = build(tmp_path, SUPPORT, wrong_gap=True)
    run = workflow.run(workflow.start(TASK)["run_id"])
    result = run["return_reviews"][0]
    assert result["gap_forecast_match"] is False and result["continuation_match"] is True
    assert result["predicted_missing_after"][0]["types"] == ["coach_statement"]
    assert result["observed_missing_after"][0]["types"] == []
    assert result["status"] == "deviation_from_declared_plan"
    assert verify(tmp_path, run["run_id"])["decisions_verified"] == 2


def test_matching_query_proposal_does_not_imply_it_executed_past_the_budget(tmp_path):
    workflow, _, calls = build(tmp_path, SUPPORT, deviate=True, support_query=True)
    run = workflow.run(workflow.start(TASK, max_requests=1, max_llm_calls=3)["run_id"])
    result = run["return_reviews"][0]
    assert result["continuation_match"] and result["proposed_continuation"]["kind"] == "query"
    assert result["continuation_scope"] == "next_proposal_not_execution_or_optimality"
    assert len(calls) == 1 and run["state"] == "STOPPED" and run["reason"] == "REQUEST_BUDGET_EXHAUSTED"
    assert verify(tmp_path, run["run_id"])["raw_observations_verified"] == 1


@pytest.mark.parametrize("status,outcome", [("NO_DATA", "no_answer"), ("FAILED", "failure"),
    ("INVALID_CONTENT", "failure"), ("LATE_RETURN", "failure")])
def test_absence_and_failure_are_not_opposition(status, outcome):
    receipt = {"status": status, "inquiry_contract": {"known_urls_before": []}}
    assert classify_return(receipt, {"outcome": "source_challenge"}, {"links": []}) == (outcome, [])


def test_new_links_are_discovery_without_a_claim_of_relevance():
    receipt = {"status": "OK", "inquiry_contract": {"known_urls_before": ["https://example.com/old"]}}
    doc = {"links": [{"url": "https://example.com/old"}, {"url": "https://example.com/navigation"}]}
    assert classify_return(receipt, {"outcome": "no_usable_extracted_evidence"}, doc) == ("discovery", ["https://example.com/navigation"])
    receipt["inquiry_contract"]["known_urls_before"].append("https://example.com/navigation")
    assert classify_return(receipt, {"outcome": "no_usable_extracted_evidence"}, doc) == ("no_answer", [])


@pytest.mark.parametrize("mistake", ["duplicate_branch", "foreign_target", "unknown_requirement", "numeric_gain"])
def test_invalid_or_invented_forecasts_are_rejected(tmp_path, mistake):
    workflow, _, _ = build(tmp_path, SUPPORT)
    state = workflow.public(workflow.start(TASK))
    response = answer()
    forecast = response["candidates"][0]["return_model"]
    if mistake == "duplicate_branch": forecast["branches"][-1]["outcome"] = "no_answer"
    if mistake == "foreign_target": forecast["branches"][0]["missing_after"][0]["hypothesis_id"] = "future"
    if mistake == "unknown_requirement": forecast["branches"][0]["missing_after"][0]["types"] = ["standings"]
    if mistake == "numeric_gain": forecast["expected_gain"] = .9
    with pytest.raises((ValueError, ValidationError)):
        workflow.protocol.validate(response, state)


def test_invariance_ignores_failure_plans_and_does_not_assert_zero_value():
    forecast = model(challenge_query=False)
    forecast["branches"][-1]["next_action"].update(kind="wait")
    summary = summarize_model(forecast)
    assert not summary["continuation_sensitive"] and summary["expected_gain"] is None
    assert summary["successful_gap_variants"] == 2
    for branch in forecast["branches"]:
        branch["missing_after"][0]["types"] = ["standings", "competition_rules"]
    forecast["branches"][0]["missing_after"][0]["types"].reverse()
    assert summarize_model(forecast)["successful_gap_variants"] == 1
