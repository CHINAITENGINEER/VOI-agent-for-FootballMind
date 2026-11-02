"""Constructed interventions verify evidence-dependent inquiry, not real efficacy."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

import pytest

from football_agent.inquiry import InquiryWorkflow
from football_agent.inquiry_hypotheses import HypothesisProtocol, project
from football_agent.inquiry_tools import OpenWebTools
from football_agent.inquiry_verify import verify
from football_spider.storage.raw_archive import RawArchive


NOW = datetime(2026, 9, 26, 15, tzinfo=timezone.utc)
TASK = {"task_id": "constructed-match", "objective": "Investigate rotation, retaining scope uncertainty.",
        "deadline": (NOW + timedelta(hours=1)).isoformat(), "initial_facts": [],
        "seed_urls": ["https://example.com/squad"]}
H = {"hypothesis_id": "rotation", "question_id": "q", "statement": "The team rotated its starting lineup.",
     "hypothesis_type": "actual_rotation", "subject": "team-a", "scope": "match", "scope_key": "match-a",
     "required_evidence": ["official_lineup", "lineup_comparison"]}


def response(kind="query", url="https://example.com/squad", claims=None):
    return {"questions": [{"question_id": "q", "question": "Is rotation established for this match?",
            "decision_relevance": "Distinguish squad context from actual rotation.",
            "answer_could_change": "Decide whether to seek a final lineup.", "based_on_documents": []}],
            "hypotheses": [deepcopy(H)], "claim_updates": claims or [],
            "candidates": [{"candidate_id": "c", "question_id": "q", "tool": "read_url", "url": url, "query": None,
                "rationale": "The roster can identify a more specific personnel question.",
                "answer_could_change": "Choose a lineup check or defer.",
                "inquiry_intent": {"hypothesis_ids": ["rotation"], "purpose": "context",
                    "expected_evidence": ["roster"], "subject": "team-a", "scope": "campaign", "scope_key": "window-a",
                    "supports_change": "Seek the final team.", "challenges_change": "Check the named unavailable players.",
                    "no_answer_change": "Seek another source or defer."}}] if kind == "query" else [],
            "decision": {"kind": kind, "candidate_id": "c" if kind == "query" else None,
                         "reason": "Use returned evidence to choose the next question.", "revisit_at": None}}


def claim(identity="roster", doc="doc-a", *, kind="roster", scope="campaign", key="window-a", relation="supports"):
    return {"claim_id": identity, "question_id": "q", "text": "The source lists players for the window.",
            "status": "source_claim", "citations": [{"document_id": doc, "quote": "Squad for the window"}],
            "evidence": {"kind": kind, "subject": "team-a", "scope": scope, "scope_key": key},
            "hypothesis_links": [{"hypothesis_id": "rotation", "relation": relation}]}


def test_roster_context_cannot_fill_match_rotation_requirements():
    state = project([H], [claim()])[0]
    assert state["status"] == "context_only"
    assert set(state["missing_support_types"]) == {"official_lineup", "lineup_comparison"}
    assert state["context"][0]["reason"] == "scope_mismatch"
    assert state["forecast_utility"] is None


@pytest.mark.parametrize("mismatch", ["subject", "scope_key", "kind", "inference"])
def test_a_quote_does_not_override_applicability_or_inference(mismatch):
    item = claim(kind="official_lineup", scope="match", key="match-a")
    if mismatch == "inference":
        item["status"] = "hypothesis"
    else:
        item["evidence"][mismatch] = {"subject": "team-b", "scope_key": "other-match", "kind": "roster"}[mismatch]
    assert project([H], [item])[0]["status"] == "context_only"


def test_lineup_needs_comparison_and_opposing_source_remains_visible():
    lineup = claim("lineup", kind="official_lineup", scope="match", key="match-a")
    comparison = claim("comparison", kind="lineup_comparison", scope="match", key="match-a")
    assert project([H], [lineup])[0]["status"] == "partial_source_support"
    assert project([H], [lineup, comparison])[0]["status"] == "complete_source_support"
    opposite = claim("opposite", doc="doc-b", kind="lineup_comparison", scope="match", key="match-a", relation="challenges")
    result = project([H], [lineup, comparison, opposite])[0]
    assert result["status"] == "mixed_source_signals" and result["semantic_verification"] == "unverified"


def test_requirements_and_claims_cannot_be_rewritten_to_manufacture_resolution():
    protocol = HypothesisProtocol()
    public = {"task": TASK, "now": NOW.isoformat(), "documents": [], "questions": [], "claims": [], "hypotheses": []}
    changed = response()
    changed["hypotheses"][0]["required_evidence"] = ["roster"]
    with pytest.raises(ValueError, match="required evidence"):
        protocol.validate(changed, public)
    public["hypotheses"] = [H]
    changed = response()
    changed["hypotheses"][0]["statement"] = "A different easier hypothesis."
    with pytest.raises(ValueError, match="immutable"):
        protocol.validate(changed, public)


class Provider:
    def __init__(self, answers):
        self.answers, self.seen = answers, []

    def configuration(self):
        return {"provider": "constructed", "configured": True}

    def decide(self, public):
        self.seen.append(deepcopy(public))
        answer = self.answers[len(self.seen)-1]
        return {"response": answer(public) if callable(answer) else deepcopy(answer),
                "receipt": {"real_provider_call": False}}


def build(tmp_path, answers, body):
    calls = []
    class Session:
        def get(self, url, **kwargs):
            calls.append(url)
            text = body if url.endswith("/squad") else "Official lineup still pending"
            return SimpleNamespace(content=text.encode(), status_code=200, headers={"content-type": "text/html"}, close=lambda: None)
        def close(self):
            pass
    provider = Provider(answers)
    tools = OpenWebTools(RawArchive(tmp_path / "evidence"), ["example.com"], clock=lambda: NOW, session_factory=Session)
    workflow = InquiryWorkflow(tmp_path / "runs", provider, tools, clock=lambda: NOW, protocol=HypothesisProtocol())
    return workflow, provider, calls


@pytest.mark.parametrize("announced", [True, False])
def test_actual_document_drives_next_query_with_frozen_intent_and_offline_replay(tmp_path, announced):
    def after_document(public):
        assert public["hypothesis_states"][0]["status"] == "unresolved"
        doc = public["documents"][0]
        if "Squad for the window" in doc["text"]:
            item = claim(doc=doc["document_id"])
            return response(url="https://example.com/lineup", claims=[item])
        return response("defer")
    body = ('<main>Squad for the window<a href="/lineup">Final team</a></main>' if announced
            else '<main>No usable personnel details available</main>')
    workflow, provider, calls = build(tmp_path, [response(), after_document, response("defer")], body)
    run = workflow.run(workflow.start(TASK, max_requests=2, max_llm_calls=3)["run_id"])
    assert run["state"] == "DEFERRED"
    assert len(calls) == (2 if announced else 1)
    assert "Squad for the window" not in json.dumps(provider.seen[0])
    first = run["query_reviews"][0]
    assert first["intent"] == run["receipts"][0]["inquiry_contract"]["intent"]
    assert first["state_before"][0]["status"] == "unresolved"
    assert first["outcome"] == ("context_only" if announced else "no_usable_extracted_evidence")
    assert first["counterfactual_gain"] is None
    assert all(d["expected_gain"] is None for d in run["decisions"])
    review = verify(tmp_path, run["run_id"])
    assert review["query_reviews_replayed"] == len(calls)
    assert review["new_network_requests"] == 0 and len(calls) == (2 if announced else 1)


def test_cross_document_synthesis_is_not_attributed_to_one_query():
    from football_agent.inquiry_hypotheses import review_query
    item = claim("synthesis", kind="official_lineup", scope="match", key="match-a")
    item["citations"].append({"document_id": "doc-b", "quote": "Another document"})
    receipt = {"request_id": "r", "document_id": "doc-b", "status": "OK", "inquiry_contract": {
        "intent": response()["candidates"][0]["inquiry_intent"], "state_before": project([H], [])}}
    review = review_query(receipt, [H], [item], project([H], [item]))
    assert review["multi_document_attribution_unknown"] == ["synthesis"]
    assert review["outcome"] == "no_usable_extracted_evidence"
    assert review["state_after"][0]["status"] == "partial_source_support"


def test_self_consistent_public_text_mutation_is_detected_against_actual_receipt(tmp_path):
    from football_agent.audit import digest
    workflow, _, calls = build(tmp_path, [response(), response("defer")], "<main>Original acquired text</main>")
    run = workflow.run(workflow.start(TASK)["run_id"])
    # Rehashing a journal is not enough: the verifier must bind the public text
    # to the separately archived HTTP receipt, rather than just match IDs.
    previous = None
    for event in run["events"]:
        for doc in event["snapshot"].get("documents", []):
            doc["text"] = "An invented lineup"
        if event["type"] in {"inquiry_reasoning_reserved", "inquiry_reasoning_returned"}:
            public = workflow.store.get_blob(run["run_id"], event["data"]["input_ref"])
            if public["documents"]:
                public["documents"][0]["text"] = "An invented lineup"
                event["data"]["input_ref"] = workflow.store.put_blob(run["run_id"], public)
        event["previous_hash"] = previous
        event["event_hash"] = digest({k: v for k, v in event.items() if k != "event_hash"})
        previous = event["event_hash"]
    run["documents"] = deepcopy(run["events"][-1]["snapshot"]["documents"])
    workflow.store.save(run)
    assert workflow.store.verify(run["run_id"])["status"] == "OK"
    with pytest.raises(ValueError, match="document content differs"):
        verify(tmp_path, run["run_id"])
    assert len(calls) == 1
