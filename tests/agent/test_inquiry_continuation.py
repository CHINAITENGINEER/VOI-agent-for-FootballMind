from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from football_agent.audit import atomic_write, digest
from football_agent.decision_risk_experiment import proposal
from football_agent.inquiry import InquiryWorkflow
from football_agent.inquiry_continuation import ContinuationPolicy
from football_agent.inquiry_evidence_state import EvidenceStateProtocol
from football_agent.inquiry_tools import OpenWebTools
from football_agent.inquiry_verify import verify
from football_spider.storage.raw_archive import RawArchive


NOW = datetime(2026, 9, 27, tzinfo=timezone.utc)
URL = "https://example.com/report_a"
TASK = {"task_id": "constructed-continuation", "objective": "Check the source statement and preserve unknowns.",
        "deadline": (NOW+timedelta(hours=1)).isoformat(), "initial_facts": [], "seed_urls": [URL]}
RSS = '<rss><channel><item><title>Official report</title><link>https://example.com/report_a</link></item></channel></rss>'


def reply(kind="query", *, search=False, revisit=None):
    response = proposal(["report_a"] if kind == "query" else [], kind=kind)
    response["decision"]["revisit_at"] = revisit
    if search:
        response["candidates"][0].update(tool="search_web", url=None, query="source lineup report")
    return response


class Provider:
    def __init__(self, responses):
        self.responses, self.seen = responses, []
    def configuration(self):
        return {"provider": "constructed-continuation-test", "configured": True}
    def decide(self, public):
        self.seen.append(deepcopy(public))
        item = self.responses[len(self.seen)-1]
        return {"response": item(public) if callable(item) else deepcopy(item), "receipt": {"real_provider_call": False}}


def setup(tmp_path, responses, *, body=None, policy=True, protocol=None):
    stamp, calls = [NOW], []
    class Session:
        def get(self, url, **kwargs):
            calls.append(url)
            payload = RSS if "bing.com" in url else body(len(calls)) if callable(body) else body or "<main>Source statement.</main>"
            status, payload = payload if isinstance(payload, tuple) else (200, payload)
            return SimpleNamespace(content=payload.encode(), status_code=status, headers={}, close=lambda: None)
        def close(self):
            pass
    class Constructed(OpenWebTools):
        def configuration(self):
            return {**super().configuration(), "execution_mode": "constructed_transport_no_network"}
    tools = Constructed(RawArchive(tmp_path/"evidence"), ["example.com"], session_factory=Session, clock=lambda: stamp[0])
    provider = Provider(responses)
    workflow = InquiryWorkflow(tmp_path/"runs", provider, tools, protocol=protocol or EvidenceStateProtocol(),
        clock=lambda: stamp[0], continuation=ContinuationPolicy() if policy else None)
    return workflow, provider, calls, stamp


def extracted(public, kind="stop"):
    response = reply(kind)
    doc = public["documents"][-1]
    response["claim_updates"] = [{"claim_id": "source-answer", "question_id": "q1", "status": "source_claim",
        "text": "The acquired source contains this statement; external truth remains unverified.",
        "citations": [{"document_id": doc["document_id"], "quote": doc["text"]}],
        "evidence": {"kind": "official_lineup", "subject": "Team-A", "scope": "match", "scope_key": "fixture-M"},
        "hypothesis_links": [{"hypothesis_id": "h1", "relation": "supports"}]}]
    return response


@pytest.mark.parametrize("llm_calls,requests", [(1, 0), (2, 1)])
def test_page_read_requires_capacity_to_process_the_return(tmp_path, llm_calls, requests):
    workflow, provider, calls, _ = setup(tmp_path, [reply(), extracted])
    run = workflow.run(workflow.start(TASK, max_requests=1, max_llm_calls=llm_calls)["run_id"])
    assert len(calls) == run["budget"]["http_attempts"] == requests
    assert run["decisions"][0]["continuation_check"]["admitted"] == bool(requests)
    assert provider.seen[0]["continuation"]["llm_calls_after_this_proposal"] == llm_calls-1
    if requests:
        assert run["state"] == "STOPPED" and run["claims"][0]["citations"]
        assert len(run["query_reviews"]) == 1
    else:
        assert run["reason"] == "RETURN_PROCESSING_BUDGET_INSUFFICIENT"
        assert run["decisions"][0]["proposal_before_admission"]["kind"] == "query"
    assert verify(tmp_path, run["run_id"])["continuation_admissions_replayed"] == len(provider.seen)


@pytest.mark.parametrize("llm_calls,http_calls,allowed", [(2, 2, False), (3, 1, False), (3, 2, True)])
def test_fact_search_budget_covers_discovery_read_and_answer(tmp_path, llm_calls, http_calls, allowed):
    workflow, provider, calls, _ = setup(tmp_path, [reply(search=True), reply(), extracted])
    run = workflow.run(workflow.start(TASK, max_requests=http_calls, max_llm_calls=llm_calls)["run_id"])
    assert len(calls) == (2 if allowed else 0)
    assert run["state"] == ("STOPPED" if allowed else "DEFERRED")
    if allowed:
        assert provider.seen[1]["continuation"]["awaiting_discovered_page_selection"]
        assert run["documents"][0]["kind"] == "discovery" and run["claims"][0]["citations"][0]["document_id"] == run["documents"][1]["document_id"]
    verify(tmp_path, run["run_id"])


def test_last_reflection_keeps_extracted_evidence_before_deferring(tmp_path):
    workflow, _, calls, _ = setup(tmp_path, [reply(), lambda public: extracted(public, "reflect")])
    run = workflow.run(workflow.start(TASK, max_llm_calls=2)["run_id"])
    assert len(calls) == 1 and len(run["claims"]) == 1
    assert run["reason"] == "CONTINUATION_REASONING_BUDGET_INSUFFICIENT"
    assert run["decisions"][-1]["expected_gain"] is None
    verify(tmp_path, run["run_id"])


def test_discovery_reflection_does_not_consume_another_source_request(tmp_path):
    def reflect(public):
        response = reply("reflect")
        response["questions"].append({"question_id": "q3", "question": "Does the discovered report settle the source question?",
            "decision_relevance": "The discovered page may settle the report.", "answer_could_change": "A quoted source answer.",
            "based_on_documents": [public["documents"][-1]["document_id"]]})
        response["hypotheses"].append({**response["hypotheses"][0], "hypothesis_id": "h3", "question_id": "q3",
                                       "statement": "The discovered report answers the personnel question."})
        return response
    workflow, _, calls, _ = setup(tmp_path, [reply(search=True), reflect])
    run = workflow.run(workflow.start(TASK, max_requests=2, max_llm_calls=3)["run_id"])
    assert len(calls) == 1 and run["budget"]["llm_calls"] == 2
    assert run["decisions"][-1]["continuation_check"]["minimum_required"]["llm_calls_after_current"] == 2
    assert run["reason"] == "CONTINUATION_REASONING_BUDGET_INSUFFICIENT"
    verify(tmp_path, run["run_id"])


def waiting_run(tmp_path, *, duplicate=False, refresh_fails=False):
    due = NOW+timedelta(minutes=10)
    responses = [reply(), reply("wait", revisit=due.isoformat()), reply(), reply() if duplicate else extracted]
    workflow, provider, calls, stamp = setup(tmp_path, responses,
        body=lambda n: (403, "Unavailable") if n > 1 and refresh_fails else f"<main>Source version {n}</main>")
    run = workflow.run(workflow.start(TASK, max_requests=3, max_llm_calls=5 if duplicate else 4)["run_id"])
    return workflow, provider, calls, stamp, run, due


def test_due_wait_resumes_without_refill_and_allows_one_paid_refresh(tmp_path):
    workflow, provider, calls, stamp, waiting, due = waiting_run(tmp_path)
    assert waiting["state"] == "WAITING"
    assert workflow.step(waiting["run_id"]) == waiting
    assert workflow.resume_wait(waiting["run_id"]) == waiting
    stamp[0] = due
    resumed = workflow.resume_wait(waiting["run_id"])
    assert resumed["state"] == "READY" and resumed["budget"] == waiting["budget"]
    assert resumed["documents"] == waiting["documents"] and len(calls) == 1
    run = workflow.run(waiting["run_id"])
    assert run["state"] == "STOPPED" and len(calls) == 2
    assert run["receipts"][0]["execution_key"] == run["receipts"][1]["execution_key"]
    assert run["documents"][0]["observed_at"] != run["documents"][1]["observed_at"]
    assert "version 2" in run["claims"][0]["citations"][0]["quote"]
    assert verify(tmp_path, run["run_id"])["due_waits_resumed"] == 1


@pytest.mark.parametrize("refresh_fails", [False, True])
def test_same_query_cannot_refresh_twice_in_one_wait_window(tmp_path, refresh_fails):
    workflow, _, calls, stamp, waiting, due = waiting_run(tmp_path, duplicate=True, refresh_fails=refresh_fails)
    stamp[0] = due
    workflow.resume_wait(waiting["run_id"])
    run = workflow.run(waiting["run_id"])
    assert len(calls) == 2 and run["reason"] == "DUPLICATE_QUERY_REQUIRES_NEW_EVIDENCE_OR_WAIT"
    assert run["documents"][-1]["status"] == ("FAILED" if refresh_fails else "OK")
    verify(tmp_path, run["run_id"])


def test_wait_timestamp_is_checked_after_the_model_return(tmp_path):
    workflow, provider, calls, stamp = setup(tmp_path, [])
    def slow(public):
        stamp[0] += timedelta(minutes=11)
        return reply("wait", revisit=(NOW+timedelta(minutes=5)).isoformat())
    provider.responses.append(slow)
    run = workflow.run(workflow.start(TASK, max_llm_calls=3)["run_id"])
    assert run["reason"] == "WAIT_TIME_ELAPSED_DURING_REASONING" and not calls
    verify(tmp_path, run["run_id"])


def test_wait_past_task_deadline_stops_without_a_model_or_source_call(tmp_path):
    workflow, provider, calls, stamp, waiting, _ = waiting_run(tmp_path)
    stamp[0] = NOW+timedelta(hours=1)
    run = workflow.resume_wait(waiting["run_id"])
    assert run["reason"] == "TASK_DEADLINE_REACHED" and len(provider.seen) == 2 and len(calls) == 1
    verify(tmp_path, run["run_id"])


@pytest.mark.parametrize("cross_at", ["advance", "intent"])
def test_deadline_crossing_during_processing_or_journaling_does_not_send_request(tmp_path, cross_at):
    workflow, _, calls, stamp = setup(tmp_path, [reply()])
    if cross_at == "advance":
        original = workflow.protocol.advance
        def advance(*args):
            result = original(*args)
            stamp[0] += timedelta(hours=2)
            return result
        workflow.protocol.advance = advance
    else:
        original = workflow._event
        def event(run, kind, data):
            original(run, kind, data)
            if kind == "inquiry_acquisition_intent":
                stamp[0] += timedelta(hours=2)
        workflow._event = event
    run = workflow.run(workflow.start(TASK, max_llm_calls=2)["run_id"])
    assert not calls and run["state"] == "STOPPED" and run["budget"]["reserved_requests"] == 0
    assert run["pending"] is None
    verify(tmp_path, run["run_id"])
    if cross_at == "intent":
        from football_agent.return_dataset import export_run
        reference = {"root": str(tmp_path), "run_id": run["run_id"], "head_hash": run["events"][-1]["event_hash"],
                     "origin": "constructed", "cluster_id": "constructed-cancellation", "cluster_basis": "Same constructed deadline case."}
        rows, info = export_run(reference, as_of=run["events"][-1]["at"])
        assert rows == [] and info["unexecuted_cancelled_requests_excluded"] == 1
        before_cancel, _ = export_run(reference, as_of=NOW)
        assert len(before_cancel) == 1 and before_cancel[0]["status"] == "CENSORED_AT_EXPORT_CUTOFF"


def rehash(workflow, run):
    previous = None
    for event in run["events"]:
        event["previous_hash"] = previous
        event["event_hash"] = digest({k:v for k,v in event.items() if k != "event_hash"})
        previous = event["event_hash"]
    run.update(deepcopy(run["events"][-1]["snapshot"]))
    workflow.store.save(run)


@pytest.mark.parametrize("mutation", ["budget_limit", "early_resume", "unknown_event", "task_deadline", "fake_pause_ready", "receipt_prefix"])
def test_rehashed_trace_still_rejects_budget_or_wait_changes(tmp_path, mutation):
    workflow, _, _, stamp, waiting, due = waiting_run(tmp_path)
    stamp[0] = due
    run = workflow.resume_wait(waiting["run_id"])
    if mutation == "budget_limit":
        for event in run["events"][1:]:
            event["snapshot"]["budget"]["max_llm_calls"] += 1
    elif mutation == "early_resume":
        run["events"][-1]["data"]["resumed_at"] = (due-timedelta(seconds=1)).isoformat()
    elif mutation == "unknown_event":
        run["events"][-1]["type"] = "unverified_resume"
    elif mutation == "task_deadline":
        for event in run["events"][1:]:
            event["snapshot"]["task"]["deadline"] = (NOW+timedelta(hours=2)).isoformat()
    elif mutation == "fake_pause_ready":
        run["events"][-1]["type"] = "inquiry_paused_or_stopped"
        run["events"][-1]["data"] = {"state": "READY", "reason": "WAIT_WINDOW_REACHED", "value_stop_proven": False}
    else:
        run = workflow.run(waiting["run_id"])
        intent = [e for e in run["events"] if e["type"] == "inquiry_acquisition_intent"][-1]
        intent["snapshot"]["receipts"] = []
    rehash(workflow, run)
    with pytest.raises(ValueError):
        verify(tmp_path, run["run_id"])


def test_opt_in_guard_does_not_relabel_existing_unreserved_runs(tmp_path):
    workflow, _, calls, _ = setup(tmp_path, [reply()], policy=False)
    run = workflow.run(workflow.start(TASK, max_llm_calls=1)["run_id"])
    assert len(calls) == 1 and run["reason"] == "LLM_BUDGET_EXHAUSTED"
    assert "continuation_check" not in run["decisions"][0]
    verify(tmp_path, run["run_id"])


def test_continuation_policy_does_not_apply_to_numeric_terminal_selectors(tmp_path):
    with pytest.raises(ValueError, match="deterministic scoped selectors"):
        InquiryWorkflow(tmp_path, None, None, continuation=ContinuationPolicy(), selector=object())
