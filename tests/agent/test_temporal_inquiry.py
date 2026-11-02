from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from football_agent.inquiry import InquiryWorkflow
from football_agent.inquiry_tools import OpenWebTools
from football_agent.temporal_inquiry import TemporalInquiryGate
from football_spider.storage.raw_archive import RawArchive


NOW = datetime(2026, 9, 23, 9, tzinfo=timezone.utc)


TASK = {"task_id": "temporal-workflow", "objective": "Check a time-bound source before reporting.",
        "deadline": (NOW + timedelta(hours=1)).isoformat(),
        "initial_facts": ["Fixture identity is supplied."],
        "seed_urls": ["https://example.com/future"]}


def answer(url="https://example.com/future", kind="query"):
    return {"questions": [{"question_id": "q1", "question": "Can the source answer the question?",
                            "decision_relevance": "Timing changes the available evidence.",
                            "answer_could_change": "The report may be deferred.", "based_on_documents": []}],
            "claim_updates": [],
            "candidates": [{"candidate_id": "source", "question_id": "q1", "tool": "read_url",
                            "url": url, "query": None, "rationale": "Read the registered route.",
                            "answer_could_change": "The answer may change."}] if kind == "query" else [],
            "decision": {"kind": kind, "candidate_id": "source" if kind == "query" else None,
                         "reason": "Temporal route choice." if kind == "query" else "No eligible route remains.",
                         "revisit_at": None}}


class Provider:
    def __init__(self, choices):
        self.choices = choices

    def configuration(self):
        return {"provider": "constructed", "configured": True}

    def decide(self, public):
        choice = self.choices.pop(0)
        return {"response": deepcopy(choice), "receipt": {"real_provider_call": False, "constructed": True}}


class Session:
    def __init__(self, calls):
        self.calls = calls

    def get(self, url, **kwargs):
        self.calls.append(url)
        return SimpleNamespace(status_code=200, content=b"<html><p>Time-bound source</p></html>",
                               headers={"content-type": "text/html"}, close=lambda: None)

    def close(self):
        pass


def gate(*, available="2026-09-23T09:15:00Z", cutoff="2026-09-23T09:30:00Z"):
    return TemporalInquiryGate(
        prediction_cutoff=cutoff, kickoff="2026-09-23T10:30:00Z", allowed_hosts=["example.com"],
        entries=[{"source_id": "source-1", "url": "https://example.com/future", "publisher": "Official",
                  "information_role": "match_context", "menu_note": "Frozen route.",
                  "document_scope": "pre_match_structured", "observed_at": "2026-09-23T09:10:00Z",
                  "available_at": available, "http_status": 200, "error_class": None}],
    )


def workflow(tmp_path, temporal_gate, choices, calls):
    tools = OpenWebTools(RawArchive(tmp_path / "raw"), ["example.com"],
                         clock=lambda: NOW,
                         session_factory=lambda: Session(calls))
    return InquiryWorkflow(tmp_path / "runs", Provider(choices), tools, clock=lambda: NOW,
                           temporal_gate=temporal_gate)


def test_gate_publishes_one_eligible_route_and_rejects_late_route(tmp_path):
    late = gate(available="2026-09-23T09:45:00Z")
    public = late.public(now=NOW)
    assert public["entries"][0]["availability_status"] == "EXCLUDED_AFTER_CUTOFF"
    calls = []
    app = workflow(tmp_path, late, [answer()], calls)
    run = app.run(app.start(TASK, max_requests=1, max_llm_calls=1)["run_id"])
    assert run["state"] == "DEFERRED"
    assert run["reason"] == "INVALID_INQUIRY_RESPONSE"
    assert calls == []
    assert run["budget"]["http_attempts"] == 0


def test_same_agent_route_becomes_executable_under_later_cutoff(tmp_path):
    calls_late, calls_early = [], []
    late_gate = gate(available="2026-09-23T09:15:00Z", cutoff="2026-09-23T09:00:00Z")
    late_app = workflow(tmp_path / "late", late_gate, [answer(), answer(kind="defer")], calls_late)
    late_run = late_app.run(late_app.start(TASK, max_requests=1, max_llm_calls=2)["run_id"])
    assert late_run["reason"] == "INVALID_INQUIRY_RESPONSE" and calls_late == []

    early_gate = gate(available="2026-09-23T09:15:00Z", cutoff="2026-09-23T09:30:00Z")
    early_app = workflow(tmp_path / "early", early_gate, [answer(), answer(kind="defer")], calls_early)
    early_run = early_app.run(early_app.start(TASK, max_requests=1, max_llm_calls=2)["run_id"])
    assert calls_early == ["https://example.com/future"]
    assert early_run["budget"]["http_attempts"] == 1


def test_unknown_search_route_is_rejected_in_historical_mode(tmp_path):
    search = answer(kind="query")
    search["candidates"][0].update({"tool": "search_web", "url": None, "query": "fixture context"})
    calls = []
    app = workflow(tmp_path, gate(), [search], calls)
    run = app.run(app.start(TASK, max_requests=1, max_llm_calls=1)["run_id"])
    assert run["state"] == "DEFERRED" and run["reason"] == "INVALID_INQUIRY_RESPONSE"
    assert calls == []


def test_gate_configuration_is_stable_for_resume_and_exposes_no_body(tmp_path):
    calls = []
    app = workflow(tmp_path, gate(), [answer(), answer(kind="defer")], calls)
    run = app.start(TASK, max_requests=1, max_llm_calls=2)
    public = app.public(run)
    assert public["temporal_gate"]["entries"][0]["body_hidden"] is True
    assert "text" not in public["temporal_gate"]["entries"][0]
    assert app.get(run["run_id"])["configuration"]["temporal_gate"]["version"] == "temporal-inquiry-gate-v1"


def test_gate_rejects_invalid_time_order():
    with pytest.raises(ValueError, match="precedes"):
        gate(available="2026-09-23T09:00:00Z")

