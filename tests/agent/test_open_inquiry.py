"""Behavioral checks for question-driven inquiries, with explicitly constructed IO."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from football_agent.inquiry import InquiryWorkflow
from football_agent.inquiry_contract import validate_response
from football_agent.inquiry_provider import CodexInquiryProvider
from football_agent.inquiry_tools import OpenWebTools
from football_spider.storage.raw_archive import RawArchive


NOW = datetime(2026, 9, 26, 15, tzinfo=timezone.utc)
TASK = {"task_id": "constructed-fixture", "objective": "Confirm lineup uncertainty before reporting.",
        "deadline": (NOW + timedelta(hours=1)).isoformat(), "initial_facts": ["Fixture identity supplied for this constructed test."],
        "seed_urls": ["https://example.com/preview"]}


def answer(kind="query", url="https://example.com/preview", claims=None, question="Is the lineup final?"):
    return {"questions": [{"question_id": "q1", "question": question, "decision_relevance": "Check lineup uncertainty.",
                            "answer_could_change": "A provisional list needs later confirmation.", "based_on_documents": []}],
            "claim_updates": claims or [], "candidates": ([{"candidate_id": "c1", "question_id": "q1", "tool": "read_url",
                "url": url, "query": None, "rationale": "Consult the source.", "answer_could_change": "Choose confirmation or stop."}]
                if kind == "query" else []),
            "decision": {"kind": kind, "candidate_id": "c1" if kind == "query" else None,
                         "reason": "Need source evidence." if kind == "query" else "The remaining lineup uncertainty is explicit.",
                         "revisit_at": None}}


class Provider:
    def __init__(self, choices):
        self.choices, self.seen = choices, []

    def configuration(self):
        return {"provider": "constructed_test", "configured": True}

    def decide(self, public):
        self.seen.append(deepcopy(public))
        choice = self.choices[len(self.seen) - 1]
        response = choice(public) if callable(choice) else deepcopy(choice)
        return {"response": response, "receipt": {"real_provider_call": False, "constructed": True}}


class Session:
    def __init__(self, bodies, calls):
        self.bodies, self.calls = bodies, calls

    def get(self, url, **kwargs):
        self.calls.append(url)
        status, body = self.bodies[url]
        return SimpleNamespace(content=body.encode(), status_code=status, headers={"content-type": "text/html"}, close=lambda: None)

    def close(self):
        pass


def setup(tmp_path, provider, bodies=None, clock=None):
    calls = []
    tools = OpenWebTools(RawArchive(tmp_path / "raw"), ["example.com"], clock=clock or (lambda: NOW),
                         session_factory=lambda: Session(bodies or {}, calls))
    workflow = InquiryWorkflow(tmp_path / "runs", provider, tools, clock=clock or (lambda: NOW))
    return workflow, tools, calls


def test_actual_return_changes_next_question_and_discovers_url(tmp_path):
    def after_preview(public):
        assert len(public["documents"]) == 1
        doc = public["documents"][0]
        assert "https://example.com/team" in {link["url"] for link in doc["links"]}
        claim = {"claim_id": "lineup", "question_id": "q1", "text": "The preview is provisional.", "status": "source_claim",
                 "citations": [{"document_id": doc["document_id"], "quote": "Lineup is provisional"}]}
        return answer(url="https://example.com/team", claims=[claim], question="Has the final team been announced?")

    provider = Provider([answer(), after_preview, answer("defer")])
    workflow, tools, calls = setup(tmp_path, provider, {
        "https://example.com/preview": (200, '<html><p>Lineup is provisional</p><a href="/team">Team news</a></html>'),
        "https://example.com/team": (200, "<html><p>Final team not announced.</p></html>")})
    run = workflow.start(TASK, max_requests=2, max_llm_calls=3)
    completed = workflow.run(run["run_id"])
    assert completed["state"] == "DEFERRED"
    assert calls == ["https://example.com/preview", "https://example.com/team"]
    assert completed["claims"][0]["status"] == "source_claim"
    assert completed["decisions"][0]["expected_gain"] is None
    assert completed["budget"]["http_attempts"] == 2
    assert "Lineup is provisional" not in json.dumps(provider.seen[0])
    assert "Final team not announced" not in json.dumps(provider.seen[1])
    assert workflow.store.verify(run["run_id"])["status"] == "OK"
    before = len(calls)
    assert workflow.step(run["run_id"])["state"] == "DEFERRED"
    assert len(calls) == before


def test_failure_return_is_seen_and_budget_is_charged(tmp_path):
    def after_failure(public):
        assert public["documents"][0]["status"] == "FAILED"
        assert public["documents"][0]["text"] == ""
        assert public["budget"]["remaining_requests"] == 0
        return answer("defer")
    workflow, _, calls = setup(tmp_path, Provider([answer(), after_failure]), {"https://example.com/preview": (403, "Denied")})
    run = workflow.run(workflow.start(TASK, max_requests=1)["run_id"])
    assert run["budget"]["http_attempts"] == 1 and len(calls) == 1
    assert run["budget"]["reserved_requests"] == 0


def test_interrupted_llm_is_not_reissued_on_resume(tmp_path):
    class Interrupted(Provider):
        def decide(self, public):
            self.seen.append(public)
            raise TimeoutError("response unknown")
    provider = Interrupted([])
    workflow, _, calls = setup(tmp_path, provider)
    run_id = workflow.start(TASK)["run_id"]
    first = workflow.step(run_id)
    assert first["pending"]["kind"] == "llm"
    resumed = InquiryWorkflow(tmp_path / "runs", provider, workflow.tools, clock=lambda: NOW)
    assert resumed.step(run_id)["state"] == "DEFERRED"
    assert len(provider.seen) == 1 and not calls


@pytest.mark.parametrize("mistake", ["fabricated_quote", "unacquired", "discovery", "conflict_one_source", "invented_gain"])
def test_unsupported_evidence_and_value_claims_are_rejected(mistake):
    doc = {"document_id": "doc-one", "status": "OK", "kind": "page", "text": "The lineup is provisional."}
    public = {"documents": [doc], "questions": [], "now": NOW.isoformat(), "task": TASK}
    claim = {"claim_id": "c", "question_id": "q1", "status": "source_claim", "text": "Source describes provisional lineup.",
             "citations": [{"document_id": "doc-one", "quote": "lineup is provisional"}]}
    response = answer("stop", claims=[claim])
    if mistake == "fabricated_quote": claim["citations"][0]["quote"] = "Confirmed starting eleven"
    if mistake == "unacquired": claim["citations"][0]["document_id"] = "future-doc"
    if mistake == "discovery": doc["kind"] = "discovery"
    if mistake == "conflict_one_source": claim["status"] = "conflicting"
    if mistake == "invented_gain": response["decision"]["expected_gain"] = 0.7
    with pytest.raises(Exception):
        validate_response(response, public)


def test_deadline_and_late_document_never_become_evidence(tmp_path):
    stamp = [NOW]
    provider = Provider([answer()])
    workflow, tools, _ = setup(tmp_path, provider, {"https://example.com/preview": (200, "<html>Unexpected result</html>")}, clock=lambda: stamp[0])
    execute = tools.execute
    def late(*args, **kwargs):
        result = execute(*args, **kwargs)
        stamp[0] += timedelta(hours=2)
        return result
    tools.execute = late
    run = workflow.step(workflow.start(TASK)["run_id"])
    assert run["reason"] == "LATE_ACQUISITION_RETURN"
    assert run["documents"][0]["text"] == ""
    assert run["budget"]["http_attempts"] == 1
    with pytest.raises(ValueError): workflow.start(TASK)


def test_duplicate_query_preserves_cost_without_second_request(tmp_path):
    workflow, _, calls = setup(tmp_path, Provider([answer(), answer()]), {"https://example.com/preview": (200, "<html>Preview pending</html>")})
    run = workflow.run(workflow.start(TASK)["run_id"])
    assert run["reason"] == "DUPLICATE_QUERY_REQUIRES_NEW_EVIDENCE_OR_WAIT"
    assert len(calls) == 1


def test_unknown_url_and_domain_rejected_before_request(tmp_path):
    workflow, _, calls = setup(tmp_path, Provider([answer(url="https://example.com/invented")]))
    run = workflow.run(workflow.start(TASK)["run_id"])
    assert run["reason"] == "INVALID_INQUIRY_RESPONSE" and not calls
    with pytest.raises(ValueError):
        workflow.start({**TASK, "seed_urls": ["https://127.0.0.1/secret"]})


def test_wait_is_explicit_future_state_not_automatic_schedule(tmp_path):
    response = answer("wait")
    response["decision"]["revisit_at"] = (NOW + timedelta(minutes=10)).isoformat()
    workflow, _, calls = setup(tmp_path, Provider([response]))
    run = workflow.run(workflow.start(TASK)["run_id"])
    assert run["state"] == "WAITING" and not calls


def test_provider_adapter_uses_stdin_no_shell_and_rejects_tool_events(tmp_path):
    def runner(command, **kwargs):
        assert command[-1] == "-" and "PUBLIC_STATE_JSON" in kwargs["input"]
        assert "--ephemeral" in command and "read-only" in command
        assert kwargs.get("shell", False) is False
        output = Path(command[command.index("--output-last-message") + 1])
        output.write_text(json.dumps(answer("stop")), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout=json.dumps({"type": "turn.completed", "usage": {"input_tokens": 10}}), stderr="")
    provider = CodexInquiryProvider(tmp_path / "provider", executable="test-cli", runner=runner)
    response = provider.decide({"task": TASK})
    assert response["response"]["decision"]["kind"] == "stop"
    def unsafe(*args, **kwargs):
        result = runner(*args, **kwargs)
        result.stdout += "\n" + json.dumps({"type": "item.completed", "item": {"type": "web_search"}})
        return result
    provider._runner = unsafe
    with pytest.raises(RuntimeError, match="UNDECLARED_TOOL"):
        provider.decide({"task": TASK})


def test_article_links_survive_long_navigation_and_prior_projection_is_replayable(tmp_path):
    nav = ''.join(f'<a href="/nav/{i}">Nav {i}</a>' for i in range(100))
    body = f'<html><header>{nav}</header><main><a href="/squad">Squad announcement</a></main></html>'
    workflow, tools, _ = setup(tmp_path, Provider([]))
    obs = tools.archive.save("open_page", "https://example.com/preview", {}, body.encode(), status=200, observed_at=NOW)
    prepared = {"endpoint": "https://example.com/preview", "params": {}, "tool": "read_url", "source": "open_page"}
    current = tools.project(obs, prepared)
    assert current['links'][0]['url'] == 'https://example.com/squad'
    assert current['links_truncated'] is True
    prior = OpenWebTools(tools.archive, ['example.com'], clock=lambda: NOW, extraction_version=1).project(obs, prepared)
    assert 'https://example.com/squad' not in {a['url'] for a in prior['links']}


def test_rss_discovery_does_not_promote_snippet_into_page_evidence(tmp_path):
    workflow, tools, _ = setup(tmp_path, Provider([]))
    body = b'<rss><channel><item><title>Squad</title><link>https://example.com/squad</link><description>Example news.</description></item></channel></rss>'
    obs = tools.archive.save('open_search', 'https://www.bing.com/search', {'q':'squad','format':'rss'}, body, status=200, observed_at=NOW)
    prepared = {'endpoint':'https://www.bing.com/search', 'params':{'format':'rss','q':'squad'}, 'tool':'search_web'}
    doc = tools.project(obs, prepared)
    assert doc['kind']=='discovery' and doc['links'][0]['url']=='https://example.com/squad'
    bad = tools.archive.save('open_search', 'https://www.bing.com/search', {}, b'<html>Challenge</html>', status=200, observed_at=NOW)
    assert tools.project(bad, prepared)['status']=='INVALID_CONTENT'


def test_provider_schema_keeps_uniqueness_in_local_contract_only():
    from football_agent.inquiry_contract import RESPONSE_SCHEMA
    from football_agent.inquiry_provider import provider_schema
    assert 'uniqueItems' in json.dumps(RESPONSE_SCHEMA)
    assert 'uniqueItems' not in json.dumps(provider_schema(RESPONSE_SCHEMA))


def test_offline_verifier_checks_raw_body_and_never_fetches(tmp_path):
    from football_agent.inquiry_verify import verify
    archive = RawArchive(tmp_path / 'evidence')
    calls = []
    tools = OpenWebTools(archive, ['example.com'], clock=lambda: NOW,
                         session_factory=lambda: Session({'https://example.com/preview':(200,'<html>Preview available</html>')},calls))
    workflow = InquiryWorkflow(tmp_path/'runs', Provider([answer(),answer('stop')]), tools, clock=lambda: NOW)
    run = workflow.run(workflow.start(TASK)['run_id'])
    result = verify(tmp_path,run['run_id'])
    assert result['raw_observations_verified']==1 and result['real_completed_llm_turns']==0
    assert len(calls)==1
    obs = archive.observations()[0]
    import gzip
    (archive.root/obs.payload_path).write_bytes(gzip.compress(b'Altered source'))
    with pytest.raises(ValueError,match='checksum'):
        verify(tmp_path,run['run_id'])


def test_query_replay_ignores_order_but_preserves_values():
    from football_agent.inquiry_verify import canonical_query_url
    assert canonical_query_url('https://example.com/?q=team&format=rss') == canonical_query_url('https://example.com/?format=rss&q=team')
    assert canonical_query_url('https://example.com/?q=team&format=rss') != canonical_query_url('https://example.com/?format=rss&q=other')
