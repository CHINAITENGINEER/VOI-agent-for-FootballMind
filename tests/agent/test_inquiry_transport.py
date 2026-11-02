"""Constructed IO checks for explicitly routed operational inquiry acquisition."""
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
import requests

from football_agent.audit import digest, redact
from football_agent.inquiry import InquiryWorkflow
from football_agent.inquiry_tools import OpenWebTools
from football_agent.inquiry_transport import (
    RoutedOpenWebTools, SPFTLL_ENDPOINT, SPFTLL_PRESET_ID, request_header_profile,
)
from football_agent.inquiry_verify import verify
from football_agent.transport_probe import DiagnosticSession, exception_facts
from football_spider.storage.raw_archive import RawArchive


NOW = datetime(2026, 9, 27, 2, tzinfo=timezone.utc)
LIST_URL = SPFTLL_ENDPOINT + "?dateFormat=&notSingle=false&mKind=0&reqtype=0"


def tools_at(path, route="direct"):
    return RoutedOpenWebTools(RawArchive(path), ["slb.haoyun999.cn", "www.englandfootball.com"],
                              route=route, clock=lambda: NOW)


def prepare(tools, url=LIST_URL):
    return tools.prepare({"tool":"read_url", "url":url, "query":None},
                         {"task":{"seed_urls":[url]}, "documents":[]})


def test_route_is_required_and_configuration_contains_no_proxy_values(tmp_path, monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://user:SECRET@proxy.invalid:99")
    with pytest.raises(TypeError):
        RoutedOpenWebTools(RawArchive(tmp_path), ["slb.haoyun999.cn"])
    with pytest.raises(ValueError):
        tools_at(tmp_path, "from_model")
    config = tools_at(tmp_path).configuration()
    policy = config["transport_policy"]
    assert config["version"] == "open-web-read-v2"
    assert policy["route"] == "direct" and policy["tls_verification"] is True
    assert policy["candidate_may_override_policy"] is False
    assert "SECRET" not in json.dumps(config) and "proxy.invalid" not in json.dumps(config)
    assert tools_at(tmp_path, "environment").transport_policy()["proxy_policy"] == "inherit_environment"


@pytest.mark.parametrize("endpoint", [
    "https://slb.haoyun999.cn.evil.invalid/api/Game/GetSimpleMatchsAll",
    "https://www.englandfootball.com/api/Game/GetSimpleMatchsAll",
    "https://slb.haoyun999.cn/api/Game/GetSimpleMatchsAll/",
    "https://slb.haoyun999.cn/api/Game/GetMoreSpInfo",
    "http://slb.haoyun999.cn/api/Game/GetSimpleMatchsAll",
    "https://slb.haoyun999.cn:444/api/Game/GetSimpleMatchsAll",
])
def test_preset_is_not_sent_to_other_origins_or_paths(endpoint):
    profile = request_header_profile(endpoint)
    assert profile["preset_id"] == "research-public-page-v1"
    assert profile["headers"] == {"User-Agent":"FootballMind/0.1 research"}


def test_parameters_preset_archive_and_projection_preserve_base_contract(tmp_path, monkeypatch):
    body = b'{"code":0,"data":[{"list":[{"matchId":123,"homeChs":"Example H","awayChs":"Example A"}]}]}'
    calls=[]
    def get(session, url, **kwargs):
        calls.append((url, kwargs, session.trust_env))
        assert kwargs["params"] == {"dateFormat":"", "notSingle":"false", "mKind":"0", "reqtype":"0"}
        assert kwargs["headers"]["Referer"] == "https://xx.spftll.cn/"
        assert kwargs["headers"]["X-Requested-With"] == "XMLHttpRequest"
        assert "Authorization" not in kwargs["headers"] and "Cookie" not in kwargs["headers"]
        assert kwargs["timeout"] == (6,16) and kwargs["allow_redirects"] is False
        return SimpleNamespace(status_code=200, content=body, headers={"content-type":"application/json"}, close=lambda:None)
    monkeypatch.setattr(DiagnosticSession,"get",get)
    tools=tools_at(tmp_path); action=prepare(tools)
    base=OpenWebTools(tools.archive,tools.allowed_hosts,clock=lambda:NOW)
    assert action==prepare(base)
    run_id,request_id=str(uuid4()),str(uuid4())
    result=tools.execute(action,run_id=run_id,request_id=request_id)
    assert len(calls)==1 and calls[0][2] is False and result["http_attempts"]==1
    obs=tools.archive.observations()[0]
    assert tools.archive.read_body(obs)==body
    assert result["document"]==base.project(obs,action)==tools.project(obs,action)
    assert obs.context["inquiry_run_id"]==run_id and obs.context["inquiry_request_id"]==request_id
    context=obs.context["transport"]
    assert context["route"]=="direct" and context["request_header_preset"]==SPFTLL_PRESET_ID
    assert context["policy_sha256"]==digest(tools.configuration()["transport_policy"])
    assert context["request_headers"]==calls[0][1]["headers"]


def test_plain_page_retains_ordinary_headers_and_failure_detail_is_sanitized(tmp_path, monkeypatch):
    calls=[]
    def get(session,url,**kwargs):
        calls.append(kwargs)
        assert kwargs["headers"]=={"User-Agent":"FootballMind/0.1 research"}
        assert session.trust_env is True
        error=requests.ConnectionError("proxy credential SECRET should stay private", ConnectionRefusedError(10061,"SECRET"))
        session.failure=exception_facts(error)
        raise error
    monkeypatch.setattr(DiagnosticSession,"get",get)
    tools=tools_at(tmp_path,"environment")
    result=tools.execute(prepare(tools,"https://www.englandfootball.com/england/mens-senior-team/news"),
                         run_id=str(uuid4()),request_id=str(uuid4()))
    assert len(calls)==1 and result["document"]["status"]=="FAILED"
    assert result["document"]["text"]==""
    detail=result["observation"]["context"]["transport"]["failure_detail"]
    assert "connection_refused" in detail["categories"]
    assert "SECRET" not in json.dumps(result)


def test_model_shaped_transport_overrides_are_rejected_before_any_request(tmp_path, monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError("No acquisition expected")
    monkeypatch.setattr(DiagnosticSession,"get",forbidden)
    tools=tools_at(tmp_path)
    with pytest.raises(ValueError,match="override"):
        tools.execute({**prepare(tools),"headers":{"Authorization":"SECRET"}},run_id=str(uuid4()),request_id=str(uuid4()))
    assert tools.archive.observations()==[]


def test_complete_constructed_inquiry_and_existing_offline_replay(tmp_path, monkeypatch):
    workflow_url = LIST_URL
    calls=[]
    def get(session,url,**kwargs):
        calls.append((url,kwargs))
        return SimpleNamespace(status_code=200,content=b'{"code":0,"data":[]}',headers={},close=lambda:None)
    monkeypatch.setattr(DiagnosticSession,"get",get)
    class Provider:
        def configuration(self):return {"provider":"constructed_test","real":False}
        def decide(self,public):
            query=not public["documents"]
            question={"question_id":"q1","question":"Are fixtures returned?","decision_relevance":"Check source availability.","answer_could_change":"Retain unknown if no applicable records.","based_on_documents":[]}
            candidate={"candidate_id":"c1","question_id":"q1","tool":"read_url","url":workflow_url,"query":None,"rationale":"Read registered public fixture endpoint.","answer_could_change":"Update the source availability state."}
            response={"questions":[question],"claim_updates":[],"candidates":[candidate] if query else [],
                      "decision":{"kind":"query" if query else "defer","candidate_id":"c1" if query else None,
                                  "reason":"Need one public source observation." if query else "No applicable fixture fact is asserted.","revisit_at":None}}
            return {"response":response,"receipt":{"real_provider_call":False,"constructed":True}}
    tools=tools_at(tmp_path/"evidence")
    workflow=InquiryWorkflow(tmp_path/"runs",Provider(),tools,clock=lambda:NOW)
    task={"task_id":"constructed-transport-case","objective":"Inspect fixture source availability.",
          "deadline":(NOW+timedelta(hours=1)).isoformat(),"initial_facts":[],"seed_urls":[workflow_url]}
    run=workflow.run(workflow.start(task,max_requests=1,max_llm_calls=2)["run_id"])
    assert run["state"]=="DEFERRED" and len(calls)==1
    assert run["task"]["seed_urls"] == [LIST_URL]
    assert calls[0][1]["params"]["dateFormat"] == ""
    assert run["configuration"]["tools"]["transport_policy"]["route"]=="direct"
    proof=verify(tmp_path,run["run_id"])
    assert proof["raw_observations_verified"]==1 and proof["new_network_requests"]==0
    assert proof["real_completed_llm_turns"]==0


def test_empty_sensitive_query_is_redacted_while_public_empty_parameter_survives():
    from urllib.parse import parse_qsl, urlsplit
    public = redact("https://www.englandfootball.com/public?api_key=&dateFormat=&q=team")
    assert dict(parse_qsl(urlsplit(public).query, keep_blank_values=True)) == {
        "api_key": "[REDACTED]", "dateFormat": "", "q": "team",
    }
