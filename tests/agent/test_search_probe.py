"""Constructed diagnostics never follow redirects or repeat reserved actions."""
import json
from types import SimpleNamespace

import pytest

from football_agent.search_probe import VERSION, run_registered_probe
from football_spider.storage.raw_archive import RawArchive


def manifest(root):
    value={"schema":VERSION,"max_physical_http_requests":3,"allowed_result_hosts":["www.kleague.com"],
           "actions":[{"action_id":"probe-1","endpoint":"https://cn.bing.com/search","params":{"q":"site:www.kleague.com 2026","format":"rss"}}]}
    (root/"manifest.json").write_text(json.dumps(value),encoding="utf-8")


def test_explicit_redirect_is_archived_not_followed_or_repeated(tmp_path):
    manifest(tmp_path);calls=[]
    class Session:
        failure=None
        location="https://www.bing.com/search?q=site%3Awww.kleague.com&format=rss"
        def get(self,url,**kwargs):
            assert (tmp_path/"probe-1"/"intent.json").exists()
            assert kwargs["allow_redirects"] is False
            calls.append(url)
            return SimpleNamespace(status_code=302,content=b"Object moved",headers={"Location":self.location},close=lambda:None)
        def close(self):pass
    result=run_registered_probe(tmp_path,"probe-1",session_factory=lambda route:Session())
    assert len(calls)==1 and result["document_status"]=="FAILED"
    assert result["redirect_location"]==Session.location
    with pytest.raises(ValueError,match="repeated"):
        run_registered_probe(tmp_path,"probe-1",session_factory=lambda route:Session())
    assert len(calls)==1
    assert len(RawArchive(tmp_path/"evidence").observations())==1


def test_rss_transport_success_is_separate_from_allowed_discovery(tmp_path):
    manifest(tmp_path)
    body=b'<rss><channel><item><title>Other</title><link>https://unrelated.example/test</link></item></channel></rss>'
    class Session:
        failure=None;location=None
        def get(self,url,**kwargs):return SimpleNamespace(status_code=200,content=body,headers={},close=lambda:None)
        def close(self):pass
    result=run_registered_probe(tmp_path,"probe-1",session_factory=lambda route:Session())
    assert result["rss_item_count"]==1 and result["rss_root_tag"]=="rss"
    assert result["document_status"]=="NO_DATA" and result["allowed_link_count"]==0


def test_reserved_unknown_attempts_consume_the_physical_budget(tmp_path):
    from football_agent.audit import digest
    manifest(tmp_path)
    plan=json.loads((tmp_path/"manifest.json").read_text(encoding="utf-8"))
    ledger={"manifest_sha256":digest(plan),"attempts":[
        {"action_id":str(i),"state":"RESERVED_BEFORE_NETWORK"} for i in range(3)]}
    (tmp_path/"attempts.json").write_text(json.dumps(ledger),encoding="utf-8")
    def no_session(route):
        pytest.fail("Exhausted physical budget must be checked before session construction")
    with pytest.raises(ValueError,match="physical budget"):
        run_registered_probe(tmp_path,"probe-1",session_factory=no_session)
