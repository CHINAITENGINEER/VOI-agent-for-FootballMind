from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from football_agent.inquiry_tools import OpenWebTools
from football_agent.news_discovery import NewsDiscoveryTools
from football_agent.source_recovery import SourceRoutedOpenWebTools
from football_agent.system_tls_web import HOST, SystemTLSOpenWebTools
from football_research.data_adapters import system_tls_source
from football_spider.storage.raw_archive import RawArchive

NOW=datetime(2026,10,7,8,tzinfo=timezone.utc)
ROOT='https://www.veikkausliiga.com/'
ARTICLE=ROOT+'uutiset/observed-article/'
BODY=('<html><main><h1>Official news</h1><a href="'+ARTICLE+'">Observed news title</a>'
      '<p>Actual observed source content.</p></main></html>').encode()


def fake_client(monkeypatch,*,body=BODY,status=200,code=0,verify='0',timeout=False):
    calls=[]
    monkeypatch.setattr(system_tls_source,'_client',lambda:('curl.exe','curl mock Schannel'))
    def run(command,**kwargs):
        calls.append(command)
        Path(command[command.index('--output')+1]).write_bytes(body)
        Path(command[command.index('--dump-header')+1]).write_text('HTTP/1.1 200\r\nContent-Type: text/html; charset=utf-8\r\n')
        if timeout:
            raise subprocess.TimeoutExpired(command,30)
        return SimpleNamespace(returncode=code,stdout=f'{status:03d}\t{verify}'.encode())
    monkeypatch.setattr(subprocess,'run',run)
    return calls


def tools_at(path,*,news=False):
    cls=NewsDiscoveryTools if news else SourceRoutedOpenWebTools
    kwargs={'index_urls':[ROOT,ROOT+'uutiset/'],'projection_version':5} if news else {}
    return cls(RawArchive(path),[HOST],route_overrides={HOST:'system_tls'},clock=lambda:NOW,**kwargs)


def prepared(tools,url=ROOT,*,observed=False):
    public={'task':{'seed_urls':[] if observed else [url]},
            'documents':[{'links':[{'url':url,'title':'Actually observed'}]}] if observed else []}
    return tools.prepare({'tool':'read_url','url':url},public)


def test_parent_news_prepare_execute_passes_authorized_capability_to_new_delegate(tmp_path,monkeypatch):
    calls=fake_client(monkeypatch)
    tools=tools_at(tmp_path,news=True)
    configuration=tools.configuration()
    first=tools.execute(prepared(tools),run_id='test',request_id='first')
    assert first['document']['status']=='OK' and first['document']['kind']=='index'
    action=tools.prepare({'tool':'read_url','url':ARTICLE},
        {'task':{'seed_urls':[ROOT]},'documents':[first['document']]})
    second=tools.execute(action,run_id='test',request_id='second')
    assert second['document']['status']=='OK'
    assert len(calls)==2 and [c[-1] for c in calls]==[ROOT,ARTICLE]
    assert first['http_attempts']==second['http_attempts']==1
    assert tools.configuration()==configuration
    assert configuration['source_route_policy']['version']=='source-route-selection-v2'
    rebuilt=NewsDiscoveryTools.from_configuration(tools.archive,configuration,clock=lambda:NOW)
    assert rebuilt.configuration()==configuration
    with pytest.raises(ValueError,match='capability'):
        rebuilt.execute(action,run_id='test',request_id='unprepared')
    third=rebuilt.execute(prepared(rebuilt,ARTICLE,observed=True),run_id='test',request_id='third')
    assert third['http_attempts']==1 and len(calls)==3


def test_command_keeps_strict_tls_no_redirect_retry_credentials_or_curlrc(tmp_path,monkeypatch):
    calls=fake_client(monkeypatch)
    tools=tools_at(tmp_path)
    action=prepared(tools)
    returned=tools.execute(action,run_id='test',request_id='page')
    command=calls[0]
    assert command[:2]==['curl.exe','--disable']
    assert command[command.index('--retry')+1]=='0'
    assert command[command.index('--proto')+1]=='=https'
    assert not set(command)&{'-k','--insecure','-L','--location','--ssl-no-revoke','--ssl-revoke-best-effort',
                            '--netrc','--user','--cookie','--cookie-jar'}
    observation=tools.archive.observations()[0]
    assert tools.archive.read_body(observation)==BODY
    transport=observation.context['transport']
    assert transport['route']=='system_tls' and transport['tls_verification'] is True
    assert transport['default_system_certificate_revocation'] is True
    assert observation.context['inquiry_run_id']=='test'
    assert observation.context['inquiry_request_id']=='page'
    base=OpenWebTools(tools.archive,[HOST],clock=lambda:NOW,extraction_version=3)
    assert returned['document']==base.project(observation,action)


@pytest.mark.parametrize('status,code,verify,error',[
    (403,0,'0',None),(302,0,'0',None),(0,60,'0','CurlExit60'),
    (200,0,'1','SystemTLSVerificationOrStatusUnconfirmed'),
])
def test_single_failure_archived_without_fallback_or_retries(tmp_path,monkeypatch,status,code,verify,error):
    calls=fake_client(monkeypatch,body=b'partial response',status=status,code=code,verify=verify)
    tools=tools_at(tmp_path,news=True)
    returned=tools.execute(prepared(tools),run_id='test',request_id='failed')
    assert len(calls)==1 and len(tools.archive.observations())==1
    assert returned['http_attempts']==1 and returned['document']['status']=='FAILED'
    assert returned['observation']['error_class']==error
    assert tools.archive.read_body(tools.archive.observations()[0])==b'partial response'


def test_timeout_preserves_partial_raw_as_failed_once(tmp_path,monkeypatch):
    calls=fake_client(monkeypatch,body=b'partial',timeout=True)
    tools=tools_at(tmp_path)
    returned=tools.execute(prepared(tools),run_id='test',request_id='timeout')
    assert len(calls)==1 and returned['document']['status']=='FAILED'
    observation=tools.archive.observations()[0]
    assert observation.error_class=='TimeoutExpired' and observation.context['response_complete'] is False
    assert tools.archive.read_body(observation)==b'partial'


@pytest.mark.parametrize('url',[
    'http://www.veikkausliiga.com/', 'https://www.veikkausliiga.com.evil.invalid/',
    'https://www.veikkausliiga.com:444/', 'https://user:pass@www.veikkausliiga.com/',
    'https://127.0.0.1/', ROOT+'?token=x',ROOT+'?q=arbitrary',
    ROOT+'admin/',ROOT+'tilastot/2025/veikkausliiga/ottelut/123/',
    ROOT+'uutiset/2026/02/31/not-a-date/',
])
def test_unregistered_or_private_paths_rejected_before_io(tmp_path,monkeypatch,url):
    calls=fake_client(monkeypatch)
    tools=tools_at(tmp_path)
    with pytest.raises(ValueError):prepared(tools,url,observed=True)
    assert calls==[] and tools.archive.observations()==[]


def test_news_article_cannot_be_hand_seeded_but_native_fixture_seed_is_allowed(tmp_path,monkeypatch):
    calls=fake_client(monkeypatch)
    tools=tools_at(tmp_path)
    with pytest.raises(ValueError,match='must be observed'):
        prepared(tools,ARTICLE)
    native=ROOT+'tilastot/2026/veikkausliiga/ottelut/4353553/'
    returned=tools.execute(prepared(tools,native),run_id='test',request_id='fixture')
    assert len(calls)==1 and returned['document']['status']=='OK'


def test_forged_prepared_endpoint_or_route_override_rejected(tmp_path,monkeypatch):
    calls=fake_client(monkeypatch)
    tools=tools_at(tmp_path)
    action=prepared(tools)
    with pytest.raises(ValueError,match='override'):
        tools.execute({**action,'route':'system_tls'},run_id='t',request_id='forged-route')
    with pytest.raises(ValueError,match='capability'):
        tools.execute({**action,'endpoint':ARTICLE},run_id='t',request_id='forged-url')
    assert calls==[]


def test_system_tls_is_explicit_host_override_not_default_or_candidate_choice(tmp_path):
    archive=RawArchive(tmp_path)
    for kwargs in ({'default_route':'system_tls'},
                   {'route_overrides':{'www.bing.com':'system_tls'}},
                   {'route_overrides':{'gnistan.fi':'system_tls'}}):
        with pytest.raises(ValueError):SourceRoutedOpenWebTools(archive,[HOST,'gnistan.fi'],**kwargs)
    old=SourceRoutedOpenWebTools(archive,[HOST]).configuration()
    assert old['source_route_policy']['version']=='source-route-selection-v1'
    assert 'system_tls_policy' not in old['source_route_policy']
    assert SourceRoutedOpenWebTools.from_configuration(archive,old).configuration()==old
    config=tools_at(tmp_path/'new').configuration()
    assert SourceRoutedOpenWebTools.from_configuration(archive,config).configuration()==config
    wrong=deepcopy(config)
    wrong['source_route_policy']['version']='source-route-selection-v1'
    with pytest.raises(ValueError,match='reconstruct'):
        SourceRoutedOpenWebTools.from_configuration(archive,wrong)


def test_direct_new_tools_still_obeys_original_prepare_contract(tmp_path,monkeypatch):
    calls=fake_client(monkeypatch)
    tools=SystemTLSOpenWebTools(RawArchive(tmp_path),[HOST],clock=lambda:NOW)
    action=prepared(tools)
    returned=tools.execute(action,run_id='test',request_id='direct-class')
    assert len(calls)==1 and returned['document']['status']=='OK'
