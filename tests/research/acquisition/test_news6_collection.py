"""Opt-in v6 reaches the existing collector, typed evidence and raw replay."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import json
import socket

import pytest

from football_agent.news_discovery import NewsDiscoveryTools
from football_agent.learned_voi_replay import replay_saved_run
from football_research.acquisition.visible_state import encode_visible_state
from football_spider.storage.raw_archive import RawArchive
import test_evidence_collection as support
from test_evidence_live_workflow import job

ROOT=Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args,**kwargs):pytest.fail('News6 integration cannot acquire external data')
    for owner,name in ((socket,'getaddrinfo'),(socket,'create_connection'),(socket.socket,'connect')):
        monkeypatch.setattr(owner,name,denied)


def original_article():
    for folder in (ROOT/'var/parallel/agent/round198/live-cohort').glob('task-*/analysis/match-*'):
        archive=RawArchive(folder/'source-archive')
        for obs in archive.observations():
            if 'hjk-hankki-kaksi' in obs.endpoint:
                return obs.endpoint,archive.read_body(obs)
    raise AssertionError('Pinned R198 HJK article unavailable')


def test_public_collector_reprojects_real_html_through_v6_and_preserves_parent_before_body(tmp_path,monkeypatch):
    url,html=original_article()
    monkeypatch.setattr(support,'ARTICLE',url)
    prepared,_,_,path=support.inputs(tmp_path)
    calls=support.transport(monkeypatch)
    base=NewsDiscoveryTools.execute
    def execute(self,prepared,*,run_id,request_id):
        if prepared['endpoint']!=url:
            return base(self,prepared,run_id=run_id,request_id=request_id)
        calls.append(deepcopy(prepared))
        now=datetime.now(timezone.utc)
        obs=self.archive.save(prepared['source'],prepared['endpoint'],prepared['params'],html,
            status=200,observed_at=now,context={'requested_at':now.isoformat(),'inquiry_run_id':run_id})
        return {'document':self.project(obs,prepared),'observation':obs.model_dump(mode='json'),
            'http_attempts':1,'money':None,'money_status':'not_measured','latency_seconds':.01}
    monkeypatch.setattr(NewsDiscoveryTools,'execute',execute)
    report=support.collect(tmp_path,prepared,path,news_projection_version=6,seed=0)
    run=support.read(tmp_path/'run/run.json')
    article=next(d for d in run['documents'] if d['url']==url)
    assert article['structured_news']['role']=='article'
    selected=next(c for c in run['cycles'] if c['selected_action'] and c['selected_action'].get('url')==url)
    assert not any(d['url']==url for d in selected['public_before']['documents'])
    after=encode_visible_state(selected['public_after'])
    body=[ref for d in after['before']['documents'] for ref in d['content_refs'] if ref['role']=='body']
    assert body and not after['audit']['excluded_segments']
    assert run['final_probabilities'] is None and report['numeric_voi_training_eligible'] is False
    replay=replay_saved_run(tmp_path/'run',output=tmp_path/'replay',project_root=ROOT)
    assert replay['identical_replay'] and replay['raw_responses_reprojected']==len(calls)==4


def test_live_job_pins_configured_projection_and_resumes_without_new_requests(job):
    from football_agent import live_match_workflow as workflow
    settings=support.read(job['settings']);settings['news_projection_version']=6
    job['settings'].write_text(json.dumps(settings),encoding='utf-8')
    result=workflow.start_match(job['query'],job['output'],settings_path=job['settings'])
    assert result['status']=='awaiting_result_no_prior',result
    config=support.read(job['output']/result['run_directory']/'configuration.json')
    assert config['supervisor_metadata']['transport']['news_discovery']['projection_version']==6
    assert config['supervisor_metadata']['evidence_collection']['settings']['news_projection_version']==6
    before=deepcopy(job['calls'])
    assert workflow.resume_match(job['output'])==result and job['calls']==before
