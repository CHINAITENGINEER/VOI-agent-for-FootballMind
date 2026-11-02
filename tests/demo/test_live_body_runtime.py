"""New runtime over exact archived German bodies; no external HTTP or fitting.

Source-profile and menu admission are replayed at their original version. The
actual frozen g is called, and saved typed-v3 claims are explicitly adapted with
concept=null and validated by the new protocol. This is not a new provider run.
"""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import numpy as np
import pytest

from demo import live_g as live
from demo import live_sources
from football_agent.live_source_assertions import provider_input,bind_response,VERSION
from football_research.acquisition.semantic_f import validate_response

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'var/parallel/agent/round2/runs/live-g/runs/lg-764584968a2c435281bcf758efc8e9c8'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ArchiveTools:
    def __init__(self,backend,folder):self.backend,self.folder=backend,Path(folder)
    def prepare(self,action,public):return deepcopy(action)
    def execute(self,action,**kwargs):
        if action['tool']!='read_url':raise AssertionError('No search was performed by the original run')
        saved=next((a for a in self.backend.original['attempts'] if a['action']==action),None)
        if saved is None:raise AssertionError('Unarchived URL requested: '+str(action))
        returned=deepcopy(saved['response']);observation=returned['observation']
        original_path=SOURCE/'sources'/observation['payload_path']
        body=gzip.decompress(original_path.read_bytes())
        assert hashlib.sha256(body).hexdigest()==observation['content_hash']
        target=self.folder/observation['payload_path'];target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(original_path,target)
        metadata=original_path.with_name(original_path.name.removesuffix('.body.gz')+'.observation.json')
        shutil.copy2(metadata,target.with_name(metadata.name))
        self.backend.raw_checks.append(observation['id'])
        self.backend.source_calls.append(deepcopy(action))
        self.backend.stamp=saved['handoff_at']
        returned['replay']={'original_run_id':self.backend.original['run_id'],'new_http':False}
        return returned


class ArchiveBackend:
    def __init__(self,*,fail_initial=False,fail_repair=False):
        self.original=read(SOURCE/'run.json');self.stamp=self.original['created_at']
        self.source_calls=[];self.semantic_calls=[];self.raw_checks=[]
        self.fail_initial,self.fail_repair=fail_initial,fail_repair
        self.failed_once=False
    def initialize(self,match,folder):
        assert match['match_id']==self.original['match']['match_id']
        shutil.copytree(SOURCE/'setup',folder)
        return deepcopy(self.original['initialization'])
    def identity(self,manifest,folder,expected):
        assert manifest['market_initialization']['probabilities']==self.original['prior']
        self.stamp=self.original['identity']['verified_at']
        return deepcopy(self.original['identity'])
    def tools(self,folder,*args):return ArchiveTools(self,folder)
    def _respond(self,folder,public,kind):
        self.semantic_calls.append({'input_id':public['input_id'],'kind':kind})
        if kind=='initial' and self.fail_initial and not self.failed_once:
            self.failed_once=True
            raise ValueError('Fault injection: source quote validation failed')
        if kind=='validation_repair' and self.fail_repair:
            raise ValueError('Fault injection: repair remains invalid')
        step=next(s for s in self.original['steps'] if s['input_id']==public['input_id'])
        record=read(SOURCE/step['semantic_record'])
        # New live requests additionally expose source timestamps from this
        # same saved document. The canonical body and its identity are intact.
        assert record['input']=={k:v for k,v in public.items() if k!='source_context'}
        validate_response(record['input'],record['response'])
        # Version adaptation is explicit. It is not a new LLM interpretation.
        response={'schema':VERSION,'claims':[{**deepcopy(c),'concept':None} for c in record['response']['claims']]}
        request=provider_input(public)
        bound,binding=bind_response(public,request,response)
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        archived=folder/(public['input_id']+'.original-extraction.json')
        if not archived.exists():shutil.copy2(SOURCE/step['semantic_record'],archived)
        self.stamp=step['executed_at']
        return dict(input=deepcopy(public),response=bound,binding=binding,
            original_provider_receipt=record['receipt'],
            replay={'new_provider_http':False,'adapter':'saved_typed_v3_claims_add_explicit_null_concept',
                    'original_record_path':str(SOURCE/step['semantic_record']),
                    'original_record_sha256':sha(SOURCE/step['semantic_record'])})
    def semantics(self,folder,public,timeout):return self._respond(folder,public,'initial')
    def repair_semantics(self,folder,public,timeout):return self._respond(folder,public,'validation_repair')


def replay_current_runtime(store,*,fail_initial=False,fail_repair=False,provider_budget=None):
    backend=ArchiveBackend(fail_initial=fail_initial,fail_repair=fail_repair)
    original=backend.original;source_hash=sha(SOURCE/'run.json')
    reasons={(m['url'],m['title']):m['admission_reason'] for m in original['menus']}
    def original_admission(url,title,fixture,entries,seen):
        reason=reasons[url,title]
        return None if reason=='observed_article_link' else reason
    budget_patch=patch.object(live,'MAX_PROVIDER_CALLS',provider_budget) if provider_budget is not None else None
    if budget_patch:budget_patch.start()
    try:
        with patch.object(live,'key_from_environment',lambda:'offline-replay-no-real-credential'), \
             patch.object(live_sources,'profile',lambda binding:deepcopy(original['source_profile'])), \
             patch.object(live,'_directory_candidate_reason',original_admission):
            app=live.LiveGApplication(store,backend=backend,clock=lambda:backend.stamp)
            live.save(Path(store)/'catalog.json',dict(status='ready',updated_at=backend.stamp,
                matches=[deepcopy(original['match'])]))
            # Independently reproduce the saved original g decisions first.
            policy_checks=[];used=0
            for decision in original['decisions']:
                calculated=live.g_model.choose(app.g,decision['prequery_menu'],original['max_reads']-used,original['cost_loss_units'])
                assert calculated['action_id']==decision['action_id']
                policy_checks.append(dict(action_id=decision['action_id'],matched=True))
                if decision['action_id'] is not None:used+=1
            started=app.start(dict(match_id=original['match']['match_id'],max_reads=original['max_reads'],cost_loss_units=original['cost_loss_units']))
            # Copying the real archived setup can exceed 20 s on Windows when
            # other corpus/cache jobs contend for disk. This is a test harness
            # bound; the runtime's original simulated source clocks are intact.
            app.workers[started['run_id']].join(timeout=60)
            assert not app.workers[started['run_id']].is_alive()
            result=app.get(started['run_id'])
        assert sha(SOURCE/'run.json')==source_hash
        return app,result,backend,policy_checks
    finally:
        if budget_patch:budget_patch.stop()


def test_new_live_runtime_preserves_archived_positive_body_and_actual_g_choices(tmp_path):
    _,r,b,policy=replay_current_runtime(tmp_path)
    original=b.original
    assert r['state']=='STOPPED',r['reason']
    assert [d['action_id'] for d in r['decisions']]==[d['action_id'] for d in original['decisions']]
    assert [s['input_id'] for s in r['steps']]==[s['input_id'] for s in original['steps']]
    assert r['counts']=={'source':5,'provider':3,'body_attempts':3}
    np.testing.assert_allclose(r['before'],original['before'],atol=1e-12,rtol=0)
    np.testing.assert_allclose(r['current'],original['current'],atol=1e-12,rtol=0)
    assert r['body_pipeline']['changed_updates']==1
    assert r['body_pipeline']['modeled_claims']==2
    assert r['body_pipeline']['market_difference_l1']>0.01
    assert all(s['f_consumption']['body_consumed'] for s in r['steps'])
    assert len(b.raw_checks)==5 and len(b.semantic_calls)==3 and len(policy)==4
    assert r['g_training_eligible'] is False and r['outcome'] is None and r['gain'] is None


def test_semantic_validation_repair_is_one_extra_budgeted_call(tmp_path):
    _,r,b,_=replay_current_runtime(tmp_path,fail_initial=True)
    assert r['state']=='STOPPED',r['reason']
    assert r['counts']['provider']==4
    assert [a['kind'] for a in r['semantic_attempts']]==['initial','validation_repair','initial','initial']
    assert [a['status'] for a in r['semantic_attempts']]==['FAILED','OK','OK','OK']
    assert len(r['steps'])==3 and r['body_pipeline']['changed_updates']==1
    np.testing.assert_allclose(r['current'],b.original['current'],atol=1e-12,rtol=0)


def test_failed_repair_cannot_publish_after_or_consume_input(tmp_path):
    _,r,b,_=replay_current_runtime(tmp_path,fail_initial=True,fail_repair=True)
    assert r['state']=='FAILED'
    assert r['counts']['provider']==2 and len(b.semantic_calls)==2
    assert r['steps']==[] and r['used_input_ids']==[] and r['evidence_events']==[]
    assert r['current']==r['before'] and r['body_pipeline']['f_body_updates']==0


def test_exhausted_semantic_budget_prevents_repair_and_after(tmp_path):
    _,r,b,_=replay_current_runtime(tmp_path,fail_initial=True,provider_budget=1)
    assert r['state']=='FAILED'
    assert r['counts']['provider']==1 and len(b.semantic_calls)==1
    assert r['semantic_attempts'][0]['kind']=='initial'
    assert r['steps']==[] and r['current']==r['before'] and r['used_input_ids']==[]
