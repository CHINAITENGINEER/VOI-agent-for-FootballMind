from copy import deepcopy
import json
from pathlib import Path
import runpy

import pytest
from jsonschema import validate

from football_agent.acquisition_boundary import CandidateBatch, SHARED_FIELDS
from football_agent.experience import validate_decision
from football_agent.match_proposals import MatchCandidateProvider, MatchProposalProtocol
from football_agent.runtime_cli import main

helpers=runpy.run_path(str(Path(__file__).with_name('test_match_experience.py')))
match_decision=helpers['match_decision']


def public():
    return {'task_id':'fixture','match_input':'构造甲 vs 构造乙','task_type':'match_analysis',
            'documents':[],'seed_urls':[],'do_not_repeat':[],'final_decision_only':False}


def proposal(p):
    a=match_decision(p); b=deepcopy(a); b['query']='second fixture query'
    shared={k:a[k] for k in SHARED_FIELDS}
    return {**shared,'candidates':[{k:v for k,v in item.items() if k not in SHARED_FIELDS and k!='alternatives'}
                                  for item in [a,b]],'recommended_index':0}


def test_proposal_is_one_model_call_with_shared_analysis_and_distinct_candidates(tmp_path):
    calls=[]
    original={'response':proposal(public()),'test_double':True}
    class Provider:
        def decide(self,p):calls.append(deepcopy(p));return original
    validate(original['response'],MatchProposalProtocol.schema)
    result=MatchCandidateProvider(tmp_path,provider=Provider()).propose(public())
    assert len(calls)==1 and result['raw_provider_result']==original
    assert len(result['candidates'])==2 and result['recommended_index']==0
    for item in result['candidates']:validate_decision(item,public())
    batch=CandidateBatch.build(public(),result['candidates'],recommended_index=0,origin='structured_candidates')
    assert batch.candidates[0].candidate_id!=batch.candidates[1].candidate_id


def test_proposal_recommendation_must_refer_to_returned_candidate(tmp_path):
    class Provider:
        def decide(self,p):
            response=proposal(p);response['recommended_index']=4
            return {'response':response,'test_double':True}
    with pytest.raises(ValueError,match='OUT_OF_RANGE'):
        MatchCandidateProvider(tmp_path,provider=Provider()).propose(public())


def test_match_proposer_does_not_relabel_legacy_question(tmp_path):
    with pytest.raises(ValueError,match='MATCH_TASK'):
        MatchCandidateProvider(tmp_path,provider=object()).propose({**public(),'task_type':'legacy_inquiry'})


def test_headless_create_show_verify_make_no_model_or_source_calls(tmp_path,capsys,monkeypatch):
    def unexpected(*args,**kwargs):raise AssertionError('Offline command called provider')
    monkeypatch.setattr('football_agent.runtime_cli.MatchCandidateProvider',unexpected)
    main(['--root',str(tmp_path),'create','--match','构造甲 vs 构造乙'])
    state=json.loads(capsys.readouterr().out)
    assert state['config']['proposal_mode']=='structured_candidates'
    assert state['spent']=={'http_calls':0,'llm_calls':0}
    main(['--root',str(tmp_path),'show',state['run_id']])
    assert json.loads(capsys.readouterr().out)==state
    main(['--root',str(tmp_path),'verify',state['run_id']])
    report=json.loads(capsys.readouterr().out)
    assert report['accepted_decisions_verified']==0 and report['real_calls_made_by_replay']==0
    main(['--root',str(tmp_path),'cancel',state['run_id']])
    assert json.loads(capsys.readouterr().out)['status']=='cancelled'
