'''Constructed mechanism tests; no live tool calls or utility labels.'''
from copy import deepcopy

import pytest

from football_agent.audit import digest
from football_agent.candidate_grounded_navigation import (CandidateGroundedWorkflow,
    CandidateGroundedProtocol, grounded_candidate, project_target_choice)
from football_agent.evidence_windows import EvidenceWindows
from test_target_navigation import make as old_make, choice, query, read_new, finish

IDENTITY = 'NEXT HOME MATCH Team A VS Team B 10-09 16:30'


def convert(result):
    result = deepcopy(result)
    state = result['target_state']
    state.pop('candidate_label');state.pop('support_claim_indices')
    state['candidate'] = None
    return result


def candidate(index=0):
    return {'claim_index':index,'identity_quote':IDENTITY,'team_a_text':'Team A',
        'team_b_text':'Team B','connector_text':'VS','date_text':'10-09',
        'time_text':'16:30','scope_text':'NEXT HOME MATCH'}


def make(tmp_path,replies,**limits):
    old,rid,seen,requests,handoff,contract=old_make(tmp_path/'template',replies,**limits)
    old_run=old.get(rid)
    handoff=deepcopy(handoff)
    handoff['documents'][0]['text']=IDENTITY+'. '+'Other constructed archive context. '*80
    handoff['document_digests']={d['document_id']:digest(d) for d in handoff['documents']}
    handoff['opened']={d['document_id']:[next(iter(EvidenceWindows(d).windows))] for d in handoff['documents']}
    flow=CandidateGroundedWorkflow(tmp_path/'grounded',old.provider,old.tools,clock=old.clock)
    run=flow.start(old_run['task'],handoff,contract,**limits)
    return flow,run['run_id'],seen,requests,handoff,contract


def with_candidate(public,action='defer'):
    result=convert(choice(public,action))
    doc=public['documents'][0];window=doc['visible_windows'][0]
    result['claims']=[{'document_id':doc['document_id'],'window_id':window['window_id'],
        'quote':IDENTITY,'interpretation':'Only a home fixture clue without year.'}]
    result['target_state'].update(status='candidate_only',candidate=candidate(),
        open_requirements=['all_fixture_order','kickoff_date_or_time','reference_time_applicability'])
    return result


def test_keeps_source_date_clue_without_invented_year_or_home_away(tmp_path):
    flow,rid,seen,requests,*_=make(tmp_path,[with_candidate])
    run=flow.step(rid)
    assert run['reason']=='AGENT_DEFER' and not requests
    state=run['decisions'][0]['choice']['target_state']
    binding=grounded_candidate(state['candidate'],run['decisions'][0]['choice']['claims'],run['target_contract'])
    assert binding['candidate_label']=='Team A VS Team B | 10-09 | 16:30'
    assert binding['world_truth_verified'] is False and binding['home_away_order_verified'] is False
    assert flow.step(rid)==run


@pytest.mark.parametrize('mutation',['unrelated','cross_quote','invented_year','wrong_team','duplicate','scope_hidden','year_hidden','false_relation'])
def test_invalid_candidate_never_dispatches_query(tmp_path,mutation):
    def invalid(public):
        result=with_candidate(public)
        result.update(action='query_source',query={'tool':'read_url','url':'https://example.com/schedule','query':None},
            information_test={'requirement':'all_fixture_order','useful_if':'Full schedule resolves ordering.','irrelevant_if':'Ranking only.'})
        c=result['target_state']['candidate']
        if mutation=='unrelated':result['claims'][0]['quote']='Ranking points wins losses'
        elif mutation=='cross_quote':
            result['claims'].append({**result['claims'][0],'quote':'Ranking points wins losses'})
            c['claim_index']=1
        elif mutation=='invented_year':c['date_text']='2026-10-09'
        elif mutation=='wrong_team':c['team_a_text']='Team C'
        elif mutation=='duplicate':result['claims'][0]['quote']=IDENTITY+' / '+IDENTITY
        elif mutation=='scope_hidden':
            c['scope_text']=None
            result['target_state']['open_requirements'].remove('all_fixture_order')
        elif mutation=='year_hidden':result['target_state']['open_requirements'].remove('kickoff_date_or_time')
        else:c['connector_text']='versus'
        return result
    flow,rid,seen,requests,*_=make(tmp_path,[invalid])
    run=flow.step(rid)
    assert run['reason']=='INVALID_NAVIGATION_RESPONSE' and not requests
    assert run['budget']['llm_calls']==1 and run['budget']['http_attempts']==0


def test_unrecognized_prose_can_remain_unknown_and_agent_can_acquire(tmp_path):
    flow,rid,seen,requests,*_=make(tmp_path,[lambda p:convert(query(p)),lambda p:convert(read_new(p)),lambda p:convert(finish(p))])
    for _ in range(3):run=flow.step(rid)
    assert run['reason']=='AGENT_DEFER' and len(requests)==1
    assert run['budget']['window_reads']==1 and run['decisions'][-1]['choice']['target_state']['candidate'] is None
    feedback=flow.public(run)['prior_information_tests']
    assert feedback[-1]['assessment']['requirement']=='all_fixture_order'
    assert feedback[-1]['scope']=='this_return_for_this_requirement_only_not_source_wide_utility'
    assert feedback[-1]['claims'][0]['quote'] in seen[-1]['documents'][-1]['visible_windows'][0]['text']


def test_return_feedback_guides_later_read_without_programmed_source_choice(tmp_path):
    def assess_and_read_other(public):
        result=convert(finish(public))
        result['return_assessment']['outcome']='irrelevant'
        doc=public['documents'][0]
        result.update(action='read_window',document_id=doc['document_id'],window_id=doc['catalog'][1]['window_id'],
            information_test={'requirement':'fixture_identity','useful_if':'Another body might contain matchup.','irrelevant_if':'Unrelated activity.'})
        return result
    def decide_from_memory(public):
        feedback=public['prior_information_tests'][-1]
        assert feedback['assessment']['requirement']=='all_fixture_order'
        assert feedback['assessment']['outcome']=='irrelevant'
        assert public['last_action_result']['information_test']['requirement']=='fixture_identity'
        return convert(finish(public))
    flow,rid,seen,requests,*_=make(tmp_path,[lambda p:convert(query(p)),lambda p:convert(read_new(p)),assess_and_read_other,decide_from_memory],max_llm_calls=4)
    for _ in range(4):run=flow.step(rid)
    assert run['reason']=='AGENT_DEFER' and len(requests)==1 and run['budget']['window_reads']==2
    assert len(flow.public(run)['prior_information_tests'])==3


def test_same_table_cooccurrence_is_not_a_supported_matchup(tmp_path):
    flow,rid,*_=make(tmp_path,[])
    contract=flow.get(rid)['target_contract']
    c=candidate();c['identity_quote']='Ranking Team A 1 Team B 2'
    c.update(date_text=None,time_text=None,scope_text=None)
    with pytest.raises(ValueError,match='literal matchup'):
        grounded_candidate(c,[{'quote':c['identity_quote']}],contract)


def test_schema_rejects_freeform_label_even_when_plausible(tmp_path):
    flow,rid,*_=make(tmp_path,[])
    public=flow.public(flow.get(rid))
    with pytest.raises(Exception):project_target_choice(choice(public),public['task_contract'])
    assert CandidateGroundedProtocol.schema['properties']['target_state']['additionalProperties'] is False


@pytest.mark.parametrize('side',['left','right'])
def test_team_name_substring_does_not_ground_a_different_club(tmp_path,side):
    flow,rid,*_=make(tmp_path,[])
    c=candidate()
    c['identity_quote']=IDENTITY.replace('Team A','SuperTeam A') if side=='left' else IDENTITY.replace('Team B','Team Btown')
    with pytest.raises(ValueError,match='literal matchup'):
        grounded_candidate(c,[{'quote':c['identity_quote']}],flow.get(rid)['target_contract'])


def test_literal_matchup_in_unopened_window_cannot_be_claimed(tmp_path):
    def unopened(public):
        result=with_candidate(public)
        result['claims'][0]['window_id']=public['documents'][0]['catalog'][1]['window_id']
        return result
    flow,rid,seen,requests,*_=make(tmp_path,[unopened])
    run=flow.step(rid)
    assert run['reason']=='INVALID_NAVIGATION_RESPONSE' and not requests
    assert not run['decisions']
