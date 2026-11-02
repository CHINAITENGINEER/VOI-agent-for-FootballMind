from copy import deepcopy
import numpy as np
import pytest
from football_research.acquisition import structured_state_gain as g
from football_research.acquisition import state_candidate_gain as old
from football_research.acquisition import historical_gain_head as public_g


def public(title='Joe Gomez fitness update'):
    return {'schema':public_g.QUERY_SCHEMA,'probabilities':[.4,.3,.3],
            'acquired_documents':1,'visible_candidates':2,'article_age_hours':1.,'title':title}


def snapshot():
    return {'physical_match_id':'m','used_input_ids':['seen'],'claims':[
        {'input_id':'seen','channel':'body','entity':'Joe Gomez','team':'home',
         'facet':'availability','value':'doubtful','pending':True,'text':'Joe Gomez is doubtful.'}]}


def test_same_candidate_distinguishes_pending_resolved_and_team():
    s=snapshot();a=g.build_query(public(),s)['structured']
    s['claims'][0].update(value='available',pending=False);b=g.build_query(public(),s)['structured']
    assert a['home_doubtful']>0 and a['named_pending_entity']>0
    assert b['home_available']>0 and b['named_pending_entity']==0
    s['claims'][0]['team']='away';c=g.build_query(public(),s)['structured']
    assert c['away_available']>0 and c['home_available']==0


def test_training_and_selection_never_imply_medical_clearance():
    s=snapshot();c=deepcopy(s['claims'][0]);c.update(facet='context:training_participation',value='training_participation',pending=False)
    s['claims'].append(c);c=deepcopy(c);c.update(facet='selection',value='bench');s['claims'].append(c)
    x=g.build_query(public(),s)['structured']
    assert x['known_training_participation']>0 and x['selection_known']>0
    assert x['home_doubtful']>0 and x['home_available']==0 and x['named_pending_entity']>0


def test_no_known_fact_is_not_proof_of_absence_and_title_is_not_body():
    s=snapshot();s['claims']=[]
    x=g.build_query(public(),s)['structured']
    assert x['title_availability']==1 and x['availability_candidate_uncovered']==1
    assert x['home_unavailable']==x['home_available']==0
    assert g.build_query(public('Tickets on sale'),s)['structured']['availability_candidate_uncovered']==0


def test_duplicate_sources_do_not_inflate_structured_assertion_count():
    s=snapshot();a=g.build_query(public(),s)['structured'];c=deepcopy(s['claims'][0])
    c.update(input_id='seen2',text='A second quote');s['used_input_ids'].append('seen2');s['claims'].append(c)
    assert g.build_query(public(),s)['structured']==a


def test_mixed_values_not_silently_resolved():
    s=snapshot();c=deepcopy(s['claims'][0]);c.update(value='unavailable',pending=False);s['claims'].append(c)
    x=g.build_query(public(),s)['structured']
    assert x['availability_mixed_values']>0 and x['named_pending_entity']>0


@pytest.mark.parametrize('key',['body','outcome','prediction_after','signed_gain'])
def test_hidden_fields_rejected(key):
    p=public();p[key]='hidden'
    with pytest.raises(ValueError):g.build_query(p,snapshot())
    s=snapshot();s[key]='hidden'
    with pytest.raises(ValueError):g.build_query(public(),s)


def test_unobserved_claim_rejected_and_inputs_unchanged():
    s=snapshot();original=deepcopy(s);g.build_query(public(),s);assert s==original
    s['claims'][0]['input_id']='unread'
    with pytest.raises(ValueError):g.build_query(public(),s)


@pytest.mark.parametrize('mode',['baseline','structured','combined'])
def test_fit_roundtrip_equal_match_weight_and_signed_labels(tmp_path,mode):
    rows=[{'physical_match_id':str(i),'query':g.build_query(public(t),snapshot()),'signed_gain':y}
          for i,(t,y) in enumerate([('injury update',-.03),('Tickets',.02),('Team news',0.)])]
    m=g.fit(rows,mode);rep=g.fit([rows[0]]*8+rows[1:],mode)
    assert np.allclose(m['coefficients'],rep['coefficients'])
    path=tmp_path/'model.json';g.save(m,path);loaded=g.load(path)
    assert g.estimate(m,rows[0]['query'])==g.estimate(loaded,rows[0]['query'])
    if mode=='structured':assert m['encoder']=={'terms':[],'idf':[]}
    if mode=='baseline':
        prior=old.fit([{**r,'query':r['query']['base']} for r in rows])
        assert m['coefficients']==prior['coefficients']
    assert g.estimate(m,rows[0]['query'])<0 # Topics do not carry hand-written positive value.


def test_structured_inference_ignores_title_words_when_routes_and_state_same():
    q=g.build_query(public('Joe Gomez fitness update'),snapshot())
    rows=[{'physical_match_id':'m','query':q,'signed_gain':.01}]
    m=g.fit(rows,'structured');other=deepcopy(q);other['base']['public']['title']='unseen vocabulary'
    assert g.estimate(m,other)==g.estimate(m,q)


def test_query_rejects_hidden_packet_fields():
    q=g.build_query(public(),snapshot());m=g.fit([{'physical_match_id':'m','query':q,'signed_gain':0.}])
    q['unread_body']='hidden'
    with pytest.raises(ValueError):g.estimate(m,q)
