from copy import deepcopy
import pytest
from football_research.acquisition import state_candidate_gain as old
from football_research.acquisition import historical_gain_head as raw
from football_research.acquisition import paired_state_gain as paired


def rows():
    result=[]
    for mid in ('a','b'):
        def q(read,title):
            return old.build_query({'schema':raw.QUERY_SCHEMA,'probabilities':[.4,.3,.3],
                'acquired_documents':read,'visible_candidates':2-read,'article_age_hours':1.,'title':title},
                {'physical_match_id':mid,'used_input_ids':[],'claims':[]})
        for state,next_state,action,gain,read,title in [
            ('s','t','first',.01,0,'preview'),('s','end','second',.03,0,'fitness'),
            ('t','end','second',-.01,1,'fitness')]:
            result.append({'physical_match_id':mid,'state_id':mid+state,'next_state_id':mid+next_state,
                           'action_id':action,'body_input_id':action,'signed_gain':gain,'query':q(read,title)})
    return result


def test_same_action_pairs_and_signed_delta():
    data=rows(); pairs=paired.build_pairs(data)
    assert len(pairs)==2
    assert all(p['action_id']=='second' and p['acquired_action_id']=='first' for p in pairs)
    assert all(p['gain_delta']==pytest.approx(-.04) for p in pairs)


def test_zero_pair_weight_exactly_recovers_previous_model():
    data=rows(); before=old.fit(data); after=paired.fit(data,pair_weight=0)
    assert before['coefficients']==after['coefficients'] and before['intercept']==after['intercept']


def test_pair_loss_changes_state_delta_and_roundtrip(tmp_path):
    data=rows(); a=paired.fit(data,pair_weight=0); b=paired.fit(data,pair_weight=1)
    def error(m):
        return sum((paired.estimate(m,data[p['after_index']]['query'])-
                    paired.estimate(m,data[p['before_index']]['query'])-p['gain_delta'])**2 for p in paired.build_pairs(data))
    assert error(b)<error(a)
    path=tmp_path/'model.json';paired.save(b,path)
    assert paired.load(path)==b


def test_cross_match_and_identity_corruption_rejected():
    data=rows();data[2]['physical_match_id']='wrong'
    with pytest.raises(ValueError,match='crosses'):paired.build_pairs(data)
    data=rows();data[2]['body_input_id']='different'
    with pytest.raises(ValueError,match='identity'):paired.build_pairs(data)


def test_duplicate_action_rejected():
    data=rows();data.append(deepcopy(data[0]))
    with pytest.raises(ValueError,match='Duplicate'):paired.build_pairs(data)


@pytest.mark.parametrize('field',['outcome','body','gain_after','next_state'])
def test_hidden_inference_fields_rejected(field):
    data=rows();model=paired.fit(data);query=deepcopy(data[0]['query']);query[field]='hidden'
    with pytest.raises(ValueError,match='contract'):paired.estimate(model,query)


def test_fit_preserves_input_and_rejects_negative_pair_weight():
    data=rows();original=deepcopy(data);paired.fit(data)
    assert data==original
    with pytest.raises(ValueError,match='pair weight'):paired.fit(data,pair_weight=-1)
