"""Bounded adoption, truthful early stop and preserved frozen evaluation."""
from copy import deepcopy
import threading
from types import SimpleNamespace
import numpy as np
import pytest
from demo.defense_budget import run_with_limit, validate_limit


def backend(count=25,score=.1,fail_at=None):
    # Deliberately synthetic mechanism fixture, not a research dataset.
    b=SimpleNamespace(lock=threading.RLock(),np=np,completed_runs={},calls=[],f=SimpleNamespace(f_model_sha256='synthetic-f'))
    b.cases={'m':dict(physical_match_id='synthetic-match')}
    b.config=dict(models={name:dict(budget=3,threshold=.01,role='synthetic') for name in ['full','fixed_order','all_evidence']},
                  g_model_sha256='synthetic-g',encoder='synthetic',limitations=[])
    b.g=dict(fitted={'full':{'input':'full'}},embedding_dimension=1,target_mean=0,target_scale=1)
    b.verify=lambda:None
    b._panel=lambda case:SimpleNamespace(sequences={'synthetic-match':{'document_ids':[f'd{i}' for i in range(count)]}})
    b.candidate_features=lambda ev,mid,adopted,i,budget,dimension:(np.array([i,budget-len(adopted)]),1)
    b.predict_head=lambda head,x,mean,scale:np.full(len(x),score)
    class Evaluator:
        def __init__(self,model,panel):pass
        def predict(self,mid,adopted):
            b.calls.append(list(adopted))
            if fail_at is not None and len(adopted)>=fail_at:raise ValueError('synthetic f failure')
            delta=.001*len(adopted)
            return np.array([.4+delta,.3,.3-delta])
    b.FrozenEvaluator=Evaluator
    return b


@pytest.mark.parametrize('value',[0,21,-1,True,1.5,'20',None])
def test_reject_noninteger_or_out_of_range_limits(value):
    with pytest.raises(ValueError):validate_limit(value)


def test_twenty_adoptions_with_twenty_five_candidates_and_same_fixed_models():
    b=backend();config=deepcopy(b.config)
    r=run_with_limit(b,'m','full',20)
    assert r['body_adoptions']==20 and len(set(r['selected_ids']))==20
    assert len(r['unselected_ids'])==5 and r['stop_reason']=='budget_exhausted'
    assert r['budget']==r['requested_max_reads']==20 and r['budget_outside_training_range']
    assert b.config==config and r['threshold']==.01
    for i,step in enumerate(r['steps']):
        assert step['f_executed'] and step['used_input_ids_after']==[f'd{n}' for n in range(i+1)]
        assert step['p_after']==b.FrozenEvaluator(None,None).predict('synthetic-match',list(range(i+1))).tolist()


def test_twenty_does_not_invent_candidates_or_force_g_to_continue():
    b=backend(count=5,score=0)
    r=run_with_limit(b,'m','full',20)
    assert r['candidate_count']==5 and r['budget']==5 and r['requested_max_reads']==20
    assert r['selected_ids']==[] and len(r['unselected_ids'])==5
    assert r['stop_reason']=='predicted_gain_below_frozen_threshold'
    assert r['steps'][0]['p_after'] is None and not r['steps'][0]['f_executed']
    assert r['steps'][0]['used_input_ids_after'] is None


@pytest.mark.parametrize('policy',['fixed_order','all_evidence'])
def test_explicit_baselines_obey_same_cap(policy):
    r=run_with_limit(backend(),'m',policy,20)
    assert r['body_adoptions']==20 and not r['g_selection_executed']
    assert all(not step['candidate_scores'] for step in r['steps'])
    scarce=run_with_limit(backend(5),'m',policy,20)
    assert scarce['body_adoptions']==5 and scarce['stop_reason']=='pool_exhausted'


def test_failed_f_does_not_record_completed_run():
    b=backend(fail_at=2)
    with pytest.raises(ValueError,match='synthetic f failure'):run_with_limit(b,'m','full',20)
    assert not b.completed_runs


def test_original_three_budget_delegates_without_changing_scoring():
    b=backend(2);calls=[]
    def original(case_id,model_name):
        calls.append((case_id,model_name))
        return dict(run_id='original',candidate_count=2,budget=3,steps=[],p_after=[.4,.3,.3])
    b.run_case=original
    result=run_with_limit(b,'m','full',3)
    assert calls==[('m','full')] and not b.calls
    assert result['budget']==3 and result['effective_adoption_limit']==2
    assert result['original_frozen_evaluation'] is True
