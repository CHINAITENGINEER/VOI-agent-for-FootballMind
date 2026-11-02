from copy import deepcopy

import pytest

from football_agent.navigation_recovery import recovery_plan
from test_target_navigation import make, choice, query


def test_preserves_unknown_charge_goal_reference_and_reduces_budget(tmp_path):
    flow, rid, _, _, _, _ = make(tmp_path,[TimeoutError()])
    prior=flow.step(rid)
    before=deepcopy(prior)
    plan=recovery_plan(prior,task_id='explicit-recovery',configuration=flow.configuration())
    assert prior==before
    assert plan['limits']=={'max_llm_calls':4,'max_http_attempts':2,'max_window_reads':3}
    assert plan['prior_unknown_attempts_charged']+plan['limits']['max_llm_calls']==5
    assert plan['task_contract']['reference_time']==prior['target_contract']['reference_time']
    assert plan['task_contract']['question']==prior['target_contract']['question']
    assert plan['budget_refunded'] is False and prior['pending'] is not None


@pytest.mark.parametrize('change',['no_charge','wrong_pending','source_effect','changed_configuration','same_task','insufficient'])
def test_rejects_ambiguous_recovery_and_budget_reset(tmp_path,change):
    flow,rid,*_=make(tmp_path,[TimeoutError()])
    prior=flow.step(rid)
    config=flow.configuration()
    task_id='explicit-recovery'
    if change=='no_charge':prior['budget']['llm_calls']=0
    elif change=='wrong_pending':prior['pending']['kind']='http'
    elif change=='source_effect':prior['budget']['http_attempts']=1
    elif change=='changed_configuration':config['provider']['provider']='different-model'
    elif change=='same_task':task_id=prior['task']['task_id']
    else:prior['budget']['llm_calls']=4
    with pytest.raises(ValueError):recovery_plan(prior,task_id=task_id,configuration=config)


@pytest.mark.parametrize('mode',['voluntary_defer','source_unknown'])
def test_never_restarts_completed_choice_or_unknown_source_action(tmp_path,mode):
    flow,rid,*_=make(tmp_path,[choice if mode=='voluntary_defer' else query],source_error=TimeoutError())
    with pytest.raises(ValueError):
        recovery_plan(flow.step(rid),task_id='explicit-recovery',configuration=flow.configuration())
