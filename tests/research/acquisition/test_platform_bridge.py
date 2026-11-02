"""Mechanism tests use public protocol types; no synthetic match score claims."""
from copy import deepcopy
import json
import pytest

from football_platform.information import VisibleState, EvidenceGroup, EvidenceTimes, CostVector, ActionSpec
from football_research.acquisition.platform_bridge import experiment_actions, to_research_actions, to_research_state
from football_research.acquisition.policy import select_or_stop


def visible():
    return VisibleState(episode_id='test-episode', match_id='test-match',
        prediction_cutoff='2025-03-02T00:00:00Z', information_mode='historical_reconstruction',
        fixture={'match_id':'test-match','competition_id':'league','home_team_id':'h','away_team_id':'a',
                 'kickoff':'2025-03-03T00:00:00Z'}, base_prior={'home':1/3,'draw':1/3,'away':1/3},
        groups={}, mask={'strength':'UNACQUIRED','recent_form':'UNACQUIRED','rolling_xg':'UNACQUIRED'},
        catalog_version='catalog-v1',cost_config_version='cost-v1')


def strength():
    return EvidenceGroup(group_id='strength',values={'home_elo':1500.,'away_elo':1480.,'elo_difference':20.},
        source='test', times=EvidenceTimes(feature_available_at='2025-03-01T00:00:00Z',
            observed_at='2026-09-22T00:00:00Z', basis='historical_assumption'),
        quality='VALIDATED', version='features-v1', evidence_refs=['observation:test','D:/private.json'])


def test_config_and_action_costs_bind_joint_public_groups_once():
    actions, manifest = experiment_actions()
    assert [a['group_ids'] for a in actions] == [['elo','recent_form'],['xg']]
    assert [a['cost']['money'] for a in actions] == [.02,.03]
    assert all(a['cost'] == {'requests':1.,'money':price,'latency_ms':0.,'tokens':0.}
               for a,price in zip(actions,(.02,.03)))
    assert all(a['cost_status']=='simulated' for a in actions)
    assert manifest['config']['bindings']['results_view']==['strength','recent_form']
    assert manifest['lambda_money_unit']=='log_loss_nats_per_simulation_credit'
    assert experiment_actions()==(actions,manifest)


def test_hidden_group_labels_and_manifest_do_not_change_runtime_or_action():
    raw=visible().model_dump(mode='json')
    before=to_research_state(raw)
    actions,_=experiment_actions()
    estimates=[{'action':action,'gain':.1,'lower_gain':.05,'support_matches':30} for action in actions]
    first=select_or_stop(before,estimates,{'money':.05},{'money':1.})
    raw['groups']={'rolling_xg':{'values':{'home_rolling_xg_for':float('inf')},'times':'hidden invalid'}}
    raw['label']='away';raw['input_manifest']={'path':'private.json'};raw['fixture']['home_goals']=999
    raw['snapshot_sha256']='hidden fingerprint'
    assert to_research_state(raw)==before
    assert select_or_stop(to_research_state(raw),estimates,{'money':.05},{'money':1.})==first
    encoded=json.dumps(before)
    assert 'private' not in encoded and 'snapshot' not in encoded and 'home_goals' not in encoded


def test_observed_group_allowlist_and_historical_clock_distinction():
    state=visible()
    state.groups['strength']=strength();state.mask['strength']='OBSERVED'
    projected=to_research_state(state)
    group=projected['groups']['elo']
    assert group['values']=={'home_elo':1500.,'away_elo':1480.}
    assert group['evidence_refs']==['observation:test']
    assert projected['mask']==['elo']


def test_operational_future_ingestion_is_rejected():
    state=visible();state.information_mode='operational'
    state.groups['strength']=strength();state.mask['strength']='OBSERVED'
    with pytest.raises(ValueError,match='unavailable'):
        to_research_state(state)


def test_failed_mask_preserves_reason_and_history_for_bounded_retry():
    raw=visible().model_dump(mode='json')
    raw['mask']['strength']='FAILED'
    raw['history']=[{'action_id':'results_view','request_id':'request-1','status':'TIMEOUT',
                     'group_ids':['strength','recent_form'],'reason':'timeout','sequence':1}]
    raw['groups']['strength']={'values':{'home_elo':999999},'reason':'private/hidden'}
    state=to_research_state(raw)
    assert state['groups']['elo']=={'status':'source_missing','values':{},'reason':'timeout'}
    assert state['action_history'][0]['action_id']=='results_view'
    actions,_=experiment_actions()
    decision=select_or_stop(state,[{'action':actions[0],'gain':.1,'support_matches':30}],{}, {'money':1})
    assert decision['reason']=='attempt_limit'


@pytest.mark.parametrize('quality,status',[('STALE','stale'),('INVALID','source_missing'),('SOURCE_MISSING','source_missing')])
def test_bad_or_stale_quality_exposes_no_numeric_values(quality,status):
    state=visible();group=strength();group.quality=quality
    state.groups['strength']=group;state.mask['strength']='OBSERVED'
    result=to_research_state(state)['groups']['elo']
    assert result['status']==status and result['values']=={}


def test_unknown_cost_and_unconfigured_capability_remain_explicit():
    _,manifest=experiment_actions()
    action=deepcopy(manifest['catalog'][0]);action['expected_cost']=CostVector().model_dump(mode='json')
    action['capability_status']='NEEDS_CONFIGURATION'
    action['future_missing']=True;action['values']={'home_elo':1234}
    mapped=to_research_actions([action])[0]
    assert not mapped['available'] and mapped['cost_status']=='unknown'
    assert all(value is None for value in mapped['cost'].values())
    assert 'values' not in mapped and 'future_missing' not in mapped


def test_invalid_group_identity_is_rejected():
    state=visible();group=strength();group.group_id='rolling_xg'
    state.groups['strength']=group;state.mask['strength']='OBSERVED'
    with pytest.raises(ValueError,match='identity'):
        to_research_state(state)


def test_platform_policy_boundary_only_accepts_serialized_public_input():
    from football_research.acquisition.platform_demo import decide_public
    _, manifest = experiment_actions()
    state = visible().model_dump(mode='json')

    class Partial:
        def predict_partial(self, projected):
            assert not any(key in projected for key in ('label', 'manifest', 'snapshot_sha256'))
            return {'probabilities': [.4,.3,.3]}

    class Gain:
        def estimate(self, projected, actions, prediction):
            assert prediction == {'probabilities': [.4,.3,.3]}
            return [{'action':action, 'gain':-.01, 'lower_gain':-.02, 'support_matches':30}
                    for action in actions]

    with pytest.raises(TypeError, match='serialized JSON'):
        decide_public(state, json.dumps(manifest['catalog']), Partial(), Gain(), 1.)
    first = decide_public(json.dumps(state),json.dumps(manifest['catalog']),Partial(),Gain(),1.)
    assert first['decision']['reason']=='nonpositive_net_gain'
    state['label']='home';state['manifest']={'private_path':'hidden'}
    state['groups']['rolling_xg']={'values':{'home_rolling_xg_for':9999}}
    second=decide_public(json.dumps(state),json.dumps(manifest['catalog']),Partial(),Gain(),1.)
    assert first['prediction']==second['prediction']
    assert first['decision']==second['decision']
