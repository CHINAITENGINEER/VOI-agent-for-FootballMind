"""Unknown-value exploration preserves empirical and operational boundaries."""
from copy import deepcopy
from pathlib import Path
import runpy

import pytest

from football_agent.acquisition_boundary import CandidateBatch, materialize_choice, policy_identity
from football_agent import runtime_exploration as exploration
from football_agent import runtime_source_claims as claims
from football_agent.runtime_voi_model import RuntimeVoIModel
from football_agent.state_bound_runtime_policy import StateBoundVoISelectionPolicy


H = runpy.run_path(str(Path(__file__).with_name('test_runtime_source_claim_policy.py')))
SOURCE = H['SOURCE_HELPERS']
HOME = H['HOME_QUALIFICATION']
AWAY = 'away_reported_qualification_state'


def selector(config=None, *, count=2, cost=.3, allow=True):
    return StateBoundVoISelectionPolicy(
        RuntimeVoIModel(config if config is not None else H['conditional_package']()),
        allow_synthetic=allow,
        exploration={'max_actions': count, 'max_declared_cost': cost})


def public_for(tmp_path, policy, *, operations=(), accepted=()):
    _, public, _ = SOURCE['source_public'](tmp_path)
    return bind(public, policy, operations=operations, accepted=accepted)


def bind(public, policy, *, operations=(), accepted=()):
    public = deepcopy(public)
    public['selection_policy'] = policy_identity(policy)
    public['seconds_to_deadline'] = 300
    public[exploration.STATE_FIELD] = exploration.project_state(
        public['selection_policy'], list(operations), list(accepted), public['match_input'])
    return public


def menu(public, items=(), *, names=('a', 'b'), recommended=0):
    original = H['claim_batch'](public, list(items))
    by_name = {candidate.response['query'].rsplit(' ', 1)[-1]: candidate.response
               for candidate in original.candidates}
    return CandidateBatch.build(public, [by_name[name] for name in names],
                                origin='synthetic_exploration_test', recommended_index=recommended)


def choose(policy, batch):
    choice = policy.select(batch.context, batch)
    return choice, choice.record()['diagnostics']


def accepted_exploration(policy, public, *, operation_id='accepted-first'):
    batch = menu(public)
    choice, diagnostic = choose(policy, batch)
    assert diagnostic['selection_basis'] == exploration.BASIS
    response = materialize_choice(batch.context, batch, choice)
    record = exploration.reservation(public, response, choice.record(), operation_id)
    operation = {'kind': 'model', 'status': 'returned', 'operation_id': operation_id,
                 'decision_digest': record['decision_digest'], exploration.RESERVATION: record}
    return operation, response


def selected_route(choice, diagnostic):
    return diagnostic['state_bound_voi']['route_bindings'][choice.candidate_id]['route_id']


def test_default_v2_and_v3_policy_identities_remain_frozen():
    old = StateBoundVoISelectionPolicy(RuntimeVoIModel(H['HELPERS']['package']()), allow_synthetic=True)
    source = StateBoundVoISelectionPolicy(RuntimeVoIModel(H['claim_package']()), allow_synthetic=True)
    assert policy_identity(old)['fingerprint'] == '03a2ab15adb61ff435ebc5b94181b047fa186da2f16a0242ee1f5733c6dd3c3e'
    assert policy_identity(source)['fingerprint'] == '4364c407dbd415b0ac514b46b47ff4128b816d62896c30d3b24d215176a5b4c6'
    assert old.version == 'state-bound-runtime-policy-v2'
    assert source.version == 'state-bound-runtime-policy-v3'
    assert 'bounded_exploration' not in old.configuration
    assert 'bounded_exploration' not in source.configuration


def test_explicit_allowance_has_distinct_frozen_v4_identity():
    options = {'max_actions': 2, 'max_declared_cost': .3}
    policy = StateBoundVoISelectionPolicy(RuntimeVoIModel(H['claim_package']()),
                                        allow_synthetic=True, exploration=options)
    original = policy_identity(policy)
    options['max_actions'] = 16
    exposed = policy.configuration
    exposed['bounded_exploration']['max_declared_cost'] = 9
    assert policy_identity(policy) == original
    assert original['version'] == 'state-bound-runtime-policy-v4'
    assert original['configuration']['bounded_exploration']['selection_probability'] is None
    assert policy_identity(selector(H['claim_package'](), count=1)) != original
    assert policy_identity(selector(H['claim_package'](), cost=.2)) != original
    with pytest.raises(ValueError, match='MODEL_V2'):
        selector(H['HELPERS']['package']())


@pytest.mark.parametrize('partial', [False, True])
def test_fully_or_partly_unknown_source_context_can_explore_without_value_upgrade(tmp_path, partial):
    config = H['claim_package'](keys=[HOME, AWAY])
    policy = selector(config)
    public = public_for(tmp_path, policy)
    items = [SOURCE['claim'](public)] if partial else []
    before = deepcopy(public)
    batch = menu(public, items, names=('a',))
    choice, diagnostic = choose(policy, batch)
    assert choice.candidate_id == batch.candidates[0].candidate_id and choice.terminal_action is None
    assert diagnostic['selection_basis'] == exploration.BASIS
    value = diagnostic['selected_value']
    assert value['value_status'] == 'VALUE_UNMEASURED'
    assert value['expected_value'] is None and value['net_value'] is None
    assert diagnostic['query_value_decision'] is None
    assert diagnostic['bounded_exploration']['selection_probability'] is None
    observed = diagnostic['state_bound_voi']['projection']['context']
    assert observed[HOME] == ('already_qualified' if partial else 'unknown')
    assert observed[AWAY] == 'unknown'
    assert public == before


def test_cold_start_with_no_documents_can_explore_a_generated_registered_route():
    from football_agent.runtime_fixture_bridge import prepare_fixture_identity

    policy = selector()
    public = H['HELPERS']['public']()
    public[claims.CAPABILITY_FIELD] = claims.capability()
    public[claims.HISTORY_FIELD] = []
    public['fixture_identity'] = prepare_fixture_identity(None, public, [])
    public = bind(public, policy)
    batch = menu(public)
    choice, diagnostic = choose(policy, batch)
    assert diagnostic['selection_basis'] == exploration.BASIS
    assert choice.candidate_id in {candidate.candidate_id for candidate in batch.candidates}
    assert diagnostic['state_bound_voi']['projection']['context']['operation_document_count'] == 0
    assert diagnostic['selected_value']['expected_value'] is None
    assert public['documents'] == []


@pytest.mark.parametrize('support', ['no_history', 'insufficient_support', 'cross_cost'])
def test_unmeasured_empirical_support_can_use_declared_allowance(tmp_path, support):
    config = H['conditional_package']()
    if support == 'no_history':
        config['history'] = []
    elif support == 'insufficient_support':
        config['history'] = [row for row in config['history']
                             if int(row['link']['link_id'].rsplit('-', 1)[-1]) < 2]
    else:
        for row in config['history']:
            row['link']['observed_risk_reduction'] = .2
    policy = selector(config)
    public = public_for(tmp_path, policy)
    choice, diagnostic = choose(policy, menu(public, [SOURCE['claim'](public)]))
    assert diagnostic['selection_basis'] == exploration.BASIS
    interval = diagnostic['state_bound_voi']['intervals'][choice.candidate_id]
    if support == 'cross_cost':
        assert interval['support_status'] == 'supported_interval'
        assert interval['lower_bound'] <= interval['cost'] < interval['upper_bound']
    else:
        expected = 'no_matching_history' if support == 'no_history' else support
        assert interval['support_status'] == expected
    assert diagnostic['selected_value']['expected_value'] is None


def test_supported_positive_action_precedes_cheaper_unmeasured_route(tmp_path):
    config = H['conditional_package']()
    config['history'] = [row for row in config['history'] if row['link']['candidate_id'] == 'a']
    config['routes'][1]['cost'] = .001
    # The exploration-only cost cap must not suppress supported acquisition.
    policy = selector(config, cost=.01)
    public = public_for(tmp_path, policy)
    choice, diagnostic = choose(policy, menu(public, [SOURCE['claim'](public)], recommended=1))
    assert diagnostic['selection_basis'] == 'modeled_positive_net_value'
    assert selected_route(choice, diagnostic) == 'a'
    assert diagnostic['selected_value']['net_value'] > 0
    assert diagnostic['selected_value']['cost'] > .01


def test_supported_nonpositive_menu_keeps_model_scoped_stop(tmp_path):
    config = H['conditional_package']()
    for row in config['history']:
        row['link']['observed_risk_reduction'] = -.5
    policy = selector(config)
    public = public_for(tmp_path, policy)
    choice, diagnostic = choose(policy, menu(public, [SOURCE['claim'](public)]))
    assert choice.terminal_action == 'stop' and choice.candidate_id is None
    assert diagnostic['stop_kind'] == 'MODEL_SCOPED_STOP'
    assert diagnostic['query_value_decision']['stop_certificate']['maximum_modeled_net_value'] <= 0


def test_negative_route_stays_excluded_while_distinct_unknown_route_explores(tmp_path):
    config = H['conditional_package']()
    config['history'] = [row for row in config['history'] if row['link']['candidate_id'] == 'b']
    policy = selector(config)
    public = public_for(tmp_path, policy)
    batch = menu(public, [SOURCE['claim'](public)], recommended=1)
    choice, diagnostic = choose(policy, batch)
    assert diagnostic['selection_basis'] == exploration.BASIS
    assert selected_route(choice, diagnostic) == 'a'
    assert 'not_unmeasured' in diagnostic['bounded_exploration']['exclusions'][batch.candidates[1].candidate_id]


def test_unknown_alias_does_not_reopen_supported_nonpositive_information_group(tmp_path):
    config = H['conditional_package']()
    config['routes'][0]['action_id'] = config['routes'][1]['action_id']
    config['history'] = [row for row in config['history'] if row['link']['candidate_id'] == 'b']
    policy = selector(config)
    public = public_for(tmp_path, policy)
    batch = menu(public, [SOURCE['claim'](public)])
    choice, diagnostic = choose(policy, batch)
    assert choice.terminal_action == 'defer' and choice.candidate_id is None
    assert diagnostic['stop_kind'] == 'WITHHELD_OPEN_WORLD_VALUE_UNKNOWN'
    assert 'information_group_supported_nonpositive' in diagnostic['bounded_exploration']['exclusions'][batch.candidates[0].candidate_id]


def test_earlier_positive_acquisition_excludes_whole_group_from_later_exploration(tmp_path):
    config = H['conditional_package']()
    config['routes'][1]['action_id'] = config['routes'][0]['action_id']
    for row in config['history']:
        row['link']['action_id'] = 'discover-a'
    policy = selector(config)
    public = public_for(tmp_path, policy)
    first = menu(public, [SOURCE['claim'](public)])
    chosen, diagnostic = choose(policy, first)
    assert diagnostic['selection_basis'] == 'modeled_positive_net_value'
    response = materialize_choice(first.context, first, chosen)
    later = bind(public, policy, accepted=[response])
    assert later[exploration.STATE_FIELD]['reserved_actions'] == 0
    choice, diagnostic = choose(policy, menu(later, names=('b',)))
    assert choice.terminal_action == 'defer'
    reasons = next(iter(diagnostic['bounded_exploration']['exclusions'].values()))
    assert 'information_group_already_accepted' in reasons


def test_declared_cost_cap_is_independent_of_action_and_http_allowances(tmp_path):
    config = H['conditional_package']()
    config['routes'][1]['cost'] = .25
    policy = selector(config, count=2, cost=.3)
    public = public_for(tmp_path, policy)
    operation, response = accepted_exploration(policy, public)
    later = bind(public, policy, operations=[operation], accepted=[response])
    state = later[exploration.STATE_FIELD]
    assert state['remaining_actions'] == 1 and state['remaining_declared_cost'] == .2
    assert later['remaining']['http_calls'] > 0
    choice, diagnostic = choose(policy, menu(later, names=('b',)))
    assert choice.terminal_action == 'defer'
    assert 'declared_cost_allowance_exceeded' in next(iter(diagnostic['bounded_exploration']['exclusions'].values()))


def test_exact_remaining_decimal_cost_is_spendable(tmp_path):
    config = H['conditional_package']()
    config['routes'][1]['cost'] = .2
    policy = selector(config, count=2, cost=.3)
    public = public_for(tmp_path, policy)
    operation, response = accepted_exploration(policy, public)
    later = bind(public, policy, operations=[operation], accepted=[response])
    choice, diagnostic = choose(policy, menu(later, names=('b',)))
    assert diagnostic['selection_basis'] == exploration.BASIS
    assert selected_route(choice, diagnostic) == 'b'
    assert diagnostic['selected_value']['cost'] == .2


def test_terminal_only_proposals_do_not_invent_an_exploration_action(tmp_path):
    policy = selector()
    public = public_for(tmp_path, policy)
    response = H['HELPERS']['match_decision'](public, 'stop')
    response[claims.FIELD] = []
    batch = CandidateBatch.build(public, [response], origin='synthetic_terminal', recommended_index=0)
    choice, diagnostic = choose(policy, batch)
    assert choice.terminal_action == 'defer' and choice.candidate_id is None
    assert diagnostic['stop_kind'] == 'NO_FEASIBLE_REGISTERED_ACTION'
    assert diagnostic['selection_basis'] != exploration.BASIS


def test_action_cap_still_limits_zero_declared_cost_routes(tmp_path):
    config = H['conditional_package']()
    for route in config['routes']:
        route['cost'] = 0
    policy = selector(config, count=1)
    public = public_for(tmp_path, policy)
    operation, response = accepted_exploration(policy, public)
    later = bind(public, policy, operations=[operation], accepted=[response])
    assert later[exploration.STATE_FIELD]['remaining_declared_cost'] == .3
    choice, diagnostic = choose(policy, menu(later, names=('b',)))
    assert choice.terminal_action == 'defer'
    assert diagnostic['bounded_exploration']['status'] == 'exploration_action_budget_exhausted'


@pytest.mark.parametrize('boundary', ['http', 'final', 'deadline'])
def test_remaining_exploration_never_reopens_operational_limits(tmp_path, boundary):
    policy = selector()
    public = public_for(tmp_path, policy)
    if boundary == 'http':
        public['remaining']['http_calls'] = 0
    elif boundary == 'final':
        public['final_decision_only'] = True
    else:
        public['seconds_to_deadline'] = 0
    choice, diagnostic = choose(policy, menu(public))
    assert choice.terminal_action == 'defer' and choice.candidate_id is None
    if boundary != 'deadline':
        assert diagnostic['stop_kind'] == 'BUDGET_EXHAUSTED'
    assert diagnostic['selection_basis'] != exploration.BASIS


@pytest.mark.parametrize('equal_cost', [False, True])
def test_cost_then_route_selection_is_independent_of_llm_recommendation_and_order(tmp_path, equal_cost):
    config = H['conditional_package']()
    if not equal_cost:
        config['routes'][1]['cost'] = .05
    policy = selector(config)
    public = public_for(tmp_path, policy)
    selected = []
    for names in (('a', 'b'), ('b', 'a')):
        for recommendation in (0, 1):
            choice, diagnostic = choose(policy, menu(public, names=names, recommended=recommendation))
            selected.append(selected_route(choice, diagnostic))
            assert diagnostic['bounded_exploration']['selection_probability'] is None
    assert selected == [('a' if equal_cost else 'b')] * 4


@pytest.mark.parametrize('bad_route', ['unregistered', 'source_version'])
def test_undeclared_or_wrong_transport_route_is_not_explored(tmp_path, bad_route):
    config = H['conditional_package']()
    if bad_route == 'source_version':
        config['routes'][0]['source_version'] = 'old-transport'
        for row in config['history']:
            if row['link']['candidate_id'] == 'a':
                row['link']['source_version'] = 'old-transport'
    policy = selector(config)
    public = public_for(tmp_path, policy)
    batch = menu(public, names=('a',))
    if bad_route == 'unregistered':
        response = batch.candidates[0].response
        response['query'] += ' unregistered discovery'
        batch = CandidateBatch.build(public, [response], origin='synthetic_unregistered', recommended_index=0)
    choice, diagnostic = choose(policy, batch)
    assert choice.terminal_action == 'defer' and choice.candidate_id is None
    assert 'exact_registered_route_required' in diagnostic['bounded_exploration']['exclusions'][batch.candidates[0].candidate_id]


@pytest.mark.parametrize('blocked', ['synthetic', 'future_calibration', 'missing_capability'])
def test_exploration_preserves_model_deployment_and_clock_blocks(tmp_path, blocked):
    config = H['conditional_package']()
    if blocked == 'future_calibration':
        config['calibration_as_of'] = '2099-01-01T00:00:00Z'
    policy = selector(config, allow=blocked != 'synthetic')
    public = public_for(tmp_path, policy)
    if blocked == 'missing_capability':
        del public[claims.CAPABILITY_FIELD]
    choice, diagnostic = choose(policy, menu(public))
    assert choice.terminal_action == 'defer' and choice.candidate_id is None
    expected = {'synthetic': 'synthetic_history_disabled',
                'future_calibration': 'calibration_after_run_creation',
                'missing_capability': 'source_claim_capability_required'}[blocked]
    assert diagnostic['state_bound_voi']['blocked_reason'] == expected
    assert diagnostic['bounded_exploration']['status'] == 'base_policy_takes_precedence'
