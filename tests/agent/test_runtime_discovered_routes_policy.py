"""New tool actions retain unknown value and frozen exploration boundaries."""
from copy import deepcopy
from pathlib import Path
import runpy

import pytest

from football_agent.acquisition_boundary import CandidateBatch, materialize_choice, policy_identity
from football_agent import runtime_discovered_routes as discovery
from football_agent import runtime_exploration as exploration
from football_agent.runtime_voi_model import RuntimeVoIModel
from football_agent.state_bound_runtime_policy import StateBoundVoISelectionPolicy


E = runpy.run_path(str(Path(__file__).with_name('test_runtime_exploration_policy.py')))
V, SOURCE = E['H'], E['SOURCE']
public_for, bind, menu, choose = (E[name] for name in ('public_for', 'bind', 'menu', 'choose'))
PROFILE = {'search_web': {'bing': .1}, 'read_url': .1}
URL = 'https://example.org/new-match-report'


def policy(config=None, *, count=3, cost=.3, profile=None):
    return StateBoundVoISelectionPolicy(
        RuntimeVoIModel(V['conditional_package']() if config is None else config),
        allow_synthetic=True,
        exploration={'max_actions': count, 'max_declared_cost': cost},
        discovered_routes=deepcopy(PROFILE) if profile is None else profile)


def response(public, *, query='New official fixture report', url=None, items=()):
    result = menu(public, items, names=('a',)).candidates[0].response
    result['query'] = query
    if url is not None:
        result.update(action='read_url', query=None, engine=None, url=url)
    return result


def candidates(public, *responses, recommended=0):
    return CandidateBatch.build(public, list(responses), origin='synthetic_discovered_route_test',
                                recommended_index=recommended)


def accepted(public, batch, selected, *, operation_id='accepted-discovery'):
    result = materialize_choice(batch.context, batch, selected)
    record = exploration.reservation(public, result, selected.record(), operation_id, batch=batch)
    operation = {'kind': 'model', 'status': 'returned', 'operation_id': operation_id,
                 'decision_digest': record['decision_digest'], exploration.RESERVATION: record}
    return operation, result, record


def test_new_search_explores_without_borrowing_empirical_gain(tmp_path):
    selector = policy()
    public = public_for(tmp_path, selector)
    before = deepcopy(public)
    batch = candidates(public, response(public))
    selected, diagnostic = choose(selector, batch)
    cid = batch.candidates[0].candidate_id
    voi = diagnostic['state_bound_voi']
    assert selected.candidate_id == cid and selected.terminal_action is None
    assert diagnostic['selection_basis'] == exploration.BASIS
    assert voi['route_bindings'][cid]['status'] == discovery.BOUND
    assert cid not in voi['intervals']
    assert voi['discovered_routes'][cid]['empirical_support'] is False
    assert voi['discovered_routes'][cid]['semantic_relevance'] == 'unassessed'
    assert voi['discovered_routes'][cid]['origin'] == {'kind': 'search_tool'}
    value = diagnostic['selected_value']
    assert value['value_status'] == 'VALUE_UNMEASURED'
    assert value['expected_value'] is None and value['net_value'] is None
    assert value['cost'] == .1 and diagnostic['query_value_decision'] is None
    _, _, record = accepted(public, batch, selected)
    assert record['declared_cost'] == .1 and record['realized_information_gain'] is None
    assert record['numeric_voi_training_eligible'] is False
    assert public == before


def test_new_body_read_binds_exact_observed_link(tmp_path):
    selector = policy()
    archive, public, originals = SOURCE['source_public'](tmp_path)
    document = public['documents'][-1]
    document['links'] = [{'url': URL, 'title': 'Official match report'}]
    originals[-1]['links'] = deepcopy(document['links'])
    SOURCE['rebind'](archive, public, originals)
    public = bind(public, selector)
    batch = candidates(public, response(public, url=URL))
    selected, diagnostic = choose(selector, batch)
    route = diagnostic['state_bound_voi']['discovered_routes'][selected.candidate_id]
    assert route['request'] == {'action': 'read_url', 'url': URL}
    assert route['origin'] == {'kind': 'observed_link', 'document_id': document['document_id'],
                               'link_index': 0, 'url': URL}
    assert route['source_id'] == 'web:example.org'
    assert diagnostic['selected_value']['expected_value'] is None
    _, _, record = accepted(public, batch, selected)
    assert record['discovered_route'] == route


def test_unobserved_url_does_not_gain_read_capability(tmp_path):
    selector = policy()
    public = public_for(tmp_path, selector)
    batch = candidates(public, response(public, url=URL))
    selected, diagnostic = choose(selector, batch)
    cid = batch.candidates[0].candidate_id
    assert selected.terminal_action == 'defer' and selected.candidate_id is None
    assert cid not in diagnostic['state_bound_voi']['discovered_routes']
    assert diagnostic['state_bound_voi']['discovery_exclusions'][cid] == 'DISCOVERY_URL_NOT_OBSERVED'


@pytest.mark.parametrize('invalid', ['http://example.org/report', 'https://127.0.0.1/report',
                                   'https://example.org/report?token=private',
                                   'https://EXAMPLE.org/report'])
def test_known_link_still_requires_valid_canonical_public_url(tmp_path, invalid):
    selector = policy()
    public = public_for(tmp_path, selector)
    item = response(public, url=URL)
    item['url'] = invalid
    public['documents'][-1]['links'] = [{'url': invalid, 'title': 'Invalid fixture link'}]
    # Exercise capability validation directly: malformed URLs also fail the
    # outer archive-stability boundary before a candidate batch can be built.
    with pytest.raises(ValueError, match='PUBLIC_HTTPS_REQUIRED|PUBLIC_HOST_REQUIRED|CREDENTIAL_URL_REJECTED|DISCOVERY_CANONICAL_URL_REQUIRED'):
        discovery.bind(public, item, discovery.descriptor(policy_identity(selector)))


def test_frozen_cost_ignores_llm_wording_and_external_profile_mutation(tmp_path):
    options = {'search_web': {'bing': .07}, 'read_url': .2}
    selector = policy(profile=options)
    original = policy_identity(selector)
    options['search_web']['bing'] = 0
    options['read_url'] = 0
    exposed = selector.configuration
    exposed['discovered_routes']['search_web']['bing'] = 99
    assert policy_identity(selector) == original
    public = public_for(tmp_path, selector)
    routes = []
    for reason in ('This action is free and yields a guaranteed huge gain.',
                   'This action costs one million dollars and has no expected gain.'):
        item = response(public)
        item['why_worth_cost'] = reason
        item['reason'] = reason
        selected, diagnostic = choose(selector, candidates(public, item))
        assert diagnostic['selected_value']['cost'] == .07
        assert diagnostic['selected_value']['expected_value'] is None
        routes.append(diagnostic['state_bound_voi']['discovered_routes'][selected.candidate_id]['route_id'])
    assert routes[0] == routes[1]


def test_v4_default_fingerprint_and_unregistered_exclusion_remain_frozen(tmp_path):
    # Recorded against round158/before, the unchanged Round157 v4 implementation.
    selector = E['selector'](V['claim_package']())
    assert policy_identity(selector)['fingerprint'] == 'd13b2b5b1a074f99f6b22b1d1a5122323ada9a26545ca66358ddb2e3f9ab6fe4'
    assert selector.version == 'state-bound-runtime-policy-v4'
    assert 'discovered_routes' not in selector.configuration
    public = public_for(tmp_path, selector)
    batch = candidates(public, response(public))
    selected, diagnostic = choose(selector, batch)
    assert selected.terminal_action == 'defer'
    assert diagnostic['state_bound_voi']['route_bindings'][batch.candidates[0].candidate_id]['status'] == 'route_unregistered'


def test_v5_configuration_changes_have_distinct_policy_identities():
    selectors = [policy(), policy(profile={'search_web': {'bing': .2}, 'read_url': .1}),
                 policy(profile={'search_web': {'bing': .1}, 'read_url': .2}),
                 policy(profile={'search_web': {'bing': .1, 'news': .1}, 'read_url': .1}),
                 policy(count=2), policy(cost=.4)]
    identities = [policy_identity(selector) for selector in selectors]
    assert all(identity['version'] == 'state-bound-runtime-policy-v5' for identity in identities)
    assert len({identity['fingerprint'] for identity in identities}) == len(identities)
    assert discovery.descriptor(identities[0]) == selectors[0].configuration['discovered_routes']
    assert exploration.descriptor(identities[0])['scope'] == 'generated_registered_or_capability_bound_unmeasured_routes'


def test_discovery_requires_explicit_exploration():
    with pytest.raises(ValueError, match='DISCOVERY_REQUIRES_EXPLICIT_EXPLORATION'):
        StateBoundVoISelectionPolicy(RuntimeVoIModel(V['claim_package']()),
                                    allow_synthetic=True, discovered_routes=PROFILE)


@pytest.mark.parametrize('profile', [
    {}, {'search_web': {'bing': .1}},
    {'search_web': {'bing': .1}, 'read_url': .1, 'extra': 1},
    {'search_web': {'unknown_engine': .1}, 'read_url': .1},
    {'search_web': [], 'read_url': .1},
    {'search_web': {}, 'read_url': None},
    {'search_web': {'bing': True}, 'read_url': .1},
    {'search_web': {'bing': -.1}, 'read_url': .1},
    {'search_web': {'bing': float('nan')}, 'read_url': .1},
    {'search_web': {'bing': float('inf')}, 'read_url': .1},
    {'search_web': {'bing': 10 ** 400}, 'read_url': .1},
    {'search_web': {'bing': '0.1'}, 'read_url': .1},
    {'search_web': {'bing': .1}, 'read_url': False},
    {'search_web': {'bing': .1}, 'read_url': -.1},
    {'search_web': {'bing': .1}, 'read_url': float('nan')},
    {'search_web': {'bing': .1}, 'read_url': 10 ** 400},
])
def test_invalid_cost_profile_is_rejected(profile):
    with pytest.raises(ValueError, match='DISCOVERY_'):
        policy(profile=profile)


@pytest.mark.parametrize('profile', [{'search_web': {}, 'read_url': 0},
                                   {'search_web': {'bing': 0}, 'read_url': None}])
def test_zero_cost_and_explicitly_disabled_tools_are_valid(profile):
    selector = policy(profile=profile)
    assert discovery.descriptor(policy_identity(selector)) is not None


def test_read_cost_none_disables_new_reads_even_for_known_catalog_url(tmp_path):
    selector = policy(profile={'search_web': {'bing': .1}, 'read_url': None})
    public = public_for(tmp_path, selector)
    url = public['source_catalog']['entries'][0]['url']
    batch = candidates(public, response(public, url=url))
    selected, diagnostic = choose(selector, batch)
    assert selected.terminal_action == 'defer'
    assert diagnostic['state_bound_voi']['discovery_exclusions'][batch.candidates[0].candidate_id] == 'DISCOVERY_READ_COST_UNDECLARED'


@pytest.mark.parametrize('invalid', ['query_blank', 'query_too_long', 'engine_unknown',
                                   'engine_without_cost', 'closed_question', 'unknown_question'])
def test_new_action_requires_executable_tool_and_open_question(tmp_path, invalid):
    selector = policy()
    public = public_for(tmp_path, selector)
    item = response(public)
    if invalid == 'query_blank': item['query'] = '  \t '
    elif invalid == 'query_too_long': item['query'] = 'x' * 361
    elif invalid == 'engine_unknown': item['engine'] = 'invented'
    elif invalid == 'engine_without_cost': item['engine'] = 'news'
    elif invalid == 'closed_question': item['analysis_plan'][0]['status'] = 'supported'
    else: item['active_question_id'] = 'question-not-in-plan'
    batch = candidates(public, item)
    selected, diagnostic = choose(selector, batch)
    assert selected.terminal_action == 'defer'
    assert not diagnostic['state_bound_voi']['discovered_routes']
    assert batch.candidates[0].candidate_id in diagnostic['state_bound_voi']['discovery_exclusions']


def test_registered_nonpositive_query_does_not_block_different_new_information(tmp_path):
    selector = policy()
    public = public_for(tmp_path, selector)
    batch = candidates(public, response(public, items=[SOURCE['claim'](public)]))
    selected, diagnostic = choose(selector, batch)
    cid = batch.candidates[0].candidate_id
    voi = diagnostic['state_bound_voi']
    assert selected.candidate_id == cid and selected.terminal_action is None
    assert len(batch.candidates) == 1 and voi['route_bindings'][cid]['status'] == discovery.BOUND
    assert any(guard['registered_route_id'] == 'b' for guard in voi['discovery_action_guards'])
    assert voi['registered_route_intervals']['b']['upper_bound'] <= .1
    assert diagnostic['selection_basis'] == exploration.BASIS
    assert cid not in diagnostic['bounded_exploration']['exclusions']
    value = diagnostic['selected_value']
    assert value['value_status'] == 'VALUE_UNMEASURED' and value['net_value'] is None


def test_registered_nonpositive_action_absent_from_menu_guards_normalized_alias(tmp_path):
    selector = policy()
    public = public_for(tmp_path, selector)
    items = [SOURCE['claim'](public)]
    known = menu(public, items, names=('b',)).candidates[0].response
    alias = response(public, query='  ' + known['query'].replace('source b', 'ＳＯＵＲＣＥ   Ｂ') + '  ', items=items)
    assert discovery.action_key(alias) == discovery.action_key(known)
    batch = candidates(public, alias)
    selected, diagnostic = choose(selector, batch)
    cid = batch.candidates[0].candidate_id
    voi = diagnostic['state_bound_voi']
    assert selected.terminal_action == 'defer'
    assert voi['route_bindings'][cid]['status'] == discovery.BOUND
    assert any(guard['registered_route_id'] == 'b' and guard['action_key'] == discovery.action_key(alias)
               for guard in voi['discovery_action_guards'])
    assert 'registered_nonpositive_action_guard' in diagnostic['bounded_exploration']['exclusions'][cid]
    value = next(row['value'] for row in diagnostic['candidates'] if row['candidate_id'] == cid)
    assert value['value_status'] == 'VALUE_UNMEASURED' and value['net_value'] is None


def test_nonpositive_registered_template_over_runtime_query_limit_supplies_no_guard():
    route = deepcopy(next(route for route in V['conditional_package']()['routes']
                          if route['route_id'] == 'b'))
    route['match']['query_template'] = '{match_input} ' + 'x' * 350
    match_input = 'A sufficiently long match label'
    request = discovery.registered_request_payload(route, match_input)
    assert len(request['query']) > 360
    intervals = {'b': {'support_status': 'supported_interval', 'upper_bound': 0.0}}
    assert discovery.registered_guards([route], intervals, match_input) == []


@pytest.mark.parametrize('mismatch', ['source_version', 'source_id'])
def test_wrong_transport_or_source_negative_history_does_not_guard_new_capability(tmp_path, mismatch):
    config = V['conditional_package']()
    value = 'old-transport' if mismatch == 'source_version' else 'search:duckduckgo'
    next(route for route in config['routes'] if route['route_id'] == 'b')[mismatch] = value
    for row in config['history']:
        if row['link']['candidate_id'] == 'b': row['link'][mismatch] = value
    selector = policy(config)
    public = public_for(tmp_path, selector)
    items = [SOURCE['claim'](public)]
    known = menu(public, items, names=('b',)).candidates[0].response
    batch = candidates(public, response(public, query='  ' + known['query'].upper() + '  ', items=items))
    selected, diagnostic = choose(selector, batch)
    assert diagnostic['state_bound_voi']['registered_route_intervals']['b']['upper_bound'] <= .1
    assert diagnostic['state_bound_voi']['discovery_action_guards'] == []
    assert selected.candidate_id == batch.candidates[0].candidate_id
    assert diagnostic['selection_basis'] == exploration.BASIS


def test_registered_positive_candidate_precedes_cheaper_dynamic_choice(tmp_path):
    selector = policy(profile={'search_web': {'bing': .001}, 'read_url': .001})
    public = public_for(tmp_path, selector)
    items = [SOURCE['claim'](public)]
    static = menu(public, items, names=('a',)).candidates[0].response
    batch = candidates(public, response(public, items=items), static, recommended=0)
    selected, diagnostic = choose(selector, batch)
    assert selected.candidate_id == batch.candidates[1].candidate_id
    assert diagnostic['selection_basis'] == 'modeled_positive_net_value'
    assert diagnostic['selected_value']['net_value'] > 0
    assert exploration.reservation(public, static, selected.record(), 'positive', batch=batch) is None


@pytest.mark.parametrize('boundary', ['ambiguous', 'source_version', 'source_id'])
def test_existing_exact_match_is_never_rebound_as_dynamic_on_failure(tmp_path, boundary):
    config = V['conditional_package']()
    public = public_for(tmp_path, policy(config))
    if boundary == 'ambiguous':
        config['routes'][1]['match']['query_template'] = public['match_input'] + ' source a'
    else:
        value = 'old-transport' if boundary == 'source_version' else 'search:duckduckgo'
        config['routes'][0][boundary] = value
        for row in config['history']:
            if row['link']['candidate_id'] == 'a': row['link'][boundary] = value
    selector = policy(config)
    public = bind(public, selector)
    batch = menu(public, names=('a',))
    selected, diagnostic = choose(selector, batch)
    cid = batch.candidates[0].candidate_id
    assert selected.terminal_action == 'defer'
    assert cid not in diagnostic['state_bound_voi']['discovered_routes']
    expected = 'ambiguous_route' if boundary == 'ambiguous' else 'runtime_source_route_mismatch'
    assert diagnostic['state_bound_voi']['route_bindings'][cid]['status'] == expected


def test_normalized_dynamic_action_key_covers_prior_static_positive_and_new_question(tmp_path):
    selector = policy()
    public = public_for(tmp_path, selector)
    first = menu(public, [SOURCE['claim'](public)], names=('a',))
    selected, first_diagnostic = choose(selector, first)
    assert first_diagnostic['selection_basis'] == 'modeled_positive_net_value'
    prior = materialize_choice(first.context, first, selected)
    later = bind(public, selector, accepted=[prior])
    rewritten = response(later, query='  ' + '   '.join(prior['query'].upper().split()) + '  ')
    rewritten['analysis_plan'] = [{'id': 'new_question', 'question': 'Different framing of the unknown',
                                  'decision_relevance': 'Another analysis consequence', 'status': 'open'}]
    rewritten['active_question_id'] = 'new_question'
    batch = candidates(later, rewritten)
    selected, diagnostic = choose(selector, batch)
    assert later[exploration.STATE_FIELD]['reserved_actions'] == 0
    assert discovery.action_key(prior) in later[exploration.STATE_FIELD]['used_discovered_action_keys']
    assert selected.terminal_action == 'defer'
    assert 'normalized_action_already_accepted' in diagnostic['bounded_exploration']['exclusions'][batch.candidates[0].candidate_id]
    assert diagnostic['state_bound_voi']['discovery_action_guards'] == []


def test_prior_dynamic_alias_blocks_later_exact_registered_unmeasured_action(tmp_path):
    selector = policy()
    public = public_for(tmp_path, selector)
    exact = menu(public, names=('a',)).candidates[0].response
    alias = response(public, query='  ' + exact['query'].upper() + '  ')
    batch = candidates(public, alias)
    selected, diagnostic = choose(selector, batch)
    assert diagnostic['selection_basis'] == exploration.BASIS
    assert diagnostic['state_bound_voi']['route_bindings'][selected.candidate_id]['status'] == discovery.BOUND
    operation, prior, _ = accepted(public, batch, selected)
    later = bind(public, selector, operations=[operation], accepted=[prior])
    batch = menu(later, names=('a',))
    selected, diagnostic = choose(selector, batch)
    cid = batch.candidates[0].candidate_id
    assert discovery.action_key(prior) == discovery.action_key(batch.candidates[0].response)
    assert later[exploration.STATE_FIELD]['reserved_actions'] == 1
    assert later[exploration.STATE_FIELD]['remaining_actions'] > 0
    assert later[exploration.STATE_FIELD]['used_route_groups'] == []
    assert diagnostic['state_bound_voi']['route_bindings'][cid]['status'] == 'exact_route'
    assert selected.terminal_action == 'defer'
    assert 'normalized_action_already_accepted' in diagnostic['bounded_exploration']['exclusions'][cid]


def test_accepted_dynamic_search_stays_used_after_question_and_unicode_rewrite(tmp_path):
    selector = policy()
    public = public_for(tmp_path, selector)
    batch = candidates(public, response(public, query='Official team report'))
    selected, _ = choose(selector, batch)
    operation, prior, _ = accepted(public, batch, selected)
    later = bind(public, selector, operations=[operation], accepted=[prior])
    rewritten = response(later, query='  ＯＦＦＩＣＩＡＬ   team\tREPORT  ')
    rewritten['analysis_plan'][0].update(id='different_id', question='A rewritten question')
    rewritten['active_question_id'] = 'different_id'
    batch = candidates(later, rewritten)
    selected, diagnostic = choose(selector, batch)
    assert discovery.action_key(rewritten) == discovery.action_key(prior)
    assert selected.terminal_action == 'defer'
    assert 'normalized_action_already_accepted' in diagnostic['bounded_exploration']['exclusions'][batch.candidates[0].candidate_id]


def test_semantic_rewrites_with_distinct_payloads_still_share_total_allowance(tmp_path):
    selector = policy(count=2, cost=.2)
    public = public_for(tmp_path, selector)
    operations, turns, keys = [], [], []
    for index, query in enumerate(('Official lineup announcement', 'Confirmed starting eleven publication')):
        snapshot = bind(public, selector, operations=operations, accepted=turns)
        batch = candidates(snapshot, response(snapshot, query=query))
        selected, diagnostic = choose(selector, batch)
        assert diagnostic['selection_basis'] == exploration.BASIS
        operation, item, _ = accepted(snapshot, batch, selected, operation_id=f'accepted-{index}')
        operations.append(operation)
        turns.append(item)
        keys.append(discovery.action_key(item))
    assert keys[0] != keys[1]
    later = bind(public, selector, operations=operations, accepted=turns)
    batch = candidates(later, response(later, query='Which players will start this fixture'))
    selected, diagnostic = choose(selector, batch)
    assert selected.terminal_action == 'defer'
    assert diagnostic['bounded_exploration']['status'] == 'exploration_action_budget_exhausted'
    assert later[exploration.STATE_FIELD]['reserved_declared_cost'] == .2


def test_dynamic_reservation_requires_complete_candidate_batch(tmp_path):
    selector = policy()
    public = public_for(tmp_path, selector)
    batch = candidates(public, response(public), response(public, query='Another new report'))
    selected, _ = choose(selector, batch)
    item = materialize_choice(batch.context, batch, selected)
    with pytest.raises(ValueError, match='DISCOVERY_REPLAY_CANDIDATE_BATCH_REQUIRED'):
        exploration.reservation(public, item, selected.record(), 'missing-batch')
    with pytest.raises(ValueError, match='DISCOVERY_RESERVATION_BINDING_MISMATCH'):
        exploration.reservation(public, item, selected.record(), 'partial-batch',
                                batch=candidates(public, item))
    _, _, record = accepted(public, batch, selected)
    assert record['declared_cost'] == .1


@pytest.mark.parametrize('mutation', ['cost', 'origin', 'question', 'remove_unselected'])
def test_reservation_rebuilds_dynamic_bindings_instead_of_trusting_diagnostics(tmp_path, mutation):
    selector = policy()
    public = public_for(tmp_path, selector)
    batch = candidates(public, response(public), response(public, query='Another new report'))
    selected, _ = choose(selector, batch)
    item = materialize_choice(batch.context, batch, selected)
    forged = selected.record()
    routes = forged['diagnostics']['state_bound_voi']['discovered_routes']
    route = routes[selected.candidate_id]
    if mutation == 'cost': route['cost'] = 0
    elif mutation == 'origin': route['origin'] = {'kind': 'invented'}
    elif mutation == 'question': route['question']['question'] = 'Invented retrospective question'
    else:
        other = next(cid for cid in routes if cid != selected.candidate_id)
        del routes[other]
    with pytest.raises(ValueError, match='DISCOVERY_RESERVATION_BINDING_MISMATCH'):
        exploration.reservation(public, item, forged, 'forged', batch=batch)
