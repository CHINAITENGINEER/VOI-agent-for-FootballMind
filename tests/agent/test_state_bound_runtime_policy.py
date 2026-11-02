"""End-to-end synthetic support; no real source or independent gain labels."""
from copy import deepcopy
import json
from pathlib import Path
import runpy

import pytest

from football_agent.acquisition_boundary import CandidateBatch, materialize_choice, policy_identity
from football_agent.acquisition_replay import verify_run
from football_agent.experience import ExperienceService, now
from football_agent.experience_sources import project_body, VERSION as TRANSPORT_VERSION
from football_agent.runtime_voi_model import RuntimeVoIModel, VERSION as MODEL_VERSION
from football_agent.state_bound_runtime_policy import StateBoundVoISelectionPolicy, observed_runtime_context
from football_agent.voi_evidence_bridge import VoIEvidenceLink

HELPERS = runpy.run_path(str(Path(__file__).with_name('test_match_return_assessment.py')))
match_decision, actual_review = HELPERS['match_decision'], HELPERS['actual_review']


def package(cells=(('a', 0, .9),), *, keys=None):
    keys = keys or ['operation_document_count']
    routes = [{'route_id': name, 'match': {'action': 'search_web', 'engine': 'bing',
               'query_template': '{match_input} source ' + name}, 'source_id': 'search:bing',
               'source_version': TRANSPORT_VERSION, 'action_id': 'discover-' + name,
               'cost': .1, 'unit': 'decision_loss'} for name in sorted({x[0] for x in cells})]
    history = []
    for cell, (name, count, gain) in enumerate(cells):
        for i in range(60):
            key = f'{cell}-{i}'
            link = VoIEvidenceLink(link_id='link-' + key, trace_id='trace-' + key,
                episode_id='episode-' + key, target_key=f'spftll:{1000 + cell * 60 + i}',
                request_id='request-' + key, action_id='discover-' + name, candidate_id=name,
                evidence_entry_id='entry-' + key, evidence_availability='available',
                evidence_quality='VALIDATED', requested_at='2026-09-20T09:00:00Z',
                source_version=TRANSPORT_VERSION, source_id='search:bing',
                prediction_cutoff='2026-09-20T10:00:00Z', expected_information_value=None,
                observed_risk_reduction=gain, calibration_error=None, status='linked_available',
                dependency_key='dependency-' + key)
            context = {key: count if key.startswith('operation_') else 'qualified' for key in keys}
            history.append({'link': link.as_dict(), 'context': context,
                            'label_available_at': '2026-09-21T00:00:00Z'})
    return {'version': MODEL_VERSION, 'evidence_basis': 'synthetic',
        'calibration_as_of': '2026-09-29T00:00:00Z', 'loss_unit': 'decision_loss',
        'context_keys': keys, 'min_support': 5, 'confidence': .9,
        'family_size': len(routes), 'routes': routes, 'history': history}


def policy(config=None, *, allow=True):
    return StateBoundVoISelectionPolicy(RuntimeVoIModel(config or package()), allow_synthetic=allow)


def public(**overrides):
    return {'task_id': 'fixture', 'task_type': 'match_analysis', 'match_input': 'spftll:42',
        'reference_utc': '2026-10-01T09:00:00Z', 'snapshot_as_of': '2026-10-01T10:00:00Z',
        'documents': [], 'remaining': {'http_calls': 4, 'llm_calls': 4},
        'final_decision_only': False, **overrides}


def batch(p=None, names=('a',), recommended_index=0):
    p = p or public()
    responses = []
    for name in names:
        response = match_decision(p)
        response['query'] = p['match_input'] + ' source ' + name
        responses.append(response)
    return CandidateBatch.build(p, responses, origin='synthetic_proposer', recommended_index=recommended_index)


def select(selector, b=None):
    b = b or batch()
    choice = selector.select(b.context, b)
    return choice, json.loads(choice.diagnostics_json)['state_bound_voi']


def test_algorithm_overrides_llm_preference_and_maps_stable_route_to_current_candidate():
    b = batch(names=('low', 'high'))
    chosen, diag = select(policy(package((('low', 0, -.5), ('high', 0, .9)))), b)
    assert chosen.candidate_id == b.candidates[1].candidate_id != b.recommended_id
    assert diag['route_bindings'][chosen.candidate_id]['route_id'] == 'high'
    assert diag['intervals'][chosen.candidate_id]['candidate_id'] == 'high'
    assert diag['snapshot_digest'] == b.context.snapshot_digest
    assert diag['numeric_voi_calibrated'] is False


def test_interval_crossing_cost_defers_instead_of_using_positive_mean():
    chosen, diag = select(policy(package((('a', 0, .2),))))
    assert chosen.terminal_action == 'defer'
    interval = next(iter(diag['intervals'].values()))
    assert interval['mean_value'] > interval['cost']
    assert interval['lower_bound'] < interval['cost'] < interval['upper_bound']


@pytest.mark.parametrize('override,reason', [
    ({'match_input': '主队 vs 客队'}, 'canonical_target_unresolved'),
    ({'match_input': 'spftll:042'}, 'canonical_target_unresolved'),
    ({'reference_utc': '2026-09-28T09:00:00Z'}, 'calibration_after_run_creation'),
    ({'task_type': 'legacy_inquiry'}, 'match_task_required'),
    ({'documents': [{'observed_at': '2026-10-02T09:00:00Z'}]}, 'incomplete_observation_clocks'),
    ({'documents': [{}]}, 'incomplete_observation_clocks'),
])
def test_unsupported_actor_state_is_explicitly_unmeasured(override, reason):
    chosen, diag = select(policy(), batch(public(**override)))
    assert chosen.terminal_action == 'defer' and diag['blocked_reason'] == reason
    assert not diag['intervals']


def test_synthetic_is_disabled_by_default_and_live_service_rejects_opt_in(tmp_path):
    chosen, diag = select(policy(allow=False))
    assert chosen.terminal_action == 'defer' and diag['blocked_reason'] == 'synthetic_history_disabled'
    with pytest.raises(ValueError, match='TEST_FIXTURE_MODE'):
        ExperienceService(tmp_path, selection_policy=policy())


def test_missing_history_and_current_target_only_history_have_no_fallback():
    config = package()
    for row in config['history']:
        row['link']['target_key'] = 'spftll:42'
    chosen, diag = select(policy(config))
    assert chosen.terminal_action == 'defer'
    assert next(iter(diag['intervals'].values()))['dependency_count'] == 0
    config['history'] = []
    assert select(policy(config))[0].terminal_action == 'defer'


def test_football_context_is_not_inferred_from_scores_or_llm_assertions():
    p = public(match_identity={'status': 'identified', 'label': 'qualified'},
        user_context=['qualified'], current_assessment={'answer': 'qualified'},
        documents=[{'observed_at': '2026-10-01T09:30:00Z', 'status': 'readable',
            'text': 'qualified', 'score': {'home': 3, 'away': 0},
            'context': {'qualification_state': 'qualified'},
            'return_observation': {'evidence_surface': 'body_text'}}])
    config = package(keys=['qualification_state'])
    chosen, diag = select(policy(config), batch(p))
    assert chosen.terminal_action == 'defer'
    assert next(iter(diag['intervals'].values()))['support_status'] == 'context_incomplete'
    assert diag['projection']['context']['qualification_state'] == 'unknown'
    assert diag['projection']['context']['operation_document_count'] == 1
    assert 'score' not in json.dumps(diag['projection'])


def test_source_version_mismatch_and_similar_queries_do_not_borrow_support():
    assert select(policy(), batch(names=('almost a',)))[0].terminal_action == 'defer'
    config = package()
    config['routes'][0]['source_version'] = 'older-adapter'
    for row in config['history']:
        row['link']['source_version'] = 'older-adapter'
    chosen, diag = select(policy(config))
    assert chosen.terminal_action == 'defer'
    assert next(iter(diag['route_bindings'].values()))['status'] == 'runtime_source_route_mismatch'


def test_ambiguous_runtime_route_after_template_expansion_is_unknown():
    config = package((('a', 0, .9), ('b', 0, .9)))
    config['routes'][1]['match']['query_template'] = 'spftll:42 source a'
    chosen, diag = select(policy(config))
    assert chosen.terminal_action == 'defer'
    assert next(iter(diag['route_bindings'].values()))['status'] == 'ambiguous_route'


def test_candidate_identity_changes_with_snapshot_and_old_selection_is_rejected():
    first, second = batch(), batch(public(snapshot_as_of='2026-10-01T10:01:00Z'))
    chosen, _ = select(policy(), first)
    assert chosen.candidate_id != second.candidates[0].candidate_id
    with pytest.raises(ValueError, match='STALE_OR_CROSS_TASK'):
        materialize_choice(second.context, second, chosen)
    assert select(policy(), second)[0].candidate_id == second.candidates[0].candidate_id


def test_resource_exhaustion_retains_budget_reason_even_with_positive_support():
    chosen, _ = select(policy(), batch(public(remaining={'http_calls': 0})))
    assert chosen.terminal_action == 'defer'
    assert json.loads(chosen.diagnostics_json)['selection_basis'] == 'resource_budget_gate'


class CountingTools:
    def __init__(self):
        self.actions = []

    def execute(self, action, *, run_id, reserve_request, cancelled):
        self.actions.append(deepcopy(action))
        url = 'https://example.org/search'
        request_id = reserve_request(url)
        assert request_id
        document = project_body(b'<article>Constructed source return only.</article>', url, 200)
        document.update(document_id=f'doc-{len(self.actions)}', observed_at=now().isoformat(),
                        source_published_at=None)
        return {'document': document, 'requests': [{'request_id': request_id, 'http_status': 200, 'error': None}]}


def run_scenario(root):
    config = package((('a', 0, -.5), ('b', 0, .9), ('a', 1, -.5),
                      ('c', 1, .9), ('d', 2, -.5)))
    selector, tools, seen = policy(config), CountingTools(), []

    class Proposer:
        def propose(self, p):
            seen.append(deepcopy(p))
            names = [('a', 'b'), ('a', 'c'), ('d',)][min(len(p['documents']), 2)]
            responses = []
            for name in names:
                response = match_decision(p)
                response['query'] = p['match_input'] + ' source ' + name
                response['return_assessment'] = actual_review(p)
                responses.append(response)
            return {'candidates': responses, 'recommended_index': 0, 'test_double': True}

    service = ExperienceService(root, selection_policy=selector, tools_factory=lambda f: tools,
        proposer_factory=lambda f, e: Proposer(), mode='test_fixture', enable_return_assessment=True)
    run = service.create({'match': 'spftll:42'})['run_id']
    service._drive(run, True)
    return service, run, tools, seen


def test_durable_entry_selects_two_useful_sources_replans_and_stops_with_replay(tmp_path):
    service, run, tools, seen = run_scenario(tmp_path)
    try:
        state = service.get(run)
        assert state['last_error'] is None
        assert [t['action'] for t in state['turns']] == ['search_web', 'search_web', 'stop']
        assert [a['query'] for a in tools.actions] == ['spftll:42 source b', 'spftll:42 source c']
        assert state['spent'] == {'http_calls': 2, 'llm_calls': 3}
        assert state['question_to_user'] is None and state['user_context'] == []
        assert [len(p['documents']) for p in seen] == [0, 1, 2]
        assert all(p['snapshot_as_of'] >= p['reference_utc'] for p in seen)
        diagnostics = []
        for trace in state['selection_trace']:
            folder = service.folder(run) / 'operations' / trace['operation_id']
            snapshot = json.loads((folder / 'input.json').read_text(encoding='utf-8'))
            selected = json.loads((folder / 'policy-output.json').read_text(encoding='utf-8'))
            diag = selected['diagnostics']['state_bound_voi']
            assert diag['projection']['snapshot_as_of'] == snapshot['snapshot_as_of']
            diagnostics.append(diag)
        assert [d['projection']['context']['operation_document_count'] for d in diagnostics] == [0, 1, 2]
        replay = verify_run(service.folder(run))
        assert replay['accepted_decisions_verified'] == 3
        assert replay['linked_http_receipts'] == 2 and replay['real_calls_made_by_replay'] == 0
        assert state['numeric_voi'] is None
    finally:
        service.close()


def test_resume_changed_history_identity_pauses_before_provider_calls(tmp_path):
    original = policy()
    service = ExperienceService(tmp_path, selection_policy=original, mode='test_fixture')
    run = service.create({'match': 'spftll:42'})['run_id']
    service.close()
    changed = package()
    changed['family_size'] = 3
    resumed = ExperienceService(tmp_path, selection_policy=policy(changed), mode='test_fixture')
    try:
        state = resumed.step(run)
        state = resumed.get(run)
        assert state['status'] == 'paused'
        assert 'POLICY_OR_PROPOSER_CHANGED' in state['last_error']
        assert state['spent'] == {'http_calls': 0, 'llm_calls': 0}
    finally:
        resumed.close()


def test_cli_requires_artifact_rejects_synthetic_and_creates_unmeasured_empty_model(tmp_path, capsys):
    from football_agent.runtime_cli import main
    args = ['--root', str(tmp_path / 'runs'), '--selection-policy', 'state-bound-voi']
    with pytest.raises(SystemExit):
        main(args + ['create', '--match', 'spftll:42'])
    config = package()
    path = tmp_path / 'model.json'
    path.write_text(json.dumps(config), encoding='utf-8')
    with pytest.raises(SystemExit):
        main(args + ['--value-model', str(path), 'create', '--match', 'spftll:42'])
    capsys.readouterr()
    config.update(evidence_basis='independently_evaluated', history=[])
    path.write_text(json.dumps(config), encoding='utf-8')
    main(args + ['--value-model', str(path), 'create', '--match', 'spftll:42'])
    state = json.loads(capsys.readouterr().out)
    assert state['spent'] == {'http_calls': 0, 'llm_calls': 0}
    assert state['selection_method'] == 'STATE_BOUND_EMPIRICAL_INTERVAL_SELECTION'
    assert not state['config']['selection_policy']['configuration']['numeric_voi_calibrated']


def test_snapshot_clock_is_required_and_no_reference_time_fallback():
    p = public()
    del p['snapshot_as_of']
    with pytest.raises(ValueError, match='TIMESTAMP_REQUIRED'):
        select(policy(), batch(p))
    p = public(snapshot_as_of='2026-10-01T08:00:00Z')
    with pytest.raises(ValueError, match='BEFORE_CREATION'):
        select(policy(), batch(p))


@pytest.mark.parametrize('change', ['live_default', 'model', 'proposal_mode'])
def test_queued_numeric_action_revalidates_policy_and_mode_before_source_execution(tmp_path, change):
    class Proposer:
        def propose(self, p):
            b = batch(p)
            return {'candidates': [c.response for c in b.candidates], 'recommended_index': 0}

    original = ExperienceService(tmp_path, selection_policy=policy(), mode='test_fixture',
                                 proposer_factory=lambda f, e: Proposer())
    run = original.create({'match': 'spftll:42'})['run_id']
    original.step(run)
    before = original.get(run)
    assert before['phase'] == 'fetch' and before['spent']['http_calls'] == 0
    original.close()
    tools = CountingTools()
    options = {'tools_factory': lambda f: tools}
    if change != 'live_default':
        config = package()
        if change == 'model':
            config['family_size'] = 2
        options.update(selection_policy=policy(config), mode='test_fixture')
        if change != 'proposal_mode':
            options['proposer_factory'] = lambda f, e: Proposer()
    resumed = ExperienceService(tmp_path, **options)
    try:
        after = resumed.step(run)
        assert after['status'] == 'paused' and 'QUEUED_VOI_POLICY_OR_MODE_CHANGED' in after['last_error']
        assert not tools.actions and after['spent'] == before['spent']
        assert after['queued_action'] == before['queued_action']
    finally:
        resumed.close()
    # An unchanged configuration may still finish the already chosen action.
    correct = ExperienceService(tmp_path, selection_policy=policy(), mode='test_fixture',
        proposer_factory=lambda f, e: Proposer(), tools_factory=lambda f: tools)
    try:
        after = correct.step(run)
        assert after['phase'] == 'plan' and after['spent']['http_calls'] == 1
        assert len(tools.actions) == 1
    finally:
        correct.close()


def test_numeric_plan_cannot_resume_a_fixture_run_as_live_even_with_same_model(tmp_path):
    config = package()
    config['evidence_basis'] = 'independently_evaluated'
    selector = policy(config, allow=False)
    original = ExperienceService(tmp_path, selection_policy=selector, mode='test_fixture')
    run = original.create({'match': 'spftll:42'})['run_id']
    original.close()
    resumed = ExperienceService(tmp_path, selection_policy=selector)
    try:
        after = resumed.step(run)
        assert after['status'] == 'paused' and after['spent'] == {'http_calls': 0, 'llm_calls': 0}
        assert 'POLICY_OR_PROPOSER_CHANGED' in after['last_error']
    finally:
        resumed.close()
