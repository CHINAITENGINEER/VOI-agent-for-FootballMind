"""Immediate-return supervision, using constructed clocks and a frozen real f.

The fixture below is protocol input, not an independently observed football
match. Public raw entrypoints must still replay and verify original archives.
"""
from copy import deepcopy
from hashlib import sha256
import json
import math
import socket

import pytest

from football_agent.learned_voi_web import _web_action
from football_research.acquisition.context_gain import _signed_records, _load_corpus
from football_research.acquisition.context_predictor import _state
from football_research.acquisition.state import parse_time
from football_research.acquisition.visible_state import encode_visible_state
from football_research.data_adapters.context_training import _visible_prefixes, VERSION
from football_research.data_adapters.prospective_text import _digest
from test_context_gain import frozen, registration
from test_context_predictor import WORDS


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        pytest.fail('Immediate context supervision must not use the network')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'getaddrinfo', denied)


def example(frozen, *, status='OK', late=False, labels=True):
    model = frozen[0]
    mid, physical = 'provider:B0', 'B0'
    action = _web_action('read_url', 'https://www.uefa.com/news/body/',
                         cost={'requests': 1., 'money': .01})
    parent = {'document_id': 'parent', 'raw_sha256': 'a'*64,
        'url': 'https://www.uefa.com/news/', 'kind': 'index', 'status': 'OK',
        'observed_at': '2027-01-20T09:59:00+00:00',
        'available_at': '2027-01-20T09:59:01+00:00',
        'source_published_at': None, 'text': 'Unstructured navigation',
        'links': [{'url': action['url'], 'title': WORDS[0],
            'observed_anchors': [{'anchor_index': 0, 'text': WORDS[0], 'text_truncated': False}]}]}
    document = {'document_id': 'body', 'raw_sha256': 'b'*64, 'url': action['url'],
        'kind': 'article', 'status': status, 'text': WORDS[2], 'links': [],
        'observed_at': '2027-01-20T10:00:02+00:00', 'available_at': '2027-01-20T10:00:03+00:00',
        'source_published_at': None,
        'structured_news': {'version': 'structured-news-discovery-v3', 'role': 'article',
            'segments': [{'text': WORDS[2], 'start': 0, 'end': len(WORDS[2]),
                'content_source': 'json_ld_article_body', 'json_pointer': 'script[0].articleBody'}]}}
    budget = {'limits': {'requests': 2., 'money': .02},
              'spent': {'money': 0., 'requests': 0., 'latency_ms': 0., 'tokens': 0.}}
    before = {'state': {'match_id': mid, 'information_mode': 'operational',
        'base': {'kickoff': '2027-01-20T19:00:00+00:00'},
        'prediction_cutoff': '2027-01-20T18:00:00+00:00',
        'open_text': {'prior_probabilities': [.45, .3, .25]}, 'action_history': []},
        'documents': [parent], 'attempts': [], 'decision_at': '2027-01-20T10:00:01+00:00',
        'prediction': {'probabilities': [.45, .3, .25]}, 'budget': budget}
    before['prediction'] = model.predict(encode_visible_state(before), physical_match_id=physical)
    returned = '2027-01-20T18:00:01+00:00' if late else '2027-01-20T10:00:04+00:00'
    admission = {'status': 'admitted' if status == 'OK' else 'not_admitted',
                 'reason': 'constructed_actual_return'}
    spent = {**budget['spent'], 'money': .01, 'requests': 1.}
    after = deepcopy(before)
    after['decision_at'] = returned
    after['budget']['spent'] = spent
    after['state']['action_history'].append({'action_id': action['action_id']})
    after['attempts'].append({'action_id': action['action_id'], 'information_key': action['information_key'],
        'status': status, 'admission': admission['status']})
    if status == 'OK':
        after['documents'].append(document)
    if not late:
        after['prediction'] = model.predict(encode_visible_state(after), physical_match_id=physical)
    cycle = {'step': 0, 'public_before': before, 'generated_actions': [action],
        'selected_action': action, 'execution_status': 'returned', 'returned_at': returned,
        'public_after': None if late else after, 'public_state_after': after['state'],
        'prediction_before': before['prediction']['probabilities'],
        'prediction_after': after['prediction']['probabilities'], 'admission': admission,
        'charged_cost': spent, 'actual_cost': {'money': None, 'requests': 1., 'latency_ms': 20., 'tokens': None},
        'update': {'status': 'not_updated', 'reason': 'after_information_cutoff'} if late else {'status': 'updated'}}
    run = {'schema': 'learned-voi-open-execution-v4', 'run_id': 'constructed-immediate-B', 'cycles': [cycle]}
    if not late:
        terminal = deepcopy(after)
        terminal['decision_at'] = '2027-01-20T17:00:00+00:00'
        terminal['prediction'] = model.predict(encode_visible_state(terminal), physical_match_id=physical)
        run['cycles'].append({'step': 1, 'selected_action': None, 'generated_actions': [],
                              'public_before': terminal})
    episode = {'step': 0, 'action': action, 'response': document,
        'decision_at': before['decision_at'], 'observed_at': document['observed_at'],
        'available_at': document['available_at'], 'response_observation_eligible': True,
        'training_exclusion_reasons': [] if labels else ['independent_settlement_missing'],
        'selection_probability': 1., 'selection_scope': 'conditional_on_generated_eligible_menu_only',
        'selection_basis': 'bounded_unknown_information_exploration'}
    settlements = {mid: [{'home_goals': 0, 'away_goals': 1, 'status': 'final', 'score_period': 'regulation',
        'information_mode': 'operational', 'kickoff': '2027-01-20T19:00:00+00:00',
        'observed_at': '2027-01-20T22:00:00+00:00', 'available_at': '2027-01-20T22:00:01+00:00',
        'source': 'constructed_independent_result_fixture', 'source_reference': 'constructed:independent-score',
        'revision_id': 'constructed-1', 'input_manifest': {'score_fixture': 'c'*64}}]} if labels else {}
    identity = {'original_match_id': mid, 'source_event_id': physical}
    return run, [episode], identity, settlements


def project(parts):
    return _visible_prefixes(*parts, parse_time('2027-02-01T00:00:00+00:00'))


def derive(frozen, parts):
    rows, transitions = project(parts)
    plan = registration(frozen, [{'match_id': 'provider:B0', 'source_event_id': 'B0'}])
    return _signed_records(rows, transitions, {parts[0]['run_id']: plan}, model=frozen[0],
        model_sha=frozen[2], as_of='2027-02-01T00:00:00+00:00', origin='constructed_mechanism')


def test_immediate_target_uses_return_clock_not_age_changed_next_decision(frozen):
    parts = example(frozen)
    run = parts[0]
    assert run['cycles'][0]['prediction_after'] != run['cycles'][1]['public_before']['prediction']['probabilities']
    rows, transitions = project(parts)
    assert len(rows) == 3 and len(transitions) == 1
    pair = transitions[0]
    assert pair['after_prefix_id'] == rows[1]['prefix_id']
    assert rows[1]['visible_state']['before']['decision_at'] == run['cycles'][0]['returned_at']
    assert rows[2]['visible_state']['before']['decision_at'] == '2027-01-20T17:00:00+00:00'
    features, targets, exclusions = derive(frozen, parts)
    assert not exclusions and len(targets) == 1
    target = targets[0]
    actual = math.log(run['cycles'][0]['prediction_after'][2])-math.log(run['cycles'][0]['prediction_before'][2])
    wrong = math.log(run['cycles'][1]['public_before']['prediction']['probabilities'][2])-math.log(run['cycles'][0]['prediction_before'][2])
    assert target['signed_gain'] == actual and actual != wrong
    assert WORDS[0] in features[0]['inputs']['state']['texts']
    assert WORDS[2] not in features[0]['inputs']['state']['texts']
    assert WORDS[2] in _state(rows[1]['visible_state'])['texts']
    assert _state(rows[1]['visible_state'])['numeric'] != _state(rows[2]['visible_state'])['numeric']


@pytest.mark.parametrize('change', ['next_body', 'next_history', 'after_clock', 'after_forecast',
                                  'after_cost', 'extra_document', 'after_hidden', 'returned_clock'])
def test_causal_or_immediate_snapshot_changes_are_rejected(frozen, change):
    parts = example(frozen)
    run = parts[0]
    first, terminal = run['cycles']
    if change == 'next_body':
        terminal['public_before']['documents'][0]['text'] = 'Unobserved body substitution'
    elif change == 'next_history':
        terminal['public_before']['attempts'][0]['status'] = 'FAILED'
    elif change == 'after_clock':
        first['public_after']['decision_at'] = '2027-01-20T10:00:05+00:00'
    elif change == 'after_forecast':
        first['prediction_after'] = [.1, .2, .7]
    elif change == 'after_cost':
        first['public_after']['budget']['spent'] = {'requests': 0., 'money': 0.}
    elif change == 'extra_document':
        first['public_after']['documents'].append(deepcopy(first['public_after']['documents'][-1]))
    elif change == 'after_hidden':
        first['public_after']['attempts'] = []
    elif change == 'returned_clock':
        first['returned_at'] = '2027-01-20T09:00:00+00:00'
    with pytest.raises(ValueError):
        project(parts)


def test_failed_response_keeps_actual_failure_state_and_signed_gain(frozen):
    parts = example(frozen, status='FAILED')
    rows, pairs = project(parts)
    assert rows[1]['visible_state']['before']['documents'] == rows[0]['visible_state']['before']['documents']
    assert pairs[0]['response_status'] == 'FAILED'
    _, targets, exclusions = derive(frozen, parts)
    assert not exclusions and len(targets) == 1
    target = targets[0]
    assert target['signed_gain'] == math.log(target['prediction_after'][2])-math.log(target['prediction_before'][2])
    assert target['prediction_after'] != target['prediction_before']


def test_late_return_keeps_paid_attempt_but_has_no_gain_or_prematch_after(frozen):
    parts = example(frozen, late=True)
    rows, pairs = project(parts)
    assert len(rows) == len(pairs) == 1
    assert pairs[0]['after_prefix_id'] is None
    assert pairs[0]['gain_exclusion_reason'] == 'actual_return_after_prediction_cutoff'
    assert pairs[0]['actual_cost']['requests'] == 1
    features, targets, exclusions = derive(frozen, parts)
    assert features == targets == []
    assert exclusions == [{'run_id': parts[0]['run_id'], 'step': 0,
        'reason': 'actual_return_after_prediction_cutoff', 'returned_at': '2027-01-20T18:00:01+00:00',
        'response_status': 'OK', 'actual_cost': pairs[0]['actual_cost']}]


def test_missing_independent_result_retains_pending_states_without_gain(frozen):
    parts = example(frozen, labels=False)
    rows, pairs = project(parts)
    assert len(rows) == 3 and len(pairs) == 1
    assert all('outcome' not in row for row in rows)
    features, targets, exclusions = derive(frozen, parts)
    assert features == targets == []
    assert exclusions[0]['reason'] == 'independent_settlement_missing'


def test_measured_request_overrun_is_retained_without_zero_gain(frozen):
    parts = example(frozen)
    run = parts[0]
    run['cycles'] = run['cycles'][:1]
    cycle = run['cycles'][0]
    cycle.update(public_after=None, resource_overrun=True,
        update={'status': 'not_updated', 'reason': 'actual_resource_budget_overrun'})
    cycle['charged_cost']['requests'] = 3.
    cycle['actual_cost']['requests'] = 3.
    rows, pairs = project(parts)
    assert len(rows) == len(pairs) == 1
    assert pairs[0]['after_prefix_id'] is None
    assert pairs[0]['gain_exclusion_reason'] == 'actual_resource_budget_overrun'
    features, targets, excluded = derive(frozen, parts)
    assert features == targets == []
    assert excluded[0]['reason'] == 'actual_resource_budget_overrun'
    assert excluded[0]['actual_cost']['requests'] == 3.
    cycle['charged_cost']['requests'] = 1.
    with pytest.raises(ValueError, match='actually exceeded physical limits'):
        project(parts)


def test_exploration_only_overrun_retains_cost_without_after_or_gain(frozen):
    parts = example(frozen)
    run = parts[0]
    run['cycles'] = run['cycles'][:1]
    cycle = run['cycles'][0]
    cycle.update(public_after=None, resource_overrun=True,
        exploration_limits={'requests': 1., 'latency_ms': 10.},
        exploration_charged_cost={'requests': 1., 'latency_ms': 20.},
        update={'status': 'not_updated', 'reason': 'actual_resource_budget_overrun'})
    rows, pairs = project(parts)
    assert len(rows) == 1 and pairs[0]['exploration_charged_cost']['latency_ms'] == 20.
    features, targets, excluded = derive(frozen, parts)
    assert features == targets == []
    assert excluded[0]['reason'] == 'actual_resource_budget_overrun'
    assert excluded[0]['actual_cost']['latency_ms'] == 20.
    cycle['exploration_charged_cost']['latency_ms'] = 5.
    with pytest.raises(ValueError, match='actually exceeded physical limits'):
        project(parts)


def test_old_version_still_rejects_age_changed_next_prediction(frozen):
    parts = example(frozen)
    parts[0]['schema'] = 'learned-voi-open-execution-v3'
    with pytest.raises(ValueError, match='After state is not the actual next visible snapshot'):
        project(parts)


def test_v4_corpus_loader_reencodes_each_original_before_and_immediate_after(frozen, tmp_path):
    parts = example(frozen)
    rows, transitions = project(parts)
    run = parts[0]
    source, corpus = tmp_path/'source', tmp_path/'corpus'
    source.mkdir()
    corpus.mkdir()
    config = {'run_id': run['run_id']}
    run['configuration_digest'] = _digest(config)
    rows, transitions = project(parts)
    def write(path, value):
        path.write_text(json.dumps(value), encoding='utf-8')
    write(source/'run.json', run)
    write(source/'configuration.json', config)
    report = {'schema': VERSION, 'evidence_origin': 'constructed_mechanism',
        'input_bindings': {str(path): sha256(path.read_bytes()).hexdigest() for path in source.iterdir()},
        'records_sha256': _digest(rows), 'pending_sha256': _digest([]),
        'transitions_sha256': _digest(transitions), 'registration_inputs_sha256': _digest([])}
    for name, value in [('conversion-report', report), ('prefix-records', rows),
                        ('pending-prefixes', []), ('actual-transitions', transitions), ('registration-inputs', [])]:
        write(corpus/(name+'.json'), value)
    loaded = _load_corpus(corpus)
    assert loaded[0] == rows and loaded[1] == transitions
    assert loaded[0][1]['lineage']['public_snapshot'] == 'public_after'
