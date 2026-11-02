from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from football_agent.audit import atomic_write, digest
from football_agent.live_acquisition_comparison import (
    LiveAcquisitionArm, check_arm, read_json, validate_assessment,
    validate_selection, visible_document,
)


NOW = datetime(2026, 9, 30, 1, tzinfo=timezone.utc)


def config(policy='adaptive_information'):
    return {'run_id': 'case-' + policy, 'policy': policy,
            'task': {'task_id': 'task-a', 'question': 'Find the next fixture.',
                     'seed_urls': ['https://example.org/a', 'https://example.org/b']},
            'allowed_hosts': ['example.org'],
            'fixed_candidates': [candidate('a'), candidate('b')],
            'limits': {'max_http_calls': 2, 'max_llm_calls': 3,
                       'max_windows': 2, 'window_characters': 24},
            'deadline_utc': (NOW + timedelta(minutes=10)).isoformat()}


def candidate(key):
    return {'tool': 'read_url', 'url': 'https://example.org/' + key, 'query': None}


def choice(action='query', key='a', previous='none'):
    return {'task_id': 'task-a', 'action': action,
            'candidate': candidate(key) if action == 'query' else None,
            'unknown_to_resolve': 'opponent', 'useful_if': 'dated fixture',
            'irrelevant_if': 'menu only', 'previous_return': previous, 'reason': 'Check evidence.'}


def answer():
    return {'task_id': 'task-a', 'status': 'unresolved', 'answer': '', 'claims': [],
            'remaining_unknowns': ['date'], 'reason': 'No verified date.'}


class Provider:
    def __init__(self, responses):
        self.responses, self.inputs = list(responses), []

    def decide(self, public):
        self.inputs.append(deepcopy(public))
        value = self.responses.pop(0)
        if isinstance(value, BaseException):
            raise value
        if callable(value):
            value = value(public)
        return {'response': value, 'receipt': {'real_provider_call': True, 'tool_events': []}}


class Tools:
    def __init__(self, statuses=None):
        self.calls, self.statuses = [], statuses or ['OK', 'OK']

    def prepare(self, c, public):
        if c['url'] not in public['task']['seed_urls']:
            raise ValueError('unknown URL')
        return c

    def execute(self, prepared, **kwargs):
        self.calls.append(deepcopy(prepared))
        index = len(self.calls) - 1
        status = self.statuses[index]
        return {'document': {'document_id': 'doc-' + str(index),
                'url': prepared['url'], 'status': status, 'text': 'menu only' if index == 0 else 'fixture found',
                'links': [], 'raw_sha256': 'fixture'}, 'http_attempts': 1}


def run(tmp_path, cfg=None, selections=None, assessment=None, tools=None, clock=None):
    selector = Provider(selections or [choice(), choice('stop', previous='irrelevant')])
    assessor = Provider([answer()] if assessment is None else assessment)
    tools = tools or Tools()
    arm = LiveAcquisitionArm(tmp_path, cfg or config(), selector, assessor, tools,
                             clock=clock or (lambda: NOW))
    for _ in range(12):
        state = arm.step()
        if state['terminal']:
            break
    assert state['terminal']
    return state, selector, assessor, tools, arm


def test_actual_return_conditions_second_action_and_stops(tmp_path):
    def react(public):
        assert public['documents'][0]['visible_text'] == 'menu only'
        return choice('defer', previous='irrelevant')
    state, selector, assessor, tools, _ = run(tmp_path, selections=[choice(), react])
    assert state['acquisition_end'] == 'AGENT_DEFER'
    assert state['spent'] == {'http_calls': 1, 'llm_calls': 3, 'windows': 1}
    assert len(tools.calls) == 1
    assert set(assessor.inputs[0]) == {'task', 'documents'}
    assert check_arm(tmp_path, config())['all_passed']


def test_fixed_sequence_same_ceilings_actual_lower_reasoning_spend(tmp_path):
    state, selector, assessor, tools, _ = run(tmp_path, cfg=config('fixed_order'))
    assert not selector.inputs
    assert tools.calls == config()['fixed_candidates']
    assert state['spent'] == {'http_calls': 2, 'llm_calls': 1, 'windows': 2}
    assert state['assessment_validation']['independent_semantic_verification'] is False
    assert check_arm(tmp_path, config('fixed_order'))['all_passed']


def test_adaptive_changes_source_then_budget_stops(tmp_path):
    state, _, _, tools, _ = run(tmp_path, selections=[choice(), choice(key='b', previous='irrelevant')])
    assert tools.calls[1] == candidate('b')
    assert state['spent'] == {'http_calls': 2, 'llm_calls': 3, 'windows': 2}
    assert state['acquisition_end'] == 'ACQUISITION_BUDGET_REACHED'


def test_stop_does_not_resolve_task(tmp_path):
    state, _, _, tools, _ = run(tmp_path, selections=[choice('stop')])
    assert state['assessment']['status'] == 'unresolved'
    assert state['spent']['http_calls'] == 0 and not tools.calls


def test_failed_http_document_charged_and_visible(tmp_path):
    state, selector, _, _, _ = run(tmp_path, tools=Tools(['FAILED']))
    assert selector.inputs[1]['documents'][0]['status'] == 'FAILED'
    assert state['spent']['http_calls'] == state['spent']['windows'] == 1


def test_provider_failure_charged_no_retry(tmp_path):
    state, selector, _, tools, arm = run(tmp_path, selections=[TimeoutError()])
    assert state['spent']['llm_calls'] == 2
    assert state['acquisition_end'] == 'SELECTION_FAILED'
    arm.step()
    assert len(selector.inputs) == 1 and not tools.calls


def test_pending_call_on_resume_is_not_reissued(tmp_path):
    arm = LiveAcquisitionArm(tmp_path, config(), Provider([]), Provider([]), Tools(), clock=lambda: NOW)
    state = arm._load()
    arm._reserve(state, 'selection', {'task': config()['task']})
    resumed = arm.step()
    assert resumed['terminal'] == 'UNKNOWN_PENDING_CALL_REQUIRES_RECONCILIATION'
    assert resumed['spent']['llm_calls'] == 1


def test_applied_selection_is_resumed_at_http(tmp_path):
    selector, assessor, tools = Provider([choice()]), Provider([answer()]), Tools()
    arm = LiveAcquisitionArm(tmp_path, config(), selector, assessor, tools, clock=lambda: NOW)
    state = arm.step()
    assert state['phase'] == 'retrieve' and not tools.calls
    resumed = LiveAcquisitionArm(tmp_path, config(), Provider([]), assessor, tools, clock=lambda: NOW)
    state = resumed.step()
    assert len(tools.calls) == 1 and state['spent']['llm_calls'] == 1


def test_deadline_has_no_calls(tmp_path):
    state, s, a, t, _ = run(tmp_path, clock=lambda: NOW + timedelta(hours=1))
    assert state['terminal'] == 'DEADLINE_EXPIRED'
    assert not s.inputs and not a.inputs and not t.calls


def test_late_provider_response_preserved_but_not_executed(tmp_path):
    times = [NOW]
    def late(_):
        times[0] += timedelta(hours=1)
        return choice()
    state, _, _, t, _ = run(tmp_path, selections=[late], clock=lambda: times[0])
    assert state['operations'][0]['status'] == 'late'
    assert (tmp_path / 'operations/op-001/result.json').exists()
    assert not t.calls and state['spent']['llm_calls'] == 1


def test_changed_config_does_not_reset_budget(tmp_path):
    _, _, _, _, arm = run(tmp_path)
    arm.config['task']['question'] = 'changed'
    with pytest.raises(ValueError, match='ARM_CONFIG_CHANGED'):
        arm.step()


@pytest.mark.parametrize('alter', ['task', 'null_candidate', 'empty_test', 'terminal_candidate'])
def test_invalid_choices(alter):
    c = choice()
    if alter == 'task': c['task_id'] = 'other'
    if alter == 'null_candidate': c['candidate'] = None
    if alter == 'empty_test': c['useful_if'] = ''
    if alter == 'terminal_candidate': c['action'] = 'stop'
    with pytest.raises(ValueError):
        validate_selection(c, 'task-a')


def test_window_has_explicit_scope_and_hidden_quote_rejected():
    d = visible_document({'document_id': 'd', 'url': 'https://example.org',
                         'status': 'OK', 'text': 'menu only fixture', 'links': []}, 9)
    assert d['window_truncated'] and d['window_end'] == 9
    a = answer()
    a['claims'] = [{'statement': 'fixture', 'document_id': 'd', 'quote': 'fixture'}]
    with pytest.raises(ValueError, match='UNSUPPORTED_QUOTATION'):
        validate_assessment(a, config()['task'], [d])


def test_answer_with_gaps_rejected(tmp_path):
    a = answer()
    a['status'] = 'answered'
    state, *_ = run(tmp_path, assessment=[a])
    assert state['terminal'].startswith('INVALID_ASSESSMENT_')


def test_input_tampering_detected(tmp_path):
    run(tmp_path)
    p = tmp_path / 'operations/op-001/input.json'
    value = read_json(p)
    value['task']['question'] = 'sibling result injected'
    atomic_write(p, value)
    assert 'op-001:input_digest' in check_arm(tmp_path, config())['errors']


def test_provider_and_http_failures_are_distinct(tmp_path):
    class Broken(Tools):
        def execute(self, *args, **kwargs):
            raise OSError('fixture failure')
    state, *_ = run(tmp_path, tools=Broken())
    assert state['acquisition_end'] == 'HTTP_FAILED'
    assert state['spent'] == {'http_calls': 1, 'llm_calls': 2, 'windows': 0}
    assert check_arm(tmp_path, config())['all_passed']
