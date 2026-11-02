import json
import subprocess
from pathlib import Path

import pytest

from football_agent.experience import ExperienceService, public_state
from football_agent.provider_outcome import (
    accepted_outcome,
    classify_failure,
)


URL = 'https://example.org/fixture'


@pytest.mark.parametrize(
    'exc,phase,stage,category,unknown',
    [
        (subprocess.TimeoutExpired(['codex'], 5), 'provider_call', 'process_timeout', 'timeout', True),
        (OSError('runner failed'), 'provider_call', 'process_start', 'runner_os_error', True),
        (RuntimeError('PROVIDER_NON_JSON_EVENT'), 'provider_call', 'stream_parse', 'non_json_event', True),
        (RuntimeError('PROVIDER_USED_UNDECLARED_TOOL'), 'provider_call', 'stream_validation', 'undeclared_tool', False),
        (RuntimeError('PROVIDER_TURN_FAILED'), 'provider_call', 'turn_completion', 'turn_failed', True),
        (RuntimeError('PROVIDER_FILE_STREAM_MISMATCH'), 'provider_result', 'file_consistency', 'file_stream_mismatch', False),
    ],
)
def test_failure_contract_exposes_stage_without_provider_text(exc, phase, stage, category, unknown):
    outcome = classify_failure(exc, phase=phase, diagnostic_ref='provider/attempt/diagnostic.json')
    assert outcome['schema_version'] == 'provider-outcome-v1'
    assert outcome['status'] == 'failed'
    assert (outcome['stage'], outcome['category']) == (stage, category)
    assert outcome['response_accepted'] is False
    assert outcome['budget_charged'] is True
    assert outcome['automatic_retry'] is False
    assert outcome['remote_outcome'] == ('unknown' if unknown else 'not_accepted')
    assert 'runner failed' not in json.dumps(outcome, ensure_ascii=False)
    assert 'diagnostic_ref' in outcome


def test_accepted_contract_is_a_small_committed_receipt():
    outcome = accepted_outcome(diagnostic_ref=None)
    assert outcome == {
        'schema_version': 'provider-outcome-v1',
        'status': 'accepted',
        'stage': 'decision_committed',
        'category': 'accepted',
        'error_code': None,
        'error_hash': None,
        'response_accepted': True,
        'diagnostic_ref': None,
        'redaction': 'provider text and credentials excluded',
        'budget_charged': True,
        'automatic_retry': False,
        'same_operation_reusable': False,
        'remote_outcome': 'accepted',
        'next_step': 'continue_with_validated_decision',
    }


class FailingProvider:
    def __init__(self, exc):
        self.exc = exc

    def decide(self, public):
        raise self.exc


def _create(service):
    return service.create({'question': '核实比赛信息', 'seed_urls': [URL]})['run_id']


def test_service_persists_provider_failure_boundary_without_http_or_turn(tmp_path):
    service = ExperienceService(
        tmp_path,
        provider_factory=lambda folder, effort: FailingProvider(RuntimeError('PROVIDER_TURN_FAILED')),
        mode='test_fixture',
    )
    try:
        run_id = _create(service)
        service._drive(run_id, True)
        state = service.get(run_id)
        outcome = state['last_provider_outcome']
        assert state['status'] == 'paused'
        assert state['spent'] == {'http_calls': 0, 'llm_calls': 1}
        assert state['pending'] is None
        assert state['turns'] == [] and state['documents'] == []
        assert outcome['stage'] == 'turn_completion'
        assert outcome['category'] == 'turn_failed'
        assert outcome['remote_outcome'] == 'unknown'
        assert public_state(state)['last_provider_outcome'] == outcome
        operation = state['operations'][-1]
        assert operation['status'] == 'failed'
        assert operation['provider_outcome'] == outcome
        failure = next((state_path.parent / 'failure.json'
                        for state_path in (service.folder(run_id) / 'operations').glob('*/input.json')))
        saved = json.loads(failure.read_text(encoding='utf-8'))
        assert saved['provider_outcome'] == outcome
        assert 'PROVIDER_TURN_FAILED' in saved['provider_outcome']['error_code']
    finally:
        service.close()


def test_previous_validated_turn_survives_later_provider_failure(tmp_path):
    calls = {'count': 0}

    def provider_factory(folder, effort):
        class Provider:
            def decide(self, public):
                calls['count'] += 1
                if calls['count'] == 2:
                    raise RuntimeError('PROVIDER_NON_JSON_EVENT')
                return {'response': {
                    'task_id': public['task_id'], 'action': 'ask_user', 'url': None,
                    'query': None, 'engine': None, 'question_to_user': '请补充比赛日期。',
                    'unknown_to_resolve': '比赛日期', 'useful_if': '可对应到唯一赛事',
                    'why_worth_cost': '避免混淆同名比赛', 'previous_return': 'none',
                    'assessment': {'status': 'unresolved', 'answer': '待确认', 'claims': [],
                                   'remaining_unknowns': ['比赛日期']},
                    'alternatives': [], 'reason': '仍缺少比赛身份信息',
                }, 'test_double': True}
        return Provider()

    service = ExperienceService(tmp_path, provider_factory=provider_factory, mode='test_fixture')
    try:
        run_id = _create(service)
        service._drive(run_id, True)
        assert service.get(run_id)['status'] == 'needs_input'
        service.answer(run_id, '2026-10-01')
        service.step(run_id)
        state = service.get(run_id)
        assert state['status'] == 'paused'
        assert state['spent']['llm_calls'] == 2
        assert len(state['turns']) == 1
        assert state['turns'][0]['action'] == 'ask_user'
        assert state['last_provider_outcome']['category'] == 'non_json_event'
        assert state['last_provider_outcome']['response_accepted'] is False
    finally:
        service.close()
