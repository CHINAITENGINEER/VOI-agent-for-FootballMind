'''Process failures stay unknown; diagnostics do not retry or become evidence.'''
import hashlib
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from football_agent.inquiry_provider import CodexInquiryProvider
from football_agent.inquiry_provider_diagnostics import MAX_PARSE_BYTES, record_process_diagnostic
from football_agent.inquiry import InquiryWorkflow
from football_agent.inquiry_tools import OpenWebTools
from football_agent.inquiry_verify import verify
from football_spider.storage.raw_archive import RawArchive


def read_diagnostic(root):
    paths = list(root.glob('*/diagnostic.json'))
    assert len(paths) == 1
    return json.loads(paths[0].read_text(encoding='utf-8'))


@pytest.mark.parametrize('encoding', ['bytes', 'text', 'absent'])
def test_timeout_has_diagnostic_and_keeps_original_exception_without_retry(tmp_path, encoding):
    stdout = chr(10).join(json.dumps(x) for x in [
        {'type': 'thread.started'}, {'type': 'turn.started'},
        {'type': 'item.completed', 'item': {'type': 'reasoning', 'text': 'PRIVATE_REASONING'}}])
    stderr = 'Authorization: SECRET_TOKEN; stream disconnected; reconnecting 1/5'
    if encoding == 'bytes':
        stdout, stderr = stdout.encode(), stderr.encode()
    if encoding == 'absent':
        stdout, stderr = None, None
    failure = subprocess.TimeoutExpired('test-cli', 120, output=stdout, stderr=stderr)
    calls = []
    def runner(command, **kwargs):
        calls.append(command)
        raise failure
    provider = CodexInquiryProvider(tmp_path, executable='test-cli', runner=runner)
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        provider.decide({'task': 'public'})
    assert caught.value is failure and len(calls) == 1
    data = read_diagnostic(tmp_path)
    assert data['category'] == 'timeout'
    assert data['remote_outcome'] == data['remote_billing'] == 'unknown'
    assert not data['retry_performed'] and not data['budget_refunded']
    assert data['observed_turn_started'] == (encoding != 'absent')
    assert data['diagnostic_signature_flags']['stream_retry'] == (encoding != 'absent')
    assert 'PRIVATE_REASONING' not in json.dumps(data) and 'SECRET_TOKEN' not in json.dumps(data)
    assert not list(tmp_path.glob('*/receipt.json'))


def test_late_response_is_not_promoted_to_accepted_turn(tmp_path):
    def runner(command, **kwargs):
        Path(command[command.index('--output-last-message') + 1]).write_text('{}', encoding='utf-8')
        raise subprocess.TimeoutExpired(command, 120, output=json.dumps({'type': 'turn.completed'}).encode())
    with pytest.raises(subprocess.TimeoutExpired):
        CodexInquiryProvider(tmp_path, executable='test-cli', runner=runner).decide({})
    data = read_diagnostic(tmp_path)
    assert data['observed_turn_completed'] and data['response_file_exists']
    assert data['response_sha256'] == hashlib.sha256(b'{}').hexdigest()
    assert not data['response_accepted'] and data['remote_outcome'] == 'unknown'
    assert not list(tmp_path.glob('*/receipt.json'))


def test_partial_and_non_object_events_are_counted_without_crashing(tmp_path):
    body = chr(10).join([json.dumps({'type': 'turn.started'}), 'null', '[]', '{'])
    data = record_process_diagnostic(tmp_path, stdout=body, category='timeout', elapsed_seconds=1, timeout_seconds=1)
    assert data['malformed_or_partial_json_lines'] == 3
    assert data['observed_turn_started']


def test_event_names_cannot_export_arbitrary_model_text(tmp_path):
    body = chr(10).join(json.dumps(e) for e in [
        {'type': 'PRIVATE_EVENT', 'item': {'type': 'PRIVATE_TYPE', 'text': 'PRIVATE_REASONING'}},
        {'type': ['bad'], 'item': {'type': ['bad']}},
        {'type': 'error', 'message': 'HTTP 401 SECRET_TOKEN'}])
    data = record_process_diagnostic(tmp_path, stdout=body, category='timeout', elapsed_seconds=1, timeout_seconds=1)
    assert data['event_type_counts'] == {'other': 2, 'error': 1}
    assert data['item_type_counts'] == {'other': 2}
    assert data['diagnostic_signature_flags']['authentication']
    assert 'PRIVATE' not in json.dumps(data) and 'SECRET' not in json.dumps(data)


def test_parsing_is_bounded_and_hash_covers_full_output(tmp_path):
    body = b' ' * MAX_PARSE_BYTES + json.dumps({'type': 'turn.completed'}).encode()
    data = record_process_diagnostic(tmp_path, stdout=body, category='timeout', elapsed_seconds=1, timeout_seconds=1)
    assert data['stdout']['parse_truncated'] and data['stdout']['parsed_bytes'] == MAX_PARSE_BYTES
    assert data['stdout']['sha256'] == hashlib.sha256(body).hexdigest()
    assert not data['observed_turn_completed']


@pytest.mark.parametrize('failure', [FileNotFoundError('missing executable'), PermissionError('not started')])
def test_runner_os_error_preserves_exception(tmp_path, failure):
    def runner(*args, **kwargs):
        raise failure
    with pytest.raises(type(failure)) as caught:
        CodexInquiryProvider(tmp_path, executable='test-cli', runner=runner).decide({})
    assert caught.value is failure
    assert read_diagnostic(tmp_path)['category'] == 'runner_os_error'


@pytest.mark.parametrize('signal,flag', [
    ('HTTP 429 Too many requests', 'rate_limit'),
    ('error sending request', 'network'),
    ('Invalid schema for response_format', 'schema_rejected'),
    ('unknown feature: test', 'cli_argument')])
def test_diagnostic_flags_never_accept_a_failed_turn(tmp_path, signal, flag):
    def runner(*args, **kwargs):
        return SimpleNamespace(returncode=1, stdout='', stderr=signal)
    with pytest.raises(RuntimeError, match='PROVIDER_TURN_FAILED'):
        CodexInquiryProvider(tmp_path, executable='test-cli', runner=runner).decide({})
    data = read_diagnostic(tmp_path)
    assert data['diagnostic_signature_flags'][flag] and not data['response_accepted']


def test_success_keeps_validated_receipt_separate_from_raw_process_diagnostic(tmp_path):
    def runner(command, **kwargs):
        Path(command[command.index('--output-last-message') + 1]).write_text('{}', encoding='utf-8')
        return SimpleNamespace(returncode=0, stdout=json.dumps({'type': 'turn.completed'}), stderr='')
    response = CodexInquiryProvider(tmp_path, executable='test-cli', runner=runner).decide({})
    assert response['receipt']['real_provider_call'] is True
    assert read_diagnostic(tmp_path)['observed_turn_completed']


def test_diagnostic_write_failure_does_not_mask_timeout(tmp_path, monkeypatch):
    failure = subprocess.TimeoutExpired('test-cli', 120)
    def runner(*args, **kwargs):
        raise failure
    def broken_diagnostic(*args, **kwargs):
        raise OSError('Disk failure with PRIVATE_PATH')
    monkeypatch.setattr('football_agent.inquiry_provider.record_process_diagnostic', broken_diagnostic)
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        CodexInquiryProvider(tmp_path, executable='test-cli', runner=runner).decide({})
    assert caught.value is failure
    assert failure.__notes__ == ['Diagnostic persistence failed: OSError']


@pytest.mark.parametrize('stdout', ['SECRET_OUTPUT', 'null', '[]',
    json.dumps({'type': 'item.completed', 'item': 'SECRET_ITEM'})])
def test_invalid_events_fail_closed_without_exporting_raw_error_text(tmp_path, stdout):
    def runner(*args, **kwargs):
        return SimpleNamespace(returncode=1, stdout=stdout, stderr='SECRET_CREDENTIAL')
    with pytest.raises(RuntimeError, match='PROVIDER_NON_JSON_EVENT'):
        CodexInquiryProvider(tmp_path, executable='test-cli', runner=runner).decide({})
    for path in tmp_path.glob('*/*.json'):
        assert 'SECRET_' not in path.read_text(encoding='utf-8')


def test_error_event_payload_is_not_copied_into_process_record(tmp_path):
    def runner(*args, **kwargs):
        return SimpleNamespace(returncode=1, stdout=json.dumps({'type': 'error',
            'message': 'SECRET_CREDENTIAL HTTP 401'}), stderr='')
    with pytest.raises(RuntimeError, match='PROVIDER_TURN_FAILED'):
        CodexInquiryProvider(tmp_path, executable='test-cli', runner=runner).decide({})
    for path in tmp_path.glob('*/*.json'):
        assert 'SECRET_' not in path.read_text(encoding='utf-8')


def test_diagnosed_timeout_retains_pending_budget_and_never_becomes_source_evidence(tmp_path):
    now = datetime(2026, 9, 29, 2, tzinfo=timezone.utc)
    calls = []
    def runner(command, **kwargs):
        calls.append(command)
        raise subprocess.TimeoutExpired(command, 120,
            output=json.dumps({'type': 'turn.started'}).encode(), stderr=b'error sending request')
    def unexpected_network():
        raise AssertionError('No football source call should occur')
    provider = CodexInquiryProvider(tmp_path / 'provider', executable='test-cli', runner=runner)
    tools = OpenWebTools(RawArchive(tmp_path / 'evidence'), ['example.com'],
                         clock=lambda: now, session_factory=unexpected_network)
    workflow = InquiryWorkflow(tmp_path / 'runs', provider, tools, clock=lambda: now)
    task = {'task_id': 'constructed-timeout', 'objective': 'Find the next useful source',
            'deadline': (now + timedelta(hours=1)).isoformat(),
            'initial_facts': [], 'seed_urls': ['https://example.com/preview']}
    run = workflow.run(workflow.start(task, max_requests=2, max_llm_calls=4)['run_id'])
    assert run['state'] == 'DEFERRED' and run['pending']['kind'] == 'llm'
    assert run['budget']['llm_calls'] == 1 and run['budget']['http_attempts'] == 0
    assert not run['documents'] and not run['claims'] and not run['budget']['llm_usage']
    assert workflow.step(run['run_id']) == run and len(calls) == 1
    proof = verify(tmp_path, run['run_id'])
    assert proof['real_completed_llm_turns'] == 0
    assert read_diagnostic(tmp_path / 'provider')['diagnostic_signature_flags']['network']


def test_item_error_message_participates_in_diagnosis(tmp_path):
    data = record_process_diagnostic(tmp_path, stdout=json.dumps({'type': 'item.completed',
        'item': {'type': 'error', 'message': 'Authentication failed; SECRET_TOKEN'}}),
        category='timeout', elapsed_seconds=1, timeout_seconds=1)
    assert data['diagnostic_signature_flags']['authentication']
    assert data['public_errors']['items'][0]['message'].startswith('Authentication failed')
    assert 'SECRET_TOKEN' not in json.dumps(data)


def test_required_host_enabled_without_enabling_evidence_tools_or_changing_model(tmp_path):
    def runner(command, **kwargs):
        disabled = {command[i + 1] for i, part in enumerate(command[:-1]) if part == '--disable'}
        enabled = {command[i + 1] for i, part in enumerate(command[:-1]) if part == '--enable'}
        assert enabled == {'code_mode_host'} and 'code_mode_host' not in disabled
        assert {'shell_tool', 'browser_use', 'apps', 'plugins', 'multi_agent'} <= disabled
        assert '-m' not in command and '--model' not in command
        assert command[command.index('--sandbox') + 1] == 'read-only'
        assert kwargs.get('shell', False) is False
        Path(command[command.index('--output-last-message') + 1]).write_text('{}', encoding='utf-8')
        return SimpleNamespace(returncode=0, stdout=json.dumps({'type': 'turn.completed'}), stderr='')
    provider = CodexInquiryProvider(tmp_path, executable='test-cli', runner=runner)
    assert provider.configuration()['version'] == 'codex-inquiry-json-v2'
    assert provider.configuration()['code_mode_host'] == 'enabled_required_runtime'
    provider.decide({})


def test_disabled_host_is_explicit_diagnostic_signal(tmp_path):
    data = record_process_diagnostic(tmp_path, stdout=json.dumps({'type': 'item.completed',
        'item': {'type': 'error', 'message': 'Code Mode is unavailable because code-mode host is disabled.'}}),
        category='timeout', elapsed_seconds=1, timeout_seconds=1)
    assert data['diagnostic_signature_flags']['code_mode_host_disabled']
