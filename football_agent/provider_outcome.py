'''Stable, redacted outcome contract for model/provider decision attempts.

The provider may expose a large amount of process output while an Agent is
running. The task state only needs a small contract that answers three
questions: where the attempt stopped, whether a decision was accepted, and
what recovery boundary applies. This module deliberately stores error codes
and hashes, never provider text or credentials.
'''
from __future__ import annotations

import hashlib
import json
import subprocess


VERSION = 'provider-outcome-v1'

_KNOWN_CODES = {
    'CODEX_CLI_NOT_CONFIGURED': ('process_start', 'cli_not_configured'),
    'PROVIDER_NON_JSON_EVENT': ('stream_parse', 'non_json_event'),
    'PROVIDER_EVENT_SHAPE': ('stream_parse', 'event_shape'),
    'PROVIDER_ITEM_SHAPE': ('stream_parse', 'item_shape'),
    'PROVIDER_USED_UNDECLARED_TOOL': ('stream_validation', 'undeclared_tool'),
    'PROVIDER_TURN_FAILED': ('turn_completion', 'turn_failed'),
    'PROVIDER_MESSAGE_TEXT': ('message_parse', 'message_not_text'),
    'PROVIDER_FILE_STREAM_MISMATCH': ('file_consistency', 'file_stream_mismatch'),
    'PROPOSER_RECOMMENDATION_OUT_OF_RANGE': ('candidate_validation', 'recommendation_invalid'),
    'TASK_MISMATCH': ('candidate_validation', 'candidate_invalid'),
    'FINAL_DECISION_REQUIRED': ('decision_validation', 'decision_invalid'),
    'HTTP_BUDGET_EXHAUSTED': ('decision_validation', 'decision_invalid'),
    'LOCAL_QUERY_UNAVAILABLE_OR_BUDGET_EXHAUSTED': ('decision_validation', 'decision_invalid'),
    'POLICY_CHANGED_DURING_SELECTION': ('policy_selection', 'policy_changed'),
}

_PHASE_DEFAULTS = {
    'provider_call': ('provider_call', 'provider_error'),
    'provider_result': ('provider_result', 'provider_result_invalid'),
    'candidate_build': ('candidate_build', 'candidate_batch_invalid'),
    'candidate_validation': ('candidate_validation', 'candidate_invalid'),
    'policy_selection': ('policy_selection', 'policy_error'),
    'decision_validation': ('decision_validation', 'decision_invalid'),
    'persistence': ('persistence', 'persistence_error'),
}


def _safe_code(exc):
    if isinstance(exc, subprocess.TimeoutExpired):
        return 'TIMEOUT'
    if isinstance(exc, OSError):
        return 'RUNNER_OS_ERROR'
    if isinstance(exc, json.JSONDecodeError):
        return 'JSON_DECODE_ERROR'
    text = str(exc)
    # Provider and application validators use compact uppercase sentinel codes.
    # Extract only a known token so user text cannot enter the public contract.
    for code in sorted(_KNOWN_CODES, key=len, reverse=True):
        if code in text:
            return code
    if exc.__class__.__name__ == 'ValidationError' and 'jsonschema' in exc.__class__.__module__:
        return 'JSONSCHEMA_VALIDATION_ERROR'
    if isinstance(exc, ValueError):
        return 'VALUE_ERROR'
    return exc.__class__.__name__[:80] or 'UNKNOWN_ERROR'


def _message_hash(exc):
    return hashlib.sha256(str(exc).encode('utf-8', errors='replace')).hexdigest()


def _classification(exc, phase):
    code = _safe_code(exc)
    if code == 'TIMEOUT':
        return 'process_timeout', 'timeout'
    if code == 'RUNNER_OS_ERROR':
        return 'process_start', 'runner_os_error'
    if code == 'JSON_DECODE_ERROR':
        return 'message_parse' if phase in {'provider_result', 'candidate_build'} else phase, 'message_not_json'
    if code == 'JSONSCHEMA_VALIDATION_ERROR':
        return 'schema_validation', 'schema_invalid'
    if code in _KNOWN_CODES:
        return _KNOWN_CODES[code]
    return _PHASE_DEFAULTS.get(phase, ('unknown', 'unknown_error'))


def _recovery(category):
    # A completed or partially completed remote turn may have an unknown
    # billing/outcome state. The service therefore never retries it silently.
    uncertain = category in {'timeout', 'runner_os_error', 'non_json_event', 'turn_failed'}
    return {
        'budget_charged': True,
        'automatic_retry': False,
        'same_operation_reusable': False if uncertain else True,
        'remote_outcome': 'unknown' if uncertain else 'not_accepted',
        'next_step': 'inspect_attempt_then_start_new_budgeted_attempt',
    }


def classify_failure(exc, *, phase='provider_call', diagnostic_ref=None):
    '''Return a stable public failure record without leaking provider output.'''
    stage, category = _classification(exc, phase)
    record = {
        'schema_version': VERSION,
        'status': 'failed',
        'stage': stage,
        'category': category,
        'error_code': _safe_code(exc),
        'error_hash': _message_hash(exc),
        'response_accepted': False,
        'diagnostic_ref': diagnostic_ref,
        'redaction': 'code-and-hash-only; provider text, stderr and credentials excluded',
    }
    record.update(_recovery(category))
    return record


def accepted_outcome(*, diagnostic_ref=None):
    '''Return the compact receipt placed on a committed decision attempt.'''
    return {
        'schema_version': VERSION,
        'status': 'accepted',
        'stage': 'decision_committed',
        'category': 'accepted',
        'error_code': None,
        'error_hash': None,
        'response_accepted': True,
        'diagnostic_ref': diagnostic_ref,
        'redaction': 'provider text and credentials excluded',
        'budget_charged': True,
        'automatic_retry': False,
        'same_operation_reusable': False,
        'remote_outcome': 'accepted',
        'next_step': 'continue_with_validated_decision',
    }


def provider_receipt_reference(provider, run_folder):
    """Return a run-relative receipt path for a real provider attempt."""
    attempt = getattr(provider, 'last_attempt_folder', None)
    if not attempt:
        return None
    try:
        from pathlib import Path
        path = Path(attempt).resolve() / 'receipt.json'
        root = Path(run_folder).resolve()
        if not path.exists():
            return None
        return path.relative_to(root).as_posix()
    except (OSError, ValueError, TypeError):
        return None


def diagnostic_reference(provider, run_folder):
    '''Return a run-relative diagnostic path when the provider exposes one.'''
    attempt = getattr(provider, 'last_attempt_folder', None)
    if not attempt:
        return None
    try:
        from pathlib import Path
        path = Path(attempt).resolve() / 'diagnostic.json'
        root = Path(run_folder).resolve()
        return path.relative_to(root).as_posix()
    except (OSError, ValueError, TypeError):
        return None
