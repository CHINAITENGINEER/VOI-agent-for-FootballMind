'''Bounded process observations; no model reasoning or credentials retained.'''
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from .audit import atomic_write
from .inquiry_provider_errors import public_error_summary

MAX_PARSE_BYTES = 1024 * 1024
EVENT_TYPES = frozenset({'thread.started', 'turn.started', 'turn.completed', 'turn.failed',
    'item.started', 'item.updated', 'item.completed', 'error'})
ITEM_TYPES = frozenset({'agent_message', 'reasoning', 'error', 'command_execution',
    'file_change', 'mcp_tool_call', 'web_search', 'todo_list', 'plan'})


def _stream(value):
    raw = b'' if value is None else value if isinstance(value, bytes) else value.encode('utf-8', errors='replace')
    return raw, {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
                 'parsed_bytes': min(len(raw), MAX_PARSE_BYTES),
                 'parse_truncated': len(raw) > MAX_PARSE_BYTES}


def record_process_diagnostic(folder, *, stdout=None, stderr=None, category,
                              elapsed_seconds, timeout_seconds, returncode=None, exception_type=None):
    '''Record observations, never infer remote completion or refund a call.'''
    folder = Path(folder)
    raw, out = _stream(stdout)
    err_raw, err = _stream(stderr)
    events, items = Counter(), Counter()
    malformed = 0
    error_text = []
    for line in raw[:MAX_PARSE_BYTES].splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError('Event must be an object')
        except (ValueError, UnicodeError):
            malformed += 1
            continue
        kind = event.get('type')
        events[kind if isinstance(kind, str) and kind in EVENT_TYPES else 'other'] += 1
        item = event.get('item')
        if isinstance(item, dict):
            item_kind = item.get('type')
            items[item_kind if isinstance(item_kind, str) and item_kind in ITEM_TYPES else 'other'] += 1
            if item_kind == 'error' and isinstance(item.get('message'), str):
                error_text.append(item['message'][:4096])
        if kind in ('error', 'turn.failed'):
            error_text.append(json.dumps(event, ensure_ascii=False)[:4096])
    text = err_raw[:MAX_PARSE_BYTES].decode('utf-8', errors='replace') + chr(10) + chr(10).join(error_text)
    signatures = {
        'authentication': r'(?<![0-9])401(?![0-9])|unauthori[sz]ed|authentication (?:failed|required)|invalid api key',
        'rate_limit': r'(?<![0-9])429(?![0-9])|too many requests|rate.?limit',
        'network': r'failed to connect|connection (?:refused|reset|closed)|dns error|tls error|error sending request',
        'stream_retry': r'reconnecting|retrying|stream disconnected',
        'schema_rejected': r'invalid schema|schema.*(?:invalid|unsupported)|unsupported.*schema',
        'cli_argument': r'unexpected argument|unrecognized (?:option|argument)|unknown feature',
        'code_mode_host_disabled': r'code.mode host is disabled',
    }
    response_path = folder / 'response.json'
    response_exists = response_path.is_file()
    result = {
        'schema_version': 'inquiry-provider-diagnostic-v1',
        'category': category, 'exception_type': exception_type,
        'stage': 'process_observation_before_adapter_validation',
        'elapsed_seconds': max(0.0, elapsed_seconds), 'timeout_seconds': timeout_seconds,
        'returncode': returncode, 'stdout': out, 'stderr': err,
        'event_type_counts': dict(events), 'item_type_counts': dict(items),
        'malformed_or_partial_json_lines': malformed,
        'public_errors': public_error_summary(stdout),
        'observed_thread_started': bool(events['thread.started']),
        'observed_turn_started': bool(events['turn.started']),
        'observed_turn_completed': bool(events['turn.completed']),
        'observed_turn_failed': bool(events['turn.failed']),
        'diagnostic_signature_flags': {k: bool(re.search(v, text, re.I)) for k, v in signatures.items()},
        'response_file_exists': response_exists,
        'response_sha256': hashlib.sha256(response_path.read_bytes()).hexdigest() if response_exists else None,
        'response_accepted': False, 'remote_outcome': 'unknown', 'remote_billing': 'unknown',
        'retry_performed': False, 'budget_refunded': False,
        'raw_reasoning_recorded': False, 'raw_stderr_recorded': False,
        'interpretation': 'Observed signals only; flags do not establish root cause or remote outcome.',
    }
    atomic_write(folder / 'diagnostic.json', result)
    return result
