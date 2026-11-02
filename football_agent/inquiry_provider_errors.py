'''Extract bounded public CLI error messages, excluding reasoning and stderr.'''
import hashlib
import json
import re


def _redact_message(value):
    text = str(value)[:4096]
    for pattern in (
        r'https?://[^ ]+',
        r'(?i)(?:authorization|api[_-]?key|password|secret|cookie)[ :=]+[^ ,;]+',
        r'(?i)bearer +[^ ,;]+',
        r'(?i)(?:secret|private|sk)[_-][a-z0-9_-]+',
        r'[A-Za-z0-9_.-]{24,}',
    ):
        text = re.sub(pattern, '[REDACTED]', text)
    return ' '.join(text.split())[:500]


def public_error_summary(stdout):
    raw = b'' if stdout is None else stdout if isinstance(stdout, bytes) else stdout.encode('utf-8', errors='replace')
    output = []
    for line in raw[:1024 * 1024].splitlines():
        try:
            event = json.loads(line)
        except (ValueError, UnicodeError):
            continue
        if not isinstance(event, dict):
            continue
        kind, item = event.get('type'), event.get('item')
        payload = None
        if kind in ('item.started', 'item.updated', 'item.completed') and isinstance(item, dict) and item.get('type') == 'error':
            payload = item
        elif kind in ('error', 'turn.failed'):
            payload = event.get('error') if isinstance(event.get('error'), dict) else event
        if payload is None or not isinstance(payload.get('message'), str):
            continue
        message = payload['message']
        output.append({'event_type': kind, 'message': _redact_message(message),
                       'message_sha256': hashlib.sha256(message.encode()).hexdigest()})
        if len(output) >= 8:
            break
    return {'version': 'cli-public-errors-v1', 'items': output,
            'maximum_items': 8, 'maximum_message_characters': 500,
            'reasoning_excluded': True, 'stderr_excluded': True,
            'redaction': 'URLs, credential labels and token-shaped strings redacted; errors are not evidence'}
