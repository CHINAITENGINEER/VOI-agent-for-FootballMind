import json

from football_agent.inquiry_provider_errors import public_error_summary


def test_item_error_is_retained_and_reasoning_is_excluded():
    body = chr(10).join(json.dumps(e) for e in [
        {'type': 'item.completed', 'item': {'type': 'reasoning', 'text': 'PRIVATE_REASONING'}},
        {'type': 'item.completed', 'item': {'type': 'error', 'message': 'Model connection failed'}}])
    result = public_error_summary(body.encode())
    assert len(result['items']) == 1
    assert result['items'][0]['message'] == 'Model connection failed'
    assert 'PRIVATE_REASONING' not in json.dumps(result)


def test_error_summary_redacts_urls_credentials_and_token_shapes():
    error = 'HTTP 401 Authorization: SECRET_TOKEN; Bearer secret123; https://private.example/?token=abc'
    result = public_error_summary(json.dumps({'type': 'error', 'message': error}))
    message = result['items'][0]['message']
    assert 'HTTP 401' in message
    assert 'SECRET_TOKEN' not in message and 'secret123' not in message and 'private.example' not in message


def test_nested_failure_message_is_retained():
    result = public_error_summary(json.dumps({'type': 'turn.failed', 'error': {'message': 'Provider unavailable'}}))
    assert result['items'][0]['message'] == 'Provider unavailable'


def test_summary_ignores_partial_nonobject_and_nonmessage_data():
    body = chr(10).join(['{', 'null', '[]', json.dumps({'type': 'error', 'message': ['bad']})])
    assert public_error_summary(body)['items'] == []
    assert public_error_summary(None)['items'] == []


def test_summary_is_bounded():
    body = chr(10).join(json.dumps({'type': 'error', 'message': 'Failure ' * 1000}) for _ in range(20))
    result = public_error_summary(body)
    assert len(result['items']) == 8
    assert all(len(i['message']) <= 500 for i in result['items'])
