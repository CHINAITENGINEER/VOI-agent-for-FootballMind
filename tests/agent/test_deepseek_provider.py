"""Contract tests for the direct DeepSeek semantic inquiry provider.

These tests never contact DeepSeek.  A sender seam is used so that the tests
exercise the provider's request contract and its hostile-response boundary.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from football_agent.deepseek_provider import DeepSeekInquiryProvider


class DummyProtocol:
    version = "dummy-inquiry-v1"
    prompt = "Return the inquiry JSON only."
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }


PUBLIC = {"match": {"home": "A", "away": "B"}, "baseline": [0.4, 0.3, 0.3]}
SECRET = "sk-test-provider-secret"


def _completion(content='{"answer":"ok"}', *, usage=None, finish_reason="stop",
               choices=1, message_extra=None, model="deepseek-flash"):
    message = {"role": "assistant", "content": content}
    if message_extra:
        message.update(message_extra)
    choice = {"index": 0, "message": message, "finish_reason": finish_reason}
    return {
        "id": "chatcmpl-test",
        "model": model,
        "choices": [choice for _ in range(choices)],
        "usage": usage if usage is not None else {
            "prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19,
        },
    }


class Sender:
    def __init__(self, completion=None, error=None):
        self.completion = completion or _completion()
        self.error = error
        self.calls = []

    def __call__(self, url, *, payload, api_key, timeout_seconds):
        self.calls.append((url, payload, api_key, timeout_seconds))
        if self.error is not None:
            raise self.error
        return self.completion


@pytest.fixture
def provider(tmp_path):
    def make(sender, **kwargs):
        return DeepSeekInquiryProvider(
            tmp_path, protocol=DummyProtocol,
            api_key=SECRET, sender=sender, **kwargs,
        )
    return make


def test_success_parses_one_json_choice_and_preserves_sanitized_usage(provider, tmp_path):
    sender = Sender()
    result = provider(sender).decide(PUBLIC)

    assert result["response"] == {"answer": "ok"}
    assert result["receipt"]["provider"] == "deepseek-api"
    assert result["receipt"]["real_provider_call"] is False
    assert result["receipt"]["requests"] == 1
    assert result["receipt"]["usage"] == [{
        "prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19,
    }]
    assert result["receipt"]["model_requested"] == "deepseek-v4-pro"
    assert sender.calls[0][2] == SECRET
    assert len(sender.calls) == 1
    outputs = list(tmp_path.rglob("response.json"))
    assert len(outputs) == 1
    assert json.loads(outputs[0].read_text(encoding="utf-8")) == result["response"]


def test_configuration_has_model_and_protocol_but_never_api_key(provider):
    sender = Sender()
    p = provider(sender, model="deepseek-reasoner", timeout_seconds=37,
                 max_tokens=321, thinking="enabled")
    config = p.configuration()
    encoded = json.dumps(config, ensure_ascii=False)
    assert config["provider"] == "deepseek-api"
    assert config["model"] == "deepseek-reasoner"
    assert config["timeout_seconds"] == 37
    assert config["max_tokens"] == 321
    assert config["thinking"] == "enabled"
    assert config["inquiry_protocol"] == DummyProtocol.version
    assert SECRET not in encoded
    assert "api_key" not in config


def test_request_is_structured_json_single_call_without_tools(provider):
    sender = Sender()
    provider(sender, model="deepseek-chat", max_tokens=777,
             thinking="disabled", timeout_seconds=23).decide(PUBLIC)
    url, payload, key, timeout = sender.calls[0]
    assert url == "https://api.deepseek.com/chat/completions"
    assert key == SECRET
    assert timeout == 23
    assert payload["model"] == "deepseek-chat"
    assert payload["max_tokens"] == 777
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["response_format"] == {"type": "json_object"}
    assert "tools" not in payload
    messages = payload["messages"]
    system = next(item["content"] for item in messages if item["role"] == "system")
    user = next(item["content"] for item in messages if item["role"] == "user")
    assert DummyProtocol.prompt in system
    # Check the actual schema object, independent of formatting or markers.
    decoder = json.JSONDecoder()
    embedded = []
    for position, character in enumerate(system):
        if character == "{":
            try:
                embedded.append(decoder.raw_decode(system[position:])[0])
            except ValueError:
                pass
    assert DummyProtocol.schema in embedded
    assert json.loads(user.split("PUBLIC_STATE_JSON:", 1)[1]) == PUBLIC
    assert len(sender.calls) == 1


@pytest.mark.parametrize("completion", [
    _completion(choices=0),
    _completion(finish_reason="length"),
    _completion(choices=2),
    _completion(message_extra={"tool_calls": [{"id": "x"}]}),
    _completion(content="not-json"),
    _completion(content='{"answer":NaN}'),
    _completion(content='{"answer":1,"answer":2}'),
    _completion(content=""),
    _completion(content="   "),
    _completion(content=None),
    _completion(content="[]"),
    _completion(content='{"answer":1e999}'),
    _completion(content='{"answer":{"bad":1e999}}'),
    _completion(content='{"answer":[1e999]}'),
    _completion(message_extra={"refusal": "cannot comply"}),
    _completion(message_extra={"function_call": {"name": "read_url", "arguments": "{}"}}),
    _completion(finish_reason=None),
    _completion(finish_reason="tool_calls"),
])
def test_hostile_or_ambiguous_completion_is_rejected(completion, provider):
    with pytest.raises(RuntimeError):
        provider(Sender(completion=completion)).decide(PUBLIC)


@pytest.mark.parametrize("usage", [
    {"prompt_tokens": "12", "completion_tokens": 7, "total_tokens": 19},
    {"prompt_tokens": 12, "completion_tokens": -1, "total_tokens": 11},
    {"prompt_tokens": True},
    {"prompt_tokens": 10**14},
    {"completion_tokens_details": {"reasoning_tokens": "9"}},
    {"prompt_tokens_details": {"cached_tokens": -1}},
    {"prompt_tokens_details": "invalid"},
])
def test_usage_must_be_integer_counters_only(usage, provider):
    with pytest.raises(RuntimeError):
        provider(Sender(completion=_completion(usage=usage))).decide(PUBLIC)


def test_usage_preserves_known_nested_counts_and_strips_unknown_metadata(provider):
    usage = {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19,
             "prompt_cache_hit_tokens": 8, "prompt_cache_miss_tokens": 4,
             "completion_tokens_details": {"reasoning_tokens": 3,
                                           "arbitrary_text": "do-not-copy"},
             "prompt_tokens_details": {"cached_tokens": 8,
                                       "arbitrary_text": "do-not-copy"},
             "debug_info": {"authorization": SECRET}}
    result = provider(Sender(completion=_completion(usage=usage))).decide(PUBLIC)
    assert result["receipt"]["usage"] == [{
        "prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19,
        "prompt_cache_hit_tokens": 8, "prompt_cache_miss_tokens": 4,
        "completion_tokens_details": {"reasoning_tokens": 3},
        "prompt_tokens_details": {"cached_tokens": 8},
    }]
    assert SECRET not in json.dumps(result["receipt"])


def test_persisted_receipt_retains_counters_and_failed_usage(provider):
    p = provider(Sender(completion=_completion(usage={'prompt_cache_hit_tokens': 8,
                          'completion_tokens': 7}, finish_reason='length')))
    with pytest.raises(RuntimeError):
        p.decide(PUBLIC)
    saved = json.loads((Path(p.last_attempt_folder)/'receipt.json').read_text())
    assert saved == p.last_receipt
    assert saved['usage'] == [{'prompt_cache_hit_tokens': 8, 'completion_tokens': 7}]
    assert saved['status'] == 'failed'
    assert saved['latency_seconds'] >= 0


def test_runtime_keeps_failed_provider_receipt_in_budget(tmp_path):
    from football_agent.semantic_scenario_runtime import SemanticScenarioRuntime
    runtime = object.__new__(SemanticScenarioRuntime)
    runtime.calls, runtime.semantic_call_budget = [], 3
    runtime.output, runtime.progress = tmp_path, lambda _: None
    runtime.provider_factory = lambda folder, protocol: DeepSeekInquiryProvider(
        folder, protocol=protocol, api_key=SECRET,
        sender=Sender(completion=_completion(finish_reason='length')))
    with pytest.raises(RuntimeError):
        runtime._semantic(DummyProtocol(), PUBLIC, lambda x: x)
    assert len(runtime.calls) == 1
    saved = json.loads((tmp_path/'semantic-calls/record-000.json').read_text())
    assert saved['status'] == 'failed'
    assert saved['receipt']['usage'][0]['total_tokens'] == 19


def test_http_boundary_does_not_follow_redirect_or_expose_error_body(monkeypatch):
    import httpx
    original = httpx.AsyncClient
    def factory(**kwargs):
        assert kwargs['follow_redirects'] is False
        return original(**kwargs, transport=httpx.MockTransport(
            lambda _: httpx.Response(302, headers={'location': 'https://other.example'},
                                    text=SECRET)))
    monkeypatch.setattr(httpx, 'AsyncClient', factory)
    with pytest.raises(RuntimeError, match='DEEPSEEK_HTTP_302') as error:
        DeepSeekInquiryProvider._request('https://api.deepseek.com/chat/completions',
            payload={}, api_key=SECRET, timeout_seconds=1)
    assert SECRET not in str(error.value)


def test_http_timeout_covers_entire_response(monkeypatch):
    import asyncio
    import httpx
    original = httpx.AsyncClient
    class SlowBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            while True:
                yield b' '
                await asyncio.sleep(.02)
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kwargs: original(**kwargs,
        transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=SlowBody()))))
    with pytest.raises(RuntimeError, match='DEEPSEEK_TIMEOUT'):
        DeepSeekInquiryProvider._request('https://api.deepseek.com/chat/completions',
            payload={}, api_key=SECRET, timeout_seconds=.07)


def test_sender_failure_is_safe_and_does_not_leak_key_or_provider_message(provider, tmp_path):
    sender = Sender(error=RuntimeError(f"authorization failed with {SECRET}"))
    with pytest.raises(RuntimeError) as caught:
        provider(sender).decide(PUBLIC)
    assert SECRET not in str(caught.value)
    assert "authorization failed" not in str(caught.value)
    assert len(sender.calls) == 1
    artifacts = list(tmp_path.rglob("*.json"))
    assert artifacts
    for artifact in artifacts:
        assert SECRET not in artifact.read_text(encoding="utf-8")


def test_missing_explicit_key_fails_before_sender_even_when_env_exists(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-other-test-env-key")
    sender = Sender()
    p = DeepSeekInquiryProvider(tmp_path,
                                protocol=DummyProtocol, api_key="",
                                sender=sender)
    with pytest.raises(RuntimeError):
        p.decide(PUBLIC)
    assert sender.calls == []


def test_business_schema_validation_remains_in_runtime_for_repair(provider):
    # The transport enforces JSON shape. The protocol validator in the runtime
    # must still see valid JSON with semantic errors to use its bounded repair.
    result = provider(Sender(completion=_completion(content='{"wrong":"field"}'))).decide(PUBLIC)
    assert result["response"] == {"wrong": "field"}


def test_response_model_identity_is_recorded_separately_from_request(provider):
    result = provider(Sender(completion=_completion(model="deepseek-response-version")),
                      model="deepseek-request-alias").decide(PUBLIC)
    assert result["receipt"]["model_requested"] == "deepseek-request-alias"
    assert result["receipt"]["response_model"] == "deepseek-response-version"


def test_enabled_thinking_uses_declared_bounded_effort(provider):
    sender = Sender()
    p = provider(sender, thinking='enabled')
    p.decide(PUBLIC)
    assert sender.calls[0][1]['reasoning_effort'] == 'low'
    assert p.configuration()['reasoning_effort'] == 'low'


@pytest.mark.parametrize("kwargs", [
    {"model": "../ bad model"}, {"max_tokens": 0}, {"max_tokens": True},
    {"max_tokens": -1}, {"thinking": "auto"}, {"timeout_seconds": 0},
])
def test_invalid_configuration_fails_before_network(provider, kwargs):
    sender = Sender()
    with pytest.raises(ValueError):
        provider(sender, **kwargs)
    assert sender.calls == []
