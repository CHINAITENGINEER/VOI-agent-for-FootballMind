"""Direct, bounded DeepSeek transport; the runtime alone owns VoI decisions."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from uuid import uuid4

from .inquiry_contract import PROMPT, RESPONSE_SCHEMA


BASE_URL = "https://api.deepseek.com"
CHAT_COMPLETIONS_URL = BASE_URL + "/chat/completions"
DEFAULT_MODEL = "deepseek-v4-pro"
_IDENTIFIER = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._:/-]{0,119}\Z")
_TOKEN_COUNTS = {
    "prompt_tokens", "completion_tokens", "total_tokens", "input_tokens",
    "output_tokens", "reasoning_tokens", "cached_tokens",
    "prompt_cache_hit_tokens", "prompt_cache_miss_tokens",
}


class DeepSeekProviderError(RuntimeError):
    """Internal errors contain static codes, never remote messages or headers."""


def atomic_write(path, value):
    """Write sanitized provider artifacts without redacting token COUNTERS.

    Authorization headers and reasoning_content never enter these objects.
    Generic key-name redaction would incorrectly erase prompt_cache_hit_tokens
    and max_tokens and make the returned and persisted receipts disagree.
    """
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode('utf-8')
    fd, temporary = tempfile.mkstemp(prefix='.deepseek-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def resolve_api_key(api_key=None):
    """Read only an explicit value or DEEPSEEK_API_KEY in this process."""
    value = os.environ.get("DEEPSEEK_API_KEY") if api_key is None else api_key
    return value.strip() if isinstance(value, str) and value.strip() else None


def _strict_json(text):
    if not isinstance(text, str) or not text.strip():
        raise DeepSeekProviderError("DEEPSEEK_EMPTY_CONTENT")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise DeepSeekProviderError("DEEPSEEK_DUPLICATE_JSON_KEY")
            result[key] = value
        return result

    def reject(_):
        raise DeepSeekProviderError("DEEPSEEK_NONFINITE_JSON")

    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            reject(value)
        return result

    try:
        result = json.loads(text, object_pairs_hook=pairs, parse_constant=reject,
                            parse_float=finite_float)
    except (ValueError, RecursionError):
        raise DeepSeekProviderError("DEEPSEEK_INVALID_JSON") from None
    if not isinstance(result, dict):
        raise DeepSeekProviderError("DEEPSEEK_JSON_OBJECT_REQUIRED")
    return result


def _safe_usage(value):
    """Preserve documented counters; never copy arbitrary metadata strings."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise DeepSeekProviderError("DEEPSEEK_USAGE_INVALID")
    clean = {}
    for key, count in value.items():
        if key in _TOKEN_COUNTS:
            if type(count) is not int or not 0 <= count <= 10**12:
                raise DeepSeekProviderError("DEEPSEEK_USAGE_INVALID")
            clean[key] = count
        elif key in {"prompt_tokens_details", "completion_tokens_details"}:
            clean[key] = _safe_usage(count)
    return clean


async def _send_http(url, *, payload, api_key, timeout_seconds):
    import httpx

    async def send():
        # No redirects, SDK retries, tool execution, or provider fallback.
        async with httpx.AsyncClient(timeout=timeout_seconds, follow_redirects=False) as client:
            async with client.stream("POST", url, json=payload, headers={
                    "Authorization": "Bearer " + api_key,
                    "Accept": "application/json"}) as response:
                if not 200 <= response.status_code < 300:
                    raise DeepSeekProviderError(f"DEEPSEEK_HTTP_{response.status_code}")
                chunks, size = [], 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > 8 * 1024 * 1024:
                        raise DeepSeekProviderError("DEEPSEEK_RESPONSE_TOO_LARGE")
                    chunks.append(chunk)
                try:
                    return _strict_json(b"".join(chunks).decode("utf-8"))
                except UnicodeDecodeError:
                    raise DeepSeekProviderError("DEEPSEEK_RESPONSE_NOT_UTF8") from None
    try:
        # Covers connection, headers and the complete body even when the server
        # keeps sending whitespace; a socket inactivity timeout alone does not.
        return await asyncio.wait_for(send(), timeout=timeout_seconds)
    except (TimeoutError, httpx.TimeoutException):
        raise DeepSeekProviderError("DEEPSEEK_TIMEOUT") from None
    except httpx.HTTPError:
        raise DeepSeekProviderError("DEEPSEEK_TRANSPORT_ERROR") from None


class DeepSeekInquiryProvider:
    version = "deepseek-inquiry-json-v1"

    def __init__(self, workspace, *, protocol=None, model=None,
                 timeout_seconds=120, max_tokens=8192, thinking="enabled",
                 api_key=None, sender=None):
        self.workspace = Path(workspace).resolve()
        self.protocol = protocol
        self.model = DEFAULT_MODEL if model is None else model
        if not isinstance(self.model, str) or not _IDENTIFIER.fullmatch(self.model):
            raise ValueError("Explicit model must be a bounded model identifier")
        if type(timeout_seconds) not in (int, float) or not 1 <= timeout_seconds <= 180:
            raise ValueError("Provider deadline must be 1..180 seconds")
        if type(max_tokens) is not int or not 1 <= max_tokens <= 32768:
            raise ValueError("max_tokens must be an integer between 1 and 32768")
        if thinking not in {"enabled", "disabled"}:
            raise ValueError("thinking must be enabled or disabled")
        self.timeout_seconds, self.max_tokens, self.thinking = timeout_seconds, max_tokens, thinking
        self._api_key, self._sender = resolve_api_key(api_key), sender
        self.last_attempt_folder, self.last_receipt = None, None

    def configuration(self):
        return {"provider": "deepseek-api", "version": self.version,
            "configured": bool(self._api_key), "authentication": "process_env_or_explicit_argument",
            "base_url": BASE_URL, "model": self.model, "thinking": self.thinking,
            "reasoning_effort": "low" if self.thinking == "enabled" else None,
            "max_tokens": self.max_tokens, "timeout_seconds": self.timeout_seconds,
            "response_format": "json_object_with_local_schema_and_semantic_validation",
            "inquiry_protocol": getattr(self.protocol, "version", None),
            "tools": [], "automatic_retries": 0,
            "auth_validated_by_actual_call": False, "cost_status": "usage_only_price_not_inferred"}

    def _payload(self, public):
        prompt = self.protocol.prompt if self.protocol else PROMPT
        schema = self.protocol.schema if self.protocol else RESPONSE_SCHEMA
        system = (prompt + "\nReturn ONLY a JSON object conforming to OUTPUT_JSON_SCHEMA. "
                  "Do not emit Markdown, tool calls or commentary.\nOUTPUT_JSON_SCHEMA:\n" +
                  json.dumps(schema, ensure_ascii=False, allow_nan=False))
        payload = {"model": self.model, "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "PUBLIC_STATE_JSON:\n" +
                json.dumps(public, ensure_ascii=False, allow_nan=False)}],
            "response_format": {"type": "json_object"}, "thinking": {"type": self.thinking},
            "max_tokens": self.max_tokens, "stream": False}
        if self.thinking == "enabled":
            payload["reasoning_effort"] = "low"
        return payload

    @staticmethod
    def _request(url, *, payload, api_key, timeout_seconds):
        return asyncio.run(_send_http(url, payload=payload, api_key=api_key,
                                     timeout_seconds=timeout_seconds))

    def decide(self, public):
        self.last_receipt = None
        if not self._api_key:
            raise DeepSeekProviderError("DEEPSEEK_API_KEY_NOT_CONFIGURED")
        payload = self._payload(public)
        if self._api_key in json.dumps(payload, ensure_ascii=False):
            raise DeepSeekProviderError("DEEPSEEK_CREDENTIAL_IN_INPUT")
        folder = self.workspace / str(uuid4())
        folder.mkdir(parents=True)
        self.last_attempt_folder = str(folder)
        atomic_write(folder / "input.json", public)
        atomic_write(folder / "response.schema.json", self.protocol.schema if self.protocol else RESPONSE_SCHEMA)
        atomic_write(folder / "configuration.json", self.configuration())
        started = time.monotonic()
        receipt = {"provider": "deepseek-api", "version": self.version,
            "real_provider_call": self._sender is None, "requests": 1,
            "requests_unit": "chat_completion_http_attempt", "usage": [],
            "latency_seconds": 0., "model_requested": self.model, "response_model": None,
            "request_sha256": hashlib.sha256(json.dumps(payload, ensure_ascii=False,
                sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest(),
            "tool_events": [], "actual_money": None, "money_status": "not_reported",
            "thinking": self.thinking, "status": "pending", "automatic_retries": 0}
        try:
            result = (self._sender or self._request)(CHAT_COMPLETIONS_URL, payload=payload,
                api_key=self._api_key, timeout_seconds=self.timeout_seconds)
            if not isinstance(result, dict) or result.get("error") is not None:
                raise DeepSeekProviderError("DEEPSEEK_RESPONSE_INVALID")
            receipt["usage"] = [_safe_usage(result.get("usage"))]
            model = result.get("model")
            if isinstance(model, str) and _IDENTIFIER.fullmatch(model) and self._api_key not in model:
                receipt["response_model"] = model
            choices = result.get("choices")
            if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
                raise DeepSeekProviderError("DEEPSEEK_SINGLE_CHOICE_REQUIRED")
            choice = choices[0]
            message = choice.get("message")
            if not isinstance(message, dict) or message.get("role") != "assistant":
                raise DeepSeekProviderError("DEEPSEEK_MESSAGE_INVALID")
            if message.get("tool_calls") or message.get("function_call"):
                raise DeepSeekProviderError("DEEPSEEK_UNEXPECTED_TOOL_CALL")
            if message.get("refusal"):
                raise DeepSeekProviderError("DEEPSEEK_REFUSAL")
            if choice.get("finish_reason") != "stop":
                raise DeepSeekProviderError("DEEPSEEK_INCOMPLETE_OUTPUT")
            content = message.get("content")
            if isinstance(content, str) and self._api_key in content:
                raise DeepSeekProviderError("DEEPSEEK_CREDENTIAL_IN_OUTPUT")
            response = _strict_json(content)
            atomic_write(folder / "response.json", response)
            receipt["response_sha256"] = hashlib.sha256((folder / "response.json").read_bytes()).hexdigest()
            receipt["content_sha256"] = hashlib.sha256(content.encode("utf-8")).hexdigest()
            receipt["status"] = "completed"
            return {"response": response, "receipt": receipt}
        except Exception as exc:
            code = str(exc) if type(exc) is DeepSeekProviderError else "DEEPSEEK_REQUEST_FAILED"
            receipt.update(status="failed", error_code=code)
            atomic_write(folder / "failure.json", {"error_code": code,
                "remote_billing": "unknown", "automatic_retry": False})
            raise DeepSeekProviderError(code) from None
        finally:
            receipt["latency_seconds"] = time.monotonic() - started
            self.last_receipt = receipt
            atomic_write(folder / "receipt.json", receipt)
