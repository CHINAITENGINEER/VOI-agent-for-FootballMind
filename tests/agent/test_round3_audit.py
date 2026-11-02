from football_agent.audit import redact, atomic_write
import json


def test_cost_vector_and_numeric_token_usage_survive_without_credentials(tmp_path):
    value = {"tokens": {"amount": 0, "basis": "simulated", "unit": "token"},
             "lambda_cost": {"tokens": 0.001}, "usage": {"input_tokens": 9, "output_tokens": 3},
             "api_key": "private", "access_token": "private", "other": {"tokens": "private"}}
    path = tmp_path / "record.json"
    atomic_write(path, value)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["tokens"] == value["tokens"]
    assert saved["lambda_cost"] == value["lambda_cost"]
    assert saved["usage"] == value["usage"]
    assert saved["api_key"] == saved["access_token"] == saved["other"]["tokens"] == "[REDACTED]"


def test_token_measure_does_not_allow_strings_or_nested_credentials():
    assert redact({"input_tokens": "credential"})["input_tokens"] == "[REDACTED]"
    assert redact({"tokens": {"amount": "credential", "unit": "token", "basis": "unknown"}})["tokens"] == "[REDACTED]"
    assert redact({"tokens": {"amount": 2, "unit": "token", "basis": "known", "secret": "credential"}})["tokens"] == "[REDACTED]"
