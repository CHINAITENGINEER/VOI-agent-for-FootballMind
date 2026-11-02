import json
from types import SimpleNamespace

import pytest

from football_agent.audit import stored_digest
from football_agent.experience_provider import ExperienceProvider
from football_agent.provider_contract import VERSION, build_receipt, verify_receipt
from football_agent.experience import ExperienceProtocol


def _events():
    return [
        {"type": "turn.started"},
        {"type": "item.completed", "item": {"type": "agent_message"}},
        {"type": "turn.completed", "usage": {"input_tokens": 1}},
    ]


def _receipt():
    public = {"task_id": "t1", "documents": []}
    return build_receipt(
        provider_version="experience-streamed-provider-v1", protocol=ExperienceProtocol,
        public=public, prompt="P", schema_bytes=b"schema", stream_bytes=b"{}",
        output_bytes=b"{}", stdout_bytes=b"{}", events=_events(), command=["codex"],
        disabled_features=("shell_tool",), enabled_features=("code_mode_host",),
        cleanup_after_completed=False, returncode=0, elapsed_seconds=0.1)


def test_contract_binds_public_state_protocol_and_no_tools():
    public = {"task_id": "t1", "documents": []}
    receipt = _receipt()
    checked = verify_receipt(receipt, public=public, protocol=ExperienceProtocol)
    assert checked["status"] == "verified"
    assert checked["semantic_validation"] == "delegated_to_experience_service"
    assert receipt["numeric_voi_calibrated"] is False


@pytest.mark.parametrize("field,mutator,error", [
    ("public_input_digest", lambda r: r.update(public_input_digest="x"), "PROVIDER_CONTRACT_DIGEST_MISMATCH"),
    ("tool_events", lambda r: r["capability_policy"].update(tool_events=["read_url"]), "PROVIDER_CONTRACT_DIGEST_MISMATCH"),
])
def test_tampered_receipt_is_rejected(field, mutator, error):
    receipt = _receipt()
    mutator(receipt)
    with pytest.raises(ValueError, match=error):
        verify_receipt(receipt, public={"task_id": "t1", "documents": []}, protocol=ExperienceProtocol)


def test_public_or_protocol_mismatch_is_detected_after_self_integrity():
    receipt = _receipt()
    receipt["public_input_digest"] = stored_digest({"task_id": "other", "documents": []})
    receipt["contract_digest"] = stored_digest({k:v for k,v in receipt.items() if k != "contract_digest"})
    with pytest.raises(ValueError, match="PROVIDER_PUBLIC_INPUT_DIGEST_MISMATCH"):
        verify_receipt(receipt, public={"task_id": "t1", "documents": []}, protocol=ExperienceProtocol)


def test_experience_provider_emits_contract_receipt(tmp_path):
    response = {
        "task_id": "t1", "action": "stop", "url": None, "query": None,
        "engine": None, "question_to_user": None, "unknown_to_resolve": "待确认",
        "useful_if": "来源明确回答", "why_worth_cost": "降低未知",
        "previous_return": "none",
        "assessment": {"status": "unresolved", "answer": "待确认",
                         "claims": [], "remaining_unknowns": ["待确认"]},
        "alternatives": [], "reason": "当前未有可核验来源",
    }
    def runner(command, **kwargs):
        output = command[command.index("--output-last-message") + 1]
        with open(output, "w", encoding="utf-8") as fh:
            json.dump(response, fh, ensure_ascii=False)
        return SimpleNamespace(returncode=0, stdout=json.dumps({"type":"turn.started"}) + "\n" +
            json.dumps({"type":"item.completed","item":{"type":"agent_message","text":json.dumps(response, ensure_ascii=False)}}) + "\n" +
            json.dumps({"type":"turn.completed"}), stderr="")
    provider = ExperienceProvider(tmp_path, protocol=ExperienceProtocol, runner=runner)
    provider.executable = "test-cli"
    result = provider.decide({"task_id": "t1", "documents": []})
    receipt = result["receipt"]
    assert receipt["contract_version"] == VERSION
    assert receipt["public_input_digest"] == stored_digest({"task_id": "t1", "documents": []})
    assert receipt["capability_policy"]["tool_events"] == []
    assert receipt["numeric_voi_calibrated"] is False
