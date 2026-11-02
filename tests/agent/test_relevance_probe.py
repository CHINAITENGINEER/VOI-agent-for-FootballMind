from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from football_agent import relevance_probe as probe
from football_agent.audit import atomic_write, digest
from football_agent.decision_risk_experiment import make_parent, proposal


@pytest.fixture
def source(tmp_path):
    clock = datetime.now(timezone.utc)-timedelta(minutes=1)
    parent_root = tmp_path/"parent"
    parent = make_parent(parent_root, clock=lambda: clock)
    return {"root": str(parent_root), "run_id": parent["run_id"],
            "head_hash": parent["events"][-1]["event_hash"],
            "document_id": parent["documents"][0]["document_id"], "origin": "constructed"}


class ScriptedProvider:
    def decide(self, public):
        return {"response": proposal([], kind="defer"), "receipt": {"real_provider_call": False}}


def completed(tmp_path, source, *, factory=None):
    root = tmp_path/"probe"
    manifest = probe.prepare(root, source, provider_kind="constructed")
    for case in manifest["cases"]:
        probe.execute_case(root, case, manifest["allowed_hosts"],
                           provider_factory=factory or (lambda folder, protocol: ScriptedProvider()))
    return root, manifest


def test_matched_interventions_preserve_time_and_remove_all_document_material(tmp_path, source, monkeypatch):
    def forbid(*args, **kwargs):
        raise AssertionError("No new source execution is allowed")
    monkeypatch.setattr(probe.OpenWebTools, "execute", forbid)
    root, manifest = completed(tmp_path, source)
    inputs = [probe.read_json(root/c/"input.json") for c in probe.CASES]
    original = probe.read_json(root/"inherited-source.json")["documents"][0]
    assert inputs[0]["documents"] == [original]
    assert inputs[1]["documents"] == [original]
    assert inputs[2]["documents"] == []
    assert inputs[0]["task"]["objective"] != inputs[1]["task"]["objective"]
    task_control = deepcopy(inputs[1])
    task_control["task"]["objective"] = inputs[0]["task"]["objective"]
    evidence_control = deepcopy(inputs[2])
    evidence_control["documents"] = inputs[0]["documents"]
    assert task_control == evidence_control == inputs[0]
    assert inputs[2]["task"]["seed_urls"] == [original["url"]]
    for field in ("claims", "questions", "hypotheses", "context_bindings", "requirements", "action_history"):
        assert inputs[2][field] == []
    assert original["text"] not in json.dumps(inputs[2])
    proof = probe.verify_probe(root)
    assert proof["all_cases_retained"] and proof["real_completed_provider_calls"] == 0
    assert proof["new_source_requests"] == proof["new_football_observations"] == 0
    assert not proof["semantic_truth_independently_verified"]
    with pytest.raises(ValueError, match="attempted once"):
        probe.execute_case(root, manifest["cases"][0], manifest["allowed_hosts"],
                           provider_factory=lambda folder, protocol: ScriptedProvider())


@pytest.mark.parametrize("stage", ["initialization", "provider_call", "validation"])
def test_failures_are_retained_without_preventing_other_cases(tmp_path, source, stage):
    class FailingProvider:
        def decide(self, public):
            if stage == "provider_call":
                raise RuntimeError("test provider failure")
            reply = ScriptedProvider().decide(public)
            reply["response"]["decision"]["kind"] = "invalid-action"
            return reply
    def factory(folder, protocol):
        if folder.parent.name == probe.CASES[0]:
            if stage == "initialization":
                raise RuntimeError("test initialization failure")
            return FailingProvider()
        return ScriptedProvider()
    root, _ = completed(tmp_path, source, factory=factory)
    proof = probe.verify_probe(root)
    assert [r["status"] for r in proof["cases"]] == ["FAILED_PROBE_NO_RETRY"]+["VALIDATED_PROPOSAL_NOT_EXECUTION"]*2
    assert proof["cases"][0]["failure_stage"] == stage


@pytest.mark.parametrize("tamper", ["failure_relabel", "unknown_status", "case_identity", "invocation_hash"])
def test_result_or_invocation_relabeling_is_rejected(tmp_path, source, tamper):
    root, _ = completed(tmp_path, source)
    path = root/probe.CASES[0]/("invocation.json" if tamper == "invocation_hash" else "result.json")
    saved = probe.read_json(path)
    if tamper == "failure_relabel":
        saved.update(status="FAILED_PROBE_NO_RETRY", projection=None, failure_stage="validation", error_type="ValueError")
    elif tamper == "unknown_status":
        saved["status"] = "SUCCESS"
    elif tamper == "case_identity":
        saved["case_id"] = probe.CASES[1]
    else:
        saved["input_hash"] = "0"*64
    atomic_write(path, saved)
    with pytest.raises(ValueError):
        probe.verify_probe(root)


def test_rehashed_extra_input_change_does_not_pass_as_matched_intervention(tmp_path, source):
    root, manifest = completed(tmp_path, source)
    path = root/probe.CASES[1]/"input.json"
    public = probe.read_json(path)
    public["budget"]["remaining_requests"] = 8
    atomic_write(path, public)
    manifest["cases"][1]["input_hash"] = digest(public)
    atomic_write(root/"manifest.json", manifest)
    with pytest.raises(ValueError, match="intervention"):
        probe.verify_probe(root)


@pytest.mark.parametrize("tamper", ["head", "origin"])
def test_source_pin_and_constructed_origin_cannot_be_upgraded(tmp_path, source, tamper):
    if tamper == "head":
        source["head_hash"] = "0"*64
    else:
        source["origin"] = "operational_network"
    with pytest.raises(ValueError):
        probe.prepare(tmp_path/"probe", source)


@pytest.mark.parametrize("tamper", ["provider_input", "hidden_reply", "second_attempt"])
def test_real_provider_receipt_binds_case_and_retains_completed_calls(tmp_path, source, monkeypatch, tamper):
    # Only the CLI process boundary is simulated; no real CLI or network call.
    original_provider = probe.CodexInquiryProvider
    def runner(command, **kwargs):
        path = command[command.index("--output-last-message")+1]
        atomic_write(Path(path), proposal([], kind="defer"))
        return SimpleNamespace(returncode=0, stdout=json.dumps({"type": "turn.completed", "usage": {}}), stderr="")
    monkeypatch.setattr(probe, "CodexInquiryProvider", lambda folder, **kwargs:
        original_provider(folder, executable="simulated-cli", runner=runner, **kwargs))
    root = tmp_path/"probe"
    manifest = probe.prepare(root, source)
    for case in manifest["cases"]:
        probe.execute_case(root, case, manifest["allowed_hosts"])
    assert probe.verify_probe(root)["real_completed_provider_calls"] == 3  # simulated receipt flag only
    call = next((root/probe.CASES[0]/"provider").glob("*/input.json"))
    if tamper == "provider_input":
        public = probe.read_json(call)
        public["documents"] = []
        atomic_write(call, public)
    elif tamper == "hidden_reply":
        (root/probe.CASES[0]/"reply.json").unlink()
        result = probe.read_json(root/probe.CASES[0]/"result.json")
        result.update(status="FAILED_PROBE_NO_RETRY", projection=None, failure_stage="provider_call", error_type="RuntimeError")
        atomic_write(root/probe.CASES[0]/"result.json", result)
    else:
        (call.parent.parent/"another-call").mkdir()
    with pytest.raises(ValueError):
        probe.verify_probe(root)
