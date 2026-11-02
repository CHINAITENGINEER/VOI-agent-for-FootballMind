"""Matched task/evidence interventions on a pinned, acquired source document.

The provider proposes actions only. No proposed query is executed, no source
is refreshed, and no return model or policy benefit is estimated. A three-case
plan is frozen before any provider call; every outcome, including failure, stays.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

from football_spider.storage.raw_archive import RawArchive
from .audit import RunStore, atomic_write, digest
from .inquiry_contract import utc
from .inquiry_evidence_state import EvidenceStateProtocol
from .inquiry_memory import load_inherited
from .inquiry_provider import CodexInquiryProvider
from .inquiry_tools import OpenWebTools


VERSION = "acquired-evidence-relevance-probe-v1"
CASES = ("source-report-with-document", "match-lineup-with-document", "source-report-without-document")
COMMON_OBJECTIVE = (
    "这是固定资料下的行动提议诊断，网页工具在本诊断中只登记提议、不执行。"
    "保持来源原始获取时间；不使用记忆中的比赛事实补充观察。"
    "根据任务目标和当前材料，自主决定查询、反思、停止、等待或暂缓，并说明理由。"
    "不要求用完请求额度，也不要求凑足候选数量。"
)
SOURCE_TASK = (
    "核对教练在2026年9月18日发布的采访是否表述Trent Alexander-Arnold回归这次国家队大名单。"
    "交付是有原文引证的来源陈述；本任务不要求确认后来是否发生人员变化。"
)
LINEUP_TASK = (
    "以2026年9月29日捷克主场对英格兰为任务设定，判断Trent Alexander-Arnold在该场是否首发。"
    "交付需要具有本场适用范围的人员信息，保留证据不足和来源正确性未知。"
    "比赛标识是本诊断给定的研究对象，不代表本次新采集确认的事实。"
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _source_record(reference, now, deadline):
    expected = {"root", "run_id", "head_hash", "document_id", "origin"}
    if set(reference) != expected or reference["origin"] not in {"operational_network", "constructed"}:
        raise ValueError("Pinned source identity and explicit real/constructed origin required")
    run = RunStore(Path(reference["root"])/"runs").load(reference["run_id"])
    mode = run["configuration"]["tools"].get("execution_mode")
    if reference["origin"] == "operational_network" and (
            mode == "constructed_transport_no_network"
            or not run["budget"]["llm_usage"]
            or any(not r.get("real_provider_call") for r in run["budget"]["llm_usage"])):
        raise ValueError("A constructed source must not be labeled an operational acquisition")
    allowed = run["configuration"]["tools"]["allowed_hosts"]
    inherited = load_inherited(reference["root"], reference["run_id"], reference["head_hash"],
        [reference["document_id"]], now=now, deadline=deadline, allowed_hosts=allowed)
    return inherited, allowed


def _public_cases(inherited, allowed, now, deadline):
    protocol = EvidenceStateProtocol()
    document = deepcopy(inherited["documents"][0])
    # Only the already known URL is a seed. Removing the document also removes
    # its discovered links; they are not silently retained as new entry points.
    tools = OpenWebTools(RawArchive(Path("unused-relevance-probe-archive")), allowed)
    base = {"task": {"task_id": "acquired-evidence-relevance-probe", "objective": COMMON_OBJECTIVE+SOURCE_TASK,
        "deadline": utc(deadline).isoformat(), "initial_facts": [], "seed_urls": [document["url"]]},
        "now": utc(now).isoformat(), "questions": [], "claims": [], "documents": [document],
        "action_history": [], "last_decision": None,
        "budget": {"remaining_requests": 2, "remaining_llm_calls": 1},
        "tools": {**tools.configuration(), "execution_mode": "proposal_only_no_source_requests"},
        "value_model_status": "UNSUPPORTED_OPEN_TEXT", "prediction_status": "OPEN_TEXT_UPDATE_NOT_CONFIGURED"}
    base.update(protocol.public(base))
    cases = {name: deepcopy(base) for name in CASES}
    cases[CASES[1]]["task"]["objective"] = COMMON_OBJECTIVE+LINEUP_TASK
    cases[CASES[2]]["documents"] = []
    _check_interventions(cases)
    return cases


def _check_interventions(cases):
    first, task_changed, removed = (deepcopy(cases[name]) for name in CASES)
    task_changed["task"]["objective"] = first["task"]["objective"]
    if first != task_changed:
        raise ValueError("Task pair changed fields other than the task objective")
    removed["documents"] = deepcopy(first["documents"])
    if first != removed:
        raise ValueError("Evidence pair changed fields other than acquired documents")


def prepare(root, reference, *, now=None, provider_kind="codex_cli"):
    root = Path(root).resolve()
    if root.exists():
        raise ValueError("Use a new root; preserve all previous probe attempts")
    if provider_kind not in {"codex_cli", "constructed"}:
        raise ValueError("An explicit real or constructed provider is required")
    now = utc(now or datetime.now(timezone.utc))
    deadline = now+timedelta(hours=2)
    inherited, allowed = _source_record(reference, now, deadline)
    cases = _public_cases(inherited, allowed, now, deadline)
    root.mkdir(parents=True)
    atomic_write(root/"source-reference.json", reference)
    atomic_write(root/"inherited-source.json", inherited)
    records = []
    for name, public in cases.items():
        atomic_write(root/name/"input.json", public)
        records.append({"case_id": name, "input_hash": digest(public), "documents_present": bool(public["documents"])})
    source_root = Path(__file__).resolve().parent
    paths = sorted(set(source_root.glob("inquiry*.py")) | {Path(__file__), source_root/"audit.py"})
    frozen = []
    for source in paths:
        target = root/"execution-source"/source.name
        target.parent.mkdir(exist_ok=True)
        body = source.read_bytes()
        target.write_bytes(body)
        frozen.append({"path": str(source), "snapshot": str(target.relative_to(root)),
                       "sha256": hashlib.sha256(body).hexdigest()})
    manifest = {"version": VERSION, "created_at": now.isoformat(), "deadline": deadline.isoformat(),
        "source_origin": reference["origin"], "source_reference_hash": digest(reference),
        "inherited_source_hash": digest(inherited), "allowed_hosts": allowed, "cases": records,
        "sources": frozen, "provider_kind": provider_kind, "planned_provider_calls": 3, "max_concurrency": 2, "retries": 0,
        "source_requests_allowed": 0, "new_football_observations": 0,
        "scope": "matched_proposal_diagnostic_not_execution_optimality_or_policy_effectiveness",
        "source_pair_difference": "task.objective_only", "evidence_pair_difference": "documents_only",
        "initial_claims_are_preannotated": False, "numerical_value_model": None}
    atomic_write(root/"manifest.json", manifest)
    return manifest


def _projection(response, public, allowed):
    protocol = EvidenceStateProtocol()
    protocol.validate(response, public)
    tools = OpenWebTools(RawArchive(Path("unused-relevance-probe-archive")), allowed)
    prepared = {c["candidate_id"]: tools.prepare(c, public) for c in response["candidates"]}
    choice = response["decision"]
    return {"decision": deepcopy(choice), "proposed_tool": prepared.get(choice["candidate_id"]),
        "candidate_count": len(response["candidates"]), "candidates": deepcopy(response["candidates"]),
        "claim_updates": deepcopy(response["claim_updates"]), "hypotheses": deepcopy(response["hypotheses"]),
        "expected_gain": None, "executed_queries": 0, "utility": None}


def execute_case(root, case, allowed, *, provider_factory=None):
    root = Path(root)
    manifest = read_json(root/"manifest.json")
    if case not in manifest["cases"] or allowed != manifest["allowed_hosts"]:
        raise ValueError("Execution must use the predeclared case and host scope")
    if (provider_factory is not None) != (manifest["provider_kind"] == "constructed"):
        raise ValueError("Provider origin must match the frozen experiment plan")
    folder = root/case["case_id"]
    if (folder/"result.json").exists() or (folder/"invocation.json").exists():
        raise ValueError("Each predeclared case may be attempted once; no outcome-based retries")
    public = read_json(folder/"input.json")
    if digest(public) != case["input_hash"]:
        raise ValueError("The case input changed after pre-registration")
    atomic_write(folder/"invocation.json", {"started_at": datetime.now(timezone.utc).isoformat(),
        "input_hash": case["input_hash"], "provider_kind": manifest["provider_kind"], "source_requests_allowed": 0})
    result = {"case_id": case["case_id"], "source_requests_executed": 0}
    stage = "initialization"
    try:
        protocol = EvidenceStateProtocol()
        provider = (provider_factory(folder/"provider", protocol) if provider_factory else
                    CodexInquiryProvider(folder/"provider", protocol=protocol, timeout_seconds=180))
        stage = "provider_call"
        reply = provider.decide(public)
        atomic_write(folder/"reply.json", reply)
        stage = "validation"
        result.update(status="VALIDATED_PROPOSAL_NOT_EXECUTION", projection=_projection(reply["response"], public, allowed))
    except Exception as exc:
        result.update(status="FAILED_PROBE_NO_RETRY", error_type=type(exc).__name__,
                      error=str(exc)[:600], failure_stage=stage, projection=None)
    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write(folder/"result.json", result)
    return result


def verify_probe(root):
    root = Path(root).resolve()
    manifest = read_json(root/"manifest.json")
    if manifest["version"] != VERSION or tuple(c["case_id"] for c in manifest["cases"]) != CASES:
        raise ValueError("Probe plan changed or omitted a case")
    reference, inherited = read_json(root/"source-reference.json"), read_json(root/"inherited-source.json")
    if digest(reference) != manifest["source_reference_hash"] or digest(inherited) != manifest["inherited_source_hash"]:
        raise ValueError("Pinned source record changed")
    reproduced, allowed = _source_record(reference, manifest["created_at"], manifest["deadline"])
    if manifest["allowed_hosts"] != allowed or manifest["source_origin"] != reference["origin"]:
        raise ValueError("Source origin or host scope differs from the pinned source")
    if (manifest["provider_kind"] not in {"codex_cli", "constructed"}
            or manifest["planned_provider_calls"] != 3 or manifest["max_concurrency"] != 2
            or manifest["retries"] != 0 or manifest["source_requests_allowed"] != 0
            or manifest["new_football_observations"] != 0 or manifest["numerical_value_model"] is not None):
        raise ValueError("Probe acquisition, provider or value scope changed")
    if ({k:v for k,v in reproduced.items() if k != "parent_verification"}
            != {k:v for k,v in inherited.items() if k != "parent_verification"}):
        raise ValueError("Acquired source differs from the pinned original")
    expected_cases = _public_cases(inherited, allowed, manifest["created_at"], manifest["deadline"])
    for record in manifest["sources"]:
        if hashlib.sha256((root/record["snapshot"]).read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Execution source snapshot changed")
    rows, real_calls = [], 0
    for case in manifest["cases"]:
        folder = root/case["case_id"]
        public, result = read_json(folder/"input.json"), read_json(folder/"result.json")
        invocation = read_json(folder/"invocation.json")
        if public != expected_cases[case["case_id"]] or digest(public) != case["input_hash"]:
            raise ValueError("Frozen inputs differ from the declared task/evidence intervention")
        if result["source_requests_executed"] != 0:
            raise ValueError("A proposal probe must not execute an external query")
        if (invocation["input_hash"] != case["input_hash"] or invocation["source_requests_allowed"] != 0
                or invocation["provider_kind"] != manifest["provider_kind"]
                or not utc(manifest["created_at"]) <= utc(invocation["started_at"]) <= utc(result["finished_at"])):
            raise ValueError("Invocation must bind the frozen case before its result")
        if result["case_id"] != case["case_id"] or result["status"] not in {
                "VALIDATED_PROPOSAL_NOT_EXECUTION", "FAILED_PROBE_NO_RETRY"}:
            raise ValueError("Unrecognized or mismatched case result")
        reply_path = folder/"reply.json"
        if manifest["provider_kind"] == "codex_cli":
            provider_calls = [p for p in (folder/"provider").glob("*") if p.is_dir()]
            if len(provider_calls) > 1:
                raise ValueError("More than one provider attempt was archived for a frozen case")
            for call in provider_calls:
                if (call/"input.json").exists() and read_json(call/"input.json") != public:
                    raise ValueError("Provider input/output differs from the fixed case")
                if not reply_path.exists() and (call/"receipt.json").exists():
                    raise ValueError("Completed provider receipt must retain its reply")
        if reply_path.exists():
            reply = read_json(reply_path)
            real = bool(reply["receipt"].get("real_provider_call"))
            if real != (manifest["provider_kind"] == "codex_cli"):
                raise ValueError("Provider receipt origin differs from the frozen plan")
            real_calls += int(real)
            try:
                projected = _projection(reply["response"], public, allowed)
            except Exception as exc:
                if (result["status"] != "FAILED_PROBE_NO_RETRY" or result["projection"] is not None
                        or result.get("failure_stage") != "validation" or result.get("error_type") != type(exc).__name__
                        or result.get("error") != str(exc)[:600]):
                    raise ValueError("Invalid reply must retain its reproducible validation failure") from exc
            else:
                if result["status"] != "VALIDATED_PROPOSAL_NOT_EXECUTION" or result["projection"] != projected:
                    raise ValueError("Saved decision differs from the validated provider proposal")
            if real:
                matches = [path.parent for path in (folder/"provider").glob("*/receipt.json")
                           if read_json(path) == reply["receipt"]]
                if len(matches) != 1:
                    raise ValueError("Completed provider receipt must bind exactly one saved call")
                call = matches[0]
                if read_json(call/"input.json") != public or read_json(call/"response.json") != reply["response"]:
                    raise ValueError("Provider input/output differs from the fixed case")
                if hashlib.sha256((call/"response.json").read_bytes()).hexdigest() != reply["receipt"]["response_sha256"]:
                    raise ValueError("Provider response byte hash mismatch")
                process = read_json(call/"process.json")
                if process["returncode"] != 0 or process["unexpected_item_types"] or "turn.completed" not in process["event_types"]:
                    raise ValueError("Provider turn completion or tool-isolation check failed")
        elif (result["status"] != "FAILED_PROBE_NO_RETRY" or result["projection"] is not None
                or result.get("failure_stage") not in {"initialization", "provider_call"}
                or not result.get("error_type")):
            raise ValueError("A missing reply must retain an initialization or provider failure")
        rows.append(result)
    return {"status": "VERIFIED_MATCHED_PROPOSALS_NOT_POLICY_EFFECTIVENESS", "version": VERSION,
        "cases": rows, "real_completed_provider_calls": real_calls,
        "source_origin": reference["origin"], "source_documents_reused": 1,
        "new_source_requests": 0, "new_football_observations": 0,
        "semantic_truth_independently_verified": False, "counterfactual_utility_estimated": False,
        "all_cases_retained": True, "task_intervention_verified": True, "evidence_intervention_verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--source-manifest")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if not args.verify_only:
        if not args.source_manifest:
            parser.error("--source-manifest required before running the fixed cases")
        manifest = prepare(root, read_json(args.source_manifest))
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda c: execute_case(root, c, manifest["allowed_hosts"]), manifest["cases"]))
    proof = verify_probe(root)
    atomic_write(root/"verification.json", proof)
    print(json.dumps({"status": proof["status"], "real_completed_provider_calls": proof["real_completed_provider_calls"],
        "cases": [{"case_id":r["case_id"], "status":r["status"],
                   "decision":r["projection"]["decision"]["kind"] if r["projection"] else None} for r in proof["cases"]]}, ensure_ascii=True))


if __name__ == "__main__":
    main()
