"""Frozen constructed evidence, real CLI decisions, no HTTP execution.

Five predeclared cases diagnose use of an evidence-state protocol. They are
not a benchmark, causal efficacy estimate, or operational match inquiries.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil

from football_spider.storage.raw_archive import RawArchive
from .audit import atomic_write, digest
from .inquiry_hypotheses import get_protocol, project
from .inquiry_provider import CodexInquiryProvider
from .inquiry_returns import actual_continuation
from .inquiry_tools import OpenWebTools


PLAN = [("case-01", "evidence-state-inquiry-v1", "support"),
        ("case-02", "return-forecast-inquiry-v1", "challenge"),
        ("case-03", "evidence-state-inquiry-v1", "challenge"),
        ("case-04", "evidence-state-inquiry-v1", "mixed"),
        ("case-05", "evidence-state-inquiry-v1", "context")]
H1 = {"hypothesis_id": "h1", "question_id": "q1", "statement": "Player K starts in fixture M.",
      "hypothesis_type": "starting_lineup", "subject": "Team-A", "scope": "match", "scope_key": "fixture-M",
      "required_evidence": ["official_lineup"]}
H2 = {**H1, "hypothesis_id": "h2", "question_id": "q2", "statement": "Player J is available for fixture M.",
      "hypothesis_type": "availability", "required_evidence": ["injury_update"]}
STATEMENTS = {"supports": "For fixture M, Team A's official lineup statement says Player K will start.",
              "challenges": "For fixture M, Team A's official lineup statement says Player K will not start.",
              "context": "Player K was selected for Team A's training squad for the campaign."}


def make_public(protocol, scenario, now, tools):
    relations = {"support": ["supports"], "challenge": ["challenges"],
                 "mixed": ["supports", "challenges"], "context": ["context"]}[scenario]
    documents, claims = [], []
    for index, relation in enumerate(relations, 1):
        identity = f"doc-{index}"
        documents.append({"document_id": identity, "url": f"https://example.com/acquired-{index}", "kind": "page", "status": "OK",
            "http_status": None, "text": STATEMENTS[relation], "links": [], "observed_at": now.isoformat(),
            "available_at": now.isoformat(), "provenance": "constructed_fixture_no_network"})
        claims.append({"claim_id": f"claim-{index}", "question_id": "q1", "text": STATEMENTS[relation], "status": "source_claim",
            "citations": [{"document_id": identity, "quote": STATEMENTS[relation]}],
            "evidence": {"kind": "roster" if relation == "context" else "official_lineup", "subject": "Team-A",
                         "scope": "campaign" if relation == "context" else "match",
                         "scope_key": "campaign-one" if relation == "context" else "fixture-M"},
            "hypothesis_links": [{"hypothesis_id": "h1", "relation": relation}]})
    questions = [{"question_id": h["question_id"], "question": h["statement"],
                  "decision_relevance": "Two equally relevant personnel questions for a constructed decision.",
                  "answer_could_change": "Which personnel input to acquire or verify next.", "based_on_documents": []} for h in (H1, H2)]
    task = {"task_id": "constructed-personnel-inquiry", "objective":
        "为构造比赛 fixture M 整理两个同等相关的人员问题：Player K 是否首发、Player J 是否可出场。根据已有资料提出下一步行动及理由，可继续查缺口、核对、刷新、暂缓或停止。所有资料均为构造输入，不使用记忆中的比赛事实。本诊断只记录提议，不执行外部工具。",
        "deadline": (now+timedelta(hours=2)).isoformat(), "initial_facts": [{"status": "constructed_fixture",
            "statement": "Documents and initial extracted claims are controlled inputs, not real football observations."}],
        "seed_urls": ["https://example.com/lineup", "https://example.com/injury", "https://example.com/lineup-confirmation"]}
    public = {"task": task, "now": now.isoformat(), "questions": questions, "claims": claims, "documents": documents,
        "action_history": [], "last_decision": None, "budget": {"remaining_requests": 2, "remaining_llm_calls": 3},
        "tools": {**tools.configuration(), "execution_mode": "proposal_only_constructed_documents_no_http"},
        "value_model_status": "UNSUPPORTED_OPEN_TEXT", "prediction_status": "OPEN_TEXT_UPDATE_NOT_CONFIGURED",
        "hypotheses": [deepcopy(H1), deepcopy(H2)], "context_bindings": [], "query_reviews": [], "return_reviews": [], "inherited_evidence": []}
    seed = {"questions": questions, "hypotheses": [H1, H2], "claim_updates": claims, "context_bindings": [], "candidates": [],
            "decision": {"kind": "defer", "candidate_id": None, "reason": "Validate constructed initial extraction.", "revisit_at": None}}
    protocol.validate(seed, {**public, "questions": [], "claims": [], "hypotheses": []})
    public["hypothesis_states"] = project(public["hypotheses"], claims)
    public["requirements"] = protocol.project_requirements(public["hypotheses"], claims, [])
    public.update(protocol.public(public))
    return public


def prepare(root):
    if root.exists():
        raise ValueError("Use a new root; existing probe evidence must be preserved")
    root.mkdir(parents=True)
    now = datetime.now(timezone.utc)
    tools = OpenWebTools(RawArchive(root/"unused-archive"), ["example.com"])
    cases = []
    for identity, version, scenario in PLAN:
        public = make_public(get_protocol(version), scenario, now, tools)
        atomic_write(root/identity/"input.json", public)
        cases.append({"case_id": identity, "protocol": version, "scenario": scenario, "input_hash": digest(public)})
    files = sorted(Path("football_agent").glob("inquiry*.py")) + [Path(__file__), Path("football_agent/audit.py")]
    sources = []
    for file in files:
        destination = root/"execution-source"/file.name
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(file, destination)
        sources.append({"path": str(file.resolve()), "snapshot": str(destination.resolve()),
                        "sha256": hashlib.sha256(file.read_bytes()).hexdigest()})
    manifest = {"created_at": now.isoformat(), "cases": cases, "sources": sources, "planned_real_model_calls": len(cases),
                "max_concurrency": 2, "retries": 0, "http_execution": False, "constructed_initial_extraction": True,
                "interpretation": "paired_negative_case_changes_protocol_prompt_and_projection_not_field_only_causal_effect"}
    atomic_write(root/"manifest.json", manifest)
    return manifest


def execute_case(root, case):
    folder = root/case["case_id"]
    public = json.loads((folder/"input.json").read_text(encoding="utf-8"))
    assert digest(public) == case["input_hash"]
    protocol = get_protocol(case["protocol"])
    provider = CodexInquiryProvider(folder/"provider", timeout_seconds=180, protocol=protocol)
    tools = OpenWebTools(RawArchive(root/"unused-archive"), ["example.com"])
    result = {"case_id": case["case_id"], "started_at": datetime.now(timezone.utc).isoformat(), "http_executed": 0}
    try:
        reply = provider.decide(public)
        atomic_write(folder/"reply.json", reply)
        protocol.validate(reply["response"], public)
        prepared = {c["candidate_id"]: tools.prepare(c, public) for c in reply["response"]["candidates"]}
        selected = reply["response"]["decision"]["candidate_id"]
        result.update(status="VALIDATED_PROPOSAL_NOT_EXECUTION", proposed_continuation=actual_continuation(reply["response"]),
                      proposed_tool=prepared.get(selected), reason=reply["response"]["decision"]["reason"])
    except Exception as exc:
        result.update(status="FAILED_PROBE_NO_RETRY", error_type=type(exc).__name__, error=str(exc)[:600])
    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write(folder/"result.json", result)
    print(json.dumps(result, ensure_ascii=True), flush=True)
    return result


def verify_probe(root):
    manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    for source in manifest["sources"]:
        assert hashlib.sha256(Path(source["snapshot"]).read_bytes()).hexdigest() == source["sha256"]
        assert hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest() == source["sha256"]
    rows, calls = [], 0
    tools = OpenWebTools(RawArchive(root/"unused-archive"), ["example.com"])
    for case in manifest["cases"]:
        folder = root/case["case_id"]
        public = json.loads((folder/"input.json").read_text(encoding="utf-8"))
        result = json.loads((folder/"result.json").read_text(encoding="utf-8"))
        assert digest(public) == case["input_hash"]
        protocol = get_protocol(case["protocol"])
        assert public["requirements"] == protocol.project_requirements(public["hypotheses"], public["claims"], public["context_bindings"])
        assert public["hypothesis_states"] == project(public["hypotheses"], public["claims"])
        if (folder/"reply.json").exists():
            reply = json.loads((folder/"reply.json").read_text(encoding="utf-8"))
            calls += int(reply["receipt"].get("real_provider_call", False))
            if result["status"] == "VALIDATED_PROPOSAL_NOT_EXECUTION":
                protocol.validate(reply["response"], public)
                prepared = {c["candidate_id"]: tools.prepare(c, public) for c in reply["response"]["candidates"]}
                assert result["proposed_tool"] == prepared.get(reply["response"]["decision"]["candidate_id"])
                assert result["proposed_continuation"] == actual_continuation(reply["response"])
                assert result["reason"] == reply["response"]["decision"]["reason"]
        rows.append({**case, "initial_requirements": public["requirements"], **result})
    proof = {"status": "VERIFIED_FROZEN_PROPOSALS_NOT_POLICY_EFFECTIVENESS", "cases": rows,
             "real_completed_llm_turns": calls, "http_executed": 0, "new_network_requests_during_verification": 0,
             "new_football_observations": 0, "source_semantics_independently_verified": False,
             "no_counterfactual_utility_estimated": True, "all_planned_cases_retained": True}
    atomic_write(root/"verification.json", proof)
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if not args.verify_only:
        manifest = prepare(root)
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda case: execute_case(root, case), manifest["cases"]))
    proof = verify_probe(root)
    print(json.dumps({"status": proof["status"], "real_completed_llm_turns": proof["real_completed_llm_turns"],
                      "cases": len(proof["cases"]), "http_executed": 0}), flush=True)


if __name__ == "__main__":
    main()
