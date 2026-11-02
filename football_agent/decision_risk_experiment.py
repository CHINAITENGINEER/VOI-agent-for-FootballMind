"""Constructed observation environment for terminal-risk inquiry experiments.

No football data, learned likelihoods or external HTTP are supplied here. Real
CLI proposals, when requested, remain separate from the finite risk selector.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from itertools import product
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import requests

from football_spider.storage.raw_archive import RawArchive
from .audit import atomic_write, digest
from .decision_loss import FiniteDecisionLossModel
from .decision_value import ObservationTool
from .inquiry import InquiryWorkflow
from .inquiry_evidence_state import EvidenceStateProtocol
from .inquiry_provider import CodexInquiryProvider
from .inquiry_returns import OUTCOMES
from .inquiry_tools import OpenWebTools
from .inquiry_value import DecisionRiskSelector
from .inquiry_verify import verify


H1 = {"hypothesis_id": "h1", "question_id": "q1", "statement": "Player K starts in fixture M.",
      "hypothesis_type": "starting_lineup", "subject": "Team-A", "scope": "match", "scope_key": "fixture-M",
      "required_evidence": ["official_lineup"]}
H2 = {**H1, "hypothesis_id": "h2", "question_id": "q2", "statement": "Player J is available for fixture M.",
      "hypothesis_type": "availability", "required_evidence": ["injury_update"]}
TEXT_K = {"yes": "For fixture M, Team A reports that Player K will start.",
          "no": "For fixture M, Team A reports that Player K will not start."}
TEXT_J = {"yes": "For fixture M, Team A reports that Player J is available.",
          "no": "For fixture M, Team A reports that Player J is not available."}
TASK_ID = "constructed-decision-risk-inquiry"


def build_selector(k_weight=3.0):
    """A fully declared eight-state law; source availability is shared, not IID."""
    labels = tuple(f"K{k}J{j}" for k, j in product((0, 1), repeat=2))
    world = {f"K{k}J{j}S{s}": (k, j, s) for k, j, s in product((0, 1), repeat=3)}
    hypotheses = {name: [float(y == f"K{k}J{j}") for y in labels] for name, (k, j, s) in world.items()}
    prior = {name: .25 * (.8 if s else .2) for name, (k, j, s) in world.items()}
    tools = {}
    for action, cost in (("report_a", 0), ("report_b", 0), ("fill_j", .02), ("verify_k", .03), ("refresh_k", .04)):
        law = {}
        for name, (k, j, s) in world.items():
            if action.startswith("report_"):
                law[name] = (.9, .1, 0) if k else (.1, .9, 0)
            elif action == "fill_j":
                law[name] = (.9, 0, .1) if j else (0, .9, .1)
            else:
                law[name] = (float(k), float(not k), 0) if s else (0, 0, 1)
        tools[action] = ObservationTool(("yes", "no", "failure"), law, cost,
            "Constructed response law; verify_k and refresh_k share source availability S.", action)
    losses = {}
    for dk, dj in product(("yes", "no", "defer"), repeat=2):
        def single(decision, truth):
            return .2 if decision == "defer" else float((decision == "yes") != bool(truth))
        losses[f"K_{dk}__J_{dj}"] = [k_weight*single(dk, k)+single(dj, j) for k, j in product((0, 1), repeat=2)]
    model = FiniteDecisionLossModel(hypotheses, prior, tools, outcome_labels=labels,
        decision_losses=losses, cost_unit="expected_personnel_report_loss",
        assumptions=("All priors, return laws, penalties and costs are hand-declared synthetic assumptions.",
            "K and J are independent fair binary variables before observations; report errors are conditionally independent.",
            "S is a shared stationary availability variable for verify_k and refresh_k, initially P(S=1)=0.8.",
            "fill_j succeeds with probability 0.9 independently of S; successful verification is exact in this constructed environment.",
            f"Incorrect K report costs {k_weight}; incorrect J report costs 1; deferring a report costs 0.2 times its weight.",
            "No elapsed-time model, calibrated football inference or monetary conversion is asserted."))
    bindings = {a: {"url": "https://example.com/"+a, "hypothesis": deepcopy(H2 if a == "fill_j" else H1),
        "evidence_type": "injury_update" if a == "fill_j" else "official_lineup",
        "markers": deepcopy(TEXT_J if a == "fill_j" else TEXT_K), "failure_outcome": "failure"} for a in tools}
    return DecisionRiskSelector(model, bindings, task_id=TASK_ID, loss_budget=.2)


def proposal(actions, *, chosen=None, kind="query"):
    """Scripted diagnostic proposal; not counted as a real LLM decision."""
    candidates = []
    for action in actions:
        h = H2 if action == "fill_j" else H1
        candidates.append({"candidate_id": action, "question_id": h["question_id"], "tool": "read_url",
            "url": "https://example.com/"+action, "query": None, "rationale": "Compare task-relevant available sources.",
            "answer_could_change": "A personnel assertion or deferral and the following query.",
            "inquiry_intent": {"hypothesis_ids": [h["hypothesis_id"]], "purpose": "resolve",
                "expected_evidence": list(h["required_evidence"]), "subject": h["subject"], "scope": h["scope"],
                "scope_key": h["scope_key"], "supports_change": "Use the positive return.",
                "challenges_change": "Use the negative return.", "no_answer_change": "Keep uncertainty."},
            "return_model": {"branches": [{"outcome": outcome,
                "missing_after": [{"hypothesis_id": h["hypothesis_id"], "types": [] if outcome in OUTCOMES[:3] else h["required_evidence"]}],
                "next_action": {"kind": "reassess", "target_hypothesis_ids": [], "evidence_types": [],
                                "reason": "Reassess the remaining task after the actual observation."}} for outcome in OUTCOMES]}})
    return {"questions": [{"question_id": h["question_id"], "question": h["statement"],
        "decision_relevance": "Constructed personnel report decision.", "answer_could_change": "Report yes/no/defer.",
        "based_on_documents": []} for h in (H1, H2)], "hypotheses": [deepcopy(H1), deepcopy(H2)],
        "claim_updates": [], "context_bindings": [], "candidates": candidates,
        "decision": {"kind": kind, "candidate_id": (chosen or actions[0]) if kind == "query" else None,
                     "reason": "Scripted proposal for a mechanism check, not a learned policy.", "revisit_at": None}}


class ConstructedTools(OpenWebTools):
    def __init__(self, root, returns, *, clock=None):
        self.calls = []
        owner = self
        class Session:
            def get(self, url, **kwargs):
                owner.calls.append(url)
                action = url.rsplit("/", 1)[-1]
                token = returns[action]
                if token == "failure":
                    raise requests.ConnectionError("Constructed unavailable source; no network request sent")
                marker = (TEXT_J if action == "fill_j" else TEXT_K)[token]
                return SimpleNamespace(content=("<main>"+marker+"</main>").encode(), status_code=200,
                    headers={"content-type": "text/html"}, close=lambda: None)
            def close(self): pass
        super().__init__(RawArchive(root/"evidence"), ["example.com"], clock=clock, session_factory=Session)

    def configuration(self):
        return {**super().configuration(), "execution_mode": "constructed_transport_no_network"}


def task(now, selector):
    return {"task_id": TASK_ID, "objective":
        "构造任务：为 fixture M 的两个人员问题决定报告 yes/no/defer。所有网页返回均为构造资料。"
        "scoped_decision_value 提供声明假设下的模型和注册查询；引用其值时保留假设范围，不自行编造概率。"
        "h1/h2 使用 registered_queries 中的完整定义。根据已有正文和状态自主提出至多两个下一步候选，"
        "保留支持、反向或冲突来源的引文。可以查询、反思、等待、暂缓或停止。程序另行比较可建模候选的决策风险。",
        "deadline": (now+timedelta(hours=2)).isoformat(),
        "initial_facts": [{"status": "constructed_environment", "statement": "No real football observation or external HTTP is involved."}],
        "seed_urls": [b["url"] for b in selector.bindings.values()]}


def make_parent(root, *, clock=None):
    class ParentProvider:
        def configuration(self): return {"provider": "constructed_import_preparation", "configured": True}
        def decide(self, public):
            index = len(public["action_history"])
            answer = proposal([("report_a", "report_b")[index]]) if index < 2 else proposal([], kind="defer")
            return {"response": answer, "receipt": {"real_provider_call": False}}
    clock = clock or (lambda: datetime.now(timezone.utc))
    tools = ConstructedTools(root, {"report_a": "yes", "report_b": "no"}, clock=clock)
    workflow = InquiryWorkflow(root/"runs", ParentProvider(), tools, clock=clock, protocol=EvidenceStateProtocol())
    run = workflow.run(workflow.start(task(clock(), build_selector()), max_requests=2, max_llm_calls=3)["run_id"])
    if len(run["documents"]) != 2 or any(d["status"] != "OK" for d in run["documents"]):
        raise ValueError("Constructed parent preparation failed")
    verify(root, run["run_id"])
    return run


def run_experiment(root):
    if root.exists():
        raise ValueError("Use a new root to preserve every previous attempt")
    root.mkdir(parents=True)
    sources = sorted(set(Path("football_agent").glob("inquiry*.py")) | {
        Path(__file__), Path("football_agent/decision_loss.py"), Path("football_agent/decision_value.py"), Path("football_agent/audit.py")})
    frozen = []
    for source in sources:
        destination = root/"execution-source"/source.name
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, destination)
        frozen.append({"path": str(source.resolve()), "snapshot": str(destination.resolve()),
                       "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    # Same conflicting acquired reports and same potential returns in both cases.
    # Only the K loss weight changes. The private return tape is never a policy input.
    cases = [("case-01", .5), ("case-02", 3.0)]
    tape = {"verify_k": "failure", "refresh_k": "failure", "fill_j": "yes"}
    atomic_write(root/"manifest.json", {"created_at": datetime.now(timezone.utc).isoformat(), "sources": frozen,
        "cases": [{"case_id": name, "k_weight": weight, "selector": build_selector(weight).configuration()} for name, weight in cases],
        "environment_only_return_tape": tape, "max_llm_calls_per_case": 5, "max_requests_per_case": 3,
        "retries": 0, "external_http_requests": 0, "new_football_observations": 0,
        "scope": "two_constructed_mechanism_traces_not_policy_effectiveness_estimation"})
    parent = make_parent(root/"parent")
    results = []
    for name, weight in cases:
        folder = root/name
        protocol = EvidenceStateProtocol()
        selector = build_selector(weight)
        tools = ConstructedTools(folder, tape)
        provider = CodexInquiryProvider(folder/"provider", protocol=protocol, timeout_seconds=180)
        workflow = InquiryWorkflow(folder/"runs", provider, tools, protocol=protocol, selector=selector)
        run = workflow.start(task(datetime.now(timezone.utc), selector), max_requests=3, max_llm_calls=5)
        run = workflow.inherit_evidence(run["run_id"], parent_root=root/"parent", parent_run_id=parent["run_id"],
            parent_head_hash=parent["events"][-1]["event_hash"], document_ids=[d["document_id"] for d in parent["documents"]])
        print(json.dumps({"case": name, "run_id": run["run_id"], "state": "STARTED"}), flush=True)
        run = workflow.run(run["run_id"])
        workflow.export(run["run_id"], folder/"report.md")
        proof = verify(folder, run["run_id"])
        atomic_write(folder/"verification.json", proof)
        result = {"case_id": name, "k_weight": weight, "run_id": run["run_id"], "state": run["state"], "reason": run["reason"],
            "head_hash": run["events"][-1]["event_hash"], "real_completed_llm_turns": proof["real_completed_llm_turns"],
            "constructed_tool_calls": len(tools.calls), "requested_urls": tools.calls,
            "final_value_state": selector.public_summary(run), "external_http_requests": 0, "new_football_observations": 0}
        results.append(result)
        atomic_write(root/"results.json", results)
        print(json.dumps({k: v for k, v in result.items() if k != "final_value_state"}), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    run_experiment(Path(args.root).resolve())


if __name__ == "__main__":
    main()
