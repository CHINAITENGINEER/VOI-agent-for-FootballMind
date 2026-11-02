"""Prospective finite-menu experiment for autonomous information choice.

Candidate bodies are frozen before the Agent call and hidden from that call.
The prefetch is research setup, not an Agent acquisition. The selected body is
released as a logical Agent read; all bodies are later available only to an
isolated evaluator. This finite experiment never certifies open-world value.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from uuid import uuid4

from filelock import FileLock
from jsonschema import Draft202012Validator

from .audit import atomic_write, digest
from .inquiry_contract import ID, TEXT, obj, utc
from .inquiry_impact import decision_ceiling, terminal_state, validate_problem
from .inquiry_provider import CodexInquiryProvider
from .inquiry_tools import OpenWebTools, public_url
from .inquiry_transport import RoutedOpenWebTools
from .stopping_evaluation import SourceContentReview, review_input, validate_review
from .stopping_evaluation import REVIEW_SCHEMA


VERSION = "prospective-snapshot-choice-v1"
CHOICE_SCHEMA = obj({
    "target": obj({"hypothesis_id": ID, "question_id": ID, "statement": TEXT,
                     "hypothesis_type": {"enum": ["starting_lineup", "rotation_intent", "tactical_intent"]},
                     "subject": ID, "scope": {"type": "string", "const": "match"}, "scope_key": ID,
                     "required_evidence": {"type": "array", "minItems": 1, "maxItems": 2,
                                            "items": {"type": "string"}}}),
    "menu_review": {"type": "array", "minItems": 1, "maxItems": 6, "items": obj({
        "candidate_id": ID, "information_role": TEXT, "why_relevant": TEXT,
        "not_selected_reason": {"type": ["string", "null"]}})},
    "candidates": {"type": "array", "minItems": 0, "maxItems": 3, "items": obj({
        "candidate_id": ID, "rationale": TEXT, "expected_evidence": TEXT,
        "possible_answer_change": TEXT})},
    "decision": obj({"kind": {"enum": ["query", "stop", "defer"]},
                      "candidate_id": {"type": ["string", "null"]}, "reason": TEXT}),
})

CHOICE_PROMPT = """你是 FootballMind 的前瞻信息选择 Agent。你面对的是一个固定的来源报告问题。
候选菜单只提供来源角色、公开URL和研究人员事前登记的路径；所有候选正文都已在你调用前冻结，
但正文不会展示给你。你不能调用工具、读取文件、搜索网络或使用记忆补充来源事实。

请保持完整 target 不变。先审阅菜单中每个候选：它可能回答什么、为什么与当前目标相关，
未选路径写出当前理由。然后提出0到3条候选路线并选择一条，或者stop/defer。query只能选择菜单
里的candidate_id。你要判断“现在最值得先读取什么”，不是把所有候选都假装读取。

这是一个有限快照实验。不要编造概率、VoI、成本或网页内容；没有看到正文时只能描述可能的
信息作用。来源陈述不是世界真值，找不到目标声明也不是反向证据。用中文写理由，target原文
和candidate_id必须精确复制。
"""


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _time():
    return datetime.now(timezone.utc).isoformat()


def _candidate_action(url, allowed):
    planner = OpenWebTools(None, allowed)
    # A seed URL is enough for deterministic preparation; the experiment does
    # not let a response alter endpoint, parameters or transport policy.
    return planner.prepare({"tool": "read_url", "url": url}, {"task": {"seed_urls": [url]}, "documents": []})


def validate_task(task, allowed_hosts):
    if not isinstance(task, dict) or set(task) != {"task_id", "decision_problem", "candidates", "allowed_hosts"}:
        raise ValueError("A complete prospective task is required")
    if task["allowed_hosts"] != sorted(set(allowed_hosts)):
        raise ValueError("Task host catalog must be explicit and sorted")
    problem = validate_problem(task["decision_problem"])
    if problem["target"]["hypothesis_type"] != "starting_lineup":
        raise ValueError("This snapshot run uses a direct starting-lineup source endpoint")
    if not 2 <= len(task["candidates"]) <= 3:
        raise ValueError("A finite menu needs two or three candidates")
    ids, actions = set(), set()
    candidates = []
    for item in task["candidates"]:
        required = {"candidate_id", "url", "publisher", "information_role", "menu_note"}
        if set(item) != required or item["candidate_id"] in ids:
            raise ValueError("Candidate IDs and fields must be unique and fixed")
        url = public_url(item["url"], task["allowed_hosts"])
        action = _candidate_action(url, task["allowed_hosts"])
        action_hash = digest(action)
        if action_hash in actions:
            raise ValueError("Candidates must have distinct prepared reads")
        ids.add(item["candidate_id"]); actions.add(action_hash)
        candidates.append({**deepcopy(item), "url": url, "prepared": action})
    return {"task_id": task["task_id"], "decision_problem": problem, "candidates": candidates,
            "allowed_hosts": list(task["allowed_hosts"]), "menu_semantics": "finite_frozen_bodies_hidden_before_agent_call"}


def register(root, task):
    root = Path(root).resolve()
    if root.exists():
        raise ValueError("Use a new prospective experiment root")
    normalized = validate_task(task, task["allowed_hosts"])
    root.mkdir(parents=True)
    manifest = {"version": VERSION, "registered_at": _time(), "task": normalized,
                "prefetch": {"max_http_requests": len(normalized["candidates"]), "retries": 0,
                              "before_agent_call": True, "counted_as_agent_actions": False},
                "agent": {"max_provider_calls": 1, "max_logical_reads": 1, "selected_body_released": False},
                "evaluation": {"max_provider_calls": 1, "all_bodies_visible": True,
                                "separate_cost": True},
                "open_world_claim": False}
    atomic_write(root/"manifest.json", manifest)
    return manifest


def prefetch(root, *, tools_factory=None):
    root = Path(root).resolve(); manifest = read(root/"manifest.json")
    if manifest["version"] != VERSION or not manifest["prefetch"]["before_agent_call"]:
        raise ValueError("Invalid frozen prospective manifest")
    with FileLock(str(root/"experiment.lock")):
        ledger_path = root/"prefetch-ledger.json"
        ledger = read(ledger_path) if ledger_path.exists() else {"manifest_hash": digest(manifest), "attempts": []}
        if ledger["manifest_hash"] != digest(manifest) or ledger["attempts"]:
            raise ValueError("Prefetch can run only once")
        for item in manifest["task"]["candidates"]:
            req = {"candidate_id": item["candidate_id"], "request_id": str(uuid4()), "started_at": _time(),
                   "prepared": item["prepared"], "manifest_hash": digest(manifest)}
            atomic_write(root/item["candidate_id"]/"intent.json", req)
            ledger["attempts"].append({"candidate_id": item["candidate_id"], "state": "RESERVED_BEFORE_NETWORK",
                                        "intent_hash": digest(req)})
            atomic_write(ledger_path, ledger)
            archive_root = root/"snapshots"
            from football_spider.storage.raw_archive import RawArchive
            archive = RawArchive(archive_root)
            tool = tools_factory(archive, manifest) if tools_factory else RoutedOpenWebTools(
                archive, manifest["task"]["allowed_hosts"], route="environment")
            try:
                receipt = tool.execute(item["prepared"], run_id="prospective-"+manifest["registered_at"], request_id=req["request_id"])
                result = {"candidate_id": item["candidate_id"], "state": "ACQUIRED", "receipt": receipt,
                          "intent_hash": digest(req), "finished_at": _time()}
            except Exception as exc:
                result = {"candidate_id": item["candidate_id"], "state": "RECONCILIATION_REQUIRED", "receipt": None,
                          "error_class": type(exc).__name__, "intent_hash": digest(req), "finished_at": _time()}
            atomic_write(root/item["candidate_id"]/"result.json", result)
            ledger["attempts"][-1]["state"] = result["state"]
            atomic_write(ledger_path, ledger)
        return read(ledger_path)


def _check_prefetch(root, manifest):
    from football_spider.storage.raw_archive import RawArchive
    ledger = read(root/"prefetch-ledger.json")
    if ledger["manifest_hash"] != digest(manifest) or len(ledger["attempts"]) != len(manifest["task"]["candidates"]):
        raise ValueError("Prefetch ledger does not match the frozen menu")
    results = {}
    for item in manifest["task"]["candidates"]:
        result = read(root/item["candidate_id"]/"result.json")
        if result["intent_hash"] != next(a["intent_hash"] for a in ledger["attempts"] if a["candidate_id"] == item["candidate_id"]):
            raise ValueError("Prefetch result is not bound to its intent")
        if result["state"] == "ACQUIRED":
            doc = result["receipt"]["document"]
            if doc["status"] != "OK" or doc["kind"] != "page":
                raise ValueError("A prospective candidate must have an OK page or remain a failure")
            results[item["candidate_id"]] = result
        else:
            results[item["candidate_id"]] = result
    return results


def public_menu(root):
    root = Path(root).resolve(); manifest = read(root/"manifest.json")
    results = _check_prefetch(root, manifest)
    menu = []
    for item in manifest["task"]["candidates"]:
        result = results[item["candidate_id"]]
        menu.append({"candidate_id": item["candidate_id"], "publisher": item["publisher"],
                     "information_role": item["information_role"], "menu_note": item["menu_note"],
                     "url": item["url"], "body_hidden": True,
                     "snapshot_status": result["state"], "snapshot_text_not_exposed": True})
    problem = manifest["task"]["decision_problem"]
    return {"task": {"task_id": manifest["task"]["task_id"],
                      "objective": "选择最值得先读取的一条官方来源，以回答固定来源报告问题；也可以暂缓。",
                      "deadline": "2099-01-01T00:00:00+00:00"},
            "now": _time(), "target": deepcopy(problem["target"]),
            "decision_problem": deepcopy(problem), "claims": [],
            "menu": menu, "available_budget": {"logical_reads": 1, "model_calls_after_choice": 0},
            "menu_scope": "finite_predeclared_snapshot_menu_not_exhaustive_open_world",
            "body_visibility": "candidate_body_hidden_until_logical_selection",
            "agent_must_not_infer_body_from_snapshot_status": True}


class SnapshotChoiceProtocol:
    version, schema, prompt = VERSION+"-agent", CHOICE_SCHEMA, CHOICE_PROMPT


SNAPSHOT_REVIEW_SCHEMA = deepcopy(REVIEW_SCHEMA)
SNAPSHOT_REVIEW_SCHEMA["properties"].pop("target_hash", None)
SNAPSHOT_REVIEW_SCHEMA["required"] = [k for k in SNAPSHOT_REVIEW_SCHEMA["required"] if k != "target_hash"]


class SnapshotAnnotationProtocol:
    """The server binds the fixed target hash; the evaluator does not echo it."""
    version = VERSION+"-annotation"
    schema = SNAPSHOT_REVIEW_SCHEMA
    prompt = SourceContentReview.prompt + "\nThe host binds the fixed target from PUBLIC_STATE_JSON; return only rows, never invent a target hash.\n"


def normalize_snapshot_annotation(response, target):
    if not isinstance(response, dict):
        raise ValueError("Snapshot annotation must be an object")
    if "target_hash" in response and response["target_hash"] != digest(target):
        raise ValueError("A supplied target hash differs from the frozen host target")
    return {"target_hash": digest(target), "rows": response["rows"]}


def validate_choice(response, public):
    Draft202012Validator(CHOICE_SCHEMA).validate(response)
    if response["target"] != public["target"]:
        raise ValueError("The fixed target must be copied exactly")
    menu = {m["candidate_id"] for m in public["menu"]}
    rows = response["menu_review"]
    if {r["candidate_id"] for r in rows} != menu or len(rows) != len(menu):
        raise ValueError("Every finite menu item needs one review row")
    proposed = {c["candidate_id"] for c in response["candidates"]}
    if not proposed <= menu:
        raise ValueError("Agent proposed a candidate outside the frozen menu")
    if response["decision"]["kind"] == "query":
        if response["decision"]["candidate_id"] not in proposed:
            raise ValueError("Query must select one proposed menu candidate")
    elif response["decision"]["candidate_id"] is not None:
        raise ValueError("Terminal decision cannot name a candidate")
    if response["decision"]["kind"] == "query":
        for row in rows:
            if row["candidate_id"] == response["decision"]["candidate_id"] and row["not_selected_reason"] is not None:
                raise ValueError("Selected candidate cannot have a not-selected reason")
    return response


def run_agent(root):
    root = Path(root).resolve(); manifest = read(root/"manifest.json")
    public = public_menu(root)
    with FileLock(str(root/"experiment.lock")):
        if (root/"agent-response.json").exists():
            raise ValueError("Agent choice is one-shot")
        atomic_write(root/"agent-input.json", public)
        provider = CodexInquiryProvider(root/"agent-provider", timeout_seconds=180, protocol=SnapshotChoiceProtocol())
        payload = provider.decide(public)
        response = validate_choice(payload["response"], public)
        atomic_write(root/"agent-response.json", payload)
        atomic_write(root/"agent-choice.json", {"response": response, "receipt": payload["receipt"],
            "candidate_ids_seen": [m["candidate_id"] for m in public["menu"]],
            "body_text_exposed_to_agent": False, "prefetch_not_agent_cost": True})
        return response


def release_selected(root):
    root = Path(root).resolve(); manifest = read(root/"manifest.json"); response = read(root/"agent-choice.json")["response"]
    if response["decision"]["kind"] != "query":
        selected = None
    else:
        selected = response["decision"]["candidate_id"]
    candidate_ids = {c["candidate_id"] for c in manifest["task"]["candidates"]}
    if selected is not None and selected not in candidate_ids:
        raise ValueError("Selected candidate is not in frozen menu")
    docs = _check_prefetch(root, manifest)
    released = docs.get(selected, {}).get("receipt", {}).get("document") if selected else None
    record = {"released_candidate_id": selected, "logical_read_count": int(selected is not None),
              "body_exposed_after_choice": selected is not None, "document": released,
              "prefetch_http_not_agent_http": True}
    atomic_write(root/"selected-release.json", record)
    return record


def evaluate_all(root):
    root = Path(root).resolve(); manifest = read(root/"manifest.json")
    results = _check_prefetch(root, manifest)
    documents = [results[c["candidate_id"]]["receipt"]["document"] for c in manifest["task"]["candidates"]
                 if results[c["candidate_id"]]["state"] == "ACQUIRED"]
    if len(documents) != len(manifest["task"]["candidates"]):
        raise ValueError("This run requires a complete finite snapshot menu")
    plan = {"decision_problem": manifest["task"]["decision_problem"],
            "paths": [{"path_id": c["candidate_id"]} for c in manifest["task"]["candidates"]]}
    results_for_review = {c["candidate_id"]: {"state": "ACQUIRED", "receipt": results[c["candidate_id"]]["receipt"]}
                          for c in manifest["task"]["candidates"]}
    public = {"target": manifest["task"]["decision_problem"]["target"],
              "target_hash": digest(manifest["task"]["decision_problem"]["target"]),
              "paths": [{"path_id": c["candidate_id"], "document": results_for_review[c["candidate_id"]]["receipt"]["document"]}
                        for c in manifest["task"]["candidates"]]}
    atomic_write(root/"evaluation-input.json", public)
    provider = CodexInquiryProvider(root/"evaluation-provider", timeout_seconds=180, protocol=SnapshotAnnotationProtocol())
    payload = provider.decide(public)
    normalized = normalize_snapshot_annotation(payload["response"], public["target"])
    annotation = validate_review({"decision_problem": manifest["task"]["decision_problem"], "paths": plan["paths"]},
                                 results_for_review, normalized)
    payload["response"] = normalized
    atomic_write(root/"evaluation-response.json", payload)
    atomic_write(root/"evaluation-annotation.json", annotation)
    before = terminal_state(manifest["task"]["decision_problem"], [])
    claims = []
    for row in annotation["rows"]:
        for index, item in enumerate(row["evidence"]):
            claims.append({"claim_id": "snapshot-"+row["path_id"]+"-"+str(index), "status": "source_claim",
                "evidence": {"kind": "official_lineup", "subject": manifest["task"]["decision_problem"]["target"]["subject"],
                             "scope": "match", "scope_key": manifest["task"]["decision_problem"]["target"]["scope_key"]},
                "hypothesis_links": [{"hypothesis_id": manifest["task"]["decision_problem"]["target"]["hypothesis_id"],
                                      "relation": item["relation"]}],
                "citations": [{"document_id": results_for_review[row["path_id"]]["receipt"]["document"]["document_id"],
                                "quote": a["quote"]} for a in item["anchors"]]})
    per_candidate = {}
    for candidate in manifest["task"]["candidates"]:
        local = [c for c in claims if c["claim_id"].startswith("snapshot-"+candidate["candidate_id"]+"-")]
        per_candidate[candidate["candidate_id"]] = terminal_state(manifest["task"]["decision_problem"], local)
    selected = read(root/"selected-release.json")["released_candidate_id"]
    chosen_state = per_candidate.get(selected, before) if selected else before
    output = {"version": VERSION+"-result", "selected_candidate_id": selected,
        "menu_size": len(per_candidate), "agent_decision": read(root/"agent-choice.json")["response"]["decision"],
        "per_candidate_source_reports": per_candidate, "selected_projection": chosen_state,
        "full_menu_joint_projection": terminal_state(manifest["task"]["decision_problem"], claims),
        "ceiling_before": decision_ceiling(manifest["task"]["decision_problem"]["loss"], before["action"]),
        "before_action": before["action"], "selected_action_changed": chosen_state["action"] != before["action"],
        "unselected_action_changes": {k: v["action"] != before["action"] for k,v in per_candidate.items() if k != selected},
        "selection_regret": None, "expected_query_value": None, "optimal_menu_choice_proven": False,
        "snapshot_bodies_prefetched_before_agent": True, "agent_body_text_exposed_before_choice": False,
        "prefetch_http_attempts": len(manifest["task"]["candidates"]), "agent_logical_reads": int(selected is not None),
        "evaluation_cli_turns": 1, "research_setup_cost_separate": True,
        "finite_menu_not_open_world": True, "world_truth_independently_verified": False}
    atomic_write(root/"result.json", output)
    return output


def verify(root):
    root = Path(root).resolve(); manifest = read(root/"manifest.json")
    if (manifest["version"] != VERSION or not (root/"agent-input.json").exists()
            or not (root/"agent-choice.json").exists() or not (root/"result.json").exists()):
        raise ValueError("Incomplete prospective experiment")
    public = read(root/"agent-input.json"); choice = read(root/"agent-choice.json")["response"]
    if any("text" in item for item in public["menu"]):
        raise ValueError("Agent menu must not contain snapshot body text")
    validate_choice(choice, public)
    if read(root/"selected-release.json")["released_candidate_id"] != read(root/"result.json")["selected_candidate_id"]:
        raise ValueError("Selected release and result differ")
    result = read(root/"result.json")
    if result["prefetch_http_attempts"] != len(manifest["task"]["candidates"]):
        raise ValueError("Prefetch count differs from frozen menu")
    return result
