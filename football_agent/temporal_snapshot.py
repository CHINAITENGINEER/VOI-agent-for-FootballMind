"""Temporal eligibility gate for autonomous source selection.

The same source can be highly relevant in hindsight and still be unavailable at
the decision cutoff.  This module makes that distinction part of the Agent
contract: bodies are frozen for a finite experiment, while only candidates that
were operationally available before the cutoff and have an eligible document
scope may be proposed or selected.  Bodies remain hidden until a valid choice.

This is an experiment harness, not a historical truth reconstruction service.
It records the observed and operational times, keeps failures and quarantined
documents visible as metadata, and never turns a retrospective page into a
pre-match fact.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from filelock import FileLock
from jsonschema import Draft202012Validator

from .audit import atomic_write, digest
from .inquiry_contract import ID, TEXT, obj
from .inquiry_provider import CodexInquiryProvider


VERSION = "temporal-source-choice-v2"
ELIGIBLE = "ELIGIBLE"
EXCLUDED_AFTER_CUTOFF = "EXCLUDED_AFTER_CUTOFF"
QUARANTINED_SCOPE = "QUARANTINED_SCOPE"
FAILED = "FAILED"
STATUSES = {ELIGIBLE, EXCLUDED_AFTER_CUTOFF, QUARANTINED_SCOPE, FAILED}
ELIGIBLE_SCOPES = {"pre_match_preview", "pre_match_structured"}


def _time(value: str | datetime) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Temporal timestamps require a timezone")
    return parsed.astimezone(timezone.utc)


def _id(value: str) -> bool:
    return isinstance(value, str) and bool(__import__("re").fullmatch(r"[A-Za-z0-9_.:-]{1,120}", value))


def _status(candidate: dict[str, Any], cutoff: datetime) -> tuple[str, str]:
    """Return status and reason without reading candidate body text."""
    observed = _time(candidate["observed_at"])
    available = _time(candidate["available_at"])
    if available < observed:
        raise ValueError("Operational availability precedes observed acquisition")
    if candidate["error_class"] or candidate["http_status"] is None or not 200 <= candidate["http_status"] < 300:
        return FAILED, "transport_or_http_failure"
    if candidate["document_scope"] not in ELIGIBLE_SCOPES:
        return QUARANTINED_SCOPE, "document_scope_not_pre_match"
    if available > cutoff or observed > cutoff:
        return EXCLUDED_AFTER_CUTOFF, "not_available_at_prediction_cutoff"
    return ELIGIBLE, "available_before_prediction_cutoff"


CHOICE_SCHEMA = obj({
    "target": obj({"question_id": ID, "statement": TEXT, "scope": {"type": "string", "const": "match"},
                     "scope_key": ID}),
    "menu_review": {"type": "array", "minItems": 1, "maxItems": 8, "items": obj({
        "candidate_id": ID, "information_role": TEXT, "availability_reason": TEXT,
        "why_relevant": TEXT, "not_selected_reason": {"type": ["string", "null"]}})},
    "candidates": {"type": "array", "maxItems": 3, "items": obj({
        "candidate_id": ID, "rationale": TEXT, "expected_evidence": TEXT,
        "possible_answer_change": TEXT})},
    "decision": obj({"kind": {"enum": ["query", "stop", "defer"]},
                      "candidate_id": {"type": ["string", "null"]}, "reason": TEXT}),
})


CHOICE_PROMPT = """你是 FootballMind 的时间约束信息选择 Agent。
你面对一个固定的赛前问题和有限候选菜单。候选正文在你的调用前已经冻结，但正文不会展示。
你只能根据菜单角色、时间字段、请求结果和文档范围决定下一步；不得凭记忆补充页面内容。

时间语义非常严格：只有 status=ELIGIBLE 的候选可提出、可选择。它表示请求成功、文档范围为
pre_match_preview 或 pre_match_structured，并且 observed_at 与 available_at 都不晚于
prediction_cutoff。EXCLUDED_AFTER_CUTOFF 即使正文看起来最贴近问题，也属于事后材料；
QUARANTINED_SCOPE 是赛后或范围混合页面；FAILED 是请求失败。它们可以在菜单审阅中说明
为什么相关或为什么被排除，但不得被当作可读取证据，也不得据此推断正文结论。

先为每个菜单项写审阅行，准确复制candidate_id、角色和可用性理由。然后最多提出三条路线，
只能包含ELIGIBLE项；若可用来源不足以直接回答目标，选择defer或stop并说明缺口。defer时，
仍在待执行计划中的候选可以保留空的未选理由；没有进入计划的候选要说明原因。query必须
选择提出的ELIGIBLE candidate_id。不要编造概率、VoI、成本、引文、来源事实或历史发布时间。
这是有限菜单实验，不代表开放世界全局最优。用中文写理由，输出一个符合Schema的JSON对象。
"""


class TemporalChoiceProtocol:
    version = VERSION + "-agent"
    schema = CHOICE_SCHEMA
    prompt = CHOICE_PROMPT


def _validate_candidate(candidate: dict[str, Any], allowed_hosts: list[str]) -> dict[str, Any]:
    required = {"candidate_id", "url", "publisher", "information_role", "menu_note", "document_scope",
                "observed_at", "available_at", "http_status", "error_class", "snapshot_path"}
    if set(candidate) != required or not _id(candidate["candidate_id"]):
        raise ValueError("Temporal candidate fields must be complete and inert")
    from urllib.parse import urlsplit
    parsed = urlsplit(candidate["url"])
    if parsed.scheme != "https" or parsed.hostname not in allowed_hosts:
        raise ValueError("Candidate URL is outside the explicit host catalog")
    _time(candidate["observed_at"]); _time(candidate["available_at"])
    if candidate["document_scope"] not in {"pre_match_preview", "pre_match_structured", "post_match_report", "unknown"}:
        raise ValueError("Unsupported document scope")
    snapshot = Path(candidate["snapshot_path"]).resolve()
    if not snapshot.is_file():
        raise ValueError("Frozen snapshot body is missing")
    return {**deepcopy(candidate), "url": candidate["url"], "snapshot_path": str(snapshot)}


def register(root: str | Path, task: dict[str, Any]) -> dict[str, Any]:
    root = Path(root).resolve()
    if root.exists():
        raise ValueError("Use a new temporal experiment root")
    required = {"task_id", "prediction_cutoff", "kickoff", "target", "allowed_hosts", "candidates"}
    if set(task) != required or not _id(task["task_id"]):
        raise ValueError("Temporal task fields are incomplete")
    cutoff, kickoff = _time(task["prediction_cutoff"]), _time(task["kickoff"])
    if cutoff >= kickoff:
        raise ValueError("Prediction cutoff must precede kickoff")
    hosts = sorted(set(task["allowed_hosts"]))
    if not hosts or any(not isinstance(host, str) or not host for host in hosts):
        raise ValueError("A nonempty host catalog is required")
    if not 2 <= len(task["candidates"]) <= 8:
        raise ValueError("Temporal menu needs two to eight candidates")
    root.mkdir(parents=True)
    candidates, ids = [], set()
    for raw in task["candidates"]:
        candidate = _validate_candidate(raw, hosts)
        if candidate["candidate_id"] in ids:
            raise ValueError("Temporal candidate IDs must be unique")
        ids.add(candidate["candidate_id"])
        status, reason = _status(candidate, cutoff)
        src = Path(candidate["snapshot_path"])
        destination = root / "snapshots" / (candidate["candidate_id"] + ".body")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, destination)
        candidate.pop("snapshot_path")
        candidate.update({"snapshot_relpath": str(destination.relative_to(root)),
                          "snapshot_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                          "availability_status": status, "availability_reason": reason,
                          "body_hidden": True})
        candidates.append(candidate)
    manifest = {"version": VERSION, "task": {"task_id": task["task_id"],
                "prediction_cutoff": cutoff.isoformat(), "kickoff": kickoff.isoformat(),
                "target": deepcopy(task["target"]), "allowed_hosts": hosts, "candidates": candidates},
                "agent": {"max_provider_calls": 1, "max_logical_reads": 1,
                          "body_text_exposed_before_choice": False},
                "research_setup": {"frozen_before_agent": True, "new_http_requests": 0,
                                   "setup_bodies_are_not_agent_actions": True},
                "open_world_claim": False}
    atomic_write(root / "manifest.json", manifest)
    return manifest


def read(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def public_menu(root: str | Path) -> dict[str, Any]:
    manifest = read(Path(root) / "manifest.json")
    task = manifest["task"]
    menu = []
    for candidate in task["candidates"]:
        menu.append({k: candidate[k] for k in ("candidate_id", "publisher", "information_role", "menu_note",
                                                "url", "availability_status", "availability_reason")}
                    | {"body_hidden": True, "snapshot_text_not_exposed": True,
                       "selectable": candidate["availability_status"] == ELIGIBLE})
    return {"task": {"task_id": task["task_id"], "prediction_cutoff": task["prediction_cutoff"],
                      "kickoff": task["kickoff"]}, "now": datetime.now(timezone.utc).isoformat(),
            "target": deepcopy(task["target"]), "menu": menu,
            "available_budget": {"logical_reads": 1, "model_calls_after_choice": 0},
            "temporal_semantics": "only_ELIGIBLE_candidates_are_selectable",
            "body_visibility": "all_candidate_bodies_hidden_until_valid_selection"}


def validate_choice(response: dict[str, Any], public: dict[str, Any]) -> dict[str, Any]:
    Draft202012Validator(CHOICE_SCHEMA).validate(response)
    if response["target"] != public["target"]:
        raise ValueError("Temporal target was changed")
    menu = {row["candidate_id"]: row for row in public["menu"]}
    rows = response["menu_review"]
    if {row["candidate_id"] for row in rows} != set(menu) or len(rows) != len(menu):
        raise ValueError("Every temporal menu item needs one review row")
    proposed = {item["candidate_id"] for item in response["candidates"]}
    if not proposed <= set(menu):
        raise ValueError("Proposed candidate is outside the frozen menu")
    if any(not menu[c]["selectable"] for c in proposed):
        raise ValueError("Only temporally eligible candidates may be proposed")
    decision = response["decision"]
    if decision["kind"] == "query":
        if decision["candidate_id"] not in proposed or not menu[decision["candidate_id"]]["selectable"]:
            raise ValueError("Query must select a proposed eligible candidate")
    elif decision["candidate_id"] is not None:
        raise ValueError("Terminal temporal decision cannot name a candidate")
    for row in rows:
        pending = row["candidate_id"] in proposed and decision["kind"] == "defer"
        expected_null = row["candidate_id"] == decision["candidate_id"] if decision["kind"] == "query" else pending
        if expected_null and row["not_selected_reason"] is not None:
            raise ValueError("Selected candidate needs a null not-selected reason")
        if not expected_null and row["not_selected_reason"] is None:
            raise ValueError("Every unselected or unavailable candidate needs a reason")
    return response


def run_agent(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    manifest = read(root / "manifest.json")
    public = public_menu(root)
    with FileLock(str(root / "experiment.lock")):
        if (root / "agent-response.json").exists():
            raise ValueError("Temporal Agent choice is one-shot")
        atomic_write(root / "agent-input.json", public)
        provider = CodexInquiryProvider(root / "agent-provider", timeout_seconds=180,
                                       protocol=TemporalChoiceProtocol())
        payload = provider.decide(public)
        response = validate_choice(payload["response"], public)
        atomic_write(root / "agent-response.json", payload)
        atomic_write(root / "agent-choice.json", {"response": response, "receipt": payload["receipt"],
                    "body_text_exposed_to_agent": False, "candidate_ids_seen": [m["candidate_id"] for m in public["menu"]],
                    "temporally_ineligible_candidates_selectable": False})
        return response


def release_selected(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    manifest = read(root / "manifest.json")
    response = read(root / "agent-choice.json")["response"]
    selected = response["decision"]["candidate_id"] if response["decision"]["kind"] == "query" else None
    candidates = {c["candidate_id"]: c for c in manifest["task"]["candidates"]}
    if selected is not None and (selected not in candidates or candidates[selected]["availability_status"] != ELIGIBLE):
        raise ValueError("Only an eligible candidate can be released")
    document = None
    if selected is not None:
        path = root / candidates[selected]["snapshot_relpath"]
        document = {"candidate_id": selected, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "bytes": path.stat().st_size}
    release = {"released_candidate_id": selected, "logical_read_count": int(selected is not None),
               "document": document, "body_exposed_after_choice": selected is not None,
               "temporally_ineligible_body_released": False, "new_research_http_requests": 0}
    atomic_write(root / "selected-release.json", release)
    return release


def verify(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    manifest, public = read(root / "manifest.json"), read(root / "agent-input.json")
    choice = read(root / "agent-choice.json")["response"]
    if manifest["version"] != VERSION:
        raise ValueError("Temporal manifest version mismatch")
    if any("text" in row or "body" in row for row in public["menu"]):
        raise ValueError("Temporal Agent menu leaked a body")
    validate_choice(choice, public)
    release = read(root / "selected-release.json")
    if release["released_candidate_id"] != (choice["decision"]["candidate_id"] if choice["decision"]["kind"] == "query" else None):
        raise ValueError("Temporal release differs from Agent choice")
    for candidate in manifest["task"]["candidates"]:
        path = root / candidate["snapshot_relpath"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != candidate["snapshot_sha256"]:
            raise ValueError("Frozen temporal snapshot checksum mismatch")
    return {"status": "OK", "version": VERSION, "menu_size": len(public["menu"]),
            "eligible_candidates": [m["candidate_id"] for m in public["menu"] if m["selectable"]],
            "selected_candidate_id": release["released_candidate_id"],
            "new_research_http_requests": 0, "body_text_exposed_before_choice": False,
            "finite_menu_not_open_world": True}
