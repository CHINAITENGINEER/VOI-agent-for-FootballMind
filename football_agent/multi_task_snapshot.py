"""Batch runner for finite prospective source-choice tasks.

The batch reuses an already frozen snapshot menu and creates independent task
manifests, Agent choices, releases and terminal projections. Reusing snapshots
keeps the experiment focused on task-conditioned information selection rather
than mixing a new network sample into the policy comparison.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import shutil
from pathlib import Path

from .audit import atomic_write, digest
from .prospective_snapshot import VERSION as SNAPSHOT_VERSION
from .prospective_snapshot import evaluate_all, read, register, release_selected, run_agent


VERSION = "multi-task-snapshot-choice-v1"


def _time():
    return datetime.now(timezone.utc).isoformat()


def _seed_task(root: Path, task: dict, source_root: Path):
    """Register a new task and attach a copy of an existing frozen menu.

    The source bodies are copied from a completed snapshot experiment. No
    transport is invoked and the new ledger records zero research HTTP calls.
    """
    task_root = root / task["task_id"]
    register(task_root, task)
    manifest = read(task_root / "manifest.json")
    manifest["prefetch"].update({
        "max_http_requests": 0,
        "reused_frozen_snapshot": True,
        "source_experiment": str(source_root.resolve()),
    })
    manifest["research_setup"] = {
        "kind": "reused_frozen_public_snapshot",
        "source_experiment": str(source_root.resolve()),
        "source_manifest_hash": digest(read(source_root / "manifest.json")),
        "new_http_requests": 0,
        "counted_as_agent_actions": False,
    }
    atomic_write(task_root / "manifest.json", manifest)
    source_manifest = read(source_root / "manifest.json")
    source_candidates = {c["candidate_id"]: c for c in source_manifest["task"]["candidates"]}
    candidates = {c["candidate_id"]: c for c in manifest["task"]["candidates"]}
    if set(source_candidates) != set(candidates):
        raise ValueError("A reused snapshot menu must contain the same candidate IDs")
    source_snapshots = source_root / "snapshots"
    if source_snapshots.is_dir():
        shutil.copytree(source_snapshots, task_root / "snapshots")
    ledger = {"manifest_hash": digest(manifest), "attempts": [],
              "reused_source_manifest_hash": digest(source_manifest)}
    for candidate_id in candidates:
        candidate = candidates[candidate_id]
        source_result = read(source_root / candidate_id / "result.json")
        if source_result["state"] != "ACQUIRED":
            raise ValueError("Every reused candidate must have an acquired source snapshot")
        intent = {
            "candidate_id": candidate_id,
            "request_id": f"reuse-{task['task_id']}-{candidate_id}",
            "started_at": _time(),
            "prepared": candidate["prepared"],
            "manifest_hash": digest(manifest),
            "reused_source_receipt": source_result["receipt"].get("document", {}).get("document_id"),
        }
        result = {
            "candidate_id": candidate_id,
            "state": "ACQUIRED",
            "receipt": deepcopy(source_result["receipt"]),
            "intent_hash": digest(intent),
            "finished_at": _time(),
            "research_setup_reused": True,
        }
        atomic_write(task_root / candidate_id / "intent.json", intent)
        atomic_write(task_root / candidate_id / "result.json", result)
        ledger["attempts"].append({"candidate_id": candidate_id, "state": "ACQUIRED",
                                   "intent_hash": digest(intent), "reused": True})
    atomic_write(task_root / "prefetch-ledger.json", ledger)
    return task_root


def prepare_batch(root, tasks, source_root):
    """Create a new batch whose task bodies come from one frozen source run."""
    root = Path(root).resolve()
    source_root = Path(source_root).resolve()
    if root.exists():
        raise ValueError("Use a new batch root; existing artifacts are never overwritten")
    if read(source_root / "manifest.json")["version"] != SNAPSHOT_VERSION:
        raise ValueError("The source experiment is not a prospective snapshot run")
    if not tasks or len(tasks) > 6:
        raise ValueError("A batch needs one to six fixed tasks")
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise ValueError("Batch task IDs must be unique")
    root.mkdir(parents=True)
    task_roots = []
    for task in tasks:
        task_roots.append(_seed_task(root, deepcopy(task), source_root))
    manifest = {
        "version": VERSION,
        "created_at": _time(),
        "source_experiment": str(source_root),
        "source_manifest_hash": digest(read(source_root / "manifest.json")),
        "task_ids": [task["task_id"] for task in tasks],
        "task_count": len(tasks),
        "shared_frozen_menu": True,
        "new_research_http_requests": 0,
        "agent_choices_are_independent_per_task": True,
        "evaluation_is_independent_per_task": True,
        "open_world_claim": False,
    }
    atomic_write(root / "batch-manifest.json", manifest)
    return manifest


def run_batch(root):
    """Run exactly one Agent choice and one independent evaluation per task."""
    root = Path(root).resolve()
    batch = read(root / "batch-manifest.json")
    rows = []
    for task_id in batch["task_ids"]:
        task_root = root / task_id
        if (task_root / "result.json").exists():
            raise ValueError("This task already has a result; batch execution is one-shot")
        choice = run_agent(task_root)
        release = release_selected(task_root)
        result = evaluate_all(task_root)
        rows.append({
            "task_id": task_id,
            "selected_candidate_id": result["selected_candidate_id"],
            "decision_kind": choice["decision"]["kind"],
            "logical_reads": release["logical_read_count"],
            "selected_action_changed": result["selected_action_changed"],
            "selected_action_before": result["before_action"],
            "selected_action_after": result["selected_projection"]["action"],
            "unselected_action_changes": result["unselected_action_changes"],
            "expected_query_value": result["expected_query_value"],
            "selection_regret": result["selection_regret"],
            "optimal_menu_choice_proven": result["optimal_menu_choice_proven"],
            "result_path": str((task_root / "result.json").relative_to(root)),
        })
    summary = {
        "version": VERSION + "-result",
        "batch_id": root.name,
        "task_count": len(rows),
        "rows": rows,
        "new_research_http_requests": 0,
        "agent_choice_cli_turns": len(rows),
        "evaluation_cli_turns": len(rows),
        "logical_reads": sum(row["logical_reads"] for row in rows),
        "finite_menu_not_open_world": True,
        "source_position_not_world_truth": True,
        "selection_regret_measured": False,
        "expected_query_value_measured": False,
        "optimal_policy_proven": False,
    }
    atomic_write(root / "batch-result.json", summary)
    return summary


def verify_batch(root):
    root = Path(root).resolve()
    batch = read(root / "batch-manifest.json")
    result = read(root / "batch-result.json")
    if batch["version"] != VERSION or result["task_count"] != batch["task_count"]:
        raise ValueError("Batch manifest and result do not match")
    if batch["new_research_http_requests"] != 0 or result["new_research_http_requests"] != 0:
        raise ValueError("The batch must not add research HTTP")
    if not result["finite_menu_not_open_world"] or result["optimal_policy_proven"]:
        raise ValueError("Batch claims exceed finite-menu evidence")
    if {row["task_id"] for row in result["rows"]} != set(batch["task_ids"]):
        raise ValueError("Every task needs one result row")
    for row in result["rows"]:
        task_root = root / row["task_id"]
        task_result = read(task_root / "result.json")
        if task_result["selected_candidate_id"] != row["selected_candidate_id"]:
            raise ValueError("Batch row and task result differ")
        if task_result["prefetch_http_attempts"] != 3:
            raise ValueError("Each task must retain the three frozen candidate snapshots")
        if task_result["research_setup_cost_separate"] is not True:
            raise ValueError("Task research setup semantics changed")
    return result
