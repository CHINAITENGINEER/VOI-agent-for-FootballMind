"""Batch orchestration tests for task-conditioned finite source choice."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from football_agent.audit import atomic_write
from football_agent.multi_task_snapshot import VERSION, prepare_batch, verify_batch
from football_agent.prospective_snapshot import (
    prefetch, public_menu, read, register,
)


def task(task_id="source-task", statement="Official source states Player X started for Team A in match M."):
    target = {"hypothesis_id": f"h-{task_id}", "question_id": f"q-{task_id}",
        "statement": statement, "hypothesis_type": "starting_lineup", "subject": "team-a",
        "scope": "match", "scope_key": f"match-{task_id}", "required_evidence": ["official_lineup"]}
    problem = {"task_id": task_id, "decision_id": f"decision-{task_id}", "target": target,
        "semantics": "source_position_not_world_truth",
        "state_definitions": {"positive": "Source states the player started.",
                               "negative": "Source states the player did not start."},
        "loss": {"report_supported": {"positive": 0, "negative": 1},
                 "report_challenged": {"positive": 1, "negative": 0},
                 "withhold": {"positive": .2, "negative": .2}},
        "loss_basis": "Declared source-reporting preference."}
    return {"task_id": task_id, "decision_problem": problem, "allowed_hosts": ["example.com"],
        "candidates": [
            {"candidate_id": "direct", "url": "https://example.com/direct", "publisher": "Official club",
             "information_role": "match report", "menu_note": "A first-party match-specific report."},
            {"candidate_id": "background", "url": "https://example.com/background", "publisher": "Official club",
             "information_role": "team background", "menu_note": "A first-party background page."},
            {"candidate_id": "rules", "url": "https://example.com/rules", "publisher": "Competition body",
             "information_role": "competition context", "menu_note": "A rules and competition context page."},
        ]}


def source_factory(calls):
    bodies = {"/direct": "Official source states Player X started for Team A in match M.",
              "/background": "Team A has a long history.",
              "/rules": "The competition has group-stage rules."}
    def make(archive, manifest):
        class Session:
            def get(self, url, **kwargs):
                assert kwargs["allow_redirects"] is False
                calls.append(url)
                body = ("<html><article>" + bodies[url.split("example.com", 1)[1]] + "</article></html>").encode()
                return SimpleNamespace(status_code=200, content=body, headers={}, close=lambda: None)
            def close(self):
                pass
        from football_agent.inquiry_tools import OpenWebTools
        return OpenWebTools(archive, manifest["task"]["allowed_hosts"], session_factory=Session)
    return make


def source_root(tmp_path):
    root = tmp_path / "source"
    register(root, task())
    calls = []
    prefetch(root, tools_factory=source_factory(calls))
    assert len(calls) == 3
    return root


def test_prepare_batch_reuses_frozen_menu_without_network(tmp_path):
    source = source_root(tmp_path)
    batch = tmp_path / "batch"
    tasks = [task("task-a"), task("task-b", "Official source states Player X did not start for Team A in match M.")]
    manifest = prepare_batch(batch, tasks, source)
    assert manifest["version"] == VERSION
    assert manifest["task_count"] == 2
    assert manifest["new_research_http_requests"] == 0
    for item in tasks:
        menu = public_menu(batch / item["task_id"])
        assert len(menu["menu"]) == 3
        assert all(m["body_hidden"] for m in menu["menu"])
        ledger = read(batch / item["task_id"] / "prefetch-ledger.json")
        assert all(a["reused"] and a["state"] == "ACQUIRED" for a in ledger["attempts"])


def test_prepare_batch_freezes_task_identity_and_rejects_duplicates(tmp_path):
    source = source_root(tmp_path)
    with pytest.raises(ValueError, match="unique"):
        prepare_batch(tmp_path / "duplicate", [task("same"), task("same")], source)
    (tmp_path / "existing").mkdir()
    with pytest.raises(ValueError, match="new batch root"):
        prepare_batch(tmp_path / "existing", [task("one")], source)


def test_verify_batch_requires_one_result_per_task(tmp_path):
    source = source_root(tmp_path)
    batch = tmp_path / "batch"
    prepare_batch(batch, [task("task-a")], source)
    atomic_write(batch / "batch-result.json", {
        "version": VERSION + "-result", "task_count": 1, "rows": [],
        "new_research_http_requests": 0, "finite_menu_not_open_world": True,
        "optimal_policy_proven": False,
    })
    with pytest.raises(ValueError, match="Every task"):
        verify_batch(batch)


def test_verify_batch_rejects_claimed_network_or_optimality(tmp_path):
    source = source_root(tmp_path)
    batch = tmp_path / "batch"
    prepare_batch(batch, [task("task-a")], source)
    atomic_write(batch / "batch-result.json", {
        "version": VERSION + "-result", "task_count": 1,
        "rows": [{"task_id": "task-a", "selected_candidate_id": "direct"}],
        "new_research_http_requests": 1, "finite_menu_not_open_world": True,
        "optimal_policy_proven": False,
    })
    with pytest.raises(ValueError, match="research HTTP"):
        verify_batch(batch)
