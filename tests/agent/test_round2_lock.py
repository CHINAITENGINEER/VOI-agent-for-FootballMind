"""Constructed process-contention checks; no network or real probabilities."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.request import urlopen
from uuid import uuid4

import pytest

from football_agent.audit import RunBusyError, RunStore
from football_agent.engine import Workflow
from football_agent.testing import SyntheticTools, TEST_CUTOFF, TEST_MATCH_ID


WORKSPACE = Path(__file__).resolve().parents[2]


def _wait_file(path, process, timeout=10):
    deadline = time.monotonic() + timeout
    while not path.exists():
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            pytest.fail(f"Child exited {process.returncode} before ready: {stdout} {stderr}")
        if time.monotonic() >= deadline:
            process.kill()
            stdout, stderr = process.communicate()
            pytest.fail(f"Child did not signal readiness: {stdout} {stderr}")
        time.sleep(0.01)


def test_lock_reentry_crosses_store_instances_without_unlocking_outer_scope(tmp_path):
    run_id = str(uuid4())
    first = RunStore(tmp_path)
    second = RunStore(tmp_path / "nested" / "..")
    with first.lock(run_id):
        with second.lock(run_id):
            with first.lock(run_id, timeout=0.1):
                pass
        with ThreadPoolExecutor(max_workers=1) as pool:
            def contend():
                with second.lock(run_id):
                    pytest.fail("Inner exit released the outer lock")
            with pytest.raises(RunBusyError) as exc:
                pool.submit(contend).result(timeout=3)
            assert exc.value.run_id == run_id
            assert exc.value.code == "RUN_BUSY"
    with second.lock(run_id):
        pass
    assert (tmp_path / ".locks" / f"{run_id}.lock").exists()


def test_distinct_runs_do_not_block_each_other_across_threads(tmp_path):
    store = RunStore(tmp_path)
    run_id, other_id = str(uuid4()), str(uuid4())
    with store.lock(run_id), ThreadPoolExecutor(max_workers=1) as pool:
        def independent():
            with RunStore(tmp_path).lock(other_id):
                return "acquired"
        assert pool.submit(independent).result(timeout=3) == "acquired"


@pytest.mark.parametrize("timeout", [-1, float("inf"), float("nan"), True, "1"])
def test_invalid_lock_timeout_does_not_create_a_lock(tmp_path, timeout):
    with pytest.raises(ValueError):
        with RunStore(tmp_path).lock(str(uuid4()), timeout=timeout):
            pass
    assert not (tmp_path / ".locks").exists()


def test_lock_keeps_uuid_path_validation(tmp_path):
    with pytest.raises(ValueError):
        with RunStore(tmp_path).lock("../other-run"):
            pass
    assert not (tmp_path / ".locks").exists()


def test_cross_process_positive_timeout_and_crash_release(tmp_path):
    run_id = str(uuid4())
    ready, release = tmp_path / "ready", tmp_path / "release"
    script = """
import os, sys, time
from pathlib import Path
from football_agent.audit import RunStore
root, run_id = Path(sys.argv[1]), sys.argv[2]
with RunStore(root).lock(run_id):
    (root / 'ready').write_text('ready')
    while not (root / 'release').exists():
        time.sleep(0.01)
    os._exit(71)
"""
    process = subprocess.Popen([sys.executable, "-c", script, str(tmp_path), run_id],
                               cwd=WORKSPACE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        _wait_file(ready, process)
        started = time.monotonic()
        with pytest.raises(RunBusyError):
            with RunStore(tmp_path).lock(run_id, timeout=0.08):
                pass
        assert time.monotonic() - started >= 0.06
        release.write_text("exit without finally")
        process.communicate(timeout=10)
        assert process.returncode == 71
        with RunStore(tmp_path).lock(run_id, timeout=0.2):
            pass
        assert (tmp_path / ".locks" / f"{run_id}.lock").exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)


WORKER = """
import json, os, sys, time
from dataclasses import replace
from pathlib import Path
from football_agent.engine import Workflow
from football_agent.testing import SyntheticTools
root, run_id, finish = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
class CountedTools(SyntheticTools):
    def estimate_cost(self, tool, inputs):
        return 0.1
    def execute(self, tool, inputs):
        with (root / 'external-calls.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'tool': tool, 'pid': os.getpid()}) + '\\n')
            stream.flush()
            os.fsync(stream.fileno())
        (root / 'tool-running').write_text('ready')
        while not (root / 'release-tool').exists():
            time.sleep(0.01)
        if finish == 'crash':
            os._exit(72)
        return replace(super().execute(tool, inputs), cost=0.1)
runner = Workflow(CountedTools(), store_dir=root)
print(json.dumps(runner.step(run_id)['state']), flush=True)
"""


@pytest.mark.parametrize("finish", ["return", "crash"])
def test_cli_http_and_other_workflows_do_not_duplicate_active_tool_or_recover_it(tmp_path, finish):
    runner = Workflow(SyntheticTools(), store_dir=tmp_path)
    run_id = runner.start(TEST_MATCH_ID, TEST_CUTOFF)["run_id"]
    process = subprocess.Popen([sys.executable, "-c", WORKER, str(tmp_path), run_id, finish],
                               cwd=WORKSPACE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    server = None
    server_thread = None
    try:
        _wait_file(tmp_path / "tool-running", process)
        original = RunStore(tmp_path).load(run_id)
        assert original["pending_call"]["tool"] == "query_matches"
        original_events = len(original["events"])
        for method in (runner.get_run, runner.step, runner.run_to_completion):
            with pytest.raises(RunBusyError):
                method(run_id)
        other = Workflow(SyntheticTools(), store_dir=tmp_path)
        with pytest.raises(RunBusyError):
            other.get_run(run_id)

        # Invoke the real CLI restore/step path in an independent process.
        cli = subprocess.run([sys.executable, "-m", "football_agent", "--store-dir", str(tmp_path),
                              "resume", run_id, "--action", "next"],
                             cwd=WORKSPACE, capture_output=True, text=True, timeout=10)
        assert cli.returncode == 2, cli.stdout + cli.stderr
        assert "RUN_BUSY" in cli.stderr + cli.stdout

        # HTTP restore/read also takes the run lock before deciding a pending
        # call was interrupted. The request must fail promptly as busy.
        from demo.server import DemoApplication, create_server
        server = create_server("127.0.0.1", 0, DemoApplication(tmp_path))
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        with pytest.raises(HTTPError) as exc:
            urlopen(f"http://127.0.0.1:{server.server_port}/api/runs/{run_id}", timeout=5)
        assert exc.value.code == 409
        assert "RUN_BUSY" in exc.value.read().decode("utf-8")

        active = RunStore(tmp_path).load(run_id)
        assert len(active["events"]) == original_events
        assert active["pending_call"] == original["pending_call"]
        assert active["budget"]["calls"] == 1
        assert active["budget"]["spent_cost"] == 0
        (tmp_path / "release-tool").write_text("finish")
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == (72 if finish == "crash" else 0), stdout + stderr

        recovered = runner.get_run(run_id)
        assert recovered["budget"]["calls"] == 1
        assert recovered["budget"]["spent_cost"] == pytest.approx(0.1)
        assert recovered["pending_call"] is None
        if finish == "crash":
            assert recovered["state"] == "DEFERRED"
            assert recovered["budget"]["estimated_cost_charges"] == 1
            assert len(recovered["unreconciled_calls"]) == 1
            assert runner.run_to_completion(run_id)["budget"]["spent_cost"] == pytest.approx(0.1)
        else:
            assert recovered["state"] == "MATCH_READY"
            assert recovered["budget"]["estimated_cost_charges"] == 0
            assert recovered["unreconciled_calls"] == []
        calls = [json.loads(line) for line in (tmp_path / "external-calls.jsonl").read_text().splitlines()]
        assert [item["tool"] for item in calls] == ["query_matches"]
        started = [event for event in recovered["events"] if event["type"] == "tool_started"]
        returned = [event for event in recovered["events"] if event["type"] == "tool_returned"]
        assert len(started) == 1
        assert len(returned) == (0 if finish == "crash" else 1)
        assert RunStore(tmp_path).verify(run_id)["status"] == "OK"
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if server_thread is not None:
            server_thread.join(timeout=5)
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
