"""Constructed worker/deadline tests; no external network requests."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from uuid import uuid4

import pytest

from football_agent.audit import atomic_write
from football_agent.collector_refresh import ExistingCollectorRefresh, PlatformRefreshAdapter
from football_agent.platform_tools import PlatformTools
from football_agent.types import ToolResult
from football_spider.storage.raw_archive import RawArchive


def setup(tmp_path):
    service = ExistingCollectorRefresh(tmp_path / "raw", tmp_path / "journal")
    reader = PlatformTools(archive_root=tmp_path / "raw", service=service)
    inputs = {"request_id": str(uuid4()), "match_id": "spftll:123", "topic": "odds",
              "prediction_cutoff": "2026-09-20T11:25:00Z", "max_requests": 2, "timeout_seconds": 3}
    return service, reader, inputs


def test_http_200_null_data_is_not_success(tmp_path):
    service, reader, inputs = setup(tmp_path)
    archive = RawArchive(tmp_path)
    archive.save("spftll", "https://example.test/GetInfo", {}, b'{"code":0,"data":null}', status=200)
    ref = str(next((tmp_path / "raw").glob("*/*/*.observation.json")))
    raw = {"status": "OK", "requests": 1, "finished_at": datetime.now(timezone.utc).isoformat(), "observation_refs": [ref]}
    result = service._convert(raw, inputs, reader, "live", ToolResult(data=[]))
    assert result.status == "NO_DATA"
    assert result.data["observation_refs"] == [ref]
    assert result.cost == 1 and result.requests == 1
    assert result.data["monetary_cost"] is None


def test_source_failure_and_request_count_are_preserved(tmp_path):
    service, reader, inputs = setup(tmp_path)
    result = service._convert({"status": "RATE_LIMITED", "requests": 2, "finished_at": "2026-09-22T08:00:00Z",
                               "observation_refs": [], "errors": [{"category": "RATE_LIMITED"}]},
                              inputs, reader, "live", ToolResult(data=[]))
    assert result.status == "RATE_LIMITED" and result.requests == 2
    assert result.error == [{"category": "RATE_LIMITED"}]


def test_domain_service_database_and_unknown_cost_are_preserved(tmp_path):
    _, reader, inputs = setup(tmp_path)
    service = PlatformRefreshAdapter(tmp_path / "raw", tmp_path / "journal")
    raw = {"status": "NEEDS_CONFIGURATION", "requests": 1, "finished_at": "2026-09-22T08:00:00Z",
           "observation_refs": [], "platform_report": {"status": "NEEDS_DATABASE", "cost": None,
               "database": {"status": "NEEDS_DATABASE"}}}
    result = service._convert(raw, inputs, reader, "live", ToolResult(data=[]))
    assert result.status == "NEEDS_CONFIGURATION" and not result.cost_known
    assert result.requests == 1 and result.requests_known
    assert result.data["platform_status"] == "NEEDS_DATABASE"
    assert result.data["monetary_cost"] is None


def test_worker_runs_once_for_same_request_id(monkeypatch, tmp_path):
    service, reader, inputs = setup(tmp_path)
    launches = []
    class Process:
        def __init__(self, args, **kwargs):
            launches.append(args)
            atomic_write(Path(args[-1]), {"status": "NO_DATA", "requests": 2,
                "finished_at": "2026-09-22T08:00:00Z", "observation_refs": []})
        def wait(self, **kwargs): return 0
    monkeypatch.setattr(subprocess, "Popen", Process)
    first = service.refresh(inputs, reader=reader, time_mode="live")
    second = service.refresh(inputs, reader=reader, time_mode="live")
    assert first == second and len(launches) == 1
    assert launches[0][-2] == "2"
    changed = service.refresh({**inputs, "topic": "fixture"}, reader=reader, time_mode="live")
    assert changed.status == "FAILED" and len(launches) == 1


def test_deadline_kills_worker_and_marks_reservation_unknown(monkeypatch, tmp_path):
    service, reader, inputs = setup(tmp_path)
    killed = []
    class Process:
        def __init__(self, *args, **kwargs): pass
        def wait(self, timeout=None):
            if timeout is not None: raise subprocess.TimeoutExpired("fixture", timeout)
        def kill(self): killed.append(True)
    monkeypatch.setattr(subprocess, "Popen", Process)
    result = service.refresh(inputs, reader=reader, time_mode="live")
    assert result.status == "TIMEOUT" and killed == [True]
    assert not result.requests_known and not result.cost_known
    assert result.requests == 2


def test_interrupted_intent_never_retries(monkeypatch, tmp_path):
    service, reader, inputs = setup(tmp_path)
    atomic_write(service.journal_root / inputs["request_id"] / "refresh.json", {"inputs": inputs, "status": "STARTED"})
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("Repeated source call"))
    result = service.refresh(inputs, reader=reader, time_mode="live")
    assert result.status == "FAILED" and result.requests == 2 and not result.requests_known


@pytest.mark.parametrize("changes,status", [({"match_id": "unconfirmed-id"}, "NEEDS_CONFIGURATION"),
    ({"topic": "injuries"}, "NO_DATA"), ({"max_requests": 0}, "RATE_LIMITED")])
def test_invalid_scope_stops_before_source(monkeypatch, tmp_path, changes, status):
    service, reader, inputs = setup(tmp_path)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("Unexpected source call"))
    assert service.refresh({**inputs, **changes}, reader=reader, time_mode="live").status == status
