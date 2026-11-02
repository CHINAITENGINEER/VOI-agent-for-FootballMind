"""Bounded adapter for the existing platform collect() entry point.

No HTTP, source parsing, retry or quota implementation lives here. A subprocess
contains the existing collector so the Agent deadline can stop further calls.
Credits are local request-accounting units; monetary cost stays unknown.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from uuid import UUID

from .audit import RunStore, atomic_write
from .types import ToolResult


class ExistingCollectorRefresh:
    worker_mode = "--worker"
    def __init__(self, archive_root="football_spider/platform_data/raw",
                 journal_root="var/parallel/agent/round2/refresh"):
        self.archive_root = Path(archive_root).resolve()
        self.journal_root = Path(journal_root).resolve()

    def configuration(self):
        return {"adapter": "existing-platform-collector-v1", "archive_root": str(self.archive_root),
                "journal_root": str(self.journal_root), "max_requests": 2,
                "cost_basis": "one_local_credit_per_request;monetary_cost_unknown"}

    def estimate_requests(self, inputs):
        return 2

    def estimate_cost(self, inputs):
        return 2.0

    def refresh(self, inputs, *, reader, time_mode):
        request_id = str(UUID(inputs["request_id"]))
        if not re.fullmatch(r"spftll:[0-9]+", inputs["match_id"]):
            return ToolResult("NEEDS_CONFIGURATION", error="Refresh requires an evidenced spftll source ID")
        if inputs.get("topic") not in {"odds", "fixture", "result"}:
            return ToolResult("NO_DATA", error="Existing spftll entry point supports fixture/result/odds only")
        maximum = min(2, inputs.get("max_requests", 0))
        if type(maximum) is not int or maximum <= 0:
            return ToolResult("RATE_LIMITED", error="Request budget exhausted")
        if reader.archive_root != self.archive_root or reader.export_path:
            return ToolResult("NEEDS_CONFIGURATION", error="Refresh reader must use the same mutable raw archive")
        before = reader._read_evidence(inputs)
        path = self.journal_root / request_id / "refresh.json"
        with RunStore(self.journal_root).lock(request_id):
            if path.exists():
                saved = json.loads(path.read_text(encoding="utf-8"))
                if saved["inputs"] != inputs:
                    return ToolResult("FAILED", error="Request ID reused with different inputs")
                if saved.get("result"):
                    return ToolResult(**saved["result"])
                return ToolResult("FAILED", error="Interrupted platform request requires reconciliation; no automatic retry",
                                  cost=float(maximum), requests=maximum,
                                  cost_known=False, requests_known=False,
                                  data={"cost_basis": "reserved_unknown_after_interruption", "monetary_cost": None})
            atomic_write(path, {"inputs": inputs, "status": "STARTED", "configuration": self.configuration()})
            output = path.with_name("platform-result.json")
            seconds = max(0.1, float(inputs.get("timeout_seconds", 10)) - 0.75)
            interpreter = sys.executable
            if self.worker_mode == "--platform-worker":
                interpreter = str(Path(__file__).resolve().parents[1] / "football_spider/.venv/Scripts/python.exe")
            args = [interpreter, "-m", "football_agent.collector_refresh", self.worker_mode,
                    str(self.archive_root), inputs["match_id"].split(":", 1)[1], str(maximum), str(output)]
            if self.worker_mode == "--platform-worker":
                args.extend([request_id, inputs["topic"], str(max(0.1, seconds - 1))])
            process = subprocess.Popen(args, cwd=Path(__file__).resolve().parents[1], stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            try:
                process.wait(timeout=seconds)
                if not output.exists():
                    raise RuntimeError("Platform worker exited without a result")
                raw = json.loads(output.read_text(encoding="utf-8"))
                result = self._convert(raw, inputs, reader, time_mode, before)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                result = ToolResult("TIMEOUT", cost=float(maximum), requests=maximum,
                    cost_known=False, requests_known=False,
                    error="Platform worker deadline; reserved request cap pending reconciliation",
                    data={"monetary_cost": None, "cost_basis": "reserved_unknown_after_deadline",
                          "platform_result_ref": str(output), "external_effects_require_reconciliation": True})
            except Exception as exc:
                result = ToolResult("FAILED", error=str(exc), cost=float(maximum), requests=maximum,
                    cost_known=False, requests_known=False,
                    data={"monetary_cost": None, "cost_basis": "reserved_unknown_worker_failure"})
            atomic_write(path, {"inputs": inputs, "status": "COMPLETED", "result": asdict(result),
                                "configuration": self.configuration()})
            return result


    def _convert(self, raw, inputs, reader, time_mode, before):
        # HTTP success is not a successful information refresh. Reuse platform
        # normalization to verify at least one observation has requested facts.
        from football_spider.storage.raw_archive import RawArchive
        from football_spider.models.contracts import Observation
        from football_spider.normalization.sources import normalize
        archive = RawArchive(self.archive_root.parent)
        usable, validation = 0, []
        for ref in raw.get("observation_refs", []):
            try:
                obs = Observation.model_validate_json(Path(ref).read_bytes())
                payload = json.loads(archive.read_body(obs))
                if isinstance(payload, dict) and payload.get("data", object()) is None:
                    validation.append({"raw_ref": ref, "status": "NO_DATA", "message": payload.get("message")})
                    continue
                batch = normalize(obs, payload)
                source_id = inputs["match_id"].split(":", 1)[1]
                count = sum(m.source_id == source_id for m in batch.matches) + sum(q.source_match_id == source_id for q in batch.quotes)
                usable += count
                validation.append({"raw_ref": ref, "status": "OK" if count else "NO_DATA", "matching_records": count})
            except Exception as exc:
                validation.append({"raw_ref": ref, "status": "FAILED", "error": str(exc)})
        requests = raw["requests"]
        data = {"refresh": raw, "observation_refs": raw.get("observation_refs", []),
                "decision_cutoff": raw["finished_at"], "monetary_cost": None,
                "cost_basis": "one_local_credit_per_actual_platform_request;monetary_cost_unknown",
                "processing_status": "RAW_ARCHIVED_ONLY_NO_DATABASE_WRITE"}
        data["observation_validation"] = validation
        if raw["status"] != "OK":
            status = raw["status"] if raw["status"] in {"TIMEOUT", "RATE_LIMITED", "NO_DATA", "NEEDS_CONFIGURATION"} else "FAILED"
            return ToolResult(status, data=data, error=raw.get("errors"), requests=requests, cost=float(requests))
        if not usable:
            return ToolResult("FAILED" if any(v["status"] == "FAILED" for v in validation) else "NO_DATA",
                              data=data, error="No valid requested facts in archived responses", requests=requests, cost=float(requests))
        new_inputs = {**inputs, "prediction_cutoff": raw["finished_at"]} if time_mode == "live" else inputs
        after = reader._read_evidence(new_inputs)
        data["evidence"] = after.data if after.status == "OK" else []
        if time_mode == "live" and not any(e.get("raw_ref") in data["observation_refs"] for e in data["evidence"]):
            return ToolResult("NO_DATA", data=data, requests=requests, cost=float(requests),
                              error="Archived facts supplied no eligible observation at the new decision cutoff")
        old_hashes = {(e["kind"], e["content_hash"]) for e in before.data or []} if before.status == "OK" else set()
        data["content_changed"] = any((e["kind"], e["content_hash"]) not in old_hashes for e in data["evidence"])
        data["visibility_note"] = "fixed_cutoff_new_arrivals_excluded" if time_mode == "archive_replay" else "new_decision_cutoff_after_archive_commit"
        return ToolResult(data=data, requests=requests, cost=float(requests))


class PlatformRefreshAdapter(ExistingCollectorRefresh):
    """Translate the READY_CODE domain RefreshService, including unknown cost."""
    worker_mode = "--platform-worker"

    def configuration(self):
        return {**super().configuration(), "adapter": "platform-refresh-service-v1",
                "service": "football_platform.api.RefreshService",
                "cost_basis": "reserved_local_credits;platform_cost_UNKNOWN"}

    def _convert(self, raw, inputs, reader, time_mode, before):
        from dataclasses import replace
        result = super()._convert(raw, inputs, reader, time_mode, before)
        return replace(result, cost_known=False, data={**(result.data or {}),
            "cost_basis": "reserved_local_credits;platform_cost_UNKNOWN",
            "platform_status": raw.get("platform_report", {}).get("status"),
            "database": raw.get("platform_report", {}).get("database")})


def worker(archive_root, match_id, maximum, output):
    from football_spider.platform_config import PlatformSettings
    from football_spider.storage.raw_archive import RawArchive
    from football_spider.collectors.sources import collect
    archive = RawArchive(Path(archive_root).parent)
    config = PlatformSettings(data_dir=archive.root, max_requests=int(maximum), sources=["spftll"])
    today = datetime.now(timezone.utc).date()
    result = collect("spftll", config, archive, start=today, end=today, match_ids=[match_id])
    raw = result.model_dump(mode="json")
    ids = set(raw["observations"])
    raw["observation_refs"] = []
    raw["raw_observations"] = []
    for path in Path(archive_root).glob("spftll/*/*.observation.json"):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata["id"] in ids:
            raw["observation_refs"].append(str(path.resolve()))
            raw["raw_observations"].append(metadata)
    raw["finished_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write(Path(output), raw)


def platform_worker(archive_root, match_id, maximum, output, request_id, topic, timeout):
    from football_platform.api import RefreshService
    from football_spider.platform_config import PlatformSettings
    config = PlatformSettings(data_dir=Path(archive_root).parent, max_requests=int(maximum))
    report = RefreshService(config).refresh("spftll:" + match_id, (topic,), request_id,
                                           max_requests=int(maximum), timeout=float(timeout))
    refs = []
    ids = {o["id"] for o in report.get("observations", [])}
    for path in Path(archive_root).glob("spftll/*/*.observation.json"):
        if json.loads(path.read_text(encoding="utf-8"))["id"] in ids:
            refs.append(str(path.resolve()))
    status = report["status"]
    if status == "NEEDS_DATABASE":
        status = "NEEDS_CONFIGURATION"
    atomic_write(Path(output), {"platform_report": report, "status": status,
        "requests": report.get("requests_this_call", report.get("requests_reserved", 0)),
        "observation_refs": refs, "finished_at": report.get("finished_at", datetime.now(timezone.utc).isoformat()),
        "errors": report.get("errors") or ({"reason": "NEEDS_DATABASE"} if status == "NEEDS_CONFIGURATION" else [])})


if __name__ == "__main__":
    if len(sys.argv) == 9 and sys.argv[1] == "--platform-worker":
        platform_worker(*sys.argv[2:])
    elif len(sys.argv) == 6 and sys.argv[1] == "--worker":
        worker(*sys.argv[2:])
    else:
        raise SystemExit("Internal platform collector worker arguments required")
