"""Atomic local journals with content-addressed, redacted tool references."""
from __future__ import annotations

from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import UUID

from .locks import RunBusyError, run_lock


SECRET = re.compile(r"authorization|password|secret|api[_-]?key|token|cookie", re.I)
USAGE_COUNTS = {"input_tokens", "output_tokens", "total_tokens", "prompt_tokens", "completion_tokens",
                "cached_tokens", "reasoning_tokens", "max_output_tokens"}


def public_token_measure(key, value):
    if key in USAGE_COUNTS and type(value) is int:
        return True
    if key == "tokens" and type(value) in {int, float}:
        return True
    if key == "tokens" and isinstance(value, str) and value in {"token", "tokens", "log_loss_nats_per_tokens"}:
        return True
    return (key == "tokens" and isinstance(value, dict)
            and set(value) <= {"amount", "basis", "unit"}
            and value.get("unit") in {"token", "tokens"}
            and value.get("basis") in {"known", "estimated", "simulated", "unknown"}
            and (value.get("amount") is None or type(value["amount"]) in {int, float}))


def redact(value):
    if isinstance(value, dict):
        return {str(k): "[REDACTED]" if SECRET.search(str(k)) and not public_token_measure(str(k), v) else redact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str) and value.startswith(("https://", "http://")):
        parts = urlsplit(value)
        host = parts.hostname or ""
        if ":" in host:
            host = f"[{host}]"
        if parts.port:
            host += f":{parts.port}"
        query = urlencode([(k, "[REDACTED]" if SECRET.search(k) else v) for k, v in parse_qsl(parts.query, keep_blank_values=True)])
        return urlunsplit((parts.scheme, host, parts.path, query, ""))
    return value


def json_bytes(value) -> bytes:
    return json.dumps(redact(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value) -> str:
    return sha256(json_bytes(value)).hexdigest()


def stored_digest(value) -> str:
    """Verify the actual saved bytes semantically, without redacting mutations."""
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def atomic_write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".agent-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class RunStore:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def _folder(self, run_id):
        # Only generated UUIDs can address runs; never arbitrary filesystem paths.
        if str(UUID(str(run_id))) != str(run_id):
            raise ValueError("Invalid run ID")
        return self.root / run_id

    def lock(self, run_id, timeout=0.0):
        """Guard the entire read/recover/execute/write transaction for a run.

        Nesting on the same thread is supported. Lock contention raises
        RunBusyError before reading or modifying the run. The persistent file
        is never removed; OS ownership is released when a process exits.
        """
        self._folder(run_id)  # Apply the same UUID-only addressing rule.
        return run_lock(self.root / ".locks" / f"{run_id}.lock", run_id, timeout)

    def put_blob(self, run_id, value) -> str:
        key = digest(value)
        path = self._folder(run_id) / "blobs" / f"{key}.json"
        if not path.exists():
            atomic_write(path, value)
        return f"blobs/{key}.json"

    def get_blob(self, run_id, ref):
        if not re.fullmatch(r"blobs/[a-f0-9]{64}\.json", ref):
            raise ValueError("Invalid audit reference")
        value = json.loads((self._folder(run_id) / ref).read_text(encoding="utf-8"))
        if stored_digest(value) != Path(ref).stem:
            raise ValueError("Audit blob checksum mismatch")
        return value

    def save(self, run):
        folder = self._folder(run["run_id"])
        atomic_write(folder / "run.json", run)

    def load(self, run_id):
        run = self._read(run_id)
        self._verify(run)
        return run

    def _read(self, run_id):
        run = json.loads((self._folder(run_id) / "run.json").read_text(encoding="utf-8"))
        if run.get("run_id") != run_id:
            raise ValueError("Stored run ID differs from requested run")
        return run

    def verify(self, run_id):
        return self._verify(self._read(run_id))

    def _verify(self, run):
        run_id = run["run_id"]
        if not run.get("events"):
            raise ValueError("Missing audit events")
        previous = None
        for sequence, event in enumerate(run["events"], 1):
            body = {k: v for k, v in event.items() if k != "event_hash"}
            if event["seq"] != sequence or body.get("previous_hash") != previous or stored_digest(body) != event["event_hash"]:
                raise ValueError("Audit event chain mismatch")
            previous = event["event_hash"]
            for key in ("input_ref", "output_ref"):
                if event["data"].get(key):
                    self.get_blob(run_id, event["data"][key])
        expected = run["events"][-1]["snapshot"]
        if any(run.get(k) != v for k, v in expected.items()):
            raise ValueError("Run summary differs from final audit snapshot")
        return {"run_id": run_id, "status": "OK", "events_verified": len(run["events"]), "head_hash": previous}
