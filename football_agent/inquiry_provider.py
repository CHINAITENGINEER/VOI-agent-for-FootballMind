"""Explicit local Codex CLI provider; no credential-file access or model substitution."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from uuid import uuid4

from .audit import atomic_write
from .inquiry_contract import PROMPT, RESPONSE_SCHEMA
from .inquiry_provider_diagnostics import record_process_diagnostic


def provider_schema(value):
    """Provider format subset; all constraints still apply in local validation."""
    if isinstance(value, dict):
        return {k: provider_schema(v) for k, v in value.items() if k != "uniqueItems"}
    if isinstance(value, list):
        return [provider_schema(v) for v in value]
    return value


class CodexInquiryProvider:
    version = "codex-inquiry-json-v2"

    def __init__(self, workspace, *, executable=None, timeout_seconds=120, runner=None, protocol=None, model=None):
        self.workspace = Path(workspace).resolve()
        self.executable = executable or shutil.which("codex")
        self.timeout_seconds = timeout_seconds
        self._runner = runner or subprocess.run
        self.protocol = protocol
        if model is not None and (not isinstance(model, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._:/-]{0,119}", model)):
            raise ValueError("Explicit model must be a bounded model identifier")
        self.model = model
        if not 1 <= timeout_seconds <= 180:
            raise ValueError("Provider deadline must be 1..180 seconds")

    def configuration(self):
        config = {"provider": "codex-cli", "version": self.version, "configured": bool(self.executable),
                "authentication": "existing_cli_login_not_inspected", "model": "cli_default_no_override",
                "ephemeral": True, "sandbox": "read-only", "timeout_seconds": self.timeout_seconds,
                'code_mode_host': 'enabled_required_runtime',
                "auth_validated_by_actual_call": False, "cost_status": "unknown"}
        if self.protocol:
            config["inquiry_protocol"] = self.protocol.version
        if self.model is not None:
            config["model"] = self.model
            config["model_requested"] = self.model
            config["response_model"] = None
            config["model_identity_status"] = "requested_name_not_verified_weights"
        return config

    def decide(self, public):
        if not self.executable:
            raise RuntimeError("CODEX_CLI_NOT_CONFIGURED")
        folder = self.workspace / str(uuid4())
        folder.mkdir(parents=True)
        schema, output = folder / "response.schema.json", folder / "response.json"
        atomic_write(schema, provider_schema(self.protocol.schema if self.protocol else RESPONSE_SCHEMA))
        # An isolated working directory contains no research labels or archives.
        # The CLI reuses its own login; this adapter never opens auth.json.
        command = [self.executable, "exec", "--ephemeral", "--ignore-user-config",
                   "--skip-git-repo-check", "--sandbox", "read-only", "--json",
                   "--color", "never", "--cd", str(folder), "--output-schema", str(schema),
                   "--output-last-message", str(output), "-c", 'web_search="disabled"',
                   "-c", "project_doc_max_bytes=0", "-c", 'approval_policy="never"']
        for feature in ("shell_tool", "apps", "multi_agent", "browser_use", "computer_use",
                        "image_generation", "plugins", "hooks", "goals",
                        "workspace_dependencies", "skill_search"):
            command.extend(["--disable", feature])
        # Current CLI may require its execution host even for a text-only turn.
        # This does not enable shell, browser, plugins, or source acquisition.
        command.extend(['--enable', 'code_mode_host'])
        if self.model is not None:
            command.extend(["--model", self.model])
        command.append("-")
        prompt = (self.protocol.prompt if self.protocol else PROMPT) + "\nPUBLIC_STATE_JSON:\n" + json.dumps(public, ensure_ascii=False, allow_nan=False)
        atomic_write(folder / "input.json", public)
        started = time.monotonic()
        try:
            process = self._runner(command, input=prompt, capture_output=True, text=True, encoding="utf-8",
                                   timeout=self.timeout_seconds,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        except (subprocess.TimeoutExpired, OSError) as exc:
            try:
                record_process_diagnostic(folder, stdout=getattr(exc, 'stdout', None),
                    stderr=getattr(exc, 'stderr', None),
                    category='timeout' if isinstance(exc, subprocess.TimeoutExpired) else 'runner_os_error',
                    elapsed_seconds=time.monotonic() - started, timeout_seconds=self.timeout_seconds,
                    exception_type=type(exc).__name__)
            except Exception as diagnostic_error:
                exc.add_note('Diagnostic persistence failed: ' + type(diagnostic_error).__name__)
            raise
        record_process_diagnostic(folder, stdout=process.stdout, stderr=process.stderr,
            category='process_returned', elapsed_seconds=time.monotonic() - started,
            timeout_seconds=self.timeout_seconds, returncode=process.returncode)
        events = []
        for line in process.stdout.splitlines():
            try:
                event = json.loads(line)
                if not isinstance(event, dict) or not isinstance(event.get('type'), str):
                    raise ValueError('Invalid event object')
                if event.get('item') is not None and not isinstance(event['item'], dict):
                    raise ValueError('Invalid item object')
                events.append(event)
            except (ValueError, TypeError):
                atomic_write(folder / "failure.json", {"returncode": process.returncode,
                             "category": "non_json_event",
                             'line_sha256': hashlib.sha256(line.encode()).hexdigest(),
                             'stderr_sha256': hashlib.sha256(process.stderr.encode()).hexdigest(),
                             'diagnostic_ref': 'diagnostic.json'})
                raise RuntimeError("PROVIDER_NON_JSON_EVENT") from None
        # Reject responses accompanied by any direct evidence access or execution.
        # Internal reasoning messages are not exported into the project trace.
        unexpected = [e.get("item", {}).get("type") for e in events if e.get("item", {}).get("type")
                      not in {None, "agent_message", "reasoning", "error"}]
        atomic_write(folder / "process.json", {"returncode": process.returncode,
                     "event_types": [e.get("type") for e in events],
                     "item_types": [e.get("item", {}).get("type") for e in events if e.get("item")],
                     "unexpected_item_types": unexpected,
                     "errors": [{'type': e['type']} for e in events if e.get("type") in {"error", "turn.failed"}]})
        if unexpected:
            raise RuntimeError("PROVIDER_USED_UNDECLARED_TOOL")
        if process.returncode or not output.exists() or not any(e.get("type") == "turn.completed" for e in events):
            atomic_write(folder / "failure.json", {"returncode": process.returncode,
                         "stderr_sha256": hashlib.sha256(process.stderr.encode()).hexdigest(),
                         "event_types": [e.get("type") for e in events]})
            raise RuntimeError("PROVIDER_TURN_FAILED")
        response = json.loads(output.read_text(encoding="utf-8"))
        usage = [e.get("usage") for e in events if e.get("type") == "turn.completed"]
        receipt = {"provider": "codex-cli", "version": self.version, "real_provider_call": True,
                   "requests": 1, "requests_unit": "CLI_turn_provider_internal_requests_unknown",
                   "usage": usage, "latency_seconds": time.monotonic() - started,
                   "model_requested": self.model or "cli_default_no_override", "response_model": None,
                   "response_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                   "stdout_sha256": hashlib.sha256(process.stdout.encode()).hexdigest(),
                   "returncode": process.returncode, "tool_events": [],
                   "actual_money": None, "money_status": "not_reported"}
        atomic_write(folder / "receipt.json", receipt)
        return {"response": response, "receipt": receipt}
