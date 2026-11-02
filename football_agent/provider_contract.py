"""Provider transport-contract fingerprints and replay checks.

This module proves that a real model turn saw the declared public state,
protocol and capability policy, and that its persisted response matches the
streamed response. It deliberately does not claim semantic truth or calibrated
value of information.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from .audit import stored_digest

VERSION = "provider-contract-v1"


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def protocol_digest(protocol) -> str | None:
    if protocol is None:
        return None
    return stored_digest({
        "version": getattr(protocol, "version", None),
        "schema": getattr(protocol, "schema", None),
        "prompt": getattr(protocol, "prompt", None),
    })


def build_receipt(*, provider_version, protocol, public, prompt, schema_bytes,
                  stream_bytes, output_bytes, stdout_bytes, events, command,
                  disabled_features, enabled_features, cleanup_after_completed,
                  returncode, elapsed_seconds):
    """Build a stable, redacted transport receipt for one real provider turn."""
    item_types = [e.get("item", {}).get("type") for e in events
                  if isinstance(e.get("item"), dict) and e.get("item", {}).get("type")]
    event_types = [e.get("type") for e in events if isinstance(e.get("type"), str)]
    receipt = {
        "contract_version": VERSION,
        "contract_status": "transport_validated",
        "provider_version": provider_version,
        "protocol_version": getattr(protocol, "version", None),
        "protocol_digest": protocol_digest(protocol),
        "schema_sha256": _sha_bytes(schema_bytes),
        "public_input_digest": stored_digest(public),
        "prompt_sha256": _sha_bytes(prompt.encode("utf-8")),
        "response_sha256": _sha_bytes(stream_bytes),
        "stream_response_sha256": _sha_bytes(stream_bytes),
        "output_file_present": output_bytes is not None,
        "output_file_sha256": _sha_bytes(output_bytes) if output_bytes is not None else None,
        "stdout_sha256": _sha_bytes(stdout_bytes),
        "command_sha256": _sha_bytes("\0".join(str(x) for x in command).encode("utf-8")),
        "event_summary": {
            "event_types": event_types,
            "item_types": item_types,
            "turn_started_count": event_types.count("turn.started"),
            "turn_completed_count": event_types.count("turn.completed"),
            "turn_failed_count": event_types.count("turn.failed"),
        },
        "capability_policy": {
            "disabled_features": sorted(disabled_features),
            "enabled_features": sorted(enabled_features),
            "tool_events": [],
        },
        "cleanup_after_completed": bool(cleanup_after_completed),
        "returncode": returncode,
        "elapsed_seconds": elapsed_seconds,
        "semantic_validation": "delegated_to_experience_service",
        "world_truth_verified": False,
        "numeric_voi_calibrated": False,
    }
    receipt["contract_digest"] = stored_digest(receipt)
    return receipt


def verify_receipt(receipt, *, public=None, protocol=None):
    """Verify receipt self-integrity and optional input/protocol bindings."""
    if not isinstance(receipt, dict) or receipt.get("contract_version") != VERSION:
        raise ValueError("PROVIDER_CONTRACT_VERSION")
    observed = dict(receipt)
    recorded = observed.pop("contract_digest", None)
    if not recorded or stored_digest(observed) != recorded:
        raise ValueError("PROVIDER_CONTRACT_DIGEST_MISMATCH")
    if public is not None and receipt.get("public_input_digest") != stored_digest(public):
        raise ValueError("PROVIDER_PUBLIC_INPUT_DIGEST_MISMATCH")
    if protocol is not None:
        if receipt.get("protocol_version") != getattr(protocol, "version", None):
            raise ValueError("PROVIDER_PROTOCOL_VERSION_MISMATCH")
        if receipt.get("protocol_digest") != protocol_digest(protocol):
            raise ValueError("PROVIDER_PROTOCOL_DIGEST_MISMATCH")
    policy = receipt.get("capability_policy") or {}
    if policy.get("tool_events") != []:
        raise ValueError("PROVIDER_TOOL_EVENTS_PRESENT")
    if receipt.get("contract_status") != "transport_validated":
        raise ValueError("PROVIDER_CONTRACT_STATUS")
    return {
        "status": "verified",
        "contract_version": VERSION,
        "contract_digest": recorded,
        "semantic_validation": receipt.get("semantic_validation"),
        "world_truth_verified": receipt.get("world_truth_verified") is True,
        "numeric_voi_calibrated": receipt.get("numeric_voi_calibrated") is True,
    }
