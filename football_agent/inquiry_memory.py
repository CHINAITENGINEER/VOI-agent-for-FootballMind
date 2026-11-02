"""Reuse verified prior observations without a fresh source request or backdating."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from .audit import RunStore
from .inquiry_contract import utc
from .inquiry_tools import public_url


def load_inherited(root, run_id, expected_head, document_ids, *, now, deadline, allowed_hosts, ancestors=()):
    from .inquiry_verify import verify

    root = Path(root).resolve()
    identity = (str(root), run_id)
    if identity in ancestors or len(ancestors) >= 24:
        raise ValueError("Cyclic or excessive inquiry inheritance")
    if not document_ids or len(document_ids) > 6 or len(document_ids) != len(set(document_ids)):
        raise ValueError("Select one to six distinct prior documents")
    proof = verify(root, run_id, _ancestors=ancestors)
    if proof["audit"]["head_hash"] != expected_head or proof["unresolved_call"]:
        raise ValueError("Prior inquiry differs from pinned completed evidence")
    parent = RunStore(root / "runs").load(run_id)
    if parent["state"] == "READY" or parent["events"][-1]["event_hash"] != expected_head:
        raise ValueError("Prior inquiry must be frozen before reuse")
    if utc(parent["events"][-1]["at"]) > utc(now):
        raise ValueError("Pinned parent snapshot was unavailable at import time")
    by_id = {d["document_id"]: d for d in parent["documents"]}
    documents = []
    for identity in document_ids:
        doc = by_id.get(identity)
        if not doc or doc["status"] != "OK" or doc["kind"] != "page":
            raise ValueError("Only acquired usable prior pages may be inherited")
        public_url(doc["url"], allowed_hosts)
        if utc(doc["available_at"]) > utc(now) or utc(doc["available_at"]) >= utc(deadline):
            raise ValueError("Inherited document was unavailable at this decision time")
        documents.append(deepcopy(doc))
    return {"parent_root": str(root), "parent_run_id": run_id, "parent_head_hash": expected_head,
            "document_ids": list(document_ids), "imported_at": utc(now).isoformat(),
            "documents": documents, "parent_verification": proof,
            "new_http_attempts": 0, "fresh_observation": False}
