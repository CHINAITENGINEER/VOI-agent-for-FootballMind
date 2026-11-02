"""One predeclared HTTP action, using existing transport and durable intent.

This is not a discovery crawler or an autonomous policy. Callers choose among a
bounded public catalog; missing prediction/value models remain NOT_CONFIGURED.
"""
from __future__ import annotations

from football_platform.collectors.http_transport import RequestTransport, TransportError
from .acquisition_journal import AcquisitionDecisionJournal


DEFAULT_POLICY_VERSION = "manual-cup-context-capture-v1"


def capture_once(journal: AcquisitionDecisionJournal, run_id: str, *, request_id: str, action_id: str,
                 source: str, endpoint: str, params=None, headers=None, session=None,
                 candidates=None, candidate_metadata=None,
                 policy_version=None,
                 selection_basis="manual_predeclared", selection_probability=None,
                 assessment=None, reserve_requests=1, timeout=(8, 18)) -> dict:
    if policy_version is None:
        assigned = journal.read(run_id).get("policy_version")
        policy_version = assigned or DEFAULT_POLICY_VERSION
    intent = journal.begin(run_id, request_id=request_id, action_id=action_id,
                           input_parameters={"endpoint": endpoint, "params": params or {}, "source": source},
                           candidates=candidates or [action_id], policy_version=policy_version,
                           candidate_metadata=candidate_metadata,
                           selection_basis=selection_basis,
                           selection_probability=selection_probability,
                           assessment=assessment,
                           reserve_requests=reserve_requests)
    if not intent["execute"]:
        if session is not None:
            session.close()
        return {"new_http_attempts": 0, "intent_status": intent["recovery_status"],
                "receipt": intent["receipt"], "recovery": None if intent["receipt"] else journal.recover_pending(run_id)}
    receipts = []

    def persist(**facts):
        observation = journal.archive.save(
            facts["source"], facts["endpoint"], facts["params"], facts["body"], status=facts["http_status"],
            headers=facts["response_headers"], request_id=facts["request_id"], observed_at=facts["observed_at"],
            observation_id=facts["observation_id"], error_class=facts["error_class"],
            context={"journal_run_id": run_id, "journal_request_id": request_id,
                     "requested_at": facts["requested_at"].isoformat(), "attempt": facts["attempt"],
                     "scope": "prospective_predeclared_information_acquisition"})
        receipts.append(journal.complete(run_id, request_id=request_id, observation=observation))

    with RequestTransport(source, on_attempt=persist, session=session, max_attempts=1, request_budget=reserve_requests,
                          interval=0, timeout=timeout, clock=journal.clock) as transport:
        try:
            transport.get(endpoint, params, headers=headers)
            status = "RETURNED"
        except TransportError as error:
            status = error.category
        statistics = dict(transport.stats)
    return {"new_http_attempts": statistics["attempts"], "network_status": status, "statistics": statistics,
            "receipt": receipts[0] if receipts else None}
