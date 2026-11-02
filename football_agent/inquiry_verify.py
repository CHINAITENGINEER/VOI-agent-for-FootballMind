"""Offline evidence/time/budget replay for an inquiry; performs no network calls."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from football_spider.models.contracts import Observation
from football_spider.storage.raw_archive import RawArchive
from .audit import RunStore, atomic_write, digest
from .inquiry_contract import validate_response, utc
from .inquiry_tools import OpenWebTools
from .inquiry_hypotheses import get_protocol
from .inquiry_continuation import get_continuation, duplicate_query_blocked


def canonical_query_url(value):
    parts = urlsplit(value)
    return urlunsplit((parts.scheme, parts.netloc, parts.path,
                       urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True))), parts.fragment))


def verify(root, run_id, *, _ancestors=()):
    root = Path(root).resolve()
    identity = (str(root), run_id)
    if identity in _ancestors or len(_ancestors) >= 24:
        raise ValueError("Cyclic or excessive inquiry inheritance")
    ancestors = (*_ancestors, identity)
    store, archive = RunStore(root / "runs"), RawArchive(root / "evidence")
    audit, run = store.verify(run_id), store.load(run_id)
    protocol = get_protocol(run["configuration"].get("inquiry_protocol"),
                            decision_problem=run["configuration"].get("decision_problem"))
    from .inquiry_selection import get_selector
    selector = get_selector(run["configuration"].get("decision_selector"))
    continuation = get_continuation(run["configuration"].get("continuation_policy"))
    if getattr(protocol, "requires_continuation", False) and continuation is None:
        raise ValueError("Information choice is missing its continuation configuration")
    if "source_catalog" in run["configuration"]:
        from .inquiry_choice import validate_catalog
        catalog = run["configuration"]["source_catalog"]
        if (not getattr(protocol, "supports_source_catalog", False) or
                catalog != validate_catalog(catalog, run["configuration"]["tools"]["allowed_hosts"]) or
                {item["url"] for item in catalog} - set(run["task"]["seed_urls"])):
            raise ValueError("Source catalog differs from the configured inquiry seed scope")
    if continuation and selector:
        raise ValueError("LLM continuation admission does not cover deterministic selectors")
    initial_budget = run["events"][0]["snapshot"]["budget"]
    initial_task = run["events"][0]["snapshot"]["task"]
    continuation_events = {"inquiry_started", "inquiry_evidence_inherited", "inquiry_reasoning_reserved",
        "inquiry_reasoning_failed", "inquiry_reasoning_returned", "inquiry_response_rejected", "inquiry_decision",
        "inquiry_reflection_completed", "inquiry_acquisition_intent", "inquiry_acquisition_failed",
        "inquiry_acquisition_returned", "inquiry_acquisition_cancelled", "inquiry_paused_or_stopped", "inquiry_wait_resumed"}
    transport_policy = run["configuration"]["tools"].get("transport_policy")
    if transport_policy:
        from .inquiry_transport import RoutedOpenWebTools, request_header_profile
        configured_tools = RoutedOpenWebTools(archive, run["configuration"]["tools"]["allowed_hosts"],
            route=transport_policy["route"], extraction_version=1 if run["configuration"]["tools"]["version"] == "open-web-read-v1" else 2)
        if configured_tools.transport_policy() != transport_policy:
            raise ValueError("Unrecognized archived transport configuration")
    acquired, intents, decisions = [], {}, 0
    verified_decisions, verified_receipts = [], []
    deterministic_terminals = 0
    previous_event = None
    visible_documents = []
    raw_verified = 0
    real_turns = 0
    reasoning_reservations = 0
    assessments = 0
    inherited_count = 0
    projection_fields = getattr(protocol, "projection_fields", ("hypotheses", "hypothesis_states", "query_reviews")) if protocol else ()
    initial_projection = protocol.public({}) if protocol else {}
    verified_knowledge = {key: deepcopy(initial_projection[key]) for key in projection_fields}
    if protocol:
        verified_knowledge.update(questions=[], claims=[])
    seen_events, waits_resumed, cancelled_requests = [], 0, set()
    rejected_responses, expected_rejection, reply_valid = 0, None, False
    for event_index, event in enumerate(run["events"]):
        data = event["data"]
        # Knowledge changes only when a verified reply is projected below.
        # A hash-valid snapshot is not itself evidence for an earlier review.
        if protocol and event["type"] != "inquiry_decision":
            if any(event["snapshot"].get(key) != value for key, value in verified_knowledge.items()):
                raise ValueError("Knowledge differs from the previously verified projection prefix")
        if continuation and event["snapshot"]["configuration"] != run["configuration"]:
            raise ValueError("Continuation configuration changed inside the recorded run")
        if continuation:
            snapshot = event["snapshot"]
            budget = snapshot["budget"]
            expected_receipts = [*verified_receipts, {k:v for k,v in data.items() if k != "output_ref"}] if event["type"] == "inquiry_acquisition_returned" else verified_receipts
            if snapshot["receipts"] != expected_receipts:
                raise ValueError("Acquisition history differs from the verified receipt prefix")
            if snapshot["task"] != initial_task or run["task"] != initial_task:
                raise ValueError("Continuation task or deadline changed after start")
            if event["type"] not in continuation_events:
                raise ValueError("Unrecognized continuation event")
            if any(budget[k] != initial_budget[k] for k in ("max_requests", "max_llm_calls")):
                raise ValueError("Continuation budget limits changed after start")
            expected_llm = reasoning_reservations+int(event["type"] == "inquiry_reasoning_reserved")
            expected_http = raw_verified+int(event["type"] == "inquiry_acquisition_returned")
            if (budget["llm_calls"] != expected_llm or budget["http_attempts"] != expected_http
                    or budget["reserved_requests"] != int(bool(snapshot["pending"] and snapshot["pending"]["kind"] == "http"))):
                raise ValueError("Continuation counters differ from actual reservations and returns")
            if (not 0 <= budget["llm_calls"] <= budget["max_llm_calls"]
                    or not 0 <= budget["http_attempts"]+budget["reserved_requests"] <= budget["max_requests"]):
                raise ValueError("Continuation attempts exceed the initial hard budget")
            if previous_event and utc(event["at"]) < utc(previous_event["at"]):
                raise ValueError("Continuation clock moved backwards")
            if previous_event and previous_event["snapshot"]["state"] in {"STOPPED", "DEFERRED"}:
                raise ValueError("A terminal inquiry was reactivated")
            if event["type"] == "inquiry_paused_or_stopped":
                if (data["state"] not in {"WAITING", "STOPPED", "DEFERRED"}
                        or snapshot["state"] != data["state"] or snapshot["reason"] != data["reason"]):
                    raise ValueError("Paused state differs from its transition record")
                if data["state"] == "WAITING" and (not previous_event or previous_event["type"] != "inquiry_decision"
                        or not verified_decisions or verified_decisions[-1]["kind"] != "wait"
                        or snapshot["revisit_at"] != verified_decisions[-1]["revisit_at"]):
                    raise ValueError("Waiting state lacks the verified wait decision")
            elif event["type"] != "inquiry_wait_resumed" and (snapshot["state"] != "READY"
                    or previous_event and previous_event["snapshot"]["state"] != "READY"):
                raise ValueError("Active inquiry continued without a verified due resume")
        appends_decision = event["type"] in {"inquiry_decision", "inquiry_frozen_terminal"}
        expected_history = [*verified_decisions, data] if appends_decision else verified_decisions
        if event["snapshot"].get("decisions") != expected_history:
            raise ValueError("Decision history differs from the previously verified decision prefix")
        if event["type"] == "inquiry_evidence_inherited":
            from .inquiry_memory import load_inherited
            if not getattr(protocol, "supports_inheritance", False) or visible_documents or reasoning_reservations:
                raise ValueError("Inherited evidence must precede first reasoning")
            record = store.get_blob(run_id, data["input_ref"])
            reproduced = load_inherited(record["parent_root"], record["parent_run_id"], record["parent_head_hash"],
                                        record["document_ids"], now=record["imported_at"], deadline=run["task"]["deadline"],
                                        allowed_hosts=run["configuration"]["tools"]["allowed_hosts"], ancestors=ancestors)
            # The parent verification may gain extra diagnostic fields in a
            # later verifier. Pin its head and compare the actual inherited data.
            if {k:v for k,v in reproduced.items() if k != "parent_verification"} != {k:v for k,v in record.items() if k != "parent_verification"}:
                raise ValueError("Inherited evidence differs from pinned source run")
            if not utc(run["created_at"]) <= utc(record["imported_at"]) <= utc(event["at"]) < utc(run["task"]["deadline"]):
                raise ValueError("Evidence import violates current operational time")
            visible_documents.extend(deepcopy(record["documents"]))
            acquired.extend(record["document_ids"])
            if event["snapshot"]["documents"] != visible_documents:
                raise ValueError("Imported public evidence differs from source")
            inherited_count += len(record["documents"])
        if event["type"] == "inquiry_reasoning_reserved":
            reasoning_reservations += 1
        if event["type"] == "inquiry_reasoning_returned":
            public = store.get_blob(run_id, data["input_ref"])
            reply = store.get_blob(run_id, data["output_ref"])
            if public.get("last_decision") != (verified_decisions[-1] if verified_decisions else None):
                raise ValueError("Reasoning input changed the previously verified decision")
            if [d["document_id"] for d in public["documents"]] != acquired:
                raise ValueError("Reasoning input reveals unknown or omits acquired documents")
            reply_valid, expected_rejection = False, None
            try:
                (protocol.validate if protocol else validate_response)(reply["response"], public)
                validation_tools = OpenWebTools(archive, run["configuration"]["tools"]["allowed_hosts"])
                for candidate in reply["response"]["candidates"]:
                    validation_tools.prepare(candidate, public)
                reply_valid = True
            except Exception as exc:
                expected_rejection = {"error_type": type(exc).__name__, "reason": str(exc)[:500]}
                following = run["events"][event_index+1] if event_index+1 < len(run["events"]) else None
                if not following or following["type"] != "inquiry_response_rejected" or following["data"] != expected_rejection:
                    raise ValueError("Invalid reply lacks its exactly reproduced rejection record") from exc
            if continuation:
                before_call = deepcopy(event["snapshot"])
                if (before_call["budget"]["llm_calls"] != reasoning_reservations
                        or before_call["budget"]["http_attempts"] != raw_verified):
                    raise ValueError("Continuation budget differs from recorded attempts")
                before_call["budget"]["llm_calls"] -= 1
                expected_budget = {"remaining_requests": before_call["budget"]["max_requests"]-raw_verified,
                    "remaining_llm_calls": before_call["budget"]["max_llm_calls"]-before_call["budget"]["llm_calls"]}
                if public["budget"] != expected_budget or public.get("continuation") != continuation.public(before_call):
                    raise ValueError("Published continuation feasibility differs from the actual budget")
            if selector and public.get("scoped_decision_value") != selector.public_summary(event["snapshot"]):
                raise ValueError("Published decision-risk state differs from acquired evidence")
            if protocol:
                if public["documents"] != visible_documents:
                    raise ValueError("Reasoning document content differs from actual acquired evidence")
                for key in ("task", "questions", "claims", "documents"):
                    if public[key] != event["snapshot"][key]:
                        raise ValueError("Reasoning input differs from the recorded available state")
                if any(public[k] != value for k, value in protocol.public(event["snapshot"]).items()):
                    raise ValueError("Reasoning input differs from projected protocol state")
            real_turns += int(reply["receipt"].get("real_provider_call", False))
        elif event["type"] == "inquiry_response_rejected":
            if (not previous_event or previous_event["type"] != "inquiry_reasoning_returned"
                    or reply_valid or expected_rejection is None or data != expected_rejection):
                raise ValueError("Response rejection is not bound to the invalid returned reply")
            following = run["events"][event_index+1] if event_index+1 < len(run["events"]) else None
            if (not following or following["type"] != "inquiry_paused_or_stopped"
                    or following["data"]["state"] != "DEFERRED" or following["data"]["reason"] != "INVALID_INQUIRY_RESPONSE"
                    or event_index+2 != len(run["events"])):
                raise ValueError("Rejected reply must terminate without executing its proposed action")
            rejected_responses += 1
        elif event["type"] == "inquiry_decision":
            if not reply_valid:
                raise ValueError("A decision requires an admitted valid model reply")
            allowed_basis = {"llm_bounded_exploration"}
            if continuation:
                allowed_basis.add("operational_continuation_feasibility")
            if not selector and (data["expected_gain"] is not None or data["selection_basis"] not in allowed_basis):
                raise ValueError("Unsupported value was relabeled as numerical VoI")
            decisions += 1
            if protocol or continuation:
                response = reply["response"]
                selection_state = deepcopy(event["snapshot"])
                selection_state["decisions"] = selection_state["decisions"][:-1]
                selected = selector.select(selection_state, response) if selector else response["decision"]
                if continuation:
                    if not previous_event or previous_event["type"] != "inquiry_reasoning_returned" or not (
                            utc(previous_event["at"]) <= utc(data["at"]) <= utc(event["at"])):
                        raise ValueError("Continuation admission uses an invalid decision time")
                    selected = {**selected, **continuation.apply(selection_state, response, selected, now=data["at"])}
                    if ("proposal_before_admission" in data) != (not selected["continuation_check"]["admitted"]):
                        raise ValueError("Blocked proposal provenance differs from admission outcome")
                if (selector or continuation) and any(data[key] != value for key, value in selected.items()):
                    raise ValueError("Selected inquiry decision differs from scoped risk calculation")
                for key in ("kind", "candidate_id", "reason", "revisit_at"):
                    if data[key] != selected[key]:
                        raise ValueError("Recorded decision differs from provider response")
                for key in ("claim_updates", "candidates"):
                    if data[key] != response[key]:
                        raise ValueError("Recorded inquiry updates differ from provider response")
            if protocol:
                claims = {c["claim_id"]: c for c in public["claims"]}
                claims.update({c["claim_id"]: c for c in response["claim_updates"]})
                questions = {q["question_id"]: q for q in public["questions"]}
                questions.update({q["question_id"]: q for q in response["questions"]})
                if event["snapshot"]["questions"] != list(questions.values()):
                    raise ValueError("Question state differs from the verified reply")
                stored_claims = event["snapshot"]["claims"]
                updated_ids = {c["claim_id"] for c in response["claim_updates"]}
                if len(stored_claims) != len(claims):
                    raise ValueError("Claim state differs from the verified reply")
                for stored, expected in zip(stored_claims, claims.values()):
                    if stored["claim_id"] in updated_ids:
                        content = {k:v for k,v in stored.items() if k not in {"updated_at", "semantic_verification"}}
                        if (content != expected or stored.get("semantic_verification") != "llm_extraction_quote_checked_meaning_unverified"
                                or not previous_event or not utc(previous_event["at"]) <= utc(stored["updated_at"]) <= utc(event["at"])):
                            raise ValueError("Extracted claim differs from its verified reply or extraction time")
                    elif stored != expected:
                        raise ValueError("Previously extracted claim changed outside its verified reply")
                replay = {**deepcopy(public), "claims": list(claims.values()),
                          "receipts": deepcopy(event["snapshot"]["receipts"])}
                assessment = protocol.advance(replay, response)
                if assessment != data["hypothesis_assessment"]:
                    raise ValueError("Hypothesis or query review differs from deterministic projection")
                for key in projection_fields:
                    if replay[key] != event["snapshot"][key]:
                        raise ValueError("Projected knowledge differs from stored state")
                    verified_knowledge[key] = deepcopy(replay[key])
                verified_knowledge.update(questions=deepcopy(list(questions.values())), claims=deepcopy(stored_claims))
                assessments += 1
        elif event["type"] == "inquiry_frozen_terminal":
            settle = getattr(selector, "settle_after_return", None)
            if not callable(settle) or not previous_event or previous_event["type"] != "inquiry_acquisition_returned":
                raise ValueError("Frozen terminal must follow an acquired return under a supporting selector")
            snapshot = event["snapshot"]
            before = previous_event["snapshot"]
            # A deterministic terminal reads the returned evidence; no LLM
            # response, extra receipt, changed task or projected claim enters
            # between the verified acquisition and this committed decision.
            unchanged = {"task", "configuration", "documents", "receipts", "budget", "questions", "claims", "pending"}
            unchanged.update(getattr(protocol, "projection_fields", ()))
            if any(snapshot.get(key) != before.get(key) for key in unchanged):
                raise ValueError("Frozen terminal inputs differ from the actual acquired-return state")
            if snapshot["documents"] != visible_documents or not snapshot["receipts"]:
                raise ValueError("Frozen terminal lacks verified actual evidence")
            receipt = snapshot["receipts"][-1]
            if receipt["request_id"] != previous_event["data"]["request_id"]:
                raise ValueError("Frozen terminal does not bind the immediately preceding return")
            if ("at" not in data or not utc(previous_event["at"]) <= utc(data["at"]) <= utc(event["at"])
                    < utc(run["task"]["deadline"])):
                raise ValueError("Frozen terminal violates the acquisition or deadline execution time")
            selection_state = deepcopy(snapshot)
            selection_state["decisions"] = deepcopy(verified_decisions)
            selected = settle(selection_state)
            if selected is None or selected.get("kind") not in {"stop", "defer"}:
                raise ValueError("Frozen terminal has no applicable committed response policy")
            expected = {**selected, "at": data["at"], "candidates": [], "claim_updates": [],
                        "knowledge_changed": False, "decision_origin": "deterministic_frozen_policy"}
            if data != expected:
                raise ValueError("Frozen terminal differs from the verified acquisition commitment")
            deterministic_terminals += 1
        elif event["type"] == "inquiry_acquisition_intent":
            if selector or continuation:
                decision = event["snapshot"]["decisions"][-1]
                candidate = next((c for c in decision["candidates"] if c["candidate_id"] == decision["candidate_id"]), None)
                if decision["kind"] != "query" or data["selected_candidate"] != candidate:
                    raise ValueError("Executed candidate differs from scoped selected decision")
                prepared_tools = OpenWebTools(archive, run["configuration"]["tools"]["allowed_hosts"])
                if data["action"] != prepared_tools.prepare(candidate, public) or digest(data["action"]) != data["execution_key"]:
                    raise ValueError("Executed request differs from selected candidate parameters")
                if continuation:
                    history_state = {**deepcopy(event["snapshot"]), "events": seen_events}
                    if duplicate_query_blocked(history_state, data["execution_key"]):
                        raise ValueError("Repeated query lacks an unused due wait window")
            if protocol and data["inquiry_contract"] != protocol.contract(event["snapshot"], data["selected_candidate"]):
                raise ValueError("Query intent does not bind the pre-acquisition hypothesis state")
            intents[data["request_id"]] = event
        elif event["type"] == "inquiry_acquisition_returned":
            result = store.get_blob(run_id, data["output_ref"])
            if result["request_id"] in cancelled_requests:
                raise ValueError("A cancelled request was relabeled as an executed return")
            intent = intents[result["request_id"]]
            action = intent["data"]["action"]
            if protocol and data["inquiry_contract"] != intent["data"]["inquiry_contract"]:
                raise ValueError("Query intent changed after acquisition")
            observation = Observation.model_validate(result["observation"])
            archive.read_body(observation)
            payload = archive.root / observation.payload_path
            metadata = payload.with_name(payload.name[:-len(".body.gz")] + ".observation.json")
            if Observation.model_validate_json(metadata.read_bytes()).model_dump() != observation.model_dump():
                raise ValueError("Archived observation metadata mismatch")
            if observation.context.get("inquiry_run_id") != run_id or observation.context.get("inquiry_request_id") != result["request_id"]:
                raise ValueError("Observation is not bound to this acquisition")
            if (observation.endpoint != action["endpoint"] or observation.params != action["params"]
                    or observation.source != action["source"]):
                raise ValueError("Returned source differs from selected action")
            if transport_policy:
                route = observation.context.get("transport", {})
                profile = request_header_profile(action["endpoint"])
                expected_route = {"version": transport_policy["version"], "route": transport_policy["route"],
                    "policy_sha256": digest(transport_policy), "tls_verification": True,
                    "authentication": "none_no_netrc", "request_header_preset": profile["preset_id"],
                    "request_headers": profile["headers"]}
                if any(route.get(k) != v for k, v in expected_route.items()):
                    raise ValueError("Acquisition route or header preset differs from the registered transport")
            if not (utc(intent["at"]) <= utc(observation.context["requested_at"])
                    <= utc(observation.observed_at) <= utc(result["document"]["available_at"]) <= utc(event["at"])):
                raise ValueError("Acquisition violates recorded operational order")
            tools = OpenWebTools(archive, run["configuration"]["tools"]["allowed_hosts"],
                                 clock=lambda: utc(result["document"]["available_at"]),
                                 extraction_version=1 if run["configuration"]["tools"]["version"] == "open-web-read-v1" else 2)
            extracted, saved = tools.project(observation, action), dict(result["document"])
            # The JSON store sorts parameter object keys. Query order is not
            # significant for these unique-key requests; preserve every value.
            extracted["url"] = canonical_query_url(extracted["url"])
            saved["url"] = canonical_query_url(saved["url"])
            if extracted != saved:
                raise ValueError("Public document differs from deterministic raw extraction")
            if digest(action) != data["execution_key"] or result["http_attempts"] != 1:
                raise ValueError("Acquisition identity/cost mismatch")
            if protocol:
                visible = deepcopy(result["document"])
                stored_visible = event["snapshot"]["documents"][-1]
                if stored_visible["status"] == "LATE_RETURN":
                    if utc(event["at"]) < utc(run["task"]["deadline"]):
                        raise ValueError("Evidence was marked late before the deadline")
                    visible.update(status="LATE_RETURN", text="", links=[])
                elif utc(visible["available_at"]) >= utc(run["task"]["deadline"]):
                    raise ValueError("Late evidence entered usable inquiry state")
                for previous in visible_documents:
                    if previous["status"] == "OK" and visible["raw_sha256"] == previous["raw_sha256"]:
                        visible["duplicate_of"] = previous["document_id"]
                        break
                visible_documents.append(visible)
                if event["snapshot"]["documents"] != visible_documents:
                    raise ValueError("Stored document content differs from actual acquired evidence")
            acquired.append(result["document"]["document_id"])
            raw_verified += 1
        elif event["type"] == "inquiry_wait_resumed":
            if not continuation or not previous_event or previous_event["type"] != "inquiry_paused_or_stopped":
                raise ValueError("Wait resume lacks a configured continuation policy")
            before, after = previous_event["snapshot"], event["snapshot"]
            if (before["state"] != "WAITING" or before["pending"] or after["state"] != "READY"
                    or after["reason"] != "WAIT_WINDOW_REACHED" or after["revisit_at"] is not None
                    or data != {"scheduled_at": before["revisit_at"], "resumed_at": data["resumed_at"],
                                "budget_refilled": False, "new_source_observations": 0, "publication_confirmed": False}):
                raise ValueError("Wait resume differs from the paused state")
            if (not (utc(previous_event["at"]) <= utc(data["resumed_at"]) <= utc(event["at"]) < utc(run["task"]["deadline"]))
                    or utc(data["resumed_at"]) < utc(before["revisit_at"])
                    or before["budget"]["llm_calls"] >= before["budget"]["max_llm_calls"]):
                raise ValueError("Wait resumed before due time, after deadline or without model budget")
            unchanged = set(before)-{"state", "reason", "revisit_at"}
            if set(before) != set(after) or any(before[k] != after[k] for k in unchanged):
                raise ValueError("Wait resume changed evidence, configuration or remaining budget")
            waits_resumed += 1
        elif event["type"] == "inquiry_acquisition_cancelled":
            if (not continuation or not previous_event or previous_event["type"] != "inquiry_acquisition_intent"
                    or data != {"request_id": previous_event["data"]["request_id"],
                                "reason": "DEADLINE_BEFORE_TOOL_INVOCATION", "http_attempts": 0}
                    or utc(event["at"]) < utc(run["task"]["deadline"]) or event["snapshot"]["pending"]):
                raise ValueError("Invalid unexecuted acquisition cancellation")
            cancelled_requests.add(data["request_id"])
        if appends_decision:
            verified_decisions.append(deepcopy(data))
        if event["type"] == "inquiry_acquisition_returned":
            verified_receipts.append({k:deepcopy(v) for k,v in data.items() if k != "output_ref"})
        previous_event = event
        seen_events.append(event)
    if run["decisions"] != verified_decisions:
        raise ValueError("Final decisions differ from the verified decision sequence")
    if protocol and any(run.get(key) != value for key, value in verified_knowledge.items()):
        raise ValueError("Final knowledge differs from the verified projection sequence")
    if run["budget"]["http_attempts"] != raw_verified or run["budget"]["llm_calls"] != reasoning_reservations:
        raise ValueError("Actual and reserved call counts differ from trace")
    pending_http = int(bool(run["pending"] and run["pending"]["kind"] == "http"))
    if run["budget"]["reserved_requests"] != pending_http:
        raise ValueError("Unresolved request reservation mismatch")
    status = ("VERIFIED_RECORDED_PREFIX_RECONCILIATION_REQUIRED" if run["pending"] else
              "VERIFIED_REJECTED_RESPONSE_TRACE_NOT_COMPLETED_INQUIRY" if rejected_responses else
              "VERIFIED_INQUIRY_EXECUTION_NOT_VOI_EFFECTIVENESS")
    return {"status": status, "run_id": run_id,
            "audit": audit, "reasoning_reservations": reasoning_reservations, "real_completed_llm_turns": real_turns,
            "decisions_verified": decisions+deterministic_terminals,
            "llm_decisions_verified": decisions, "deterministic_terminal_decisions_verified": deterministic_terminals,
            "frozen_terminal_decisions_replayed": deterministic_terminals,
            "raw_observations_verified": raw_verified,
            "public_evidence_order_verified": True, "source_semantics_independently_verified": False,
            "inquiry_protocol": protocol.version if protocol else None,
            "hypothesis_assessments_replayed": assessments,
            "scoped_risk_decisions_replayed": decisions+deterministic_terminals if selector and not getattr(selector, "exploration", None) else 0,
            "selector_decisions_replayed": decisions+deterministic_terminals if selector else 0,
            "exploration_choices_replayed": sum(d.get("exploration_record") is not None for d in run["decisions"]),
            "numeric_value_scope": "declared_assumptions_not_calibrated" if selector and not getattr(selector, "exploration", None) else "unsupported_open_text",
            "query_reviews_replayed": len(run.get("query_reviews", [])),
            "inherited_documents_verified": inherited_count,
            "expected_gain_supported": False, "unresolved_call": pending_summary(run["pending"]), "new_network_requests": 0,
            **({"rejected_model_responses_verified": rejected_responses,
                "acquired_returns_without_valid_review": raw_verified-len(run.get("query_reviews", []))} if rejected_responses else {}),
            **({"continuation_admissions_replayed": decisions, "due_waits_resumed": waits_resumed} if continuation else {})}


def pending_summary(value):
    # A compact marker is sufficient; do not export hidden call inputs/paths.
    return {"kind": value["kind"], "outcome": "requires_reconciliation"} if value else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = verify(args.root, args.run_id)
    atomic_write(Path(args.output), result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
