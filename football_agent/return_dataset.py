"""Point-in-time, acquisition-level evidence for learning query return models.

One executed request is one row. Unselected candidates have no response label;
inherited documents are context rather than new observations. No fitted return
law or counterfactual utility is inferred by this exporter.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from urllib.parse import urlsplit

from .audit import RunStore, atomic_write, digest
from .inquiry_contract import utc
from .inquiry_exploration import DESIGN, SCOPE, VERSION as EXPLORATION_VERSION
from .inquiry_verify import verify


VERSION = "inquiry-return-dataset-v2"
ORIGINS = {"operational_network", "constructed", "unknown"}


def _review_events(run, field, cutoff):
    result = {}
    for event in run["events"]:
        if utc(event["at"]) > cutoff:
            continue
        for review in event["snapshot"].get(field, []):
            result.setdefault(review["request_id"], {"available_at": event["at"],
                "event_hash": event["event_hash"], "value": deepcopy(review)})
    return result


def _origin(run, declared):
    if declared not in ORIGINS:
        raise ValueError("Explicit evidence origin required")
    mode = run["configuration"]["tools"].get("execution_mode")
    if mode == "constructed_transport_no_network" and declared != "constructed":
        raise ValueError("Constructed transport must remain constructed evidence")
    if declared == "operational_network" and any(
        not receipt.get("real_provider_call") for receipt in run["budget"]["llm_usage"]
    ):
        raise ValueError("Operational inquiry export requires actual provider receipts")
    return declared


def _sampling_metadata(decision, state):
    """Project a replay-verified design without inventing global propensities."""
    record = decision.get("exploration_record")
    if record is None:
        return {"exploration_record": None, "selection_probability": None,
                "selection_probability_scope": None,
                "selection_probability_status": "not_logged_do_not_infer_from_chosen_action",
                "candidate_generation_probability": None, "counterfactual_utility": None}
    fields = {"version", "design", "scope", "run_id", "decision_index", "seed_commitment",
              "eligible_ids", "probabilities", "selected_p", "chosen_id", "original_candidate_id",
              "eligibility_basis", "probability_basis", "replay_scope", "candidate_generation_probability",
              "counterfactual_utility", "expected_information_value"}
    if (not isinstance(record, dict) or set(record) != fields
            or record["version"] != EXPLORATION_VERSION or record["design"] != DESIGN
            or record["scope"] != SCOPE):
        raise ValueError("Unsupported conditional exploration record")
    eligible = record["eligible_ids"]
    generated = {candidate["candidate_id"] for candidate in decision["candidates"]}
    if (not isinstance(eligible, list) or not eligible
            or any(not isinstance(candidate_id, str) for candidate_id in eligible)
            or eligible != sorted(set(eligible)) or not set(eligible) <= generated):
        raise ValueError("Exploration eligibility differs from generated candidates")
    probabilities = record["probabilities"]
    expected = 1 / len(eligible)
    if (not isinstance(probabilities, dict) or set(probabilities) != set(eligible)
            or any(type(value) not in {int, float} or value != expected for value in probabilities.values())
            or type(record["selected_p"]) not in {int, float} or record["selected_p"] != expected):
        raise ValueError("Exploration probabilities differ from the uniform design")
    config = state["configuration"].get("inquiry_exploration", {})
    if (record["chosen_id"] != decision["candidate_id"] or record["chosen_id"] not in eligible
            or record["original_candidate_id"] not in generated or record["run_id"] != state["run_id"]
            or type(record["decision_index"]) is not int
            or record["decision_index"] != len(state["decisions"])-1
            or record["seed_commitment"] != config.get("seed_commitment")):
        raise ValueError("Exploration record does not bind this frozen selected decision")
    if any(record[key] is not None for key in
           ("candidate_generation_probability", "counterfactual_utility", "expected_information_value")):
        raise ValueError("Conditional exploration does not identify global probability or utility")
    # The seed remains in the private source configuration. Strict record fields
    # prevent accidental copying of that private value into a learning export.
    return {"exploration_record": deepcopy(record), "selection_probability": record["selected_p"],
            "selection_probability_scope": SCOPE,
            "selection_probability_status": "logged_replay_verified_conditional_sampling_design",
            "candidate_generation_probability": None, "counterfactual_utility": None}


def export_run(reference, *, as_of):
    """Export a pinned audited trace with labels censored at an explicit cutoff."""
    if set(reference) != {"root", "run_id", "head_hash", "cluster_id", "origin", "cluster_basis"}:
        raise ValueError("Pinned run, evidence origin and explicit dependence-cluster basis required")
    if not reference["cluster_id"] or not reference["cluster_basis"]:
        raise ValueError("A documented dependence cluster is required")
    cutoff = utc(as_of)
    root = Path(reference["root"]).resolve()
    proof = verify(root, reference["run_id"])
    if proof["audit"]["head_hash"] != reference["head_hash"]:
        raise ValueError("Source head differs from the pinned export manifest")
    store = RunStore(root/"runs")
    run = store.load(reference["run_id"])
    origin = _origin(run, reference["origin"])
    returns = {e["data"]["request_id"]: e for e in run["events"] if e["type"] == "inquiry_acquisition_returned"}
    reviews = _review_events(run, "query_reviews", cutoff)
    forecasts = _review_events(run, "return_reviews", cutoff)
    cancelled = {e["data"]["request_id"] for e in run["events"]
                 if e["type"] == "inquiry_acquisition_cancelled" and utc(e["at"]) <= cutoff}
    rows = []
    for intent in run["events"]:
        if intent["type"] != "inquiry_acquisition_intent" or utc(intent["at"]) > cutoff:
            continue
        data, state = intent["data"], intent["snapshot"]
        request_id = data["request_id"]
        if request_id in cancelled:
            # A verified preflight cancellation sent no source request. Before
            # cancellation became observable it remains a censored intention.
            continue
        selected = data["selected_candidate"]
        decision = state["decisions"][-1]
        if decision["kind"] != "query" or decision["candidate_id"] != selected["candidate_id"]:
            raise ValueError("Acquisition does not match the recorded selected proposal")
        sampling = _sampling_metadata(decision, state)
        design = sampling["exploration_record"]
        known_ids = {d["document_id"] for d in state["documents"]}
        candidates = [{"candidate": deepcopy(candidate), "selected": candidate["candidate_id"] == selected["candidate_id"],
                       "sampling_eligible": candidate["candidate_id"] in design["eligible_ids"] if design else None,
                       "conditional_selection_probability": design["probabilities"].get(candidate["candidate_id"], 0.0) if design else None,
                       "observed_return_label": None} for candidate in decision["candidates"]]
        before = {"task": deepcopy(state["task"]), "query_decided_at": decision["at"], "request_intent_at": intent["at"],
            "selected_candidate": deepcopy(selected), "candidate_set": candidates,
            "selection_basis": decision["selection_basis"], **sampling,
            "budget": deepcopy(state["budget"]), "questions": deepcopy(state["questions"]),
            "hypotheses": deepcopy(state.get("hypotheses", [])), "requirements": deepcopy(state.get("requirements", [])),
            "claims": deepcopy(state["claims"]), "documents": deepcopy(state["documents"]),
            "prior_receipts": deepcopy(state["receipts"]), "inquiry_contract": deepcopy(data.get("inquiry_contract")),
            "tool_configuration": deepcopy(state["configuration"]["tools"])}
        after = returns.get(request_id)
        observed, classification = None, "CENSORED_AT_EXPORT_CUTOFF"
        if after and utc(after["at"]) <= cutoff:
            result = store.get_blob(run["run_id"], after["data"]["output_ref"])
            doc = after["snapshot"]["documents"][-1]
            if doc["document_id"] in known_ids:
                raise ValueError("New return leaked into its own pre-query context")
            observed = {"available_at": after["at"], "observation_id": result["observation"]["id"],
                "request_id": request_id, "document": deepcopy(doc), "http_status": doc["http_status"],
                "transport_error": doc.get("error_class"), "document_status": doc["status"],
                "raw_sha256": doc["raw_sha256"], "latency_seconds": result["latency_seconds"],
                "actual_money": result["money"], "query_review": deepcopy(reviews.get(request_id)),
                "return_review": deepcopy(forecasts.get(request_id)),
                "source_truth_label": None, "realized_decision_utility": None,
                "semantic_scope": "recorded_extraction_not_independent_truth_or_query_utility",
                "label_event_hash": after["event_hash"]}
            classification = "OBSERVED"
        rows.append({"schema_version": VERSION, "row_id": digest({"run_id": run["run_id"], "request_id": request_id}),
            "run_id": run["run_id"], "request_id": request_id, "cluster_id": reference["cluster_id"],
            "cluster_basis": reference["cluster_basis"], "origin": origin, "status": classification,
            "pre_query": before, "observed_return": observed,
            "provenance": {"root": str(root), "head_hash": reference["head_hash"], "intent_event_hash": intent["event_hash"],
                           "verification_status": proof["status"]},
            "counterfactual_response_labels_available": False})
    inherited = sum(len(e["data"]["document_ids"]) for e in run["events"]
                    if e["type"] == "inquiry_evidence_inherited" and utc(e["at"]) <= cutoff)
    return rows, {"reference": deepcopy(reference), "actual_execution_rows": len(rows),
                  "inherited_documents_not_counted": inherited,
                  "verified_head_hash": proof["audit"]["head_hash"],
                  **({"unexecuted_cancelled_requests_excluded": len(cancelled)} if cancelled else {})}


def summarize(rows):
    actual = [r for r in rows if r["origin"] == "operational_network" and r["observed_return"]]
    sampled = [r for r in rows if r["pre_query"]["exploration_record"] is not None]
    operational = [r for r in rows if r["origin"] == "operational_network"]
    operational_sampled = [r for r in sampled if r["origin"] == "operational_network"]
    singleton = sum(len(r["pre_query"]["exploration_record"]["eligible_ids"]) == 1 for r in operational_sampled)
    groups = {}
    for row in actual:
        pre, ret = row["pre_query"], row["observed_return"]
        candidate = pre["selected_candidate"]
        # Descriptive groups only. Host equality is not conditional exchangeability.
        host = urlsplit(candidate["url"]).hostname if candidate["tool"] == "read_url" else "search_provider"
        key = candidate["tool"]+":"+host
        group = groups.setdefault(key, {"requests": 0, "statuses": Counter(), "clusters": set()})
        group["requests"] += 1
        group["statuses"][ret["document_status"]] += 1
        group["clusters"].add(row["cluster_id"])
    for group in groups.values():
        group["statuses"] = dict(group["statuses"])
        group["clusters"] = sorted(group["clusters"])
    clusters = sorted({r["cluster_id"] for r in actual})
    return {"rows": len(rows), "operational_observed_requests": len(actual),
        "constructed_rows": sum(r["origin"] == "constructed" for r in rows),
        "unknown_origin_rows": sum(r["origin"] == "unknown" for r in rows),
        "unobserved_rows": sum(r["observed_return"] is None for r in rows),
        "dependence_clusters": clusters, "cluster_count": len(clusters), "groups": groups,
        "transport_status_counts": dict(Counter(r["observed_return"]["document_status"] for r in actual)),
        "fitting_status": "DESCRIPTIVE_OBSERVATIONS_ONLY_NO_CALIBRATED_LAW",
        "selection_propensities_available": bool(sampled),
        "selection_propensity_scope": SCOPE if sampled else None,
        "selection_probability_coverage": {
            "logged_rows": len(sampled), "unlogged_rows": len(rows)-len(sampled),
            "operational_logged_rows": len(operational_sampled),
            "operational_unlogged_rows": len(operational)-len(operational_sampled),
            "operational_logged_observed_rows": sum(r["observed_return"] is not None for r in operational_sampled),
            "operational_singleton_rows": singleton,
            "operational_multicandidate_rows": len(operational_sampled)-singleton},
        "global_action_propensities_available": False,
        "candidate_generation_probabilities_available": False,
        "effective_independent_sample_size": None,
        "calibrated_return_model": None, "utility_labels_available": False,
        "warnings": ["A run, HTTP attempt or inherited document is not automatically an independent task.",
            "HTTP/content availability, extracted evidence, truth and decision value are different labels.",
            "Adaptive selection and shared sources preclude an unqualified IID interpretation.",
            "Unselected actions have no observed response; policy values cannot be inferred from these counts.",
            "Logged sampling probabilities are conditional on generated eligible candidates, not global action propensities.",
            "A singleton eligible set has conditional probability one and supplies no within-state candidate comparison.",
            "Transport success rates do not identify P(return | latent match state)."]}


def build_dataset(references, *, as_of):
    rows, sources, seen, observations = [], [], {}, {}
    for reference in references:
        exported, source = export_run(reference, as_of=as_of)
        sources.append(source)
        for row in exported:
            if row["row_id"] in seen:
                if row != seen[row["row_id"]]:
                    raise ValueError("The same query identity has conflicting evidence or cluster assignments")
                continue
            ret = row["observed_return"]
            if ret and ret["observation_id"] in observations:
                raise ValueError("One raw observation was attributed to distinct physical queries")
            if ret:
                observations[ret["observation_id"]] = row["row_id"]
            rows.append(row)
            seen[row["row_id"]] = row
    rows.sort(key=lambda r: (r["pre_query"]["request_intent_at"], r["row_id"]))
    return {"schema_version": VERSION, "as_of": utc(as_of).isoformat(), "sources": sources,
            "rows": rows, "summary": summarize(rows), "data_hash": digest(rows),
            "new_network_requests": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8-sig"))
    target = Path(args.output)
    if target.exists():
        raise ValueError("Use a new output path; previous exports are immutable evidence")
    dataset = build_dataset(manifest["runs"], as_of=manifest["as_of"])
    atomic_write(target, dataset)
    print(json.dumps(dataset["summary"], ensure_ascii=True))


if __name__ == "__main__":
    main()
