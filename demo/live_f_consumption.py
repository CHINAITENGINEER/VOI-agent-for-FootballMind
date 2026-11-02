"""Inspect the actual frozen f computation; no probability changes are fabricated."""
from collections import Counter
from hashlib import sha256

import numpy as np

from football_research.acquisition import entity_evidence_updates as temporal
from football_research.acquisition import scoped_reviewed_f as scoped
from football_research.acquisition import reviewed_rich_f as rich
from football_research.acquisition.context_predictor import _digest


def consumption_audit(model, before_row, after_row, input_id, body, claims, before, after):
    resolved, resolution = temporal.resolve(after_row)
    previous, _ = temporal.resolve(before_row)
    routed, routes = scoped.route(resolved, model["head"]["mode"])
    previous_routed, _ = scoped.route(previous, model["head"]["mode"])
    head = model["head"]["head"]
    x = rich.features(head["encoder"], routed)
    x_before = rich.features(head["encoder"], previous_routed)
    coefficients = np.asarray(head["coefficients"], dtype=float)
    delta = x - x_before
    new_evidence = {_digest(e["evidence"]) for e in after_row["evidence_events"] if e["input_id"] == input_id}
    candidate_routes = [r for r in routes if r["evidence_sha256"] in new_evidence]
    counts = Counter(r["lane"] for r in candidate_routes)
    supported = sum(v for k, v in counts.items() if k != "unsupported")
    shift = float(np.abs(np.asarray(after) - np.asarray(before)).sum())
    market_shift = float(np.abs(np.asarray(after) - np.asarray(after_row["prior"])).sum())
    delta_logits = delta @ coefficients
    if shift > 1e-12:
        status, reason = "prediction_changed", "learned_evidence_residual"
    elif not claims:
        status, reason = "consumed_no_change", "no_source_claims"
    elif not supported:
        status, reason = "consumed_no_change", "no_supported_match_or_context_claims"
    elif not np.count_nonzero(np.abs(delta) > 1e-12):
        status, reason = "consumed_no_change", "duplicate_or_no_new_encoded_features"
    else:
        status, reason = "consumed_no_change", "no_discriminative_learned_residual"
    return {"schema": "footballmind-live-f-consumption-v1", "status": status, "reason": reason,
            "body_consumed": input_id in after_row["used_input_ids"], "input_id": input_id,
            "body_sha256": sha256(body.encode("utf-8")).hexdigest(), "body_characters": len(body),
            "claims_count": len(claims), "modeled_claims": supported, "route_counts": dict(counts),
            "claim_routes": candidate_routes, "model_feature_dimension": len(x),
            "nonzero_model_features": int(np.count_nonzero(np.abs(x) > 1e-12)),
            "new_nonzero_features": int(np.count_nonzero(np.abs(delta) > 1e-12)),
            "logit_delta": delta_logits.tolist(), "logit_residual": (x @ coefficients).tolist(),
            "probability_l1_change": shift, "market_probability_l1_difference": market_shift,
            "conflicts": resolution["conflicts"], "training_performed": False,
            "probability_changed_is_improvement": None}


def pipeline_summary(run):
    body_docs = [d for d in run.get("documents", []) if d.get("status") == "OK"
                 and (d.get("structured_news") or {}).get("role") == "article" and d.get("text", "").strip()]
    steps = [s for s in run.get("steps", []) if s.get("f_executed") is True]
    audits = [s.get("f_consumption") for s in steps]
    known = [a for a in audits if a is not None]
    changed = 0
    for step in steps:
        before, after = step.get("prediction_before"), step.get("prediction_after")
        if before is not None and after is not None:
            changed += float(np.abs(np.asarray(after)-np.asarray(before)).sum()) > 1e-12
    return {"schema": "footballmind-live-body-pipeline-v1", "body_documents": len(body_docs),
            "body_attempts": run.get("counts", {}).get("body_attempts", 0),
            "semantic_requests": run.get("counts", {}).get("provider", 0), "f_body_updates": len(steps),
            "claim_review_requests":sum(a.get('kind')=='claim_review' for a in run.get('semantic_attempts',[])),
            "review_rejected_claims":sum(len(s['claim_review'].get('rejected',[])) for s in steps)
                if all('claim_review' in s for s in steps) else None,
            "modeled_claims": sum(a["modeled_claims"] for a in known) if len(known) == len(steps) else None,
            "known_modeled_claims": sum(a["modeled_claims"] for a in known), "audited_updates": len(known),
            "changed_updates": int(changed),
            "semantic_rejected_bodies": sum(r.get("reason")=="semantic_validation_failed" for r in run.get("body_rejections",[])),
            "market_difference_l1": float(np.abs(np.asarray(run["current"]) - np.asarray(run["prior"])).sum())
                if run.get("current") is not None and run.get("prior") is not None else None,
            "initialization_is_body_update": False, "training_performed": False}
