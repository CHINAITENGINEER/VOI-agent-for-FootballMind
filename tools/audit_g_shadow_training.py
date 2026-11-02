"""Create a conservative g-shadow eligibility audit.

The tool never fits a g model from historical reconstruction rows. It emits a
complete deferred package that a future prospective batch can replace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BRIDGE = ROOT / "var" / "f-g-semantic-bridge-20261010-v1"
DEFAULT_STREAM = ROOT / "footballmind-information-stream"
DEFAULT_AUDIT = ROOT / "var" / "f-entity-semantic-selection-audit-20261011-v6" / "audit-report.json"
DEFAULT_REGISTRATION = ROOT / "var" / "f-entity-semantic-selection-registration-20261011-v1" / "registration.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}: JSONL row is not an object")
            result.append(value)
    return result


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def reg10_candidate(audit: dict[str, Any]) -> dict[str, Any]:
    candidates = [x for x in audit.get("candidates", []) if x.get("regularization") == 10.0]
    if not candidates:
        raise ValueError("The regularization=10 candidate is missing from the f selection audit")
    return min(candidates, key=lambda x: x.get("mean_match_log_loss", float("inf")))


def audit_bridge(bridge: Path, audit: dict[str, Any], stream: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pre = read_jsonl(bridge / "pre-read-only.jsonl")
    post = {x.get("sample_id"): x.get("audit", {}) for x in read_jsonl(bridge / "post-read-audits.jsonl")}
    targets = {x.get("sample_id"): x.get("target", {}) for x in read_jsonl(bridge / "hindsight-targets.jsonl")}
    candidate = reg10_candidate(audit)
    rows = []
    reasons = Counter()
    physical = set()
    for envelope in pre:
        sid = envelope.get("sample_id")
        q = envelope.get("query", {})
        p = post.get(sid, {})
        t = targets.get(sid, {})
        match_id = q.get("physical_match_id") or p.get("physical_match_id") or t.get("physical_match_id")
        if match_id:
            physical.add(match_id)
        action = q.get("candidate", {})
        labels = action.get("observed_labels") or []
        provenance = t.get("label_provenance") or {}
        checks = {
            "physical_match_id": bool(match_id),
            "action_id": bool(action.get("action_id")),
            "pre_query_state": bool(q.get("current") and q.get("decision_at")),
            "declared_cost": bool(action.get("cost")),
            "request_clock": bool(q.get("requested_at") or action.get("requested_at")),
            "response_clock": bool(q.get("response_at") or action.get("response_at")),
            "body_or_structured_return": bool(p.get("new_input_ids") or p.get("new_body_input_ids")),
            "post_query_state": bool(p.get("prediction_after")),
            "outcome": "outcome" in t,
            "same_f": bool(p.get("model_version") and p.get("model_version") == q.get("model_version") == t.get("model_version")),
            "source_identity": bool(action.get("url") and match_id),
            "source_version": bool(action.get("source_version") or (labels and labels[0].get("source_version"))),
            "source_available_at": bool(any(x.get("available_at") for x in labels) and q.get("information_mode") != "historical_reconstruction"),
            "before_kickoff": bool(q.get("prediction_cutoff") and q.get("fixture", {}).get("kickoff")),
            "late_return_recorded": bool(q.get("late_return") is not None or action.get("late_return") is not None),
            "prospective_marker": t.get("prospective_training_eligible") is True,
            "real_cost_receipt": action.get("cost_status") not in {"simulated", "unknown", None},
            "result_clock_not_reconstructed": provenance.get("result_clock_assumption") is not True,
            "reg10_shadow_bound": q.get("model_version") == candidate.get("model_version"),
        }
        for key, value in checks.items():
            if not value:
                reasons[key] += 1
        eligible = all(checks.values())
        if not eligible:
            reasons["row_ineligible"] += 1
        rows.append({
            "sample_id": sid,
            "physical_match_id": match_id,
            "action_id": action.get("action_id"),
            "information_mode": q.get("information_mode"),
            "checks": checks,
            "eligible": eligible,
            "target": {
                "outcome": t.get("outcome"),
                "realized_gain_nats": t.get("realized_gain_nats"),
                "prospective_training_eligible": t.get("prospective_training_eligible"),
            },
        })
    fixtures = read_jsonl(stream / "fixtures.jsonl")
    documents = read_jsonl(stream / "documents.jsonl")
    attempts = read_jsonl(stream / "attempts.jsonl")
    stream_stats = {
        "fixtures": len(fixtures),
        "documents": len(documents),
        "attempts": len(attempts),
        "historical_reconstruction_documents": sum(x.get("information_mode") == "historical_reconstruction" for x in documents),
        "documents_with_null_available_at": sum(x.get("available_at") is None for x in documents),
        "historical_attempts": sum(bool(x.get("historical_capture")) for x in attempts),
        "physical_match_ids": len({x.get("physical_match_id") for x in fixtures if x.get("physical_match_id")}),
    }
    summary = {
        "bridge_rows": len(rows),
        "bridge_physical_matches": len(physical),
        "eligible_rows": sum(x["eligible"] for x in rows),
        "eligible_physical_matches": len({x["physical_match_id"] for x in rows if x["eligible"]}),
        "failure_reasons": dict(sorted(reasons.items())),
        "stream": stream_stats,
        "reg10_candidate": {
            "model_version": candidate.get("model_version"),
            "regularization": candidate.get("regularization"),
            "coefficients_sha256": candidate.get("coefficients_sha256"),
            "forward_prediction_changed_rows": candidate.get("forward_response", {}).get("prediction_changed_rows"),
            "forward_transition_rows": candidate.get("forward_response", {}).get("transition_rows"),
            "responsive_physical_matches": candidate.get("forward_response", {}).get("responsive_physical_matches"),
        },
    }
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, default=DEFAULT_BRIDGE)
    parser.add_argument("--stream", type=Path, default=DEFAULT_STREAM)
    parser.add_argument("--f-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    audit = read_json(args.f_audit)
    registration = read_json(args.registration)
    row_audits, summary = audit_bridge(args.bridge, audit, args.stream)
    candidate = summary["reg10_candidate"]
    audit_digest = sha256(args.f_audit)
    registration_digest = sha256(args.registration)
    now = datetime.now(timezone.utc).isoformat()

    features = {
        "schema": "footballmind-g-shadow-feature-manifest-v1",
        "status": "deferred_no_eligible_rows",
        "pre_query_only": True,
        "feature_values_written": 0,
        "features": [
            "p_before_home", "p_before_draw", "p_before_away", "prediction_entropy",
            "probability_margin", "market_probability", "f_market_divergence",
            "kickoff_lead_hours", "known_document_count", "unknown_topic_count",
            "contradiction_count", "candidate_type", "source_reliability_prior",
            "declared_cost", "expected_latency", "query_step", "remaining_request_budget",
            "same_topic_query_count", "last_query_status",
        ],
    }
    write_json(args.output / "feature-manifest.json", features)
    feature_digest = sha256(args.output / "feature-manifest.json")
    write_json(args.output / "eligibility-audit.json", {
        "schema": "footballmind-g-shadow-eligibility-audit-v1",
        "status": "deferred",
        "generated_at": now,
        "decision": "g_training_deferred",
        "reason": "insufficient_timestamped_before_after_outcome_linkage",
        "g_training_eligible_rows": summary["eligible_rows"],
        "independent_physical_match_ids_reviewed": summary["bridge_physical_matches"],
        "independent_physical_match_ids_eligible": summary["eligible_physical_matches"],
        "bridge": {"path": str(args.bridge), "report_sha256": sha256(args.bridge / "report.json")},
        "f_selection_audit": {"path": str(args.f_audit), "sha256": audit_digest},
        "registration": {"path": str(args.registration), "sha256": registration_digest, "registration_id": registration.get("registration_id")},
        "summary": summary,
        "row_audits": row_audits,
        "gate_failures": [
            "historical_reconstruction_not_prospective",
            "source_available_at_not_verified",
            "request_and_response_clocks_missing",
            "simulated_or_unverified_cost",
            "reconstructed_result_clock",
            "not_bound_to_reg10_shadow_candidate",
            "independent_blind_test_not_available",
        ],
    })
    write_jsonl(args.output / "training-data.jsonl", [])
    write_json(args.output / "label-audit.json", {
        "schema": "footballmind-g-shadow-label-audit-v1",
        "status": "deferred",
        "rows_examined": len(row_audits),
        "labeled_rows": 0,
        "selected_labeled_rows": 0,
        "unselected_unlabeled_rows": 0,
        "propensity_available_rows": 0,
        "propensity_missing_rows": len(row_audits),
        "historical_targets_retained_for_audit_only": True,
        "counterfactual_value_estimation": "unavailable_or_bounded",
    })
    write_json(args.output / "split-manifest.json", {
        "schema": "footballmind-g-shadow-split-manifest-v1",
        "status": "test_not_available",
        "group_key": "physical_match_id",
        "time_ordered": True,
        "train_physical_matches": [],
        "validation_physical_matches": [],
        "test_physical_matches": [],
        "train_rows": 0,
        "validation_rows": 0,
        "test_rows": 0,
        "reason": "no_eligible_prospective_rows",
    })
    write_json(args.output / "candidate-model-comparison.json", {
        "schema": "footballmind-g-shadow-candidate-comparison-v1",
        "status": "deferred",
        "models": [{"name": x, "status": "not_fit"} for x in [
            "constant_expected_gain", "candidate_prior_baseline", "ridge",
            "lightgbm_or_xgboost", "two_stage_gain_harm",
        ]],
        "selection_bias_status": "not_estimable",
        "reason": "g_training_eligible_rows_zero",
    })
    write_json(args.output / "g-model.json", {
        "schema": "footballmind-g-shadow-v1",
        "model_version": None,
        "fit_status": "deferred",
        "model_role": "research_shadow_only",
        "production_ready": False,
        "promotion_allowed": False,
        "gain_model_enabled": False,
        "f_model_version": candidate["model_version"],
        "f_shadow_sha256": None,
        "f_shadow_sha256_scope": "candidate_not_materialized; bound to selection audit record",
        "f_shadow_candidate": candidate,
        "f_selection_audit_sha256": audit_digest,
        "registration_id": registration.get("registration_id"),
        "training_data_sha256": sha256(args.output / "training-data.jsonl"),
        "feature_manifest_sha256": feature_digest,
        "split_manifest_sha256": sha256(args.output / "split-manifest.json"),
        "selection_bias_status": "not_estimable",
        "support_count": 0,
        "uncertainty_status": "UNKNOWN",
        "unknown_policy": "UNKNOWN_OR_DEFER_NOT_ZERO",
        "counterfactual_claim": "not_established",
        "targets": ["expected_gain", "expected_harm", "probability_of_harm", "prediction_interval"],
        "cost_units": "normalized research cost units; no monetary calibration",
    })
    write_json(args.output / "policy-comparison.json", {
        "schema": "footballmind-g-shadow-policy-comparison-v1",
        "status": "deferred",
        "policies": ["stop_immediately", "fixed_order", "random", "source_reliability", "market_divergence", "g_policy", "g_policy_with_stop_gate"],
        "rows": [],
        "reason": "no_eligible_prospective_rows_and_no_independent_test",
    })
    write_json(args.output / "replay-report.json", {
        "schema": "footballmind-g-shadow-replay-report-v1",
        "status": "deferred",
        "g_replay_performed": False,
        "existing_f_g_bridge_exact_replays": 6,
        "existing_bridge_scope": "mechanism_and_f_g_interface_not_accuracy_or_autonomous_VoI_validation",
        "reason": "historical_reconstruction_targets_are_not_training_eligible",
    })
    review = (
        "FootballMind g shadow eligibility review\n\n"
        "status: DEFERRED\n"
        f"g_training_eligible_rows: {summary['eligible_rows']}\n"
        f"bridge physical matches reviewed: {summary['bridge_physical_matches']}\n"
        f"eligible physical matches: {summary['eligible_physical_matches']}\n\n"
        "The bridge preserves fixed-f before/after predictions and hindsight targets for mechanism diagnostics, "
        "but it does not prove original source availability, request/response clocks, or prospective exposure. "
        "Rows explicitly carry prospective_training_eligible=false and reconstructed result-clock assumptions, "
        "so they are excluded from g fitting.\n\n"
        f"Regularization-10 shadow reference: {candidate['model_version']}. "
        f"Selection audit SHA-256: {audit_digest}. The candidate was not materialized or promoted.\n\n"
        "No synthetic labels, counterfactual labels for unselected actions, live HTTP, or formal runtime activation "
        "were used. A future eligible batch needs at least 12 independent physical matches, at least 6 held-out "
        "matches, real request/response/availability clocks, one unchanged shadow-f artifact before and after, "
        "propensity/exposure records, and independently observed outcomes.\n"
    )
    (args.output / "review-report.txt").write_text(review, encoding="utf-8")
    files = []
    for path in sorted(args.output.iterdir()):
        if path.is_file():
            files.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    write_json(args.output / "delivery-receipt.json", {
        "schema": "footballmind-g-shadow-delivery-receipt-v1",
        "status": "deferred",
        "created_at": now,
        "files": files,
        "g_training_performed": False,
        "http_requests": 0,
        "formal_runtime_enabled": False,
    })
    print(json.dumps({
        "output": str(args.output),
        "status": "deferred",
        "g_training_eligible_rows": summary["eligible_rows"],
        "physical_matches_reviewed": summary["bridge_physical_matches"],
        "eligible_physical_matches": summary["eligible_physical_matches"],
        "failure_reasons": summary["failure_reasons"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
