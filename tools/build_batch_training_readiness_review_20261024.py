"""Build a read-only readiness review for the next f/g batch-training phase.

The project already contains several research-only artifacts that are easy to
mistake for a formal training set.  This review joins their *metadata only*
and makes the gate explicit.  It never calls the network, a provider, f, g, or
the runtime, and it never fills a missing action id, propensity, clock, menu,
or label by inference.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "var/repairs/batch-training-readiness-review-20261024-v3"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _source(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.exists(), "sha256": _sha256(path) if path.exists() else None}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_review(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)

    label_audit_path = ROOT / "var/work/fm-g-f-bound-labels-v1-20261024/g-label-audit.json"
    label_exclusion_path = ROOT / "var/work/fm-g-f-bound-labels-v1-20261024/exclusion-report.json"
    label_split_path = ROOT / "var/work/fm-g-f-bound-labels-v1-20261024/split-overlap.json"
    diagnostic_report_path = ROOT / "var/work/fm-g-batch-diagnostic-training-20261024-v3/report.json"
    diagnostic_model_path = ROOT / "var/work/fm-g-batch-diagnostic-training-20261024-v3/model.json"
    diagnostic_predictions_path = ROOT / "var/work/fm-g-batch-diagnostic-training-20261024-v3/predictions.jsonl"
    dataset_summary_path = ROOT / "var/work/fm-postread-g-dataset-v1-20261021/dataset-summary.json"
    bridge_report_path = ROOT / "var/work/fm-postread-g-label-bridge-20261021-v2/bridge-report.json"
    f_v2_validation_path = ROOT / "var/work/fm-f-v2-comparison-20261021/validation.json"
    independent_acceptance_path = ROOT / "var/repairs/independent-match-acceptance-20261024-v1/acceptance-validation.json"
    completion_path = ROOT / "var/repairs/fm-batch-data-completion-20261024-v2/completion-report.json"
    acceptance_v2_path = ROOT / "var/repairs/batch-training-acceptance-20261024-v2/acceptance.json"
    plan_path = ROOT / "var/submission/fm-g-empirical-b-collection-plan-20261010/collection-plan.json"
    plan_invariants_path = ROOT / "var/submission/fm-g-empirical-b-collection-plan-20261010/plan-invariants.json"

    label_audit = _read_json(label_audit_path)
    label_exclusion = _read_json(label_exclusion_path)
    label_split = _read_json(label_split_path)
    diagnostic = _read_json(diagnostic_report_path)
    dataset_summary = _read_json(dataset_summary_path)
    bridge = _read_json(bridge_report_path)
    f_v2_validation = _read_json(f_v2_validation_path)
    independent = _read_json(independent_acceptance_path)
    completion = _read_json(completion_path)
    acceptance_v2 = _read_json(acceptance_v2_path)
    plan = _read_json(plan_path) if plan_path.exists() else {}
    plan_invariants = _read_json(plan_invariants_path) if plan_invariants_path.exists() else {}
    diagnostic_rows = _read_jsonl(diagnostic_predictions_path)

    sources = [
        label_audit_path,
        label_exclusion_path,
        label_split_path,
        diagnostic_report_path,
        diagnostic_model_path,
        diagnostic_predictions_path,
        dataset_summary_path,
        bridge_report_path,
        f_v2_validation_path,
        independent_acceptance_path,
        completion_path,
        acceptance_v2_path,
        plan_path,
        plan_invariants_path,
    ]

    menu_rows = int(acceptance_v2.get("complete_original_menu_rows", 0))
    propensity_rows = int(acceptance_v2.get("propensity_rows", 0))
    action_rows = int(acceptance_v2.get("exact_action_id_rows", 0))
    availability_rows = int(acceptance_v2.get("source_available_at_known_rows", 0))
    formal_rows = int(acceptance_v2.get("formal_g_training_eligible_rows", 0))
    fixed_rows = int(acceptance_v2.get("fixed_f_before_after_outcome_rows", 0))
    completed_rows = int(acceptance_v2.get("rows_completed", 0))
    physical_matches = int(acceptance_v2.get("physical_matches", 0))

    blockers = [
        {
            "id": "formal_g_rows_missing",
            "severity": "hard",
            "status": "BLOCKED" if formal_rows == 0 else "PASS",
            "observed": formal_rows,
            "required": ">=1 formally eligible row before fitting; >=12 independent matches for the frozen collection plan",
            "evidence": str(acceptance_v2_path),
        },
        {
            "id": "complete_original_menu",
            "severity": "hard",
            "status": "BLOCKED" if menu_rows == 0 else "PASS",
            "observed": menu_rows,
            "required": "the full pre-query menu for every state, including selected and unselected actions",
            "evidence": str(acceptance_v2_path),
        },
        {
            "id": "action_id",
            "severity": "hard",
            "status": "BLOCKED" if action_rows < completed_rows else "PASS",
            "observed": action_rows,
            "required": "one exact action_id for every candidate exposure and selection",
            "evidence": str(acceptance_v2_path),
        },
        {
            "id": "propensity",
            "severity": "hard",
            "status": "BLOCKED" if propensity_rows < completed_rows else "PASS",
            "observed": propensity_rows,
            "required": "logged positive selection propensity for every candidate in the frozen menu",
            "evidence": str(acceptance_v2_path),
        },
        {
            "id": "source_availability_clock",
            "severity": "hard",
            "status": "BLOCKED" if availability_rows < completed_rows else "PASS",
            "observed": availability_rows,
            "required": "pre-cutoff source_available_at plus request/save/handoff clocks",
            "evidence": str(acceptance_v2_path),
        },
        {
            "id": "independent_outcome_linkage",
            "severity": "hard",
            "status": "PASS" if fixed_rows == completed_rows and fixed_rows > 0 else ("PARTIAL" if fixed_rows > 0 else "BLOCKED"),
            "observed": fixed_rows,
            "required": "same trajectory contract must link before, after, independent outcome and gain for every row admitted to formal fitting",
            "evidence": str(acceptance_v2_path),
        },
        {
            "id": "f_v2_independent_validation",
            "severity": "hard_for_f_v2_promotion",
            "status": "BLOCKED" if f_v2_validation.get("status") != "promoted" else "PASS",
            "observed": f_v2_validation.get("status"),
            "required": "independent prospective or time-valid held-out matches beyond the current four settled policy matches",
            "evidence": str(f_v2_validation_path),
        },
        {
            "id": "formal_runtime_promotion",
            "severity": "hard_for_production",
            "status": "BLOCKED",
            "observed": {"selection_allowed": diagnostic.get("selection_allowed"), "production_ready": diagnostic.get("production_ready")},
            "required": "promotion is a later gate; research fitting must not silently enable runtime",
            "evidence": str(diagnostic_report_path),
        },
    ]

    review = {
        "schema": "footballmind-batch-training-readiness-review-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "BLOCKED_BEFORE_FORMAL_BATCH_TRAINING",
        "answer_to_only_independent_data": False,
        "interpretation": (
            "The training code and diagnostic loaders exist, but the formal g contract is not present. "
            "Independent prospective matches are the largest missing input, and the required action/menu/clock "
            "fields are part of that data contract. f-v2 also lacks independent validation."
        ),
        "what_is_ready": {
            "diagnostic_g_fit": {
                "rows": diagnostic.get("rows"),
                "independent_matches": diagnostic.get("independent_matches"),
                "feature_shape": diagnostic.get("feature_shape"),
                "selection_allowed": diagnostic.get("selection_allowed"),
                "production_ready": diagnostic.get("production_ready"),
            },
            "fixed_f_label_bridge": {
                "rows": label_audit.get("rows"),
                "settled_label_rows": label_audit.get("settled_label_rows"),
                "independent_matches": label_audit.get("independent_matches"),
                "gain_sign_counts": label_audit.get("gain_sign_counts"),
                "group_split": label_split.get("status"),
            },
            "historical_postread_dataset": dataset_summary,
            "f_v2": {
                "status": f_v2_validation.get("status"),
                "formal_runtime_integration": f_v2_validation.get("formal_runtime_integration"),
                "group_splits": f_v2_validation.get("group_splits"),
            },
            "corpus": {
                "articles": acceptance_v2.get("corpus", {}).get("articles"),
                "full_text_marked": acceptance_v2.get("corpus", {}).get("full_text_marked"),
                "verified_links": acceptance_v2.get("corpus", {}).get("verified_links"),
                "warning": "corpus size is not supervised g data; automatic links are not gold labels",
            },
        },
        "blocking_matrix": blockers,
        "counts_from_current_artifacts": {
            "completed_metadata_rows": completed_rows,
            "physical_matches_in_completion": physical_matches,
            "fixed_f_before_after_outcome_rows": fixed_rows,
            "formal_g_training_eligible_rows": formal_rows,
            "exact_action_id_rows": action_rows,
            "propensity_rows": propensity_rows,
            "complete_original_menu_rows": menu_rows,
            "source_available_at_known_rows": availability_rows,
            "diagnostic_prediction_rows": len(diagnostic_rows),
            "diagnostic_gain_sign_counts": diagnostic.get("gain_class_counts"),
            "independent_catalog_records": independent.get("interpretation", {}).get("identity_records_registered"),
            "selected_catalog_fixtures": independent.get("interpretation", {}).get("agent_selected_fixtures"),
            "empirical_g_rows_in_inventory": independent.get("interpretation", {}).get("empirical_g_training_rows"),
        },
        "plan_reference": {
            "collection_plan": _source(plan_path),
            "plan_invariants": _source(plan_invariants_path),
            "plan_status": plan.get("status") or plan.get("state") or "UNKNOWN",
            "plan_invariants_status": plan_invariants.get("status") or plan_invariants.get("pass"),
        },
        "required_next_phase": {
            "phase": "prospective_full_contract_collection_then_formal_batch_fit",
            "before_training": [
                "freeze an independent physical_match_id inventory and at least two decision dates",
                "record the complete pre-query menu before every action",
                "record exact action_id and positive propensity for selected and unselected candidates",
                "record source_available_at and request/save/receipt/handoff clocks before cutoff",
                "run fixed f-v1 to write prediction_before and prediction_after with used_input_ids",
                "settle an independent outcome and compute loss_before, loss_after, and realized_gain",
                "split by physical_match_id before fitting; hold out matches, not rows",
            ],
            "fit_order": [
                "constant/zero baseline",
                "state-only and full-body-plus-state g",
                "fixed f-v1 label evaluation",
                "f-v2 only after an independent f-v2 validation set",
            ],
            "promotion_rule": "keep empirical selection UNKNOWN and production runtime disabled until the formal gates pass",
        },
        "source_artifacts": [_source(path) for path in sources],
        "network_requests": 0,
        "provider_calls": 0,
        "training_started_by_this_review": False,
        "formal_runtime_enabled": False,
    }

    _write_json(output / "readiness-review.json", review)
    with (output / "blocking-matrix.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "severity", "status", "observed", "required", "evidence"])
        writer.writeheader()
        for row in blockers:
            writer.writerow({key: json.dumps(row[key], ensure_ascii=False) if isinstance(row[key], (dict, list)) else row[key] for key in writer.fieldnames})
    (output / "README.md").write_text(
        """# Batch training readiness review\n\n"
        "This is a read-only gate over existing local artifacts. It does not train, call a provider, call the network, or enable runtime.\n\n"
        "The code path for diagnostic f/g fitting exists. Formal empirical g is still blocked because the historical bridge lacks the original pre-query menu, exact action IDs, logged propensities and mostly lacks pre-cutoff availability clocks. f-v2 remains a research prototype without independent prospective validation.\n\n"
        "The next collection must preserve the full candidate state/action contract. Existing historical rows remain diagnostic and are not silently promoted.\n""",
        encoding="utf-8",
    )
    files = []
    for path in sorted(output.iterdir()):
        if path.name == "manifest.json":
            continue
        files.append({"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)})
    _write_json(output / "manifest.json", {"schema": "footballmind-batch-training-readiness-review-manifest-v1", "files": files})
    return review


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    review = build_review(args.output)
    print(json.dumps({
        "status": review["status"],
        "answer_to_only_independent_data": review["answer_to_only_independent_data"],
        "formal_g_training_eligible_rows": review["counts_from_current_artifacts"]["formal_g_training_eligible_rows"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
