"""Build a read-only Stage 3 label audit from existing local bridge artifacts.

This joins the existing post-read candidate dataset and fixed-f bridge. It does
not perform network/provider calls, refit f, invent action IDs/propensities, or
promote historical rows to formal empirical-g training data.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(r"D:/a1-足彩")
DATASET = ROOT / r"var/work/fm-postread-g-dataset-v1-20261021/postread-g-dataset-v1.jsonl"
BRIDGE = ROOT / r"var/work/fm-postread-g-label-bridge-20261021-v2/training-rows.jsonl"
STAGE2 = ROOT / r"var/work/fm-postread-g-dataset-v1-20261021"
OUT = ROOT / r"var/work/fm-g-f-bound-labels-v1-20261024"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def outcome_class(index):
    return {0: "home", 1: "draw", 2: "away"}.get(index)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    dataset_rows = read_jsonl(DATASET)
    bridge_rows = read_jsonl(BRIDGE)
    by_sample = {row["sample_id"]: row for row in dataset_rows}
    by_bridge = {row["sample_id"]: row for row in bridge_rows}
    if len(by_sample) != len(dataset_rows):
        raise ValueError("duplicate Stage 2 sample_id")
    labels = []
    for index, row in enumerate(dataset_rows):
        sid = row["sample_id"]
        bridge = by_bridge.get(sid)
        before = (bridge or {}).get("prediction_before", row["current_state"].get("prediction_before"))
        after = (bridge or {}).get("prediction_after", row.get("post_f_audit", {}).get("prediction_after"))
        outcome = (bridge or {}).get("outcome_index")
        loss_before = (bridge or {}).get("loss_before")
        loss_after = (bridge or {}).get("loss_after")
        gain = (bridge or {}).get("realized_signed_gain")
        reasons = []
        if bridge is None:
            reasons.append("fixed_f_bridge_row_missing")
        if before is None:
            reasons.append("prediction_before_missing")
        if after is None:
            reasons.append("prediction_after_missing")
        if outcome is None:
            reasons.append("independent_outcome_missing")
        if loss_before is None or loss_after is None or gain is None:
            reasons.append("realized_gain_missing")
        # These are deliberately recorded as missing rather than recovered from order.
        reasons.extend(["action_id_missing", "propensity_missing", "complete_prequery_menu_missing"])
        if row["source"].get("source_available_at") is None:
            reasons.append("source_available_at_unknown")
        labels.append({
            "schema": "footballmind-g-f-bound-label-v1",
            "sample_id": sid,
            "physical_match_id": row["physical_match_id"],
            "split": row.get("split"),
            "action_id": None,
            "propensity": None,
            "candidate_text_id": sid,
            "state_before": {
                "prediction_before": before,
                "cutoff": row["current_state"].get("cutoff"),
                "kickoff": row["current_state"].get("kickoff"),
                "prediction_before_entropy": row["current_state"].get("prediction_before_entropy"),
                "prediction_before_max_probability": row["current_state"].get("prediction_before_max_probability"),
                "prediction_before_margin": row["current_state"].get("prediction_before_margin"),
                "existing_evidence_set": row["current_state"].get("existing_evidence_set"),
            },
            "candidate": {
                "body_path": row["body"]["body_path"],
                "body_sha256": row["body"]["body_sha256"],
                "body_vector_ref": {"path": str(DATASET), "jsonl_line": index + 1,
                                    "field": "candidate_body_vector.values",
                                    "vector_schema": row["candidate_body_vector"]["schema"],
                                    "dimension": row["candidate_body_vector"]["dimension"]},
                "source_url": row["source"]["source_url"],
                "source_domain": row["source"]["source_domain"],
            },
            "p_before": before,
            "p_after": after,
            "outcome_class": outcome_class(outcome),
            "outcome_index": outcome,
            "loss_before": loss_before,
            "loss_after": loss_after,
            "realized_gain": gain,
            "gain_formula": "log(p_after[y]) - log(p_before[y])" if gain is not None else None,
            "f_model_version": (bridge or {}).get("f_model_version"),
            "f_model_sha256": (bridge or {}).get("f_model_sha256"),
            "clocks": {
                "source_published_at": row["source"].get("source_published_at"),
                "source_available_at": row["source"].get("source_available_at"),
                "captured_at": row["source"].get("captured_at"),
                "cutoff": row["current_state"].get("cutoff"),
                "kickoff": row["current_state"].get("kickoff"),
            },
            "f_execution": {
                "body_received": row["f_adoption"].get("f_received_body"),
                "body_adopted": row["f_adoption"].get("f_adopted_candidate"),
                "modeled_claim_count": row["post_f_audit"].get("f_modeled_claim_count"),
                "prediction_changed": row["post_f_audit"].get("prediction_changed"),
                "probability_l1_shift": row["post_f_audit"].get("probability_l1_shift"),
            },
            "menu_evidence": {"complete": False, "status": "UNKNOWN",
                              "reason": "archived panel has post-read research pool, not original pre-query menu"},
            "g_input_exclusions": {"prediction_after": True, "outcome": True, "realized_gain": True,
                                   "post_kickoff_information": True},
            "eligibility_status": "diagnostic_label_only" if not reasons or all(
                r in {"action_id_missing", "propensity_missing", "complete_prequery_menu_missing", "source_available_at_unknown"}
                for r in reasons
            ) and gain is not None else "excluded",
            "exclusion_reason": reasons,
            "formal_g_training_eligible": False,
        })

    settled = [row for row in labels if row["realized_gain"] is not None]
    signs = Counter("positive" if row["realized_gain"] > 1e-12 else
                    "negative" if row["realized_gain"] < -1e-12 else "zero" for row in settled)
    groups = {}
    overlap = []
    for row in labels:
        split = row.get("split")
        if split in {"train", "validation", "test"}:
            groups.setdefault(split, set()).add(row["physical_match_id"])
    names = list(groups)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap.extend(sorted(groups[left] & groups[right]))
    audit = {
        "schema": "footballmind-g-f-bound-label-audit-v1",
        "status": "LABELS_MATERIALIZED_DIAGNOSTIC_ONLY",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(DATASET),
        "source_dataset_sha256": sha(DATASET),
        "source_bridge": str(BRIDGE),
        "source_bridge_sha256": sha(BRIDGE),
        "rows": len(labels),
        "settled_label_rows": len(settled),
        "independent_matches": len({r["physical_match_id"] for r in labels}),
        "gain_sign_counts": dict(signs),
        "split_group_overlap": sorted(set(overlap)),
        "split_group_check": "PASS" if not overlap else "FAIL",
        "formal_g_training_eligible_rows": 0,
        "network_requests": 0,
        "provider_calls": 0,
        "f_refit": False,
        "labels_do_not_enter_g_inputs": True,
        "limitations": [
            "action_id and propensity are absent in the historical bridge",
            "complete original pre-query menus are absent",
            "source_available_at is unknown for most historical rows",
            "historical rows are not prospective empirical-g trajectories",
        ],
    }
    write_jsonl(OUT / "g-f-bound-labels-v1.jsonl", labels)
    write_json(OUT / "g-label-audit.json", audit)
    write_json(OUT / "exclusion-report.json", {
        "schema": "footballmind-g-label-exclusion-report-v1",
        "rows": [{"sample_id": row["sample_id"], "physical_match_id": row["physical_match_id"],
                  "status": row["eligibility_status"], "reasons": row["exclusion_reason"]} for row in labels],
        "formal_training_eligible_rows": 0,
    })
    write_json(OUT / "split-overlap.json", {
        "schema": "footballmind-g-label-split-overlap-v1",
        "group_key": "physical_match_id",
        "groups": {key: sorted(value) for key, value in groups.items()},
        "overlap": sorted(set(overlap)),
        "status": "PASS" if not overlap else "FAIL",
    })
    with (OUT / "label-distribution.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["outcome_class", "count"])
        writer.writerows(sorted(signs.items()))
    (OUT / "README.md").write_text(
        """# Stage 3 fixed-f label package

This package joins the existing post-read dataset and fixed-f bridge using only local files.
It preserves positive, zero and negative realized gains and records excluded or missing fields.
Action IDs, propensities and complete original menus remain UNKNOWN; formal empirical-g rows are 0.
The label fields are explicitly excluded from g input construction.
""",
        encoding="utf-8",
    )
    files = {p.name: sha(p) for p in OUT.iterdir() if p.is_file() and p.name != "manifest.json"}
    write_json(OUT / "manifest.json", {"schema": "footballmind-g-f-bound-label-manifest-v1", "files": files})
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
