"""Materialize the training/deployment separation for the current g diagnostic.

This is an audit artifact, not a switch for empirical g.  Historical rows are
marked diagnostic because their propensity is unavailable; the artifact makes
that limitation explicit while preserving positive, negative, and zero labels.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from football_agent.g_label_contract import (
    LABEL_CONTRACT_SCHEMA,
    EXPLORATION_POLICY_SCHEMA,
    epsilon_greedy_propensities,
    validate_transition,
)

BASE = ROOT / "var/reviews/fm-g-nonleeds-expansion-20261015-v1"
OLD = ROOT / "var/reviews/fm-historical-transition-g-training-20261014-v2/transitions.json"
NEW = BASE / "g-diagnostic/rows.json"
OUT = ROOT / "var/reviews/fm-g-exploration-contract-20261015-v1"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def gain_class(gain: float) -> str:
    if gain > 1e-12:
        return "positive"
    if gain < -1e-12:
        return "negative"
    return "zero"


def main() -> None:
    old = read_json(OLD)
    new = read_json(NEW)
    rows = []
    for source, source_rows in (("historical_transition", old), ("nonleeds_expansion", new)):
        for i, raw in enumerate(source_rows):
            gain = float(raw.get("signed_gain", raw.get("gain")))
            row = {
                "row_id": f"{source}:{i}",
                "physical_match_id": raw["physical_match_id"],
                "source": source,
                "selected": True,
                "propensity": raw.get("propensity"),
                "predicted_signed_gain": None,
                "realized_signed_gain": gain,
                "label_status": "diagnostic_historical",
                "gain_class": gain_class(gain),
                "has_outcome": True,
                "has_f_before_after": True,
                "prospective_training_eligible": False,
            }
            errors = validate_transition(row)
            if errors:
                raise ValueError(f"{row['row_id']}: {errors}")
            row["validation"] = "PASS"
            rows.append(row)

    classes = Counter(r["gain_class"] for r in rows)
    matches = sorted({r["physical_match_id"] for r in rows})
    # A concrete example contract for a future three-action menu.  It is not
    # connected to the formal runtime and deliberately has no source values.
    example_scores = {"lineup": 0.002, "injury": -0.001, "weather": 0.0001}
    example_propensity = epsilon_greedy_propensities(
        example_scores, example_scores, epsilon=0.2
    )
    report = {
        "schema": LABEL_CONTRACT_SCHEMA,
        "status": "diagnostic_contract_materialized",
        "rows": len(rows),
        "physical_match_groups": len(matches),
        "gain_class_counts": dict(sorted(classes.items())),
        "observed_labels": "executed transitions only",
        "unexecuted_gain_policy": "realized_signed_gain=null",
        "predicted_gain_is_label": False,
        "propensity_available_rows": sum(r["propensity"] is not None for r in rows),
        "prospective_training_eligible_rows": 0,
        "selection_authority": "none",
        "empirical_g_enabled": False,
        "production_ready": False,
        "limitations": [
            "historical rows have no propensity and remain diagnostic",
            "current rows are not independent prospective B trajectories",
            "no unexecuted action is assigned a negative realized gain",
        ],
    }
    policy = {
        "schema": EXPLORATION_POLICY_SCHEMA,
        "phase": "training_collection",
        "selection_authority": "data_collection_only",
        "epsilon": 0.2,
        "propensity_required": True,
        "collect_negative_predictions": True,
        "stop_is_label": False,
        "stop_allowed": False,
        "example_menu": list(example_scores),
        "example_scores": example_scores,
        "example_propensity": example_propensity,
        "deployment_phase_rule": "g_prediction_minus_cost_may_stop",
        "training_phase_rule": "sample_with_positive_propensity_then_wait_for_outcome",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "contract-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "exploration-policy.json").write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "labeled-rows.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    (OUT / "README.md").write_text(
        "# g exploration contract\n\n"
        "This is a diagnostic contract. It preserves observed positive, negative, "
        "and zero gains, keeps unexecuted gains null, and forbids STOP from becoming "
        "a training label. Historical propensity is missing, so prospective eligible "
        "rows remain zero and empirical g stays disabled.\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
