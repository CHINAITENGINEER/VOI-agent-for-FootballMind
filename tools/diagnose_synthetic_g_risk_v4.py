"""Train-only teacher diagnosis; never refit or change the frozen v4 policy."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football_research.acquisition import synthetic_g_shared_world as teacher
from football_research.acquisition import synthetic_g_observable_v3 as v3


def diagnose(artifact, output):
    artifact, output = Path(artifact), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new diagnosis directory; frozen files are not overwritten")
    source = json.loads((artifact / "teacher-audit-dataset.json").read_text(encoding="utf-8"))
    protocol = json.loads((artifact / "protocol.json").read_text(encoding="utf-8"))
    train_ids = set(protocol["split"]["train"])
    rows = [r for r in source["rows"] if r["physical_match_id"] in train_ids]
    worlds = {w["state_id"]: w for w in source["teacher_only_worlds"]}
    gross, matched_gross, cost, actual, assumed = [], [], [], [], []
    for row in rows:
        f = row["features"]
        effective = .5 + (f["source_reliability"] - .5) * f["freshness"] * (1. - f["redundancy"])
        world = worlds[row["state_id"]]
        # This is a teacher counterfactual, not a calibrated real-world f.
        matched = teacher.score_channel(world["weights"], world["outcome_by_world"],
            teacher.ACTIONS.index(f["action"]), actual_reliability=effective,
            updater_trust=effective, response_probability=f["response_probability"])
        gross.append(row["teacher"]["expected_signed_gain"])
        matched_gross.append(matched["expected_signed_gain"])
        cost.append(row["cost_units"])
        actual.append(effective)
        assumed.append(f["updater_trust"])
    gross, matched_gross, cost, actual, assumed = map(np.asarray, (gross, matched_gross, cost, actual, assumed))
    def summarize(values):
        return {"mean": float(values.mean()), "positive_fraction": float((values > 1e-12).mean()),
                "negative_fraction": float((values < -1e-12).mean()),
                "near_zero_fraction": float((np.abs(values) <= 1e-12).mean()),
                "quantiles_10_50_90_99": np.quantile(values, [.1, .5, .9, .99]).tolist()}
    search = json.loads((artifact / "validation-selection.json").read_text(encoding="utf-8"))
    material = [r for r in search["risk_search"]
                if r["queried_states"] >= protocol["validation_constraint"]["min_queries"]
                and r["queried_matches"] >= protocol["validation_constraint"]["min_queried_matches"]]
    sparse = [r for r in search["risk_search"] if r["queried_states"] > 0]
    report = {
        "schema": "synthetic-g-risk-v4-train-only-mechanism-diagnosis-v1",
        "model_sha256": sha256((artifact / "model.json").read_bytes()).hexdigest(),
        "partition_used_for_counterfactual": "train_only", "matches": len(train_ids), "rows": len(rows),
        "matched_update_meaning": "oracle teacher source correctness substituted into fixed updater; no real calibration claim",
        "cost_zero_meaning": "teacher diagnostic ablation; no change to frozen policy costs",
        "existing_updater_gross": summarize(gross), "existing_updater_net": summarize(gross - cost),
        "oracle_matched_updater_gross": summarize(matched_gross),
        "oracle_matched_updater_net_same_cost": summarize(matched_gross - cost),
        "cost": summarize(cost),
        "source_mismatch": {"mean_effective_source_reliability": float(actual.mean()),
            "mean_updater_assumed_reliability": float(assumed.mean()),
            "updater_overtrust_fraction": float((assumed > actual + 1e-12).mean())},
        "validation_summary": {
            "status": search["status"], "rules_with_sufficient_queries": len(material),
            "rules_with_any_queries": len(sparse),
            "best_observed_conditional_harm_among_sufficient_rules": min(
                [r["negative_expected_net_given_query"] for r in material], default=None),
            "smallest_harm_nonempty_rule": min(sparse, key=lambda r: r["negative_expected_net_given_query"]) if sparse else None,
        },
        "learner_retrained": False, "test_policy_retuned": False, "runtime_modified": False,
        "network_requests": 0,
        "interpretation": [
            "The generator samples updater trust independently of degraded effective source reliability",
            "A negative-heavy teacher distribution is partly an updater/cost mechanism, not just a g capacity problem",
            "Unobserved effect strength remains heterogeneous within observable features",
            "A correct expected-utility policy may accept frequent small negative values for rarer large positive values",
            "The optional 30 percent risk preference is not identical to maximizing expected net value",
            "A future experiment must declare updater/cost regimes before its own holdout is evaluated",
        ],
    }
    if matched_gross.min() < -1e-10:
        raise AssertionError("Matched Bayesian teacher should not have materially negative expected gross information value")
    output.mkdir(parents=True, exist_ok=True)
    (output / "diagnosis.json").write_bytes(v3._canonical(report))
    (output / "source.py").write_bytes(Path(__file__).read_bytes())
    files = {p.name: {"sha256": sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
             for p in output.iterdir() if p.is_file()}
    (output / "manifest.json").write_bytes(v3._canonical({"files": files}))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.artifact, args.output), indent=2))
