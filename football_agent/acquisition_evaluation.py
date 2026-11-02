"""Offline summaries of persisted execution evidence; no tools are invoked."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def summarize_runs(runs: list[dict]) -> dict:
    """Keep simulation, unknown actual cost and model-estimated recall separate."""
    groups = {}
    for run in runs:
        mode = run.get("configuration", {}).get("backend", {}).get("mode", "unknown")
        key = (mode, run.get("strategy", "unknown"), run.get("risk_mode", "mean"))
        groups.setdefault(key, []).append(run)
    result = []
    for (mode, strategy, risk), entries in sorted(groups.items()):
        recalls = {str(k): [] for k in (1, 3, 5)}
        for run in entries:
            for event in run.get("events", []):
                audit = event.get("data", {}).get("proposal_audit", {})
                if event.get("type") != "value_estimated" or not audit.get("recall_applicable"):
                    continue
                for k, value in audit.get("proposal_recall_at_k", {}).items():
                    if k in recalls and value is not None:
                        recalls[k].append(value)
        budgets = [run.get("budget", {}) for run in entries]
        audits = [run.get("proposal_audit", {}) for run in entries]
        items = sum(a.get("proposal_items", 0) for a in audits)
        calls = sum(a.get("calls", 0) for a in audits)
        result.append({"environment_mode": mode, "strategy": strategy, "risk_mode": risk,
            "runs": len(entries), "independent_matches": len({r["match_id"] for r in entries}),
            "stop_reasons": dict(Counter(r.get("reason", "unknown") for r in entries)),
            "mean_simulated_cost": sum(b.get("simulated_cost", 0) for b in budgets)/len(entries),
            "mean_acquisition_actions": sum(b.get("actions", 0) for b in budgets)/len(entries),
            "llm_calls": sum(b.get("llm_calls", 0) for b in budgets),
            "actual_provider_money": None, "actual_provider_money_status": "not_inferred_from_reservations",
            "proposal_legal_parameter_rate": sum(a.get("legal_parameter_items", 0) for a in audits)/items if items else None,
            "duplicate_or_invalid_call_rate": sum(a.get("duplicate_or_invalid_calls", 0) for a in audits)/calls if calls else None,
            "proposal_recall_at_k": {k: sum(v)/len(v) if v else None for k, v in recalls.items()},
            "proposal_recall_reference": "numeric_estimator_positive_actions_not_oracle",
            "stop_regret": None, "stop_regret_status": "matched_offline_continuations_required"})
    return {"schema_version": "acquisition-evaluation-v1", "groups": result,
            "scope": "execution_audit; no forecast quality, online saving or LLM effectiveness claim"}


def offline_stop_regret(current_loss: float, continuations: list[dict], money_weight: float) -> float:
    """Hindsight diagnostic ONLY. Continuations must share the stopped state.

    Inputs are external evaluator scores, never part of the public policy state.
    The minimum includes STOP; selection with known outcomes is an optimistic
    diagnostic bound, not an implementable policy or unbiased expected EVSI.
    """
    numbers = [current_loss, money_weight]
    for row in continuations:
        numbers += [row["loss"], row["incremental_money"]]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in numbers):
        raise ValueError("Loss, money and utility weights must be finite and nonnegative")
    best = min([current_loss] + [r["loss"] + money_weight*r["incremental_money"] for r in continuations])
    return current_loss - best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("store_dir")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from .audit import RunStore
    store = RunStore(args.store_dir)
    runs = []
    for path in sorted(Path(args.store_dir).glob("*/run.json")):
        checked = store.verify(path.parent.name)
        if checked.get("status") != "OK":
            raise ValueError(f"Run audit failed: {path.parent.name}")
        runs.append(store.load(path.parent.name))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summarize_runs(runs), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(str(output))


if __name__ == "__main__":
    main()
