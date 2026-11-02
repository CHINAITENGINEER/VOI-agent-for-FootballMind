"""Evaluate the validation-prespecified loss-plus-cost objective, without tuning."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .experiment_round4 import _sem_time, write_json
from .experiment_round6 import _hash, _jsonl


def analyze(directory, output):
    root = Path(directory)
    design = json.loads((root/"design.json").read_text(encoding="utf-8"))
    rows = list(_jsonl(root/"episodes.jsonl"))
    weight = design["validation_utility_money_weight"]
    comparisons, means = {}, {}
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row["strategy"], row["budget_money"])][row["match_id"]] = row
    for (strategy, budget), subset in grouped.items():
        means[f"{strategy}/money={budget:g}"] = float(np.mean([r["log_loss"]+weight*r["cost"]["money"] for r in subset.values()]))
    for (strategy, budget), current in grouped.items():
        if strategy not in {"contextual-visible-context", "contextual-source-identity"}:
            continue
        for (reference, other_budget), previous in grouped.items():
            if budget != other_budget or strategy == reference:
                continue
            if current.keys() != previous.keys():
                raise ValueError("Paired match sets differ")
            blocks = defaultdict(list)
            for mid, row in current.items():
                before = previous[mid]
                year, week, _ = _sem_time(row["kickoff"]).isocalendar()
                delta = row["log_loss"]-before["log_loss"]+weight*(row["cost"]["money"]-before["cost"]["money"])
                blocks[(year, week)].append(delta)
            sums, counts = np.asarray([sum(v) for v in blocks.values()]), np.asarray([len(v) for v in blocks.values()])
            draws = np.random.default_rng(design["bootstrap_seed"]).integers(0, len(sums), size=(design["bootstrap_samples"], len(sums)))
            boot = sums[draws].sum(axis=1)/counts[draws].sum(axis=1)
            comparisons[f"{strategy}-vs-{reference}/money={budget:g}"] = {
                "mean_utility_loss_difference": float(sums.sum()/counts.sum()),
                "confidence_interval_95": np.quantile(boot, [.025, .975]).tolist(),
                "matches": int(counts.sum()), "weeks": len(blocks)}
    result = {"objective": "log_loss + frozen_validation_weight * simulated_money", "money_weight": weight,
        "means": means, "paired_comparisons": comparisons, "episode_sha256": _hash(root/"episodes.jsonl"),
        "method": "paired UTC week percentile bootstrap, each match once",
        "scope": "descriptive evaluation under prespecified exchange rate; no test selection, independent confirmation or noninferiority claim"}
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    write_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    analyze(args.run, args.output)
