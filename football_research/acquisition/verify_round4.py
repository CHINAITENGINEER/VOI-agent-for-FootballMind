"""Read-only integrity and visible-state replay checks for round four."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from .partial import PartialPredictor
from .risk import RiskAwareValueEstimator
from .value import ValueEstimator

FORBIDDEN = {"label", "labels", "outcome", "home_goals", "away_goals", "full_snapshot", "input_manifest", "source_path", "archive_path"}


def _rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _walk(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def verify(run_dir: str | Path):
    run = Path(run_dir)
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    design = json.loads((run / "design.json").read_text(encoding="utf-8"))
    implementation = json.loads((run / "implementation.json").read_text(encoding="utf-8"))
    hashes = {}
    for filename, expected in implementation.items():
        path = Path(__file__).parent / filename
        actual = sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"implementation hash mismatch: {filename}")
        hashes[filename] = actual
    model_count = 0
    for path in sorted((run / "models").glob("*.json")):
        if path.name.endswith("-risk.json"):
            RiskAwareValueEstimator.load(path)
        elif path.name.endswith("-v1-gain.json"):
            ValueEstimator.load(path)
        else:
            raise ValueError(f"unexpected model artifact {path.name}")
        model_count += 1
    if model_count != 14:
        raise ValueError(f"expected 14 model artifacts, found {model_count}")
    episodes = list(_rows(run / "episodes.jsonl"))
    if len(episodes) != report["episode_runs"]:
        raise ValueError("episode count mismatch")
    keys = set()
    for episode in episodes:
        key = tuple(episode[name] for name in ("fold", "match_id", "strategy", "budget_money", "seed"))
        if key in keys:
            raise ValueError("duplicate episode identity")
        keys.add(key)
        probabilities = episode["final_probabilities"]
        if len(probabilities) != 3 or any(value < 0 or value > 1 for value in probabilities) or abs(sum(probabilities) - 1) > 1e-12:
            raise ValueError("invalid final probabilities")
    events = actions = 0
    for trajectory in _rows(run / "trajectories.jsonl"):
        for event in trajectory["events"]:
            if any(key in FORBIDDEN for key in _walk(event["visible_state"])):
                raise ValueError("hidden label or source path crossed visible trajectory")
            if len(event["visible_state"].get("action_history", [])) != event["step"]:
                raise ValueError("future action leaked into an earlier event")
            events += 1
            if "charged_cost" in event:
                actions += 1
    result = {"status": "PASS", "run_directory": str(run.resolve()), "implementation_sha256": hashes,
              "loaded_model_artifacts": model_count, "episode_runs": len(episodes),
              "trajectory_events": events, "acquisition_actions": actions,
              "strategies": report["strategies"], "test_matches": report["test_matches"],
              "probability_checks": len(episodes), "visible_state_checks": events,
              "risk_version_prefix": "risk-aware-conditional-value-v1",
              "scope": "Artifact load, hash, normalization, action-history and visible-state checks; no new effectiveness claim."}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = verify(args.run)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

