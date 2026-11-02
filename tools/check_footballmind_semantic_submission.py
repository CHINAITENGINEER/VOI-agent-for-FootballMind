"""Offline, read-only acceptance of the semantic AIC submission evidence.

This command checks the frozen historical mechanism trace, its exact math
replay, controls, report projection, and recorded input hashes.  It never
contacts a source or a model and never rewrites the submission artifacts.

Run from the repository:

    python tools/check_footballmind_semantic_submission.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
VERSION = "footballmind-semantic-submission-acceptance-v1"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition, message, failures):
    if not condition:
        failures.append(message)


def run_acceptance(root: Path = ROOT) -> dict:
    root = Path(root).resolve()
    failures = []
    case = root / "var/submission/p7-historical-semantic/mechanism-03"
    replay_path = root / "var/submission/p7-historical-semantic/mechanism-03-replay/replay.json"
    index_path = root / "var/submission/footballmind-aic-evidence-index.md"
    paths = {
        "run.json": case / "run.json",
        "analysis.json": case / "analysis.json",
        "controls.json": case / "controls.json",
        "audit.json": case / "audit.json",
        "math-replay.json": case / "math-replay.json",
        "replay.json": replay_path,
        "report.md": case / "report.md",
        "evidence_index": index_path,
    }
    for name, path in paths.items():
        _require(path.is_file(), f"missing artifact: {name}", failures)
    if failures:
        return {"schema": VERSION, "status": "failed", "failures": failures,
                "case_directory": str(case), "network_calls": 0, "semantic_calls": 0}

    run = _load(paths["run.json"])
    analysis = _load(paths["analysis.json"])
    controls = _load(paths["controls.json"])
    audit = _load(paths["audit.json"])
    saved_replay = _load(paths["replay.json"])
    saved_math_replay = _load(paths["math-replay.json"])

    _require(run.get("status") == "completed", "mechanism run is not completed", failures)
    _require(run.get("configuration", {}).get("transport", {}).get("network_requests") == 0,
             "mechanism run reports network requests", failures)
    _require(run.get("configuration", {}).get("future_outcome_loaded") is False,
             "future outcome is visible in run configuration", failures)
    _require(len(run.get("cycles", [])) == 6, "expected six saved cycles", failures)
    _require(len(run.get("observations", [])) == 5, "expected five saved observations", failures)
    outcomes = [item.get("outcome") for item in run.get("observations", [])]
    _require(outcomes.count("supports") == 1, "expected one supports observation", failures)
    _require(sum(item.get("outcome") == "supports" and item.get("applicability") == "match_specific"
                  for item in run.get("observations", [])) == 1,
             "expected one match-specific support classification", failures)
    _require(outcomes.count("no_answer") == 4, "expected four no_answer observations", failures)

    from football_agent.semantic_scenario_runtime import replay_math

    try:
        computed = replay_math(run)
    except Exception as error:  # pragma: no cover - exercised by tampered artifacts
        failures.append(f"math replay failed: {type(error).__name__}: {error}")
        computed = None
    if computed is not None:
        _require(computed == saved_replay, "saved replay differs from exact math replay", failures)
        _require(computed == saved_math_replay, "mechanism math-replay differs from exact replay", failures)
        _require(computed.get("http_requests") == 0 and computed.get("semantic_calls") == 0,
                 "replay performed a provider or network call", failures)

    _require(controls.get("status") == "completed", "strategy controls are incomplete", failures)
    _require(controls.get("counterfactual_returns_used") == 0, "controls used counterfactual returns", failures)
    _require(controls.get("future_returns_used") == 0, "controls used future returns", failures)
    _require(controls.get("real_match_result_used") is False, "controls used match result", failures)
    _require(controls.get("empirical_gain") is None and controls.get("outcome_accuracy") is None,
             "controls contain empirical gain or accuracy", failures)

    directional = analysis.get("directional_evidence", {})
    _require(directional.get("used_for_voi") is False, "directional evidence entered VoI", failures)
    _require("方向性证据" in paths["report.md"].read_text(encoding="utf-8"),
             "report is missing directional evidence section", failures)
    _require("自主" in paths["evidence_index"].read_text(encoding="utf-8"),
             "AIC evidence index is missing autonomous-selection narrative", failures)

    expected_hashes = audit.get("input_sha256", {})
    for name in ("run.json", "analysis.json", "controls.json", "math-replay.json"):
        _require(expected_hashes.get(name) == _sha256(paths[name]),
                 f"audit hash mismatch: {name}", failures)
    _require(audit.get("status") == "passed", "mechanism audit is not passed", failures)

    return {
        "schema": VERSION,
        "status": "passed" if not failures else "failed",
        "case_directory": str(case),
        "network_calls": 0,
        "semantic_calls": 0,
        "math_replay": computed,
        "cycles": len(run.get("cycles", [])),
        "observations": len(run.get("observations", [])),
        "supports_match_specific": sum(
            item.get("outcome") == "supports" and item.get("applicability") == "match_specific"
            for item in run.get("observations", [])),
        "no_answer": outcomes.count("no_answer"),
        "directional_evidence_used_for_voi": directional.get("used_for_voi"),
        "failures": failures,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    try:
        result = run_acceptance(args.root)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        result = {"schema": VERSION, "status": "failed", "failures": [f"{type(error).__name__}: {error}"],
                  "network_calls": 0, "semantic_calls": 0}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
