"""Materialize an explicit semantic VoI case from a match-only bundle.

This command is intentionally strict.  ``prepare_footballmind_match.py``
creates the actor-visible unknown catalogue, but it does not invent a model
prior or source access.  This materializer joins that preparation with a
caller-supplied, fixture-bound source profile and structured baseline so the
semantic runtime can run without borrowing another match's evidence.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from math import isfinite
from datetime import datetime, timezone
from pathlib import Path
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from football_agent.autonomous_match_contract import VERSION as REQUEST_VERSION


_FORBIDDEN = {
    "result", "outcome", "score", "scores", "home_score", "away_score",
    "final_score", "actual_result", "future_result", "winner", "label",
    "realized_loss", "decision_loss", "settlement", "post_match",
}


def _scan_forbidden(value, path="input"):
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).casefold() in _FORBIDDEN:
                raise ValueError(f"{path}.{key} is outcome or evaluator data")
            _scan_forbidden(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _scan_forbidden(child, f"{path}[{index}]")


def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing input file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}: {exc.msg}") from exc


def _probabilities(value):
    raw = value.get("probabilities") if isinstance(value, dict) else value
    if not isinstance(raw, list) or len(raw) != 3:
        raise ValueError("baseline must contain exactly three probabilities [home, draw, away]")
    probabilities = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not isfinite(item) or item <= 0:
            raise ValueError("baseline probabilities must be finite positive numbers")
        probabilities.append(float(item))
    total = sum(probabilities)
    if abs(total - 1.0) > 1e-8:
        raise ValueError("baseline probabilities must sum to 1")
    return probabilities


def _utc_time(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"baseline must contain {field}")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"baseline {field} must be an ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"baseline {field} must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _profile(profile, request, bootstrap_override=None):
    if not isinstance(profile, dict):
        raise ValueError("source profile must be a JSON object")
    if profile.get("status") not in {"registered", "verified", "verified_fixture_with_available_public_entries"}:
        raise ValueError("source profile is not an admitted fixture-bound capability profile")
    teams = profile.get("teams")
    if not isinstance(teams, list) or len(teams) != 2:
        raise ValueError("source profile must declare exactly two teams")
    by_side = {item.get("side"): item for item in teams if isinstance(item, dict)}
    if set(by_side) != {"home", "away"}:
        raise ValueError("source profile teams must declare home and away")
    match = request["match"]
    for side in ("home", "away"):
        item = by_side[side]
        names = {str(item.get("canonical_name") or "").strip().casefold(),
                 str(item.get("input_name") or "").strip().casefold()}
        if match[side].strip().casefold() not in names:
            raise ValueError(f"source profile {side} team does not match match bundle")
    seeds = profile.get("seed_urls")
    hosts = profile.get("allowed_hosts")
    if not isinstance(seeds, list) or not seeds or not all(isinstance(url, str) and url for url in seeds):
        raise ValueError("source profile must contain nonempty seed_urls")
    if not isinstance(hosts, list) or not hosts or not all(isinstance(host, str) and host for host in hosts):
        raise ValueError("source profile must contain nonempty allowed_hosts")
    bootstrap = bootstrap_override or profile.get("bootstrap_url")
    if not isinstance(bootstrap, str) or not bootstrap:
        raise ValueError("source profile must declare bootstrap_url or pass --bootstrap-url")
    if bootstrap not in seeds:
        raise ValueError("bootstrap_url must be present in seed_urls")
    if urlsplit(bootstrap).hostname not in set(hosts):
        raise ValueError("bootstrap_url host is outside allowed_hosts")
    result = deepcopy(profile)
    result["bootstrap_url"] = bootstrap
    result["teams"] = deepcopy(teams)
    return result


def materialize(bundle, profile, baseline, *, bootstrap_url=None):
    if not isinstance(bundle, dict) or bundle.get("request", {}).get("schema") != REQUEST_VERSION:
        raise ValueError("match bundle must contain an autonomous-match-request-v1 request")
    request = deepcopy(bundle["request"])
    state = bundle.get("state", {})
    if isinstance(state, dict) and state.get("target_key") not in (None, request.get("target_key")):
        raise ValueError("match bundle request and initial state target_key differ")
    if request.get("mode") != "prematch":
        raise ValueError("semantic case materialization currently requires prematch mode")
    match = request.get("match", {})
    if not match.get("kickoff") or not match.get("prediction_cutoff"):
        raise ValueError("semantic case requires resolved kickoff and prediction_cutoff")
    _scan_forbidden(profile)
    _scan_forbidden(baseline)
    source_profile = _profile(profile, request, bootstrap_override=bootstrap_url)
    if not isinstance(baseline, dict):
        raise ValueError("structured baseline must be an object with provenance fields")
    probabilities = _probabilities(baseline)
    baseline_record = deepcopy(baseline)
    if not baseline_record.get("source") and not baseline_record.get("model_version"):
        raise ValueError("structured baseline must declare source or model_version")
    if not match.get("match_id") or baseline_record.get("match_id") != match["match_id"]:
        raise ValueError("structured baseline match_id does not match match bundle")
    if _utc_time(baseline_record.get("prediction_cutoff"), "prediction_cutoff") != match["prediction_cutoff"]:
        raise ValueError("structured baseline prediction_cutoff does not match match bundle")
    baseline_record["probabilities"] = probabilities
    record_match = {
        "match_id": match.get("match_id"),
        "match": f"{match['home']} vs {match['away']}",
        "home": match["home"], "away": match["away"],
        "competition": match.get("competition"),
        "kickoff": match["kickoff"],
        "prediction_cutoff": match["prediction_cutoff"],
        "target_key": request["target_key"],
    }
    return {
        "schema": "footballmind-semantic-case-v2",
        "match": record_match,
        "source_profile": source_profile,
        "autonomous_request": request,
        "autonomous_initial_state": deepcopy(state),
        "structured_baseline": baseline_record,
        "baseline_scope": baseline_record.get("scope", "explicit_fixture_bound_structured_baseline"),
        "text_model_prior": None,
        "text_prior_reason": "semantic VoI runtime consumes the explicit structured baseline; text value remains uncalibrated",
        "information_value": None,
        "numeric_open_text_voi_ready": False,
        "case_outcome_loaded": False,
        "network_requests": 0,
        "source_snapshot_age_disclosed": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="bundle.json from prepare_footballmind_match.py")
    parser.add_argument("--source-profile", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True,
                        help="JSON object or list with explicit [home, draw, away] probabilities")
    parser.add_argument("--bootstrap-url")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        case = materialize(_load(args.bundle), _load(args.source_profile), _load(args.baseline),
                           bootstrap_url=args.bootstrap_url)
    except ValueError as exc:
        parser.error(f"case_not_ready: {exc}")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)
    for name, value in (("case.json", case), ("request.json", case["autonomous_request"]),
                        ("initial-state.json", case["autonomous_initial_state"])):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "materialization.json").write_text(json.dumps({
        "schema": "footballmind-case-materialization-v1", "status": "ready",
        "target_key": case["autonomous_request"]["target_key"],
        "baseline_source": (case["structured_baseline"].get("source") or
                            case["structured_baseline"].get("model_version")),
        "bootstrap_url": case["source_profile"]["bootstrap_url"],
        "claims_boundary": "explicit case binding only; no empirical g calibration or outcome validation",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ready", "output": str(output),
                      "target_key": case["autonomous_request"]["target_key"],
                      "bootstrap_url": case["source_profile"]["bootstrap_url"],
                      "baseline_probabilities": case["structured_baseline"]["probabilities"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
