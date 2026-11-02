"""Read-only reproduction checks for frozen acquisition experiment artifacts.

Usage: python -m football_research.acquisition.verify --run RUN --previous OLD
       --output var/parallel/baselines/round3/final-verification.json
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path

from .data import load_episodes
from .partial import PartialPredictor
from .policy import select_or_stop
from .state import parse_time
from .value import ValueEstimator
from football_research.evaluation.splits import rolling_origin_splits

TIMING_FIELDS = {"predictor_latency_ms", "estimator_latency_ms"}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _require(condition, message):
    if not condition:
        raise ValueError("Verification failed: " + message)


def _episode_key(row):
    return tuple(row[name] for name in ("fold", "match_id", "strategy", "budget_money", "seed"))


def _semantic_episode(row):
    return {k: v for k, v in row.items() if k not in TIMING_FIELDS}


def verify(run_dir, previous_dir=None):
    run = Path(run_dir).resolve()
    root = Path(__file__).resolve().parents[2]
    report, design = _read(run / "report.json"), _read(run / "design.json")
    implementation = _read(run / "implementation.json")
    hashes = {}
    for filename, expected in implementation.items():
        _require(Path(filename).name == filename, "implementation file name")
        actual = sha256((Path(__file__).parent / filename).read_bytes()).hexdigest()
        _require(actual == expected, f"implementation SHA {filename}")
        hashes[filename] = actual
    audit = _read(run / "data-audit.json")
    rows, current_audit = load_episodes(audit["dataset_audit"]["export_directory"], lead_hours=design["lead_hours"])
    _require(current_audit["platform_manifest_sha256"] == audit["platform_manifest_sha256"], "platform export hash")
    by_id = {r["match_id"]: r for r in rows}
    folds = rolling_origin_splits(rows, initial_validation_start=design["initial_validation_start"],
        validation_window=timedelta(days=design["validation_days"]), test_window=timedelta(days=design["test_days"]))
    models, gain_models, partition_results = {}, {}, []
    saved_folds = {item["fold"]: item for item in report["folds"]}
    test_ids_seen = set()
    provenance_checks = 0
    for saved in report["folds"]:
        fid = saved["fold"]
        fold = folds[fid]
        partitions = {name: set(saved[key]) for name, key in (("train", "training_match_ids"),
                       ("validation", "validation_match_ids"), ("test", "test_match_ids"))}
        for name, ids in partitions.items():
            _require(ids == {r["match_id"] for r in getattr(fold, name)}, f"fold {fid} {name} IDs")
        _require(not (partitions["train"] & partitions["validation"] or partitions["train"] & partitions["test"] or partitions["validation"] & partitions["test"]), f"fold {fid} partition overlap")
        _require(not test_ids_seen & partitions["test"], "test match repeated across folds")
        test_ids_seen |= partitions["test"]
        predictor = PartialPredictor.load(run / "models" / f"fold-{fid}-partial.json")
        estimator = ValueEstimator.load(run / "models" / f"fold-{fid}-gain.json")
        _require(predictor.model_version == saved["model_version"], "partial version")
        _require(estimator.version == saved["gain_version"], "gain version")
        _require(parse_time(predictor.metadata["training_as_of"]) == fold.train_cutoff, "partial cutoff")
        _require(parse_time(estimator.manifest["trained_as_of"]) == fold.train_cutoff, "gain cutoff")
        for subset in predictor.metadata["subsets"].values():
            _require({r["match_id"] for r in subset["training"]} == partitions["train"], "subset training IDs")
            for item in subset["training"]:
                _require(parse_time(item["result_available_at"]) < fold.train_cutoff, "subset label availability")
        gain_ids = set(saved["gain_training_ids"])
        _require(gain_ids == set(estimator.manifest["training_match_ids"]) and gain_ids <= partitions["train"], "gain training IDs")
        for item in estimator.manifest["oof_provenance"]:
            mid = item["match_id"]
            _require(mid in gain_ids and mid not in item["oof_train_ids"], "gain OOF target membership")
            _require(set(item["oof_train_ids"]) <= partitions["train"], "gain OOF parent uses outer future IDs")
            _require(parse_time(item["oof_train_cutoff"]) < parse_time(item["prediction_cutoff"]) < parse_time(by_id[mid]["kickoff"]), "gain forward OOF cutoff")
            _require(parse_time(item["result_available_at"]) <= fold.train_cutoff, "gain label cutoff")
            provenance_checks += 1
        for selected in saved["lambda_selection"].values():
            _require(set(selected["validation_match_ids"]) == partitions["validation"], "lambda validation IDs")
            candidates = [v for v in selected["candidates"] if v["validation_utility"] is not None]
            expected = min(candidates, key=lambda v: (v["validation_utility"], -v["lambda_money"]))["lambda_money"] if candidates else max(design["lambda_money_grid"])
            _require(selected["selected"] == expected, "lambda selection rule")
        models[fid], gain_models[fid] = predictor, estimator
        partition_results.append({"fold": fid, **{k: len(v) for k, v in partitions.items()},
                                  "gain_training_matches": len(gain_ids)})
    # Verify each OOF artifact, its source hash and every parent's historical IDs.
    oof = _read(run / "oof-manifest.json")
    oof_hashes = {}
    for chunk in oof["folds"]:
        path = Path(chunk["artifact"])
        if not path.is_absolute():
            path = root / path
        _require(path.resolve().is_relative_to(run / "oof-models"), "OOF artifact outside frozen run")
        predictor = PartialPredictor.load(path)
        cutoff = parse_time(chunk["cutoff"])
        ids = set(chunk["training_match_ids"])
        _require(not ids & set(chunk["prediction_match_ids"]), "OOF train/target overlap")
        _require(ids == {r["match_id"] for r in predictor.metadata["subsets"]["0"]["training"]}, "OOF artifact training IDs")
        _require(all(parse_time(by_id[mid]["result_available_at"]) <= cutoff for mid in ids), "OOF parent future label")
        _require(all(cutoff < parse_time(by_id[mid]["prediction_cutoff"]) for mid in chunk["prediction_match_ids"]), "OOF cutoff not before target")
        oof_hashes["sha256:" + sha256(path.read_bytes()).hexdigest()] = ids
    supervision_count = 0
    for sample in _rows(run / "oof-supervision.jsonl"):
        _require(sample["oof_manifest_ref"] in oof_hashes, "unknown OOF artifact reference")
        _require(set(sample["oof_train_ids"]) == oof_hashes[sample["oof_manifest_ref"]], "OOF supervision training IDs")
        _require(sample["match_id"] not in sample["oof_train_ids"], "OOF supervision target leak")
        supervision_count += 1
    _require(supervision_count == oof["samples"], "OOF supervision count")
    # Replay all learned/conservative decision events from serialized visible
    # state. No full rows or labels are passed to the loaded predictor/selector.
    catalog = _read(run / "catalog.json")["actions"]
    replay_count, event_count, action_count, by_fold = 0, 0, 0, Counter()
    for trajectory in _rows(run / "trajectories.jsonl"):
        if trajectory["strategy"] not in {"learned-net-gain", "conservative-net-gain"}:
            continue
        fid = trajectory["fold"]
        _require(trajectory["match_id"] in set(saved_folds[fid]["test_match_ids"]), "trajectory target outside test")
        spent = {unit: 0.0 for unit in ("money", "requests", "tokens", "latency_ms")}
        for event in trajectory["events"]:
            state = event["visible_state"]
            _require(len(state.get("action_history", [])) == event["step"], f"trajectory history contains future actions at fold {fid}, step {event['step']}")
            prediction = models[fid].predict_partial(state)
            _require(prediction == event["prediction"], f"loaded probability mismatch fold {fid}")
            history = {h["action_id"] for h in state.get("action_history", [])}
            candidates = [a for a in catalog if a["available"] and a["action_id"] not in history and
                          any(state["groups"][g]["status"] == "not_acquired" for g in a["group_ids"])]
            estimates = gain_models[fid].estimate(state, candidates, prediction=prediction)
            budget = {"money": max(0, trajectory["budget_money"] - spent["money"]),
                      "requests": max(0, design["request_budget"] - spent["requests"])}
            old = event["decision"]
            decision = select_or_stop(state, estimates, budget, old["lambda_cost"],
                                     conservative=trajectory["strategy"] == "conservative-net-gain", min_support=design["gain_min_support"])
            _require(decision == old, f"loaded decision mismatch fold {fid}")
            if "charged_cost" in event:
                selected = next(a for a in candidates if a["action_id"] == decision["action_id"])
                _require(event["charged_cost"] == selected["cost"], "charged cost differs from frozen catalog")
                for unit in spent:
                    spent[unit] += event["charged_cost"][unit]
                action_count += 1
            event_count += 1
        replay_count += 1
        by_fold[fid] += 1
    _require(set(by_fold) == set(models), "not every model exercised")
    episodes = list(_rows(run / "episodes.jsonl"))
    keys = {_episode_key(r) for r in episodes}
    _require(len(keys) == len(episodes), "duplicate episode identities")
    _require(len(episodes) == report["episode_runs"], "episode report count")
    _require({r["match_id"] for r in episodes} == test_ids_seen, "episode test ID coverage")
    for item in episodes:
        probs = item["final_probabilities"]
        _require(len(probs) == 3 and all(0 <= p <= 1 for p in probs) and abs(sum(probs) - 1) < 1e-12, "episode probability normalization")
        _require(item["cost"]["money"] <= item["budget_money"] + 1e-12 and item["cost"]["requests"] <= design["request_budget"], "episode budget exceeded")
    comparisons = []
    previous_runs = [] if previous_dir is None else [previous_dir] if isinstance(previous_dir, (str, Path)) else list(previous_dir)
    for previous in previous_runs:
        old = {_episode_key(r): _semantic_episode(r) for r in _rows(Path(previous) / "episodes.jsonl")}
        current = {_episode_key(r): _semantic_episode(r) for r in episodes}
        _require(old.keys() == current.keys(), "previous/current episode identities")
        differences = [str(key) for key in current if current[key] != old[key]]
        _require(not differences, "non-timing episodes changed: " + ", ".join(differences[:3]))
        comparisons.append({"previous": str(Path(previous).resolve()), "identical_non_timing_episodes": len(current),
                            "excluded_fields": sorted(TIMING_FIELDS)})
    return {"status": "PASS", "verified_at": datetime.now(timezone.utc).isoformat(), "run_directory": str(run),
            "implementation_sha256": hashes, "verifier_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "platform_manifest_sha256": audit["platform_manifest_sha256"],
            "loaded_outer_models": len(models) + len(gain_models), "outer_folds": partition_results,
            "loaded_oof_predictors": len(oof_hashes), "oof_samples_checked": supervision_count,
            "outer_gain_provenance_records_checked": provenance_checks,
            "replayed_learned_trajectories": replay_count, "replayed_decision_events": event_count,
            "replayed_acquisition_actions": action_count, "replay_counts_by_fold": dict(by_fold),
            "episode_probability_and_budget_checks": len(episodes), "independent_test_matches": len(test_ids_seen),
            "previous_run_comparisons": comparisons,
            "scope": "Exact saved-model probability and action reproduction plus timing/identity/hash checks; no model retraining or new effectiveness evidence."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--previous", action="append", help="Repeat to compare several earlier frozen runs")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = verify(args.run, args.previous)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
