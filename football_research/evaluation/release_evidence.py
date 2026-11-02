"""Offline release evidence: recompute frozen scores without executing any policy.

This module uses only the standard library. It neither fits models nor imports an
Actor, a transport, or an old round runner. Historical labels remain explicitly
same-provider reconstructions; constructed outcomes never become football labels.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any


SCHEMA = "footballmind-release-evidence-v1"
DEFAULT_OUTPUT = "var/parallel/agent/round181/evidence"
BASE = "var/parallel/agent"


class EvidenceError(ValueError):
    """Frozen evidence or a reported claim failed verification."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def same_number(actual: float, expected: float, message: str) -> None:
    require(math.isfinite(actual) and math.isfinite(expected)
            and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10), message)


def mean(values) -> float:
    values = list(values)
    require(bool(values), "An evidence cell must contain at least one match")
    return math.fsum(values) / len(values)


def probability_loss(probabilities: list[float], outcome: int) -> float:
    require(type(outcome) is int and outcome in range(3), "Invalid evaluator outcome")
    require(len(probabilities) == 3 and all(math.isfinite(p) and 0 < p <= 1 for p in probabilities),
            "Invalid final probability vector")
    same_number(math.fsum(probabilities), 1.0, "Probabilities do not sum to one")
    return -math.log(probabilities[outcome])


def _portable(value: str | Path) -> str:
    return str(value).replace("\\", "/")


class Inputs:
    """Read immutable inputs and bind every dependency to repository-relative SHA."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.files: dict[str, dict[str, Any]] = {}

    def path(self, name: str | Path) -> Path:
        path = (self.root / _portable(name)).resolve()
        require(path.is_relative_to(self.root), f"Evidence path escapes repository: {name}")
        require(path.is_file(), f"Missing evidence: {name}")
        return path

    def bind(self, name: str | Path, expected: str | None = None) -> str:
        path = self.path(name)
        key = path.relative_to(self.root).as_posix()
        if key not in self.files:
            digest = sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            self.files[key] = {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}
        actual = self.files[key]["sha256"]
        require(expected is None or actual == expected, f"Frozen artifact SHA mismatch: {key}")
        return actual

    def read(self, name: str | Path) -> Any:
        self.bind(name)
        return json.loads(self.path(name).read_text(encoding="utf-8"))

    def jsonl(self, name: str | Path):
        self.bind(name)
        with self.path(name).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)

    def relative_hashes(self, folder: str, hashes: dict[str, str], *, fallback: str | None = None) -> None:
        for name, expected in hashes.items():
            relative = f"{folder}/{_portable(name)}"
            if not (self.root / relative).is_file() and fallback:
                relative = f"{fallback}/{_portable(name)}"
            self.bind(relative, expected)


def _label_map(inputs: Inputs, sidecars: list[str]) -> dict[str, int]:
    labels: dict[str, int] = {}
    for path in sidecars:
        for envelope in inputs.jsonl(path):
            row = envelope["evaluation"]
            require(not row["independent_outcome_verified"] and not row["numeric_voi_training_eligible"],
                    "Historical reconstruction was promoted into independent gold labels")
            mid, outcome = row["match_id"], row["outcome_index"]
            require(mid not in labels or labels[mid] == outcome, "Inconsistent historical outcome sidecars")
            labels[mid] = outcome
    return labels


def summarize_historical_records(rows: list[dict], labels: dict[str, int], *, fold_lambdas: dict,
                                 old_report_objective: bool) -> list[dict]:
    """Recompute losses with evaluator-only labels, averaging seeds within match."""
    grouped: dict[tuple, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    initial: dict[str, list[float]] = {}
    seen = set()
    for row in rows:
        mid = row["match_id"]
        require(mid in labels, f"No frozen evaluator outcome for {mid}")
        loss = probability_loss(row["final_probabilities"], labels[mid])
        same_number(loss, row["log_loss"], "Saved historical log loss differs from frozen probabilities/outcome")
        brier = math.fsum((p - int(i == labels[mid])) ** 2 for i, p in enumerate(row["final_probabilities"]))
        same_number(brier, row["brier_score"], "Saved Brier score differs")
        require(mid not in initial or initial[mid] == row["initial_probabilities"],
                "Compared strategies have different initial probabilities")
        initial[mid] = row["initial_probabilities"]
        limits = row.get("limits", {"money": .05, "requests": 2})
        cost, queries = row["cost"]["money"], row["cost"]["requests"]
        require(queries == len(row["actions"]), "Historical action/count mismatch")
        require(cost <= limits["money"] + 1e-12 and queries <= limits["requests"], "Historical budget exceeded")
        expected_cost = math.fsum({"results_view": .02, "xg_view": .03}[a] for a in row["actions"])
        same_number(cost, expected_cost, "Frozen historical query costs differ")
        coefficient = fold_lambdas[row["fold"]]
        if "lambda_money" in row:
            same_number(row["lambda_money"], coefficient, "Historical policy lambda differs from frozen fold")
        objective = loss + coefficient * cost
        if "utility_loss" in row:
            same_number(objective, row["utility_loss"], "Historical policy objective differs")
        key = (row.get("model_name", "round175"), limits["money"], limits["requests"], row["strategy"])
        identity = (*key, mid, row.get("seed"))
        require(identity not in seen, "Duplicate match/policy/budget/seed would inflate evidence")
        seen.add(identity)
        grouped[key][mid].append({"loss": loss, "cost": cost, "queries": queries, "brier": brier,
                                 "objective": objective, "reported_objective": loss + .5 * cost if old_report_objective else objective,
                                 "seed": row.get("seed"), "fold": row["fold"]})
    metrics = []
    expected_matches = set(labels)
    for (model, money, requests, policy), matches in sorted(grouped.items()):
        require(set(matches) == expected_matches, "Incomplete paired historical match coverage")
        seeds = {r["seed"] for r in next(iter(matches.values()))}
        require(all({r["seed"] for r in values} == seeds for values in matches.values()), "Unequal random seed coverage")
        metric = {"model": model, "policy": policy, "budget": {"money": money, "requests": requests},
                  "independent_matches": len(matches), "recorded_trajectories": sum(map(len, matches.values())),
                  "seed_replicates_per_match": len(seeds),
                  "aggregation": "mean within each match across recorded seeds, then equal weight matches"}
        for name, raw in (("mean_log_loss", "loss"), ("mean_brier_score", "brier"), ("mean_cost", "cost"),
                          ("mean_queries", "queries"), ("mean_policy_lambda_objective", "objective"),
                          ("mean_reported_objective", "reported_objective")):
            metric[name] = mean(mean(r[raw] for r in values) for values in matches.values())
        metrics.append(metric)
    return metrics


def _historical(inputs: Inputs) -> tuple[list[dict], dict[str, set[str]]]:
    old_dir, new_dir = f"{BASE}/round175", f"{BASE}/round175/new-period"
    old = inputs.read(f"{old_dir}/report.json")
    new = inputs.read(f"{new_dir}/report.json")
    accepted = inputs.read(f"{old_dir}/acceptance.json")
    inputs.bind(f"{old_dir}/report.json", accepted["report_sha256"])
    inputs.bind(f"{new_dir}/report.json", accepted["new_period_report_sha256"])
    freeze = inputs.read(f"{old_dir}/frozen-next-period.json")
    inputs.bind(f"{old_dir}/frozen-next-period.json", accepted["freeze_sha256"])
    baseline = inputs.read("var/parallel/baselines/round8/joint-returns-v2/report.json")
    labels = _label_map(inputs, [f"{old_dir}/folds/{f['fold']}/evaluation-sidecars.jsonl" for f in old["folds"]])
    new_labels = _label_map(inputs, [f"{new_dir}/evaluation-sidecars.jsonl"])
    require(len(labels) == 290 and len(new_labels) == 50 and not set(labels) & set(new_labels),
            "Old/new historical independent match counts differ")
    require(not set(new_labels) & set(freeze["forbidden_match_ids"]), "Frozen temporal set overlaps prior development")
    lambdas = {f["fold"]: f["lambda_money"] for f in old["folds"]}
    lambdas["new-season-2026"] = new["lambda_money"]
    rolling_splits = [{"fold": f["fold"], "training_matches": len(set(f["training_match_ids"])),
                       "validation_matches": len(set(f["validation_match_ids"])),
                       "test_matches": len(set(f["test_match_ids"]))} for f in baseline["folds"]]
    experiments = []
    specs = [("round175_reused_history", old_dir, old, labels, True),
             ("round175_frozen_temporal", new_dir, new, new_labels, True),
             ("round177_context_development", f"{BASE}/round177/evaluation/results", None, labels, False),
             ("round177_residual_development", f"{BASE}/round177/residual-calibration/results", None, labels, False)]
    for eid, folder, report, label_set, old_objective in specs:
        report = report or inputs.read(f"{folder}/report.json")
        # Frozen predictions suffice for score recomputation; do not bundle every
        # historical model/trajectory or rerun them merely to reproduce a table.
        inputs.bind(f"{folder}/paired-episodes.json", report["artifacts_sha256"]["paired-episodes.json"])
        if eid == "round177_context_development":
            prior = inputs.read(f"{BASE}/round177/evaluation/acceptance.json")
            inputs.bind(f"{folder}/report.json", prior["artifacts_sha256"]["report.json"])
            inputs.bind(f"{folder}/design.json", report["design_sha256"])
        elif eid == "round177_residual_development":
            prior = inputs.read(f"{BASE}/round177/residual-calibration/acceptance.json")
            original_hash = next(value for key, value in prior["sha256"].items() if _portable(key) == f"{folder}/report.json")
            inputs.bind(f"{folder}/report.json", original_hash)
            inputs.bind(f"{BASE}/round177/residual-calibration/design.md", report["design_sha256"])
            inputs.read(f"{folder}/input-bindings.json")
        else:
            for name, digest in report["artifacts_sha256"].items():
                if name.endswith("evaluation-sidecars.jsonl"):
                    inputs.bind(f"{folder}/{_portable(name)}", digest)
        rows = inputs.read(f"{folder}/paired-episodes.json")
        metrics = summarize_historical_records(rows, label_set, fold_lambdas=lambdas,
                                               old_report_objective=old_objective)
        reported = report["metrics"]
        for metric in metrics:
            if old_objective:
                reference = reported[metric["policy"]]
            else:
                reference = next(m for m in reported if m["model_name"] == metric["model"]
                                 and m["strategy"] == metric["policy"] and m["limits"] == metric["budget"])
            for key in ("mean_log_loss", "mean_cost", "mean_queries", "mean_brier_score"):
                same_number(metric[key], reference[key], f"{eid} {key} cannot reproduce report")
            same_number(metric["mean_reported_objective"], reference["mean_utility_loss"], "Historical report objective changed")
        temporal = eid == "round175_frozen_temporal"
        experiments.append({"id": eid, "data_kind": "real_historical_reconstruction", "source": f"{folder}/report.json",
            "metric_verification": "recomputed_from_frozen_probabilities_and_evaluator_outcomes",
            "independent_match_ids": sorted(label_set), "independent_matches": len(label_set),
            "recorded_trajectories": len(rows), "independently_verified_labels": 0,
            "label_scope": "same_provider_reconstructed_regulation_result_not_independent_gold",
            "is_fresh_confirmation": False, "new_temporal_matches_after_parameter_freeze": temporal,
            "reused_development_set": not temporal, "same_development_cohort": None if temporal else "history_290",
            "policy_cost_unit": "simulated projection credit; not provider billing",
            "policy_lambda_by_fold": {str(k): v for k, v in lambdas.items() if (k == "new-season-2026") == temporal},
            "report_objective_definition": "log_loss + 0.5 * money" if old_objective else "log_loss + frozen_fold_lambda * money",
            "policy_objective_definition": "log_loss + frozen_fold_lambda * money",
            "same_initial_predictions_verified": True, "complete_policy_cells_retained": True,
            "rolling_training_splits": rolling_splits if not temporal else [],
            "metrics": metrics, "limitations": [
                "Historical features are reconstructed rather than observed prospective news acquisitions.",
                "Evaluation outcomes come from the same provider; they are not independent result verification.",
                "This is a new temporal reconstruction after parameter freeze, not prospective confirmation." if temporal else
                "The same 290-match development cohort was reused; post-diagnosis changes are not new independent confirmation.",
                "Round175's reported fixed 0.5 cost weight differs from the policy's fold-specific weight; both are retained." if old_objective else
                "Repeated random seeds, model families and reused trajectories do not increase independent match count."]})
    return experiments, {"historical_development": set(labels), "historical_new_temporal": set(new_labels)}


def summarize_synthetic_records(rows: list[dict], *, expected_matches: set[str], version: int,
                                cost_per_query: float | None = None) -> tuple[list[dict], dict]:
    """Verify frozen potential-return scores; never rerun or tune an actor."""
    grouped = defaultdict(list)
    seen = set()
    starts = {}
    negative = substantive = unknown = 0
    for row in rows:
        identity = (row["policy"], row["budget"], row["match_id"])
        require(identity not in seen, "Duplicate synthetic policy trajectory")
        seen.add(identity)
        require(row["match_id"] in expected_matches, "Undeclared synthetic holdout match")
        start_key = (row["match_id"], row["budget"])
        require(start_key not in starts or starts[start_key] == row["initial_probabilities"], "Unfair synthetic initial state")
        starts[start_key] = row["initial_probabilities"]
        loss = probability_loss(row["final_probabilities"], row["outcome"])
        same_number(loss, row["log_loss"], "Synthetic loss does not match frozen probability/outcome")
        same_number(loss + row["cost"], row["objective"], "Synthetic loss/cost objective differs")
        require(row["cost"] <= row["budget"] + 1e-12, "Synthetic physical budget exceeded")
        require(row["paid_queries"] == len(row["transitions"]), "Synthetic charged query count differs")
        if version == 179:
            require(row["paid_queries"] <= row["max_requests"], "Synthetic request cap exceeded")
            calculated_cost = math.fsum(t["action"]["cost"]["money"] for t in row["transitions"])
        else:
            calculated_cost = len(row["transitions"]) * cost_per_query
        same_number(calculated_cost, row["cost"], "Synthetic trace costs differ")
        row_negative = row_substantive = row_unknown = 0
        previous = row["initial_probabilities"]
        for step in row["transitions"]:
            require(step["posterior_before"] == previous, "Synthetic update history is discontinuous")
            before = probability_loss(step["posterior_before"], row["outcome"])
            after = probability_loss(step["posterior_after"], row["outcome"])
            same_number(before - after, step["realized_log_loss_reduction"], "Retained negative/positive gain changed")
            gain = step["realized_log_loss_reduction"]
            row_negative += gain < 0
            row_substantive += gain < -1e-12
            row_unknown += step["update_status"] != "updated"
            previous = step["posterior_after"]
            if version == 179:
                require(step["action_visible_before_execution"] and step["action"] in step["public_menu_before"],
                        "An undiscovered action entered the execution menu")
        require(previous == row["final_probabilities"], "Synthetic final prediction differs from retained updates")
        negative += row_negative
        substantive += row_substantive
        unknown += row_unknown
        grouped[(row["budget"], row["policy"])].append({**row, "negative": row_negative,
                                                       "substantive_negative": row_substantive, "unknown": row_unknown})
    metrics = []
    for (budget, policy), group in sorted(grouped.items()):
        require({r["match_id"] for r in group} == expected_matches, "Unpaired synthetic policy coverage")
        metrics.append({"policy": policy, "budget": {"money": budget,
                        "requests": 3 if version == 179 else int(round(budget / cost_per_query))},
                        "independent_matches": len(group), "recorded_trajectories": len(group),
                        "mean_log_loss": mean(r["log_loss"] for r in group), "mean_cost": mean(r["cost"] for r in group),
                        "mean_objective": mean(r["objective"] for r in group),
                        "mean_queries": mean(r["paid_queries"] for r in group),
                        "negative_realized_updates": sum(r["negative"] for r in group),
                        "negative_realized_updates_beyond_roundoff": sum(r["substantive_negative"] for r in group),
                        "unknown_updates": sum(r["unknown"] for r in group)})
    return metrics, {"negative_realized_updates": negative, "negative_beyond_roundoff": substantive,
                     "unknown_updates": unknown}


def _synthetic(inputs: Inputs, version: int, variant: str | None = None) -> tuple[dict, set[str]]:
    shared = f"{BASE}/round{version}/evaluation"
    folder = f"{shared}/{variant}" if variant else shared
    acceptance = inputs.read(f"{folder}/acceptance.json")
    archive_only = {}
    failures_only = version == 179 and variant != "final-request-class"
    needed = {"design.json", "heldout-matches.json", "training-records.json"}
    if not failures_only:
        needed |= {"policy-records.json", "mechanism-evidence.json"}
    for name, digest in acceptance["file_sha256"].items():
        if name not in needed:
            archive_only[name] = digest
            continue
        path = f"{folder}/{name}"
        if not (inputs.root / path).is_file():
            path = (f"{shared}/corrected-request-class/{name}" if variant == "dispatch-failure-attempt"
                    and name == "frozen-discovery-model.json" else f"{shared}/{name}")
        inputs.bind(path, digest)
    design = inputs.read(f"{shared}/design.json")
    if version == 179 and not failures_only:
        require(all(acceptance["checks"].values()), "Final discovery mechanism checks did not pass")
    heldout = inputs.read(f"{shared}/heldout-matches.json")
    labels = {r["match_id"]: r["outcome"] for r in heldout}
    require(len(labels) == len(heldout) == acceptance["heldout_matches"], "Synthetic holdout uniqueness differs")
    training = inputs.read(f"{shared}/training-records.json")
    training_ids = {r["match_id"] for r in training}
    require(not training_ids & set(labels), "Synthetic training/holdout overlap")
    require(len(training_ids) == acceptance["training_matches"], "Synthetic training count differs")
    require(acceptance["production_ready"] is False, "Synthetic mechanism was promoted to production evidence")
    if failures_only:
        # Keep all failed cells, using their immutable prior acceptance as the
        # evidence level. No second copy of 100MB models or 14MB failure traces.
        metrics = [{"policy": r["policy"], "budget": {"money": r["budget"], "requests": 3},
                    "independent_matches": acceptance["heldout_matches"],
                    "recorded_trajectories": acceptance["heldout_matches"],
                    "mean_log_loss": r["mean_log_loss"], "mean_cost": r["mean_cost"],
                    "mean_objective": r["mean_objective"], "mean_queries": r["mean_paid_queries"],
                    "negative_realized_updates": r["negative_realized_updates"],
                    "negative_realized_updates_beyond_roundoff": r.get("negative_realized_updates_beyond_roundoff"),
                    "unknown_updates": r["unknown_updates"]} for r in acceptance["summary"]]
        counts = {"negative_realized_updates": acceptance["negative_realized_updates_retained"],
                  "negative_beyond_roundoff": acceptance.get("negative_realized_updates_beyond_roundoff"),
                  "unknown_updates": acceptance["unknown_updates_retained"]}
        records_count = acceptance["policy_runs"]
    else:
        records = inputs.read(f"{folder}/policy-records.json")
        require(all(labels.get(r["match_id"]) == r["outcome"] for r in records), "Policy outcome differs from frozen holdout")
        metrics, counts = summarize_synthetic_records(records, expected_matches=set(labels), version=version,
            cost_per_query=design["cost"]["per_paid_query"] if version == 178 else None)
        records_count = len(records)
        require(records_count == acceptance["policy_runs"], "Synthetic trajectory count differs")
    for metric in metrics:
        ref = next(r for r in acceptance["summary"] if r["policy"] == metric["policy"] and r["budget"] == metric["budget"]["money"])
        for key in ("mean_log_loss", "mean_cost", "mean_objective"):
            same_number(metric[key], ref[key], f"Frozen synthetic summary {key} differs")
        same_number(metric["mean_queries"], ref["mean_paid_queries"], "Frozen synthetic mean queries differs")
        require(metric["negative_realized_updates"] == ref["negative_realized_updates"], "Negative updates omitted")
        require(metric["unknown_updates"] == ref["unknown_updates"], "Unknown updates omitted")
    require(counts["negative_realized_updates"] == acceptance["negative_realized_updates_retained"], "Negative total differs")
    require({(m["policy"], m["budget"]["money"]) for m in metrics} ==
            {(policy, budget) for policy in design["policies"] for budget in design["budgets"]},
            "A preregistered policy/budget cell was omitted")
    mechanism = None if failures_only else inputs.read(f"{folder}/mechanism-evidence.json")
    if version == 178:
        mechanism_values = [{"context": r["context"], "injury_net_voi": r["injury_before"]["value"]["net_voi"],
            "injury_gross_evsi": r["injury_before"]["value"]["gross_evsi"],
            "duplicate_after_original_evsi": r["same_injury_republication_after"]["value"]["gross_evsi"]} for r in mechanism]
    elif failures_only:
        mechanism_values = {"checks": acceptance["checks"], "request_value_mode": acceptance.get("request_value_mode"),
                            "verification_scope": "frozen prior acceptance; failure trajectories not rerun or duplicated in release"}
    else:
        mechanism_values = {"checks": acceptance["checks"], "scenarios": mechanism["scenarios"],
            "request_value_mode": acceptance.get("request_value_mode"),
            "duplicate_after_original_evsi": mechanism["after_original_duplicate_value"]["value"]["gross_evsi"]}
    return {"id": f"round{version}_{variant or 'text_mechanism'}", "data_kind": "synthetic_constructed_mechanism",
        "source": f"{folder}/acceptance.json", "status": acceptance["status"],
        "metric_verification": "frozen_prior_acceptance_cells_retained_without_reexecuting_failed_attempt" if failures_only else
            "recomputed_from_frozen_policy_probabilities_outcomes_costs_and_transitions",
        "training_matches": len(training_ids), "independent_matches": len(labels),
        "independent_match_ids": sorted(labels), "recorded_trajectories": records_count,
        "archived_artifact_hash_declarations": archive_only,
        "archive_declaration_scope": "Hashes recorded by the original acceptance; these large models/old failed traces are not needed or rechecked for score recomputation.",
        "is_fresh_confirmation": False, "reused_development_set": version == 179,
        "same_development_cohort": "synthetic_discovery_240" if version == 179 else "synthetic_text_300",
        "trained_test_overlap": 0, "policy_cost_unit": "simulated loss credit", "cost_lambda": 1.0,
        "objective_definition": "log_loss + simulated_cost", "same_initial_predictions_verified": True,
        "free_information": design.get("initial_context", design.get("common_free_context")),
        "initial_menu": ["A", "B", "C"] if version == 178 else ["discovery"],
        "complete_policy_cells_retained": True, "metrics": metrics, "retained_update_counts": counts,
        "mechanism_values": mechanism_values, "limitations": [
            "Constructed alphabet and full-prefix simulation, not real football or natural-language truth.",
            "All policies receive the same free context; context discovery is not evaluated here.",
            "Same development holdout reused across medoid, dispatch failure and request-class fix; not new confirmation." if version == 179 else
            "Train/test matches are isolated, but the simulated process does not establish real-news value.",
            "One initial discovery option and identical syndicated signals make fixed and random policies equivalent here; this does not test general multi-tool ranking." if version == 179 else
            "The ordered conditional kernel is not proven order invariant.",
            "The two-step approximation uses future request classes; actual URLs are revalued only after discovery." if version == 179 else
            "A positive expected value does not imply every realized update reduces loss."]}, set(labels)


def _prospective(inputs: Inputs) -> tuple[dict, set[str]]:
    folder = f"{BASE}/round180"
    accepted = inputs.read(f"{folder}/acceptance.json")
    require(accepted["status"] == "passed", "Round180 prior acceptance not passed")
    plan = inputs.read(f"{folder}/collection-plan.json")
    prior_live = inputs.read(f"{BASE}/round179/prospective/acceptance.json")
    inputs.bind(f"{folder}/collection-plan.json", accepted["collection_source_freeze"]["plan_sha256"])
    inputs.bind(f"{folder}/identity/identity-bundle.json", accepted["identities"]["identity_bundle_sha256"])
    inputs.bind(f"{folder}/episodes-pending.json", accepted["dataset"]["episodes_file_sha256"])
    episodes = inputs.read(f"{folder}/episodes-pending.json")
    require(not episodes["training_records"], "Round180 unexpectedly acquired new training labels")
    per_match, ids, observations = [], set(), set()
    for receipt in accepted["actual_acquisition"]["runs"]:
        suffix = receipt["match_id"].rsplit(":", 1)[-1]
        original = f"{folder}/live-batch/match-{suffix}"
        run = inputs.read(f"{original}/run.json")
        inputs.bind(f"{original}/run.json", receipt["run_sha256"])
        require(receipt["match_id"] not in ids, "Duplicate prospective match counted independently")
        ids.add(receipt["match_id"])
        require(run["outcome"] is None and not run["independent_outcome_verified"]
                and not run["numeric_voi_training_eligible"], "Prospective pending data promoted to training")
        selected = [c for c in run["cycles"] if c.get("selected_action") is not None]
        require(len(selected) == receipt["requests"] <= plan["requests_per_match"], "Prospective query/budget mismatch")
        require(all(c["selected_action"] in c["generated_actions"] for c in selected), "Action absent from actual menu")
        unknown = sum(c["selection_basis"] == "bounded_unknown_information_exploration" for c in selected)
        changed = sum(any(abs(a-b) > 1e-12 for a,b in zip(c["prediction_before"], c["prediction_after"])) for c in selected)
        require(unknown == receipt["selected_unknown_value_actions"] and changed == receipt["probability_updates_beyond_roundoff"],
                "Prospective selection scope differs")
        for name in receipt["byte_identical_replay_files"]:
            actual = inputs.bind(f"{original}/{name}")
            inputs.bind(f"{folder}/replays/match-{suffix}/{name}", actual)
        archive = inputs.root / original / "source-archive"
        expected_obs = {r["observation_id"]: r for r in receipt["observations"]}
        seen_raw = set()
        for path in sorted(archive.rglob("*.observation.json")):
            raw = inputs.read(path.relative_to(inputs.root))
            oid = raw["id"]
            require(oid not in observations and oid in expected_obs, "Reused or unreported actor observation")
            observations.add(oid)
            seen_raw.add(oid)
            require(raw["context"]["inquiry_run_id"] == run["run_id"], "Diagnostic response counted as actor response")
            payload = archive / _portable(raw["payload_path"])
            inputs.bind(payload.relative_to(inputs.root))
            with gzip.open(payload, "rb") as handle:
                digest = sha256(handle.read()).hexdigest()
            require(digest == raw["content_hash"] == expected_obs[oid]["raw_sha256"], "Raw actor body SHA differs")
        require(seen_raw == set(expected_obs), "Missing actual actor raw responses")
        # runtime charged_cost is the cumulative spent vector, not per-call cost.
        simulated_cost = math.fsum(c["selected_action"]["cost"]["money"] for c in selected)
        same_number(simulated_cost, selected[-1]["charged_cost"]["money"], "Cumulative actual reservation differs")
        require(simulated_cost <= .03 + 1e-12, "Prospective simulated budget exceeded")
        per_match.append({"match_id": receipt["match_id"], "run_id": run["run_id"], "requests": len(selected),
            "budget": {"money": .03, "requests": 3}, "reserved_simulated_cost": simulated_cost,
            "actual_money_cost_known": all(c["actual_cost"]["money"] is not None for c in selected),
            "article_documents": receipt["article_documents"], "target_named_articles": receipt["target_named_article_documents"],
            "target_named_articles_without_topic_evidence": receipt["target_named_articles_without_topic_evidence"],
            "source_assertions": len(run["evidence"]["evidence"]),
            "source_assertion_applicability": receipt["source_assertion_applicability"],
            "selected_discovered_link_reads": receipt["selected_discovered_link_reads"],
            "selected_grounded_followup_queries": receipt["selected_grounded_followup_queries"],
            "unknown_value_exploration_actions": unknown, "numerically_positive_voi_actions": receipt["selected_positive_numeric_voi_actions"],
            "probability_updates": changed, "terminal": run["terminal_selection"],
            "response_status_counts": dict(Counter(o["response_status"] for o in receipt["observations"]))})
    require(len(ids) == 3 and len(observations) == 8, "Prospective independent sample count changed")
    require(ids == {r["match_id"] for r in prior_live["audit"]["runs"]}, "Round179/180 live match reuse declaration differs")
    require(accepted["independent_labels"] == accepted["eligible_real_text_training_samples"] == 0,
            "Round180 changed from pending evidence")
    return {"id": "round180_actual_prospective", "data_kind": "real_prospective_unsettled_acquisition", "source": f"{folder}/acceptance.json",
        "metric_verification": "recomputed_actual_choices_and_raw_bodies; prior_hash_bound_acceptance_for_projection_semantics",
        "independent_match_ids": sorted(ids), "independent_matches": 3, "recorded_trajectories": 3,
        "training_matches": 0, "independently_verified_labels": 0, "numeric_text_voi_training_samples": 0,
        "realized_loss": None, "realized_voi_benefit_proven": False, "numeric_text_model_deployed": False,
        "is_fresh_confirmation": False, "is_actual_pre_match_collection": True, "seed": plan["seed"],
        "per_match": per_match, "actual_requests": 8, "diagnostic_requests_excluded": 19,
        "counts": accepted["actual_acquisition"]["totals"],
        "global_unique_observed_link_urls": accepted["actual_acquisition"]["global_unique_observed_link_urls"],
        "legacy_replay_matches_counted_as_new": 0, "budget": {"money": .03, "requests": 3},
        "cost_unit": "actual request count; reserved money is simulation credit, actual provider money unknown",
        "limitations": [
            "Eight actions used bounded unknown-information exploration; none was a learned positive numerical text-VoI choice.",
            "Three article-shaped documents include one generic page without target teams; two with names are not automatically verified match evidence.",
            "Croatia source assertion requires match binding; its grounded followup failed and remains recorded.",
            "Scotland target pre-match conference was obtained but the finite topic extractor found no substantive evidence.",
            "No independent final scores attached as of this frozen acceptance; no live loss or policy benefit can be computed.",
            "These are the same three match identities used in Round179 live preparation, not three additional independent matches."]}, ids


def _claims(experiments: list[dict]) -> list[dict]:
    by_id = {e["id"]: e for e in experiments}
    definitions = [
        ("match_only_autonomous_acquisition", "用户只给比赛，系统依据当前证据生成并执行下一步信息动作", True,
         ["round180_actual_prospective"], "真实执行机制；不是已验证的数值 VoI 收益",
         "3 场 8 次真实请求，含新链接读取与原文锚定追问；当前动作全部属于未知价值下的有预算探索。"),
        ("context_conditioned_text_voi", "同一信息动作的价值随已读情境改变", True,
         ["round178_text_mechanism"], "合成机制验证", "同一伤停查询净值可正可负；不代表真实出线或轮换新闻的已证效果。"),
        ("learned_discovery_continuation", "即时 EVSI 为零的发现动作可以有正的后续价值", True,
         ["round179_final-request-class"], "有限两步、合成菜单机制", "请求类近似及受支持的有限菜单；同一开发留出集实施修复后复验。"),
        ("correlated_republication", "已获取原文后重复转载不再贡献独立信息增益", True,
         ["round178_text_mechanism", "round179_final-request-class"], "预设完全相关信号的受控实验",
         "不能据此声称已解决真实新闻语义去重、证据真实性或所有相关性建模。"),
        ("synthetic_cost_quality_tradeoff", "在声明的合成设置中以更低成本取得更好的损失加成本目标", True,
         ["round178_text_mechanism", "round179_final-request-class"], "合成成本与固定等预算比较",
         "高预算固定全查的纯预测损失略低；Round178扩大VoI预算后综合目标略差，这些不胜结果保留。"),
        ("historical_policy_superiority", "已证实自主信息获取在真实历史比赛上具有稳健的成本与质量优势", False,
         ["round175_reused_history", "round175_frozen_temporal", "round177_context_development", "round177_residual_development"],
         "不支持", "旧290场在主预算0.05/2下strict-VoI仍不优于不查询；单请求context纯loss有微小改善，但成本加权目标未占优。冻结新50场修正策略全部不查询，等于基线。"),
        ("real_news_numeric_voi_benefit", "真实赛前正文的数值 VoI 已学习并证明收益", False,
         ["round180_actual_prospective"], "不支持", "0独立赛果标签、0合格真实正文训练样本、0真实正数值VoI动作。"),
        ("open_world_optimal_stopping", "已经证明对开放互联网信息的全局最优搜索与停止", False,
         ["round179_final-request-class", "round180_actual_prospective"], "不支持",
         "有限两步受限策略值不能作为整个开放世界最优值；未知候选不能被当作零，真实运行没有最优停止证书。"),
        ("aic_reproducible_research_prototype", "可展示可复现的信息获取机制、预算对照和真实执行边界", True,
         ["round178_text_mechanism", "round179_final-request-class", "round180_actual_prospective"],
         "工程与受控研究原型", "可作为参赛原型证据，不等于算法新颖性审稿认可或真实泛化收益已经成立。"),
    ]
    return [{"claim_id": cid, "claim_zh": title, "supported_in_stated_scope": supported,
             "real_world_policy_benefit_proven": False, "scope_zh": scope, "limitation_zh": limit,
             "evidence_ids": ids, "evidence_paths": [by_id[i]["source"] for i in ids]}
            for cid, title, supported, ids, scope, limit in definitions]


def build_evidence_summary(root: Path) -> dict:
    """Read and recompute release evidence. No writes, policy execution or network."""
    inputs = Inputs(Path(root))
    experiments, cohorts = _historical(inputs)
    text, text_ids = _synthetic(inputs, 178)
    experiments.append(text)
    for variant in ("initial-medoid-attempt", "dispatch-failure-attempt", "final-request-class"):
        item, ids = _synthetic(inputs, 179, variant)
        require("synthetic_discovery" not in cohorts or cohorts["synthetic_discovery"] == ids,
                "Discovery implementation attempts changed their match cohort")
        cohorts["synthetic_discovery"] = ids
        experiments.append(item)
    actual, real_ids = _prospective(inputs)
    experiments.append(actual)
    # Use the declared repository-relative module even when a parent verifier
    # imports this function from a different checkout to inspect a portable copy.
    inputs.bind("football_research/evaluation/release_evidence.py")
    claims = _claims(experiments)
    return {"schema": SCHEMA, "status": "passed", "scope": "offline frozen-evidence release; no new experiments or fitting",
        "metric_definitions": {"log_loss": "negative natural logarithm of the frozen probability assigned to the evaluator outcome",
            "historical_aggregation": "equal weight each match after averaging repeated random seeds within that match",
            "synthetic_aggregation": "one trajectory per policy/budget/match, equal weight matches",
            "real_costs": "network request counts are actual; simulation credit is not currency or measured provider bill",
            "independent_sample": "unique match within its declared cohort, never policy runs, repeated rounds, prefixes or replays",
            "intervals": "existing bootstrap intervals are hash-bound but not presented as new confirmatory significance",
            "round175_objective_warning": "original report uses 0.5*cost while planning uses fold lambda; both reported and policy-lambda metrics are retained"},
        "sample_accounting": {"historical_development_unique_matches": len(cohorts["historical_development"]),
            "historical_new_temporal_unique_matches": len(cohorts["historical_new_temporal"]),
            "historical_union_unique_matches": len(cohorts["historical_development"] | cohorts["historical_new_temporal"]),
            "round177_additional_independent_matches": 0, "synthetic_text_holdout_matches": len(text_ids),
            "synthetic_discovery_holdout_matches": len(cohorts["synthetic_discovery"]),
            "discovery_implementation_retries_additional_matches": 0, "real_prospective_unique_matches": len(real_ids),
            "round180_additional_matches_relative_to_round179_live": 0,
            "verified_real_text_training_samples": 0,
            "warning": "Do not sum synthetic and real cohorts into a real efficacy sample; labels lack independent verification in historical reconstruction."},
        "experiments": experiments, "claims": claims,
        "research_flags": {"open_world_policy_benefit_proven": False, "real_text_numeric_voi_benefit_proven": False,
            "independent_real_text_confirmation_complete": False, "aic_algorithmic_advantage_proven": False},
        "verification": {"new_policy_runs": 0, "model_fits": 0, "network_requests": 0,
            "all_metric_cells_retained": True, "negative_and_failed_attempts_retained": True,
            "old_production_implementation_hashes_not_required_to_match_current_code": True,
            "note": "Archived artifacts are checked against their prior acceptance hashes; historical production sources can evolve without rewriting old evidence."},
        "input_bindings": dict(sorted(inputs.files.items())),
        "file_sha256": dict(sorted(inputs.files.items()))}


def verify_evidence_summary(root: Path, summary_path: Path) -> dict:
    """Verify dependencies first, then require a complete deterministic recomputation."""
    root = Path(root).resolve()
    path = Path(summary_path)
    if not path.is_absolute():
        path = root / path
    saved = json.loads(path.read_text(encoding="utf-8"))
    require(saved.get("schema") == SCHEMA, "Unexpected evidence summary schema")
    inputs = Inputs(root)
    for name, binding in saved["file_sha256"].items():
        inputs.bind(name, binding["sha256"])
    recomputed = build_evidence_summary(root)
    require(saved == recomputed, "Saved evidence summary differs from complete frozen-record recomputation")
    return {"status": "passed", "summary_sha256": sha256(path.read_bytes()).hexdigest(),
            "verified_input_files": len(saved["file_sha256"]), "experiment_entries": len(saved["experiments"]),
            "network_requests": 0, "new_policy_runs": 0, "model_fits": 0,
            "sample_accounting": saved["sample_accounting"], "research_flags": saved["research_flags"]}


def render_experiment_table(summary: dict) -> str:
    lines = ["# FootballMind 冻结实验证据表", "", "本表由已有冻结记录离线重算；没有重新训练、执行策略或联网。所有策略与不胜结果保留。", "",
        "真实历史重建、合成机制、真实未结算采集分开列出。Round175/177共享旧290场，后者不增加独立比赛；Round175新时期50场未用于选参，但仍是同源历史重建。Round178/179留出分别300/240个合成场景；Round179失败修复复用同一240场，不能算新确认集。Round180仍是Round179准备的3场，回放和再次采集不增加独立比赛。", "",
        "损失使用自然对数 log loss，越低越好。模拟成本不是人民币、美元或真实供应商账单。历史随机策略先在每场内平均种子，再按比赛等权。Round175原汇总成本权重固定0.5，与实际各折策略权重不同；两列都保留。", ""]
    for experiment in summary["experiments"]:
        lines.extend([f"## {experiment['id']}", "", f"类型：`{experiment['data_kind']}`；独立比赛/场景 {experiment['independent_matches']}；保存轨迹 {experiment['recorded_trajectories']}。", "",
                      f"冻结来源：`{experiment['source']}`，SHA见 `evidence-summary.json.file_sha256`。", ""])
        if "metrics" in experiment:
            historical = experiment["data_kind"] == "real_historical_reconstruction"
            lines.extend(["| 模型 / 策略 | 预算：模拟成本 / 请求 | log loss | 平均请求 | 平均模拟成本 | 原报告目标 | 实际策略权重目标 |",
                          "|---|---:|---:|---:|---:|---:|---:|"])
            for m in experiment["metrics"]:
                label = f"{m['model']} / {m['policy']}" if historical else m["policy"]
                reported = m["mean_reported_objective"] if historical else m["mean_objective"]
                objective = m["mean_policy_lambda_objective"] if historical else m["mean_objective"]
                lines.append(f"| {label} | {m['budget']['money']:g} / {m['budget']['requests']:g} | {m['mean_log_loss']:.6f} | {m['mean_queries']:.6f} | {m['mean_cost']:.6f} | {reported:.6f} | {objective:.6f} |")
            lines.append("")
        else:
            lines.extend(["| 比赛 | 实际请求 | 文章结构 / 含目标球队 | 提取断言 | 新链接读取 | 锚定后续查询 | 数值正 VoI 动作 |",
                          "|---|---:|---:|---:|---:|---:|---:|"])
            for row in experiment["per_match"]:
                lines.append(f"| {row['match_id']} | {row['requests']} | {row['article_documents']} / {row['target_named_articles']} | {row['source_assertions']} | {row['selected_discovered_link_reads']} | {row['selected_grounded_followup_queries']} | {row['numerically_positive_voi_actions']} |")
            lines.extend(["", "三场的比赛身份和原始响应已核验；赛果保持PENDING，0独立标签、0合格真实正文训练样本。实际8次动作皆未知价值下的有限探索，不能报告真实预测损失或数值VoI收益。", ""])
        lines.extend(["适用限制：", ""] + [f"- {x}" for x in experiment["limitations"]] + [""])
    lines.extend(["## 必须保留的结论", "",
        "- 真实历史旧290场在主预算0.05/2下，修正/上下文/残余关联 strict-VoI 仍未优于不查询。0.03/1预算下context strict纯loss为1.079009，略低于base的1.080229，但成本加权目标为1.085854，仍高于base；不能概括为所有预算、所有指标均负。冻结新50场修正策略全部不查询，与基线相等。",
        "- Round178预算0.24的固定全查纯loss为0.671697，略优于VoI的0.672146；VoI预算从0.08增加至0.24时loss+cost从0.771036略差至0.772679。",
        "- Round179固定/随机纯loss为0.900448，优于两步VoI的0.908912；两步VoI靠减少查询成本取得更好综合目标，不能写所有指标获胜。",
        "- Round179只有一个初始发现动作，转载信号相同，因此固定和随机在该机制下等价；不支持一般多工具排序优势。",
        "- Round179原medoid失败及分派失败完整保留；最终请求类修复是在同一开发留出集上的复验。",
        "- 未执行冻结LLM启发式对照，未完成真实正文数值VoI训练或真实等预算策略效益确认；这些不能包装为已完成实验。", "", "## AIC主张边界", "",
        "| 主张 | 本次证据支持 | 范围与限制 |", "|---|---|---|"])
    for claim in summary["claims"]:
        lines.append(f"| {claim['claim_zh']} | {'在声明范围内支持' if claim['supported_in_stated_scope'] else '不支持'} | {claim['scope_zh']}；{claim['limitation_zh']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    if args.command == "build":
        summary = build_evidence_summary(root)
        output.mkdir(parents=True, exist_ok=True)
        for name, value in (("evidence-summary.json", summary), ("claim-table.json", {"schema": SCHEMA, "claims": summary["claims"]})):
            (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        (output / "experiment-table.md").write_text(render_experiment_table(summary), encoding="utf-8")
        result = {"status": "built", "experiments": len(summary["experiments"]),
                  "input_files": len(summary["file_sha256"]), "output": str(output), "sample_accounting": summary["sample_accounting"]}
    else:
        result = verify_evidence_summary(root, output / "evidence-summary.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
