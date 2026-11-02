"""Freeze the existing text model from registered, raw-verified actor runs.

This is an operational data gateway, not a historical-page import or a new
learning algorithm. A global sample gate is necessary for fitting; it does
not establish conditional query support, calibration or useful information.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from football_agent.learned_voi_runtime import digest as runtime_digest
from football_research.acquisition.state import parse_time
from football_research.acquisition.text_returns import TextReturnModel
from football_research.acquisition.text_history import HistoryNeighborhoodTextReturnModel, load_text_model
from .open_text_episodes import empirical_risk_records
from .prospective_learning import EXPOSURE_VERSION, FACTS, SCOPE, _previous_inventory
from .prospective_text import (join_settlements, verify_run_identity, verify_prior_archive,
                               verify_run_learning, _digest)

VERSION = "prospective-frozen-text-training-v1"
MODEL_KINDS = {"ordered-v1": TextReturnModel, "neighborhood-v2": HistoryNeighborhoodTextReturnModel}
PARAMETERS = {"bandwidth": .2, "dirichlet_mass": 1., "min_matches": 12,
              "min_class_matches": 1, "text_clusters": 8, "action_clusters": 8,
              "min_action_similarity": .15}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+"\n")


def _registered_runs(run_folders, previous_plans, as_of):
    inventory = _previous_inventory(previous_plans, as_of)
    by_mid = {mid: row for row in inventory for mid in row["match_ids"]}
    legacy = [(plan, row) for plan in previous_plans if plan["schema"] == EXPOSURE_VERSION
              for row in plan["runs"]]
    runs, proofs, identities, seen = [], {}, {}, {}
    for folder in map(Path, run_folders):
        config, run = _read(folder/"configuration.json"), _read(folder/"run.json")
        if (run["configuration_digest"] != runtime_digest(config) or run["run_id"] != config["run_id"]
                or run["schema"] != config["schema"]):
            raise ValueError("Original run/configuration binding is inconsistent")
        if not parse_time(config["created_at"]) <= parse_time(run["finished_at"]) <= as_of:
            raise ValueError("Only actually completed runs can enter a training snapshot")
        if config["schema"] == "learned-voi-open-execution-v5":
            raise ValueError("Evidence-only acquisition has no numerical prior and cannot enter numerical predictor training")
        cohort = config.get("supervisor_metadata", {}).get("learning_cohort")
        if cohort is not None:
            if cohort.get("purpose") != "f_train":
                raise ValueError("Original calibration/confirmation runs cannot train f")
            verify_run_learning(folder, config)
        binding = verify_run_identity(folder, config)
        if binding is None:
            raise ValueError("Training requires raw-verified independent fixture identities")
        verify_prior_archive(binding["initialization_manifest"], folder/"prior-archive")
        mid, native = binding["match_id"], binding["canonical"]
        registered = by_mid.get(mid)
        if registered is None:
            raise ValueError("unknown_training_identity_inventory: "+mid)
        if registered["purpose"] != "f_train":
            raise ValueError("Training fixture belongs to the wrong cohort purpose")
        if (registered["source_event_id"] != native["source_event_id"] or
                registered["fixture_facts"] != {key: native[key] for key in FACTS}):
            raise ValueError("Registered physical fixture differs from original verified bytes")
        if parse_time(registered["first_registered_at"]) >= as_of:
            raise ValueError("Training registration must strictly precede fitting")
        run_hash = runtime_digest(run)
        if cohort is None:
            exposures = [(plan, row) for plan, row in legacy if row["run_id"] == run["run_id"]
                         and row["run_sha256"] == run_hash and row["match_id"] == mid
                         and parse_time(run["finished_at"]) <= parse_time(plan["created_at"]) < as_of]
            if not any(any(m["match_id"] == mid and m["source_event_id"] == native["source_event_id"]
                           and m["identity_binding_sha256"] == binding["binding_sha256"]
                           for m in plan["matches"]) for plan, _ in exposures):
                raise ValueError("Unregistered old runs need explicit raw-bound legacy f_train exposure")
        key = run["run_id"]
        if key in seen and seen[key] != run_hash:
            raise ValueError("Duplicate original run ID has conflicting content")
        if key in seen:
            continue
        seen[key] = run_hash
        for version, proof in _read(folder/"prior-bindings.json").items():
            if version in proofs and proofs[version] != proof:
                raise ValueError("Conflicting original prior bindings")
            proofs[version] = proof
        # Use an existing registered provider ID so the next cohort can resolve
        # f.training_match_ids without a new alias registry or historical rewrite.
        identities[mid] = {"original_match_id": mid, "representative_match_id": min(registered["match_ids"]),
            "source_event_id": native["source_event_id"], "purpose": "f_train"}
        runs.append({"folder": str(folder.resolve()), "run_id": key, "run_sha256": run_hash,
            "match_id": mid, "selected_web_steps": [cycle["step"] for cycle in run["cycles"]
                if (cycle.get("selected_action") or {}).get("tool") in {"read_url", "search_web"}]})
    return runs, proofs, identities, inventory


def _record_key(row):
    return _digest({key: (parse_time(row[key]).isoformat() if key.endswith("_at") else row[key])
                    for key in ("match_id", "prediction_cutoff", "action", "response",
                                "decision_at", "observed_at", "available_at")})


def _complete_run_records(dataset, runs):
    """A missing web episode must not silently shorten a later real history."""
    projected = empirical_risk_records(dataset)
    episodes = defaultdict(list)
    for row in dataset["episodes"]:
        episodes[row["run_id"]].append(row)
    excluded = []
    for run in runs:
        expected, actual = sorted(run["selected_web_steps"]), sorted(row["step"] for row in episodes[run["run_id"]])
        if expected != actual:
            excluded.append({"run_id": run["run_id"], "reason": "incomplete_actual_web_episode_chain",
                             "expected_steps": expected, "reconstructed_steps": actual})
    unsafe = {row["run_id"] for row in excluded}
    forbidden = {_record_key(row) for row in dataset["episodes"] if row["run_id"] in unsafe}
    records = [row for row in projected["records"] if _record_key(row) not in forbidden]
    return records, {**projected["audit"], "complete_chain_excluded_runs": excluded,
                    "complete_chain_output_records": len(records), "complete_chain_records_sha256": _digest(records)}


def _canonical_records(records, identities):
    fitted, outcomes = [], {}
    for original in records:
        identity = identities[original["match_id"]]
        event, outcome = identity["source_event_id"], original["outcome"]
        if type(outcome) is not int or outcome not in (0, 1, 2):
            raise ValueError("Qualified gateway record has no independent H/D/A label")
        if event in outcomes and outcomes[event] != outcome:
            raise ValueError("Conflicting settled outcomes for the same physical fixture")
        outcomes[event] = outcome
        fitted.append({**deepcopy(original), "match_id": identity["representative_match_id"]})
    classes = {str(label): sum(value == label for value in outcomes.values()) for label in (0, 1, 2)}
    reasons = []
    if len(outcomes) < PARAMETERS["min_matches"]:
        reasons.append("fewer_than_12_independent_qualified_physical_matches")
    if any(count < PARAMETERS["min_class_matches"] for count in classes.values()):
        reasons.append("qualified_physical_matches_do_not_cover_all_three_outcomes")
    return fitted, {"qualified_physical_matches": len(outcomes), "class_counts": classes,
        "ready_to_fit": not reasons, "not_ready_reasons": reasons,
        "gate_scope": "necessary_global_sample_gate_not_conditional_support_or_effect_validation"}


def frozen_f_from_runs(run_folders, previous_plans, result_archive, output, *, as_of=None,
                       model_kind="ordered-v1"):
    """Reverify original runs/results, then diagnose or freeze an existing f.

    previous_plans is an explicit sequence of R189 registration dictionaries.
    as_of is the caller's actual operation clock (injectable for offline tests);
    the CLI always reads wall time. There is no arbitrary-records import and no
    configurable relaxation of the existing model's support parameters.
    """
    started = datetime.now(timezone.utc)
    as_of = parse_time(as_of if as_of is not None else started)
    output = Path(output)
    if output.exists():
        raise FileExistsError("Training output exists; use a new immutable output directory")
    if model_kind not in MODEL_KINDS:
        raise ValueError("Explicit ordered-v1 or neighborhood-v2 model kind is required")
    if not isinstance(run_folders, (list, tuple)) or not run_folders:
        raise ValueError("Explicit original run folders are required, never arbitrary training records")
    runs, proofs, identities, inventory = _registered_runs(run_folders, previous_plans, as_of)
    joined = join_settlements([row["folder"] for row in runs], settlements=[], prior_bindings=proofs,
                              as_of=as_of, result_archive=result_archive)
    original_records, projection_audit = _complete_run_records(joined["dataset"], runs)
    fit_records, readiness = _canonical_records(original_records, identities)
    mapping = {"schema": "prospective-text-training-identity-map-v1", "inventory_scope": SCOPE,
        "observation_unit": "canonical_source_event_id_not_run_or_document_version",
        "mappings": [identities[mid] for mid in sorted(identities)], "registered_inventory": inventory,
        "original_records_unchanged": True}
    dataset_version = VERSION+":"+_digest({"records": fit_records, "identity_map": mapping})[:24]
    model = None
    if readiness["ready_to_fit"]:
        model = MODEL_KINDS[model_kind]().fit(fit_records, as_of=as_of, dataset_version=dataset_version,
                                            evidence_origin="real_observational", **PARAMETERS)
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("gateway-dataset.json", joined), ("original-records.json", original_records),
                        ("projection-audit.json", projection_audit), ("fit-records.json", fit_records),
                        ("identity-map.json", mapping), ("registration-inputs.json", previous_plans)):
        _write(output/name, value)
    artifact = None
    if model is not None:
        model_path = output/"frozen-text-model.json"
        model.save(model_path)
        loaded = load_text_model(model_path)
        if loaded.model_version != model.model_version or loaded._payload() != model._payload():
            raise ValueError("Frozen model failed its exact save/load verification")
        artifact = {"file": model_path.name, "path": str(model_path.resolve()),
                    "sha256": sha256(model_path.read_bytes()).hexdigest(), "model_version": model.model_version}
    report = {"schema": VERSION, "status": "frozen" if model else "not_ready", **readiness,
        "as_of": as_of.isoformat(), "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(), "model_kind": model_kind,
        "model": artifact, "dataset_version": dataset_version, "parameters": deepcopy(PARAMETERS),
        "input_runs": runs, "registered_physical_matches": len({row["source_event_id"] for row in identities.values()}),
        "qualified_records": len(fit_records), "evidence_origin": "real_observational",
        "response_kinds": dict(Counter(row["response"].get("kind", "unknown") for row in fit_records)),
        "response_statuses": dict(Counter(row["response"].get("status", "unknown") for row in fit_records)),
        "projection_audit": projection_audit, "model_training_performed": model is not None,
        "production_ready": False, "calibration_status": "not_established", "realized_voi_benefit_proven": False,
        "conditional_support": "runtime_source_action_history_prior_kernel_may_still_return_UNKNOWN",
        "history_scope": "only_actual_same_run_ordered_responses_no_imputed_counterfactuals",
        "inventory_scope": SCOPE, "historical_reconstruction_imported": False,
        "network_requests": 0, "output": str(output.resolve())}
    _write(output/"training-report.json", report)
    return report


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit")
    fit.add_argument("--run", type=Path, action="append", required=True)
    fit.add_argument("--previous-plan", type=Path, action="append", required=True)
    fit.add_argument("--result-archive", type=Path)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--model-kind", choices=MODEL_KINDS, default="ordered-v1")
    args = parser.parse_args(argv)
    try:
        report = frozen_f_from_runs(args.run, [_read(path) for path in args.previous_plan], args.result_archive,
                                    args.output, model_kind=args.model_kind)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(2, "训练入口拒绝："+str(error)+"\n")
    print(json.dumps({key: report[key] for key in ("status", "qualified_physical_matches", "class_counts",
        "qualified_records", "model", "production_ready", "output")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
