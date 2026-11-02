"""Pre-query match-cohort admission for the existing prospective collector.

The caller verifies original fixture bytes first. This module binds that exact
identity and already saved models to a supplied cohort inventory. It neither
trains a model nor certifies an undisclosed global history or an effect claim.
"""
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from football_research.acquisition.state import parse_time
from football_research.acquisition.text_history import load_text_model
from football_research.acquisition.neighborhood_text_risk import load_text_risk_model
from football_research.evaluation.policy_experiment_protocol import audit_holdout_manifest
from .prospective_text import MarketPriorPolicy, _digest

VERSION = "prospective-learning-cohort-v1"
EXPOSURE_VERSION = "prospective-training-exposure-v1"
EVIDENCE_PLAN_VERSION = 'prospective-evidence-only-cohort-v1'
PURPOSES = ("f_train", "calibration", "confirmation")
FACTS = ("home_id", "away_id", "competition", "category", "season", "stage")
SCOPE = "supplied_previously_registered_cohorts_only_not_undisclosed_global_history"


def _text(value, label):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(label+" must be an explicit nonblank string")
    return value


def _bound_matches(initializations, identity_bundle, created):
    if initializations.get("schema") != "prospective-text-initializations-v1" or not initializations.get("selected"):
        raise ValueError("A nonempty existing initialization plan is required")
    bindings = identity_bundle.get("bindings", identity_bundle)
    result, mids, events = [], set(), set()
    for manifest in initializations["selected"]:
        policy = MarketPriorPolicy.from_manifest(manifest)
        mid = policy.model.fixture["match_id"]
        binding = bindings.get(mid)
        if (not isinstance(binding, Mapping) or binding.get("status") != "CONFIRMED"
                or binding.get("match_id") != mid or binding.get("initialization_manifest") != manifest
                or binding.get("binding_sha256") != _digest({k: v for k, v in binding.items() if k != "binding_sha256"})):
            raise ValueError("The cohort requires the unchanged confirmed fixture identity binding")
        if not parse_time(binding["verified_at"]) <= created or not parse_time(policy.model.frozen_at) <= created < parse_time(policy.model.cutoff):
            raise ValueError("Cohort registration needs an already observed, unexpired initialization and identity")
        canonical = binding["canonical"]
        event = _text(canonical["source_event_id"], "canonical source_event_id")
        if mid in mids or event in events:
            raise ValueError("One physical fixture cannot enter the same batch twice")
        if parse_time(canonical["kickoff"]) != parse_time(policy.model.fixture["kickoff"]):
            raise ValueError("Canonical fixture kickoff differs from initialization")
        mids.add(mid)
        events.add(event)
        result.append({"match_id": mid, "source_event_id": event,
            "fixture_facts": {key: _text(canonical[key], key) for key in FACTS},
            "kickoff_at": policy.model.fixture["kickoff"], "prediction_cutoff": policy.model.cutoff,
            "initialization_model_version": policy.model.model_version,
            "identity_binding_sha256": binding["binding_sha256"]})
    return sorted(result, key=lambda row: row["source_event_id"])


def _check_plan(plan):
    if (not isinstance(plan, Mapping) or plan.get("schema") not in {VERSION, EXPOSURE_VERSION, EVIDENCE_PLAN_VERSION}
            or plan.get("purpose") not in PURPOSES or plan.get("inventory_scope") != SCOPE
            or not isinstance(plan.get("matches"), list) or not plan["matches"]
            or not isinstance(plan.get("prior_inventory"), list)):
        raise ValueError("Unsupported prospective learning plan")
    if plan.get("schema") == EXPOSURE_VERSION and (plan["purpose"] != "f_train" or plan.get("artifacts") != {}
            or plan.get("legacy_exposed") is not True):
        raise ValueError("Legacy exposure can only register f_train identities without a retroactive model")
    if plan.get('schema') == EVIDENCE_PLAN_VERSION and (plan['purpose'] != 'f_train'
            or plan.get('artifacts') != {} or plan.get('prior_status') != 'prior_unavailable'
            or plan.get('numeric_training_eligible') is not False):
        raise ValueError('Evidence-only cohorts record exposure without a numerical training prior')
    if plan.get("cohort_id") != plan["schema"]+":"+_digest({k: v for k, v in plan.items() if k != "cohort_id"})[:24]:
        raise ValueError("Prospective learning plan changed after registration")
    parse_time(plan["created_at"])


def _merge_inventory(rows):
    by_event, by_match = {}, {}
    for row in rows:
        event = _text(row["source_event_id"], "inventory source_event_id")
        if row.get("purpose") not in PURPOSES or set(row["fixture_facts"]) != set(FACTS):
            raise ValueError("Invalid registered fixture purpose or facts")
        facts = {key: _text(row["fixture_facts"][key], key) for key in FACTS}
        models = row.get("model_bindings")
        expected = set() if row["purpose"] == "f_train" else {"text_model"}
        if row["purpose"] == "confirmation":
            expected.add("text_risk_model")
        if not isinstance(models, Mapping) or set(models) != expected or any(
                set(spec) != {"sha256", "model_version"} or not _text(spec["model_version"], "model version")
                or len(spec["sha256"]) != 64 for spec in models.values()):
            raise ValueError("Inventory must preserve the models used to collect this cohort")
        stamp = parse_time(row["first_registered_at"])
        ids = row.get("match_ids")
        if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids):
            raise ValueError("An explicit distinct provider match identity inventory is required")
        old = by_event.get(event)
        if old and (old["purpose"] != row["purpose"] or old["fixture_facts"] != facts or old["model_bindings"] != models):
            raise ValueError("Canonical fixture cannot change cohort purpose, models or team/competition facts")
        for mid in ids:
            _text(mid, "provider match_id")
            if mid in by_match and by_match[mid] != event:
                raise ValueError("One provider match ID maps to different physical fixtures")
            by_match[mid] = event
        by_event[event] = {"source_event_id": event, "purpose": row["purpose"], "fixture_facts": facts,
            "model_bindings": deepcopy(models),
            "match_ids": sorted(set(ids+(old["match_ids"] if old else []))),
            "first_registered_at": min(stamp, parse_time(old["first_registered_at"]) if old else stamp).isoformat()}
    return [by_event[key] for key in sorted(by_event)]


def _own_inventory(plan):
    return [{"source_event_id": row["source_event_id"], "purpose": plan["purpose"],
             "fixture_facts": row["fixture_facts"], "match_ids": [row["match_id"]],
             "model_bindings": {key: {name: spec[name] for name in ("model_version", "sha256")}
                                for key, spec in plan["artifacts"].items()},
             "first_registered_at": plan["created_at"]} for row in plan["matches"]]


def _previous_inventory(plans, created):
    if not isinstance(plans, (list, tuple)):
        raise ValueError("previous_plans must be an explicit sequence of registered plan dictionaries")
    rows = []
    for previous in plans:
        # New context cohorts remain visible to later A registrations too;
        # otherwise separate entry points would have incompatible inventories.
        from football_research.acquisition.context_gain import REGISTRATION as CONTEXT_B, _registration as check_b
        from football_research.acquisition.context_confirmation import REGISTRATION as CONTEXT_C, _registration as check_c
        from .official_context_registration import CALIBRATION as OFFICIAL_B, CONFIRMATION as OFFICIAL_C, _registration as check_official
        if previous.get('schema') == CONTEXT_B:
            check_b(previous)
        elif previous.get('schema') == CONTEXT_C:
            check_c(previous)
        elif previous.get('schema') in {OFFICIAL_B, OFFICIAL_C}:
            check_official(previous)
        else:
            _check_plan(previous)
        if parse_time(previous["created_at"]) > created:
            raise ValueError("A previous cohort was not registered at this plan's creation")
        rows.extend(previous["prior_inventory"]+_own_inventory(previous))
    return _merge_inventory(rows)


def _artifact(path, model, created, *, risk=False):
    path = Path(path).resolve()
    metadata = model.manifest if risk else model.metadata
    if metadata.get("evidence_origin") != "real_observational":
        raise ValueError("Prospective learning requires real_observational models; synthetic models are offline only")
    if parse_time(metadata["training_as_of"]) > created:
        raise ValueError("A frozen model cannot be registered before its training cutoff")
    return {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest(),
            "model_version": model.model_version, "training_as_of": metadata["training_as_of"],
            "registered_at": created.isoformat()}


def _load_artifacts(artifacts, purpose, created, artifact_paths=None):
    required = set() if purpose == "f_train" else {"text_model"}
    if purpose == "confirmation":
        required.add("text_risk_model")
    if not isinstance(artifacts, Mapping) or set(artifacts) != required:
        raise ValueError("f_train needs no model, calibration needs f, confirmation needs the same f plus critic")
    overrides = {} if artifact_paths is None else artifact_paths
    if not isinstance(overrides, Mapping) or set(overrides)-required:
        raise ValueError("Archived artifact overrides must name only this plan's models")
    paths = {key: Path(overrides.get(key, spec["path"])) for key, spec in artifacts.items()}
    for key, spec in artifacts.items():
        if (parse_time(spec["registered_at"]) != created or not Path(spec["path"]).is_absolute()
                or sha256(paths[key].read_bytes()).hexdigest() != spec["sha256"]):
            raise ValueError("Registered model artifact changed or its registration clock is inconsistent")
    text = load_text_model(paths["text_model"]) if required else None
    risk = load_text_risk_model(paths["text_risk_model"], text_model=text) if "text_risk_model" in required else None
    for key, model in (("text_model", text), ("text_risk_model", risk)):
        if model is not None:
            observed = _artifact(paths[key], model, created, risk=key == "text_risk_model")
            observed["path"] = artifacts[key]["path"]
            if artifacts[key] != observed:
                raise ValueError("Registered model metadata or content differs from the saved artifact")
    return text, risk


def _training_events(model, inventory, purpose, current_events, *, risk=False, text_binding=None):
    if model is None:
        return set()
    metadata = model.manifest if risk else model.metadata
    mapping = {mid: row for row in inventory for mid in row["match_ids"]}
    events = set()
    for mid in metadata["training_match_ids"]:
        if mid not in mapping:
            raise ValueError("unknown_training_identity_inventory: "+str(mid))
        row = mapping[mid]
        if row["purpose"] != purpose:
            raise ValueError("Frozen model training match belongs to the wrong cohort purpose")
        if parse_time(row["first_registered_at"]) >= parse_time(metadata["training_as_of"]):
            raise ValueError("A training fixture was not registered before model fitting")
        if risk and row["model_bindings"]["text_model"] != text_binding:
            raise ValueError("Calibration data were collected with a different frozen text model")
        events.add(row["source_event_id"])
    if risk:
        for record in model.raw_records:
            registered = parse_time(mapping[record["match_id"]]["first_registered_at"])
            if any(not registered < parse_time(step["decision_at"]) for step in record["history"]+[record]):
                raise ValueError("Calibration registration must precede every actual training query")
    if events & current_events:
        raise ValueError("Current physical fixture was used to train a frozen model")
    return events


def prepare_learning_plan(initialization_plan, *, identity_bundle, purpose, created_at,
                          previous_plans=(), text_model_path=None, text_risk_model_path=None):
    """Register existing bytes before acquisition; perform no I/O except reads.

    created_at is the actual registration clock supplied by the orchestrator,
    not a reconstructed historical date. Original identity bytes are checked
    by the existing collector gateway, not by trusting these low-level flags.
    """
    if purpose not in PURPOSES:
        raise ValueError("Unknown acquisition learning purpose")
    created = parse_time(created_at)
    if (purpose == "f_train" and (text_model_path is not None or text_risk_model_path is not None)
            or purpose != "f_train" and text_model_path is None
            or (purpose == "confirmation") != (text_risk_model_path is not None)):
        raise ValueError("Model paths must match the declared learning purpose")
    artifacts = {}
    if text_model_path is not None:
        text = load_text_model(text_model_path)
        artifacts["text_model"] = _artifact(text_model_path, text, created)
        if text_risk_model_path is not None:
            risk = load_text_risk_model(text_risk_model_path, text_model=text)
            artifacts["text_risk_model"] = _artifact(text_risk_model_path, risk, created, risk=True)
    plan = {"schema": VERSION, "purpose": purpose, "created_at": created.isoformat(),
        "matches": _bound_matches(initialization_plan, identity_bundle, created),
        "prior_inventory": _previous_inventory(previous_plans, created), "artifacts": artifacts,
        "inventory_scope": SCOPE,
        "observation_unit": "canonical_source_event_id_not_run_or_document_version",
        "model_registration_scope": "artifact_bytes_observed_at_plan_creation_not_backdated_physical_fit_claim",
        "supervisor_only": True, "independent_effect_confirmed": False}
    plan["cohort_id"] = VERSION+":"+_digest(plan)[:24]
    _admit(plan, initialization_plan, identity_bundle, created, previous_plans)
    return plan


def _admit(plan, initialization_plan, identity_bundle, as_of, previous_plans, artifact_paths=None):
    _check_plan(plan)
    if plan["schema"] != VERSION:
        raise ValueError("Legacy training exposure is inventory, never a prospective acquisition plan")
    created = parse_time(plan["created_at"])
    if created > as_of:
        raise ValueError("Acquisition plan is from the future")
    matches = _bound_matches(initialization_plan, identity_bundle, created)
    if matches != plan["matches"]:
        raise ValueError("Current batch differs from the pre-query match cohort")
    if any(as_of >= parse_time(row["prediction_cutoff"]) for row in matches):
        raise ValueError("Learning cohort prematch horizon has expired")
    prior = _merge_inventory(plan["prior_inventory"]+_previous_inventory(previous_plans, created))
    if any(parse_time(row["first_registered_at"]) > created for row in prior):
        raise ValueError("Prior inventory includes a future registration")
    inventory = _merge_inventory(prior+_own_inventory(plan))
    text, risk = _load_artifacts(plan["artifacts"], plan["purpose"], created, artifact_paths)
    current = {row["source_event_id"] for row in matches}
    f_events = _training_events(text, prior, "f_train", current)
    text_binding = ({key: plan["artifacts"]["text_model"][key] for key in ("model_version", "sha256")}
                    if text is not None else None)
    g_events = _training_events(risk, prior, "calibration", current, risk=True, text_binding=text_binding)
    if f_events & g_events:
        raise ValueError("Predictor and critic training share a physical fixture")
    partitions = {name: sorted(row["source_event_id"] for row in inventory if row["purpose"] == purpose)
                  for name, purpose in zip(("train", "validation", "test"), PURPOSES)}
    audit = audit_holdout_manifest(partitions, prior_calibration_match_ids=sorted(f_events | g_events),
        prior_policy_search_match_ids=[], episode_match_ids=sorted(current), declared_independent=False)
    if any(item["code"] not in {"EMPTY_PARTITION", "INDEPENDENCE_NOT_DECLARED"} for item in audit["issues"]):
        raise ValueError("Registered match inventory violates the existing holdout protocol")
    binding = {**deepcopy(plan), "holdout_audit": audit}
    by_match = {row["match_id"]: {**deepcopy(row), "cohort_id": plan["cohort_id"],
        "purpose": plan["purpose"], "created_at": plan["created_at"], "supervisor_only": True,
        "text_model_version": text.model_version if text else None,
        "text_risk_model_version": risk.model_version if risk else None} for row in matches}
    return {"binding": binding, "by_match": by_match, "text_model": text, "risk_model": risk}


def validate_learning_plan(learning_plan, *, initialization_plan, identity_bundle, as_of,
                           previous_plans=(), artifact_paths=None):
    """Validate immediately before collection and return the exact loaded models."""
    _check_plan(learning_plan)
    as_of = parse_time(as_of)
    if not parse_time(learning_plan["created_at"]) < as_of:
        raise ValueError("Learning plan/model registration must strictly precede actual acquisition")
    return _admit(learning_plan, initialization_plan, identity_bundle, as_of, previous_plans, artifact_paths)


def import_training_exposure(run_folders, *, created_at):
    """Import only raw-verified old runs as exposed f-training identities.

    This new registration does not qualify any response or label for training,
    reconstruct a prior registration clock, or create selection propensities.
    """
    from .prospective_text import verify_run_identity, verify_prior_archive
    from football_agent.learned_voi_runtime import digest as runtime_digest
    created = parse_time(created_at)
    if not isinstance(run_folders, (list, tuple)) or not run_folders:
        raise ValueError("Explicit original run folders are required")
    matches, runs, seen = [], [], {}
    for folder in map(Path, run_folders):
        config = json.loads((folder/"configuration.json").read_text(encoding="utf-8"))
        run = json.loads((folder/"run.json").read_text(encoding="utf-8"))
        if (run["configuration_digest"] != runtime_digest(config) or run["run_id"] != config["run_id"]
                or run["schema"] != config["schema"]):
            raise ValueError("Legacy run/configuration binding is inconsistent")
        if not parse_time(config["created_at"]) <= parse_time(run["finished_at"]) <= created:
            raise ValueError("Training exposure can only be registered after the actual run finished")
        if config.get('schema') == 'learned-voi-open-execution-v5':
            raise ValueError('Evidence-only exposure already has its own registration; a missing numerical prior cannot be imported')
        if config.get("supervisor_metadata", {}).get("context_calibration") is not None:
            raise ValueError("Context calibration exposure cannot be reclassified as f_train")
        if config.get("supervisor_metadata", {}).get("context_confirmation") is not None:
            raise ValueError("Context confirmation exposure cannot be reclassified as f_train")
        existing = config.get("supervisor_metadata", {}).get("learning_cohort")
        if existing is not None and existing.get("purpose") != "f_train":
            raise ValueError("Calibration/confirmation exposure cannot be reclassified as f_train")
        binding = verify_run_identity(folder, config)
        if binding is None:
            raise ValueError("Legacy exposure needs raw-verified independent fixture identity")
        verify_prior_archive(binding["initialization_manifest"], folder/"prior-archive")
        native = binding["canonical"]
        policy = MarketPriorPolicy.from_manifest(binding["initialization_manifest"])
        record = {"match_id": binding["match_id"], "source_event_id": native["source_event_id"],
            "fixture_facts": {key: native[key] for key in FACTS}, "kickoff_at": native["kickoff"],
            "prediction_cutoff": policy.model.cutoff, "initialization_model_version": policy.model.model_version,
            "identity_binding_sha256": binding["binding_sha256"]}
        key, content = run["run_id"], runtime_digest(run)
        if key in seen and seen[key] != content:
            raise ValueError("Duplicate legacy run ID has conflicting content")
        if key in seen:
            continue
        seen[key] = content
        matches.append(record)
        runs.append({"run_id": key, "folder": str(folder.resolve()), "run_sha256": content,
                     "finished_at": run["finished_at"], "match_id": record["match_id"]})
    plan = {"schema": EXPOSURE_VERSION, "purpose": "f_train", "created_at": created.isoformat(),
        "matches": matches, "prior_inventory": [], "artifacts": {}, "legacy_exposed": True,
        "inventory_scope": SCOPE, "runs": runs, "supervisor_only": True,
        "observation_unit": "canonical_source_event_id_not_run_or_document_version",
        "historical_preregistration_claimed": False, "selection_probabilities_imputed": False,
        "training_eligibility": "existing_raw_response_and_independent_settlement_gateway_still_required"}
    inventory = _merge_inventory(_own_inventory(plan))
    plan["unique_matches"] = len(inventory)
    plan["cohort_id"] = EXPOSURE_VERSION+":"+_digest(plan)[:24]
    return plan


def main(argv=None):
    """Write one actual-time plan exclusively; never perform network requests."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--initialization-plan", "--initplan", type=Path, required=True)
    prepare.add_argument("--identity", type=Path, required=True)
    prepare.add_argument("--purpose", choices=PURPOSES, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--previous-plan", type=Path, action="append", default=[])
    prepare.add_argument("--text-model", type=Path)
    prepare.add_argument("--text-risk-model", type=Path)
    legacy = commands.add_parser("import-training-exposure")
    legacy.add_argument("--run", type=Path, action="append", required=True)
    legacy.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("A learning plan already exists; registration cannot be overwritten")
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    if args.command == "prepare":
        plan = prepare_learning_plan(read(args.initialization_plan), identity_bundle=read(args.identity),
            purpose=args.purpose, created_at=datetime.now(timezone.utc),
            previous_plans=[read(path) for path in args.previous_plan],
            text_model_path=args.text_model, text_risk_model_path=args.text_risk_model)
    else:
        plan = import_training_exposure(args.run, created_at=datetime.now(timezone.utc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status": "registered_not_collected", "cohort_id": plan["cohort_id"],
        "purpose": plan["purpose"], "unique_matches": len({row["source_event_id"] for row in plan["matches"]}),
        "registered_runs": len(plan.get("runs", [])), "output": str(args.output.resolve()),
        "network_requests": 0, "model_training_performed": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    main()
