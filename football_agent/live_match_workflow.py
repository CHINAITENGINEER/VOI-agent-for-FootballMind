"""Persistent match-only jobs over the existing prospective research pipeline.

One invocation advances work that is due now. It never waits for kickoff or
launches a background worker. Completed I/O is adopted after verification;
ambiguous partial acquisition is retained for reconciliation, not repeated.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from filelock import FileLock

from football_research.acquisition.state import parse_time
from football_research.data_adapters.prospective_learning import prepare_learning_plan, _previous_inventory
from football_research.data_adapters.prospective_text import (
    collect_prepared_batch, join_settlements, verify_prior_archive,
    verify_run_identity, verify_run_learning,
)
from football_research.data_adapters.verified_fixtures import (
    verify_identity_binding, main as capture_results,
)
from football_spider.storage.raw_archive import atomic_write, json_bytes, RawArchive
from .learned_voi_runtime import digest
from .match_source_profiles import resolve_verified_profile

ROOT = Path(__file__).resolve().parents[1]
VERSION = "footballmind-live-job-v1"
SETTINGS_VERSION = "footballmind-live-settings-v1"
TERMINAL = {"unsupported", "not_found", "ambiguous", "not_prematch", "unavailable",
    "budget_exhausted", "expired", "training_eligible", "settled_ineligible", "result_unconfirmed", "duplicate_fixture",
    "evidence_settled_prior_unavailable"}


def _now():
    return datetime.now(timezone.utc)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _stage(output, state, status, *, reason=None, **extra):
    state.update(status=status, updated_at=_now().isoformat(), **extra)
    if reason is not None:
        state["reason"] = reason
    else:
        state.pop("reason", None)
    state["transitions"].append({"status": status, "at": state["updated_at"], "reason": reason})
    atomic_write(output / "state.json", json_bytes(state))


def load_settings(path=None):
    settings = _read(path or ROOT / "config/live-workflow.json")
    if settings.get("schema") != SETTINGS_VERSION or settings.get("purpose") != "f_train":
        raise ValueError("This live job version supports the registered f_train stage only")
    if 'news_projection_version' in settings and (type(settings['news_projection_version']) is not int
            or settings['news_projection_version'] not in {5,6}):
        raise ValueError('Explicit news projection version must be 5 or 6')
    if 'official_horizon_days' in settings and (type(settings['official_horizon_days']) is not int
            or settings['official_horizon_days'] not in {7,14}):
        raise ValueError('Explicit official preparation horizon must be 7 or 14 days')
    if 'official_schedule_group' in settings and settings['official_schedule_group'] not in ('2', '3'):
        raise ValueError('Explicit official schedule group must be the registered string 2 or 3')
    for key, low, high in (("preparation_http_budget", 1, 12), ("actor_requests_per_match", 1, 4),
        ("settlement_delay_minutes", 110, 1440), ("settlement_retry_minutes", 30, 1440),
        ("settlement_max_attempts", 1, 3)):
        if type(settings.get(key)) is not int or not low <= settings[key] <= high:
            raise ValueError("Invalid live workflow setting: " + key)
    if type(settings.get("seed")) is not int or settings.get("route") not in {"direct", "environment"}:
        raise ValueError("Explicit seed and transport route required")
    if not isinstance(settings.get("previous_learning_plans"), list):
        raise ValueError("Learning history must be an explicit deployment inventory")
    if not isinstance(settings.get("task_root"), str) or not settings["task_root"].strip():
        raise ValueError("A persistent task registry directory is required")
    if type(settings.get("official_evidence_fallback",False)) is not bool:
        raise ValueError("official_evidence_fallback must be an explicit boolean when supplied")
    return settings


def start_match(query, output, *, settings_path=None, match_date=None):
    """Start a new persistent job. The only task-specific input is the match."""
    if not isinstance(query, (str, dict)) or not query:
        raise ValueError("A match ID or ordered team pair is required")
    settings = load_settings(settings_path)
    if match_date is not None:
        # Validate before persisting a job; date remains a user identity hint.
        datetime.strptime(match_date, "%Y-%m-%d")
    _prepare_query({"query": query, "date": match_date, "input_timezone": "Asia/Shanghai"})
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    with FileLock(str(output / ".lock"), timeout=0):
        previous = []
        registry_root = Path(settings["task_root"])
        if not registry_root.is_absolute():
            registry_root = ROOT / registry_root
        registry_root.mkdir(parents=True, exist_ok=True)
        with FileLock(str(registry_root / ".registry.lock"), timeout=0):
            registry_path = registry_root / "registry.json"
            registry = _read(registry_path) if registry_path.exists() else {"tasks": []}
            history = list(settings["previous_learning_plans"])
            for existing in registry["tasks"]:
                known = Path(existing)
                if (known / "learning-plan.json").exists():
                    _, known_state = _load_job(known)
                    if known_state.get("learning_plan_sha256"):
                        history.append(str(known / "learning-plan.json"))
            if str(output) not in registry["tasks"]:
                registry["tasks"].append(str(output))
            atomic_write(registry_path, json_bytes(registry))
        seen = set()
        for reference in history:
            source = Path(reference)
            if not source.is_absolute():
                source = ROOT / source
            plan = _read(source)
            key = digest(plan)
            if key in seen:
                continue
            seen.add(key)
            target = output / "history" / f"plan-{len(previous):03}.json"
            _write(target, plan)
            previous.append({"path": target.relative_to(output).as_posix(), "sha256": _hash(target)})
        request = {"schema": VERSION, "query": query, "date": match_date,
            "input_timezone": "Asia/Shanghai", "created_at": _now().isoformat(),
            "settings": settings, "previous_plans": previous}
        _write(output / "request.json", request)
        state = {"schema": VERSION, "request_sha256": _hash(output / "request.json"),
            "transitions": [], "settlement_attempts": 0, "numeric_voi_status": "UNKNOWN_pending_real_response_training",
            "realized_voi_benefit_proven": False, "background_worker_started": False}
        _stage(output, state, "new")
    return resume_match(output)


def _load_job(output):
    request, state = _read(output / "request.json"), _read(output / "state.json")
    if state.get("schema") != VERSION or state.get("request_sha256") != _hash(output / "request.json"):
        raise ValueError("Persistent job input changed")
    for ref in request["previous_plans"]:
        if _hash(output / ref["path"]) != ref["sha256"]:
            raise ValueError("Registered learning history changed")
    for name, expected in state.get("prepared_sha256", {}).items():
        if _hash(output / name) != expected:
            raise ValueError("Prepared input changed: " + name)
    if state.get("learning_plan_sha256") and _hash(output / "learning-plan.json") != state["learning_plan_sha256"]:
        raise ValueError("Learning registration changed")
    for name, expected in state.get("actor_sha256", {}).items():
        if _hash(output / name) != expected:
            raise ValueError("Completed actor archive changed: " + name)
    for name, expected in state.get("result_sha256", {}).items():
        if _hash(output / name) != expected:
            raise ValueError("Completed result archive changed: " + name)
    return request, state


def _prepare_query(request):
    query = request["query"]
    if isinstance(query, dict):
        return query
    if query.isdigit() or query.startswith(("spftll:match:","veikkausliiga:2026:match:","palloliitto:match:")):
        return query
    for separator in (" vs ", " 对阵 ", "对阵", " v "):
        if separator in query:
            home, away = query.split(separator, 1)
            if home.strip() and away.strip():
                return {"home": home.strip(), "away": away.strip(), "date": request["date"],
                    "timezone": request["input_timezone"]}
    raise ValueError("Use the match ID or 'home vs away' with an optional --date")


def _prepared(output):
    base = output / "preparation"
    plan, bundle = _read(base / "initializations.json"), _read(base / "identity-bundle.json")
    if len(plan.get("selected", [])) != 1 or len(bundle.get("bindings", {})) != 1:
        raise ValueError("A live task must bind exactly one physical match")
    manifest = plan["selected"][0]
    mid = manifest["market_initialization"]["fixture"]["match_id"]
    binding = bundle["bindings"][mid]
    if binding["initialization_manifest"] != manifest:
        raise ValueError("Prepared identity and market initialization disagree")
    verify_prior_archive(manifest, base / "market-archive")
    verify_identity_binding(binding, archive_root=base / "identity-archive", as_of=_now())
    profile = resolve_verified_profile(binding, version=(4 if binding.get("schema") == "finland-stage-fixture-identity-v1" else 3))
    return plan, bundle, mid, profile


def _collection(output, request, state):
    plan, bundle, mid, profile = _prepared(output)
    prepared = _read(output / "preparation/prepare-result.json")
    prior = plan["selected"][0]["market_initialization"]
    now = _now()
    if now >= parse_time(prior["prediction_cutoff"]):
        _stage(output, state, "expired", reason="forecast_cutoff_elapsed_before_acquisition")
        return
    if (now - parse_time(prior["quote"]["observed_at"])).total_seconds() > 3600:
        _stage(output, state, "expired", reason="prepared_market_snapshot_older_than_one_hour")
        return
    history = [output / ref["path"] for ref in request["previous_plans"]]
    learning_path = output / "learning-plan.json"
    if not learning_path.exists():
        registry_root = Path(request["settings"]["task_root"])
        if not registry_root.is_absolute():
            registry_root = ROOT / registry_root
        registry_root.mkdir(parents=True, exist_ok=True)
        with FileLock(str(registry_root / ".registry.lock"), timeout=0):
            event = bundle["bindings"][mid]["canonical"]["source_event_id"]
            previous = [_read(path) for path in history]
            registered = {row["source_event_id"] for row in _previous_inventory(previous, now)}
            registry_path = registry_root / "registry.json"
            if registry_path.exists():
                for known in _read(registry_path)["tasks"]:
                    known = Path(known)
                    if known != output and (known / "learning-plan.json").exists():
                        other = _read(known / "learning-plan.json")
                        registered.update(row["source_event_id"] for row in _previous_inventory([other], now))
            if event in registered:
                _stage(output, state, "duplicate_fixture", reason="physical_fixture_already_registered_no_additional_actor_budget",
                    match_id=mid, source_event_id=event)
                return
            learning = prepare_learning_plan(plan, identity_bundle=bundle, purpose="f_train", created_at=now,
                previous_plans=previous)
            _write(learning_path, learning)
    _stage(output, state, "registered", match_id=mid,
        source_event_id=bundle["bindings"][mid]["canonical"]["source_event_id"],
        learning_plan_sha256=_hash(learning_path))
    config = request["settings"]
    batch = output / "analysis"
    _stage(output, state, "collecting", actor_intent_at=_now().isoformat())
    try:
        collect_prepared_batch(output / "preparation/initializations.json", output=batch,
            archive_root=output / "preparation/market-archive",
            identity_bundle=output / "preparation/identity-bundle.json", identity_archive=output / "preparation/identity-archive",
            allowed_hosts=profile["allowed_hosts"], seed_urls=(),
            source_profile_version=(4 if bundle["bindings"][mid].get("schema") == "finland-stage-fixture-identity-v1" else 3),
            news_provider="bing_rss", news_projection_version=config.get('news_projection_version',prepared["collection"]["news_projection_version"]),
            evidence_version="match-evidence-extraction-v5",
            candidate_discovery_version=prepared["collection"]["candidate_discovery_version"],
            requests_per_match=config["actor_requests_per_match"],
            seed=config["seed"], route=config["route"], learning_plan_path=learning_path,
            news_route_overrides=({host: mode for host, mode in config["news_route_overrides"].items()
                                  if host in profile["allowed_hosts"]} if "news_route_overrides" in config else None),
            previous_learning_plan_paths=history)
    except Exception as error:
        _stage(output, state, "reconciliation_required", reason="acquisition_incomplete_never_automatically_repeat",
            failed_stage="collecting", error_type=type(error).__name__, error_message=str(error))
        return
    _adopt_collection(output, request, state)


def _adopt_collection(output, request, state):
    """Accept a complete batch after a lost process-level completion marker."""
    try:
        _verify_and_adopt_collection(output, request, state)
    except (ValueError, OSError, KeyError, TypeError, StopIteration) as error:
        _stage(output, state, "reconciliation_required", reason="actor_archive_verification_failed_no_repeat",
            failed_stage="collecting", error_type=type(error).__name__, error_message=str(error))


def _verify_and_adopt_collection(output, request, state):
    batch = output / "analysis"
    if not (batch / "batch-report.json").exists():
        _stage(output, state, "reconciliation_required", reason="partial_actor_archive_requires_reconciliation",
            failed_stage="collecting")
        return
    report = _read(batch / "batch-report.json")
    if report.get("unique_matches") != 1 or len(report.get("runs", [])) != 1:
        raise ValueError("Incomplete or inconsistent single-match batch")
    mid = state["match_id"]
    folder = batch / ("match-" + mid.rsplit(":", 1)[-1])
    run, config = _read(folder / "run.json"), _read(folder / "configuration.json")
    if (report["runs"][0]["match_id"] != mid or run["configuration_digest"] != digest(config)
            or config["initial_state"]["match_id"] != mid
            or report["http_requests"] != run["measured_resources"]["requests"]
            or report["http_requests"] > request["settings"]["actor_requests_per_match"]):
        raise ValueError("Saved actor summary disagrees with the original run")
    if config["supervisor_metadata"]["input_bindings"]["initialization_plan_sha256"] != _hash(output / "preparation/initializations.json"):
        raise ValueError("Actor belongs to a different preparation plan")
    verify_run_identity(folder, config)
    verify_run_learning(folder, config)
    verify_prior_archive(config["policy_manifest"], folder / "prior-archive")
    for obs in RawArchive(folder / "source-archive").observations():
        RawArchive(folder / "source-archive").read_body(obs)
    terminal = run.get("terminal_selection", {})
    if "reconciliation" in str(terminal.get("reason", "")):
        _stage(output, state, "reconciliation_required", reason=terminal["reason"], failed_stage="collecting")
        return
    if not (folder / "analysis.json").is_file() or not (folder / "report.md").is_file():
        raise ValueError("Completed analysis report is missing")
    # Reuse the real zero-network replayer: it checks every persisted cycle,
    # response projection, action selection, cost settlement and final state.
    from .learned_voi_replay import replay_saved_run
    with TemporaryDirectory(prefix="actor-verification-", dir=output) as temporary:
        verification = replay_saved_run(folder, output=Path(temporary) / "replay", project_root=ROOT)
    kickoff = parse_time(config["initial_state"]["base"]["kickoff"])
    due = kickoff + timedelta(minutes=request["settings"]["settlement_delay_minutes"])
    _stage(output, state, "awaiting_result", actor_requests=report["http_requests"],
        run_directory=folder.relative_to(output).as_posix(), result_due_at=due.isoformat(),
        actor_sha256={p.relative_to(output).as_posix(): _hash(p)
            for p in [*sorted(p for p in folder.rglob("*") if p.is_file()), batch / "batch-report.json"]},
        actor_verification={"identical_replay": verification["identical_replay"],
            "raw_responses_reprojected": verification["raw_responses_reprojected"], "network_requests": 0},
        report= (folder / "report.md").relative_to(output).as_posix())


def _settle(output, request, state):
    now = _now()
    if now < parse_time(state["result_due_at"]):
        return
    folder = output / state["run_directory"]
    binding = verify_run_identity(folder, _read(folder / "configuration.json"))
    finnish = binding.get("schema") in {"finland-club-fixture-identity-v1", "finland-stage-fixture-identity-v1"}
    if not finnish and not binding["canonical"]["source_event_id"].startswith("uefa:match:"):
        _stage(output, state, "result_unconfirmed", reason="result_capture_not_registered_for_this_source")
        return
    attempt = state["settlement_attempts"] + 1
    settings = request["settings"]
    if attempt > settings["settlement_max_attempts"]:
        _stage(output, state, "result_unconfirmed", reason="bounded_result_attempts_exhausted")
        return
    destination = output / "results" / f"attempt-{attempt:02}"
    _stage(output, state, "settling", settlement_attempts=attempt,
        settlement_path=destination.relative_to(output).as_posix(), settlement_intent_at=now.isoformat())
    try:
        if finnish:
            from football_research.data_adapters.finland_result_capture import capture_finland_result
            capture_finland_result(binding, identity_archive_root=output / "preparation/identity-archive",
                output=destination, clock=_now)
        else:
            capture_results(["capture-results", "--identity-bundle", str(output / "preparation/identity-bundle.json"),
                "--identity-archive", str(output / "preparation/identity-archive"), "--output", str(destination),
                "--result-source", "official-api", "--route", settings["route"], "--max-http-requests", "1"])
    except Exception as error:
        _stage(output, state, "reconciliation_required", reason="result_acquisition_incomplete_no_automatic_retry",
            failed_stage="settling", error_type=type(error).__name__)
        return
    _adopt_settlement(output, request, state)


def _adopt_settlement(output, request, state):
    try:
        _verify_and_adopt_settlement(output, request, state)
    except (ValueError, OSError, KeyError, TypeError) as error:
        _stage(output, state, "reconciliation_required", reason="result_archive_verification_failed_no_repeat",
            failed_stage="settling", error_type=type(error).__name__, error_message=str(error))


def _verify_and_adopt_settlement(output, request, state):
    destination = output / state["settlement_path"]
    if not (destination / "capture-receipts.json").exists():
        _stage(output, state, "reconciliation_required", reason="result_receipt_missing_no_repeat", failed_stage="settling")
        return
    receipts = _read(destination / "capture-receipts.json")
    archive = RawArchive(destination)
    observations = archive.observations()
    folder = output / state["run_directory"]
    binding = verify_run_identity(folder, _read(folder / "configuration.json"))
    finnish = binding.get("schema") in {"finland-club-fixture-identity-v1", "finland-stage-fixture-identity-v1"}
    if finnish:
        from football_research.data_adapters.finland_result_capture import verify_capture_receipts
        verify_capture_receipts(binding, destination, as_of=_now())
        if not receipts['receipts']:
            raise ValueError('Due settlement adoption requires an actual result request')
    elif (receipts["http_requests_actual"] != 1 or receipts.get("max_http_requests") != 1
            or len(receipts.get("receipts", [])) != 1 or len(observations) != 1):
        raise ValueError("Result capture must contain exactly one actual request")
    event_id = binding["canonical"]["source_event_id"]
    expected_url = "https://match.uefa.com/v5/matches/" + event_id.rsplit(":", 1)[-1]
    receipt = receipts["receipts"][0]
    observation = next((item for item in observations if item.id == receipt['observation_id']), None)
    if observation is None:
        raise ValueError('Result receipt refers to an absent raw response')
    if not finnish and (not event_id.startswith("uefa:match:") or receipt.get("http_attempts") != 1
            or receipt.get("observation_id") != observation.id
            or receipt.get("raw_sha256") != observation.content_hash
            or receipt.get("url") != observation.endpoint or observation.endpoint != expected_url
            or parse_time(receipt["observed_at"]) != observation.observed_at
            or parse_time(receipt["requested_at"]) != parse_time(observation.context["requested_at"])
            or not parse_time(receipt["requested_at"]) <= observation.observed_at <= _now()):
        raise ValueError("Result receipt does not identify its single archived response")
    archive.read_body(observation)
    joined = join_settlements([folder], settlements=[], prior_bindings=_read(folder / "prior-bindings.json"),
        as_of=_now(), result_archive=destination)
    audit_path = destination / "training-qualification.json"
    # Recompute now rather than copying an earlier eligibility claim.
    atomic_write(audit_path, json_bytes(joined))
    checks = joined["independent_result_verification"]
    if len(checks) != 1:
        raise ValueError("Result qualification must contain exactly one original fixture")
    counts = sum(_read(p)["http_requests_actual"] for p in (output / "results").glob("*/capture-receipts.json"))
    common = {"result_requests": counts, "qualification_path": audit_path.relative_to(output).as_posix(),
        "qualified_response_rows": len(joined["training_records"]),
        "result_sha256": {**state.get("result_sha256", {}), **{
            p.relative_to(output).as_posix(): _hash(p) for p in sorted(destination.rglob("*")) if p.is_file()}}}
    if all(check["status"] == "VERIFIED" for check in checks):
        _stage(output, state, "training_eligible" if joined["training_records"] else "settled_ineligible", **common)
    elif state["settlement_attempts"] >= request["settings"]["settlement_max_attempts"]:
        _stage(output, state, "result_unconfirmed", reason="bounded_result_attempts_exhausted", **common)
    else:
        due = _now() + timedelta(minutes=request["settings"]["settlement_retry_minutes"])
        _stage(output, state, "awaiting_result", reason="independent_result_not_final_or_unavailable",
            result_due_at=due.isoformat(), **common)


def resume_match(output, *, settle=False):
    """Advance now; no hidden process, sleep or unrecorded I/O retry."""
    output = Path(output).resolve()
    with FileLock(str(output / ".lock"), timeout=0):
        request, state = _load_job(output)
        if state["status"] in TERMINAL:
            return _summary(output, state)
        from . import evidence_live_workflow
        if evidence_live_workflow.handles(request,state):
            return evidence_live_workflow.advance(output,request,state,settle=settle)
        if state["status"] == "reconciliation_required":
            if state.get("failed_stage") == "preparing":
                # The preparation adapter reconciles its own durable receipts.
                # It will not repeat a request with an unresolved intent.
                _stage(output, state, "preparing")
            else:
                if state.get("failed_stage") == "collecting" and (output / "analysis/batch-report.json").exists():
                    _adopt_collection(output, request, state)
                elif state.get("failed_stage") == "settling" and (output / state["settlement_path"] / "capture-receipts.json").exists():
                    _adopt_settlement(output, request, state)
                return _summary(output, state)
        if state["status"] in {"new", "preparing"}:
            from football_research.data_adapters.live_fixture_preparation import prepare_live_fixture
            resuming = state["status"] == "preparing"
            _stage(output, state, "preparing")
            result = prepare_live_fixture(_prepare_query(request), output=output / "preparation",
                max_http_requests=request["settings"]["preparation_http_budget"], resume=resuming)
            if result["status"] != "ready":
                if evidence_live_workflow.try_fallback(output,request,state,result):
                    return evidence_live_workflow.advance(output,request,state,settle=settle)
                status = "reconciliation_required" if result["status"] == "interrupted" else result["status"]
                _stage(output, state, status, reason=result.get("reason", result["status"]), failed_stage="preparing",
                    preparation_result="preparation/prepare-result.json")
                return _summary(output, state)
            _prepared(output)
            _stage(output, state, "prepared", prepared_sha256={p: _hash(output / p) for p in
                ("preparation/initializations.json", "preparation/identity-bundle.json", "preparation/prepare-result.json")})
        if state["status"] in {"prepared", "registered"}:
            _collection(output, request, state)
        elif state["status"] == "collecting":
            _adopt_collection(output, request, state)
        elif state["status"] == "settling":
            _adopt_settlement(output, request, state)
        if settle and state["status"] == "awaiting_result":
            _settle(output, request, state)
        return _summary(output, state)


def _summary(output, state):
    return {**state, "output": str(output), "stage_state_is_mutable": True,
        "raw_responses_immutable": True, "worker_mode": "explicit_single_tick_no_background_service"}


def status_match(output):
    output = Path(output).resolve()
    _, state = _load_job(output)
    return _summary(output, state)


def tick_tasks(task_root, *, limit=20):
    """Run one bounded due-work scan; starting a persistent timer is separate."""
    if type(limit) is not int or not 1 <= limit <= 20:
        raise ValueError("A tick may inspect at most twenty registered tasks")
    task_root = Path(task_root).resolve()
    task_dirs = {p.parent for p in task_root.glob("*/state.json")}
    if (task_root / "registry.json").exists():
        task_dirs.update(Path(p) for p in _read(task_root / "registry.json")["tasks"]
                         if (Path(p) / "state.json").is_file())
    results = []
    for task in sorted(task_dirs):
        if len(results) >= limit:
            break
        try:
            snapshot = status_match(task)
            if snapshot["status"] in TERMINAL or snapshot["status"] == "reconciliation_required":
                continue
            if snapshot["status"] in {"awaiting_result","awaiting_result_no_prior"} and parse_time(snapshot["result_due_at"]) > _now():
                continue
            results.append(resume_match(task, settle=True))
        except (ValueError, OSError) as error:
            results.append({"output": str(task), "status": "failed", "error_type": type(error).__name__, "reason": str(error)})
    return {"schema": VERSION, "tasks_advanced": len(results), "results": results,
        "background_worker_started": False}


def start_batch(queries, output, *, settings_path=None, match_date=None):
    """Freeze an explicit ordered match cohort, then execute each job once."""
    if not isinstance(queries, list) or not 1 <= len(queries) <= 20:
        raise ValueError("A batch needs one to twenty explicit match selections")
    normalized = [digest(_prepare_query({"query": q, "date": match_date, "input_timezone": "Asia/Shanghai"}))
                  for q in queries]
    if len(set(normalized)) != len(normalized):
        raise ValueError("Duplicate match selections cannot inflate the batch")
    settings = load_settings(settings_path)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "settings.json", settings)
    _write(output / "batch.json", {"schema": "footballmind-live-cohort-v1", "created_at": _now().isoformat(),
        "settings_sha256": _hash(output / "settings.json"), "date": match_date,
        "selection": "explicit_ordered_match_inputs_not_outcome_selected", "purpose": "f_train",
        "tasks": [{"query": q, "directory": f"task-{i:03}-{key[:12]}"}
            for i, (q, key) in enumerate(zip(queries, normalized))],
        "actor_request_cap": len(queries) * settings["actor_requests_per_match"],
        "preparation_request_cap": len(queries) * settings["preparation_http_budget"],
        "target_count_is_not_independent_support_or_training_readiness": True})
    return resume_batch(output)


def resume_batch(output):
    output = Path(output).resolve()
    with FileLock(str(output / ".batch.lock"), timeout=0):
        plan = _read(output / "batch.json")
        if plan["settings_sha256"] != _hash(output / "settings.json"):
            raise ValueError("Frozen batch settings changed")
        results = []
        for task in plan["tasks"]:
            directory = output / task["directory"]
            try:
                if (directory / "state.json").exists():
                    result = resume_match(directory)
                elif directory.exists():
                    result = {"status": "reconciliation_required", "reason": "partial_task_creation",
                        "output": str(directory)}
                else:
                    result = start_match(task["query"], directory, settings_path=output / "settings.json",
                        match_date=plan["date"])
            except (ValueError, OSError) as error:
                result = {"status": "failed", "output": str(directory), "reason": str(error)}
            results.append(result)
        summary = {"schema": plan["schema"], "batch_sha256": _hash(output / "batch.json"),
            "status": "completed" if all(r.get("run_directory") for r in results) else "incomplete",
            "results": results, "attempted_tasks": len(results),
            "collected_tasks": sum(bool(r.get("run_directory")) for r in results),
            "confirmed_physical_matches": len({r["source_event_id"] for r in results if r.get("source_event_id")}),
            "realized_voi_benefit_proven": False, "background_worker_started": False}
        atomic_write(output / "batch-result.json", json_bytes(summary))
        return summary


def replay_match(output, destination):
    from .learned_voi_replay import replay_saved_run
    output = Path(output).resolve()
    _, state = _load_job(output)
    if not state.get("run_directory"):
        raise ValueError("This task has no completed actor run")
    return replay_saved_run(output / state["run_directory"], output=destination, project_root=ROOT)
