"""Export existing prospective-text runs to the audited information-stream format.

The exporter is deliberately a projection layer.  It does not issue HTTP
requests, recalculate probabilities, or create a second archive.  It reads the
immutable ``configuration.json``, ``cycle-*.json`` and ``receipt-*.json``
artifacts produced by :func:`collect_prepared_batch`, copies referenced
RawArchive files without changing their bytes, and writes the six canonical
JSONL files consumed by ``football_platform.information.stream_validator``.

Runs created before this exporter existed are labelled
``historical_reconstruction`` unless a run configuration explicitly proves a
prospective capture.  This keeps a retrospective export useful for schema
inspection without presenting it as an online Agent menu or as empirical g.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from math import fsum, isfinite
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping


DATA_FILES = (
    "fixtures.jsonl",
    "documents.jsonl",
    "discovery.jsonl",
    "attempts.jsonl",
    "context_snapshots.jsonl",
    "labels/results.jsonl",
)
POLICY_ARMS = {"voi", "fixed_order", "random_explore", "full_probe", "not_configured"}
CONTEXT_FIELDS = ("ranking", "points", "qualification_status", "schedule",
                  "injuries", "lineup_status", "odds", "weather")
PREDICTOR_CONTEXT_SCHEMAS = {"runtime-predictor-context-v2", "runtime-predictor-context-v3", "runtime-predictor-context-v4"}
CONTEXT_SCHEMAS = {"runtime-context-v1", *PREDICTOR_CONTEXT_SCHEMAS}


def _load(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


def _iso(value: Any) -> str | None:
    parsed = _time(value)
    return parsed.isoformat() if parsed else None


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False) + "\n")
            count += 1
    return count


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory_provenance(value: Mapping[str, Any] | str | Path | None,
                          *, base_dir: Path | None = None) -> dict[str, Any] | None:
    """Describe an external fixture-inventory report without copying it.

    Inventory runs are intentionally kept outside the canonical stream package
    because their RawArchive is owned by the inventory adapter.  The stream
    package still needs a tamper-evident reference to the exact report used to
    decide that the package is empty.  When the report is readable we compute
    its digest locally; a caller-supplied digest is retained separately and is
    never treated as verified unless it matches the bytes we read.
    """
    if value is None:
        return None
    if isinstance(value, Mapping):
        raw_path = value.get("path") or value.get("inventory_report")
        supplied_hash = (value.get("sha256") or value.get("raw_sha256") or
                         value.get("inventory_report_sha256"))
        supplied_run_id = value.get("run_id") or value.get("inventory_run_id")
    else:
        raw_path = str(value)
        supplied_hash = None
        supplied_run_id = None
    if not raw_path:
        return {"path": None, "resolved_path": None, "run_id": supplied_run_id,
                "sha256": None, "provided_sha256": supplied_hash,
                "hash_status": "missing_path"}

    display_path = str(raw_path)
    candidate = Path(display_path).expanduser()
    candidates = [candidate] if candidate.is_absolute() else [
        (base_dir / candidate) if base_dir is not None else None,
        Path.cwd() / candidate,
    ]
    path = next((item.resolve() for item in candidates if item is not None and item.is_file()), None)
    # Keep the original path in the report.  ``resolved_path`` is an audit
    # aid, not a replacement: moving an external run must remain visible.
    result: dict[str, Any] = {
        "path": display_path,
        "resolved_path": str(path) if path is not None else None,
        "run_id": supplied_run_id,
        "provided_run_id": supplied_run_id,
        "sha256": None,
        "provided_sha256": supplied_hash,
        "hash_status": "unavailable",
    }
    if path is None:
        return result
    actual = _sha256(path)
    result["sha256"] = actual
    if supplied_hash:
        result["hash_status"] = "verified" if str(supplied_hash).lower() == actual else "mismatch"
    else:
        result["hash_status"] = "computed"
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        loaded = None
    if isinstance(loaded, Mapping):
        result["run_id"] = loaded.get("run_id")
        result["run_id_status"] = ("mismatch" if supplied_run_id is not None
            and supplied_run_id != result["run_id"] else "matched" if supplied_run_id is not None
            else "read_from_report")
        if loaded.get("schema") is not None:
            result["schema"] = loaded.get("schema")
    return result


def assign_policy_arm(*, physical_match_id: str, kickoff_at: str,
                      arm: str, assigned_at: str, policy_version: str,
                      random_seed: int | None = None,
                      propensity_logging: bool = False) -> dict[str, Any]:
    """Create the pre-kickoff arm record used by an export.

    This helper only records a declared experimental assignment.  It never
    claims that a numeric VoI model ran.  ``random_explore`` requires a seed
    and explicit propensity logging; an absent probability remains ``null``.
    """
    if arm not in POLICY_ARMS - {"not_configured"}:
        raise ValueError(f"unknown policy arm: {arm!r}")
    assigned, kickoff = _time(assigned_at), _time(kickoff_at)
    if assigned is None or kickoff is None or not assigned < kickoff:
        raise ValueError("policy arm must be assigned before kickoff with timezone-aware times")
    if arm == "random_explore" and (type(random_seed) is not int or not propensity_logging):
        raise ValueError("random_explore requires an integer random_seed and propensity_logging=true")
    return {
        "physical_match_id": physical_match_id,
        "policy_arm": arm,
        "arm_assigned_at": assigned.isoformat(),
        "policy_version": policy_version,
        "random_seed": random_seed,
        "propensity_logging": bool(propensity_logging),
        "selection_probability_basis": None,
    }


def _verified_market_fixture(config: Mapping[str, Any], match_folder: Path) -> Mapping[str, Any]:
    """Return a fixture only after reproducing its frozen prior from raw bytes."""
    from .prospective_text import verify_prior_archive

    policy = config.get("policy_manifest")
    if isinstance(policy, Mapping) and isinstance(policy.get("base_policy"), Mapping):
        policy = policy["base_policy"]
    market = policy.get("market_initialization") if isinstance(policy, Mapping) else None
    fixture = market.get("fixture") if isinstance(market, Mapping) else None
    if not isinstance(fixture, Mapping):
        return {}
    declared = _match_identity(config)["physical_match_id"]
    if declared not in {"unknown", fixture.get("match_id")}:
        return {}
    try:
        verify_prior_archive(policy, match_folder / "prior-archive")
    except (OSError, ValueError, TypeError, KeyError):
        return {}
    return fixture


def _match_identity(config: Mapping[str, Any], *,
                    verified_market_fixture: Mapping[str, Any] | None = None) -> dict[str, Any]:
    initial = config.get("initial_state") or {}
    base = initial.get("base") or {}
    evidence = config.get("evidence_projection") or {}
    match = evidence.get("match") if isinstance(evidence, Mapping) else {}
    match = match if isinstance(match, Mapping) else {}
    supervisor = config.get("supervisor_metadata") or {}
    fixture = supervisor.get("fixture") if isinstance(supervisor, Mapping) else {}
    fixture = fixture if isinstance(fixture, Mapping) else {}
    verified = supervisor.get("verified_fixture_identity") if isinstance(supervisor, Mapping) else {}
    verified = verified if isinstance(verified, Mapping) else {}
    canonical = verified.get("canonical") if isinstance(verified, Mapping) else {}
    canonical = canonical if isinstance(canonical, Mapping) else {}
    market_fixture = verified_market_fixture or {}
    mid = match.get("match_id") or initial.get("match_id") or base.get("match_id")
    if mid is not None and market_fixture.get("match_id") != mid:
        market_fixture = {}
    mid = mid or market_fixture.get("match_id")
    kickoff = (match.get("kickoff_at") or base.get("kickoff") or
               canonical.get("kickoff") or market_fixture.get("kickoff"))
    home = (match.get("home_team") or fixture.get("home_team") or
            canonical.get("home_name") or market_fixture.get("home_team_name") or base.get("home_team_id"))
    away = (match.get("away_team") or fixture.get("away_team") or
            canonical.get("away_name") or market_fixture.get("away_team_name") or base.get("away_team_id"))
    competition = (match.get("competition") or fixture.get("competition") or
                   canonical.get("competition") or market_fixture.get("competition_name") or "unknown")
    season = (match.get("season") or fixture.get("season") or canonical.get("season")
              or market_fixture.get("season") or "unknown")
    source_url = (canonical.get("source_url") or fixture.get("source_url") or
                  (verified.get("canonical") or {}).get("source_url"))
    # Registration is the act of freezing this fixture into the run. A
    # source observation (legacy fixture_available_at) does not prove that
    # the collector registered the match at the same time.
    registered_at = None
    registration_timing_basis = "not_recorded"
    for label, container in (("evidence_projection.match", match),
                             ("supervisor_metadata.fixture", fixture),
                             ("verified_market_fixture", market_fixture),
                             ("configuration", config)):
        for field in ("registered_at", "fixture_registered_at"):
            if container.get(field) is not None:
                registered_at = _iso(container.get(field))
                registration_timing_basis = f"explicit_{label}.{field}"
                if registered_at is None:
                    registration_timing_basis += ":invalid_timezone_aware_time"
                break
        if registration_timing_basis != "not_recorded":
            break
    market = _market_initialization(config)
    if registration_timing_basis == "not_recorded" and market_fixture and market.get("frozen_at") is not None:
        registered_at = _iso(market.get("frozen_at"))
        registration_timing_basis = "verified_market_initialization.frozen_at"
        if registered_at is None:
            registration_timing_basis += ":invalid_timezone_aware_time"
    if registration_timing_basis == "not_recorded" and config.get("run_id") and mid:
        registered_at = _iso(config.get("created_at"))
        if registered_at is not None:
            registration_timing_basis = "bound_runtime_configuration.created_at"

    timing_basis = (match.get("time_basis") or fixture.get("time_basis") or
                    canonical.get("time_basis") or market_fixture.get("time_basis"))
    source_observed_at = None
    source_observed_at_basis = "not_recorded"
    for label, container in (("evidence_projection.match", match),
                             ("supervisor_metadata.fixture", fixture),
                             ("verified_market_fixture", market_fixture)):
        for field in ("source_observed_at", "observed_at"):
            if _iso(container.get(field)) is not None:
                source_observed_at = _iso(container.get(field))
                source_observed_at_basis = f"explicit_{label}.{field}"
                break
        if source_observed_at is not None:
            break
    if source_observed_at is None and market_fixture and "verified_observation_at" in str(market_fixture.get("time_basis") or ""):
        source_observed_at = _iso(market_fixture.get("fixture_available_at"))
        if source_observed_at is not None:
            source_observed_at_basis = "verified_market_fixture.fixture_available_at_with_verified_observation_at_basis"
    source_timezone = (match.get("source_timezone") or fixture.get("source_timezone") or
                       canonical.get("source_timezone") or market_fixture.get("source_timezone"))
    if source_timezone is None and "Asia/Shanghai_source_clock" in str(timing_basis or ""):
        source_timezone = "Asia/Shanghai"
    original_time_text = None
    for container in (match, fixture, canonical, market_fixture):
        for field in ("original_time_text", "original_time"):
            if container.get(field) is not None:
                original_time_text = container.get(field)
                break
        if original_time_text is not None:
            break
    return {"physical_match_id": mid or "unknown", "kickoff_at": kickoff,
            "home_team": home or "unknown", "away_team": away or "unknown",
            "competition": competition or "unknown", "season": season or "unknown",
            "fixture_source_url": source_url or "", "registered_at": registered_at,
            "registration_timing_basis": registration_timing_basis,
            "source_observed_at": source_observed_at,
            "source_observed_at_basis": source_observed_at_basis,
            "source_timezone": source_timezone, "original_time_text": original_time_text,
            "timing_basis": timing_basis,
            "registration_source": market_fixture.get("source") or "prospective_text.configuration"}


def _mode(config: Mapping[str, Any], kickoff: datetime | None, *, requested: str | None) -> str:
    if requested is not None:
        if requested not in {"prospective_capture", "archived_snapshot", "historical_reconstruction"}:
            raise ValueError(f"invalid information mode: {requested!r}")
        if requested in {"prospective_capture", "archived_snapshot"} and not (
                config.get("information_mode") == requested
                or config.get(f"{requested}_proven") is True):
            raise ValueError(f"export mode cannot upgrade an unproven run to {requested}")
        return requested
    if config.get("information_mode") in {"prospective_capture", "archived_snapshot", "historical_reconstruction"}:
        return str(config["information_mode"])
    # The old runtime did not record a mode.  A run is only prospective when
    # it was explicitly marked as such; absence of proof is reconstruction.
    if config.get("prospective_capture_proven") is True:
        return "prospective_capture"
    return "historical_reconstruction"


def _raw_copy(batch: Path, match_folder: Path, raw_path: str | None, output: Path) -> str | None:
    if not raw_path:
        return None
    # RawArchive records use backslashes on Windows.  Resolve relative to the
    # per-match source archive, then copy both immutable payload and metadata.
    archive_relative = Path(str(raw_path).replace("\\", "/"))
    relative = archive_relative
    if relative.parts and relative.parts[0].lower() == "raw":
        relative = Path(*relative.parts[1:])
    source_archive = match_folder / "source-archive"
    source = (source_archive / archive_relative).resolve()
    if not source.is_file():
        source = (source_archive / relative).resolve()
    root = source_archive.resolve()
    if not source.is_file() or (source != root and root not in source.parents):
        return None
    destination_relative = Path("raw") / relative
    destination = output / destination_relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(source, destination)
    sidecar = source.with_name(source.name.replace(".body.gz", ".observation.json"))
    if sidecar.is_file():
        side_destination = destination.with_name(destination.name.replace(".body.gz", ".observation.json"))
        if not side_destination.exists():
            shutil.copy2(sidecar, side_destination)
    return destination_relative.as_posix()


def _identity_raw_copy(match_folder: Path, observation_ids: Iterable[str], output: Path,
                       *, config: Mapping[str, Any] | None = None) -> tuple[str | None, str | None, str | None]:
    """Copy an already archived fixture identity response into the export.

    The prospective text collector stores the independent fixture response in
    the per-match ``identity-archive``.  This projection copies that existing
    RawArchive body and sidecar; it never creates a new response or digest.
    """
    from football_spider.storage.raw_archive import RawArchive

    wanted = {str(value) for value in observation_ids if value}
    roots: list[tuple[Path, dict[str, str | None]]] = []
    if wanted:
        roots.append((match_folder / "identity-archive", {value: None for value in wanted}))

    # A normal spftll prospective run has no independent identity archive. Its
    # frozen fixture is instead bound to specific observations in prior-archive.
    # Reconstruct the fixture and quote with the existing verifier before using
    # those references; never pick the first payload in this shared archive.
    policy = config.get("policy_manifest") if isinstance(config, Mapping) else None
    if isinstance(policy, Mapping) and isinstance(policy.get("base_policy"), Mapping):
        policy = policy["base_policy"]
    market = policy.get("market_initialization") if isinstance(policy, Mapping) else None
    fixture = market.get("fixture") if isinstance(market, Mapping) else None
    prior_refs: dict[str, str | None] = {}
    if isinstance(fixture, Mapping) and isinstance(fixture.get("input_manifest"), (list, tuple)):
        for entry in fixture["input_manifest"]:
            if not isinstance(entry, str):
                continue
            parts = dict(item.split(":", 1) for item in entry.split("|") if ":" in item)
            observation_id, expected = parts.get("observation"), parts.get("sha256")
            if observation_id and expected and len(expected) == 64 and all(char in "0123456789abcdefABCDEF" for char in expected):
                if observation_id in prior_refs and prior_refs[observation_id] != expected.lower():
                    return None, None, None
                prior_refs[observation_id] = expected.lower()
    if prior_refs and config is not None and _verified_market_fixture(config, match_folder):
        roots.append((match_folder / "prior-archive", prior_refs))

    for root, references in roots:
        if not root.is_dir():
            continue
        try:
            observations = RawArchive(root).observations()
        except (OSError, ValueError):
            continue
        for observation in observations:
            if str(observation.id) not in references:
                continue
            expected = references[str(observation.id)]
            if expected is not None and observation.content_hash != expected:
                continue
            source = (root / observation.payload_path).resolve()
            if not source.is_file() or root.resolve() not in source.parents:
                continue
            try:
                RawArchive(root).read_body(observation)
            except (OSError, ValueError):
                continue
            relative = Path("raw") / Path(str(observation.payload_path).replace("\\", "/")).relative_to("raw")
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copy2(source, destination)
            elif _sha256(destination) != _sha256(source):
                raise ValueError("Immutable raw payload collision during fixture export")
            sidecar = source.with_name(source.name.replace(".body.gz", ".observation.json"))
            if sidecar.is_file():
                side_destination = destination.with_name(destination.name.replace(".body.gz", ".observation.json"))
                if not side_destination.exists():
                    shutil.copy2(sidecar, side_destination)
                elif _sha256(side_destination) != _sha256(sidecar):
                    raise ValueError("Immutable raw observation collision during fixture export")
            return relative.as_posix(), observation.content_hash, str(observation.id)
    return None, None, None


def _verified_archive_body(root: Path, relative: str | None, expected_hash: str | None) -> bytes | None:
    """Verify an existing response using RawArchive's existing digest contract."""
    if not relative or not expected_hash:
        return None
    from football_spider.models.contracts import Observation
    from football_spider.storage.raw_archive import RawArchive
    path = (root / relative).resolve()
    if root.resolve() not in path.parents or not path.name.endswith(".body.gz"):
        return None
    sidecar = path.with_name(path.name.replace(".body.gz", ".observation.json"))
    try:
        observation = Observation.model_validate_json(sidecar.read_bytes())
        if observation.content_hash != expected_hash:
            return None
        # The copied payload keeps its RawArchive-relative path.  Do not
        # validate a different payload named by a tampered sidecar.
        if (root / observation.payload_path).resolve() != path:
            return None
        return RawArchive(root).read_body(observation)
    except (OSError, ValueError, TypeError):
        return None


def _menu_proven(root: Path, relative: str | None, expected_hash: str | None,
                 config: Mapping[str, Any], cycle: Mapping[str, Any], step: int,
                 decision_at: str | None, kickoff: datetime | None) -> bool:
    if (config.get("trajectory_schema") != "prospective-acquisition-trajectory-v1"
            or not _time(decision_at) or not kickoff or not _time(decision_at) < kickoff):
        return False
    raw = _verified_archive_body(root, relative, expected_hash)
    if raw is None:
        return False
    try:
        archived = json.loads(raw)
    except (ValueError, UnicodeError):
        return False
    return bool(isinstance(archived, Mapping)
        and archived.get("schema") == "prospective-menu-archive-v1"
        and archived.get("run_id") == config.get("run_id")
        and archived.get("step") == step
        and _iso(archived.get("observed_at")) == decision_at
        and archived.get("generated_actions") == cycle.get("generated_actions")
        and archived.get("menu") == cycle.get("menu")
        and archived.get("public_before") == cycle.get("public_before"))


def _information_partition(state: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    if isinstance(state.get("state"), Mapping):
        state = state["state"]
    groups = state.get("groups") or {}
    known = [str(key) for key, value in groups.items()
             if isinstance(value, Mapping) and value.get("status") == "observed"]
    unknown = [str(key) for key, value in groups.items()
               if not isinstance(value, Mapping) or value.get("status") != "observed"]
    return known, unknown


def _market_initialization(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Find the recorded market binding; never calculate a new forecast."""
    manifest = config.get("policy_manifest")
    while isinstance(manifest, Mapping):
        market = manifest.get("market_initialization")
        if isinstance(market, Mapping):
            return market
        manifest = manifest.get("base_policy")
    return {}


def _context_fields(state: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """Project only explicit structured values and their own observation clocks.

    Topic mentions in documents are not structured injury/lineup/weather facts.
    The snapshot clock cannot supply a missing per-field observation clock, and
    publication, source quote and capture times cannot supply availability.
    """
    nested = state.get("state") if isinstance(state.get("state"), Mapping) else state
    containers = [state, nested]
    for parent in (state, nested):
        for name in ("context_fields", "context", "groups"):
            if isinstance(parent.get(name), Mapping):
                containers.append(parent[name])
    result = {}
    for field in CONTEXT_FIELDS:
        supplied, found = None, False
        for container in containers:
            if field in container:
                supplied, found = container[field], True
                break
        basis = "explicit_runtime_context" if found else "not_recorded"
        if not found and field == "odds":
            quote = _market_initialization(config).get("quote")
            if isinstance(quote, Mapping):
                supplied, found, basis = quote, True, "recorded_market_initialization_quote"
        metadata = supplied if isinstance(supplied, Mapping) else {}
        status = str(metadata.get("status") or ("observed" if found and supplied is not None else "unknown"))
        if status.lower() in {"unknown", "not_acquired", "not_configured", "source_missing", "unavailable"}:
            value = None
        elif isinstance(supplied, Mapping):
            if "value" in supplied:
                value = supplied["value"]
            elif "values" in supplied:
                value = supplied["values"]
            else:
                excluded = {"observed_at", "captured_at", "available_at", "quote_time", "source_timestamp",
                            "availability_status", "timing_basis", "status", "source", "source_url",
                            "raw_path", "raw_sha256", "input_manifest"}
                value = {key: item for key, item in supplied.items() if key not in excluded} or None
        else:
            value = supplied
        available = _iso(metadata.get("available_at"))
        entry = {"value": value, "status": status,
                 "observed_at": _iso(metadata.get("observed_at")),
                 "captured_at": _iso(metadata.get("captured_at")),
                 "available_at": available,
                 "availability_status": "known" if available else "unknown",
                 "timing_basis": metadata.get("timing_basis") or basis}
        for key in ("source", "source_url", "raw_path", "raw_sha256", "input_manifest"):
            if key in metadata:
                entry[key] = metadata[key]
        if field == "odds":
            entry["quote_time"] = _iso(metadata.get("quote_time") or metadata.get("source_timestamp"))
            entry["quote_time_basis"] = ("explicit_quote_time" if metadata.get("quote_time") else
                "recorded_provider_source_timestamp" if metadata.get("source_timestamp") else "not_recorded")
        result[field] = entry
    return result


def _forecast_fields(config: Mapping[str, Any], before: Mapping[str, Any] | None,
                     after: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Keep a frozen market prior separate from a configured fitted predictor."""
    manifest = config.get("policy_manifest") or {}
    manifest = manifest if isinstance(manifest, Mapping) else {}
    trained = bool(
        (manifest.get("schema") == "learned-joint-strict-voi-v1"
         and manifest.get("model_version") and manifest.get("trained_as_of"))
        or (manifest.get("predictor_version") and manifest.get("predictor_sha256"))
        or (isinstance(config.get("text_response_model"), Mapping)
            and config["text_response_model"].get("model_version")
            and config["text_response_model"].get("sha256")))
    market = _market_initialization(config)
    prior = _probabilities({"prediction_before": market.get("probabilities")})
    return {
        "prediction_before": before if trained else None,
        "prediction_after": after if trained else None,
        "prediction_status": "trained_f_configured" if trained else "market_prior_only" if market else "model_not_configured",
        "calibration_status": (manifest.get("calibration_status") or "not_established") if trained else "not_established",
        "market_prior": prior,
        "market_prior_basis": "recorded_market_initialization" if market else None,
    }


def _f_consumption_required(config: Mapping[str, Any]) -> bool:
    """Return whether this run declares a fitted f whose input use must be auditable.

    The flag is deliberately derived only from the frozen configuration.  It
    is not inferred from a probability change, a returned document, or a
    non-empty claim list.  Runs without a declared predictor remain compatible
    with the legacy stream contract.
    """
    manifest = config.get("policy_manifest") or {}
    manifest = manifest if isinstance(manifest, Mapping) else {}
    if (manifest.get("schema") == "learned-joint-strict-voi-v1"
            and manifest.get("model_version") and manifest.get("trained_as_of")):
        return True
    if manifest.get("predictor_version") and manifest.get("predictor_sha256"):
        return True
    text_model = config.get("text_response_model")
    return (isinstance(text_model, Mapping) and bool(text_model.get("model_version"))
            and bool(text_model.get("sha256")))


def _context_raw_fields(batch: Path, folder: Path, target: Path,
                        timing: Mapping[str, Any], *, snapshot_kind: str,
                        run_id: str | None) -> dict[str, Any]:
    relative = _raw_copy(batch, folder, timing.get("raw_path"), target)
    digest = timing.get("raw_sha256")
    body = _verified_archive_body(target, relative, digest)
    timing_proven = False
    save_return_proven = False
    predictor_timing: dict[str, Any] = {}
    if body is not None:
        try:
            payload = json.loads(body)
            sidecar = (target / relative).with_name(Path(relative).name.replace(".body.gz", ".observation.json"))
            observation = _load(sidecar, {})
            bound_observation = (str(timing.get("raw_observation_id")) == str(observation.get("id"))
                and timing.get("raw_observation_id") is not None and observation.get("content_hash") == digest)
            save_return_proven = bool(bound_observation
                and timing.get("capture_timing_basis") == "measured_local_utc_archive_save_return"
                and timing.get("persistence_boundary") == "return_observed_fsync_not_instrumented"
                and _time(timing.get("archive_save_started_at")) is not None
                and _time(timing.get("archive_save_started_at")) == _time(observation.get("observed_at"))
                and _time(timing.get("archive_save_returned_at")) is not None
                and _time(timing.get("captured_at")) == _time(timing.get("archive_save_returned_at"))
                and _time(timing.get("archive_save_started_at")) <= _time(timing.get("archive_save_returned_at")))
            timing_proven = bool(
                isinstance(payload, Mapping) and payload.get("schema") in CONTEXT_SCHEMAS
                and payload.get("snapshot_kind") == snapshot_kind
                and run_id is not None and payload.get("run_id") == run_id
                and _time(timing.get("observed_at")) is not None
                and _time(timing.get("available_at")) is not None
                and _time(timing.get("prediction_cutoff")) is not None
                and bound_observation
                and all(_time(timing.get(key)) == _time(payload.get(key))
                        for key in ("observed_at", "available_at", "prediction_cutoff"))
                and all(timing.get(key) == payload.get(key) for key in
                        ("availability_status", "availability_basis", "parent_attempt_id")))
            if isinstance(payload, Mapping) and payload.get("schema") in {"runtime-predictor-context-v3", "runtime-predictor-context-v4"}:
                forecast = _time(payload.get("forecast_as_of"))
                deadline = _time(payload.get("acquisition_deadline"))
                timing_proven = bool(timing_proven and forecast is not None and deadline is not None
                    and forecast == _time(timing.get("forecast_as_of"))
                    and forecast == _time(payload.get("prediction_cutoff"))
                    and forecast <= _time(payload.get("observed_at")) and forecast <= deadline)
                processing_ready = True
                if payload.get("schema") == "runtime-predictor-context-v4":
                    from .predictor_context_pairs import predictor_processing_valid
                    timing_proven = bool(timing_proven and predictor_processing_valid(payload)
                        and all(payload.get(key) == timing.get(key)
                                for key in ("prediction_processing", "prediction_pipeline_version")))
                    processing_ready = (isinstance(payload.get("prediction_processing"), Mapping)
                        and payload["prediction_processing"].get("status") == "ready")
                if timing_proven:
                    # A saved audit of an earlier forecast is not a new forecast.
                    # Do not turn the later handoff/save clocks into its cutoff.
                    state = payload.get("state") or {}
                    base = (state.get("base") or {}) if isinstance(state, Mapping) else {}
                    kickoff = _time(base.get("kickoff")) if isinstance(base, Mapping) else None
                    captured = _time(timing.get("captured_at")) if save_return_proven else None
                    eligible = (processing_ready and captured <= deadline and captured < kickoff
                                if captured is not None and kickoff is not None else None)
                    predictor_timing = {"raw_context_schema": payload["schema"],
                        "forecast_as_of": payload["forecast_as_of"], "forecast_eligible": eligible,
                        "recording_scope": "pre_match_context" if eligible is True else
                            "audit_only" if eligible is False else "unknown"}
                    if payload.get("schema") == "runtime-predictor-context-v4":
                        predictor_timing.update(prediction_pipeline_version=payload["prediction_pipeline_version"],
                            prediction_processing=deepcopy(payload["prediction_processing"]))
        except (TypeError, ValueError, UnicodeError):
            timing_proven = False
    return {"raw_path": relative, "raw_sha256": digest,
            "raw_observation_id": timing.get("raw_observation_id"),
            "raw_digest_verified": body is not None,
            "raw_context_timing_proven": timing_proven,
            "raw_context_timing_basis": ("local_callback_return_receipt_bound_to_raw_observation_and_payload"
                if save_return_proven and timing_proven else "immutable_raw_payload_state_handoff_independent_of_archive_completion"
                if timing_proven else "unknown"),
            "context_capture_completion_proven": save_return_proven,
            "captured_at": _iso(timing.get("captured_at")) if save_return_proven else None,
            "capture_status": "known" if save_return_proven else "unknown",
            "source_context_captured_at": _iso(timing.get("captured_at")),
            "source_context_captured_at_basis": "callback_projection_preserved_separately_from_completion_proof",
            "archive_save_started_at": _iso(timing.get("archive_save_started_at")),
            "archive_save_returned_at": _iso(timing.get("archive_save_returned_at")),
            "capture_timing_basis": timing.get("capture_timing_basis") if save_return_proven else "unknown",
            "persistence_boundary": timing.get("persistence_boundary") or "not_instrumented",
            **predictor_timing}


def _context_cutoff_fields(config: Mapping[str, Any], timing: Mapping[str, Any],
                           observed_at: str | None) -> dict[str, Any]:
    """Keep a measured snapshot cutoff separate from the acquisition deadline.

    Legacy runs commonly stored kickoff-minus-one-minute as the run deadline.
    It does not establish a snapshot's actual prediction cutoff.  Only retain
    that legacy value as a cutoff when it was already at/before the observation;
    otherwise leave the missing snapshot cutoff unknown instead of backfilling.
    """
    initial = config.get("initial_state") or {}
    projection = config.get("evidence_projection") or {}
    match = projection.get("match") or {}
    planned = (_iso(timing.get("acquisition_deadline")) or
               _iso(initial.get("prediction_cutoff")) or _iso(match.get("prediction_cutoff")))
    cutoff = _iso(timing.get("prediction_cutoff"))
    basis = "runtime_context_timing" if cutoff is not None else "not_recorded"
    if cutoff is None and planned is not None and _time(observed_at) is not None:
        if _time(planned) <= _time(observed_at):
            cutoff, basis = planned, "recorded_cutoff_at_or_before_snapshot"
    return {"prediction_cutoff": cutoff, "prediction_cutoff_basis": basis,
            "acquisition_deadline": planned}


def _context_payload(target: Path, raw_fields: Mapping[str, Any]) -> Mapping[str, Any]:
    body = _verified_archive_body(target, raw_fields.get("raw_path"), raw_fields.get("raw_sha256"))
    if body is not None:
        try:
            payload = json.loads(body)
            if isinstance(payload, Mapping) and payload.get("schema") in CONTEXT_SCHEMAS:
                return payload
        except (TypeError, ValueError, UnicodeError):
            pass
    return {}


def _projection_ledger(projection: Any) -> dict[str, Any] | None:
    """Explicit semantic-consumption ledger recorded on one public prediction.

    Returns ``None`` when this projection records no ledger at all.  An explicit
    empty list is a real ``not_consumed`` record and is returned unchanged.
    """
    if not isinstance(projection, Mapping):
        return None
    if isinstance(projection.get("used_semantic_input_ids"), list):
        ids, basis = projection["used_semantic_input_ids"], "prediction.used_semantic_input_ids"
    else:
        nested = projection.get("projection")
        if not (isinstance(nested, Mapping) and isinstance(nested.get("used_input_ids"), list)):
            return None
        ids, basis = nested["used_input_ids"], "prediction.projection.used_input_ids"
    consumed = [str(value) for value in ids if isinstance(value, (str, int, float)) and str(value)]
    return {"f_consumed_input_ids": consumed,
            "f_consumption_status": "consumed" if consumed else "not_consumed",
            "f_consumption_basis": basis}


def _consumption_projection(states: Iterable[Any]) -> Mapping[str, Any] | None:
    """Resolve the public prediction projection that carries the ledger.

    Candidates are ordered by the caller: the after-action public projection
    precedes its state slice, because ``public_state_after`` is a state slice
    whose presence without a ``prediction`` must not hide a recorded
    projection.  The first projection that records an explicit ledger wins;
    when none records one, the first recorded projection is kept so its
    modeled-document count is still exported.  Records are never merged across
    projections.
    """
    projections = [state["prediction"] for state in states
                   if isinstance(state, Mapping) and isinstance(state.get("prediction"), Mapping)]
    return next((projection for projection in projections if _projection_ledger(projection) is not None),
                projections[0] if projections else None)


def _cycle_consumption_ledger(cycle: Any) -> dict[str, Any] | None:
    """Same-cycle ledger projected by the runtime, accepted only verbatim.

    ``autonomous_runtime_loop`` writes one explicit record per action cycle
    (``f_consumed_input_ids``, ``f_consumption_status``, ``f_consumption_basis``,
    ``f_modeled_read_documents``).  A malformed, duplicated or internally
    inconsistent record is not an f consumption proof: it stays unknown and is
    never repaired or promoted.
    """
    if not isinstance(cycle, Mapping) or "f_consumed_input_ids" not in cycle:
        return None
    ids, status = cycle.get("f_consumed_input_ids"), cycle.get("f_consumption_status")
    basis, modeled = cycle.get("f_consumption_basis"), cycle.get("f_modeled_read_documents")
    recorded = ([item.strip() for item in ids]
                if isinstance(ids, list) and all(isinstance(item, str) and item.strip() for item in ids)
                else None)
    if (recorded is None or len(set(recorded)) != len(recorded)
            or status not in {"consumed", "not_consumed"}
            or (status == "consumed") != bool(recorded)
            or not isinstance(basis, str) or not basis.strip() or basis in {"unknown", "not_recorded"}
            or (modeled is not None and (type(modeled) is not int or modeled < 0))):
        return None
    return {"f_consumed_input_ids": recorded, "f_consumption_status": status,
            "f_consumption_basis": basis, "f_modeled_read_documents": modeled}


def _predictor_context_fields(config: Mapping[str, Any], payload: Mapping[str, Any],
                              raw_fields: Mapping[str, Any],
                              runtime_prediction: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Project saved f outputs; never execute f or substitute the market prior.

    Predictor-only payloads store predictions beside the state, not inside it.
    Read them only when this raw payload is bound to the requested run, snapshot
    kind and recorded handoff. Legacy payloads retain their existing projection.
    """
    if (payload.get("schema") not in PREDICTOR_CONTEXT_SCHEMAS
            or raw_fields.get("raw_context_timing_proven") is not True):
        return {}

    def prediction(key):
        value = payload.get(key)
        names = ("home", "draw", "away")
        if (not isinstance(value, Mapping) or set(value) != set(names)
                or any(type(value[n]) not in {int, float} for n in names)):
            return None
        numbers = [float(value[n]) for n in names]
        if not all(isfinite(n) and 0 <= n <= 1 for n in numbers) or abs(fsum(numbers) - 1.) > 1e-6:
            return None
        return dict(zip(names, numbers))

    # Some entity-f runtimes keep the explicit consumption ledger in the
    # cycle's public prediction projection rather than in the raw context
    # envelope.  The caller passes that projection only when it is bound to
    # this exact cycle; missing fields remain unknown.
    projection = runtime_prediction if isinstance(runtime_prediction, Mapping) else None
    if projection is None:
        nested = payload.get("prediction")
        projection = nested if isinstance(nested, Mapping) else None
    ledger = _projection_ledger(projection)
    consumed = ledger["f_consumed_input_ids"] if ledger is not None else None
    consumption_status = ledger["f_consumption_status"] if ledger is not None else "unknown"
    consumption_basis = ledger["f_consumption_basis"] if ledger is not None else "not_recorded"
    modeled_documents = projection.get("modeled_read_documents") if projection is not None else None
    return {**_forecast_fields(config, prediction("prediction_before"), prediction("prediction_after")),
            **({key: deepcopy(payload.get(key)) for key in ("prediction_pipeline_version", "prediction_processing")}
               if payload.get("schema") == "runtime-predictor-context-v4" else {}),
            "prediction_basis": "immutable_predictor_context_payload",
            "predictor_model_versions": deepcopy(payload.get("predictor_model_versions")),
            "raw_context_schema": payload["schema"],
            "f_consumed_input_ids": consumed,
            "f_consumption_status": consumption_status,
            "f_consumption_basis": consumption_basis,
            "f_modeled_read_documents": modeled_documents,
            "f_consumption_required": _f_consumption_required(config)}


def _context_information(state: Mapping[str, Any], fields: Mapping[str, Any],
                          documents: Any) -> dict[str, Any]:
    """Describe only information present in this exact recorded state."""
    current = state.get("state") if isinstance(state.get("state"), Mapping) else state
    known, unknown = _information_partition(current)
    base = current.get("base") or {}
    if isinstance(base, Mapping):
        # Runtime states keep fixture identity at the top level today, while
        # adapters may persist it inside ``base``.  Either explicit shape is
        # sufficient; an empty source projection is not.
        fixture_id = (current.get("match_id") or base.get("match_id") or
                       base.get("fixture_id"))
        if fixture_id and base.get("kickoff"):
            known.append("fixture")
        if base.get("competition_id"):
            known.append("competition")
    for field in CONTEXT_FIELDS:
        detail = fields[field]
        if detail.get("value") is not None and detail.get("status", "").lower() == "observed":
            known.append(field)
        else:
            unknown.append(field)
    ids = []
    for document in documents if isinstance(documents, list) else []:
        if isinstance(document, Mapping) and isinstance(document.get("document_id"), str) and document["document_id"]:
            if document["document_id"] not in ids:
                ids.append(document["document_id"])
    known.extend("document:" + document_id for document_id in ids)
    return {"known_information": list(dict.fromkeys(known)),
            "unknown_information": [name for name in dict.fromkeys(unknown) if name not in known],
            "known_document_ids": ids,
            "known_information_basis": "export_projection_of_explicit_state_and_snapshot_documents",
            "unknown_information_basis": "export_projection_of_explicit_state; missing_fields_not_agent_value_judgments"}


def _context_documents(payload: Mapping[str, Any], state: Mapping[str, Any],
                       fallback: Any = None) -> list[Any]:
    """Use raw context documents, then the recorded pre-decision public state.

    A raw context payload with an explicit empty ``documents`` list is a real
    empty observation and must stay empty.  The fallback is used only when an
    older runtime payload did not carry the documents key at all.
    """
    if isinstance(payload, Mapping) and "documents" in payload:
        return payload.get("documents") if isinstance(payload.get("documents"), list) else []
    if isinstance(state, Mapping) and isinstance(state.get("documents"), list):
        return state["documents"]
    return fallback if isinstance(fallback, list) else []


def _prediction_mapping(value: Any) -> dict[str, float] | None:
    """Normalize one recorded 1X2 prediction without repairing it.

    Runtime snapshots exist in both the historical list form and the newer
    named mapping form.  The exporter must accept either representation, but
    it must never renormalize, fill missing values, or infer a probability
    from another field.  A malformed value therefore remains ``None`` and is
    rejected by the downstream label gates.
    """
    if isinstance(value, Mapping):
        if set(value) != {"home", "draw", "away"}:
            return None
        raw_values = [value.get("home"), value.get("draw"), value.get("away")]
    elif isinstance(value, (list, tuple)) and len(value) == 3:
        raw_values = list(value)
    else:
        return None
    try:
        numbers = [float(item) for item in raw_values]
    except (TypeError, ValueError):
        return None
    if not all(isfinite(item) for item in numbers):
        return None
    if any(item < 0 or item > 1 for item in numbers):
        return None
    if abs(fsum(numbers) - 1.0) > 1e-9:
        return None
    return {"home": numbers[0], "draw": numbers[1], "away": numbers[2]}


def _probabilities(cycle: Mapping[str, Any]) -> dict[str, float] | None:
    values = cycle.get("prediction_before")
    before = _prediction_mapping(values)
    if before is not None:
        return before
    public_before = cycle.get("public_before")
    nested = public_before.get("prediction", {}).get("probabilities") if isinstance(public_before, Mapping) else None
    return _prediction_mapping(nested)


def _candidate_rows(cycle: Mapping[str, Any]) -> list[dict[str, Any]]:
    actions = cycle.get("generated_actions")
    if not isinstance(actions, list):
        actions = []
    return [dict(action) for action in actions if isinstance(action, Mapping) and action.get("action_id")]


def _visible_document_links(cycle: Mapping[str, Any]) -> dict[str, Any]:
    """Project only the links present in the saved pre-decision public state.

    Provider RSS items and links returned by this cycle's request are not a
    substitute for the Agent's earlier context.  Missing link metadata stays
    unknown; an empty recorded list remains distinguishable from a missing one.
    """
    public = cycle.get("public_before")
    documents = public.get("documents") if isinstance(public, Mapping) else None
    if not isinstance(documents, list):
        return {"visible_document_links": None, "visible_document_links_status": "not_recorded"}
    links = []
    complete = True
    for document_order, document in enumerate(documents):
        if not isinstance(document, Mapping) or not isinstance(document.get("links"), list):
            complete = False
            continue
        for link_order, link in enumerate(document["links"]):
            if not isinstance(link, Mapping):
                complete = False
                continue
            links.append({"document_id": document.get("document_id"),
                "document_order": document_order, "link_order": link_order,
                "url": link.get("url"), "title": link.get("title"),
                "link": deepcopy(dict(link))})
    return {"visible_document_links": links,
            "visible_document_links_status": "recorded" if complete else "partially_recorded",
            "visible_document_links_basis": "cycle.public_before.documents.links",
            "visible_document_links_order_basis": "zero_based_document_order_then_zero_based_link_order"}


def _document_link_metadata(document: Mapping[str, Any]) -> dict[str, Any]:
    """Retain saved projection metadata without re-extracting source bytes."""
    keys = ("links", "links_truncated", "eligible_link_count", "text_truncated",
            "status", "http_status", "error_class", "publication_status",
            "claim_verification", "filter_status", "filtering_status", "filter_reason",
            "extraction_version", "parser_version", "news_projection_version",
            "projection_provenance", "structured_news")
    result = {key: deepcopy(document[key]) for key in keys if key in document}
    # Future source adapters can add extraction/filter facts without losing them
    # in a fixed export allowlist.  This is the saved runtime projection, not a
    # newly claimed source publication, candidate menu, or provider raw feed.
    result["runtime_document_metadata"] = {key: deepcopy(value) for key, value in document.items()
        if key not in {"text", "links", "document_id", "url"}}
    result["document_projection_basis"] = "saved_receipt_document_only_no_export_time_extraction"
    return result


def _result_visibility_fields(cycle: Mapping[str, Any]) -> dict[str, Any]:
    """The Agent can observe a returned receipt even when no body is admitted."""
    observed = _iso(cycle.get("result_observed_at")) or _iso(cycle.get("returned_at"))
    available = _iso(cycle.get("receipt_available_at")) or _iso(cycle.get("returned_at"))
    return {"result_observed": observed is not None, "result_observed_at": observed,
            "receipt_available_at": available,
            "receipt_availability_basis": (cycle.get("receipt_availability_basis") or
                "recorded_runtime_return_checkpoint") if available else "unknown"}


def _document_timing_fields(document: Mapping[str, Any], observation: Mapping[str, Any],
                            handoff: Mapping[str, Any], receipt: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Keep response observation, archive invocation, and Agent handoff distinct."""
    context = observation.get("context") or {}
    context = context if isinstance(context, Mapping) else {}
    captured = None
    capture_basis = "unknown"
    archive_started = None
    archive_returned = None
    if context.get("archive_save_timing_basis") == "measured_local_utc_archive_save_start":
        archive_started = _iso(context.get("archive_save_started_at"))
    saved_timing = (receipt or {}).get("archive_save_timing") or {}
    return_proven = (isinstance(saved_timing, Mapping)
        and str(saved_timing.get("observation_id")) == str(observation.get("id"))
        and observation.get("id") is not None
        and saved_timing.get("raw_sha256") == observation.get("content_hash")
        and saved_timing.get("raw_sha256") is not None
        and saved_timing.get("archive_return_timing_basis") == "measured_local_utc_archive_save_return"
        and archive_started is not None
        and _iso(saved_timing.get("archive_save_started_at")) == archive_started
        and _time(saved_timing.get("archive_save_returned_at")) is not None
        and _time(archive_started) <= _time(saved_timing.get("archive_save_returned_at")))
    if return_proven:
        archive_returned = _iso(saved_timing.get("archive_save_returned_at"))
        captured, capture_basis = archive_returned, "measured_local_utc_archive_save_return"
    elif document.get("capture_status") == "known" and document.get("capture_timing_basis") == "measured_local_archive_capture":
        captured = _iso(document.get("captured_at"))
        if captured is not None:
            capture_basis = "measured_local_archive_capture"
    proven_handoff = (handoff.get("raw_context_timing_proven") is True
                      and handoff.get("availability_basis") == "runtime_state_handoff")
    membership_proven = proven_handoff and handoff.get("raw_document_membership_proven") is True
    document_id = document.get("document_id")
    admitted = (document_id in handoff.get("known_document_ids", [])
                if membership_proven and isinstance(document_id, str) else None)
    available = _iso(handoff.get("available_at")) if admitted is True else None
    return {
        "observed_at": _iso(observation.get("observed_at")) or _iso(document.get("observed_at")),
        "observed_at_basis": "raw_response_observation" if observation.get("observed_at") else "saved_document_observation",
        "captured_at": captured,
        "capture_status": "known" if captured else "unknown",
        "capture_timing_basis": capture_basis,
        "archive_save_started_at": archive_started,
        "archive_save_returned_at": archive_returned,
        "archive_return_timing_basis": "measured_local_utc_archive_save_return" if return_proven else "unknown",
        "persistence_boundary": "return_observed_fsync_not_instrumented" if return_proven else "unknown",
        "document_admitted": admitted,
        "document_admission_basis": "verified_raw_context_document_membership" if membership_proven else "unknown",
        "document_available_at": available,
        "available_at": available,
        "availability_status": "known" if available else "unknown",
        "availability_basis": "verified_runtime_state_handoff" if available else None,
        "availability_snapshot_id": handoff.get("snapshot_id") if available else None,
        "availability_raw_path": handoff.get("raw_path") if available else None,
        "availability_raw_sha256": handoff.get("raw_sha256") if available else None,
        "source_projection_available_at": _iso(document.get("available_at")),
        "source_projection_available_at_basis": "projection_start_clock_not_agent_handoff",
    }


def _link_document_versions(documents: list[dict[str, Any]]) -> None:
    """Link successive captures of one exact URL only with strict clock order."""
    resources: dict[str, list[dict[str, Any]]] = {}
    for document in documents:
        url = document.get("canonical_url")
        if url and _time(document.get("observed_at")) is not None:
            resources.setdefault(url, []).append(document)
    for versions in resources.values():
        versions.sort(key=lambda row: _time(row["observed_at"]))
        previous = None
        for document in versions:
            if previous is not None and _time(previous["observed_at"]) < _time(document["observed_at"]):
                document["previous_version_id"] = previous.get("version_id") or previous["document_id"]
                document["version_link_basis"] = "same_exact_canonical_url_strict_response_observation_order"
            previous = document


def _receipt_for(match_folder: Path, step: int, selected_id: str | None) -> dict[str, Any]:
    receipt = _load(match_folder / f"receipt-{step:03d}.json", {}) or {}
    if selected_id and receipt.get("action_id") not in {None, selected_id}:
        # Some runtime versions omit action_id in the receipt.  Never guess a
        # different document; the observation is still available for audit.
        return receipt
    return receipt


def _request_provenance(receipt: Mapping[str, Any], observation: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve runtime and journal identities separately from the HTTP ID."""
    context = observation.get("context") or {}
    context = context if isinstance(context, Mapping) else {}
    inner = receipt.get("receipt") or {}
    inner = inner if isinstance(inner, Mapping) else {}
    journal = receipt.get("acquisition_journal") or inner.get("acquisition_journal")
    journal = dict(journal) if isinstance(journal, Mapping) else None
    journal_request_id = context.get("journal_request_id") or (journal or {}).get("request_id")
    if journal is None and (context.get("journal_run_id") or journal_request_id):
        journal = {"run_id": context.get("journal_run_id"), "request_id": journal_request_id,
                   "reference_basis": "raw_observation_context"}
    candidates = (
        (context.get("inquiry_request_id"), "observation.context.inquiry_request_id"),
        (journal_request_id, "acquisition_journal_request_id"),
        (inner.get("request_id"), "receipt.request_id"),
        (observation.get("request_id"), "observation.request_id"),
    )
    request_id, basis = next(((value, origin) for value, origin in candidates if value), (None, None))
    return {"request_id": request_id, "request_id_basis": basis,
            "inquiry_request_id": context.get("inquiry_request_id"),
            "journal_request_id": journal_request_id,
            "transport_request_id": observation.get("request_id"),
            "acquisition_journal": journal}


def _match_run_folders(batch: Path) -> list[Path]:
    """Discover the existing flat layout and one bounded batch nesting level."""
    root = batch.resolve()
    folders = [path for path in batch.glob("match-*") if path.is_dir()]
    folders.extend(path for path in batch.glob("batch-*/match-*") if path.is_dir())
    for path in folders:
        if not path.resolve().is_relative_to(root):
            raise ValueError("Source match folder escapes the batch root")
    return sorted(folders, key=lambda path: path.relative_to(batch).as_posix())


def _journal_source_path(batch: Path, folder: Path, provenance: Mapping[str, Any]) -> str | None:
    journal = provenance.get("acquisition_journal")
    if not isinstance(journal, Mapping) or journal.get("path_scope") != "batch_relative":
        return None
    root = journal.get("root")
    if not isinstance(root, str) or not root:
        return None
    relative = Path(root.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        return None
    source = (folder.parent / relative).resolve()
    return source.relative_to(batch).as_posix() if source.is_relative_to(batch) else None


def _status(cycle: Mapping[str, Any], receipt: Mapping[str, Any]) -> str:
    admission = cycle.get("admission") or {}
    if admission.get("reason") in {"provider_no_data", "empty_response", "no_results"}:
        return "empty"
    execution = str(cycle.get("execution_status") or receipt.get("status") or "").lower()
    if execution in {"timeout", "timed_out"}:
        return "timeout"
    if execution in {"failed", "error", "http_error"} or receipt.get("status") in {"FAILED", "HTTP_ERROR"}:
        return "failed"
    document = receipt.get("document") or {}
    if document.get("status") in {"NO_DATA", "EMPTY"}:
        return "empty"
    if document or receipt.get("status") in {"OK", "RETURNED"}:
        return "returned"
    return "unknown"


def export_batch(batch_dir: str | Path, output: str | Path, *, mode: str | None = None,
                 arm_assignments: Mapping[str, Mapping[str, Any]] | None = None,
                 inventory_report: Mapping[str, Any] | str | Path | None = None) -> dict[str, Any]:
    """Project one or more existing runtime matches into canonical JSONL.

    ``batch_dir`` is read-only.  ``output`` must not already exist.  The
    return value is a compact report; detailed audit evidence is written to
    ``reports/prospective-capture-report.json`` and ``quality-report.txt``.
    """
    batch, target = Path(batch_dir).resolve(), Path(output)
    if target.exists():
        raise FileExistsError("output exists; exports are immutable")
    if not batch.is_dir():
        raise FileNotFoundError(batch)
    # Preflight the complete bounded source set before creating an output.
    # Multiple runs of one physical match are versions, not independent
    # fixtures; combining them would collide in canonical IDs and bias counts.
    prepared_runs = []
    source_match_paths: dict[str, str] = {}
    for folder in _match_run_folders(batch):
        config = _load(folder / "configuration.json", {}) or {}
        identity = _match_identity(config,
            verified_market_fixture=_verified_market_fixture(config, folder))
        mid = identity["physical_match_id"]
        relative = folder.relative_to(batch).as_posix()
        if mid in source_match_paths:
            raise ValueError(f"Duplicate physical_match_id {mid!r}: {source_match_paths[mid]} and {relative}")
        source_match_paths[mid] = relative
        prepared_runs.append((folder, config, identity))
    target.mkdir(parents=True)
    (target / "labels").mkdir()
    (target / "reports").mkdir()
    (target / "derived").mkdir()
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in DATA_FILES}
    exported_at = datetime.now(timezone.utc).isoformat()
    runs_meta: list[dict[str, Any]] = []
    prospective = 0
    proven_menus = 0
    selected_count = unselected_count = 0
    contexts_before_kickoff = 0
    terminal_reasons: dict[str, str | None] = {}
    final_timings: dict[str, Mapping[str, Any]] = {}
    for folder, config, identity in prepared_runs:
        run = _load(folder / "run.json", {}) or {}
        mid = identity["physical_match_id"]
        kickoff = _time(identity.get("kickoff_at"))
        run_mode = _mode(config, kickoff, requested=mode)
        assignment = dict((arm_assignments or {}).get(mid) or {})
        policy_arm = assignment.get("policy_arm") or config.get("policy_arm") or "not_configured"
        if policy_arm not in POLICY_ARMS:
            raise ValueError(f"invalid policy arm for {mid}: {policy_arm!r}")
        arm_assigned_at = assignment.get("arm_assigned_at") or config.get("arm_assigned_at")
        if arm_assigned_at and kickoff and _time(arm_assigned_at) and not _time(arm_assigned_at) < kickoff:
            raise ValueError(f"policy arm assignment is not pre-kickoff for {mid}")
        registration = identity.get("registered_at")
        identity_binding = ((config.get("supervisor_metadata") or {}).get("verified_fixture_identity")
                            if isinstance(config.get("supervisor_metadata"), Mapping) else {})
        identity_binding = identity_binding if isinstance(identity_binding, Mapping) else {}
        identity_observation_ids = [entry.get("observation_id") for entry in
                                    (identity_binding.get("observations") or [])
                                    if isinstance(entry, Mapping)]
        fixture_raw_rel, fixture_raw_hash, fixture_observation_id = _identity_raw_copy(
            folder, identity_observation_ids, target, config=config)
        if fixture_raw_rel and not identity["fixture_source_url"]:
            fixture_observation = _load((target / fixture_raw_rel).with_name(
                Path(fixture_raw_rel).name.replace(".body.gz", ".observation.json")), {})
            if isinstance(fixture_observation, Mapping):
                identity["fixture_source_url"] = fixture_observation.get("endpoint") or ""
        rows["fixtures.jsonl"].append({
            "fixture_id": f"{mid}:fixture",
            "physical_match_id": mid,
            "competition": identity["competition"], "season": str(identity["season"]),
            "home_team": identity["home_team"], "away_team": identity["away_team"],
            "kickoff_at": identity["kickoff_at"], "kickoff": identity["kickoff_at"],
            "kickoff_timezone": (_time(identity["kickoff_at"]).tzname() if _time(identity["kickoff_at"]) else None),
            "registered_at": registration, "fixture_registered_at": registration,
            "registration_timing_basis": identity["registration_timing_basis"],
            "registration_source": identity["registration_source"],
            "source_observed_at": identity["source_observed_at"],
            "source_observed_at_basis": identity["source_observed_at_basis"],
            "source_timezone": identity["source_timezone"],
            "original_time_text": identity["original_time_text"],
            "timing_basis": identity["timing_basis"],
            "fixture_source_url": identity["fixture_source_url"],
            "identity_source_url": identity["fixture_source_url"],
            "information_mode": run_mode, "policy_arm": policy_arm,
            "arm_assigned_at": arm_assigned_at,
            "policy_version": assignment.get("policy_version") or config.get("policy_version") or config.get("policy_arm_version") or "not_configured",
            "random_seed": assignment.get("random_seed", config.get("random_seed")),
            "propensity_logging": assignment.get("propensity_logging", config.get("propensity_logging", False)),
            "raw_path": fixture_raw_rel, "raw_sha256": fixture_raw_hash,
            "raw_observation_id": fixture_observation_id,
            "pre_kickoff_frozen": bool(_time(registration) and kickoff and _time(registration) < kickoff),
        })
        if run_mode == "prospective_capture":
            prospective += 1
        cycles = run.get("cycles") or []
        if not isinstance(cycles, list):
            cycles = []
        terminal = run.get("terminal_selection") or run.get("terminal") or {}
        terminal_reasons[mid] = terminal.get("reason") if isinstance(terminal, Mapping) else None
        final_timing = run.get("final_state_timing") or {}
        if isinstance(final_timing, Mapping):
            final_timings[mid] = {**final_timing, **_context_raw_fields(batch, folder, target, final_timing,
                snapshot_kind="final", run_id=config.get("run_id")),
                **_context_cutoff_fields(config, final_timing, _iso(final_timing.get("observed_at")))}
            final_payload = _context_payload(target, final_timings[mid])
            final_timings[mid]["raw_document_membership_proven"] = bool(
                final_timings[mid].get("raw_context_timing_proven")
                and isinstance(final_payload.get("documents"), list))
            final_timings[mid]["projected_state"] = (final_payload.get("state")
                if isinstance(final_payload.get("state"), Mapping) else run.get("final_state") or {})
            final_timings[mid]["projected_documents"] = _context_documents(
                final_payload, final_timings[mid]["projected_state"], run.get("documents"))
            final_timings[mid]["projected_config"] = config
            final_timings[mid]["projected_prediction"] = _probabilities({"prediction_before": run.get("final_probabilities")})
            final_timings[mid]["projected_predictor_fields"] = _predictor_context_fields(
                config, final_payload, final_timings[mid])
        all_candidates: dict[str, dict[str, Any]] = {}
        for step, cycle in enumerate(cycles):
            if not isinstance(cycle, Mapping):
                continue
            decision_at = (cycle.get("decision_at") or (cycle.get("public_before") or {}).get("decision_at") or
                           (cycle.get("public_before") or {}).get("observed_at"))
            decision_at = _iso(decision_at)
            menu_id = f"menu-{mid}-{step:04d}"
            selected = (cycle.get("selected_action") or {}).get("action_id")
            if not selected:
                selected = (cycle.get("exploration_record") or {}).get("chosen_id")
            exploration = cycle.get("exploration_record") or {}
            probabilities = cycle.get("selection_distribution")
            if not isinstance(probabilities, Mapping):
                probabilities = exploration.get("probabilities") if isinstance(exploration, Mapping) else {}
            if not isinstance(probabilities, Mapping):
                probabilities = {}
            # A pre-trajectory run does not prove that its sampler was the
            # declared policy arm.  Keep those old probabilities out of the
            # propensity field instead of upgrading them into evidence.
            if run_mode != "prospective_capture" or policy_arm == "not_configured":
                probabilities = {}
            candidates = _candidate_rows(cycle)
            recorded_menu = {item.get("action_id"): item for item in cycle.get("menu", [])
                             if isinstance(item, Mapping)}
            for rank, action in enumerate(candidates):
                all_candidates[action["action_id"]] = {**action, "rank": rank,
                    "selected": action["action_id"] == selected,
                    "selection_probability": probabilities.get(action["action_id"])}
            visible = []
            from .pre_read_metadata import candidate_pre_read_metadata
            for rank, action in enumerate(candidates):
                action_id = action["action_id"]
                candidate_id = f"{mid}:{step}:{action_id}"
                recorded = recorded_menu.get(action_id) or {}
                probability = probabilities.get(action_id)
                probability_basis = (recorded.get("selection_probability_basis") or
                                     cycle.get("selection_probability_basis") or exploration.get("probability_basis"))
                all_candidates[action_id]["candidate_id"] = candidate_id
                visible.append({
                    "candidate_id": candidate_id,
                    "action_id": action_id,
                    "action_type": action.get("tool") or action.get("information_key") or "unknown",
                    "url": action.get("url") or action.get("query") or action_id,
                    "query": action.get("query"),
                    "title": action.get("title") or action.get("query") or action_id,
                    "snippet": action.get("snippet") or action.get("query") or "",
                    "rank": rank,
                    "selected_by_policy": action_id == selected,
                    "selection_probability": probability,
                    "selection_probability_basis": probability_basis if probability is not None else "not_recorded",
                    "not_selected_reason": recorded.get("not_selected_reason"),
                    "estimated_cost": action.get("cost"),
                    "cost_status": action.get("cost_status"),
                    **candidate_pre_read_metadata(cycle, action),
                })
            menu_raw_rel = _raw_copy(batch, folder,
                cycle.get("menu_raw_path") or cycle.get("raw_menu_path"), target)
            menu_raw_hash = cycle.get("menu_raw_sha256") or cycle.get("raw_menu_sha256")
            menu_proven = run_mode == "prospective_capture" and _menu_proven(
                target, menu_raw_rel, menu_raw_hash, config, cycle, step, decision_at, kickoff)
            discovery_available = decision_at if menu_proven else None
            for candidate in visible:
                candidate["pre_read_metadata"].update(menu_raw_path=menu_raw_rel, menu_raw_sha256=menu_raw_hash)
            rows["discovery.jsonl"].append({
                "discovery_id": f"discovery-{mid}-{step:04d}", "physical_match_id": mid,
                "observed_at": decision_at, "available_at": discovery_available,
                "availability_status": "known" if discovery_available else "unknown",
                "availability_basis": "agent_decision_clock" if discovery_available else None,
                "query_or_parent": None, "query": None, "parent_url": None,
                "provider": "runtime.generated_actions", "visible_candidates": visible,
                **_visible_document_links(cycle),
                "menu_id": menu_id, "policy_version": config.get("policy_version") or config.get("policy_arm_version") or "not_configured",
                "policy_arm": policy_arm,
                "original_agent_menu_proven": menu_proven,
                "information_mode": run_mode, "raw_path": menu_raw_rel, "raw_sha256": menu_raw_hash,
                "status": "returned", "truncation_or_cap": None,
            })
            if menu_proven:
                proven_menus += 1
            receipt = _receipt_for(folder, step, selected)
            receipt_path = folder / f"receipt-{step:03d}.json"
            obs = (receipt.get("receipt") or {}).get("observation") or {}
            request_provenance = _request_provenance(receipt, obs)
            document = receipt.get("document") or {}
            result_visibility = _result_visibility_fields(cycle)
            raw_rel = _raw_copy(batch, folder, obs.get("payload_path"), target)
            raw_hash = obs.get("content_hash")
            selected_row = None
            selected_attempt_id = None
            for action in visible:
                action_id = action.get("action_id")
                chosen = action.get("selected_by_policy") is True
                if chosen:
                    requested = (_iso(obs.get("context", {}).get("requested_at")) or
                                 _iso(cycle.get("request_started_at")))
                    response_observed = _iso(obs.get("observed_at"))
                    completed = _iso(cycle.get("returned_at")) or response_observed
                    cost = cycle.get("actual_cost") or {}
                    requests = cost.get("requests")
                    if type(requests) in {int, float} and requests >= 0 and requests == int(requests):
                        requests = int(requests)
                    status = _status(cycle, receipt)
                    selected_row = {
                        "attempt_id": cycle.get("attempt_id") or f"attempt-{mid}-{step:04d}-{action_id}",
                        "physical_match_id": mid,
                        "discovery_id": f"discovery-{mid}-{step:04d}",
                        "candidate_id": action["candidate_id"], "action_id": action_id,
                        "tool": (cycle.get("selected_action") or {}).get("tool") or action.get("action_type"),
                        "source": ((cycle.get("selected_action") or {}).get("source")
                                   or action.get("source") or action.get("action_type") or "unknown"),
                        "started_at": requested, "completed_at": completed,
                        "requested_at": requested, "request_started_at": requested,
                        "response_observed_at": response_observed, "response_at": response_observed,
                        **result_visibility,
                        **request_provenance,
                        "source_journal_root_path": _journal_source_path(batch, folder, request_provenance),
                        "source_receipt_path": receipt_path.relative_to(batch).as_posix() if receipt_path.is_file() else None,
                        "source_receipt_sha256": _sha256(receipt_path) if receipt_path.is_file() else None,
                        "kickoff_at": identity.get("kickoff_at"),
                        "information_mode": run_mode,
                        "decision_at": decision_at, "status": status,
                        "selected_by_policy": True,
                        "selection_basis": (cycle.get("selection_basis") if run_mode == "prospective_capture" else "not_configured") or "not_configured",
                        "policy_version": config.get("policy_version") or config.get("policy_arm_version") or "not_configured", "policy_arm": policy_arm,
                        "f_consumption_required": _f_consumption_required(config),
                        "selection_probability": action.get("selection_probability"),
                        "selection_probability_basis": action.get("selection_probability_basis") or "not_recorded",
                        "http_status": obs.get("http_status"), "error_code": obs.get("error_class"),
                        "returned_version_ids": [document.get("document_id")] if document.get("document_id") else [],
                        "returned_document_id": document.get("document_id"),
                        "request_count": requests,
                        "latency_ms": cost.get("latency_ms"),
                        "token_usage_if_available": cost.get("tokens"), "monetary_cost_if_known": cost.get("money"),
                        "raw_path": raw_rel, "raw_sha256": raw_hash,
                        "raw_digest_verified": _verified_archive_body(target, raw_rel, raw_hash) is not None,
                    }
                    selected_attempt_id = selected_row["attempt_id"]
                    selected_count += 1
                else:
                    selected_row = {
                        "attempt_id": f"not-requested-{mid}-{step:04d}-{action_id}",
                        "physical_match_id": mid, "discovery_id": f"discovery-{mid}-{step:04d}",
                        "candidate_id": action["candidate_id"], "action_id": action_id,
                        "tool": action.get("action_type"), "source": action.get("action_type") or "not_requested",
                        "started_at": None, "completed_at": None, "requested_at": None,
                        "response_observed_at": None, "decision_at": decision_at,
                        "result_observed": False, "result_observed_at": None, "receipt_available_at": None,
                        "receipt_availability_basis": "not_requested", "document_admitted": None,
                        "document_available_at": None, "document_admission_basis": "not_requested",
                        "kickoff_at": identity.get("kickoff_at"), "information_mode": run_mode,
                        "f_consumption_required": _f_consumption_required(config),
                        "status": "not_requested", "selected_by_policy": False,
                        "selection_basis": (cycle.get("selection_basis") if run_mode == "prospective_capture" else "not_configured") or "not_configured",
                        "policy_version": config.get("policy_version") or config.get("policy_arm_version") or "not_configured", "policy_arm": policy_arm,
                        "selection_probability": action.get("selection_probability"),
                        "selection_probability_basis": action.get("selection_probability_basis") or "not_recorded",
                        "not_selected_reason": action.get("not_selected_reason") or "not_selected_by_policy",
                        "http_status": None, "error_code": None, "returned_version_ids": [],
                        "request_count": 0, "latency_ms": None, "token_usage_if_available": None,
                        "monetary_cost_if_known": None, "raw_path": None, "raw_sha256": None,
                    }
                    unselected_count += 1
                rows["attempts.jsonl"].append(selected_row)
            before = cycle.get("public_before") or {}
            public_after = cycle.get("public_after") or {}
            after = cycle.get("public_state_after") or public_after
            # The public after-action projection is the primary consumption
            # record; ``public_state_after`` is a state slice that usually
            # carries no ``prediction`` and must never hide it.
            before_consumption = _consumption_projection((before,))
            after_consumption = _consumption_projection((public_after, after))
            before_prob = _probabilities(cycle)
            # ``prediction_after`` was historically serialized as a list and
            # newer predictor contexts serialize named probabilities.  Use
            # the same strict normalizer for both; never copy ``before`` or
            # renormalize an invalid after value.
            after_prob = _prediction_mapping(cycle.get("prediction_after"))
            for kind, state, consumption, parent, observed, probability, timing in (
                ("before_action", before, before_consumption, None, decision_at, before_prob,
                 cycle.get("before_state_timing") or {}),
                ("after_action", after, after_consumption, selected_attempt_id,
                 _iso(cycle.get("returned_at")) or _iso(obs.get("observed_at")), after_prob,
                 cycle.get("after_state_timing") or {}),
            ):
                if observed is None:
                    continue
                if isinstance(timing, Mapping):
                    observed = _iso(timing.get("observed_at")) or observed
                    captured = _iso(timing.get("captured_at"))
                    available = _iso(timing.get("available_at"))
                    availability_status = timing.get("availability_status") or ("known" if available else "unknown")
                else:
                    captured, available, availability_status = None, None, "unknown"
                context_raw = _context_raw_fields(batch, folder, target, timing if isinstance(timing, Mapping) else {},
                    snapshot_kind=kind, run_id=config.get("run_id"))
                context_payload = _context_payload(target, context_raw)
                context_state = context_payload.get("state") if isinstance(context_payload.get("state"), Mapping) else state
                context_state = context_state if isinstance(context_state, Mapping) else {}
                context_documents = _context_documents(context_payload, context_state)
                field_values = _context_fields(context_state, config)
                predictor_fields = _predictor_context_fields(config, context_payload, context_raw, consumption)
                if kind == "after_action" and predictor_fields.get("f_consumption_status") == "unknown":
                    # A projection without an explicit ledger leaves the record
                    # unknown; the runtime's own same-cycle ledger may then
                    # supply it verbatim.  Only the after-action handoff may
                    # read it, because it describes that action's f evaluation.
                    predictor_fields = {**predictor_fields, **(_cycle_consumption_ledger(cycle) or {})}
                rows["context_snapshots.jsonl"].append({
                    "snapshot_id": f"snapshot-{mid}-{step:04d}-{kind}", "physical_match_id": mid,
                    "sequence": step * 2 + (1 if kind == "after_action" else 0),
                    "snapshot_kind": kind, "parent_attempt_id": parent,
                    "observed_at": observed, "as_of": observed, "captured_at": captured,
                    "available_at": available, "availability_status": availability_status,
                    "availability_basis": timing.get("availability_basis") if isinstance(timing, Mapping) else None,
                    **context_raw,
                    "raw_document_membership_proven": bool(context_raw.get("raw_context_timing_proven")
                        and isinstance(context_payload.get("documents"), list)),
                    **_context_cutoff_fields(config, timing if isinstance(timing, Mapping) else {}, observed),
                    **_context_information(context_state, field_values, context_documents),
                    **_forecast_fields(config, before_prob, after_prob if kind == "after_action" else None),
                    **predictor_fields,
                    **field_values,
                    "state_delta": None, "stop_reason": None, "information_mode": run_mode,
                })
                if kickoff and _time(observed) and _time(observed) < kickoff:
                    contexts_before_kickoff += 1
            handoff = next((row for row in reversed(rows["context_snapshots.jsonl"])
                if row.get("parent_attempt_id") == selected_attempt_id
                and selected_attempt_id is not None and row.get("snapshot_kind") == "after_action"), {})
            document_timing = _document_timing_fields(document, obs, handoff, receipt.get("receipt") or {})
            admission_fields = {key: document_timing[key] for key in (
                "document_admitted", "document_admission_basis", "document_available_at")}
            selected_attempt = next((row for row in reversed(rows["attempts.jsonl"])
                if selected_attempt_id is not None and row.get("attempt_id") == selected_attempt_id), None)
            if selected_attempt is not None:
                selected_attempt.update(admission_fields)
                selected_attempt.update({key: document_timing[key] for key in (
                    "archive_save_started_at", "archive_save_returned_at", "archive_return_timing_basis", "persistence_boundary")})
            if handoff:
                handoff.update({**result_visibility, **admission_fields,
                    "returned_document_id": document.get("document_id")})
                if selected_attempt is not None:
                    selected_attempt.update({key: handoff.get(key) for key in (
                        "f_consumed_input_ids", "f_consumption_status", "f_consumption_basis",
                        "f_modeled_read_documents", "f_consumption_required")})
            if document.get("document_id") and document.get("text") is not None:
                rows["documents.jsonl"].append({
                    "document_id": document["document_id"], "version_id": document["document_id"],
                    "physical_match_ids": [mid], "physical_match_id": mid,
                    "canonical_url": document.get("url") or obs.get("endpoint") or "",
                    "url": document.get("url") or obs.get("endpoint") or "",
                    "publisher": document.get("publisher") or obs.get("source") or "unknown",
                    "title": document.get("title") or "", "body_text": document.get("text") or "",
                    "body": document.get("text") or "", "language": document.get("language") or "unknown",
                    **_document_link_metadata(document),
                    "source_published_at": _iso(document.get("source_published_at")),
                    "source_modified_at": _iso(document.get("source_modified_at")),
                    **document_timing,
                    **result_visibility,
                    "source_time_raw": document.get("source_time_raw") or "",
                    "original_time_text": document.get("source_time_raw") or "",
                    "timezone_basis": document.get("timezone_basis") or "source_declared_or_unknown",
                    "timing_basis": document.get("timing_basis") or "source_declared_or_unknown",
                    "time_precision": document.get("time_precision") or "unknown",
                    "time_evidence": document.get("time_evidence") or "runtime_projection",
                    "information_mode": run_mode, "archive_snapshot_at": None,
                    "archive_snapshot_url": None, "historical_snapshot_proven": run_mode != "prospective_capture",
                    "raw_path": raw_rel, "raw_sha256": raw_hash, "previous_version_id": None,
                    "document_type": document.get("kind") or "unknown", "document_kind": document.get("kind") or "unknown",
                    "match_binding_basis": "runtime_fixture_binding", "fixture_binding_basis": "runtime_fixture_binding",
                    "duplicate_cluster_id": f"cluster-{document['document_id']}",
                })
        # Ensure an initial snapshot exists even when candidate generation had no cycles.
        # Every run receives an explicit initial state snapshot.  It may have
        # null probabilities when the model was not configured, but its clock
        # remains the actual frozen configuration clock.
        initial_state = config.get("initial_state") or {}
        initial_timing = config.get("initial_state_timing") or {}
        initial_timing = initial_timing if isinstance(initial_timing, Mapping) else {}
        initial_time = _iso(initial_timing.get("observed_at")) or _iso(config.get("created_at"))
        initial_available = _iso(initial_timing.get("available_at"))
        initial_clocks = {"observed_at": initial_time, "as_of": initial_time,
                          "captured_at": _iso(initial_timing.get("captured_at")),
                          "available_at": initial_available,
                          "availability_status": "known" if initial_available else "unknown",
                          "availability_basis": initial_timing.get("availability_basis"),
                          **_context_raw_fields(batch, folder, target, initial_timing,
                              snapshot_kind="initial", run_id=config.get("run_id")),
                          **_context_cutoff_fields(config, initial_timing, initial_time)}
        initial_payload = _context_payload(target, initial_clocks)
        initial_clocks["raw_document_membership_proven"] = bool(
            initial_clocks.get("raw_context_timing_proven")
            and isinstance(initial_payload.get("documents"), list))
        initial_state = (initial_payload.get("state") if isinstance(initial_payload.get("state"), Mapping)
                         else initial_state if isinstance(initial_state, Mapping) else {})
        initial_fields = _context_fields(initial_state, config)
        initial_information = _context_information(
            initial_state, initial_fields, _context_documents(initial_payload, initial_state))
        if cycles:
            rows["context_snapshots.jsonl"].insert(0, {
                "snapshot_id": f"snapshot-{mid}-initial", "physical_match_id": mid,
                "sequence": 0, "snapshot_kind": "initial", "parent_attempt_id": None,
                **initial_clocks,
                **initial_information,
                **_forecast_fields(config, _probabilities({"prediction_before": (initial_state.get("prediction") or {}).get("probabilities")}) if isinstance(initial_state, Mapping) else None),
                **_predictor_context_fields(config, initial_payload, initial_clocks),
                **initial_fields,
                "state_delta": None, "stop_reason": None, "information_mode": run_mode,
            })
        else:
            rows["context_snapshots.jsonl"].append({
                "snapshot_id": f"snapshot-{mid}-initial", "physical_match_id": mid,
                "sequence": 0, "snapshot_kind": "initial", "parent_attempt_id": None,
                **initial_clocks,
                **initial_information,
                **_forecast_fields(config, None),
                **_predictor_context_fields(config, initial_payload, initial_clocks),
                **initial_fields,
                "state_delta": None,
                "stop_reason": terminal_reasons.get(mid), "information_mode": run_mode,
            })
        runs_meta.append({"physical_match_id": mid, "mode": run_mode, "policy_arm": policy_arm,
                          "source_run_path": source_match_paths[mid],
                          "source_configuration_sha256": (_sha256(folder / "configuration.json")
                              if (folder / "configuration.json").is_file() else None),
                          "source_run_sha256": (_sha256(folder / "run.json")
                              if (folder / "run.json").is_file() else None),
                          "kickoff_at": identity.get("kickoff_at"), "cycles": len(cycles)})

    # Add initial/final context rows when the runtime had cycles and mark the
    # terminal reason without inventing a prediction.
    for meta in runs_meta:
        mid = meta["physical_match_id"]
        matching = [row for row in rows["context_snapshots.jsonl"] if row.get("physical_match_id") == mid]
        if matching:
            last = matching[-1]
            final = dict(last)
            # Final state has its own raw proof.  A previous action receipt's
            # visibility/admission event must not leak into this later snapshot.
            for key in ("result_observed", "result_observed_at", "receipt_available_at",
                        "receipt_availability_basis", "document_admitted", "document_admission_basis",
                        "document_available_at", "returned_document_id"):
                final.pop(key, None)
            final.update({"snapshot_id": f"snapshot-{mid}-final", "snapshot_kind": "final",
                          "sequence": max(row.get("sequence", 0) for row in matching) + 1,
                          "stop_reason": terminal_reasons.get(mid) or "run_complete"})
            timing = final_timings.get(mid) or {}
            final.update({"observed_at": _iso(timing.get("observed_at")),
                          "as_of": _iso(timing.get("observed_at")),
                          "captured_at": _iso(timing.get("captured_at")),
                          "available_at": _iso(timing.get("available_at")),
                          "availability_status": "known" if _iso(timing.get("available_at")) else "unknown",
                          "availability_basis": timing.get("availability_basis"),
                          "prediction_cutoff": timing.get("prediction_cutoff"),
                          "prediction_cutoff_basis": timing.get("prediction_cutoff_basis", "not_recorded"),
                          "acquisition_deadline": timing.get("acquisition_deadline"),
                          "raw_path": timing.get("raw_path"), "raw_sha256": timing.get("raw_sha256"),
                          "raw_observation_id": timing.get("raw_observation_id"),
                          "raw_digest_verified": timing.get("raw_digest_verified", False),
                          "raw_context_timing_proven": timing.get("raw_context_timing_proven", False),
                          "raw_document_membership_proven": timing.get("raw_document_membership_proven", False),
                          **{key: timing.get(key) for key in ("raw_context_timing_basis", "context_capture_completion_proven",
                              "archive_save_started_at", "archive_save_returned_at", "capture_timing_basis", "persistence_boundary",
                              "capture_status", "source_context_captured_at", "source_context_captured_at_basis",
                              "raw_context_schema", "forecast_as_of", "forecast_eligible", "recording_scope")}})
            for key in ("prediction_pipeline_version", "prediction_processing"):
                if key in timing:
                    final[key] = deepcopy(timing[key])
                else:
                    final.pop(key, None)
            final_state = timing.get("projected_state") or {}
            final_config = timing.get("projected_config") or {}
            final_fields = _context_fields(final_state, final_config)
            final.update({**final_fields,
                          **_context_information(final_state, final_fields, timing.get("projected_documents")),
                          **_forecast_fields(final_config, timing.get("projected_prediction")),
                          **(timing.get("projected_predictor_fields") or {}),
                          "parent_attempt_id": timing.get("parent_attempt_id")})
            rows["context_snapshots.jsonl"].append(final)

    for values in rows.values():
        for row in values:
            mid = row.get("physical_match_id")
            if mid in source_match_paths:
                row["source_run_path"] = source_match_paths[mid]
    _link_document_versions(rows["documents.jsonl"])
    for relative in DATA_FILES:
        _write_jsonl(target / relative, rows[relative])
    # Derive candidates only after all six canonical files are assembled.  The
    # helper applies the same temporal/menu/raw/result gates to prospective and
    # historical rows and leaves ineligible rows auditable.
    from .prospective_outcomes import derive_g_candidates
    derived_rows = derive_g_candidates(
        fixtures=rows["fixtures.jsonl"], discoveries=rows["discovery.jsonl"],
        attempts=rows["attempts.jsonl"], contexts=rows["context_snapshots.jsonl"],
        results=rows["labels/results.jsonl"], documents=rows["documents.jsonl"], archive_root=target)
    _write_jsonl(target / "derived" / "g-training-candidates.jsonl", derived_rows)
    files: dict[str, dict[str, Any]] = {}
    for relative in DATA_FILES + ("derived/g-training-candidates.jsonl",):
        path = target / relative
        files[relative] = {"path": relative, "sha256": _sha256(path),
                           "row_count": sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())}
    report = {
        "schema": "footballmind-prospective-stream-v2",
        "status": "exported",
        "source_batch": str(batch.resolve()),
        "fixtures_registered": len(rows["fixtures.jsonl"]),
        "fixtures_before_kickoff": sum(1 for row in rows["fixtures.jsonl"] if row.get("pre_kickoff_frozen")),
        "menus_observed_before_kickoff": sum(1 for row in rows["discovery.jsonl"] if _time(row.get("observed_at")) and _time(row.get("observed_at")) < _time(next((f.get("kickoff_at") for f in rows["fixtures.jsonl"] if f.get("physical_match_id") == row.get("physical_match_id")), None))),
        "original_agent_menus_proven": proven_menus,
        "attempts_before_kickoff": sum(1 for row in rows["attempts.jsonl"] if row.get("decision_at")),
        "selected_actions": selected_count, "unselected_actions": unselected_count,
        # Count the complete canonical context stream, including the explicit
        # initial and final snapshots appended after cycle projection.  The
        # earlier cycle-local counter intentionally remains available while
        # building rows, but must not under-report the exported file.
        "contexts_before_kickoff": sum(
            1 for row in rows["context_snapshots.jsonl"]
            if _time(row.get("observed_at"))
            and _time(row.get("observed_at")) < _time(next(
                (f.get("kickoff_at") for f in rows["fixtures.jsonl"]
                 if f.get("physical_match_id") == row.get("physical_match_id")),
                None,
            ))
        ),
        "outcomes_after_kickoff": len(rows["labels/results.jsonl"]),
        "raw_digest_checks": f"{sum(1 for row in rows['attempts.jsonl'] if row.get('raw_sha256'))}/{sum(1 for row in rows['attempts.jsonl'] if row.get('raw_path'))}",
        # These counts must describe the derived canonical labels written just
        # above.  The old exporter hard-coded eligible=0, which made a valid
        # settled row look ineligible until a later finalizer happened to
        # rewrite the report.
        "g_training_eligible_rows": sum(1 for row in derived_rows if row.get("label_status") == "eligible"),
        "g_training_ineligible_rows": sum(1 for row in derived_rows if row.get("label_status") == "ineligible"),
        "g_training_count_basis": "derived/g-training-candidates.jsonl.label_status",
        "available_prospective_fixtures": sum(1 for row in rows["fixtures.jsonl"] if row.get("information_mode") == "prospective_capture"),
        "shortage_reason": "no verified pre-kickoff prospective fixtures in the supplied batch" if prospective == 0 else None,
        "runs": runs_meta,
    }
    inventory = _inventory_provenance(inventory_report, base_dir=batch)
    if inventory is not None:
        report.update(inventory_provenance=inventory,
            inventory_report=inventory["path"], inventory_run_id=inventory["run_id"],
            inventory_report_sha256=inventory["sha256"])
    modes = sorted({str(row.get("information_mode")) for row in rows["fixtures.jsonl"]
                    if row.get("information_mode")})
    manifest_information_mode = modes[0] if len(modes) == 1 else ("mixed" if modes else None)
    (target / "manifest.json").write_text(json.dumps({"schema": "footballmind-prospective-stream-v2", "schema_version": "footballmind-prospective-stream-v2",
        "information_mode": manifest_information_mode,
        "source": str(batch.resolve()), "files": files, "report": report}, ensure_ascii=False,
        sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (target / "reports" / "prospective-capture-report.json").write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    quality = ["Prospective Information Stream Status", "", *(f"{key}: {value}" for key, value in report.items() if key not in {"runs"}),
               "", "empirical_g: deferred", "historical reconstruction rows are excluded from online empirical_g."]
    (target / "quality-report.txt").write_text("\n".join(quality) + "\n", encoding="utf-8")
    from .prospective_stream_reports import finalize_stream_package
    return finalize_stream_package(target, report)


def export_empty_package(output: str | Path, *, inventory_report: Mapping[str, Any] | str | Path | None = None,
                         shortage_reason: str = "no_future_premier_league_rows_in_spftll_current_window") -> dict[str, Any]:
    """Create a valid, auditable empty prospective package.

    This is used when the live, pre-kickoff inventory contains no qualifying
    fixture.  It writes the canonical files and provenance report, but never
    invents a fixture, menu, action, timestamp, or outcome.  Existing output
    directories are immutable and are never overwritten.
    """
    target = Path(output)
    if target.exists():
        raise FileExistsError("output exists; exports are immutable")
    target.mkdir(parents=True)
    (target / "labels").mkdir()
    (target / "reports").mkdir()
    (target / "derived").mkdir()
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in DATA_FILES}
    for relative in DATA_FILES:
        _write_jsonl(target / relative, rows[relative])
    _write_jsonl(target / "derived" / "g-training-candidates.jsonl", [])
    files: dict[str, dict[str, Any]] = {}
    for relative in DATA_FILES + ("derived/g-training-candidates.jsonl",):
        path = target / relative
        files[relative] = {"path": relative, "sha256": _sha256(path), "row_count": 0}
    inventory = _inventory_provenance(inventory_report)
    report = {
        "schema": "footballmind-prospective-stream-v2", "status": "empty_inventory",
        "fixtures_registered": 0, "fixtures_before_kickoff": 0,
        "menus_observed_before_kickoff": 0, "original_agent_menus_proven": 0,
        "attempts_before_kickoff": 0, "selected_actions": 0, "unselected_actions": 0,
        "contexts_before_kickoff": 0, "outcomes_after_kickoff": 0,
        "raw_digest_checks": "0/0", "g_training_eligible_rows": 0,
        "g_training_ineligible_rows": 0, "available_prospective_fixtures": 0,
        "shortage_reason": shortage_reason,
        "inventory_report": inventory["path"] if inventory else None,
        "inventory_run_id": inventory["run_id"] if inventory else None,
        "inventory_report_sha256": inventory["sha256"] if inventory else None,
        "inventory_provenance": inventory,
    }
    manifest = {"schema": "footballmind-prospective-stream-v2",
                "schema_version": "footballmind-prospective-stream-v2",
                "information_mode": "prospective_capture", "files": files,
                "report": report}
    (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (target / "reports" / "prospective-capture-report.json").write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    quality = ["Prospective Information Stream Status", "", *(f"{key}: {value}" for key, value in report.items()),
               "", "available prospective fixtures: 0", f"shortage reason: {shortage_reason}",
               "empirical_g: deferred", "No candidate menu or action was fabricated."]
    (target / "quality-report.txt").write_text("\n".join(quality) + "\n", encoding="utf-8")
    from .prospective_stream_reports import finalize_stream_package
    return finalize_stream_package(target, report)


__all__ = ["DATA_FILES", "POLICY_ARMS", "assign_policy_arm", "export_batch", "export_empty_package"]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("prospective_capture", "archived_snapshot", "historical_reconstruction"))
    parser.add_argument("--inventory-report", type=Path,
                        help="Existing fixture inventory report to bind by path, run ID and SHA-256")
    args = parser.parse_args()
    print(json.dumps(export_batch(args.batch, args.output, mode=args.mode,
        inventory_report=args.inventory_report), ensure_ascii=False, indent=2))
