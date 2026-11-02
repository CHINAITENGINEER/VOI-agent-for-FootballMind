"""Offline preflight for an entity-semantic f before/after collection.

The preflight is deliberately a projection audit.  It reads an existing run,
its immutable menu observations and (optionally) its canonical stream; it never
starts a collector, calls a model provider, settles a result, or trains g.
The output is a new, immutable audit directory and the input run is left
byte-for-byte untouched.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import socket
import sys
from typing import Any, Iterator, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


V4_PROCESSING_SCHEMA = "entity-semantic-prediction-processing-v1"
V4_CONTEXT_SCHEMA = "runtime-predictor-context-v4"


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if result.tzinfo is None or result.utcoffset() is None:
        return None
    return result


def _probabilities(value: Any) -> list[float] | None:
    """Extract and validate a public 1X2 probability vector.

    The runner only accepts an explicit vector.  It never reconstructs a
    forecast from text, odds, a title, or a retained result.
    """
    candidate = value
    if isinstance(candidate, Mapping):
        if set(candidate) == {"home", "draw", "away"}:
            candidate = [candidate[key] for key in ("home", "draw", "away")]
        elif "probabilities" in candidate:
            candidate = candidate["probabilities"]
        elif "prediction" in candidate:
            return _probabilities(candidate["prediction"])
    if not isinstance(candidate, (list, tuple)) or len(candidate) != 3:
        return None
    if any(type(item) not in {int, float} or not math.isfinite(float(item))
           or not 0 <= float(item) <= 1 for item in candidate):
        return None
    values = [float(item) for item in candidate]
    if abs(sum(values) - 1.0) > 1e-7:
        return None
    return values


def _find_run(path: Path) -> tuple[Path, Path]:
    path = path.resolve()
    if path.is_file():
        if path.name != "run.json":
            raise ValueError("--run must be a run directory or run.json")
        return path.parent, path.parent.parent
    direct = path / "run.json"
    if direct.is_file():
        return path, path.parent
    matches = sorted(path.rglob("run.json")) if path.is_dir() else []
    if len(matches) != 1:
        raise ValueError("--run must resolve to exactly one run.json")
    return matches[0].parent, matches[0].parent.parent


def _load_raw(path: Path) -> bytes:
    data = path.read_bytes()
    return gzip.decompress(data) if path.suffix.lower() == ".gz" else data


def _raw_candidates(run_dir: Path, source_root: Path, raw_path: Any) -> list[Path]:
    if not isinstance(raw_path, str) or not raw_path:
        return []
    rel = Path(raw_path.replace("\\", "/"))
    # RawArchive payload paths are relative to the archive root.  Keep all
    # fallbacks local and deterministic; no URL/path is ever fetched.
    return [
        run_dir / rel,
        run_dir / "source-archive" / rel,
        source_root / rel,
        source_root / "source-archive" / rel,
    ]


def _raw_menu_check(cycle: Mapping[str, Any], run_dir: Path, source_root: Path,
                    errors: list[str], warnings: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "unavailable", "path": cycle.get("menu_raw_path"),
                              "observation_id": cycle.get("menu_raw_observation_id"),
                              "declared_sha256": cycle.get("menu_raw_sha256")}
    declared = cycle.get("menu_raw_sha256")
    if not isinstance(declared, str) or len(declared) != 64:
        errors.append(f"cycle {cycle.get('step')}: menu_raw_sha256 is required")
        return result
    if not isinstance(cycle.get("menu_raw_observation_id"), str) or not cycle.get("menu_raw_observation_id"):
        errors.append(f"cycle {cycle.get('step')}: menu_raw_observation_id is required")
        return result
    candidates = [item for item in _raw_candidates(run_dir, source_root, cycle.get("menu_raw_path"))
                  if item.is_file()]
    if not candidates:
        errors.append(f"cycle {cycle.get('step')}: retained menu raw payload is missing")
        return result
    path = candidates[0]
    try:
        payload = _load_raw(path)
    except (OSError, EOFError, gzip.BadGzipFile) as exc:
        errors.append(f"cycle {cycle.get('step')}: retained menu payload cannot be read: {exc}")
        return result
    actual = _sha256_bytes(payload)
    result.update(path=str(path), actual_sha256=actual, sha256_match=actual == declared)
    if actual != declared:
        errors.append(f"cycle {cycle.get('step')}: menu raw SHA-256 differs")
        return result
    result["status"] = "verified"
    try:
        parsed = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        warnings.append(f"cycle {cycle.get('step')}: menu raw is not JSON; hash still verified")
        return result
    expected_ids = [item.get("action_id") for item in (cycle.get("generated_actions") or [])
                    if isinstance(item, Mapping)]
    found_ids = None
    for container in (parsed, parsed.get("cycle") if isinstance(parsed, Mapping) else None,
                      parsed.get("menu") if isinstance(parsed, Mapping) else None):
        if not isinstance(container, Mapping):
            continue
        for key in ("generated_actions", "actions", "menu"):
            rows = container.get(key)
            if isinstance(rows, list) and all(isinstance(row, Mapping) and "action_id" in row for row in rows):
                found_ids = [row["action_id"] for row in rows]
                break
        if found_ids is not None:
            break
    if found_ids is None:
        warnings.append(f"cycle {cycle.get('step')}: raw menu JSON has no explicit action list")
    elif found_ids != expected_ids:
        errors.append(f"cycle {cycle.get('step')}: raw menu action order differs from frozen menu")
    else:
        result["menu_action_ids_match"] = True
    return result


def _processing_audit(timing: Mapping[str, Any] | None, *, side: str,
                      cycle: Mapping[str, Any], errors: list[str], warnings: list[str]) -> dict[str, Any]:
    if not isinstance(timing, Mapping):
        errors.append(f"cycle {cycle.get('step')}: {side} timing is missing")
        return {"status": "missing", "side": side}
    processing = timing.get("prediction_processing")
    pipeline = timing.get("prediction_pipeline_version")
    result = {"side": side, "schema": V4_CONTEXT_SCHEMA if processing is not None else "legacy",
              "pipeline_version": pipeline, "processing": processing}
    if processing is None:
        warnings.append(f"cycle {cycle.get('step')}: {side} has no v4 prediction_processing")
        return result
    if processing.get("schema") != V4_PROCESSING_SCHEMA:
        errors.append(f"cycle {cycle.get('step')}: {side} processing schema is not v1")
        return result
    status = processing.get("status")
    if status not in {"ready", "not_updated"}:
        errors.append(f"cycle {cycle.get('step')}: {side} processing status is invalid")
        return result
    observed = _time(timing.get("observed_at") or timing.get("captured_at"))
    input_at = _time(processing.get("input_at"))
    available_at = _time(processing.get("available_at"))
    effective_at = _time(processing.get("effective_at"))
    forecast_as_of = _time(timing.get("forecast_as_of"))
    if status == "ready":
        if processing.get("reason") is not None:
            errors.append(f"cycle {cycle.get('step')}: ready {side} has a failure reason")
        if None in (input_at, available_at, effective_at, forecast_as_of, observed):
            errors.append(f"cycle {cycle.get('step')}: ready {side} is missing a clock")
        elif effective_at != max(input_at, available_at):
            errors.append(f"cycle {cycle.get('step')}: {side} effective_at is not max(input,available)")
        elif forecast_as_of != effective_at:
            errors.append(f"cycle {cycle.get('step')}: {side} forecast_as_of differs from effective_at")
        elif not (input_at <= effective_at <= observed and available_at <= effective_at):
            errors.append(f"cycle {cycle.get('step')}: {side} processing clock order is invalid")
        if side == "after":
            receipt_available = _time(cycle.get("receipt_available_at"))
            if receipt_available is None or input_at != receipt_available:
                errors.append(f"cycle {cycle.get('step')}: after input_at must equal receipt_available_at")
        if timing.get("forecast_eligible") is False or timing.get("recording_scope") == "audit_only":
            errors.append(f"cycle {cycle.get('step')}: ready {side} cannot be audit_only")
    else:
        if not isinstance(processing.get("reason"), str) or not processing["reason"].strip():
            errors.append(f"cycle {cycle.get('step')}: not_updated {side} requires a reason")
        if processing.get("effective_at") is not None:
            errors.append(f"cycle {cycle.get('step')}: not_updated {side} effective_at must be null")
        if timing.get("forecast_eligible") is not False or timing.get("recording_scope") != "audit_only":
            errors.append(f"cycle {cycle.get('step')}: not_updated {side} must be audit_only")
    return result


def _f_consumed(cycle: Mapping[str, Any]) -> tuple[bool | None, str]:
    for key in ("f_consumed", "document_f_consumed"):
        if type(cycle.get(key)) is bool:
            return cycle[key], f"explicit_{key}"
    update = cycle.get("update")
    if isinstance(update, Mapping) and update.get("status") == "not_updated":
        return False, "explicit_update_not_updated"
    if isinstance(update, Mapping) and update.get("status") == "updated":
        semantic = cycle.get("semantic_prediction")
        if isinstance(semantic, Mapping) and update.get("model_version"):
            return True, "explicit_semantic_update_model_version"
    return None, "not_recorded"


def _labels_audit(run: Mapping[str, Any], cycles: list[Mapping[str, Any]], errors: list[str]) -> dict[str, Any]:
    outcome = run.get("outcome")
    run_values = {key: run.get(key) for key in ("realized_signed_gain", "gain", "net_gain")}
    if outcome is None and any(value is not None for value in run_values.values()):
        errors.append("run carries gain fields without an independent outcome")
    rows = []
    for cycle in cycles:
        update = cycle.get("update") if isinstance(cycle.get("update"), Mapping) else {}
        blocked = update.get("status") == "not_updated"
        values = {key: cycle.get(key) for key in ("outcome", "realized_signed_gain", "gain", "net_gain")}
        non_null = [key for key, value in values.items() if value is not None]
        if blocked and non_null:
            errors.append(f"cycle {cycle.get('step')}: failed/late context carries a label")
        rows.append({"step": cycle.get("step"), "outcome": values["outcome"],
                     "realized_signed_gain": values["realized_signed_gain"],
                     "net_gain": values["net_gain"], "label_status": "audit_only" if blocked else "placeholder"})
    if outcome is None:
        status = "placeholder_null_no_run_outcome"
    else:
        status = "source_declared_outcome_preserved"
    return {"run_outcome": outcome, "run_gain_fields": run_values, "status": status, "rows": rows,
            "zero_label_invention_check": not any(
                row["realized_signed_gain"] == 0 for row in rows if row["label_status"] == "audit_only")}


def audit_preflight(run_path: str | Path, *, stream: str | Path | None = None,
                    require_v4: bool = False) -> dict[str, Any]:
    run_dir, source_root = _find_run(Path(run_path))
    run = _read(run_dir / "run.json")
    config_path = run_dir / "configuration.json"
    config = _read(config_path) if config_path.is_file() else {}
    errors: list[str] = []
    warnings: list[str] = []
    cycles = run.get("cycles")
    if not isinstance(cycles, list):
        errors.append("run.cycles must be a list")
        cycles = []
    menu_checks, clock_checks, f_rows = [], [], []
    pipeline_versions: set[str] = set()
    for cycle in cycles:
        if not isinstance(cycle, Mapping):
            errors.append("cycle must be an object")
            continue
        actions = cycle.get("generated_actions")
        if not isinstance(actions, list) or not actions:
            errors.append(f"cycle {cycle.get('step')}: generated_actions menu is empty")
            actions = []
        ids = [item.get("action_id") for item in actions if isinstance(item, Mapping)]
        if len(ids) != len(set(ids)) or any(not isinstance(item, str) or not item for item in ids):
            errors.append(f"cycle {cycle.get('step')}: action IDs are not unique/inert")
        menu = cycle.get("menu")
        if not isinstance(menu, list) or [item.get("action_id") for item in menu if isinstance(item, Mapping)] != ids:
            errors.append(f"cycle {cycle.get('step')}: menu does not exactly cover generated_actions")
        selected = [item for item in menu or [] if isinstance(item, Mapping) and item.get("selected") is True]
        selected_action = cycle.get("selected_action")
        selected_id = selected_action.get("action_id") if isinstance(selected_action, Mapping) else None
        if len(selected) > 1 or (selected and selected_id != selected[0].get("action_id")):
            errors.append(f"cycle {cycle.get('step')}: selected action is not a menu member or is not unique")
        menu_checks.append(_raw_menu_check(cycle, run_dir, source_root, errors, warnings))
        before = _processing_audit(cycle.get("before_state_timing"), side="before",
                                   cycle=cycle, errors=errors, warnings=warnings)
        after = _processing_audit(cycle.get("after_state_timing"), side="after",
                                  cycle=cycle, errors=errors, warnings=warnings)
        clock_checks.append({"step": cycle.get("step"), "before": before, "after": after})
        for check in (before, after):
            if check.get("pipeline_version"):
                pipeline_versions.add(check["pipeline_version"])
        if cycle.get("prediction_pipeline_version"):
            pipeline_versions.add(cycle["prediction_pipeline_version"])
        declared_pipeline = cycle.get("prediction_pipeline_version")
        public_before_prediction = (cycle.get("public_before") or {}).get("prediction")
        if isinstance(public_before_prediction, Mapping):
            declared_pipeline = declared_pipeline or public_before_prediction.get("prediction_pipeline_version")
        timing_pipelines = {item.get("pipeline_version") for item in (before, after)
                            if item.get("pipeline_version")}
        if declared_pipeline and timing_pipelines and declared_pipeline not in timing_pipelines:
            errors.append(f"cycle {cycle.get('step')}: declared prediction pipeline differs from v4 timing")
        if _probabilities((cycle.get("public_before") or {}).get("prediction")) is None:
            errors.append(f"cycle {cycle.get('step')}: public_before has no normalized prediction")
        if _probabilities(cycle.get("prediction_before")) is None or _probabilities(cycle.get("prediction_after")) is None:
            errors.append(f"cycle {cycle.get('step')}: before/after prediction is missing or invalid")
        consumed, basis = _f_consumed(cycle)
        f_rows.append({"step": cycle.get("step"), "f_consumed": consumed, "basis": basis,
                       "prediction_changed": cycle.get("prediction_before") != cycle.get("prediction_after")})
    if len(pipeline_versions) > 1:
        errors.append("prediction_pipeline_version changes within one run")
    if require_v4 and any(check["before"]["schema"] != V4_CONTEXT_SCHEMA or check["after"]["schema"] != V4_CONTEXT_SCHEMA
                          for check in clock_checks):
        errors.append("v4 is required but one or more cycles are legacy or missing processing")
    labels = _labels_audit(run, cycles, errors)
    stream_report = None
    stream_path = Path(stream).resolve() if stream is not None else None
    if stream_path is not None:
        try:
            from football_platform.information.stream_validator import validate_stream
            stream_report = validate_stream(stream_path)
            if not stream_report.get("valid", False):
                errors.append("canonical information stream validator rejected the stream")
        except Exception as exc:  # audit must explain a validator failure, not hide it
            errors.append(f"canonical information stream validator failed: {exc}")
    report = {
        "schema": "footballmind-entity-semantic-f-preflight-audit-v1",
        "status": "PASS" if not errors else "BLOCKED",
        "offline": True,
        "network_requests": 0,
        "training_performed": False,
        "formal_runtime_started": False,
        "source_run": str(run_dir),
        "source_run_sha256": _sha256(run_dir / "run.json"),
        "configuration_sha256": _sha256(config_path) if config_path.is_file() else None,
        "policy_arm": run.get("policy_arm"),
        "selection_authority": "empirical_g_only",
        "menu_freeze": {"cycles": len(cycles), "checks": menu_checks,
                        "complete": not any(item.startswith("cycle ") and "menu" in item for item in errors)},
        "v4_clocks": {"checks": clock_checks, "pipeline_versions": sorted(pipeline_versions),
                       "consistent": len(pipeline_versions) <= 1},
        "f_consumed": f_rows,
        "labels": labels,
        "stream_validation": stream_report,
        "errors": errors,
        "warnings": warnings,
        "limitations": [
            "This audit verifies declared f processing and clocks; it does not estimate g or prove causal information value.",
            "A missing explicit f_consumed flag is reported as null rather than inferred from HTTP or unchanged probabilities.",
            "Outcome and gain remain null until an independent result ledger is joined.",
        ],
    }
    return report


@contextmanager
def _network_denied() -> Iterator[None]:
    original = (socket.socket.connect, socket.socket.connect_ex,
                socket.getaddrinfo, socket.create_connection)
    def deny(*args, **kwargs):
        raise AssertionError("entity-semantic f preflight is offline; network is forbidden")
    socket.socket.connect = socket.socket.connect_ex = socket.getaddrinfo = socket.create_connection = deny
    try:
        yield
    finally:
        socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo, socket.create_connection = original


def write_audit(report: Mapping[str, Any], output: str | Path) -> dict[str, Any]:
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(f"output already exists: {destination}")
    destination.mkdir(parents=True)
    audit_path = destination / "preflight-audit.json"
    audit_path.write_text(json.dumps(dict(report), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    review = [
        "FootballMind entity-semantic f collection preflight",
        f"status: {report['status']}",
        "network_requests: 0 (offline read-only audit)",
        f"cycles: {report['menu_freeze']['cycles']}",
        f"pipeline_versions: {', '.join(report['v4_clocks']['pipeline_versions']) or 'none'}",
        f"errors: {len(report['errors'])}; warnings: {len(report['warnings'])}",
        "outcome/gain: placeholders only; no zero label is created for a failed or late context",
    ]
    (destination / "review-report.txt").write_text("\n".join(review) + "\n", encoding="utf-8")
    files = []
    for path in sorted(destination.iterdir()):
        if path.is_file():
            files.append({"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)})
    manifest = {"schema": "footballmind-entity-semantic-f-preflight-manifest-v1",
                "files": files, "sha256_scope": "all_files_except_manifest"}
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return {"output": str(destination), "status": report["status"],
            "audit_sha256": _sha256(audit_path), "manifest_sha256": _sha256(destination / "manifest.json"),
            "errors": len(report["errors"]), "warnings": len(report["warnings"])}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Existing run directory or run.json")
    parser.add_argument("--output", required=True, help="New audit directory; must not already exist")
    parser.add_argument("--stream", default=None, help="Optional existing information-stream directory")
    parser.add_argument("--require-v4", action="store_true", help="Block legacy/missing v4 processing")
    args = parser.parse_args(argv)
    with _network_denied():
        report = audit_preflight(args.run, stream=args.stream, require_v4=args.require_v4)
    result = write_audit(report, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
