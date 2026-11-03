"""Pinned, offline replay of the existing development demonstration bundle.

This entry does not train a model, discover sources, download weights, or create
a new test split. Planning only reads JSON and hashes input bytes. Outcomes, when
explicitly requested, are read through the separate settlement API after all
predictions have completed.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import platform
import re
import socket
import sys
import time
from unittest.mock import patch

SCHEMA = "footballmind-frozen-replay-experiment-v1"
HEADS = frozenset({"state_ridge", "fulltext_mlp_seed20261027", "fixed_order"})
FIELDS = frozenset({"schema", "experiment_id", "output_dir", "dataset_role",
                    "case_ids", "heads", "max_reads", "expected_manifest_sha256", "settle"})


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate experiment field: " + key)
        result[key] = value
    return result


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=_object)


def _relative(root, value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Bundle input paths must be nonempty forward-slash paths")
    path = (root / value).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError("Bundle input escapes release directory")
    return path


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def experiment_plan(config_path, workspace_config):
    """Validate pinned inputs without importing Torch/sklearn or creating files."""
    filename = Path(config_path).expanduser().resolve()
    config_bytes = filename.read_bytes()
    raw = json.loads(config_bytes.decode("utf-8-sig"), object_pairs_hook=_object)
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA:
        raise ValueError("Unsupported experiment schema")
    unknown = set(raw) - FIELDS
    if unknown:
        raise ValueError("Unknown experiment fields: " + ", ".join(sorted(unknown)))
    identifier = raw.get("experiment_id")
    if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", identifier):
        raise ValueError("experiment_id must be a portable nonempty identifier")
    expected = raw.get("expected_manifest_sha256")
    if not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected):
        raise ValueError("expected_manifest_sha256 must explicitly pin the release manifest")
    heads = raw.get("heads", [workspace_config.default_head])
    if (not isinstance(heads, list) or not heads or any(not isinstance(h, str) or h not in HEADS for h in heads)
            or len(heads) != len(set(heads))):
        raise ValueError("heads must list unique approved frozen strategies")
    maximum = raw.get("max_reads", 3)
    if type(maximum) is not int or not 1 <= maximum <= 20:
        raise ValueError("max_reads must be an integer from 1 through 20")
    if type(raw.get("settle", False)) is not bool:
        raise ValueError("settle must be an explicit boolean")
    role = raw.get("dataset_role", "g_dev")
    if not isinstance(role, str) or not role:
        raise ValueError("dataset_role must be a nonempty role")

    root = Path(workspace_config.release_dir).resolve()
    manifest_file = root / "manifest.json"
    if _sha(manifest_file) != expected:
        raise ValueError("Release manifest SHA differs from experiment pin")
    manifest = _read(manifest_file)
    if manifest.get("bundle_id") != workspace_config.release_id:
        raise ValueError("Unexpected release ID")
    hashes = manifest.get("files_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Release has no file bindings")
    bound = {}
    for name, digest in sorted(hashes.items()):
        source = _relative(root, name)
        if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest) or _sha(source) != digest:
            raise ValueError("Frozen input SHA mismatch: " + name)
        bound[name] = digest
    for name in ("bundle-config.json", "footballmind_defense.py", "data/cases.json", "data/embeddings.npz"):
        if name not in bound:
            raise ValueError("Required frozen input is not bound: " + name)
    binding = _read(root / "bundle-config.json")
    if (binding.get("f_model_sha256") != workspace_config.f_sha256
            or binding.get("g_model_sha256") != workspace_config.g_sha256):
        raise ValueError("Workspace f/g binding differs from frozen release")
    for head in heads:
        if head not in binding.get("models", {}):
            raise ValueError("Strategy is absent from the pinned bundle: " + head)
    f_path = binding.get("f_model_path", "models/f") + "/model.pt"
    g_path = binding.get("g_model_path", "models/g/g-models.joblib")
    if bound.get(f_path) != workspace_config.f_sha256 or bound.get(g_path) != workspace_config.g_sha256:
        raise ValueError("Model bytes are not pinned to the workspace f/g hashes")

    cases = _read(root / "data/cases.json")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Frozen dataset must contain explicit cases")
    by_id = {}
    by_match = set()
    for item in cases:
        if not isinstance(item, dict):
            raise ValueError("Invalid frozen case")
        case_id, match_id = item.get("case_id"), item.get("physical_match_id")
        if (not isinstance(case_id, str) or not case_id or case_id in by_id
                or not isinstance(match_id, str) or not match_id or match_id in by_match):
            raise ValueError("Duplicate or missing case/match identity")
        by_id[case_id] = item
        by_match.add(match_id)
    requested = raw.get("case_ids")
    if requested is not None and (not isinstance(requested, list) or not requested
            or any(not isinstance(c, str) for c in requested) or len(requested) != len(set(requested))):
        raise ValueError("case_ids must be unique explicit IDs, or omitted for the role")
    if requested is not None and set(requested) - set(by_id):
        raise ValueError("Unknown case ID")
    chosen = sorted(requested if requested is not None else [k for k, v in by_id.items() if v.get("split_role") == role])
    if not chosen or any(by_id[c].get("split_role") != role for c in chosen):
        raise ValueError("Selected cases do not all belong to dataset_role")

    experiments = Path(workspace_config.experiments_dir).resolve()
    output_name = raw.get("output_dir", identifier)
    if not isinstance(output_name, str) or not output_name:
        raise ValueError("output_dir must be a nonempty path")
    output = (experiments / Path(output_name).expanduser()).resolve()
    if output == experiments or not output.is_relative_to(experiments):
        raise ValueError("Output must be a fresh child of configured experiments_dir")
    if output == root or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Output overlaps frozen inputs")
    if filename == output or filename.is_relative_to(output):
        raise ValueError("Output overlaps experiment configuration")
    if output.exists():
        raise FileExistsError("Use a new experiment output directory: " + str(output))
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    identity = {"config_sha256": config_hash, "manifest_sha256": expected,
                "case_ids": chosen, "heads": heads, "max_reads": maximum}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"schema": SCHEMA, "experiment_id": identifier, "config_path": str(filename),
            "config_sha256": config_hash, "release_dir": str(root), "manifest_sha256": expected,
            "inputs_sha256": bound, "dataset_role": role, "case_ids": chosen,
            "physical_match_ids": [by_id[c]["physical_match_id"] for c in chosen],
            "matches": len(chosen), "document_occurrences": sum(len(by_id[c].get("documents", [])) for c in chosen),
            "heads": heads, "max_reads": maximum, "output_dir": str(output),
            "settle": raw.get("settle", False), "requested_config": raw, "fingerprint": fingerprint,
            "f_model_sha256": workspace_config.f_sha256, "g_model_sha256": workspace_config.g_sha256,
            "scope": "frozen_development_bundle_replay_not_new_blind_evaluation",
            "network_requests": 0, "training_calls": 0, "weights_loaded": False,
            "outcomes_used_in_prediction": False}


def _load_backend(release_dir):
    # The frozen implementation imports its own bound inference modules.
    # Match the online adapter's Windows native-library initialization order.
    import sklearn.preprocessing  # noqa: F401
    spec = importlib.util.spec_from_file_location("footballmind_experiment_release", release_dir / "footballmind_defense.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FreezeApplication(bundle_dir=release_dir)


@contextmanager
def _offline():
    def blocked(*args, **kwargs):
        raise RuntimeError("Experiment replay prohibits network access")
    with patch.object(socket, "create_connection", blocked), patch.object(socket, "getaddrinfo", blocked), patch.object(socket.socket, "connect", blocked), patch.object(socket.socket, "connect_ex", blocked):
        yield


def _write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _line(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()


def run_experiment(config_path, workspace_config):
    """Replay selected cases with pinned weights; failures never become zero gain."""
    plan = experiment_plan(config_path, workspace_config)
    output = Path(plan["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "plan.json", plan)
    _write(output / "config.json", plan["requested_config"])
    dependencies = {}
    for name in ("numpy", "torch", "scikit-learn", "joblib", "scipy"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    sources = [Path(__file__), Path(__file__).with_name("config.py"),
               Path(__file__).resolve().parents[1] / "demo/defense_budget.py"]
    provenance = {"started_at": _timestamp(), "python": sys.version,
                  "platform": platform.platform(), "dependencies": dependencies,
                  "workspace": workspace_config.as_dict(), "workspace_fingerprint": workspace_config.fingerprint(),
                  "code_sha256": {str(p): _sha(p) for p in sources},
                  "network_requests": 0, "training_calls": 0}
    _write(output / "provenance.json", provenance)
    summary = {"status": "RUNNING", "experiment_id": plan["experiment_id"],
               "output_dir": str(output), "scope": plan["scope"], "planned_predictions": len(plan["heads"]) * plan["matches"],
               "completed_predictions": 0, "failed_predictions": 0, "settled_predictions": 0,
               "network_requests": 0, "training_calls": 0, "outcomes_used_in_prediction": False}
    started = time.perf_counter()
    errors = []
    outcomes = []
    try:
        from demo.defense_budget import run_with_limit
        with _offline():
            backend = _load_backend(Path(plan["release_dir"]))
            completed = []
            with (output / "predictions.jsonl").open("x", encoding="utf-8", newline="\n") as predictions, (output / "failures.jsonl").open("x", encoding="utf-8", newline="\n") as failures:
                for case_id in plan["case_ids"]:
                    for head in plan["heads"]:
                        row_started = time.perf_counter()
                        try:
                            result = run_with_limit(backend, case_id, head, plan["max_reads"])
                            if not isinstance(result, dict) or result.get("status") != "STOPPED":
                                raise ValueError("Frozen replay did not produce a completed prediction")
                            for key in ("p_before", "p_after"):
                                probabilities = result.get(key)
                                if (not isinstance(probabilities, list) or len(probabilities) != 3
                                        or any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities)
                                        or not math.isclose(sum(probabilities), 1, abs_tol=1e-6)):
                                    raise ValueError("Invalid frozen prediction: " + key)
                            _line(predictions, {"case_id": case_id, "head": head, "elapsed_seconds": time.perf_counter() - row_started,
                                                "prediction": result})
                            completed.append((case_id, head, result))
                            summary["completed_predictions"] += 1
                        except Exception as exc:
                            failure = {"case_id": case_id, "head": head, "status": "FAILED",
                                       "error_type": type(exc).__name__, "reason": str(exc),
                                       "gain": None, "p_after": None, "elapsed_seconds": time.perf_counter() - row_started}
                            errors.append(failure)
                            _line(failures, failure)
                            summary["failed_predictions"] += 1
                # Settlement happens only after every strategy prediction. No label
                # is passed to the frozen inference or candidate scoring methods.
                if plan["settle"]:
                    with (output / "settlements.jsonl").open("x", encoding="utf-8", newline="\n") as settlements:
                        for case_id, head, result in completed:
                            settlement = backend.settle_run(result)
                            _line(settlements, {"case_id": case_id, "head": head, "settlement": settlement})
                            if settlement.get("status") == "settled_historical_diagnostic":
                                outcomes.append((head, settlement))
                                summary["settled_predictions"] += 1
        summary["status"] = "COMPLETED" if not errors else "COMPLETED_WITH_FAILURES"
    except Exception as exc:
        summary.update(status="FAILED", error_type=type(exc).__name__, reason=str(exc))
    summary["finished_at"] = _timestamp()
    summary["elapsed_seconds"] = time.perf_counter() - started
    if outcomes:
        summary["historical_diagnostic_metrics"] = {
            head: {"matches": len(rows), "mean_log_loss": sum(r["loss_after"] for r in rows) / len(rows),
                   "mean_realized_gain": sum(r["realized_gain"] for r in rows) / len(rows)}
            for head in plan["heads"] if (rows := [r for h, r in outcomes if h == head])}
    _write(output / "summary.json", summary)
    _write(output / "artifacts.json", {p.name: _sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    return summary
