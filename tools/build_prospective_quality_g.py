"""Build a read-only audit package for the prospective frozen-f quality g.

The command deliberately has no network client and never enables a policy.  It
reconstructs pre-query features from frozen streams, re-runs the raw-bound
label admission gate, and records why rows remain pending or excluded.  When
settled outcomes are later appended to the source stream, the same command can
be rerun with a new output directory.

Example::

    python tools/build_prospective_quality_g.py \
      --entries-json var/submission/fm-g-quality-g-entries.json \
      --output var/submission/fm-g-quality-g-assembly-20261011-v1

The entries JSON is an object with an ``entries`` array.  Each entry contains
``stream`` and ``registration`` paths.  Paths are resolved relative to the
current working directory, then recorded as absolute paths in the package.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

# Direct ``python tools/...`` puts only the tools directory on sys.path.  Keep
# the runner usable from the repository root without requiring installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from football_research.acquisition.prospective_quality_g import assemble, fit_and_evaluate


SCRIPT_VERSION = "prospective-quality-g-audit-builder-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def _manifest(root: Path):
    return {
        "schema": "prospective-quality-g-audit-manifest-v1",
        "builder": SCRIPT_VERSION,
        "files": [
            {"path": str(path.relative_to(root)).replace("\\", "/"), "sha256": _sha256(path),
             "bytes": path.stat().st_size}
            for path in sorted(root.rglob("*")) if path.is_file() and path.name != "manifest.json"
        ],
    }


def build(entries_json: Path, output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty audit directory: {output}")
    config = _read(entries_json)
    entries = config.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("entries JSON must contain a non-empty entries array")
    normalized = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"stream", "registration"}:
            raise ValueError("Each entry must contain exactly stream and registration")
        normalized.append({"stream": str(Path(entry["stream"]).resolve()),
                           "registration": str(Path(entry["registration"]).resolve())})
    output.mkdir(parents=True, exist_ok=False)
    dataset = assemble(normalized)
    models, evaluation = fit_and_evaluate(dataset)
    dataset["source_bindings"] = sorted(dataset["source_bindings"],
                                         key=lambda row: (row["physical_match_id"], row["pipeline"]))
    _write_json(output / "dataset.json", dataset)
    _write_json(output / "fit-report.json", evaluation)
    saved_models = []
    for pipeline, model in sorted(models.items()):
        short = hashlib.sha256(pipeline.encode("utf-8")).hexdigest()[:16]
        model_path = output / f"model-{short}.json"
        model.save(model_path)
        saved_models.append({"pipeline": pipeline, "path": model_path.name,
                             "sha256": _sha256(model_path), "model_version": model.model_version})
    summary = dataset["summary"]
    validation = {
        "schema": "prospective-quality-g-audit-v1",
        "builder": SCRIPT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_config_sha256": _sha256(entries_json),
        "source_config": str(entries_json.resolve()),
        "dataset_schema": dataset["schema"],
        "selected_attempts": summary["selected_attempts"],
        "physical_matches": summary["physical_matches"],
        "prequery_feature_rows": summary["prequery_feature_rows"],
        "valid_predictor_pairs": summary["valid_predictor_pairs"],
        "eligible_rows": summary["eligible_rows"],
        "models_fitted": evaluation["models_fitted"],
        "saved_models": saved_models,
        "strata": evaluation["strata"],
        "network_requests": 0,
        "selection_allowed": False,
        "production_ready": False,
        "empirical_status": "waiting_for_independent_settled_outcomes",
        "synthetic_prior_used": False,
        "training_performed": bool(models),
        "limitations": [
            "No independent settled outcome is present in the supplied streams.",
            "Rows with missing or invalid post-action f provenance are excluded.",
            "Prediction pipelines and pre-registered physical-match partitions are never pooled.",
            "This package is a research audit and cannot authorize online g selection.",
        ],
    }
    _write_json(output / "validation.json", validation)
    (output / "README.md").write_text(
        "# Prospective frozen-f quality-g assembly\n\n"
        "This package is a read-only reconstruction of pre-query features and the raw-bound quality-target gate. "
        "It contains no network calls and cannot enable g. `eligible_rows=0` means no independent settled outcome "
        "was admitted at build time; it is not a zero-gain label. Rebuild into a new directory after outcome "
        "settlement and independent raw/result verification.\n",
        encoding="utf-8",
    )
    manifest = _manifest(output)
    _write_json(output / "manifest.json", manifest)
    zip_path = output.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output).as_posix())
    return {"output": str(output.resolve()), "zip": str(zip_path.resolve()),
            "zip_sha256": _sha256(zip_path), "validation": validation,
            "manifest_sha256": _sha256(output / "manifest.json")}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entries-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build(args.entries_json, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

