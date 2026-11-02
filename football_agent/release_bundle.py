"""Freeze the backend and its declared evidence without copying workspace state.

The bundle is a local research handoff, not a new experiment or a deployment.
All inputs retain their relative paths and original bytes. Its manifest is an
integrity inventory, not a signature or a substitute for the scientific audit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import sys


VERSION = "footballmind-backend-release-v1"
MANIFEST = "release-manifest.json"
SKIP = {"__pycache__", ".venv", ".git", "tests", "data", "platform_data", "logs"}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def _file(root, relative):
    if not isinstance(relative, str) or "\\" in relative or ":" in relative:
        raise ValueError("Release paths must use relative forward slashes")
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or any(p in {".", ".."} for p in path.parts):
        raise ValueError("Release path must remain inside its root")
    resolved = (root/relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Release path escapes its root")
    return resolved


def _files_under(root, relative):
    path = _file(root, relative)
    if path.is_file():
        return {path.relative_to(root).as_posix()}
    if not path.is_dir():
        raise ValueError(f"Missing release input: {relative}")
    found = set()
    for folder, dirs, files in os.walk(path, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in SKIP and not d.startswith("."))
        for name in files:
            if name.startswith(".") or name.endswith((".pyc", ".lock")):
                continue
            item = Path(folder)/name
            if item.is_symlink():
                raise ValueError(f"Release cannot silently copy a symbolic link: {item}")
            found.add(item.relative_to(root).as_posix())
    return found


def declared_assets(root, evidence_summary):
    """Select code plus only the paths declared by the deployment and evidence."""
    root = Path(root).resolve()
    selected = set()
    for package in ("football_agent", "football_research", "football_platform", "football_spider"):
        selected.update(p for p in _files_under(root, package) if p.endswith(".py"))
    for relative in ("config/footballmind.json", "football_agent/requirements-frozen.txt",
                     "docs/agent/backend-release.md", "docs/aic", "tools/verify_footballmind_release.py", evidence_summary):
        selected.update(_files_under(root, relative))
    # All asset paths are explicit in the deployment. Descriptive strings and
    # URLs are never treated as paths or evaluated as code.
    config = _read(_file(root, "config/footballmind.json"))

    def assets(value):
        if isinstance(value, dict):
            if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
                path = _file(root, value["path"])
                if sha256(path.read_bytes()).hexdigest() != value["sha256"]:
                    raise ValueError(f"Deployment source changed: {value['path']}")
                yield value["path"]
            for v in value.values():
                yield from assets(v)
        elif isinstance(value, list):
            for v in value:
                yield from assets(v)
        elif isinstance(value, str) and value.startswith("var/parallel/agent/"):
            yield value

    for relative in assets(config):
        selected.update(_files_under(root, relative))
    summary = _read(_file(root, evidence_summary))
    bindings = summary.get("input_bindings", summary.get("source_bindings", []))
    if isinstance(bindings, dict):
        bindings = [{"path": key, **value} for key, value in bindings.items()]
    if not bindings:
        raise ValueError("Evidence must declare its original input bindings")
    for binding in bindings:
        relative = binding.get("path") if isinstance(binding, dict) else None
        if not relative:
            raise ValueError("Every evidence binding must name a relative input path")
        path = _file(root, relative)
        if sha256(path.read_bytes()).hexdigest() != binding["sha256"]:
            raise ValueError(f"Evidence source changed before packaging: {relative}")
        selected.add(relative)
    # Include the rendered tables produced in the same build as the summary.
    for name in ("experiment-table.md", "claim-table.json"):
        path = Path(evidence_summary).with_name(name).as_posix()
        selected.update(_files_under(root, path))
    # Preserve the AIC brief's direct local references in the handoff too.
    # Historical reference documents are not recursively expanded into all rounds.
    for document in (root/"docs/aic").glob("*.md"):
        for target in re.findall(r"\]\(([^)]+)\)", document.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("#"):
                continue
            referenced = (document.parent/target.split("#", 1)[0]).resolve()
            if not referenced.is_relative_to(root) or not referenced.is_file():
                raise ValueError(f"AIC document references a missing local file: {target}")
            selected.add(referenced.relative_to(root).as_posix())
    return sorted(selected)


def freeze_files(root, output, paths, *, evidence_summary=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Release output already exists; choose a new directory")
    paths = sorted(paths)
    if not paths or len(paths) != len(set(paths)) or MANIFEST in paths:
        raise ValueError("Release must contain unique files and no prior manifest")
    records = []
    # Validate the entire selection before starting a new immutable output.
    for relative in paths:
        path = _file(root, relative)
        if not path.is_file() or path.is_relative_to(output):
            raise ValueError(f"Invalid release file: {relative}")
        body = path.read_bytes()
        records.append({"path": relative, "sha256": sha256(body).hexdigest(), "bytes": len(body)})
    output.mkdir(parents=True)
    for record in records:
        source, target = _file(root, record["path"]), _file(output, record["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha256(target.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Source changed during release copy; incomplete bundle is retained")
    manifest = {"schema": VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "headless_research_backend_and_frozen_evidence",
        "validated_platform": {"system": platform.system(), "python": platform.python_version()},
        "real_text_numeric_voi_validated": False, "open_world_policy_benefit_proven": False,
        "frozen_live_match_count": 3, "historical_runs_are_not_fresh_predictions": True,
        "evidence_summary": evidence_summary, "files": records,
        "file_count": len(records), "bytes": sum(r["bytes"] for r in records)}
    _write(output/MANIFEST, manifest)
    return verify_bundle(output)


def verify_bundle(root):
    root = Path(root).resolve()
    manifest = _read(root/MANIFEST)
    if manifest.get("schema") != VERSION:
        raise ValueError("Unknown release manifest")
    records = manifest.get("files")
    if not isinstance(records, list) or not records or manifest.get("file_count") != len(records):
        raise ValueError("Release inventory count differs")
    seen, total = set(), 0
    for record in records:
        relative = record["path"]
        if relative in seen or relative == MANIFEST:
            raise ValueError("Repeated release file")
        seen.add(relative)
        path = _file(root, relative)
        body = path.read_bytes()
        if len(body) != record["bytes"] or sha256(body).hexdigest() != record["sha256"]:
            raise ValueError(f"Release file changed: {relative}")
        total += len(body)
    if total != manifest.get("bytes"):
        raise ValueError("Release inventory size differs")
    return {"schema": "footballmind-release-integrity-v1", "status": "passed",
        "file_count": len(seen), "bytes": total,
        "manifest_sha256": sha256((root/MANIFEST).read_bytes()).hexdigest(),
        "integrity_only": True, "replay_and_experiment_verification_required_separately": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-summary", default="var/parallel/agent/round181/evidence/evidence-summary.json")
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            result = freeze_files(args.root, args.output, declared_assets(args.root, args.evidence_summary),
                                  evidence_summary=args.evidence_summary)
        else:
            if args.output.exists():
                raise ValueError("Verification output already exists")
            result = verify_bundle(args.root)
            _write(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(f"发布包校验失败：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
