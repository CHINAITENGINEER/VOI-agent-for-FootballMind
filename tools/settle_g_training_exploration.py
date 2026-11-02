"""Materialize post-match g labels from an immutable prospective run.

Usage is deliberately explicit: callers provide each canonical JSONL stream
file and a new output directory.  The command performs no HTTP and never
modifies the source run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from football_research.data_adapters.training_exploration_settlement import (
    settle_training_exploration_rows,
    write_settlement_package,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: expected JSON object")
            rows.append(value)
    return rows


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--discoveries", required=True, type=Path)
    parser.add_argument("--attempts", required=True, type=Path)
    parser.add_argument("--contexts", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--documents", type=Path)
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    paths = {
        name: path for name, path in {
            "fixtures": args.fixtures, "discoveries": args.discoveries,
            "attempts": args.attempts, "contexts": args.contexts,
            "results": args.results, "documents": args.documents,
        }.items() if path is not None
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        parser.error("missing input file(s): " + ", ".join(missing))
    loaded = {name: _read_jsonl(path) for name, path in paths.items()}
    rows, report = settle_training_exploration_rows(
        fixtures=loaded["fixtures"], discoveries=loaded["discoveries"],
        attempts=loaded["attempts"], contexts=loaded["contexts"],
        results=loaded["results"], documents=loaded.get("documents"),
        archive_root=args.archive_root,
    )
    source_manifest = {
        "source_files": {name: {"path": str(path), "sha256": _sha(path), "bytes": path.stat().st_size}
                         for name, path in paths.items()},
        "archive_root": str(args.archive_root) if args.archive_root else None,
    }
    out = write_settlement_package(args.output, rows=rows, report=report, source_manifest=source_manifest)
    files = []
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            files.append({"path": path.name, "bytes": path.stat().st_size, "sha256": _sha(path)})
    (out / "manifest.json").write_text(json.dumps(files, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_out = {**report, "output": str(out), "manifest_sha256": _sha(out / "manifest.json")}
    print(json.dumps(report_out, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
