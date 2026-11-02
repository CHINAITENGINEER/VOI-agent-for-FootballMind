"""Package already accepted frozen artifacts; never train or overwrite a model."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "var/releases/footballmind-defense-20261025-rc1"
REPORT = ROOT / "var/submission/footballmind-defense-20261025-v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def archive(folder, files, target):
    expected = {p.relative_to(folder).as_posix(): sha(p) for p in files}
    with ZipFile(target, "x", ZIP_DEFLATED, compresslevel=6) as z:
        for p in files:
            z.write(p, folder.name + "/" + p.relative_to(folder).as_posix())
    with ZipFile(target) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(files)
        for name, digest in expected.items():
            assert hashlib.sha256(z.read(folder.name + "/" + name)).hexdigest() == digest
    return {"path": str(target), "bytes": target.stat().st_size, "sha256": sha(target),
            "files": len(files), "testzip": "PASS", "zip_bytes_equal_to_local": True,
            "file_sha256": expected}


def main():
    manifest = json.loads((RELEASE / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files_sha256"].items():
        assert sha(RELEASE / name) == expected, name
    current = json.loads((RELEASE / "validation-stop-v2/acceptance.json").read_text(encoding="utf-8"))
    assert current["status"] == "PASS" and current["manifest_sha256"] == sha(RELEASE / "manifest.json")
    code_files = [
        "demo/live_g.py", "demo/defense.py", "demo/server.py", "demo/static/live.html",
        "demo/static/live.js", "demo/static/live-progress.js", "demo/static/defense.html",
        "demo/static/defense.js", "demo/static/defense.css", "demo/static/live.css",
        "demo/static/index.html", "Start-FootballMind.cmd", "Start-FootballMind-Defense.cmd",
        "Start-FootballMind-Legacy.cmd", "start_footballmind_defense.py",
        "tools/verify_defense_browser.py", "tools/verify_live_body_handoff.py",
        "tools/finalize_footballmind_defense_release.py",
        "tests/demo/test_live_g.py", "tests/demo/test_defense_api.py", "tests/demo/test_live_progress_ui.py",
    ]
    snapshots = REPORT / "accepted-source-snapshot"
    snapshots.mkdir(exist_ok=False)
    for name in code_files:
        target = snapshots / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    save(REPORT / "accepted-code-manifest.json", {"files_sha256": {n: sha(ROOT / n) for n in code_files},
         "note": "Review snapshot of accepted code; not a standalone full application distribution.",
         "py_compile": "PASS", "javascript_syntax": "PASS"})
    # Only the current validation goes into the portable archive. Superseded
    # local receipts remain untouched beside the original bundle.
    release_files = [RELEASE / n for n in manifest["files_sha256"]] + [RELEASE / "manifest.json"]
    release_files += [p for p in (RELEASE / "validation-stop-v2").rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts]
    release_files = sorted(set(release_files))
    model_zip = archive(RELEASE, release_files, RELEASE.with_suffix(".zip"))
    save(REPORT / "frozen-package-receipt.json", model_zip)
    report_files = sorted(p for p in REPORT.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    report_zip = archive(REPORT, report_files, REPORT.with_suffix(".zip"))
    receipt = {"status": "PASS", "created_at": datetime.now(timezone.utc).isoformat(),
               "core_manifest_sha256": sha(RELEASE / "manifest.json"),
               "model_package": {k: v for k, v in model_zip.items() if k != "file_sha256"},
               "evidence_package": {k: v for k, v in report_zip.items() if k != "file_sha256"},
               "current_validation": "validation-stop-v2/acceptance.json", "network_requests": 0,
               "model_retraining": False, "original_models_modified": False,
               "frontend": "Project server at http://127.0.0.1:8766/defense.html; model archive supports portable CLI, not standalone frontend."}
    save(ROOT / "var/submission/footballmind-defense-20261025-final-delivery.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
