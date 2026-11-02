"""Inventory existing Round60+ artifacts without rerunning or rewriting them."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
ROUND85_HASH_PATHS = {
    "metrics_source": "football_research/evaluation/historical_policy_metrics.py",
    "metrics_tests": "tests/research/evaluation/test_historical_policy_metrics.py",
    "replay_script": "var/parallel/agent/round85/run_historical_policy_metrics_replay.py",
    "pairing_script": "var/parallel/agent/round85/run_strategy_pairing_audit.py",
    "replay": "var/parallel/agent/round85/historical-policy-metrics-replay.json",
    "pairing_audit": "var/parallel/agent/round85/strategy-pairing-audit.json",
    "readme": "var/parallel/agent/round85/README.md",
}


def read_log(path):
    raw = path.read_bytes()
    encoding = "utf-16" if raw[:2] in (b'\xff\xfe', b'\xfe\xff') else "utf-8-sig"
    return raw.decode(encoding, errors="replace")


def inventory():
    status_path = ROOT / "docs/parallel/status/agent.md"
    status_text = status_path.read_text(encoding="utf-8-sig")
    declared = {}
    for value, number in re.findall(r"status=(ROUND(\d+)_[A-Z0-9_]+)", status_text):
        declared.setdefault(int(number), value)
    rounds = []
    for folder in sorted((ROOT / "var/parallel/agent").glob("round*"), key=lambda p: int(p.name[5:]) if p.name[5:].isdigit() else -1):
        if not folder.is_dir() or not folder.name[5:].isdigit() or int(folder.name[5:]) < 60:
            continue
        number = int(folder.name[5:])
        files = sorted(p.name for p in folder.iterdir() if p.is_file())
        tests = []
        for log in sorted(folder.glob("*.txt")):
            summaries = re.findall(r"[0-9]+ passed[^\r\n]*", read_log(log))
            if summaries:
                tests.append({"file": log.relative_to(ROOT).as_posix(), "summary": summaries[-1],
                              "evidence": "stored test log; not rerun by this inventory"})
        acceptance = folder / "acceptance.json"
        acceptance_summary = None
        if acceptance.is_file():
            try:
                data = json.loads(acceptance.read_text(encoding="utf-8-sig"))
                checks = data.get("checks", {})
                acceptance_summary = {
                    "status": data.get("status"), "check_count": len(checks),
                    "recorded_checks_all_true": bool(checks) and all(v is True for v in checks.values()),
                    "artifact_hash_count": len(data.get("artifact_hashes", {})),
                }
            except (OSError, ValueError, TypeError) as exc:
                acceptance_summary = {"error": str(exc)}
        rounds.append({"round": number, "directory": folder.relative_to(ROOT).as_posix(),
                       "files": files, "declared_status": declared.get(number),
                       "accept_script_present": "accept_round.py" in files,
                       "acceptance": acceptance_summary, "stored_test_logs": tests})
    verified_hashes = {}
    hash_rounds = [(85, ROUND85_HASH_PATHS)]
    hash_rounds.extend((item['round'], {}) for item in rounds if item['round'] >= 86)
    for number, aliases in hash_rounds:
        path = ROOT / f"var/parallel/agent/round{number}/acceptance.json"
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        results = {}
        for key, digest in data.get("artifact_hashes", {}).items():
            relative = aliases.get(key, key)
            target = (ROOT / relative).resolve()
            if not target.is_relative_to(ROOT.resolve()):
                results[key] = False
                continue
            results[key] = target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == digest
        verified_hashes[str(number)] = {"count": len(results), "all_match": bool(results) and all(results.values()), "checks": results}
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        'workspace': str(ROOT), 'scope': 'disk artifacts, stored logs and available Round85+ acceptance hashes; no chat-server diagnosis',
        "latest_directory": max((r["round"] for r in rounds), default=None),
        "latest_declared_round": max(declared, default=None),
        "rounds": rounds, "verified_artifact_hashes": verified_hashes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inventory()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + chr(10), encoding="utf-8")
    print(json.dumps({"latest_directory": report["latest_directory"], "latest_declared_round": report["latest_declared_round"],
                      "round_count": len(report["rounds"]), "verified_artifact_hashes": report["verified_artifact_hashes"]}, ensure_ascii=False, indent=2))
    if any(not value["all_match"] for value in report["verified_artifact_hashes"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
