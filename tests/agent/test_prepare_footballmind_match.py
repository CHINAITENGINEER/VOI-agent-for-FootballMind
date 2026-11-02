import json
import subprocess
import sys


def test_match_bootstrap_cli_creates_headless_bundle(tmp_path):
    output = tmp_path / "run"
    result = subprocess.run([
        sys.executable, "tools/prepare_footballmind_match.py",
        "--match", "Santos vs Flamengo",
        "--competition", "Brasileirao",
        "--kickoff", "2026-10-10T20:00:00+08:00",
        "--output", str(output),
    ], capture_output=True, text=True, check=True)
    summary = json.loads(result.stdout)
    assert summary["status"] == "prepared"
    assert summary["unknown_count"] == 7
    assert json.loads((output / "initial-state.json").read_text(encoding="utf-8"))["contract"]["no_user_scope_question"]

