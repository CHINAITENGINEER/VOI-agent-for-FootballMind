from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from football_agent.autonomous_mechanism_acceptance import (
    run_autonomous_mechanism_acceptance,
)
from football_agent.scenario_fixture_pack import generate_scenario_fixtures


def test_offline_mechanism_acceptance_passes_all_built_in_scenarios():
    report = run_autonomous_mechanism_acceptance()

    assert report["status"] == "passed"
    assert report["fixture_count"] == len(generate_scenario_fixtures())
    assert report["summary"]["failed"] == 0
    assert report["empirical_g"] == {"status": "deferred", "used_for_selection": False}


def test_acceptance_is_not_a_result_evaluator():
    report = run_autonomous_mechanism_acceptance()

    assert report["mode"] == "offline_mechanism_audit"
    assert all("result_leakage" not in name for name in report["failed_checks"])
    assert all(result["passed"] for name, result in report["checks"].items()
               if name.endswith("actor_has_no_evaluator_or_result_fields"))
    assert report["interpretation"].startswith("通过表示信息选择契约")
    assert "准确" in report["interpretation"]


def test_acceptance_fails_closed_when_actor_fixture_is_tampered():
    fixtures = generate_scenario_fixtures()
    fixtures[0]["public_snapshot"]["known_information"]["result"] = "1-0"

    report = run_autonomous_mechanism_acceptance(fixtures)

    assert report["status"] == "failed"
    assert any("fixture_valid" in item for item in report["failed_checks"])


def test_cli_emits_json_and_writes_report(tmp_path: Path):
    output = tmp_path / "mechanism.json"
    command = [sys.executable, "tools/check_footballmind_autonomous_mechanism.py",
               "--output", str(output)]
    completed = subprocess.run(command, capture_output=True, text=True,
                               encoding="utf-8", check=False)

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["status"] == "passed"
    assert json.loads(output.read_text(encoding="utf-8"))["summary"]["failed"] == 0
