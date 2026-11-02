from __future__ import annotations

import json

import pytest

from football_platform.information.stream_validator import validate_stream
from football_research.data_adapters.prospective_stream import (
    assign_policy_arm,
    export_empty_package,
)


def test_assign_policy_arm_is_predeclared_and_requires_random_seed():
    assigned = assign_policy_arm(
        physical_match_id="m-1",
        kickoff_at="2026-10-12T19:00:00+00:00",
        arm="fixed_order",
        assigned_at="2026-10-12T12:00:00+00:00",
        policy_version="policy-v1",
    )
    assert assigned["policy_arm"] == "fixed_order"
    assert assigned["random_seed"] is None
    with pytest.raises(ValueError):
        assign_policy_arm(
            physical_match_id="m-1",
            kickoff_at="2026-10-12T19:00:00+00:00",
            arm="random_explore",
            assigned_at="2026-10-12T12:00:00+00:00",
            policy_version="policy-v1",
        )


def test_empty_live_inventory_is_a_valid_canonical_stream(tmp_path):
    output = tmp_path / "stream"
    report = export_empty_package(
        output,
        inventory_report={"path": "inventory/run-report.json", "run_id": "run-1"},
        shortage_reason="no_future_premier_league_rows",
    )
    assert report["available_prospective_fixtures"] == 0
    assert report["g_training_eligible_rows"] == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == manifest["schema_version"] == "footballmind-prospective-stream-v2"
    checked = validate_stream(output)
    assert checked["valid"] is True
    assert checked["errors"] == []
