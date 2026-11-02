import json
from pathlib import Path

from football_agent.autonomous_loop_report import VERSION, build_report, render_markdown


ROOT = Path(__file__).resolve().parents[2]


def artifacts():
    loop = json.loads((ROOT / "var/parallel/agent/round81/real-autonomous-loop-replay.json").read_text(encoding="utf-8"))
    policy = json.loads((ROOT / "var/parallel/agent/round78/autonomous-acquisition-replay.json").read_text(encoding="utf-8"))
    return loop, policy


def test_report_selects_only_public_loop_fields():
    loop, policy = artifacts()
    report = build_report(loop, policy)
    assert report["schema_version"] == VERSION
    assert report["steps"][0]["selected_candidate"] == "qualification-refresh"
    encoded = json.dumps(report, ensure_ascii=False).lower()
    assert '"true_outcome"' not in encoded
    assert '"sampled_result"' not in encoded
    assert report["hidden_result_included"] is False


def test_report_preserves_replanning_trace_and_evidence_references():
    loop, policy = artifacts()
    report = build_report(loop, policy)
    assert report["transition_events"][-1] == "STOPPED"
    assert report["evidence_refs"]
    assert len(report["steps"]) == 4
    assert report["steps"][0]["context_delta_after_step"]["qualification"] == "already_qualified"


def test_markdown_contains_selection_reason_and_cost_table():
    loop, policy = artifacts()
    markdown = render_markdown(build_report(loop, policy))
    assert "qualification-refresh" in markdown
    assert "reserve-strength" in markdown
    assert "固定特征与 VoI 对照" in markdown
    assert "获取成本" in markdown
    assert "true_outcome" not in markdown


def test_report_rejects_wrong_schema():
    loop, policy = artifacts()
    report = build_report(loop, policy)
    report["schema_version"] = "wrong"
    try:
        render_markdown(report)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong schema should be rejected")


def test_report_is_json_serializable():
    loop, policy = artifacts()
    report = build_report(loop, policy)
    json.dumps(report, ensure_ascii=False)
