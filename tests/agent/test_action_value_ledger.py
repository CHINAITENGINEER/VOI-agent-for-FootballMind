from copy import deepcopy
from pathlib import Path
import json
import runpy

import pytest

from football_agent.action_value_ledger import build_action_value_ledger
from football_agent.acquisition_replay import verify_run
from football_agent.audit import atomic_write


helper = runpy.run_path(str(Path(__file__).with_name("test_match_return_assessment.py")))
fixture_runtime = helper["fixture_runtime"]


def test_ledger_joins_selection_cost_return_and_followup(tmp_path):
    service, run, _ = fixture_runtime(tmp_path)
    try:
        service._drive(run, True)
        ledger = build_action_value_ledger(service.folder(run))
        assert ledger["schema_version"] == "action-value-ledger-v1"
        assert [entry["action"] for entry in ledger["entries"]] == ["search_web", "read_url", "stop"]
        first, second, terminal = ledger["entries"]
        assert first["selection_resource_cost"] == {"llm_calls": 1}
        assert first["resource_cost"] == {"http_calls": 1, "llm_calls": 0, "local_reads": 0}
        assert first["acquisition_status"] == "returned"
        assert first["return_observation"]["semantic_relevance"] == "unassessed"
        assert first["return_assessment"]["outcome"] == "discovery_only"
        assert first["realized_decision_gain"] is None
        assert first["numeric_voi_training_eligible"] is False
        assert first["next_selection_operation_id"] == second["selection_operation_id"]
        assert second["return_assessment"]["outcome"] == "partial"
        assert terminal["acquisition_status"] == "not_requested"
        assert terminal["resource_cost"] == {"http_calls": 0, "llm_calls": 0, "local_reads": 0}
        assert ledger["summary"]["resource_cost"] == {"http_calls": 2, "llm_calls": 0, "local_reads": 0}
        assert ledger["numeric_voi_calibrated"] is False
        assert verify_run(service.folder(run))["action_value_ledger_entries"] == 3
    finally:
        service.close()


def test_ledger_is_deterministic_and_exported(tmp_path):
    service, run, _ = fixture_runtime(tmp_path)
    try:
        service._drive(run, True)
        folder = service.folder(run)
        one = build_action_value_ledger(folder)
        two = build_action_value_ledger(folder)
        assert one == two
        assert service.export(run)["action_value_ledger"] == one
    finally:
        service.close()


def test_ledger_preserves_independent_local_read_cost(tmp_path):
    local = runpy.run_path(str(Path(__file__).with_name("test_local_evidence.py")))
    calls = []
    rows = [local["fixture_row"](matchId=1000 + i, home="构造甲队" + str(i)) for i in range(20)]
    rows[-1]["home"] = "窗口外目标甲队"

    class Proposer:
        def propose(self, public):
            if not public["documents"]:
                response = helper["decision"](public, "read_url")
                response["url"] = public["source_catalog"]["entries"][0]["url"]
            elif len(public["documents"]) == 1:
                response = local["local_decision"](public, team_name="窗口外目标甲队")
                response[helper["review"].FIELD] = helper["actual_review"](public, "insufficient_context")
            else:
                response = helper["decision"](public)
            return {"candidates": [response], "recommended_index": 0, "test_double": True}

    service = helper["ExperienceService"](
        tmp_path,
        proposer_factory=lambda f, e: Proposer(),
        tools_factory=lambda f: local["h"]["make_tools"](f, calls, {"recent": local["recent_payload"](rows)}),
        mode="test_fixture", enable_return_assessment=True)
    try:
        run = service.create({"match": "窗口外目标甲队 vs 构造乙队", "limits": {"http_calls": 1}})["run_id"]
        service._drive(run, True)
        ledger = build_action_value_ledger(service.folder(run))
        query = next(entry for entry in ledger["entries"] if entry["action"] == "query_acquired")
        assert query["resource_cost"]["http_calls"] == 0
        assert query["resource_cost"]["local_reads"] == 1
        assert ledger["summary"]["resource_cost"]["http_calls"] == 1
        assert ledger["summary"]["resource_cost"]["local_reads"] == 1
    finally:
        service.close()


def test_ledger_rejects_changed_action_link(tmp_path):
    service, run, _ = fixture_runtime(tmp_path)
    try:
        service._drive(run, True)
        folder = service.folder(run)
        link_path = next((folder / "acquisitions").glob("*/selection-link.json"))
        link = json.loads(link_path.read_text(encoding="utf-8"))
        link["action_digest"] = "0" * 64
        atomic_write(link_path, link)
        with pytest.raises(ValueError, match="REPLAY_ACQUISITION_LINK_MISMATCH|ACTION_LEDGER_ACTION_LINK_MISMATCH"):
            build_action_value_ledger(folder)
    finally:
        service.close()
