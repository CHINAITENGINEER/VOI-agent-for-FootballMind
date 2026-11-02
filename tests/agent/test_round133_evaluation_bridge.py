import runpy
from pathlib import Path

from football_agent.round133_evaluation_bridge import (
    build_evaluation_episodes_from_run,
    ledger_to_evaluation_episodes,
    load_operation_proposals,
    sanitize_public_snapshot,
)
from football_agent.action_value_ledger import build_action_value_ledger
from football_agent.value_evaluation_contract import PARTIAL_OBSERVATION, validate_evaluation_episode


def test_sanitize_preserves_match_identity_and_removes_prior_return_judgement():
    source = {
        "match_identity": {"label": "甲队 vs 乙队", "status": "unresolved"},
        "question_return_assessments": [
            {"assessment": {"outcome": "partial", "reason": "actor judgement"}},
        ],
        "documents": [{
            "document_id": "doc", "status": "readable",
            "legacy": {"result": {"score": "2-1"}},
        }],
    }
    projected = sanitize_public_snapshot(source)
    assert projected["match_identity"] == {
        "match_label": "甲队 vs 乙队", "status": "unresolved"
    }
    assert "question_return_assessments" not in projected
    assert projected["documents"] == [{"document_id": "doc", "status": "readable", "legacy": {}}]


def test_real_match_experience_run_is_partial_and_uses_proposal_menu(tmp_path):
    helper = runpy.run_path(str(Path(__file__).with_name("test_action_value_ledger.py")))
    service, run, _ = helper["fixture_runtime"](tmp_path)
    try:
        service._drive(run, True)
        folder = service.folder(run)
        ledger = build_action_value_ledger(folder)
        proposals = load_operation_proposals(folder)
        episodes = ledger_to_evaluation_episodes(ledger, proposals=proposals)
        convenience = build_evaluation_episodes_from_run(folder, ledger)
        assert convenience == episodes

        assert len(episodes) == len(ledger["entries"]) == 3
        for entry, episode in zip(ledger["entries"], episodes):
            operation_id = entry["selection_operation_id"]
            proposal = proposals[operation_id]
            expected = [row["candidate_id"] for row in proposal["candidates"]]
            assert episode["pre_action"]["candidate_ids"] == expected
            assert episode["pre_action"]["candidate_generation_complete"] is True
            assert episode["pre_action"]["snapshot"]["match_identity"]["match_label"]
            assert "label" not in episode["pre_action"]["snapshot"]["match_identity"]
            assert "question_return_assessments" not in episode["pre_action"]["snapshot"]
            report = validate_evaluation_episode(episode)
            assert report["status"] == PARTIAL_OBSERVATION
            assert [reason["code"] for reason in report["reasons"]] == [
                "MISSING_EVALUATION_OUTCOME"
            ]
    finally:
        service.close()
