from copy import deepcopy

import pytest

from football_agent.audit import atomic_write
from football_agent.temporal_snapshot import (
    ELIGIBLE, EXCLUDED_AFTER_CUTOFF, FAILED, QUARANTINED_SCOPE,
    public_menu, register, validate_choice, verify,
)


def task(tmp_path):
    bodies = {}
    candidates = []
    specs = [
        ("eligible", "pre_match_structured", "2026-09-22T09:00:00Z", "2026-09-22T09:01:00Z", 200, None),
        ("late", "pre_match_preview", "2026-09-26T09:00:00Z", "2026-09-26T09:01:00Z", 200, None),
        ("quarantine", "unknown", "2026-09-22T09:00:00Z", "2026-09-22T09:01:00Z", 200, None),
        ("failed", "post_match_report", "2026-09-22T09:00:00Z", "2026-09-22T09:01:00Z", None, "ConnectionError"),
    ]
    for cid, scope, observed, available, status, error in specs:
        path = tmp_path / f"{cid}.body"
        path.write_text(f"private body for {cid}", encoding="utf-8")
        bodies[cid] = path
        candidates.append({"candidate_id": cid, "url": f"https://example.com/{cid}",
                           "publisher": "Source", "information_role": f"role-{cid}",
                           "menu_note": "frozen candidate", "document_scope": scope,
                           "observed_at": observed, "available_at": available,
                           "http_status": status, "error_class": error,
                           "snapshot_path": str(path)})
    return {"task_id": "temporal-task", "prediction_cutoff": "2026-09-23T09:30:00Z",
            "kickoff": "2026-09-23T10:30:00Z",
            "target": {"question_id": "q-context", "statement": "The source can answer the match context.",
                       "scope": "match", "scope_key": "match-1"},
            "allowed_hosts": ["example.com"], "candidates": candidates}


def valid_response(public, selected="eligible", kind="query"):
    rows = []
    for item in public["menu"]:
        chosen = item["candidate_id"] == selected and kind == "query"
        rows.append({"candidate_id": item["candidate_id"], "information_role": item["information_role"],
                     "availability_reason": item["availability_reason"],
                     "why_relevant": "It could affect the fixed question.",
                     "not_selected_reason": None if chosen else "It is not the selected available route."})
    return {"target": public["target"], "menu_review": rows,
            "candidates": [{"candidate_id": selected, "rationale": "It is available and relevant.",
                            "expected_evidence": "A source answer.", "possible_answer_change": "It may alter the report."}] if kind == "query" else [],
            "decision": {"kind": kind, "candidate_id": selected if kind == "query" else None,
                         "reason": "Temporal finite-menu choice."}}


def test_registration_classifies_time_scope_and_failure_without_exposing_bodies(tmp_path):
    root = tmp_path / "run"
    register(root, task(tmp_path))
    public = public_menu(root)
    statuses = {row["candidate_id"]: row["availability_status"] for row in public["menu"]}
    assert statuses == {"eligible": ELIGIBLE, "late": EXCLUDED_AFTER_CUTOFF,
                        "quarantine": QUARANTINED_SCOPE, "failed": FAILED}
    assert all(row["body_hidden"] and row["snapshot_text_not_exposed"] and "text" not in row for row in public["menu"])
    assert all(not row["selectable"] for row in public["menu"] if row["candidate_id"] != "eligible")


@pytest.mark.parametrize("selected", ["late", "quarantine", "failed"])
def test_choice_rejects_a_temporally_ineligible_candidate(tmp_path, selected):
    root = tmp_path / "run"
    register(root, task(tmp_path))
    public = public_menu(root)
    response = valid_response(public, selected=selected)
    with pytest.raises(ValueError, match="eligible"):
        validate_choice(response, public)


def test_defer_requires_reasons_for_all_hidden_or_excluded_paths(tmp_path):
    root = tmp_path / "run"
    register(root, task(tmp_path))
    public = public_menu(root)
    response = valid_response(public, kind="defer")
    assert validate_choice(response, public)["decision"]["kind"] == "defer"


def test_selected_release_is_bound_to_eligible_choice_and_verify_preserves_hashes(tmp_path):
    root = tmp_path / "run"
    register(root, task(tmp_path))
    public = public_menu(root)
    atomic_write(root / "agent-input.json", public)
    response = valid_response(public)
    atomic_write(root / "agent-choice.json", {"response": response, "receipt": {},
                                               "body_text_exposed_to_agent": False,
                                               "candidate_ids_seen": [x["candidate_id"] for x in public["menu"]],
                                               "temporally_ineligible_candidates_selectable": False})
    from football_agent.temporal_snapshot import release_selected
    release = release_selected(root)
    assert release["released_candidate_id"] == "eligible"
    assert release["temporally_ineligible_body_released"] is False
    assert verify(root)["status"] == "OK"


def test_time_order_is_rejected(tmp_path):
    value = task(tmp_path)
    value["candidates"][0]["available_at"] = "2026-09-21T08:59:00Z"
    with pytest.raises(ValueError, match="precedes"):
        register(tmp_path / "run", value)

