"""Finite snapshot choice: menu-only Agent input and isolated all-menu evaluation."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from football_agent.audit import atomic_write, digest
from football_agent.prospective_snapshot import (
    CHOICE_PROMPT, VERSION, evaluate_all, prefetch, public_menu, read, register,
    release_selected, validate_choice, verify,
)
from football_agent.inquiry_impact import terminal_state
from football_agent.inquiry_tools import OpenWebTools


def task():
    target = {"hypothesis_id": "h-lineup", "question_id": "q-lineup", "statement": "Official source states Player X started for Team A in match M.",
        "hypothesis_type": "starting_lineup", "subject": "team-a", "scope": "match", "scope_key": "match-m",
        "required_evidence": ["official_lineup"]}
    problem = {"task_id": "snapshot-task", "decision_id": "lineup-decision", "target": target,
        "semantics": "source_position_not_world_truth", "state_definitions": {"positive": "Source states the player started.", "negative": "Source states the player did not start."},
        "loss": {"report_supported": {"positive": 0, "negative": 1}, "report_challenged": {"positive": 1, "negative": 0}, "withhold": {"positive": .2, "negative": .2}},
        "loss_basis": "Declared source-reporting preference."}
    return {"task_id": "snapshot-task", "decision_problem": problem, "allowed_hosts": ["example.com"], "candidates": [
        {"candidate_id": "direct", "url": "https://example.com/direct", "publisher": "Official club", "information_role": "match report", "menu_note": "A first-party match-specific report."},
        {"candidate_id": "background", "url": "https://example.com/background", "publisher": "Official club", "information_role": "team background", "menu_note": "A first-party background page."},
        {"candidate_id": "rules", "url": "https://example.com/rules", "publisher": "Competition body", "information_role": "competition context", "menu_note": "A rules and competition context page."},
    ]}


def factory(root, calls):
    bodies = {"/direct": "Official source states Player X started for Team A in match M.",
              "/background": "Team A has a long history and a large squad.",
              "/rules": "The competition has group-stage rules."}
    def make(archive, manifest):
        class Session:
            def get(self, url, **kwargs):
                assert kwargs["allow_redirects"] is False
                calls.append(url)
                body = ("<html><article>" + bodies[url.split("example.com", 1)[1]] + "</article></html>").encode()
                return SimpleNamespace(status_code=200, content=body, headers={}, close=lambda: None)
            def close(self): pass
        return OpenWebTools(archive, manifest["task"]["allowed_hosts"], session_factory=Session,
                            clock=lambda: __import__("datetime").datetime.now(__import__("datetime").timezone.utc))
    return make


def setup(tmp_path):
    root = tmp_path/"snapshot"; register(root, task()); calls=[]
    prefetch(root, tools_factory=factory(root, calls))
    return root, calls


def response(root, selected="direct", kind="query"):
    public = public_menu(root); target = public["target"]; ids=[m["candidate_id"] for m in public["menu"]]
    rows=[{"candidate_id": i, "information_role": next(m["information_role"] for m in public["menu"] if m["candidate_id"]==i),
           "why_relevant": "It may contribute to the fixed question.", "not_selected_reason": None if i==selected and kind=="query" else "Another route is more direct."} for i in ids]
    candidates=[{"candidate_id": selected, "rationale": "This is the most direct available route.", "expected_evidence": "A source statement.", "possible_answer_change": "It could change the report."}] if kind=="query" else []
    return {"target":target,"menu_review":rows,"candidates":candidates,
            "decision":{"kind":kind,"candidate_id":selected if kind=="query" else None,"reason":"Finite menu choice."}}


def test_prefetch_is_before_agent_and_public_menu_hides_bodies(tmp_path):
    root,calls=setup(tmp_path)
    public=public_menu(root)
    assert len(calls)==3 and all(m["body_hidden"] and m["snapshot_text_not_exposed"] for m in public["menu"])
    assert all("text" not in m for m in public["menu"])
    assert public["body_visibility"] == "candidate_body_hidden_until_logical_selection"
    assert read(root/"manifest.json")["prefetch"]["counted_as_agent_actions"] is False


def test_prefetch_cannot_repeat_or_change_frozen_menu(tmp_path):
    root,_=setup(tmp_path)
    with pytest.raises(ValueError, match="only once"):
        prefetch(root, tools_factory=factory(root, []))
    manifest=read(root/"manifest.json"); manifest["task"]["candidates"][0]["url"]="https://example.com/other"
    atomic_write(root/"manifest.json",manifest)
    with pytest.raises(ValueError): public_menu(root)


@pytest.mark.parametrize("mutation", ["target", "outside", "missing_row", "selected_reason"])
def test_choice_contract_rejects_target_or_menu_tampering(tmp_path, mutation):
    root,_=setup(tmp_path); public=public_menu(root); value=response(root)
    if mutation=="target": value["target"]["statement"]="Other target."
    elif mutation=="outside": value["candidates"][0]["candidate_id"]="outside"
    elif mutation=="missing_row": value["menu_review"].pop()
    else: value["menu_review"][0]["not_selected_reason"]="Should not be here." 
    with pytest.raises(ValueError): validate_choice(value,public)


def test_release_exposes_only_selected_snapshot_and_not_all(tmp_path):
    root,_=setup(tmp_path); choice=response(root,"direct")
    atomic_write(root/"agent-choice.json", {"response":choice,"receipt":{"real_provider_call":True},"candidate_ids_seen":["direct","background","rules"],"body_text_exposed_to_agent":False})
    released=release_selected(root)
    assert released["released_candidate_id"]=="direct" and released["logical_read_count"]==1
    assert "Player X started" in released["document"]["text"]
    assert read(root/"selected-release.json")["prefetch_http_not_agent_http"] is True


def test_stop_releases_no_body(tmp_path):
    root,_=setup(tmp_path); choice=response(root,kind="defer")
    atomic_write(root/"agent-choice.json", {"response":choice,"receipt":{},"candidate_ids_seen":["direct","background","rules"],"body_text_exposed_to_agent":False})
    released=release_selected(root)
    assert released["released_candidate_id"] is None and released["document"] is None


def test_all_menu_evaluation_separates_selected_and_unselected_projection(tmp_path, monkeypatch):
    root,_=setup(tmp_path); choice=response(root,"background")
    atomic_write(root/"agent-choice.json", {"response":choice,"receipt":{},"candidate_ids_seen":["direct","background","rules"],"body_text_exposed_to_agent":False})
    release_selected(root)
    def fake_decide(self, public):
        rows=[]
        for item in public["paths"]:
            if item["path_id"]=="direct":
                rows.append({"path_id":"direct","verdict":"direct_support","rationale":"Exact source line.","evidence":[{"relation":"supports","explanation":"Exact target statement.","anchors":[{"role":"statement","quote":"Official source states Player X started for Team A in match M."}]}]})
            else:
                rows.append({"path_id":item["path_id"],"verdict":"no_direct_evidence","rationale":"No target statement.","evidence":[]})
        return {"response":{"target_hash":public["target_hash"],"rows":rows},"receipt":{"real_provider_call":True,"tool_events":[]}}
    monkeypatch.setattr("football_agent.prospective_snapshot.CodexInquiryProvider.decide", fake_decide)
    result=evaluate_all(root)
    assert result["selected_candidate_id"]=="background"
    assert result["selected_projection"]["action"]=="withhold"
    assert result["per_candidate_source_reports"]["direct"]["action"]=="report_supported"
    assert result["unselected_action_changes"]=={"direct":True,"rules":False}
    assert result["selection_regret"] is None and result["optimal_menu_choice_proven"] is False
    assert result["prefetch_http_attempts"]==3 and result["agent_logical_reads"]==1


def test_verify_rejects_body_leak_or_selected_release_mismatch(tmp_path):
    root,_=setup(tmp_path); choice=response(root)
    atomic_write(root/"agent-input.json", public_menu(root))
    atomic_write(root/"agent-choice.json", {"response":choice,"receipt":{},"candidate_ids_seen":["direct","background","rules"],"body_text_exposed_to_agent":False})
    release_selected(root)
    atomic_write(root/"result.json", {"selected_candidate_id":"background","prefetch_http_attempts":3})
    with pytest.raises(ValueError): verify(root)
