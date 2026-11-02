"""Witnessed rule inputs, unchanged propositions and autonomous next-query hooks."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from football_agent.audit import digest
from football_agent.inquiry import InquiryWorkflow
from football_agent.inquiry_continuation import ContinuationPolicy
from football_agent.inquiry_probability import ReturnProbabilityProtocol
from football_agent.inquiry_returns import OUTCOMES
from football_agent.inquiry_rule_scope import RuleScopeProtocol
from football_agent.inquiry_tools import OpenWebTools
from football_agent.inquiry_verify import verify
from football_spider.storage.raw_archive import RawArchive


NOW = datetime(2026, 9, 27, 3, tzinfo=timezone.utc)
EDITION = "The League One 2026 competition rules govern this season."
MEMBERS = "League One 2026 participating clubs: Team A and Team B."
RULE = "The top six teams after the regular rounds enter Group A."
FIXTURE = "Fixture M: Team A plays Team B in League One 2026 on 2026-09-27."
TABLE = "At 2026-09-26T12:00:00Z, Team A has 30 points with two matches remaining."
BODY = " ".join((EDITION, MEMBERS, RULE, FIXTURE))
H = {"hypothesis_id": "h1", "question_id": "q1", "hypothesis_type": "qualification",
     "statement": "Team A has qualified before fixture M.", "subject": "Team-A",
     "scope": "match", "scope_key": "fixture-M", "required_evidence": ["competition_rules", "standings"]}
TASK = {"task_id": "constructed-rule-scope", "objective": "Choose information for qualification reasoning.",
        "deadline": (NOW+timedelta(hours=2)).isoformat(), "initial_facts": [],
        "seed_urls": ["https://example.com/rules", "https://example.com/standings"]}


def claims(document_id="doc-one"):
    rule = {"claim_id": "rule", "question_id": "q1", "text": RULE, "status": "source_claim",
        "citations": [{"document_id": document_id, "quote": q} for q in (EDITION, RULE)],
        "evidence": {"kind": "competition_rules", "subject": "LeagueOne", "scope": "campaign", "scope_key": "LeagueOne:2026"},
        "hypothesis_links": [{"hypothesis_id": "h1", "relation": "context"}]}
    fixture = {"claim_id": "fixture", "question_id": "q1", "text": FIXTURE, "status": "source_claim",
        "citations": [{"document_id": document_id, "quote": FIXTURE}],
        "evidence": {"kind": "other", "subject": "Team-A", "scope": "match", "scope_key": "fixture-M"},
        "hypothesis_links": [{"hypothesis_id": "h1", "relation": "context"}]}
    return [rule, fixture]


def binding(document_id="doc-one"):
    def anchor(quote, **mentions):
        return {"document_id": document_id, "quote": quote, **{k+"_mention": v for k, v in mentions.items()}}
    return {"binding_id": "rule-to-match", "claim_id": "rule", "fixture_claim_id": "fixture", "hypothesis_id": "h1",
        "competition_id": "LeagueOne", "campaign_key": "LeagueOne:2026", "season": "2026", "team_id": "Team-A", "fixture_id": "fixture-M",
        "rule_edition": anchor(EDITION, competition="League One", season="2026"),
        "team_in_edition": anchor(MEMBERS, competition="League One", season="2026", team="Team A"),
        "fixture_in_edition": anchor(FIXTURE, competition="League One", season="2026"),
        "team_in_fixture": anchor(FIXTURE, team="Team A"),
        "explanation": "Source-reported edition, participation and fixture context; semantic identity is unverified."}


def response(kind="query", *, target="rules", updates=None, links=None):
    evidence = "competition_rules" if target == "rules" else "standings"
    candidate = {"candidate_id": "c-"+target, "question_id": "q1", "tool": "read_url",
        "url": "https://example.com/"+target, "query": None, "rationale": "Inspect the next source-reported input.",
        "answer_could_change": "Change which input is missing, without certifying qualification.",
        "inquiry_intent": {"hypothesis_ids": ["h1"], "purpose": "context", "expected_evidence": [evidence],
            "subject": "LeagueOne" if target == "rules" else "Team-A", "scope": "campaign", "scope_key": "LeagueOne:2026",
            "supports_change": "Use an applicable source input.", "challenges_change": "Retain opposing statements.",
            "no_answer_change": "Retain uncertainty."},
        "return_model": {"branches": [{"outcome": outcome,
            "missing_after": [{"hypothesis_id": "h1", "types": list(H["required_evidence"])}],
            "next_action": {"kind": "reassess", "target_hypothesis_ids": [], "evidence_types": [],
                            "reason": "Use the actual acquired evidence."}} for outcome in OUTCOMES]},
        "outcome_forecast": {"status": "abstain", "probabilities": None, "basis_document_ids": [],
                             "reason": "No numerical forecast in this constructed scope test."}}
    candidates = [candidate] if kind == "query" else []
    return {"questions": [{"question_id": "q1", "question": H["statement"], "decision_relevance": "Determine useful next information.",
        "answer_could_change": "Choose a remaining gap.", "based_on_documents": []}], "hypotheses": [deepcopy(H)],
        "claim_updates": updates or [], "context_bindings": [], "rule_scope_bindings": links or [], "candidates": candidates,
        "choice_review": {"alternatives": [{"candidate_id": c["candidate_id"], "information_advantage": "Addresses a remaining input.",
            "source_limitations": "Source assertions and applicability are unverified.", "evidence_document_ids": [],
            "not_selected_reason": None} for c in candidates], "tradeoff": "Choose after inspecting current evidence.",
            "missing_routes": "Not an exhaustive action set."},
        "decision": {"kind": kind, "candidate_id": candidate["candidate_id"] if candidates else None,
                     "reason": "Constructed proposal to exercise the acquisition mechanism.", "revisit_at": None}}


def public():
    return {**RuleScopeProtocol().public({}), "task": TASK, "now": NOW.isoformat(), "questions": [], "claims": [],
        "budget": {"remaining_requests": 3, "remaining_llm_calls": 4}, "tools": {"allowed_hosts": ["example.com"]},
        "documents": [{"document_id": "doc-one", "kind": "page", "status": "OK", "text": BODY, "url": TASK["seed_urls"][0], "links": []}]}


def setup(root, responses, *, protocol=None, clock=lambda: NOW):
    calls, inputs = [], []
    class Session:
        def get(self, url, **kwargs):
            calls.append(url)
            text = BODY if url.endswith("rules") else TABLE
            return SimpleNamespace(status_code=200, content=text.encode(), headers={}, close=lambda: None)
        def close(self): pass
    class Tools(OpenWebTools):
        def configuration(self):
            return {**super().configuration(), "execution_mode": "constructed_transport_no_network"}
    class Provider:
        def configuration(self): return {"provider": "constructed-scope-test"}
        def decide(self, state):
            inputs.append(deepcopy(state))
            item = responses[len(inputs)-1]
            return {"response": item(state) if callable(item) else deepcopy(item), "receipt": {"real_provider_call": False}}
    tools = Tools(RawArchive(root/"evidence"), ["example.com"], session_factory=Session, clock=clock)
    workflow = InquiryWorkflow(root/"runs", Provider(), tools, clock=clock, protocol=protocol or RuleScopeProtocol(),
                               continuation=ContinuationPolicy(), source_catalog=[])
    return workflow, calls, inputs


def test_cross_subject_rule_supplies_only_an_input_and_keeps_original_evidence(tmp_path):
    def extract(state):
        doc = state["documents"][-1]["document_id"]
        return response("defer", updates=claims(doc), links=[binding(doc)])
    workflow, calls, _ = setup(tmp_path, [response(), extract])
    run = workflow.run(workflow.start(TASK, max_requests=1, max_llm_calls=2)["run_id"])
    requirement = run["requirements"][0]
    assert requirement["present_types"] == ["competition_rules"] and requirement["missing_types"] == ["standings"]
    assert requirement["proposition_direction"] == "undetermined" and not requirement["prerequisites_complete"]
    assert run["hypothesis_states"][0]["status"] == "context_only"
    assert run["claims"][0]["evidence"]["subject"] == "LeagueOne" and run["context_bindings"] == []
    path = requirement["slots"][0]["evidence_paths"][0]
    assert path["via"] == "witnessed_rule_scope" and path["identity_verification"] == "unverified"
    assert run["return_reviews"][0]["outcome"] == "context_only"
    assert run["return_probability_reviews"][0]["evaluation"]["status"] == "abstained_not_scored"
    assert len(calls) == 1 and verify(tmp_path, run["run_id"])["decisions_verified"] == 2


@pytest.mark.parametrize("mistake", ["wrong_competition", "wrong_campaign", "wrong_team", "wrong_fixture", "season", "different_literal_competition",
    "different_literal_team", "unacquired", "quote", "mention", "wrong_doc_role", "rule_anchor", "support", "rotation", "coaching", "fixture_scope", "same_id"])
def test_scope_mismatches_and_unwitnessed_promotions_are_rejected(mistake):
    state = public()
    reply = response("reflect", updates=claims(), links=[binding()])
    link = reply["rule_scope_bindings"][0]
    if mistake.startswith("wrong_") and mistake != "wrong_doc_role":
        field = {"wrong_competition": "competition_id", "wrong_campaign": "campaign_key", "wrong_team": "team_id", "wrong_fixture": "fixture_id"}[mistake]
        link[field] = "different"
    elif mistake == "season": link["season"] = "2025"
    elif mistake == "different_literal_competition":
        text = FIXTURE.replace("League One", "League Two")
        state["documents"][0]["text"] += " " + text
        link["fixture_in_edition"].update(quote=text, competition_mention="League Two")
    elif mistake == "different_literal_team": link["team_in_fixture"]["team_mention"] = "Team B"
    elif mistake == "unacquired": link["team_in_edition"]["document_id"] = "unknown-doc"
    elif mistake == "quote": link["team_in_edition"]["quote"] = "Invented League One 2026 membership of Team A."
    elif mistake == "mention": link["team_in_fixture"]["team_mention"] = "Team C"
    elif mistake == "wrong_doc_role":
        state["documents"].append({**state["documents"][0], "document_id": "other-doc"})
        link["team_in_edition"]["document_id"] = "other-doc"
    elif mistake == "rule_anchor": reply["claim_updates"][0]["citations"] = [reply["claim_updates"][0]["citations"][1]]
    elif mistake == "support": reply["claim_updates"][0]["hypothesis_links"][0]["relation"] = "supports"
    elif mistake == "rotation":
        reply["hypotheses"][0].update(hypothesis_type="rotation_intent", required_evidence=["coach_statement"])
    elif mistake == "coaching": reply["claim_updates"][0]["evidence"]["kind"] = "coach_statement"
    elif mistake == "fixture_scope": reply["claim_updates"][1]["evidence"]["scope"] = "campaign"
    else:
        reply["context_bindings"] = [{"binding_id": link["binding_id"], "hypothesis_id": "h1", "claim_id": "rule",
            "relation": "competition_rule_applies_to_match", "explanation": "Duplicate ID.",
            "witnesses": [{"document_id": "doc-one", "quote": MEMBERS}], "information_as_of": None, "time_witnesses": []}]
    with pytest.raises(ValueError): RuleScopeProtocol().validate(reply, state)


@pytest.mark.parametrize("longer,mention", [("20260", "2026"), ("2026-27", "2026"), ("League 12", "League 1")])
def test_numeric_prefixes_are_not_full_edition_or_competition_mentions(longer, mention):
    from football_agent.inquiry_rule_scope import _contains_mention
    assert not _contains_mention("Source states "+longer+" here.", mention, season=mention == "2026")


def test_legacy_cross_subject_rejection_is_unchanged():
    new = response("reflect", updates=claims(), links=[binding()])
    RuleScopeProtocol().validate(new, public())
    old = deepcopy(new)
    old.pop("rule_scope_bindings")
    old["context_bindings"] = [{"binding_id": "legacy-b", "hypothesis_id": "h1", "claim_id": "rule",
        "relation": "competition_rule_applies_to_match", "explanation": "The sources state the relation.",
        "witnesses": [{"document_id": "doc-one", "quote": MEMBERS}], "information_as_of": None, "time_witnesses": []}]
    with pytest.raises(ValueError, match="Unsupported applicability"):
        ReturnProbabilityProtocol().validate(old, public())
    assert "rule_scope_bindings" not in ReturnProbabilityProtocol.schema["properties"]


def test_new_binding_alone_can_reflect_but_existing_binding_is_immutable():
    state = public()
    state.update(hypotheses=[deepcopy(H)], claims=claims(), questions=response()["questions"])
    update = response("reflect", links=[binding()])
    update["hypotheses"] = []
    assert RuleScopeProtocol().validate(update, state) == update
    state["rule_scope_bindings"] = [binding()]
    with pytest.raises(ValueError, match="Reflection requires new"):
        RuleScopeProtocol().validate(update, state)
    update["rule_scope_bindings"][0]["explanation"] = "Rewritten justification."
    with pytest.raises(ValueError, match="immutable"):
        RuleScopeProtocol().validate(update, state)


def test_inherited_rule_changes_next_query_without_refresh_or_parent_labels(tmp_path):
    parent, parent_calls, _ = setup(tmp_path/"parent", [response(), response("defer")])
    prior = parent.run(parent.start(TASK, max_requests=1, max_llm_calls=2)["run_id"])
    def extract_existing(state):
        doc = state["documents"][0]["document_id"]
        return response("reflect", updates=claims(doc), links=[binding(doc)])
    def choose_gap(state):
        assert state["requirements"][0]["missing_types"] == ["standings"]
        assert state["context_bindings"] == [] and state["rule_scope_bindings"]
        assert state["return_probability_reviews"] == []
        return response(target="standings")
    child, child_calls, inputs = setup(tmp_path/"child", [extract_existing, choose_gap, response("defer")], clock=lambda: NOW+timedelta(minutes=5))
    run = child.start(TASK, max_requests=1, max_llm_calls=3)
    run = child.inherit_evidence(run["run_id"], parent_root=tmp_path/"parent", parent_run_id=prior["run_id"],
        parent_head_hash=prior["events"][-1]["event_hash"], document_ids=[prior["documents"][0]["document_id"]])
    final = child.run(run["run_id"])
    assert len(parent_calls) == len(child_calls) == 1 and child_calls[0].endswith("/standings")
    assert final["documents"][0]["observed_at"] == prior["documents"][0]["observed_at"]
    contract = final["receipts"][0]["inquiry_contract"]
    assert contract["rule_scope_before"] == final["rule_scope_bindings"]
    assert contract["requirements_before"][0]["missing_types"] == ["standings"]
    assert len(final["return_reviews"]) == 1 and final["requirements"][0]["missing_types"] == ["standings"]
    assert inputs[0]["claims"] == [] and inputs[0]["rule_scope_bindings"] == []
    assert verify(tmp_path/"child", final["run_id"])["inherited_documents_verified"] == 1


def test_rehashed_scope_projection_cannot_be_changed_between_decisions(tmp_path):
    def extract(state):
        doc = state["documents"][-1]["document_id"]
        return response("defer", updates=claims(doc), links=[binding(doc)])
    workflow, _, _ = setup(tmp_path, [response(), extract])
    run = workflow.run(workflow.start(TASK, max_requests=1, max_llm_calls=2)["run_id"])
    event = run["events"][-1]
    event["snapshot"]["rule_scope_bindings"][0]["season"] = "2025"
    event["event_hash"] = digest({k: v for k, v in event.items() if k != "event_hash"})
    run.update(deepcopy(event["snapshot"]))
    workflow.store.save(run)
    workflow.store.verify(run["run_id"])
    with pytest.raises(ValueError, match="projection prefix"):
        verify(tmp_path, run["run_id"])


def test_matching_literals_do_not_claim_semantic_or_identity_verification():
    state = public()
    reply = response("defer", updates=claims(), links=[binding()])
    RuleScopeProtocol().validate(reply, state)
    semantics = state["rule_scope_semantics"]
    assert semantics["entity_resolution"] == "unverified_source_reported_aliases"
    assert semantics["rule_section_scope"] == semantics["relation_entailment"] == "unverified"
    assert semantics["information_value"] is None


@pytest.mark.parametrize("role", ["fixture_in_edition", "team_in_fixture"])
def test_another_fixture_on_the_same_page_cannot_supply_the_cited_match_context(role):
    state = public()
    other = "Fixture N: Team A plays Team C in League One 2026 on 2026-10-02."
    state["documents"][0]["text"] += " " + other
    reply = response("defer", updates=claims(), links=[binding()])
    reply["rule_scope_bindings"][0][role]["quote"] = other
    with pytest.raises(ValueError, match="another match"):
        RuleScopeProtocol().validate(reply, state)
