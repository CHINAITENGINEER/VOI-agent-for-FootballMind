"""Protocol boundary tests, not claims of semantic grounding accuracy."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from football_agent.inquiry_provider import CodexInquiryProvider
from football_agent.semantic_scenario_protocol import (
    ProposalProtocol, DirectionalProposalProtocol, ObservationProtocol,
    ActionExtensionProtocol, DirectionalActionExtensionProtocol,
    validate_proposal, validate_directional_proposal, validate_observation,
    validate_action_extension, validate_directional_action_extension,
    public_projection, no_answer, observation_public_projection, OBSERVATION_INPUT_VERSION,
    CandidateDiscoveryProtocol, validate_discovery,
)


def proposal():
    return {"schema": ProposalProtocol.version,
        "hypotheses": [
            {"id": "limited", "description": "主力不可用", "prior": .4,
             "outcome_probabilities": [.4, .3, .3]},
            {"id": "available", "description": "主力可用", "prior": .6,
             "outcome_probabilities": [.6, .2, .2]}],
        "actions": [{"action_id": "read:article", "question": "主力本场可用吗？",
            "observation_group": "availability", "likelihoods": [
                {"hypothesis_id": "limited", "probabilities": [.1, .7, .2]},
                {"hypothesis_id": "available", "probabilities": [.7, .1, .2]}]}],
        "assumptions": ["条件概率来自语义模型假设，未经统计校准。"]}


def test_each_protocol_schema_node_declares_type_for_actual_provider():
    def inspect(node):
        assert "type" in node
        if node["type"] == "object":
            for child in node["properties"].values():
                inspect(child)
        elif node["type"] == "array":
            inspect(node["items"])
    for protocol in (ProposalProtocol, DirectionalProposalProtocol, ObservationProtocol,
                     ActionExtensionProtocol, DirectionalActionExtensionProtocol,
                     CandidateDiscoveryProtocol):
        inspect(protocol.schema)


CANDIDATES = [{"action_id": "read:article", "tool": "read_url", "title": "Fixture update", "url": "https://example.org/update"}]
DOCUMENTS = [{"document_id": "page:1", "title": "Ready for the club fixture",
              "text": "The striker is available for Saturday's fixture against Leeds.",
              "kind": "page", "status": "OK"}]


def observation():
    return {"schema": ObservationProtocol.version, "action_id": "read:article",
        "outcome": "supports", "evidence": [{"document_id": "page:1",
            "quote": "available for Saturday's fixture against Leeds"}],
        "applicability": "match_specific", "reason": "正文明确涉及本场。"}


def test_evidence_only_projection_keeps_exact_body_and_question_without_numeric_expectations():
    source = proposal()
    action = source["actions"][0]
    action["direction_contract"] = {"supports": ["home_positive"], "refutes": ["home_negative"]}
    match = {"match_id": "m", "home_name": "Home", "away_name": "Away",
             "kickoff": "2026-10-10T00:00:00Z", "future_outcome": "secret",
             "baseline": [.9, .05, .05]}
    public = observation_public_projection(match=match, action=action, documents=DOCUMENTS)
    assert public["input_contract"] == OBSERVATION_INPUT_VERSION
    assert set(public) == {"input_contract", "match", "action", "documents"}
    assert set(public["action"]) == {"action_id", "question", "direction_contract"}
    assert "baseline" not in public["match"] and "future_outcome" not in public["match"]
    assert public["documents"][0]["text"] == DOCUMENTS[0]["text"]
    assert validate_observation(observation(), public["action"], public["documents"])["outcome"] == "supports"
    action["question"] = "changed outside"
    assert public["action"]["question"] != action["question"]


def test_proposal_covers_observed_candidates_and_returns_detached_copy():
    value = proposal()
    clean = validate_proposal(value, CANDIDATES)
    assert clean == value
    clean["hypotheses"][0]["prior"] = .9
    assert value["hypotheses"][0]["prior"] == .4


def directional_proposal():
    value = deepcopy(proposal())
    value["schema"] = DirectionalProposalProtocol.version
    value["actions"][0]["direction_contract"] = {
        "supports": ["home_negative"], "refutes": ["home_positive"]}
    return value


def test_directional_proposal_freezes_support_refute_polarity():
    clean = validate_directional_proposal(directional_proposal(), CANDIDATES)
    assert clean == directional_proposal()
    bad = directional_proposal()
    bad["actions"][0]["direction_contract"]["refutes"] = ["home_negative"]
    with pytest.raises(ValueError, match="disjoint"):
        validate_directional_proposal(bad, CANDIDATES)


@pytest.mark.parametrize("bucket,values,pattern", [
    ("supports", ["home_positive", "home_negative"], "at most one team direction"),
    ("refutes", ["away_positive", "away_negative"], "at most one team direction"),
    ("supports", ["mixed", "home_negative"], "cannot mix mixed and team direction"),
])
def test_directional_contract_rejects_ambiguous_polarity_buckets(bucket, values, pattern):
    response = directional_proposal()
    response["actions"][0]["direction_contract"][bucket] = values
    # Keep the test focused on the bucket-width rule rather than the earlier
    # disjointness rule when the mutated support list contains the default
    # refuted direction.
    if values == ["home_positive", "home_negative"]:
        response["actions"][0]["direction_contract"]["refutes"] = ["away_positive"]
    with pytest.raises(ValueError, match=pattern):
        validate_directional_proposal(response, CANDIDATES)


@pytest.mark.parametrize("where,key,value", [
    ("root", "selected_action", "read:article"),
    ("action", "gain", 1), ("action", "url", "https://not-observed.org"),
    ("root", "cost", 0),
])
def test_proposal_cannot_make_decisions_or_expand_capabilities(where, key, value):
    response = proposal()
    target = response if where == "root" else response["actions"][0]
    target[key] = value
    with pytest.raises(ValueError, match="schema rejected"):
        validate_proposal(response, CANDIDATES)


def test_unknown_action_omitted_candidate_duplicate_hypothesis_rejected():
    response = proposal()
    response["actions"][0]["action_id"] = "read:hidden"
    with pytest.raises(ValueError, match="public candidate"):
        validate_proposal(response, CANDIDATES)
    with pytest.raises(ValueError, match="cover every"):
        validate_proposal(proposal(), CANDIDATES + [{"action_id": "search:new"}])
    response = proposal()
    response["hypotheses"][1]["id"] = "limited"
    with pytest.raises(ValueError, match="Duplicate hypothesis"):
        validate_proposal(response, CANDIDATES)


@pytest.mark.parametrize("value", [0, 1, float("nan"), float("inf"), True])
def test_nonpositive_nonfinite_or_boolean_probability_is_not_repaired(value):
    response = proposal()
    response["hypotheses"][0]["prior"] = value
    with pytest.raises(ValueError):
        validate_proposal(response, CANDIDATES)


def test_likelihoods_cannot_turn_missing_answer_into_evidence():
    response = proposal()
    response["actions"][0]["likelihoods"][1]["probabilities"] = [.6, .1, .3]
    with pytest.raises(ValueError, match="hypothesis-independent"):
        validate_proposal(response, CANDIDATES)
    response = proposal()
    response["hypotheses"][0]["outcome_probabilities"] = [.4, .3, .4]
    with pytest.raises(ValueError, match="sum to one"):
        validate_proposal(response, CANDIDATES)


def test_extension_cannot_rewrite_frozen_hypotheses_or_repeat_old_actions():
    extension = {"schema": ActionExtensionProtocol.version,
                 "actions": proposal()["actions"], "assumptions": ["当前历史条件下的似然。"]}
    clean = validate_action_extension(extension, CANDIDATES, ["limited", "available"])
    assert clean == extension
    with pytest.raises(ValueError, match="unique public"):
        validate_action_extension(extension, CANDIDATES, ["limited", "available"], ["read:article"])
    extension["hypotheses"] = proposal()["hypotheses"]
    with pytest.raises(ValueError, match="schema rejected"):
        validate_action_extension(extension, CANDIDATES, ["limited", "available"])


def test_directional_extension_requires_the_same_contract():
    extension = {"schema": DirectionalActionExtensionProtocol.version,
                 "actions": deepcopy(proposal()["actions"]),
                 "assumptions": ["当前历史条件下的方向合同。"]}
    with pytest.raises(ValueError, match="schema rejected"):
        validate_directional_action_extension(extension, CANDIDATES, ["limited", "available"])
    extension["actions"][0]["direction_contract"] = {
        "supports": ["home_negative"], "refutes": ["home_positive"]}
    assert validate_directional_action_extension(
        extension, CANDIDATES, ["limited", "available"]) == extension


def test_only_acquired_body_exact_quote_can_support_match_specific_observation():
    response = observation()
    assert validate_observation(response, CANDIDATES[0], DOCUMENTS) == response
    for quote in [DOCUMENTS[0]["title"], "available for this fixture against Leeds", "available for Saturday ... Leeds"]:
        bad = deepcopy(response)
        bad["evidence"][0]["quote"] = quote
        with pytest.raises(ValueError, match="exact body substring"):
            validate_observation(bad, CANDIDATES[0], DOCUMENTS)


@pytest.mark.parametrize("field,value", [("kind", "search_snippet"), ("status", "FAILED")])
def test_failed_or_discovery_return_is_not_body_evidence(field, value):
    docs = deepcopy(DOCUMENTS)
    docs[0][field] = value
    with pytest.raises(ValueError, match="failed return or discovery snippet"):
        validate_observation(observation(), CANDIDATES[0], docs)


@pytest.mark.parametrize("applicability", ["team_context", "unresolved"])
def test_general_team_context_cannot_update_fixture_beliefs(applicability):
    response = observation()
    response["applicability"] = applicability
    clean = validate_observation(response, CANDIDATES[0], DOCUMENTS)
    assert clean["outcome"] == "no_answer"
    assert response["outcome"] == "supports"
    assert "保持原情景权重" in clean["reason"]


def test_unreturned_document_wrong_action_and_unquoted_answer_rejected():
    with pytest.raises(ValueError, match="unreturned document"):
        validate_observation(observation(), CANDIDATES[0], [])
    with pytest.raises(ValueError, match="differs from executed"):
        validate_observation(observation(), {"action_id": "read:other"}, DOCUMENTS)
    response = observation()
    response["evidence"] = []
    with pytest.raises(ValueError, match="needs acquired body"):
        validate_observation(response, CANDIDATES[0], DOCUMENTS)
    failed = no_answer(CANDIDATES[0], "本次真实查询失败，无法回答。")
    assert validate_observation(failed, CANDIDATES[0], []) == failed


def test_public_projection_excludes_future_returns_and_training_labels():
    public = public_projection(match={"home_name": "A", "away_name": "B", "result": [3, 0]},
        baseline={"probabilities": [.5, .3, .2], "hidden_labels": [1]},
        documents=[{**DOCUMENTS[0], "hidden_prior": "SECRET"}],
        candidates=[{**CANDIDATES[0], "body": "SECRET", "gain": 1}],
        budget={"remaining_requests": 3, "secret_value": "SECRET"})
    assert "SECRET" not in json.dumps(public)
    assert "result" not in public["match"]
    assert "gain" not in public["candidates"][0]
    assert "hidden_labels" not in public["baseline"]
    assert public["documents"][0]["text"] == DOCUMENTS[0]["text"]


def test_discovery_proposes_queries_without_deciding_or_manufacturing_sources():
    response = {"schema": CandidateDiscoveryProtocol.version, "queries": [
        {"query": "Arsenal Leeds October 10 match availability", "information_need": "确认本场不确定的阵容。"}]}
    assert validate_discovery(response) == response
    assert validate_discovery({"schema": CandidateDiscoveryProtocol.version, "queries": []})["queries"] == []
    response["queries"][0]["gain"] = .1
    with pytest.raises(ValueError, match="schema rejected"):
        validate_discovery(response)


@pytest.mark.parametrize("query", ["", "   ", "x" * 301, "query\nnext"])
def test_discovery_rejects_unbounded_or_blank_query(query):
    with pytest.raises(ValueError):
        validate_discovery({"schema": CandidateDiscoveryProtocol.version, "queries": [
            {"query": query, "information_need": "需要信息。"}]})


def test_discovery_rejects_duplicates_and_extra_conclusion():
    response = {"schema": CandidateDiscoveryProtocol.version, "queries": [
        {"query": "Arsenal Leeds", "information_need": "确认本场未知。"},
        {"query": " arsenal   leeds ", "information_need": "仍是相同查询。"}]}
    with pytest.raises(ValueError, match="distinct"):
        validate_discovery(response)
    response["queries"] = response["queries"][:1]
    response["decision"] = "query"
    with pytest.raises(ValueError, match="schema rejected"):
        validate_discovery(response)


@pytest.mark.parametrize("explicit_model", [None, "test-model-version"])
def test_provider_records_requested_model_without_claiming_resolved_weights(tmp_path, explicit_model):
    commands = []
    def runner(command, **kwargs):
        commands.append(command)
        output = Path(command[command.index("--output-last-message") + 1])
        output.write_text(json.dumps({"ok": True}), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout=json.dumps({"type": "turn.completed", "usage": {}}), stderr="")
    provider = CodexInquiryProvider(tmp_path, executable="codex", runner=runner,
                                   protocol=ProposalProtocol(), model=explicit_model)
    result = provider.decide({"transport_only": True})
    assert result["receipt"]["model_requested"] == (explicit_model or "cli_default_no_override")
    assert result["receipt"]["response_model"] is None
    assert ("--model" in commands[0]) == (explicit_model is not None)
    if explicit_model is not None:
        assert commands[0][commands[0].index("--model") + 1] == explicit_model
        assert provider.configuration()["model_identity_status"] == "requested_name_not_verified_weights"


@pytest.mark.parametrize("model", ["", "a b", "--enable", "model\n--other", 42])
def test_provider_model_identifier_validation(tmp_path, model):
    with pytest.raises(ValueError, match="model identifier"):
        CodexInquiryProvider(tmp_path, model=model)
