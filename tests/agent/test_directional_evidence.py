"""Directional evidence is opt-in and must not turn availability into impact."""
from copy import deepcopy

import pytest

from football_agent.directional_evidence import (
    DIRECTIONAL_VERSION,
    decorate_legacy_observation,
    directional_update,
    directional_update_for_action,
    validate_directional_observation,
)


ACTION = {"action_id": "read:fixture-news"}
DOCS = [{
    "document_id": "doc:fixture-news",
    "kind": "article",
    "status": "OK",
    "text": (
        "The manager confirmed that the home side will miss its main striker "
        "for Saturday's fixture, a major attacking loss."
    ),
}]


def record(**changes):
    value = {
        "schema": DIRECTIONAL_VERSION,
        "action_id": ACTION["action_id"],
        "outcome": "supports",
        "direction": "home_negative",
        "subject": "home",
        "direction_basis": "explicit_effect",
        "directional_claim": "正文明确说明主队主力前锋缺阵且构成进攻损失。",
        "evidence": [{
            "document_id": DOCS[0]["document_id"],
            "quote": "miss its main striker for Saturday's fixture, a major attacking loss",
        }],
        "applicability": "match_specific",
        "reason": "正文明确涉及本场并陈述对主队进攻的影响。",
    }
    value.update(changes)
    return value


def test_explicit_direction_requires_exact_body_and_exposes_update():
    clean = validate_directional_observation(record(), ACTION, DOCS)
    update = directional_update(clean)
    assert update["direction"] == "home_negative"
    assert update["subject"] == "home"
    assert update["evidence_document_ids"] == ["doc:fixture-news"]
    assert update["semantics"].startswith("source_semantic_direction")


def test_action_bound_update_rejects_direction_that_contract_does_not_allow():
    action = {
        "action_id": ACTION["action_id"],
        "direction_contract": {
            "supports": ["home_negative"],
            "refutes": ["home_positive"],
        },
    }
    clean = validate_directional_observation(record(), ACTION, DOCS)
    assert directional_update_for_action(clean, action)["direction"] == "home_negative"
    contradictory = deepcopy(clean)
    contradictory["direction"] = "home_positive"
    contradictory["directional_claim"] = "正文明确说明主队进攻增强。"
    assert directional_update_for_action(contradictory, action) is None
    malformed = {"action_id": ACTION["action_id"],
                 "direction_contract": {"supports": "home_negative", "refutes": []}}
    assert directional_update_for_action(clean, malformed) is None


def test_availability_only_health_report_is_unknown_and_inert():
    response = record(
        direction="unknown",
        subject="unknown",
        direction_basis="availability_only",
        directional_claim="正文只说明球员仍在康复，未说明对本场结果的方向性影响。",
    )
    clean = validate_directional_observation(response, ACTION, DOCS)
    assert clean["direction"] == "unknown"
    assert directional_update(clean) is None


def test_unknown_direction_cannot_be_promoted_by_supports_relation():
    response = record(
        direction="unknown",
        subject="unknown",
        direction_basis="availability_only",
        directional_claim="正文有健康状态信息，但没有影响方向。",
    )
    clean = validate_directional_observation(response, ACTION, DOCS)
    assert clean["outcome"] == "supports"
    assert directional_update(clean) is None


def test_unknown_direction_repair_identifies_subject_and_stays_inert():
    # A real provider returned a correct availability-only classification but
    # used the article's team as subject. The repair must identify that field,
    # and correcting it must not turn availability into a probability update.
    response = record(direction="unknown", subject="home",
                      direction_basis="availability_only", outcome="refutes")
    with pytest.raises(ValueError, match="subject=unknown"):
        validate_directional_observation(response, ACTION, DOCS)
    assert response["subject"] == "home"
    response["subject"] = "unknown"
    clean = validate_directional_observation(response, ACTION, DOCS)
    assert clean["outcome"] == "refutes"
    assert directional_update(clean) is None


def test_unknown_direction_repair_identifies_invalid_basis():
    with pytest.raises(ValueError, match="direction_basis=availability_only"):
        validate_directional_observation(
            record(direction="unknown", subject="unknown", direction_basis="explicit_effect"),
            ACTION, DOCS)


@pytest.mark.parametrize("direction,subject,basis", [
    ("home_negative", "away", "explicit_effect"),
    ("away_positive", "away", "conflicting_effect"),
    ("mixed", "home", "conflicting_effect"),
    ("home_positive", "unknown", "availability_only"),
])
def test_direction_subject_and_basis_must_agree(direction, subject, basis):
    with pytest.raises(ValueError, match="direction"):
        validate_directional_observation(record(direction=direction, subject=subject,
                                                direction_basis=basis), ACTION, DOCS)


def test_no_answer_cannot_carry_directional_effect():
    response = record(outcome="no_answer")
    clean = validate_directional_observation(response, ACTION, DOCS)
    assert clean["direction"] == "unknown"
    assert clean["direction_basis"] == "none"
    assert directional_update(clean) is None


def test_non_match_specific_direction_is_downgraded_without_mutating_input():
    response = record(applicability="team_context")
    clean = validate_directional_observation(response, ACTION, DOCS)
    assert clean["outcome"] == "no_answer"
    assert clean["direction"] == "unknown"
    assert clean["subject"] == "unknown"
    assert clean["direction_basis"] == "none"
    assert response["outcome"] == "supports"
    assert directional_update(clean) is None


def test_exact_quote_and_failed_documents_remain_rejected():
    bad = record(evidence=[{"document_id": "doc:fixture-news", "quote": "The title says injury"}])
    with pytest.raises(ValueError, match="exact body substring"):
        validate_directional_observation(bad, ACTION, DOCS)
    failed = deepcopy(DOCS)
    failed[0]["kind"] = "search_snippet"
    with pytest.raises(ValueError, match="failed return"):
        validate_directional_observation(record(), ACTION, failed)


def test_legacy_observation_can_be_decorated_as_inert_unknown():
    legacy = {
        "schema": "semantic-scenario-observation-v1",
        "action_id": ACTION["action_id"], "outcome": "supports", "evidence": [],
        "applicability": "unresolved", "reason": "旧协议记录。",
    }
    decorated = decorate_legacy_observation(legacy)
    assert decorated["schema"] == DIRECTIONAL_VERSION
    assert decorated["direction"] == "unknown"
    assert directional_update(decorated) is None
    assert legacy["schema"] == "semantic-scenario-observation-v1"
