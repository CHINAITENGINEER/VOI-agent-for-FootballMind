"""A conservative, opt-in extension for directional evidence.

``semantic_scenario_protocol`` intentionally keeps the frozen mechanism-03
contract small: an accepted body can be ``supports``/``refutes`` or
``no_answer``.  That relation says whether a source answered *the action's
question*; it does not say whether the source is good or bad for the home or
away team.  This module adds that missing axis without changing the frozen
protocol or any saved run.

The extension is deliberately inert by default.  A body which merely reports
an injury, rehabilitation, training or selection status receives
``direction=unknown`` unless it explicitly states a match-relevant effect.
Callers must use :func:`directional_update` before allowing a directional
label to influence a model.  A legacy observation can therefore be recorded
alongside this field while continuing to replay under the old model.

These labels are source-semantic classifications, not independently verified
truth and not calibrated effect sizes.
"""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator, ValidationError


DIRECTIONAL_VERSION = "semantic-directional-observation-v1"
DIRECTIONS = (
    "home_negative",
    "home_positive",
    "away_negative",
    "away_positive",
    "mixed",
    "unknown",
)
_TEAM_DIRECTIONS = frozenset(DIRECTIONS[:4])
_SUBJECTS = ("home", "away", "both", "unknown")
_BASES = (
    "explicit_effect",       # text explicitly states an effect for a team
    "conflicting_effect",    # text gives opposing/team-mixed effects
    "availability_only",     # status is reported, effect is not stated
    "ambiguous_effect",      # effect may exist but direction is unresolved
    "none",                  # no usable body or no directional assertion
)


_ID = {"type": "string", "pattern": "^[a-zA-Z0-9_.:-]{1,240}$"}
_TEXT = {"type": "string", "minLength": 1, "maxLength": 2400}
_DIRECTION = {"type": "string", "enum": list(DIRECTIONS)}
_SUBJECT = {"type": "string", "enum": list(_SUBJECTS)}
_BASIS = {"type": "string", "enum": list(_BASES)}
_EVIDENCE = {
    "type": "array", "maxItems": 8,
    "items": {
        "type": "object",
        "properties": {
            "document_id": _ID,
            "quote": {"type": "string", "minLength": 8, "maxLength": 1200},
        },
        "required": ["document_id", "quote"],
        "additionalProperties": False,
    },
}


# All fields are required on purpose.  An absent direction must be represented
# as the explicit value ``unknown`` rather than silently treated as positive or
# negative by a downstream adapter.
DIRECTIONAL_SCHEMA = {
    "type": "object",
    "properties": {
        "schema": {"type": "string", "const": DIRECTIONAL_VERSION},
        "action_id": _ID,
        "outcome": {"type": "string", "enum": ["supports", "refutes", "no_answer"]},
        "direction": _DIRECTION,
        "subject": _SUBJECT,
        "direction_basis": _BASIS,
        "directional_claim": _TEXT,
        "evidence": _EVIDENCE,
        "applicability": {"type": "string", "enum": ["match_specific", "team_context", "unresolved"]},
        "reason": _TEXT,
    },
    "required": [
        "schema", "action_id", "outcome", "direction", "subject",
        "direction_basis", "directional_claim", "evidence", "applicability",
        "reason",
    ],
    "additionalProperties": False,
}


DIRECTIONAL_PROMPT_SUFFIX = """\n
Add a directional classification only when the acquired BODY explicitly
supports a match-relevant effect.  `direction` means effect relative to the
named team: home_negative/home_positive/away_negative/away_positive.  Use
`mixed` for explicitly conflicting or both-team effects and `unknown` when the
body merely says a player is injured, recovering, training, selected or absent
without stating the effect on this fixture.  The source's existence of an
injury is not itself negative evidence.  `direction_basis` must be
explicit_effect for one-team labels, conflicting_effect for mixed, and
availability_only/ambiguous_effect/none for unknown.  `directional_claim` is
the bounded source-semantic claim, not world-truth verification or a numeric
effect size.  The numerical solver must ignore directional_update(None).
`subject` labels the resolved EFFECT, not merely the team mentioned in the
article: use home for home_negative/home_positive, away for
away_negative/away_positive, both for mixed, and unknown for unknown.
In particular, an availability-only report about the home team must still
use direction=unknown, subject=unknown, direction_basis=availability_only.
"""


def _validate_schema(response: Mapping[str, Any]) -> None:
    try:
        Draft202012Validator(DIRECTIONAL_SCHEMA).validate(response)
    except ValidationError as exc:
        raise ValueError("Directional evidence schema rejected: " + exc.message) from None


def _document_id(document: Mapping[str, Any]):
    if "id" in document and "document_id" in document and document["id"] != document["document_id"]:
        raise ValueError("Conflicting public document identities")
    return document.get("document_id", document.get("id"))


def _exact_evidence(response: Mapping[str, Any], documents: Sequence[Mapping[str, Any]]) -> None:
    by_id = {_document_id(document): document for document in documents}
    if None in by_id or len(by_id) != len(documents):
        raise ValueError("Missing or duplicate document identities")
    for item in response["evidence"]:
        document = by_id.get(item["document_id"])
        if document is None:
            raise ValueError("Directional evidence cites an unreturned document")
        if document.get("status", "OK") != "OK" or document.get("kind", "page") not in {"page", "article"}:
            raise ValueError("Directional evidence cites failed return or discovery snippet")
        body = document.get("text")
        if not isinstance(body, str) or item["quote"] not in body:
            raise ValueError("Directional quote is not an exact body substring")


def _expected_subject(direction: str) -> str:
    if direction.startswith("home_"):
        return "home"
    if direction.startswith("away_"):
        return "away"
    if direction == "mixed":
        return "both"
    return "unknown"


def validate_directional_observation(
    response: Mapping[str, Any], action: Mapping[str, Any], documents: Sequence[Mapping[str, Any]]
) -> dict:
    """Validate and conservatively canonicalize a directional observation.

    The old ``outcome`` remains in the returned record for audit/replay.  The
    new direction is a separate, opt-in axis.  Unsupported applicability or a
    failed return is downgraded to ``outcome=no_answer, direction=unknown``;
    this mirrors the old protocol's safety gate while preserving the original
    response in the caller's input object.
    """
    _validate_schema(response)
    if response["action_id"] != action["action_id"]:
        raise ValueError("Directional observation action_id differs from executed action")
    _exact_evidence(response, documents)
    clean = deepcopy(dict(response))

    direction = clean["direction"]
    basis = clean["direction_basis"]
    subject = clean["subject"]
    if direction in _TEAM_DIRECTIONS:
        if basis != "explicit_effect" or subject != _expected_subject(direction):
            raise ValueError("A team-direction label requires explicit_effect and matching subject")
        if not clean["evidence"]:
            raise ValueError("A directional effect requires acquired body evidence")
    elif direction == "mixed":
        if basis != "conflicting_effect" or subject != "both":
            raise ValueError("mixed direction requires conflicting_effect and both subjects")
        if not clean["evidence"]:
            raise ValueError("A mixed directional effect requires acquired body evidence")
    else:  # unknown
        if subject != "unknown":
            raise ValueError("unknown direction requires subject=unknown, even when the article names a team")
        if basis not in {"availability_only", "ambiguous_effect", "none"}:
            raise ValueError("unknown direction requires direction_basis=availability_only, ambiguous_effect or none")

    # Only a match-specific answer with exact body evidence can be accepted as
    # a source-semantic answer.  Unknown directional evidence is still useful
    # as an audit record, but it is deliberately inert for belief updates.
    if clean["outcome"] in {"supports", "refutes"}:
        if not clean["evidence"] or clean["applicability"] != "match_specific":
            clean["outcome"] = "no_answer"
            clean["direction"] = "unknown"
            clean["subject"] = "unknown"
            clean["direction_basis"] = "none"
            clean["directional_claim"] = "未获得本场适用的方向性正文证据。"
            clean["reason"] = ("未获得本场适用的方向性证据，保持原情景权重。" + clean["reason"])[:2400]
    elif clean["outcome"] == "no_answer":
        # A failed/no-answer return cannot smuggle a directional claim into a
        # posterior, even if a provider accidentally emitted one.
        clean["direction"] = "unknown"
        clean["subject"] = "unknown"
        clean["direction_basis"] = "none"
        clean["directional_claim"] = "没有可用于本场方向判断的正文证据。"

    return clean


def directional_update(observation: Mapping[str, Any]) -> dict | None:
    """Return an explicit effect update, or ``None`` when direction is inert.

    This is the only adapter a future directional Bayesian model should call.
    It prevents an old ``supports`` relation, or an availability-only injury
    report, from being interpreted as home/away positive or negative evidence.
    """
    _validate_schema(observation)
    if observation["outcome"] not in {"supports", "refutes"}:
        return None
    if observation["applicability"] != "match_specific" or not observation["evidence"]:
        return None
    direction = observation["direction"]
    if direction not in _TEAM_DIRECTIONS and direction != "mixed":
        return None
    expected_basis = "conflicting_effect" if direction == "mixed" else "explicit_effect"
    if observation["direction_basis"] != expected_basis:
        return None
    return {
        "direction": direction,
        "subject": observation["subject"],
        "claim": observation["directional_claim"],
        "evidence_document_ids": [item["document_id"] for item in observation["evidence"]],
        "semantics": "source_semantic_direction_not_independent_truth_or_calibrated_effect",
    }


def directional_update_for_action(
    observation: Mapping[str, Any], action: Mapping[str, Any]
) -> dict | None:
    """Apply the directional update only when it is bound to the action.

    ``directional_update`` remains available for replaying old, already-gated
    records.  New callers should use this action-bound adapter so a valid
    observation cannot be reused with a different question whose contract has
    the opposite polarity.
    """
    if not isinstance(action, Mapping):
        raise ValueError("directional action must be an object")
    if observation.get("action_id") != action.get("action_id"):
        return None
    contract = action.get("direction_contract")
    if not isinstance(contract, Mapping):
        return None
    outcome = observation.get("outcome")
    direction = observation.get("direction")
    if outcome in {"supports", "refutes"}:
        allowed = contract.get(outcome, ())
        # Do not treat a malformed scalar such as ``"home_negative"`` as a
        # sequence of characters.  The proposal validator normally guarantees
        # lists, but this adapter is also a defensive boundary for callers
        # that replay or construct action records directly.
        if not isinstance(allowed, (list, tuple, set, frozenset)) or direction not in allowed:
            return None
    return directional_update(observation)


def decorate_legacy_observation(
    observation: Mapping[str, Any], *,
    direction: str = "unknown",
    subject: str = "unknown",
    direction_basis: str = "none",
    directional_claim: str = "未进行方向性判断。",
) -> dict:
    """Add an explicit inert direction to a legacy observation.

    This helper does not alter the legacy object in place and is intended for
    reports or adapters that want one uniform record.  It does not reclassify
    existing ``supports``/``refutes`` semantics.
    """
    value = deepcopy(dict(observation))
    value.update({"schema": DIRECTIONAL_VERSION, "direction": direction,
                  "subject": subject, "direction_basis": direction_basis,
                  "directional_claim": directional_claim})
    _validate_schema(value)
    return value


__all__ = [
    "DIRECTIONAL_VERSION", "DIRECTIONS", "DIRECTIONAL_SCHEMA",
    "DIRECTIONAL_PROMPT_SUFFIX", "validate_directional_observation",
    "directional_update", "directional_update_for_action", "decorate_legacy_observation",
]
