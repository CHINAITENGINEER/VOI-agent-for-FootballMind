"""Live-only explicit concept extraction for the existing reviewed f branches.

Frozen typed-v3 and historical records are unchanged. Quotes and kind/value
validation retain the existing canonical body binding. A concept is a source
assertion supplied by the extractor, never inferred from a URL or probability.
"""
from copy import deepcopy
import json

from jsonschema import Draft202012Validator

from football_agent.semantic_request_binding import TypedRequestBoundSourceAssertionProtocol
from football_research.acquisition.context_predictor import _digest
from football_research.acquisition.semantic_f import (
    EXTRACTION_VERSION, public_input, validate_response,
)

VERSION = "request-bound-football-live-concepts-v1"
CONCEPTS = {
    "confirmed_starter": "The source explicitly names the player in the official starting XI for this fixture; not a predicted lineup.",
    "confirmed_bench": "The source explicitly names the player on the official substitute bench for this fixture.",
    "confirmed_squad": "The source explicitly confirms selection in the match squad for this fixture; squad membership is not medical clearance.",
    "training_participation": "The text explicitly reports that the named player actually participated in training. Mentoring, giving advice, being a role model, watching others, or generic discussion about training does not establish personal participation. Participation does not establish ability to play.",
    "partial_training": "The text explicitly describes partial/individual/limited training, not full match availability.",
    "recovery_not_ready": "The text explicitly states recovery is progressing but the player is not yet ready to play.",
    "match_decision_pending": "The source explicitly leaves a selection/fitness decision pending; preserve its condition and time.",
    "club_eligibility_pending": "Registration or other stated eligibility for this club/fixture is still pending.",
    "possible_bench": "The source explicitly says a bench appearance is possible, not confirmed.",
    "observed_lineup_change": "The text explicitly compares official selections and reports a change; do not reconstruct a change from memory.",
    "season_ambition": "An explicit season-level objective, not a mathematical must-win condition for this match.",
    "match_ambition": "An explicit objective or intention for the upcoming match; not a tactical attacking plan.",
    "season_attacking_ambition": "An explicit season-level attacking intention; not an established tactical plan for this fixture.",
    "match_preparation": "The text explicitly describes this team's preparations for the upcoming opponent/fixture.",
}
SEMANTIC_CAUTION = """
Every assertion's subject, action, cause, condition and time must be supported
by the selected quote. A defender teaching or advising younger teammates is
not evidence that the defender participated in training. Do not assign
training_participation to that defender unless their actual participation is
explicitly reported. 'Lost a player' or 'the situation with a player' does not
establish injury, suspension or a medical cause. Preserve unspecified reasons
as unspecified. A video introduction or programme advert is only the acquired
page text: do not invent the unseen transcript, photos or publication contents.

The response has a HARD maximum of 24 claims. Return one complete valid JSON
response within that limit, never a truncated response. Do not repeat the same
entity/fact/condition/time as separate assertions or in multiple feature lanes.
Distinct named players may have separate claims. Pure historical/statistical
detail may be omitted rather than exhaust the claim budget; never invent a
modeled assertion to replace an omitted detail.

For ANY direct typed predicate (availability, selection, tactical_plan,
schedule or stakes), concept MUST be null. In particular,
kind=availability/value=doubtful/concept=match_decision_pending is INVALID.
If a direct availability predicate is supported, use it with concept=null;
do not also emit a duplicate concept assertion for the same fact. A non-null
concept is permitted ONLY with kind=other and value=unmodeled. These rules
apply to every claim individually, including a validation-repair response.

An ambition is an explicitly stated objective or intention. Achieved wins,
points, table positions, goals scored/conceded, historical results and current
form are NOT season_ambition, season_attacking_ambition or match_ambition.
Such facts, if retained, use other/unmodeled with concept=null. Do not infer
an intention from performance statistics. Preserve conditional and expected
wording such as could, monitored, look set, if and pending in the statement;
do not turn those source assertions into confirmed medical clearance or a
guaranteed future event.
"""
SCHEMA = deepcopy(TypedRequestBoundSourceAssertionProtocol.schema)
SCHEMA["properties"]["schema"] = {"const": VERSION}
CLAIM_SCHEMA = SCHEMA["properties"]["claims"]["items"]
CLAIM_SCHEMA["properties"]["concept"] = {"enum": [None, *CONCEPTS]}
CLAIM_SCHEMA["required"].append("concept")
CLAIM_SCHEMA["allOf"] = [{
    "if": {"properties": {"concept": {"enum": list(CONCEPTS)}}, "required": ["concept"]},
    "then": {"properties": {"kind": {"const": "other"}, "value": {"const": "unmodeled"}}},
}]
PROMPT = TypedRequestBoundSourceAssertionProtocol.prompt + """

This live protocol additionally supports an explicit concept field on EVERY
claim. Use concept=null for an ordinary typed direct predicate. For a supported
context or selection concept below, use kind=other, value=unmodeled and the
exact concept name; this routes a separate feature branch, not an availability
claim. Other unsupported assertions use other/unmodeled with concept=null.
Only assign a concept actually supported by an exact body quote. Training,
recovery, lineup selection and medical availability remain distinct. Keep
historical/unresolved scopes when applicable; do not turn an old match into the
current fixture. Empty claims are valid when nothing can be attributed.
Preserve quotation language and exact continuous characters for all languages.
When source_context is supplied, it contains only observed source metadata.
Use its declared publication date together with the fixture date to avoid
transplanting an old fixture's news into this match. Missing publication is
UNKNOWN, not evidence of recency. Do not infer first public availability from
the response observation clock. Quotes must still come from text only.
Its fixture_competition_name identifies the selected competition: men's,
women's, youth and senior squads must stay distinct even if club names match.
Do not transfer a women's/youth assertion to the men's senior fixture.
""" + SEMANTIC_CAUTION + "\n".join(f"{key}: {value}" for key, value in CONCEPTS.items())


class LiveSourceAssertionProtocol:
    version, prompt, schema = VERSION, PROMPT, SCHEMA


class RepairLiveSourceAssertionProtocol:
    version, schema = VERSION, SCHEMA
    prompt = PROMPT + """
This is a single bounded validation-repair pass over the same original body.
Emit a fresh complete response. Check all quotes, kind/value combinations and
concept fields against the schema. Do not invent a claim to replace an invalid
claim. Do not emit transport hashes or input_id.
"""


SOURCE_CONTEXT_FIELDS = frozenset({"source_published_at", "publication_raw", "response_observed_at", "publication_status",
                                    "fixture_competition_name", "fixture_season"})


def with_source_context(public, document, candidate, fixture=None):
    result = deepcopy(public)
    raw = document.get("publication_raw") or candidate.get("publication_raw")
    if raw is not None and not isinstance(raw,str):
        raw = json.dumps(raw,ensure_ascii=False,sort_keys=True,allow_nan=False)
    fixture=fixture or {}
    result["source_context"] = {
        "source_published_at": document.get("source_published_at") or candidate.get("source_published_at"),
        "publication_raw": raw,
        "response_observed_at": document.get("observed_at"),
        "publication_status": document.get("publication_status"),
        "fixture_competition_name": fixture.get("competition_name"),
        "fixture_season": fixture.get("season"),
    }
    return result


def canonical_body(public):
    body = {k: v for k, v in public.items() if k != "source_context"}
    if public_input(public["fixture"], public, public["channel"]) != body:
        raise ValueError("Canonical live body input differs")
    if "source_context" in public:
        context = public["source_context"]
        if not isinstance(context, dict) or set(context) != SOURCE_CONTEXT_FIELDS or any(
                value is not None and not isinstance(value, str) for value in context.values()):
            raise ValueError("Only explicit source clock/publication metadata is allowed")
    return body


def provider_input(public, *, protocol=LiveSourceAssertionProtocol):
    if protocol not in (LiveSourceAssertionProtocol, RepairLiveSourceAssertionProtocol):
        raise ValueError("Unknown live extraction protocol")
    canonical_body(public)
    request = {k: deepcopy(v) for k, v in public.items() if k != "input_id"}
    request["schema"] = protocol.version
    return request


def bind_response(public, request, response, *, protocol=LiveSourceAssertionProtocol):
    if request != provider_input(public, protocol=protocol):
        raise ValueError("Live provider request binding differs")
    Draft202012Validator(protocol.schema).validate(response)
    base_claims = [{k: deepcopy(v) for k, v in claim.items() if k != "concept"} for claim in response["claims"]]
    validate_response(canonical_body(public), {"schema": EXTRACTION_VERSION, "input_id": public["input_id"], "claims": base_claims})
    bound = {"schema": VERSION, "input_id": public["input_id"], "claims": deepcopy(response["claims"])}
    binding = {"schema": "host-bound-live-concept-response-v1", "canonical_input_sha256": _digest(public),
               "provider_input_sha256": _digest(request), "provider_response_sha256": _digest(response),
               "bound_response_sha256": _digest(bound), "binding_basis": "synchronous_request_and_exact_body_quotes"}
    return bound, binding
