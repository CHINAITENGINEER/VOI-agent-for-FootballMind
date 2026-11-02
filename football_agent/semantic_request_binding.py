"""Opt-in, request-scoped source extraction; old evidence bytes stay immutable.

The LLM supplies quoted assertions, not transport identifiers. The caller
binds the response to the exact synchronous request and validates every quote
before adapting it to the existing frozen f representation. This does not
repair or reinterpret a response from the legacy echo-ID protocol.
"""
from copy import deepcopy

from jsonschema import Draft202012Validator

from football_research.acquisition.context_predictor import _digest
from football_research.acquisition.semantic_f import (
    EXTRACTION_VERSION, PROMPT, SCHEMA, VALUES, public_input, validate_response,
)

VERSION = "request-bound-football-source-assertions-v2"
BINDING_SCHEMA = "host-bound-source-assertion-response-v1"
PROJECTION_SCHEMA = "observed-news-language-navigation-filter-v2"
LANGUAGE_CONTROLS = frozenset({
    "english", "ع", "العربية", "arabic", "español", "espanol", "spanish",
    "thai", "ไทย", "中文", "简体中文", "繁體中文", "chinese", "日本語",
    "한국어", "français", "francais", "deutsch", "português", "portugues",
    "italiano", "bahasa indonesia", "bahasa melayu", "tiếng việt",
    "select language", "choose language", "language", "languages",
})
PROJECTION_MANIFEST = {
    "schema": PROJECTION_SCHEMA,
    "base_projection": "observed-news-titles-v1",
    "rule": "exact_casefolded_whitespace_normalized_observed_language_control",
    "excluded_labels": sorted(LANGUAGE_CONTROLS),
    "scope": "structural_ui_only_not_match_relevance_or_information_value",
}

_WIRE_SCHEMA = deepcopy(SCHEMA)
_WIRE_SCHEMA["properties"].pop("input_id")
_WIRE_SCHEMA["required"].remove("input_id")
_WIRE_SCHEMA["properties"]["schema"] = {"const": VERSION}
_WIRE_PROMPT = PROMPT.replace(
    "Return the exact JSON schema. Copy input_id.",
    "Return the exact JSON schema. Do not return input_id or any transport hash. "
    "The caller binds this single response to this single supplied input.")


class RequestBoundSourceAssertionProtocol:
    version, prompt, schema = VERSION, _WIRE_PROMPT, _WIRE_SCHEMA


TYPED_VERSION = "request-bound-football-source-assertions-v3"
_TYPED_SCHEMA = deepcopy(_WIRE_SCHEMA)
_TYPED_SCHEMA["properties"]["schema"] = {"const": TYPED_VERSION}
_TYPED_SCHEMA["properties"]["claims"]["items"]["oneOf"] = [
    {"properties": {"kind": {"const": kind}, "value": {"enum": list(values)}}}
    for kind, values in VALUES.items()
]
_TYPED_PROMPT = _WIRE_PROMPT + "\n\nThe ONLY allowed kind/value pairs are:\n" + "\n".join(
    f"{kind}: {', '.join(values)}" for kind, values in VALUES.items()
) + """\nChoose kind and value together. unmodeled is valid ONLY with kind=other,
never with availability, selection, tactical_plan, schedule or stakes.
For example, 'No players are suspended' describes the squad as a whole and
does not establish every player's availability: preserve it as other/unmodeled
if retained. Do not invent an availability claim to fit the vocabulary.
Before returning JSON, verify every pair against this table, and every quote
against the supplied text. Do not rewrite or shorten quotations with ellipses.
"""


class TypedRequestBoundSourceAssertionProtocol:
    version, prompt, schema = TYPED_VERSION, _TYPED_PROMPT, _TYPED_SCHEMA


# A validation-repair call is still the same typed-v3 wire contract.  The
# protocol differs only in the system instructions: it is an explicit,
# versioned second interpretation of the same acquired text, never a local
# rewrite of the first response.  Keeping the schema/version identical means
# the repaired response must pass the exact same host binding and semantic
# validation as an ordinary response.
REPAIR_TYPED_PROMPT_VERSION = "request-bound-football-source-assertions-v3-repair-v1"
_REPAIR_TYPED_PROMPT = _TYPED_PROMPT + """

This is a bounded validation-repair pass for the same single input. Re-read
the supplied text and emit a fresh complete JSON object. Do not copy an
invalid kind/value pair from an earlier attempt. Before returning, check each
claim against the allowed-pair table above. If a statement does not support a
modeled pair, use kind=other and value=unmodeled, or omit it. Never emit
kind=other with a non-unmodeled value, and never emit value=unmodeled with a
non-other kind. Preserve exact continuous quotes from the supplied text.
"""


class RepairTypedRequestBoundSourceAssertionProtocol:
    # The response remains typed-v3 on the wire; only the prompt is versioned
    # so the policy manifest can bind the repair behavior explicitly.
    version, prompt, schema = TYPED_VERSION, _REPAIR_TYPED_PROMPT, _TYPED_SCHEMA


def _known_protocol(protocol):
    if protocol not in (RequestBoundSourceAssertionProtocol,
                        TypedRequestBoundSourceAssertionProtocol,
                        RepairTypedRequestBoundSourceAssertionProtocol):
        raise ValueError("Unregistered request-bound extraction protocol")
    return protocol


def provider_input(public, *, protocol=RequestBoundSourceAssertionProtocol):
    """Validate the canonical local input and expose no opaque ID to copy."""
    if public_input(public["fixture"], public, public["channel"]) != public:
        raise ValueError("Canonical source input binding differs")
    result = {k: deepcopy(v) for k, v in public.items() if k != "input_id"}
    result["schema"] = _known_protocol(protocol).version
    return result


def bind_response(public, request, response, *, protocol=RequestBoundSourceAssertionProtocol):
    """Called only on a response returned by this request, or its saved replay.

    Do not pass a legacy response here: extra fields and old schemas fail.
    The wire response is archived separately, including validation failures.
    """
    protocol = _known_protocol(protocol)
    if request != provider_input(public, protocol=protocol):
        raise ValueError("Provider request differs from the local source input")
    Draft202012Validator(protocol.schema).validate(response)
    bound = {"schema": EXTRACTION_VERSION, "input_id": public["input_id"],
             "claims": deepcopy(response["claims"])}
    validate_response(public, bound)
    binding = {"schema": BINDING_SCHEMA,
        "canonical_input_sha256": _digest(public),
        "provider_input_sha256": _digest(request),
        "provider_response_sha256": _digest(response),
        "bound_response_sha256": _digest(bound),
        "binding_basis": "synchronous_call_return_not_model_echo"}
    return bound, binding


def filter_language_controls(links):
    """Only already visible link labels are inspected; bodies are untouched."""
    kept, excluded = [], []
    for index, link in enumerate(links):
        title = link.get("title")
        plain = " ".join(title.split()).casefold() if isinstance(title, str) else None
        if plain in LANGUAGE_CONTROLS:
            excluded.append({"index": index, "url": link.get("url"), "title": title,
                "reason": "observed_language_navigation_control"})
        else:
            kept.append(deepcopy(link))
    return kept, {"schema": PROJECTION_SCHEMA,
        "manifest_sha256": _digest(PROJECTION_MANIFEST),
        "input_labels": len(links), "retained_labels": len(kept),
        "excluded_labels": len(excluded), "excluded": excluded,
        "scope": PROJECTION_MANIFEST["scope"]}
