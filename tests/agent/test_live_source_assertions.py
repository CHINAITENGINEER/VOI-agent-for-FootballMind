from copy import deepcopy

import pytest
from jsonschema import ValidationError

from football_agent.live_source_assertions import (
    LiveSourceAssertionProtocol as Protocol, bind_response, provider_input,
    with_source_context,
)
from football_agent.semantic_request_binding import TypedRequestBoundSourceAssertionProtocol
from football_research.acquisition.semantic_f import public_input


def example():
    body = "Christie has returned to training, but it is too early for him to play against Liverpool."
    fixture = {"physical_match_id": "test:fixture", "home_name": "Liverpool", "away_name": "Bournemouth",
               "kickoff": "2026-10-26T14:00:00+00:00"}
    public = public_input(fixture, {"url": "https://example.test/team-news", "title": "Team news", "text": body}, "body")
    claim = {"team": "away", "entity": "Christie", "kind": "other", "value": "unmodeled",
             "scope": "this_match", "statement": "Training participation does not establish match availability.",
             "quotes": [body], "concept": "training_participation"}
    return public, {"schema": Protocol.version, "claims": [claim]}


def test_new_explicit_concept_keeps_exact_source_binding_and_old_contract():
    public, response = example();request = provider_input(public)
    bound, binding = bind_response(public, request, response)
    assert bound["input_id"] == public["input_id"]
    assert bound["claims"][0]["concept"] == "training_participation"
    assert bound["claims"][0]["kind"] == "other"
    assert "input_id" not in request
    assert "concept" not in TypedRequestBoundSourceAssertionProtocol.schema["properties"]["claims"]["items"]["properties"]
    assert binding["binding_basis"] == "synchronous_request_and_exact_body_quotes"


def test_training_cannot_share_a_claim_with_medical_available_predicate():
    public, response = example()
    response["claims"][0].update(kind="availability", value="available")
    with pytest.raises(ValidationError):bind_response(public,provider_input(public),response)


def test_concept_does_not_relax_verbatim_quote_check():
    public, response = example()
    response["claims"][0]["quotes"]=["Christie is fully fit and available."]
    with pytest.raises(ValueError,match="exact acquired text"):
        bind_response(public,provider_input(public),response)


def test_direct_claim_uses_null_concept_and_preserves_pair_validation():
    public, response = example()
    response["claims"][0].update(concept=None,kind="availability",value="unavailable")
    assert bind_response(public,provider_input(public),response)[0]["claims"][0]["concept"] is None
    response["claims"][0]["value"]="unmodeled"
    with pytest.raises(ValidationError):bind_response(public,provider_input(public),response)


def test_no_claims_is_a_valid_neutral_extraction():
    public, _ = example(); response={"schema":Protocol.version,"claims":[]}
    assert bind_response(public,provider_input(public),response)[0]["claims"]==[]


def test_no_prediction_or_outcome_enters_extraction_request():
    public, _ = example(); public["prediction_after"]=[.3,.3,.4]
    with pytest.raises(ValueError,match="Canonical live body"):
        provider_input(public)


def test_observed_publication_context_is_bound_but_does_not_change_body_identity():
    public,response=example()
    contextual=with_source_context(public,{"observed_at":"2026-10-25T09:00:00Z"},
        {"source_published_at":None,"publication_raw":"24 October 2026"})
    assert contextual["input_id"]==public["input_id"]
    request=provider_input(contextual)
    assert request["source_context"]["publication_raw"]=="24 October 2026"
    assert request["source_context"]["source_published_at"] is None
    assert bind_response(contextual,request,response)[0]["input_id"]==public["input_id"]
    contextual["source_context"]["realized_gain"]=.1
    with pytest.raises(ValueError,match="explicit source"):
        provider_input(contextual)


def test_relative_publication_label_list_stays_literal_without_invented_time():
    public,response=example()
    contextual=with_source_context(public,{"source_published_at":"2026-10-24T08:31:00Z"},
        {"publication_raw":["a day ago","a day ago"]},{"competition_name":"Premier League","season":None})
    assert contextual["source_context"]["publication_raw"]=='["a day ago", "a day ago"]'
    assert contextual["source_context"]["source_published_at"]=="2026-10-24T08:31:00Z"
    assert contextual["source_context"]["fixture_competition_name"]=="Premier League"
    assert bind_response(contextual,provider_input(contextual),response)[0]["input_id"]==public["input_id"]
