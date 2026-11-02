from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest
from jsonschema import ValidationError

from football_agent import live_claim_review as v1
from football_agent import live_claim_review_v2 as v2
from football_agent import live_source_assertions as source
from football_research.acquisition.context_predictor import _digest
from football_research.acquisition.semantic_f import VALUES, public_input


def example():
    body=("Barcola did not train on Thursday. Bradley trained normally on Friday.\n"
          "Ryan Gravenberch is eligible for selection after serving a suspension.\n"
          "I want to win games. We need points and would love to win them.")
    fixture={"physical_match_id":"test:review-v2","home_name":"Liverpool","away_name":"Brighton",
        "kickoff":"2026-10-25T14:00:00Z","competition_name":"Premier League","season":None}
    public=public_input(fixture,{"url":"https://example.test/news","title":"Team news","text":body},"body")
    public=source.with_source_context(public,{"observed_at":"2026-10-25T09:00:00Z"},
        {"source_published_at":"2026-10-24T12:00:00Z"},fixture)
    base={"team":"home","kind":"other","value":"unmodeled","scope":"this_match"}
    claims=[
        {**base,"entity":"Bradley","concept":"training_participation","statement":"Bradley trained normally on Friday.","quotes":["Bradley trained normally on Friday."]},
        {**base,"entity":"Barcola","concept":"training_participation","statement":"Barcola did not train on Thursday.","quotes":["Barcola did not train on Thursday."]},
        {**base,"entity":"Ryan Gravenberch","concept":"confirmed_squad","statement":"Ryan Gravenberch is eligible for selection.",
         "quotes":["Ryan Gravenberch is eligible for selection after serving a suspension."]},
        {**base,"entity":"Liverpool","kind":"stakes","value":"must_win","concept":None,
         "statement":"The coach wants to win and needs points.","quotes":["I want to win games. We need points and would love to win them."]},
    ]
    return public,claims


def supported(index,claim):
    result={"index":index,"duplicate":False,"duplicate_reason":"No earlier identical source fact."}
    for dimension in v2.DIMENSIONS:
        result[dimension]="not_applicable" if dimension=="concept_support" and claim["concept"] is None else "supported"
        result[v2.REASONS[dimension]]="Explicit source and the supplied specific definition checked."
    return result


def response_for(claims):
    return {"schema":v2.VERSION,"reviews":[supported(i,c) for i,c in enumerate(claims)]}


def test_input_carries_complete_source_immutable_claims_and_exact_versioned_definitions():
    public,claims=example();old=deepcopy((public,claims));request=v2.provider_input(public,claims)
    assert request["source"]["text"]==public["text"]
    assert request["source"]["source_context"]==public["source_context"]
    assert "input_id" not in request["source"]
    assert request["body_sha256"]==sha256(public["text"].encode("utf-8")).hexdigest()
    assert request["criteria"]["direct_kind_value_pairs"]==VALUES
    assert request["criteria"]["concept_definitions"]==source.CONCEPTS
    assert set(request["criteria"]["predicate_definitions"])=={k+":"+x for k,values in VALUES.items() for x in values}
    for i,c in enumerate(claims):
        assert request["claims"][i]["claim"]==c
        assert request["claims"][i]["predicate_definition"]==v2.PREDICATE_DEFINITIONS[c["kind"]+":"+c["value"]]
        assert request["claims"][i]["concept_definition"]==(source.CONCEPTS[c["concept"]] if c["concept"] else None)
    assert "necessary" in request["claims"][3]["predicate_definition"]
    assert "would love to win" in request["claims"][3]["predicate_definition"]
    assert "sufficient" in request["criteria"]["predicate_definitions"]["stakes:draw_sufficient"]
    assert (public,claims)==old


def test_true_statements_with_wrong_concepts_or_predicates_are_rejected_independently():
    public,claims=example();response=response_for(claims)
    response["reviews"][1].update(concept_support="unsupported",concept_reason="training_participation requires actual participation in this event, but the statement says did not train on Thursday.")
    response["reviews"][2].update(concept_support="unsupported",concept_reason="confirmed_squad requires actual announced selection; eligible for selection is not that.")
    response["reviews"][3].update(predicate_support="unsupported",predicate_reason="stakes:must_win requires competitive necessity, whereas want/love and need points express a generic wish.")
    original=deepcopy((public,claims,response))
    review,binding=v2.bind_response(public,claims,v2.provider_input(public,claims),response)
    assert review["retained_indices"]==[0]
    assert [e["index"] for e in review["rejected"]]==[1,2,3]
    assert all(e["statement_support"]=="supported" for e in review["rejected"])
    assert all(e["verdict"]=="unsupported" for e in review["rejected"])
    assert "concept_support" in review["rejected"][0]["reason"]
    assert "predicate_support" in review["rejected"][2]["reason"]
    assert binding["review_criteria_sha256"]==_digest(v2.CRITERIA)
    assert binding["bound_review_sha256"]==_digest(review)
    assert (public,claims,response)==original


@pytest.mark.parametrize("dimension",v2.DIMENSIONS)
@pytest.mark.parametrize("value",["unsupported","uncertain"])
def test_any_non_supported_applicable_dimension_blocks_retention(dimension,value):
    public,claims=example();response=response_for(claims)
    response["reviews"][0][dimension]=value
    review,_=v2.bind_response(public,claims,v2.provider_input(public,claims),response)
    assert 0 not in review["retained_indices"]
    assert review["rejected"][0]["verdict"]==value


def test_duplicate_is_independent_of_four_support_checks():
    public,claims=example();response=response_for(claims)
    response["reviews"][1].update(duplicate=True,duplicate_of=0,duplicate_reason="Same source event already represented.")
    review,_=v2.bind_response(public,claims,v2.provider_input(public,claims),response)
    assert 1 not in review["retained_indices"]
    assert review["rejected"][0]["verdict"]=="duplicate"
    assert review["rejected"][0]["statement_support"]=="supported"


@pytest.mark.parametrize("index,value",[(0,"not_applicable"),(3,"supported"),(3,"unsupported"),(3,"uncertain")])
def test_concept_not_applicable_is_exactly_bound_to_original_null(index,value):
    public,claims=example();response=response_for(claims);response["reviews"][index]["concept_support"]=value
    with pytest.raises(ValueError,match="exactly when"):
        v2.bind_response(public,claims,v2.provider_input(public,claims),response)


@pytest.mark.parametrize("field",[*v2.DIMENSIONS,*v2.REASONS.values(),"duplicate","duplicate_reason"])
def test_each_dimension_reason_and_duplicate_check_is_mandatory(field):
    public,claims=example();response=response_for(claims);del response["reviews"][0][field]
    with pytest.raises(ValidationError):v2.bind_response(public,claims,v2.provider_input(public,claims),response)


@pytest.mark.parametrize("field",[*v2.REASONS.values(),"duplicate_reason"])
def test_dimension_reasons_cannot_be_blank(field):
    public,claims=example();response=response_for(claims);response["reviews"][0][field]=" \n\t "
    with pytest.raises(ValueError,match="cannot be blank"):
        v2.bind_response(public,claims,v2.provider_input(public,claims),response)


@pytest.mark.parametrize("indices",[[0,1,2],[0,1,2,2],[0,1,2,4],[0,1,2,3,4]])
def test_every_original_index_reviewed_once_only(indices):
    public,claims=example();response=response_for(claims)
    response["reviews"]=[supported(i,claims[0]) for i in indices]
    with pytest.raises((ValueError,ValidationError)):
        v2.bind_response(public,claims,v2.provider_input(public,claims),response)


@pytest.mark.parametrize("target",[1,2,4,-1,True])
def test_duplicate_of_cannot_be_self_forward_absent_or_boolean(target):
    public,claims=example();response=response_for(claims)
    response["reviews"][1].update(duplicate=True,duplicate_of=target)
    with pytest.raises((ValueError,ValidationError)):
        v2.bind_response(public,claims,v2.provider_input(public,claims),response)


def test_duplicate_of_requires_duplicate_true():
    public,claims=example();response=response_for(claims);response["reviews"][1]["duplicate_of"]=0
    with pytest.raises(ValidationError):v2.bind_response(public,claims,v2.provider_input(public,claims),response)


@pytest.mark.parametrize("field,value",[("verdict","supported"),("claim",{}),("quotes",["Rewritten"]),
    ("concept","confirmed_starter"),("gain",.1),("prediction_after",[.2,.3,.5])])
def test_model_cannot_shortcut_dimensions_or_rewrite_facts(field,value):
    public,claims=example();response=response_for(claims);response["reviews"][0][field]=value
    with pytest.raises(ValidationError):v2.bind_response(public,claims,v2.provider_input(public,claims),response)


@pytest.mark.parametrize("mutation",["body","context","claim","predicate_definition","concept_definition","criteria","index","prediction"])
def test_all_source_claims_and_review_definitions_are_bound(mutation):
    public,claims=example();request=v2.provider_input(public,claims);response=response_for(claims)
    if mutation=="body":request["source"]["text"]+="Modified"
    elif mutation=="context":request["source"]["source_context"]["fixture_competition_name"]="Other squad"
    elif mutation=="claim":request["claims"][0]["claim"]["statement"]="Changed"
    elif mutation in ("predicate_definition","concept_definition"):request["claims"][0][mutation]="Changed definition"
    elif mutation=="criteria":request["criteria"]["predicate_definitions"]["stakes:must_win"]="Any desire to win"
    elif mutation=="index":request["claims"][0]["index"]=1
    elif mutation=="prediction":request["f_probability"]=[.2,.3,.5]
    with pytest.raises(ValueError,match="binding differs"):
        v2.bind_response(public,claims,request,response)


def test_no_outcome_or_forecast_can_enter_public_review_input():
    public,claims=example();public["realized_gain"]=.1
    with pytest.raises(ValueError):v2.provider_input(public,claims)


def test_empty_claims_skip_and_legacy_contract_remains_distinct():
    public,claims=example()
    with pytest.raises(ValueError,match="no provider request"):v2.provider_input(public,[])
    review,binding=v2.skip_review(public,[])
    assert review["status"]=="skipped_empty_claims"
    assert review["retained_indices"]==review["verdicts"]==review["rejected"]==[]
    assert binding["provider_calls"]==0 and binding["provider_response_sha256"] is None
    assert binding["review_criteria_sha256"]==_digest(v2.CRITERIA)
    assert v1.VERSION!=v2.VERSION and "predicate_support" not in v1.SCHEMA["properties"]["reviews"]["items"]["properties"]
    legacy={"schema":v1.VERSION,"reviews":[{"index":i,"verdict":"supported","reason":"Legacy decision."} for i in range(len(claims))]}
    with pytest.raises(ValidationError):v2.bind_response(public,claims,v2.provider_input(public,claims),legacy)


def test_actual_a807_claim_definitions_bound_without_modifying_v1_records():
    root=Path(__file__).resolve().parents[2]
    run=root/"var/parallel/agent/round2/runs/live-g/runs/lg-a807f42ea5284385a1cb98474e68d6fe"
    data=json.loads((run/"run.json").read_text(encoding="utf-8"))
    step=next(s for s in data["steps"] if "team-news-premier" in s["url"])
    path=run/step["semantic_record"];old=path.read_bytes();record=json.loads(old)
    request=v2.provider_input(record["input"],record["response"]["claims"])
    assert len(request["claims"])==24
    assert request["claims"][1]["claim"]["concept"]=="training_participation"
    assert "actually participated" in request["claims"][1]["concept_definition"]
    assert request["claims"][4]["claim"]["concept"]=="confirmed_squad"
    assert "confirms selection" in request["claims"][4]["concept_definition"]
    assert path.read_bytes()==old


def test_actual_e336_wish_is_a_predicate_control_not_a_new_api_claim():
    root=Path(__file__).resolve().parents[2]
    run=root/"var/parallel/agent/round2/runs/live-g/runs/lg-e33662b5d3c748c5ab508f39dda5aab3"
    data=json.loads((run/"run.json").read_text(encoding="utf-8"))
    step=next(s for s in data["steps"] if "andoni-iraola-press-conference-liverpool" in s["url"])
    record=json.loads((run/step["semantic_record"]).read_text(encoding="utf-8"))
    claims=record["response"]["claims"];request=v2.provider_input(record["input"],claims)
    assert claims[3]["kind"]=="stakes" and claims[3]["value"]=="must_win"
    assert "would love" in claims[3]["statement"]
    # Constructed dimension response tests conjunction only; not a claimed new
    # provider result. The source document is not combined with TeamNews.
    response=response_for(claims)
    response["reviews"][3].update(predicate_support="unsupported",predicate_reason="Generic wish to win does not establish stakes:must_win's necessary competitive condition.")
    review,_=v2.bind_response(record["input"],claims,request,response)
    assert 3 not in review["retained_indices"]
