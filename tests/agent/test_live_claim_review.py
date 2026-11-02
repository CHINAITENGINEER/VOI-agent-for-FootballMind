from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest
from jsonschema import ValidationError

from football_agent import live_source_assertions as source
from football_agent.live_claim_review import (
    LiveClaimReviewProtocol as Protocol, bind_response, provider_input, skip_review,
)
from football_research.acquisition.context_predictor import _digest
from football_research.acquisition.semantic_f import public_input


def example():
    body = ("Barcola did not train on Thursday. Bradley trained normally on Friday.\n"
            "Ryan Gravenberch is eligible for selection after serving a suspension.")
    fixture = {"physical_match_id": "test:review", "home_name": "Liverpool", "away_name": "Brighton",
               "kickoff": "2026-10-25T14:00:00Z", "competition_name": "Premier League", "season": None}
    public = public_input(fixture,{"url":"https://example.test/news", "title":"Team news", "text":body},"body")
    public = source.with_source_context(public,{"observed_at":"2026-10-25T09:00:00Z"},
        {"source_published_at":"2026-10-24T12:00:00Z"},fixture)
    claims = [
        {"team":"home","entity":"Bradley","kind":"other","value":"unmodeled","concept":"training_participation",
         "scope":"this_match","statement":"Bradley trained normally on Friday.","quotes":["Bradley trained normally on Friday."]},
        {"team":"home","entity":"Barcola","kind":"other","value":"unmodeled","concept":"training_participation",
         "scope":"this_match","statement":"Barcola did not train on Thursday.","quotes":["Barcola did not train on Thursday."]},
        {"team":"home","entity":"Ryan Gravenberch","kind":"other","value":"unmodeled","concept":"confirmed_squad",
         "scope":"this_match","statement":"Ryan Gravenberch is eligible for selection.",
         "quotes":["Ryan Gravenberch is eligible for selection after serving a suspension."]},
    ]
    response = {"schema":Protocol.version,"reviews":[
        {"index":0,"verdict":"supported","reason":"The acquired body explicitly reports Friday participation."},
        {"index":1,"verdict":"unsupported","reason":"Thursday non-participation does not support a positive training concept."},
        {"index":2,"verdict":"unsupported","reason":"Eligibility is not a confirmed squad announcement."},
    ]}
    return public,claims,response


def test_only_supported_original_indices_can_enter_f_without_rewriting_facts():
    public,claims,response = example();original = deepcopy((public,claims,response))
    request = provider_input(public,claims)
    review,binding = bind_response(public,claims,request,response)
    assert review["retained_indices"] == [0]
    assert [x["index"] for x in review["rejected"]] == [1,2]
    assert review["status"] == "completed"
    assert "claims" not in review
    assert "input_id" not in request["source"]
    assert request["source"]["text"] == public["text"]
    assert request["source"]["source_context"] == public["source_context"]
    assert request["claims"] == [{"index":i,"claim":c} for i,c in enumerate(claims)]
    assert binding["canonical_input_sha256"] == _digest(public)
    assert binding["original_claims_sha256"] == _digest(claims)
    assert binding["provider_input_sha256"] == _digest(request)
    assert binding["provider_response_sha256"] == _digest(response)
    assert binding["bound_review_sha256"] == _digest(review)
    assert binding["body_sha256"] == sha256(public["text"].encode("utf-8")).hexdigest()
    assert review["source_truth_independently_verified"] is False
    assert review["information_value_assessed"] is False
    assert review["rewrites_original_claims"] is False
    assert (public,claims,response) == original


def test_model_order_does_not_change_original_claim_order():
    public,claims,response=example();response["reviews"].reverse()
    review,_=bind_response(public,claims,provider_input(public,claims),response)
    assert [r["index"] for r in review["verdicts"]] == [0,1,2]


@pytest.mark.parametrize("verdict", ["unsupported","uncertain","duplicate"])
def test_no_non_supported_verdict_is_admitted(verdict):
    public,claims,response=example();response["reviews"][0]["verdict"]=verdict
    review,_=bind_response(public,claims,provider_input(public,claims),response)
    assert review["retained_indices"] == []
    assert len(review["rejected"]) == 3


def test_duplicate_can_point_only_to_an_earlier_original_index():
    public,claims,response=example()
    response["reviews"][2].update(verdict="duplicate",duplicate_of=0)
    review,_=bind_response(public,claims,provider_input(public,claims),response)
    assert review["rejected"][-1]["duplicate_of"] == 0
    assert review["retained_indices"] == [0]


@pytest.mark.parametrize("index,target", [(0,0),(0,1),(1,1),(1,2),(2,3),(2,-1),(2,True)])
def test_invalid_duplicate_references_are_rejected(index,target):
    public,claims,response=example()
    response["reviews"][index].update(verdict="duplicate",duplicate_of=target)
    with pytest.raises((ValueError,ValidationError)):
        bind_response(public,claims,provider_input(public,claims),response)


def test_supported_verdict_cannot_carry_duplicate_of():
    public,claims,response=example();response["reviews"][0]["duplicate_of"]=0
    with pytest.raises(ValidationError):bind_response(public,claims,provider_input(public,claims),response)


def test_whitespace_only_review_reason_is_rejected():
    public,claims,response=example();response["reviews"][0]["reason"]=" \n\t "
    with pytest.raises(ValueError,match="cannot be blank"):
        bind_response(public,claims,provider_input(public,claims),response)


@pytest.mark.parametrize("indices", [[0,1],[0,1,1],[0,1,3],[0,1,2,3],[0,1,True],[-1,1,2]])
def test_review_must_cover_exactly_each_original_index_once(indices):
    public,claims,response=example()
    response["reviews"]=[{"index":i,"verdict":"supported","reason":"Claim checked."} for i in indices]
    with pytest.raises((ValueError,ValidationError)):
        bind_response(public,claims,provider_input(public,claims),response)


@pytest.mark.parametrize("field,value", [("claim",{"entity":"Invented"}),("quotes",["Invented quote"]),
    ("statement","Invented statement"),("concept","confirmed_starter"),("prediction_after",[.2,.3,.5]),
    ("gain",.1),("supported_indices",[0,1,2])])
def test_review_response_cannot_inject_replacement_facts_or_predictions(field,value):
    public,claims,response=example();response["reviews"][0][field]=value
    with pytest.raises(ValidationError):bind_response(public,claims,provider_input(public,claims),response)


@pytest.mark.parametrize("extra", ["prediction_before","prediction_after","realized_gain","outcome_class","g_score"])
def test_no_probability_or_label_fields_can_enter_public_review_input(extra):
    public,claims,_=example();public[extra]=.2
    with pytest.raises(ValueError):provider_input(public,claims)


@pytest.mark.parametrize("mutation", ["body","context","claim","index","extra","body_sha"])
def test_request_is_bound_to_exact_source_context_and_original_claims(mutation):
    public,claims,response=example();request=provider_input(public,claims)
    if mutation=="body":request["source"]["text"]+="Modified"
    elif mutation=="context":request["source"]["source_context"]["fixture_competition_name"]="Women's competition"
    elif mutation=="claim":request["claims"][0]["claim"]["statement"]="Modified"
    elif mutation=="index":request["claims"][0]["index"]=1
    elif mutation=="extra":request["f_probability"]=[.2,.3,.5]
    elif mutation=="body_sha":request["body_sha256"]="0"*64
    with pytest.raises(ValueError,match="binding differs"):bind_response(public,claims,request,response)


def test_original_claims_require_valid_typed_pairs_and_exact_source_quotes():
    public,claims,_=example();claims[0].update(kind="availability",value="doubtful")
    with pytest.raises(ValidationError):provider_input(public,claims)
    _,claims,_=example();claims[0]["quotes"]=["An invented quotation absent from the acquired body."]
    with pytest.raises(ValueError):provider_input(public,claims)


def test_original_claim_limit_remains_24_and_no_truncation_occurs():
    public,claims,_=example();claims=[deepcopy(claims[0]) for _ in range(25)]
    original=deepcopy(claims)
    with pytest.raises(ValidationError):provider_input(public,claims)
    assert claims==original


def test_empty_claims_skip_locally_without_provider_input_or_facts():
    public,_,_=example()
    with pytest.raises(ValueError,match="no provider request"):provider_input(public,[])
    review,binding=skip_review(public,[])
    assert review["status"]=="skipped_empty_claims"
    assert review["verdicts"]==review["retained_indices"]==review["rejected"]==[]
    assert binding["provider_calls"]==0
    assert binding["provider_input_sha256"] is None and binding["provider_response_sha256"] is None
    assert binding["bound_review_sha256"]==_digest(review)
    _,claims,_=example()
    with pytest.raises(ValueError,match="Only empty"):skip_review(public,claims)


def test_title_only_review_is_rejected_and_unknown_protocol_is_rejected():
    public,claims,_=example()
    title=public_input(public["fixture"],public,"title")
    with pytest.raises(ValueError,match="title-only"):provider_input(title,claims)
    with pytest.raises(ValueError,match="Unknown"):provider_input(public,claims,protocol=source.LiveSourceAssertionProtocol)


def test_real_saved_team_news_is_reviewable_without_mutating_original_extraction():
    root=Path(__file__).resolve().parents[2]
    run=root/"var/parallel/agent/round2/runs/live-g/runs/lg-a807f42ea5284385a1cb98474e68d6fe"
    data=json.loads((run/"run.json").read_text(encoding="utf-8"))
    step=next(s for s in data["steps"] if "team-news-premier" in s["url"])
    record_path=run/step["semantic_record"]
    original=record_path.read_bytes();record=json.loads(original)
    request=provider_input(record["input"],record["response"]["claims"])
    assert len(request["claims"])==24
    assert request["source"]["text"]==step["body"]
    # An explicitly constructed protocol-control verdict is not a claimed new
    # API review or independent truth. It verifies rejection/index mechanics.
    verdicts=[{"index":i,"verdict":"supported","reason":"Protocol control; original source reviewed separately."}
              for i in range(24)]
    verdicts[1].update(verdict="unsupported",reason="Negative Thursday event has a positive training concept.")
    verdicts[4].update(verdict="unsupported",reason="Eligibility is not confirmed actual squad selection.")
    review,binding=bind_response(record["input"],record["response"]["claims"],request,
        {"schema":Protocol.version,"reviews":verdicts})
    assert set(review["retained_indices"])==set(range(24))-{1,4}
    assert binding["original_claims_sha256"]==_digest(record["response"]["claims"])
    assert record_path.read_bytes()==original
