from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest
from jsonschema import ValidationError

from football_agent import live_source_assertions as ordinary
from football_agent.live_source_spans import (
    MAX_SPAN_CHARACTERS, SpanRepairLiveSourceAssertionProtocol as Protocol,
    bind_response, body_spans, provider_input,
)
from football_agent.semantic_request_binding import TypedRequestBoundSourceAssertionProtocol
from football_research.acquisition.context_predictor import _digest
from football_research.acquisition.semantic_f import public_input


def example(body=None, *, title="Team news"):
    body = body or "Christie has returned to training, but it is too early for him to play against Liverpool."
    fixture = {"physical_match_id": "test:fixture", "home_name": "Liverpool", "away_name": "Bournemouth",
               "kickoff": "2026-10-26T14:00:00+00:00", "competition_name": "Premier League", "season": "2026/27"}
    public = public_input(fixture, {"url": "https://example.test/team-news", "title": title, "text": body}, "body")
    public = ordinary.with_source_context(public, {"observed_at": "2026-10-25T09:00:00Z"},
        {"source_published_at": "2026-10-24T16:00:00Z", "publication_raw": "24 October 2026"}, fixture)
    claim = {"team": "away", "entity": "Christie", "kind": "other", "value": "unmodeled",
             "scope": "this_match", "statement": "Training participation does not establish match availability.",
             "quote_ids": ["q0001"], "concept": "training_participation"}
    return public, {"schema": Protocol.version, "claims": [claim]}


def test_span_protocol_copies_only_host_body_and_preserves_source_context_binding():
    public, response = example();request = provider_input(public)
    bound, binding = bind_response(public, request, response)
    assert "input_id" not in request
    assert request["text"] == public["text"]
    assert request["source_context"] == public["source_context"]
    assert request["body_sha256"] == sha256(public["text"].encode("utf-8")).hexdigest()
    assert bound["schema"] == ordinary.VERSION
    assert bound["input_id"] == public["input_id"]
    assert bound["claims"][0]["quotes"] == [public["text"]]
    assert "quote_ids" not in bound["claims"][0]
    assert bound["claims"][0]["concept"] == "training_participation"
    assert binding["canonical_input_sha256"] == _digest(public)
    assert binding["provider_input_sha256"] == _digest(request)
    assert binding["provider_response_sha256"] == _digest(response)
    assert binding["bound_response_sha256"] == _digest(bound)
    assert binding["body_sha256"] == request["body_sha256"]
    assert binding["selected_spans"] == [{"claim_index": 0, "quote_id": "q0001", "start": 0,
        "end": len(public["text"]), "text_sha256": request["body_sha256"]}]
    assert binding["semantic_truth_independently_verified"] is False
    assert binding["modifies_previous_failed_claims"] is False


@pytest.mark.parametrize("length", [8, 1599, 1600, 1601, 1607, 1608, 3199, 3200, 3201, 3207, 10001])
def test_long_paragraph_is_not_truncated_or_given_unusable_tiny_tail(length):
    body = "甲" * length;spans = body_spans(body)
    assert "".join(s["text"] for s in spans) == body
    assert spans[0]["start"] == 0 and spans[-1]["end"] == length
    assert all(8 <= len(s["text"]) <= MAX_SPAN_CHARACTERS for s in spans)
    assert all(left["end"] == right["start"] for left,right in zip(spans, spans[1:]))
    assert body_spans(body) == spans


@pytest.mark.parametrize("separator", ["\n\n", "\r\n\r\n", "\r\r", "\n \t\n"])
def test_paragraphs_keep_every_original_character_and_unicode_offset(separator):
    body = "😀球员é参加训练，未获放行。" + separator + "第二段：awaiting a decision." + separator + "\n尾段😀未截断。  "
    spans = body_spans(body)
    assert "".join(s["text"] for s in spans) == body
    assert spans[0]["text"] == body[:body.index(separator)+len(separator)]
    for span in spans:
        assert span["text"] == body[span["start"]:span["end"]]
        assert span["end"]-span["start"] == len(span["text"])
    # Emoji, CJK and composed accent have distinct UTF-8/UTF-16 lengths;
    # these positions are explicitly Python Unicode code points.
    assert len(spans[0]["text"].encode("utf-8")) != spans[0]["end"]
    assert len(spans[0]["text"].encode("utf-16-le"))//2 != spans[0]["end"]


def test_tail_can_be_selected_without_copying_or_rewriting_its_text():
    body = "背景信息 " * 1200 + "\n\nThe player will not play against Liverpool because of injury."
    public, response = example(body);request = provider_input(public)
    tail = request["quote_spans"][-1]
    response["claims"][0].update(concept=None, kind="availability", value="unavailable", quote_ids=[tail["quote_id"]])
    bound,binding = bind_response(public,request,response)
    assert "will not play" in bound["claims"][0]["quotes"][0]
    assert binding["selected_spans"][0]["end"] == len(body)
    assert request["span_policy"]["truncated_characters"] == 0


def test_title_never_creates_spans_and_title_only_input_is_rejected():
    public,response = example(title="Christie is fully fit and will start")
    changed,_ = example(public["text"], title="A different unrelated title")
    request = provider_input(public);other = provider_input(changed)
    assert request["quote_spans"] == other["quote_spans"]
    assert request["body_sha256"] == other["body_sha256"]
    assert all("fully fit" not in span["text"] for span in request["quote_spans"])
    title_public = public_input(public["fixture"], public, "title")
    with pytest.raises(ValueError,match="title-only"):
        provider_input(title_public)
    response["claims"][0]["quotes"] = [public["title"]]
    with pytest.raises(ValidationError):bind_response(public,request,response)


@pytest.mark.parametrize("ids", [["q9999"], ["q0000"], ["q0001", "q0001"], ["q1"], [], ["q0001"]*5])
def test_unknown_duplicate_missing_and_malformed_ids_are_rejected(ids):
    public,response = example();response["claims"][0]["quote_ids"] = ids
    with pytest.raises((ValueError,ValidationError)):
        bind_response(public,provider_input(public),response)


@pytest.mark.parametrize("field,value", [("quotes",["Injected fake text"]),("start",1),("input_id","invented"),
                                          ("body_sha256","invented"),("quote_text","invented")])
def test_wire_cannot_inject_quote_text_or_transport_fields(field,value):
    public,response = example();response["claims"][0][field] = value
    with pytest.raises(ValidationError):bind_response(public,provider_input(public),response)


@pytest.mark.parametrize("mutation", ["span_text", "span_start", "body_sha", "context", "title", "body"])
def test_request_tampering_cannot_be_bound(mutation):
    public,response = example();request = provider_input(public)
    if mutation == "span_text":request["quote_spans"][0]["text"] = "Injected source"
    elif mutation == "span_start":request["quote_spans"][0]["start"] = 1
    elif mutation == "body_sha":request["body_sha256"] = "0"*64
    elif mutation == "context":request["source_context"]["fixture_competition_name"] = "Women's Super League"
    elif mutation == "title":request["title"] = "Modified headline"
    elif mutation == "body":request["text"] += " Modified body"
    with pytest.raises(ValueError,match="binding differs"):
        bind_response(public,request,response)


def test_training_concept_cannot_become_medical_available():
    public,response = example()
    response["claims"][0].update(kind="availability",value="available")
    with pytest.raises(ValidationError):bind_response(public,provider_input(public),response)
    response["claims"][0].update(kind="other",value="available",concept=None)
    with pytest.raises(ValidationError):bind_response(public,provider_input(public),response)
    response["claims"][0].update(kind="availability",value="unavailable",concept=None)
    assert bind_response(public,provider_input(public),response)[0]["claims"][0]["concept"] is None


def test_old_protocol_schemas_are_unchanged_and_repair_response_cannot_enter_old_binder():
    public,response = example()
    for protocol in (ordinary.LiveSourceAssertionProtocol,ordinary.RepairLiveSourceAssertionProtocol,
                     TypedRequestBoundSourceAssertionProtocol):
        properties = protocol.schema["properties"]["claims"]["items"]["properties"]
        assert "quotes" in properties and "quote_ids" not in properties
    with pytest.raises(ValidationError):ordinary.bind_response(public,ordinary.provider_input(public),response)
    with pytest.raises(ValueError,match="Unknown body-span"):
        provider_input(public,protocol=ordinary.LiveSourceAssertionProtocol)


def test_normal_and_span_prompts_share_semantic_and_cardinality_rules():
    for protocol in (ordinary.LiveSourceAssertionProtocol, ordinary.RepairLiveSourceAssertionProtocol, Protocol):
        assert ordinary.SEMANTIC_CAUTION in protocol.prompt
        assert "HARD maximum of 24 claims" in protocol.prompt
        assert "concept MUST be null" in protocol.prompt
        assert "Do not infer" in protocol.prompt
        assert "NOT season_ambition" in protocol.prompt
        assert protocol.schema["properties"]["claims"]["maxItems"] == 24
    # The frozen historical typed-v3 contract receives no prompt changes.
    assert ordinary.SEMANTIC_CAUTION not in TypedRequestBoundSourceAssertionProtocol.prompt


def test_too_many_span_claims_are_rejected_as_whole_response_not_truncated():
    public,response=example()
    response["claims"]=[dict(response["claims"][0],entity="Player "+str(i)) for i in range(25)]
    original=deepcopy(response)
    with pytest.raises(ValidationError):bind_response(public,provider_input(public),response)
    assert response==original


def test_empty_claims_are_neutral_and_do_not_synthesize_evidence():
    public,_ = example();response = {"schema":Protocol.version,"claims":[]}
    bound,binding = bind_response(public,provider_input(public),response)
    assert bound["claims"] == [] and binding["selected_spans"] == []


def test_same_source_span_can_support_two_separately_attributed_entities():
    public,response = example("Christie and Gomez both trained, but neither is yet fit to play.")
    response["claims"].append(dict(response["claims"][0],entity="Gomez"))
    bound,binding = bind_response(public,provider_input(public),response)
    assert len(bound["claims"]) == 2
    assert [x["claim_index"] for x in binding["selected_spans"]] == [0,1]


def test_exact_provenance_does_not_claim_semantic_truth_or_rewrite_failed_response():
    public,response = example();failed_response = deepcopy(response)
    # Structural provenance checks intentionally do not act as an independent
    # semantic oracle: an erroneous statement may still cite a real span.
    response["claims"][0]["statement"] = "The model supplied a semantically wrong interpretation."
    _,binding = bind_response(public,provider_input(public),response)
    assert binding["semantic_truth_independently_verified"] is False
    assert failed_response["claims"][0]["statement"] != response["claims"][0]["statement"]
    assert binding["modifies_previous_failed_claims"] is False


def test_real_liverpool_failed_article_full_body_is_partitioned_without_network():
    run = Path(__file__).resolve().parents[2]/"var/parallel/agent/round2/runs/live-g/runs/lg-8b2cb8d051c64e6586af484f2c8f8f37"
    paths = list((run/"provider").glob("*/input.json"))
    assert len(paths) == 3
    requests = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    lengths = []
    for old in requests:
        # Request body is the canonical source, not the invalid response's
        # quotes or any selection of its failed claims.
        canonical = public_input(old["fixture"],old,"body")
        if "source_context" in old:canonical["source_context"] = deepcopy(old["source_context"])
        request = provider_input(canonical)
        assert "".join(s["text"] for s in request["quote_spans"]) == old["text"]
        assert request["source_context"] == old["source_context"]
        assert request["body_sha256"] == sha256(old["text"].encode("utf-8")).hexdigest()
        assert request["quote_spans"][-1]["end"] == len(old["text"])
        lengths.append(len(old["text"]))
    assert len(set(lengths)) == 2  # first body, and initial/repair of the next body
