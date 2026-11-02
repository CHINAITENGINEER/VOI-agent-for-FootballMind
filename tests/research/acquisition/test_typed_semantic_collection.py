"""Joint extraction vocabulary, preserved failures, real-f mocked collection."""
from copy import deepcopy
import json

import pytest
from jsonschema import Draft202012Validator, ValidationError

from football_agent.semantic_request_binding import (
    TYPED_VERSION, TypedRequestBoundSourceAssertionProtocol as Protocol,
    provider_input, bind_response,
)
from football_research.acquisition.semantic_f import VALUES, validate_response
from tests.research.acquisition.test_request_bound_semantic_collection import source_input, assertions
from tests.research.acquisition.test_entity_semantic_f_collection import (
    ROOT, TRACE_FILE, no_network, collect_entity, replay_without_provider, exported_quality_rows,
)
from tests.research.acquisition.test_verified_collection import verified_inputs


@pytest.mark.parametrize("kind,value", [(kind, value) for kind, values in VALUES.items() for value in values])
def test_every_declared_joint_pair_can_be_expressed_without_predicting_effects(kind, value):
    response = {"schema": TYPED_VERSION, "claims": assertions()}
    response["claims"][0].update(kind=kind, value=value)
    Draft202012Validator(Protocol.schema).validate(response)


@pytest.mark.parametrize("kind,value", [("availability", "unmodeled"), ("other", "available"),
                                      ("selection", "recovering"), ("stakes", "defensive")])
def test_joint_pairs_rejected_by_declared_schema_itself(kind, value):
    response = {"schema": TYPED_VERSION, "claims": assertions()}
    response["claims"][0].update(kind=kind, value=value)
    with pytest.raises(ValidationError):
        Draft202012Validator(Protocol.schema).validate(response)


def test_actual_v2_body_failure_stays_a_failure_and_input_is_not_repaired():
    path = ROOT / "var/f-entity-semantic-g-shadow-cumulative-b-20261011-pipelinev2-liverpool-v1/batch/match-3003906" / TRACE_FILE
    if not path.is_file():
        pytest.skip("Collector archive not included in this checkout")
    raw = path.read_bytes()
    failed = next(a for e in json.loads(raw)["events"] for a in e["attempts"] if a["status"] == "FAILED")
    assert failed["input"]["channel"] == "body"
    with pytest.raises(ValueError, match="predicate/value mismatch"):
        bind_response(failed["input"], failed["provider_input"], failed["provider_response"])
    # Even the stronger request contract must not relabel an old response as
    # a successful new call. Its raw bytes and status are still authoritative.
    request = provider_input(failed["input"], protocol=Protocol)
    with pytest.raises(ValidationError):
        bind_response(failed["input"], request, failed["provider_response"], protocol=Protocol)
    assert path.read_bytes() == raw


def test_v3_preserves_quote_verification_and_host_binding():
    item = source_input()
    request = provider_input(item, protocol=Protocol)
    response = {"schema": TYPED_VERSION, "claims": assertions()}
    bound, _ = bind_response(item, request, response, protocol=Protocol)
    assert validate_response(item, bound) == assertions()
    assert "input_id" not in request
    bad = deepcopy(response)
    bad["claims"][0]["quotes"] = ["An unobserved but plausible statement"]
    with pytest.raises(ValueError, match="exact acquired"):
        bind_response(item, request, bad, protocol=Protocol)


@pytest.mark.parametrize("mode", ["ok", "bad_quote", "failure", "late_provider"])
def test_v3_collection_body_consumption_quality_target_and_exact_replay(verified_inputs, monkeypatch, mode):
    folder, run, journal, report, calls = collect_entity(verified_inputs, monkeypatch,
        mode=mode, protocol_version="v3", language_links=True)
    assert calls and any(c["channel"] == "body" for c in calls)
    assert not any(c["text"] in {"English", "ع", "Español", "Thai"} for c in calls)
    trace = json.loads((folder / TRACE_FILE).read_bytes())
    assert trace["schema"] == "entity-semantic-shadow-extraction-trace-v3"
    for event in trace["events"]:
        for attempt in event["attempts"]:
            if attempt["status"] == "OK":
                bound, binding = bind_response(attempt["input"], attempt["provider_input"],
                    attempt["provider_response"], protocol=Protocol)
                assert bound == attempt["response"] and binding == attempt["response_binding"]
    replay_without_provider(folder, verified_inputs, monkeypatch)
    eligible = [r for r in exported_quality_rows(folder, verified_inputs)
                if r.get("g_quality_target", {}).get("status") == "eligible"]
    if mode == "ok":
        assert len(eligible) == 1 and eligible[0]["g_quality_target"]["realized_signed_gain"] != 0
    else:
        assert eligible == []
