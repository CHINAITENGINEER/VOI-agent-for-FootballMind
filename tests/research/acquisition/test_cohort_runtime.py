"""Offline cohort -> actual policy -> archive -> replay contract checks.

All fixture/model data in this file are constructed test inputs.  Their
real_observational marker exercises the production origin gate; these are
never exported as real research samples or evidence of policy benefit.
"""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import socket

import pytest

from football_agent.learned_voi_replay import replay_saved_run
from football_agent.learned_voi_web import _web_action
from football_research.acquisition.text_returns import TextReturnModel
from football_research.acquisition.text_risk import TextRiskCalibrator
from football_research.data_adapters.prospective_learning import prepare_learning_plan
from football_research.data_adapters.prospective_text import (
    MarketPriorPolicy, _digest, _save, join_settlements,
)
from test_verified_collection import verified_inputs, _collect


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Cohort integration tests must never use the network")
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _past_inputs(inputs, *, registered, prefix, event_start):
    """Construct 12 separate prior fixture registrations, not real captures."""
    manifests, bindings = [], {}
    for index in range(12):
        old = inputs["binding"]["initialization_manifest"]["market_initialization"]
        fixture, quote = deepcopy(old["fixture"]), deepcopy(old["quote"])
        mid = prefix + str(index)
        fixture.update(match_id=mid, kickoff=(registered+timedelta(hours=1)).isoformat(),
                       fixture_available_at=(registered-timedelta(minutes=2)).isoformat())
        quote.update(match_id=mid, observed_at=(registered-timedelta(minutes=1)).isoformat())
        policy = MarketPriorPolicy(fixture, quote, frozen_at=registered,
                                   prediction_cutoff=registered+timedelta(minutes=59))
        binding = deepcopy(inputs["binding"])
        binding.update(match_id=mid, initialization_manifest=policy.manifest,
                       initialization_sha256=_digest(policy.manifest),
                       verified_at=registered.isoformat())
        binding["canonical"].update(source_event_id="uefa:match:"+str(event_start+index),
                                    kickoff=fixture["kickoff"])
        binding["binding_sha256"] = _digest({k: v for k, v in binding.items() if k != "binding_sha256"})
        manifests.append(policy.manifest)
        bindings[mid] = binding
    return {"schema": "prospective-text-initializations-v1", "selected": manifests}, {"bindings": bindings}


def _records(initializations, *, registered):
    texts = [
        "Croatia may rotate their starting lineup against Spain. Spain will assess player fitness before the match.",
        "Spain have a complete first team and settled defensive formation against Croatia.",
        "Croatia and Spain report balanced performances and uncertainty in midfield selection.",
    ]
    action = _web_action("search_web", "Croatia Spain", cost={"money": .01, "requests": 1.})
    records = []
    for index, manifest in enumerate(initializations["selected"]):
        market = manifest["market_initialization"]
        records.append({"match_id": market["fixture"]["match_id"], "outcome": index % 3,
            "prior_probabilities": market["probabilities"], "prediction_cutoff": market["prediction_cutoff"],
            "kickoff": market["fixture"]["kickoff"],
            "result_available_at": (registered+timedelta(hours=3)).isoformat(),
            "decision_at": (registered+timedelta(minutes=1)).isoformat(),
            "observed_at": (registered+timedelta(minutes=1, seconds=1)).isoformat(),
            "available_at": (registered+timedelta(minutes=1, seconds=2)).isoformat(),
            "history": [], "action": action,
            "response": {"status": "OK", "kind": "article", "text": texts[index % 3]}})
    return records


def _registered(inputs, purpose):
    root, now = inputs["root"], inputs["now"]
    previous, fpath, gpath = [], None, None
    if purpose != "f_train":
        a_at = now-timedelta(days=4)
        ai, ab = _past_inputs(inputs, registered=a_at, prefix="f-fixture-", event_start=8100000)
        a = prepare_learning_plan(ai, identity_bundle=ab, purpose="f_train", created_at=a_at)
        previous.append(a)
        f = TextReturnModel().fit(_records(ai, registered=a_at), as_of=now-timedelta(days=3),
            dataset_version="constructed-contract-fixture-only", evidence_origin="real_observational")
        fpath = root/"original-f.json"
        f.save(fpath)
        if purpose == "confirmation":
            b_at = now-timedelta(days=2)
            bi, bb = _past_inputs(inputs, registered=b_at, prefix="critic-fixture-", event_start=8200000)
            b = prepare_learning_plan(bi, identity_bundle=bb, purpose="calibration", created_at=b_at,
                                       previous_plans=[a], text_model_path=fpath)
            previous.append(b)
            g = TextRiskCalibrator().fit(_records(bi, registered=b_at), text_model=f,
                as_of=now-timedelta(days=1), dataset_version="constructed-contract-fixture-only",
                evidence_origin="real_observational")
            gpath = root/"original-critic.json"
            g.save(gpath)
    previous_paths = []
    for index, plan in enumerate(previous):
        path = root/f"previous-{index}.json"
        _save(path, plan)
        previous_paths.append(path)
    plan = prepare_learning_plan(inputs["plan"], identity_bundle={"bindings": {
        inputs["binding"]["match_id"]: inputs["binding"]}}, purpose=purpose, created_at=now,
        previous_plans=previous, text_model_path=fpath, text_risk_model_path=gpath)
    path = root/"learning-plan.json"
    _save(path, plan)
    return path, previous_paths, plan, fpath, gpath


@pytest.mark.parametrize("purpose", ["f_train", "calibration", "confirmation"])
def test_actual_policy_bound_before_http_and_portable_replay(verified_inputs, monkeypatch, purpose):
    inputs = verified_inputs
    path, previous, plan, fpath, gpath = _registered(inputs, purpose)
    folder, calls, report = _collect(inputs, monkeypatch, learning_plan_path=path,
                                     previous_learning_plan_paths=previous)
    config, run = _read(folder/"configuration.json"), _read(folder/"run.json")
    manifest = config["policy_manifest"]
    if purpose == "f_train":
        assert manifest["schema"] == "prospective-market-prior-policy-v1"
        assert "text_response_model" not in config
    else:
        assert manifest["schema"] == ("learned-text-strict-voi-v1" if purpose == "calibration"
                                       else "empirical-text-risk-policy-v1")
        assert config["text_response_model"]["sha256"] == sha256(fpath.read_bytes()).hexdigest()
        assert config["text_response_model"]["model_version"] == plan["artifacts"]["text_model"]["model_version"]
        if purpose == "confirmation":
            assert config["text_risk_model"]["model_version"] == plan["artifacts"]["text_risk_model"]["model_version"]
    assert config["supervisor_metadata"]["learning_cohort"]["source_event_id"] == "uefa:match:2046001"
    assert len(calls) == report["http_requests"] <= 2
    assert calls, "The fixed actor must actually traverse the acquisition path"
    for cycle in run["cycles"]:
        public = json.dumps(cycle["public_before"], sort_keys=True)
        assert plan["cohort_id"] not in public
        assert "learning_cohort" not in public and "prior_inventory" not in public
        if cycle["selected_action"]:
            assert datetime.fromisoformat(cycle["public_before"]["decision_at"]) > inputs["now"]
    proofs = _read(folder/"prior-bindings.json")
    audit = join_settlements([folder], settlements=[], prior_bindings=proofs, as_of=datetime.now(timezone.utc))
    assert audit["training_records"] == []
    assert audit["independent_result_verification"][0]["status"] == "PENDING"
    assert audit["dataset"]["episodes"]
    for episode in audit["dataset"]["episodes"]:
        assert episode["response_observation_eligible"]
        assert all(reason.startswith("independent_settlement_")
                   for reason in episode["training_exclusion_reasons"]), episode["training_exclusion_reasons"]
    # The recorded run must not depend on models at the original absolute paths.
    for source in (fpath, gpath):
        if source is not None:
            source.write_text("original source intentionally unavailable", encoding="utf-8")
    replay = replay_saved_run(folder, output=inputs["root"]/"replay", project_root=inputs["root"])
    assert replay["identical_replay"] and replay["replay_network_requests"] == 0


@pytest.mark.parametrize("change", ["plan_digest", "artifact_bytes", "future_registration"])
def test_cohort_failure_is_before_any_acquisition(verified_inputs, monkeypatch, change):
    inputs = verified_inputs
    path, previous, plan, fpath, _ = _registered(inputs, "calibration")
    if change == "plan_digest":
        plan["matches"][0]["source_event_id"] = "uefa:match:9999999"
        _save(path, plan)
    elif change == "artifact_bytes":
        fpath.write_bytes(fpath.read_bytes()+b" ")
    else:
        # A well-formed declared future registration still cannot run now.
        future = inputs["now"]+timedelta(minutes=1)
        plan = prepare_learning_plan(inputs["plan"], identity_bundle={"bindings": {
            inputs["binding"]["match_id"]: inputs["binding"]}}, purpose="calibration", created_at=future,
            previous_plans=[_read(p) for p in previous], text_model_path=fpath)
        _save(path, plan)
    with pytest.raises(ValueError):
        _collect(inputs, monkeypatch, learning_plan_path=path, previous_learning_plan_paths=previous)
    assert not (inputs["root"]/"batch").exists()


def test_new_link_and_unread_body_are_not_available_at_initial_decision(verified_inputs, monkeypatch):
    inputs = verified_inputs
    hidden = "https://www.uefa.com/uefanationsleague/news/0199-unseen-fixture-article/"
    sentinel = "UNREAD_SCOPE_SENTINEL"
    inputs["body"] = inputs["body"].replace(b"Croatia may rotate", (sentinel+" Croatia may rotate").encode())
    inputs["body"] = inputs["body"].replace(b"</html>", (f'<a href="{hidden}">Further squad report</a></html>').encode())
    path, previous, _, _, _ = _registered(inputs, "f_train")
    folder, _, _ = _collect(inputs, monkeypatch, learning_plan_path=path, previous_learning_plan_paths=previous)
    run = _read(folder/"run.json")
    earlier_documents = []
    observed_link = False
    for cycle in run["cycles"]:
        public = json.dumps(cycle["public_before"], sort_keys=True)
        menu = json.dumps(cycle["generated_actions"], sort_keys=True)
        earlier = json.dumps(earlier_documents, sort_keys=True)
        if hidden not in earlier:
            assert hidden not in public and hidden not in menu
        if sentinel not in earlier:
            assert sentinel not in public
        if cycle["selected_action"]:
            document = _read(folder/f"receipt-{cycle['step']:03}.json")["document"]
            if cycle.get("admission", {}).get("status") == "admitted":
                earlier_documents.append(document)
                observed_link = observed_link or hidden in json.dumps(document)
    assert observed_link, "The negative initial-state assertion must be followed by an actual observed link"
