"""Raw-backed training gate integration with constructed, offline-only fixtures.

The source bytes, fixture identities, clocks and scores below are simulated
test inputs.  They test the complete real-data gateway contract, and must never
be exported as real observational samples or evidence of predictive benefit.
"""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from html import escape
import json
from pathlib import Path
import shutil
import socket

import pytest

from football_agent.learned_voi_replay import replay_saved_run
from football_research.acquisition.text_history import load_text_model
from football_research.data_adapters.prospective_learning import (
    import_training_exposure, prepare_learning_plan,
)
from football_research.data_adapters.prospective_text import _save, select_initializations
from football_research.data_adapters.prospective_training import frozen_f_from_runs
from football_research.data_adapters.spftll import load_spftll_archive
from football_research.data_adapters.verified_fixtures import verify_fixture_identity
from football_spider.storage.raw_archive import RawArchive
from test_cohort_runtime import _registered
from test_verified_collection import verified_inputs, _collect


def _forbidden(*args, **kwargs):
    pytest.fail("Training integration fixtures and verification are strictly offline")


def _deny_network(mp):
    mp.setattr(socket, "create_connection", _forbidden)
    mp.setattr(socket, "getaddrinfo", _forbidden)
    mp.setattr(socket.socket, "connect", _forbidden)
    mp.setattr(socket.socket, "connect_ex", _forbidden)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    _deny_network(monkeypatch)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _source_inputs(root, *, at, local_id, event_id, kickoff=None):
    """Create independent quote and official identity bytes, then use gateways."""
    kickoff = kickoff or at+timedelta(hours=3)
    prior = RawArchive(root/"market")
    source = {"matchId": local_id, "leagueId": 1864, "leagueChs": "欧国联",
        "homeId": 11, "awayId": 12, "homeChs": "克罗地亚", "awayChs": "西班牙",
        "footMatchState": 0,
        "matchTime": kickoff.astimezone(timezone(timedelta(hours=8))).replace(tzinfo=None).isoformat(),
        "spfWinFoot": 2., "spfEqualFoot": 3., "spfLoseFoot": 4.}
    prior.save("spftll", "https://slb.haoyun999.cn/api/Game/GetSimpleMatchsAll", {},
        json.dumps({"code": 0, "data": [{"list": [source]}]}).encode(),
        status=200, observed_at=at-timedelta(seconds=20))
    plan = select_initializations(load_spftll_archive(prior.root/"raw", as_of=at),
                                  as_of=at-timedelta(seconds=10), limit=1)
    url = f"https://www.uefa.com/uefanationsleague/match/{event_id}--croatia-vs-spain/"
    event = {"@type": "SportsEvent", "name": "Croatia vs Spain", "url": url,
        "description": "UEFA Nations League 2027 League phase: Croatia vs Spain",
        "startDate": kickoff.isoformat(), "eventStatus": "EventScheduled"}
    native = {"id": str(event_id), "seasonYear": 2027, "status": "Upcoming",
        "competition": {"age": "Adult", "sex": "Male", "sportsType": "Football",
            "teamCategory": "National", "displayName": "UEFA Nations League", "id": "2014"},
        "round": {"displayName": "League phase", "mode": "Group"},
        "kickOffTime": {"dateTime": kickoff.isoformat()}}
    for side, ident, name in (("home", "101", "Croatia"), ("away", "102", "Spain")):
        event[side+"Team"] = {"@type": "SportsTeam", "name": name,
            "sameAs": f"https://www.uefa.com/uefanationsleague/teams/{ident}--{name.lower()}/"}
        native[side+"Team"] = {"id": ident, "internationalName": name}
    article = {"@type": "LiveBlogPosting", "mainEntity": event,
        "liveBlogUpdate": [{"@type": "BlogPosting", "datePublished": (at-timedelta(minutes=3)).isoformat(),
            "articleBody": "Croatia may rotate their starting lineup against Spain. "
                "Spain will assess player fitness before the match."}]}
    body = ('<html><script type="application/ld+json">'+json.dumps(article)+
            '</script><div data-options="'+escape(json.dumps(native), quote=True)+'"></div></html>').encode()
    identity = RawArchive(root/"identity")
    observed = identity.save("open_page", url, {}, body, status=200, observed_at=at-timedelta(seconds=5))
    binding = verify_fixture_identity(plan["selected"][0], archive_root=identity.root,
        observation_ids=[observed.id], aliases={"home": ["克罗地亚", "Croatia"], "away": ["西班牙", "Spain"]},
        as_of=at)
    assert binding["status"] == "CONFIRMED"
    _save(root/"plan.json", plan)
    _save(root/"bindings.json", {"bindings": {binding["match_id"]: binding}})
    return {"root": root, "plan": plan, "binding": binding, "body": body, "now": at, "native": native}


def _score(archive, inputs, outcome):
    value = deepcopy(inputs["native"])
    kickoff = datetime.fromisoformat(value["kickOffTime"]["dateTime"])
    value.update(status="FINISHED", fullTimeAt=(kickoff+timedelta(minutes=96)).isoformat(),
        competition={"id": "2014", "code": "UNL", "age": "ADULT", "sex": "MALE",
                     "sportsType": "FOOTBALL", "teamCategory": "NATIONAL"},
        round={"mode": "GROUP", "competitionId": "2014", "seasonYear": "2027",
               "metaData": {"name": "League phase", "type": "GROUP_STANDINGS"}},
        group={"competitionId": "2014", "seasonYear": "2027", "teams": ["101", "102"]},
        competitionPhase="TOURNAMENT", type="GROUP_STAGE")
    home, away = ((3, 0), (1, 1), (0, 2))[outcome]
    value["score"] = {"regular": {"home": home, "away": away}, "total": {"home": home, "away": away}}
    observed = kickoff+timedelta(hours=2)
    archive.save("open_page", "https://match.uefa.com/v5/matches/"+value["id"], {},
        json.dumps(value).encode(), status=200, observed_at=observed,
        context={"requested_at": (observed-timedelta(seconds=1)).isoformat()})


@pytest.fixture(scope="module")
def archived_training_cohort(tmp_path_factory):
    """13 archived runs, 13 provider IDs, 12 official fixtures and all classes."""
    import football_agent.inquiry_tools as web_tools
    import football_agent.learned_voi_runtime as runtime_module
    import football_research.data_adapters.prospective_text as collector_module
    import test_verified_collection as fixture_module
    root = tmp_path_factory.mktemp("constructed-raw-training-contract")
    now = datetime.now(timezone.utc).replace(microsecond=0)
    base = now-timedelta(days=40)
    clock = [base+timedelta(milliseconds=100)]
    class FixtureClock(datetime):
        @classmethod
        def now(cls, tz=None):
            clock[0] += timedelta(milliseconds=10)
            return clock[0].astimezone(tz) if tz is not None else clock[0].replace(tzinfo=None)
    runs, single, alias = [], None, None
    results = RawArchive(root/"independent-results")
    with pytest.MonkeyPatch.context() as mp:
        _deny_network(mp)
        for module in (web_tools, runtime_module, collector_module, fixture_module):
            mp.setattr(module, "datetime", FixtureClock)
        for index in range(12):
            at = base+timedelta(days=index)
            clock[0] = at+timedelta(milliseconds=100)
            inputs = _source_inputs(root/f"source-{index}", at=at, local_id=1000+index, event_id=2047000+index)
            _, calls, collected = _collect(inputs, mp)
            folder = Path(collected["runs"][0]["folder"])
            assert len(calls) == 2
            runs.append(folder)
            _score(results, inputs, index % 3)
            if index == 0:
                single = folder
                clock[0] = at+timedelta(minutes=1, milliseconds=100)
                alias_inputs = _source_inputs(root/"provider-alias", at=at+timedelta(minutes=1),
                    local_id=2000, event_id=2047000, kickoff=at+timedelta(hours=3))
                _, _, alias_collected = _collect(alias_inputs, mp)
                alias = Path(alias_collected["runs"][0]["folder"])
    runs.append(alias)
    registered = now-timedelta(days=2)
    exposure = import_training_exposure(runs, created_at=registered)
    assert exposure["unique_matches"] == 12 and len(exposure["runs"]) == 13
    return {"root": root, "runs": runs, "single": single, "alias": alias,
            "result_archive": results.root, "exposure": exposure, "as_of": now-timedelta(days=1)}


def _train(data, output, *, runs=None, previous=None, kind="ordered-v1"):
    return frozen_f_from_runs(data["runs"] if runs is None else runs,
        [data["exposure"]] if previous is None else previous, data["result_archive"], output,
        as_of=data["as_of"], model_kind=kind)


def test_same_official_event_multiple_provider_runs_cannot_satisfy_support(archived_training_cohort, tmp_path):
    data = archived_training_cohort
    # Duplicate path plus an independently acquired provider alias remain one match.
    report = _train(data, tmp_path/"not-ready", runs=[data["single"], data["single"], data["alias"]])
    assert report["status"] == "not_ready" and report["qualified_physical_matches"] == 1
    assert report["class_counts"] == {"0": 1, "1": 0, "2": 0}
    assert report["model"] is None and report["production_ready"] is False
    assert not (tmp_path/"not-ready/frozen-text-model.json").exists()


def test_no_arbitrary_records_or_unregistered_matches_can_enter(archived_training_cohort, tmp_path):
    data = archived_training_cohort
    with pytest.raises(TypeError):
        frozen_f_from_runs([], [], data["result_archive"], tmp_path/"raw-record-bypass",
            records=[{"match_id": "invented", "outcome": 0, "evidence_origin": "real_observational"}])
    with pytest.raises(ValueError):
        _train(data, tmp_path/"unregistered", previous=[])
    assert not (tmp_path/"raw-record-bypass/frozen-text-model.json").exists()
    assert not (tmp_path/"unregistered/frozen-text-model.json").exists()


def test_raw_results_missing_cannot_publish_a_model(archived_training_cohort, tmp_path):
    data = archived_training_cohort
    empty_results = RawArchive(tmp_path/"no-result-bytes")
    report = frozen_f_from_runs(data["runs"], [data["exposure"]], empty_results.root, tmp_path/"pending",
                               as_of=data["as_of"])
    assert report["status"] == "not_ready" and report["qualified_physical_matches"] == 0
    assert report["model"] is None and not (tmp_path/"pending/frozen-text-model.json").exists()


def test_twelve_matches_without_all_outcome_classes_still_do_not_publish(archived_training_cohort, tmp_path):
    data = archived_training_cohort
    original, one_class = RawArchive(data["result_archive"]), RawArchive(tmp_path/"one-class-results")
    for obs in original.observations():
        value = json.loads(original.read_body(obs))
        value["score"] = {"regular": {"home": 3, "away": 0}, "total": {"home": 3, "away": 0}}
        one_class.save(obs.source, obs.endpoint, obs.params, json.dumps(value).encode(),
            status=obs.http_status, observed_at=obs.observed_at, context=obs.context)
    report = frozen_f_from_runs(data["runs"], [data["exposure"]], one_class.root, tmp_path/"missing-classes",
                               as_of=data["as_of"])
    assert report["status"] == "not_ready" and report["qualified_physical_matches"] == 12
    assert report["class_counts"] == {"0": 12, "1": 0, "2": 0}
    assert report["model"] is None and not (tmp_path/"missing-classes/frozen-text-model.json").exists()


def test_actual_calibration_cohort_cannot_be_reclassified_as_predictor_training(verified_inputs, monkeypatch, tmp_path):
    path, previous_paths, plan, _, _ = _registered(verified_inputs, "calibration")
    folder, _, _ = _collect(verified_inputs, monkeypatch, learning_plan_path=path,
                             previous_learning_plan_paths=previous_paths)
    previous = [_read(p) for p in previous_paths]+[plan]
    with pytest.raises(ValueError):
        frozen_f_from_runs([folder], previous, RawArchive(tmp_path/"empty-results").root,
                           tmp_path/"wrong-cohort", as_of=datetime.now(timezone.utc))
    assert not (tmp_path/"wrong-cohort/frozen-text-model.json").exists()


def test_missing_earlier_receipt_cannot_shorten_history_of_later_response(archived_training_cohort, tmp_path):
    data = archived_training_cohort
    copied = tmp_path/"incomplete-original"
    shutil.copytree(data["single"], copied)
    run = _read(copied/"run.json")
    first = next(cycle["step"] for cycle in run["cycles"] if cycle.get("selected_action"))
    receipt = copied/f"receipt-{first:03}.json"
    receipt.rename(copied/f"receipt-{first:03}.preserved-but-unavailable")
    report = _train(data, tmp_path/"incomplete-history", runs=[copied])
    assert report["status"] == "not_ready" and report["qualified_records"] == 0
    assert report["qualified_physical_matches"] == 0 and report["model"] is None
    excluded = report["projection_audit"]["complete_chain_excluded_runs"]
    assert len(excluded) == 1 and excluded[0]["reason"] == "incomplete_actual_web_episode_chain"
    assert _read(tmp_path/"incomplete-history/original-records.json") == []


@pytest.mark.parametrize("kind", ["ordered-v1", "neighborhood-v2"])
def test_raw_gate_publishes_loadable_f_without_claiming_effect(archived_training_cohort, verified_inputs,
                                                             tmp_path, monkeypatch, kind):
    data = archived_training_cohort
    output = tmp_path/"fitted"
    report = _train(data, output, kind=kind)
    assert report["status"] == "frozen" and report["qualified_physical_matches"] == 12
    assert report["class_counts"] == {"0": 4, "1": 4, "2": 4}
    assert report["production_ready"] is False
    model_path = Path(report["model"]["path"])
    assert sha256(model_path.read_bytes()).hexdigest() == report["model"]["sha256"]
    model = load_text_model(model_path)
    assert model.metadata["evidence_origin"] == "real_observational"
    assert model.parameters["min_matches"] == 12 and model.parameters["min_class_matches"] >= 1
    assert len(model.metadata["training_match_ids"]) == 12
    assert model.metadata["production_ready"] is False
    plan = prepare_learning_plan(verified_inputs["plan"], identity_bundle={"bindings": {
        verified_inputs["binding"]["match_id"]: verified_inputs["binding"]}}, purpose="calibration",
        created_at=datetime.now(timezone.utc), previous_plans=[data["exposure"]], text_model_path=model_path)
    plan_path, old_plan = verified_inputs["root"]/"learning.json", verified_inputs["root"]/"exposure.json"
    _save(plan_path, plan)
    _save(old_plan, data["exposure"])
    folder, _, _ = _collect(verified_inputs, monkeypatch, learning_plan_path=plan_path,
                             previous_learning_plan_paths=[old_plan])
    config = _read(folder/"configuration.json")
    assert config["text_response_model"]["model_version"] == model.model_version
    assert config["supervisor_metadata"]["learning_cohort"]["purpose"] == "calibration"
    replay = replay_saved_run(folder, output=tmp_path/"replay", project_root=tmp_path)
    assert replay["identical_replay"] and replay["replay_network_requests"] == 0
