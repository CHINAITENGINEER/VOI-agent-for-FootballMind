"""Pure offline live-acquisition contract tests using synthetic source replies.

No network, production RawArchive, repository or real fixture is used here.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import hashlib
import json
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from football_platform.information.live import LiveEvidenceEnvironment
from football_spider.models.contracts import Observation


MID = "spftll:match:12345"
CUTOFF = "2025-10-10T12:00:00+00:00"
REFRESHED = "2025-10-10T12:05:00+00:00"
FINISHED = "2025-10-10T12:06:00+00:00"
EARLIER = "2025-10-10T11:50:00+00:00"
EARLIER_FINISH = "2025-10-10T11:51:00+00:00"


def fixture():
    return {
        "match_id": MID, "competition_id": "spftll:competition:1", "season": None,
        "home_team_id": "spftll:team:11", "away_team_id": "spftll:team:22",
        "home_team_name": "Synthetic Home", "away_team_name": "Synthetic Away",
        "kickoff": "2025-10-11T12:00:00+00:00",
    }


def source_match(mid="12345", *, home_id=11, away_id=22, finished=False,
                 home_name="Synthetic Home", away_name="Synthetic Away"):
    return {
        "matchId": mid, "homeId": home_id, "awayId": away_id,
        "homeChs": home_name, "awayChs": away_name, "leagueId": 1, "leagueChs": "英超",
        "matchTime": "2025-10-01T12:00:00+00:00" if finished else "2025-10-11T12:00:00+00:00",
        "finishedTitle": "完场" if finished else "未开", "footMatchState": -1 if finished else 0,
        "homeScore": 2 if finished else None, "awayScore": 1 if finished else None,
        "homeHalfScore": 1 if finished else None, "awayHalfScore": 0 if finished else None,
    }


def detail_payload():
    row = source_match()
    row["homeBattleList"] = [source_match("9001", home_id=11, away_id=33, finished=True)]
    row["awayBattleList"] = [source_match("9002", home_id=44, away_id=22, finished=True)]
    row["private_target_label"] = "TARGET_LABEL_CANARY"
    return {"code": 0, "data": row}


def odds_payload():
    return {"code": 0, "data": {"footballMoreSpInfo": {
        "matchId": "12345", "spf_win": 2.0, "spf_draw": 3.5, "spf_lost": 4.0,
    }}}


class MemoryArchive:
    def __init__(self):
        self.bodies = {}
        self.read_ids = []

    def put(self, payload, *, endpoint="GetInfo", observed_at=REFRESHED,
            http_status=200, error_class=None, source="spftll"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        obs = Observation(
            id=str(uuid4()), source=source, endpoint=f"https://synthetic.invalid/{endpoint}",
            params={"matchId": "12345"}, observed_at=datetime.fromisoformat(observed_at),
            http_status=http_status, error_class=error_class,
            payload_path="synthetic-private-do-not-expose.gz",
            payload_format="json" if not isinstance(payload, bytes) else "bytes",
            content_hash=hashlib.sha256(body).hexdigest(),
            context={"synthetic_test_fixture": True, "requested_source_match_id": "12345"},
        )
        self.bodies[obs.id] = body
        return obs

    def read_body(self, observation):
        self.read_ids.append(observation.id)
        return self.bodies[observation.id]


class FakeRefreshService:
    def __init__(self, archive, reports):
        self.archive = archive
        self.config = SimpleNamespace(database_url=None)
        self.reports = deepcopy(reports)
        self.calls = []

    def refresh(self, match_id, **kwargs):
        self.calls.append({"match_id": match_id, **kwargs})
        if not self.reports:
            raise AssertionError("Unexpected extra external request")
        return deepcopy(self.reports.pop(0))


def report(archive, payload=None, *, endpoint="GetInfo", observed_at=REFRESHED,
           finished_at=FINISHED, http_status=200, error_class=None,
           status="NEEDS_DATABASE", network_status="OK", processing=None,
           cached=False, requests=1, elapsed_seconds=2.5):
    obs = archive.put(detail_payload() if payload is None else payload, endpoint=endpoint,
                      observed_at=observed_at, http_status=http_status, error_class=error_class)
    return {
        "status": status, "network_status": network_status, "match_id": "spftll:12345",
        "observations": [obs.model_dump(mode="json")],
        "processing": [] if processing is None else processing,
        "finished_at": finished_at, "elapsed_seconds": elapsed_seconds,
        "requests_this_call": requests, "cached": cached,
        "errors": [] if error_class is None else [{"category": error_class}],
        "database": {"status": "NEEDS_DATABASE"},
        "cost": None, "cost_status": "UNKNOWN", "provider_credits": None,
    }


def environment(reports=None, *, archive=None, max_requests=2, timeout=30):
    archive = archive or MemoryArchive()
    service = FakeRefreshService(archive, reports if reports is not None else [report(archive)])
    env = LiveEvidenceEnvironment(service, fixture=fixture(),
                                  fixture_observed_at="2025-10-10T10:00:00+00:00",
                                  max_requests=max_requests, timeout=timeout)
    env.start(CUTOFF)
    return env, service


def assert_zero(cost):
    assert all(getattr(cost, part).amount == 0 for part in ("money", "requests", "latency_ms", "tokens"))


def test_catalog_binds_fixture_and_history_to_one_response():
    env, _ = environment()
    specs = {item.action_id: item for item in env.catalog()}
    assert specs["fixture_history"].group_ids == ["fixture_refresh", "history"]
    assert specs["fixture_history"].dependency_key == "GetInfo"
    assert specs["odds"].dependency_key == "GetMoreSpInfo"
    assert specs["fixture_history"].expected_cost.requests.amount == 1
    assert specs["fixture_history"].expected_cost.money.amount is None
    assert specs["fixture_history"].expected_cost.money.basis == "unknown"


def test_refresh_is_bounded_and_does_not_reveal_new_evidence_at_old_cutoff():
    env, service = environment(max_requests=2, timeout=30)
    initial = env.visible_state()
    result = env.acquire("fixture_history", "first")
    assert len(service.calls) == 1
    call = service.calls[0]
    assert call["match_id"] == MID
    assert call["topics"] == ("fixture", "history")
    assert call["max_requests"] == 1
    assert call["timeout"] == 30
    UUID(call["request_id"])
    assert result.status == "NEEDS_NEW_CUTOFF"
    assert result.revealed_groups == {}
    assert result.suggested_new_cutoff.isoformat() == FINISHED
    assert result.resulting_cutoff == initial.prediction_cutoff
    assert len(result.evidence_refs) == 1
    assert result.evidence_refs[0].startswith("observation:")
    assert result.charged_cost.requests.amount == 1
    assert result.charged_cost.money.amount is None
    assert env.visible_state().groups == {}
    assert all(value == "UNACQUIRED" for value in env.visible_state().mask.values())
    assert initial.groups == {} and initial.history == []


def test_explicit_advance_reveals_bound_groups_but_never_target_score():
    env, _ = environment()
    initial = env.visible_state()
    env.acquire("fixture_history", "first")
    updated = env.advance_cutoff(FINISHED)
    assert set(updated.groups) == {"fixture_refresh", "history"}
    assert updated.mask["fixture_refresh"] == updated.mask["history"] == "OBSERVED"
    assert updated.groups["history"].values["home_matches"] == 1
    assert updated.groups["history"].values["home_goals_for"] == 2
    assert updated.groups["history"].values["away_goals_for"] == 1
    assert updated.groups["history"].quality == "PARTIAL"
    assert updated.groups["fixture_refresh"].times.ingested_at is None
    assert updated.groups["fixture_refresh"].times.operational_available_at.isoformat() == FINISHED
    encoded = updated.model_dump_json()
    assert "CANARY" not in encoded
    assert "private-do-not-expose" not in encoded
    assert "homeScore" not in encoded and "awayScore" not in encoded
    assert "observation:" in encoded
    assert initial.prediction_cutoff.isoformat() == CUTOFF
    assert initial.groups == {}


def test_advance_before_availability_keeps_new_groups_hidden():
    env, _ = environment()
    env.acquire("fixture_history", "first")
    middle = env.advance_cutoff("2025-10-10T12:04:00+00:00")
    assert middle.groups == {}
    assert all(value == "UNACQUIRED" for value in middle.mask.values())
    assert set(env.advance_cutoff(FINISHED).groups) == {"fixture_refresh", "history"}


def test_existing_observation_before_cutoff_can_be_revealed_immediately():
    archive = MemoryArchive()
    saved = report(archive, observed_at=EARLIER, finished_at=EARLIER_FINISH)
    env, _ = environment([saved], archive=archive)
    result = env.acquire("fixture_history", "already-available")
    assert result.status == "ACQUIRED"
    assert set(result.revealed_groups) == {"fixture_refresh", "history"}
    assert result.revealed_groups["history"].times.feature_available_at <= env.visible_state().prediction_cutoff


def test_same_request_is_free_no_new_latency_and_cannot_switch_action():
    env, service = environment()
    first = env.acquire("fixture_history", "stable")
    saved = env.visible_state()
    repeated = env.acquire("fixture_history", "stable")
    assert repeated.repeated_request is True
    assert repeated.evidence_refs == first.evidence_refs
    assert repeated.attempts == 0
    assert_zero(repeated.charged_cost)
    assert len(service.calls) == 1
    assert env.visible_state() == saved
    with pytest.raises(ValueError, match="another action"):
        env.acquire("odds", "stable")
    assert len(service.calls) == 1


def test_domain_cached_report_does_not_charge_historical_elapsed_seconds(monkeypatch):
    import football_platform.information.live as live_module
    clock = iter((10.0, 10.125))
    monkeypatch.setattr(live_module.time, "perf_counter", lambda: next(clock))
    archive = MemoryArchive()
    cached = report(archive, cached=True, requests=0, elapsed_seconds=25,
                    observed_at=EARLIER, finished_at=EARLIER_FINISH)
    env, _ = environment([cached], archive=archive)
    result = env.acquire("fixture_history", "restored-domain-operation")
    assert result.charged_cost.requests.amount == 0
    # Reused provider work is not recharged. Actual local cache-read latency is
    # a distinct measured cost: 125 ms here, never the original 25-second call.
    assert result.charged_cost.latency_ms.amount == 125
    assert env.visible_state().budget.spent["latency_ms"] == 125
    assert result.revealed_groups["fixture_refresh"].cache_reused is True


def test_request_budget_accounts_binding_once_and_prevents_third_call():
    archive = MemoryArchive()
    env, service = environment([report(archive), report(archive, odds_payload(), endpoint="GetMoreSpInfo")], archive=archive)
    first = env.acquire("fixture_history", "first")
    second = env.acquire("odds", "second")
    blocked = env.acquire("fixture_history", "third")
    assert first.charged_cost.requests.amount == second.charged_cost.requests.amount == 1
    assert blocked.status == "BUDGET_EXCEEDED"
    assert blocked.attempts == 0
    assert_zero(blocked.charged_cost)
    assert len(service.calls) == 2
    assert env.visible_state().budget.spent["requests"] == 2


def test_single_request_budget_leaves_other_action_unacquired():
    env, service = environment(max_requests=1)
    env.acquire("fixture_history", "first")
    assert env.acquire("odds", "second").status == "BUDGET_EXCEEDED"
    assert len(service.calls) == 1
    assert env.visible_state().mask["odds"] == "UNACQUIRED"


def test_empty_data_is_visible_and_cost_is_retained():
    archive = MemoryArchive()
    empty = report(archive, {"code": 0, "data": None}, status="NO_DATA", network_status="NO_DATA")
    env, _ = environment([empty], archive=archive)
    result = env.acquire("fixture_history", "empty")
    assert result.status == "NO_DATA"
    assert result.revealed_groups == {}
    assert result.mask_updates == {"fixture_refresh": "SOURCE_MISSING", "history": "SOURCE_MISSING"}
    assert result.charged_cost.requests.amount == 1
    assert result.evidence_refs
    assert env.visible_state().budget.spent["requests"] == 1


def test_timeout_is_classified_and_charged_once():
    archive = MemoryArchive()
    timeout = report(archive, b"", http_status=None, error_class="TIMEOUT", status="FAILED", network_status="FAILED")
    env, service = environment([timeout], archive=archive)
    result = env.acquire("fixture_history", "timeout")
    assert result.status == "TIMEOUT"
    assert result.charged_cost.requests.amount == 1
    assert result.evidence_refs
    assert result.mask_updates == {"fixture_refresh": "FAILED", "history": "FAILED"}
    assert_zero(env.acquire("fixture_history", "timeout").charged_cost)
    assert len(service.calls) == 1


@pytest.mark.parametrize("payload", [b"<html>bad synthetic response</html>", {"code": 5, "data": {}}])
def test_bad_response_preserves_cost_and_audit_refs(payload):
    archive = MemoryArchive()
    env, _ = environment([report(archive, payload)], archive=archive)
    result = env.acquire("fixture_history", "bad")
    assert result.status == "BAD_DATA"
    assert not result.revealed_groups
    assert result.charged_cost.requests.amount == 1
    assert result.evidence_refs
    assert env.visible_state().mask["fixture_refresh"] == "FAILED"


def test_target_quality_quarantine_is_not_revealed_on_advance():
    archive = MemoryArchive()
    payload = detail_payload()
    payload["data"].update(homeScore=0, awayScore=0, homeHalfScore=1)
    env, _ = environment([report(archive, payload)], archive=archive)
    result = env.acquire("fixture_history", "quarantined")
    assert result.status == "BAD_DATA"
    assert not env.advance_cutoff(FINISHED).groups


@pytest.mark.parametrize("failure", ["unfinished_report", "processing_failed", "processing_quarantined"])
def test_uncommitted_or_failed_processing_is_not_revealed(failure):
    archive = MemoryArchive()
    saved = report(archive)
    if failure == "unfinished_report":
        saved["status"] = "RUNNING"
    else:
        saved["status"] = "PARTIAL"
        saved["processing"] = [{"observation_id": saved["observations"][0]["id"],
                                 "status": "FAILED" if failure == "processing_failed" else "QUARANTINED"}]
    env, _ = environment([saved], archive=archive)
    result = env.acquire("fixture_history", "not-committed")
    assert result.status in {"FAILED", "BAD_DATA"}
    assert not result.revealed_groups
    assert not env.advance_cutoff(FINISHED).groups
    assert result.charged_cost.requests.amount == 1


def test_wrong_root_identity_cannot_be_rescued_by_target_in_history():
    archive = MemoryArchive()
    payload = detail_payload()
    target = deepcopy(payload["data"])
    payload["data"] = source_match("67890", home_id=55, away_id=66)
    payload["data"]["battleList"] = [target]
    saved = report(archive, payload, status="PARTIAL", network_status="PARTIAL")
    saved["errors"] = [{"category": "TARGET_ID_MISMATCH"}]
    env, _ = environment([saved], archive=archive)
    result = env.acquire("fixture_history", "wrong-root")
    assert result.status in {"BAD_DATA", "FAILED"}
    assert result.revealed_groups == {}
    assert env.advance_cutoff(FINISHED).groups == {}


def test_report_request_identity_must_match_live_fixture():
    archive = MemoryArchive()
    saved = report(archive)
    saved["match_id"] = "spftll:98765"
    env, _ = environment([saved], archive=archive)
    result = env.acquire("fixture_history", "wrong-report")
    assert result.status in {"BAD_DATA", "FAILED"}
    assert env.advance_cutoff(FINISHED).groups == {}


def test_current_odds_are_never_backdated_to_old_cutoff():
    archive = MemoryArchive()
    saved = report(archive, odds_payload(), endpoint="GetMoreSpInfo")
    env, service = environment([saved], archive=archive)
    result = env.acquire("odds", "current-quote")
    assert service.calls[0]["topics"] == ("odds",)
    assert result.status == "NEEDS_NEW_CUTOFF"
    assert "odds" not in env.visible_state().groups
    advanced = env.advance_cutoff(FINISHED)
    quote = advanced.groups["odds"]
    assert quote.values["home"] == 2.0
    assert quote.values["bookmaker"] is None
    assert quote.values["quote_time_semantics"] == "current_observation_only"
    assert quote.times.observed_at.isoformat() == REFRESHED
    assert quote.times.feature_available_at.isoformat() == FINISHED


def test_wrong_odds_target_and_incomplete_market_are_bad_data():
    archive = MemoryArchive()
    wrong = odds_payload()
    wrong["data"]["footballMoreSpInfo"]["matchId"] = "67890"
    incomplete = odds_payload()
    incomplete["data"]["footballMoreSpInfo"].pop("spf_draw")
    env, _ = environment([report(archive, wrong, endpoint="GetMoreSpInfo"),
                           report(archive, incomplete, endpoint="GetMoreSpInfo")], archive=archive)
    assert env.acquire("odds", "wrong-target").status == "BAD_DATA"
    assert env.acquire("odds", "incomplete").status == "BAD_DATA"
    assert env.visible_state().budget.spent["requests"] == 2
    assert not env.advance_cutoff(FINISHED).groups


def test_target_result_never_enters_history_aggregate():
    archive = MemoryArchive()
    payload = detail_payload()
    duplicate = source_match()
    duplicate.update(homeScore=99, awayScore=88)
    payload["data"]["battleList"] = [duplicate]
    env, _ = environment([report(archive, payload)], archive=archive)
    env.acquire("fixture_history", "history-filter")
    state = env.advance_cutoff(FINISHED)
    values = state.groups["history"].values
    assert values["home_matches"] == values["away_matches"] == 1
    assert values["home_goals_for"] == 2
    assert values["away_goals_for"] == 1
    assert all("12345" not in ref for ref in state.groups["history"].evidence_refs if ref.startswith("source_match:"))


def test_cup_and_inconsistent_halves_are_excluded_from_history():
    archive = MemoryArchive()
    payload = detail_payload()
    bad_half = source_match("8001", home_id=11, away_id=33, finished=True)
    bad_half["homeHalfScore"] = 9
    cup = source_match("8002", home_id=11, away_id=33, finished=True)
    cup["leagueChs"] = "Synthetic Cup"
    payload["data"]["homeBattleList"].extend([bad_half, cup])
    env, _ = environment([report(archive, payload)], archive=archive)
    env.acquire("fixture_history", "filter")
    values = env.advance_cutoff(FINISHED).groups["history"].values
    assert values["home_matches"] == 1
    assert values["home_goals_for"] == 2


def test_returned_acquisition_objects_are_detached_from_internal_state():
    archive = MemoryArchive()
    env, _ = environment([report(archive, observed_at=EARLIER, finished_at=EARLIER_FINISH)], archive=archive)
    result = env.acquire("fixture_history", "detached")
    result.revealed_groups["history"].values["home_goals_for"] = 999
    result.revealed_groups["fixture_refresh"].times.feature_available_at = datetime.fromisoformat("2099-01-01T00:00:00+00:00")
    result.charged_cost.requests.amount = 99
    state = env.visible_state()
    assert state.groups["history"].values["home_goals_for"] == 2
    assert state.groups["fixture_refresh"].times.feature_available_at <= state.prediction_cutoff
    assert state.history[0].charged_cost.requests.amount == 1
    assert state.budget.spent["requests"] == 1


@pytest.mark.parametrize("cutoff", [CUTOFF, "2025-10-10T11:00:00+00:00", "2025-10-11T12:00:00+00:00"])
def test_invalid_advance_is_rejected_without_state_changes(cutoff):
    env, _ = environment()
    before = env.visible_state()
    with pytest.raises(ValueError):
        env.advance_cutoff(cutoff)
    assert env.visible_state() == before


@pytest.mark.parametrize("limits", [{"max_requests": 0}, {"max_requests": 3}, {"max_requests": True}, {"timeout": 0}, {"timeout": 61}])
def test_invalid_external_bounds_fail_before_requests(limits):
    with pytest.raises(ValueError):
        environment(**limits)


def test_second_start_cannot_reset_live_budget():
    env, service = environment()
    env.acquire("fixture_history", "first")
    before = env.visible_state()
    with pytest.raises(ValueError, match="One bounded live episode"):
        env.start(CUTOFF)
    assert env.visible_state() == before
    assert len(service.calls) == 1


def test_invalid_request_or_unknown_action_does_not_call_service():
    env, service = environment()
    before = env.visible_state()
    for invalid in ("", "x" * 201, None, 123):
        with pytest.raises(ValueError):
            env.acquire("fixture_history", invalid)
    with pytest.raises(ValueError):
        env.acquire("weather", "unknown-capability")
    assert env.visible_state() == before
    assert service.calls == []


def test_archive_read_failure_is_observable_and_preserves_spent_cost(monkeypatch):
    env, service = environment()
    def unreadable(_observation):
        raise OSError("synthetic archived payload temporarily unreadable")
    monkeypatch.setattr(service.archive, "read_body", unreadable)
    result = env.acquire("fixture_history", "archive-failure")
    assert result.status in {"FAILED", "BAD_DATA"}
    assert result.charged_cost.requests.amount == 1
    assert result.evidence_refs
    assert env.visible_state().budget.spent["requests"] == 1
    assert env.visible_state().mask["fixture_refresh"] == "FAILED"
    repeated = env.acquire("fixture_history", "archive-failure")
    assert repeated.repeated_request is True
    assert_zero(repeated.charged_cost)
    assert len(service.calls) == 1


def test_older_cached_projection_does_not_replace_newer_pending_evidence():
    archive = MemoryArchive()
    newer_payload = detail_payload()
    newer_payload["data"]["homeChs"] = "Synthetic Home Revised Name"
    newer = report(archive, newer_payload)
    older = report(archive, detail_payload(), observed_at=EARLIER,
                    finished_at=EARLIER_FINISH, cached=True, requests=0)
    env, _ = environment([newer, older], archive=archive)
    assert env.acquire("fixture_history", "newer").status == "NEEDS_NEW_CUTOFF"
    older_result = env.acquire("fixture_history", "older-cached")
    assert older_result.revealed_groups["fixture_refresh"].values["home_team_name"] == "Synthetic Home"
    advanced = env.advance_cutoff(FINISHED)
    assert advanced.groups["fixture_refresh"].values["home_team_name"] == "Synthetic Home Revised Name"
    assert advanced.groups["fixture_refresh"].times.observed_at.isoformat() == REFRESHED
