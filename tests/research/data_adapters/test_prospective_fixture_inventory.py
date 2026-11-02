from datetime import datetime, timedelta, timezone
import json

import pytest

from football_spider.models.contracts import Observation
from football_spider.storage.raw_archive import RawArchive
from football_research.data_adapters.prospective_fixture_inventory import (
    MAX_DETAIL_MATCHES,
    MAX_REQUESTS,
    collect_inventory,
    project_future_fixtures,
    project_future_matches,
    project_inventory_windows,
    select_competition_fixtures,
)


def _row(mid, league, match_time, *, home="Home", away="Away"):
    return {
        "matchId": mid,
        "matchTime": match_time,
        "homeId": mid * 10,
        "awayId": mid * 10 + 1,
        "home": home,
        "away": away,
        "league": league,
        "leagueId": 36 if league == "英超" else 999,
        "footMatchState": 0,
        "homeScore": None,
        "awayScore": None,
    }


def _save_list(archive, rows, observed_at):
    payload = {"code": 0, "data": [{"list": rows}]}
    return archive.save(
        "spftll",
        "https://slb.haoyun999.cn/api/Game/GetSimpleMatchsAll",
        {},
        json.dumps(payload, ensure_ascii=False).encode(),
        status=200,
        observed_at=observed_at,
    )


def test_projection_filters_date_and_league_and_preserves_unknown_availability(tmp_path):
    archive = RawArchive(tmp_path)
    observed = datetime(2026, 10, 9, 1, 0, tzinfo=timezone.utc)
    # spftll matchTime is offset-less Asia/Shanghai text.
    obs = _save_list(
        archive,
        [
            _row(1, "英超", "2026-10-09 20:00:00", home="Alpha", away="Beta"),
            _row(2, "德甲", "2026-10-10 20:00:00"),
            _row(3, "英超", "2026-10-08 20:00:00"),
        ],
        observed,
    )
    fixtures, audit = project_future_fixtures(
        archive,
        [obs],
        window_start=observed,
        window_end=datetime(2026, 10, 16, 1, 0, tzinfo=timezone.utc),
    )
    assert [row["source_match_id"] for row in fixtures] == ["1"]
    row = fixtures[0]
    assert row["registered_at"] == observed.isoformat()
    assert row["registered_at"] < row["kickoff_at"]
    assert row["available_at"] is None
    assert row["availability_status"] == "unknown"
    assert row["raw_sha256"] == obs.content_hash
    assert audit["future_premier_league_fixtures"] == 1


def test_projection_uses_first_observation_for_registration(tmp_path):
    archive = RawArchive(tmp_path)
    first_time = datetime(2026, 10, 9, 0, 0, tzinfo=timezone.utc)
    later_time = datetime(2026, 10, 9, 0, 30, tzinfo=timezone.utc)
    row = _row(7, "Premier League", "2026-10-12 20:00:00")
    first = _save_list(archive, [row], first_time)
    later = _save_list(archive, [row], later_time)
    fixtures, _ = project_future_fixtures(
        archive,
        [later, first],
        window_start=first_time,
        window_end=datetime(2026, 10, 16, 0, 0, tzinfo=timezone.utc),
    )
    assert fixtures[0]["registration_observation_id"] == first.id
    assert fixtures[0]["registered_at"] == first_time.isoformat()


def test_budget_constants_are_bounded():
    assert MAX_REQUESTS <= 6
    assert MAX_DETAIL_MATCHES <= 1


def test_full_english_league_name_and_shared_source_id_bind_to_pl(tmp_path):
    archive = RawArchive(tmp_path)
    observed = datetime(2026, 10, 9, 0, 0, tzinfo=timezone.utc)
    first = _row(1, "英格兰超级联赛", "2026-10-11 20:00:00")
    other = _row(2, "source alternate name", "2026-10-12 20:00:00")
    first["leagueId"] = other["leagueId"] = 36
    obs = _save_list(archive, [first, other], observed)
    fixtures, _ = project_future_fixtures(archive, [obs], window_start=observed,
                                         window_end=observed + timedelta(days=21))
    assert [row["source_match_id"] for row in fixtures] == ["1", "2"]
    assert {row["competition_key"] for row in fixtures} == {"Premier League"}
    assert fixtures[1]["competition"] == "source alternate name"


def _candidate(mid, league, *, complete=True, odds=False, conflict=False):
    return {"physical_match_id": f"spftll:match:{mid}", "source_match_id": str(mid),
            "competition": league, "competition_key": league,
            "competition_source_id": league, "home_team": "A" if complete else "", "away_team": "B",
            "kickoff_at": f"2026-10-{10 + mid:02}T10:00:00+00:00",
            "inventory_quality": {"complete_1x2_observed": odds, "schedule_conflict_observed": conflict}}


def test_single_fallback_does_not_mix_pl_but_retains_its_candidates():
    rows = [_candidate(1, "Premier League"), _candidate(2, "Premier League")]
    rows += [_candidate(mid, "League B") for mid in range(3, 7)]
    rows += [_candidate(mid, "League C") for mid in range(7, 10)]
    selected, report = select_competition_fixtures(rows)
    assert {row["competition_key"] for row in selected} == {"League B"}
    assert report["premier_league_candidates_retained"] == ["spftll:match:1", "spftll:match:2"]
    assert report["selected_competition"] == "League B"
    assert report["selection_uses_outcomes"] is False


def test_fallback_tie_uses_measured_odds_coverage_without_claiming_stability():
    rows = [_candidate(1, "League A"), _candidate(2, "League B", odds=True)]
    selected, report = select_competition_fixtures(rows)
    assert selected[0]["competition_key"] == "League B"
    assert report["fallback_scores"][0]["complete_1x2_observed"] == 1


def test_six_pl_fixtures_take_priority_over_larger_fallback():
    rows = [_candidate(mid, "Premier League") for mid in range(1, 7)]
    rows += [_candidate(mid, "League B") for mid in range(7, 15)]
    selected, report = select_competition_fixtures(rows)
    assert len(selected) == 6
    assert report["selected_competition"] == "Premier League"
    assert report["fallback_reason"] is None


def test_21_day_shortage_expands_same_raw_to_42_days_without_network(tmp_path):
    archive = RawArchive(tmp_path)
    observed = datetime(2026, 10, 9, 0, 0, tzinfo=timezone.utc)
    rows = [_row(1, "英格兰超级联赛", "2026-10-11 20:00:00")]
    rows += [_row(mid, "英格兰超级联赛", f"2026-11-{mid:02} 20:00:00") for mid in range(2, 7)]
    obs = _save_list(archive, rows, observed)
    selected, candidates, report = project_inventory_windows(archive, [obs], window_start=observed)
    assert report["expanded_to_maximum"] is True
    assert len(candidates) == len(selected) == 6
    assert report["primary_projection"]["future_fixture_count"] == 1
    assert report["window_end"] == (observed + timedelta(days=42)).isoformat()
    assert report["window_extension_basis"] == "filter_same_current_list_response_not_date_lookup"


def test_three_fixture_primary_sample_does_not_force_extension(tmp_path):
    archive = RawArchive(tmp_path)
    observed = datetime(2026, 10, 9, 0, 0, tzinfo=timezone.utc)
    obs = _save_list(archive, [_row(mid, "德甲", f"2026-10-{10 + mid} 20:00:00") for mid in range(1, 4)], observed)
    selected, _, report = project_inventory_windows(archive, [obs], window_start=observed)
    assert len(selected) == 3
    assert report["expanded_to_maximum"] is False
    assert report["window_end"] == (observed + timedelta(days=21)).isoformat()


def test_inventory_freezes_plan_and_uses_existing_collector_once_for_empty_window(tmp_path, monkeypatch):
    from football_spider.models.contracts import CollectionResult
    import football_research.data_adapters.prospective_fixture_inventory as module
    observed = datetime(2026, 10, 9, 0, 0, tzinfo=timezone.utc)
    calls = []
    def fake_collect(config, *, start, end, match_ids=None):
        plan = json.loads((config.data_dir / "selection-plan.json").read_text(encoding="utf-8"))
        assert plan["frozen_at"] == observed.isoformat()
        calls.append((start, end, match_ids))
        obs = _save_list(RawArchive(config.data_dir), [], observed)
        return CollectionResult(source="spftll", observations=[obs.id], requests=1)
    monkeypatch.setattr(module, "_run_collection", fake_collect)
    report = collect_inventory(tmp_path / "new-run", now=observed)
    assert len(calls) == 1
    assert report["window"]["expanded_to_maximum"] is True
    assert report["fixtures_registered"] == 0
    assert report["request_budget"]["actual_archived_attempts"] == 1
    with pytest.raises(FileExistsError):
        collect_inventory(tmp_path / "new-run", now=observed)


def test_maximum_window_cannot_exceed_42_days(tmp_path):
    with pytest.raises(ValueError):
        project_inventory_windows(RawArchive(tmp_path), [], window_start=datetime.now(timezone.utc), max_horizon_days=43)
