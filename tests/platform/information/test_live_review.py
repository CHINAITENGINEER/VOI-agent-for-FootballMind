"""Additional live-episode review cases found after the initial contract suite.

These tests deliberately use the existing synthetic helpers and never make an
HTTP request.  They cover temporal monotonicity and content-change reporting,
which are easy to miss when the RefreshService report is otherwise valid.
"""
from __future__ import annotations

import pytest

from tests.platform.information.test_live import (
    EARLIER,
    EARLIER_FINISH,
    FINISHED,
    REFRESHED,
    detail_payload,
    environment,
    report,
)


def test_older_cached_response_after_cutoff_advance_cannot_regress_visible_state():
    """A late cache hit must not replace a newer pending observation.

    The first response is newer than the initial cutoff, so it is staged until
    an explicit cutoff advance.  Once published, an older cached response is
    still useful for audit but must not move the visible state backwards.
    """
    from tests.platform.information.test_live import MemoryArchive

    archive = MemoryArchive()
    newer_payload = detail_payload()
    newer_payload["data"]["homeChs"] = "Synthetic Home Revised Name"
    newer = report(archive, newer_payload, observed_at=REFRESHED, finished_at=FINISHED)
    older = report(archive, detail_payload(), observed_at=EARLIER,
                   finished_at=EARLIER_FINISH, cached=True, requests=0)
    env, _ = environment([newer, older], archive=archive)

    assert env.acquire("fixture_history", "newer") .status == "NEEDS_NEW_CUTOFF"
    env.advance_cutoff(FINISHED)
    before = env.visible_state()
    late = env.acquire("fixture_history", "older-after-advance")
    after = env.visible_state()

    assert after.groups["fixture_refresh"].values["home_team_name"] == "Synthetic Home Revised Name"
    assert after.groups["fixture_refresh"].version == before.groups["fixture_refresh"].version
    assert late.content_changed is False


def test_refreshed_earlier_kickoff_bounds_explicit_cutoff_advance():
    """The latest observed kickoff, rather than the stale seed fixture, bounds a live episode."""
    from tests.platform.information.test_live import MemoryArchive, FakeRefreshService, fixture
    from football_platform.information.live import LiveEvidenceEnvironment

    archive = MemoryArchive()
    moved = detail_payload()
    moved["data"]["matchTime"] = "2025-10-10T11:00:00+00:00"
    moved_report = report(archive, moved, observed_at="2025-10-10T10:05:00+00:00",
                          finished_at="2025-10-10T10:06:00+00:00")
    service = FakeRefreshService(archive, [moved_report])
    env = LiveEvidenceEnvironment(service, fixture=fixture(),
                                  fixture_observed_at="2025-10-10T09:00:00+00:00")
    env.start("2025-10-10T10:00:00+00:00")
    assert env.acquire("fixture_history", "moved-kickoff").status == "NEEDS_NEW_CUTOFF"

    # The seed fixture says 12:00, while the refreshed target says 11:00.  A
    # post-11:00 cutoff must not be accepted as a prematch live state.
    with pytest.raises(ValueError):
        env.advance_cutoff("2025-10-10T11:30:00+00:00")


def test_same_payload_on_new_request_is_observable_as_unchanged():
    """A second charged refresh with identical content is not a content change."""
    from tests.platform.information.test_live import MemoryArchive

    archive = MemoryArchive()
    first = report(archive, detail_payload(), observed_at=EARLIER,
                   finished_at=EARLIER_FINISH)
    second = report(archive, detail_payload(), observed_at=EARLIER,
                    finished_at=EARLIER_FINISH)
    env, _ = environment([first, second], archive=archive)
    assert env.acquire("fixture_history", "first-content").status == "ACQUIRED"
    result = env.acquire("fixture_history", "second-content")

    assert result.status == "UNCHANGED"
    assert result.content_changed is False
    assert result.charged_cost.requests.amount == 1
    assert result.evidence_refs
