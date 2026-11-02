import pytest
from demo.live_candidate_time import candidate_time_scope, observed_calendar_day

@pytest.mark.parametrize("value,expected",[
    ("24 September 2025","2025-09-24"),("23 oktober 2026","2026-10-23"),
    ("2026-10-24T14:00:00+02:00","2026-10-24"),("2026年10月24日","2026-10-24"),
    ("25 octobre 2026","2026-10-25"),("October 24, 2026","2026-10-24"),
])
def test_observed_multilingual_calendar_metadata(value,expected):
    assert observed_calendar_day(value).isoformat()==expected

def test_unknown_date_not_filled_from_url_or_capture():
    r=candidate_time_scope({"url":"https://club.test/news/2026/10/25/story","source_published_at":None},"2026-10-25T09:00:00Z")
    assert r["status"]=="unknown" and r["calendar_day"] is None

def test_old_news_is_stale_without_fabricating_a_timestamp():
    r=candidate_time_scope({"publication_raw":"24 September 2025"},"2026-10-25T09:00:00Z")
    assert r["status"]=="stale" and r["source_published_at_imputed"] is False

def test_date_only_timezone_boundary_is_conservative():
    assert candidate_time_scope({"publication_raw":"27 October 2026"},"2026-10-25T23:00:00Z")["status"]=="recent"
    assert candidate_time_scope({"publication_raw":"28 October 2026"},"2026-10-25T23:00:00Z")["status"]=="future"

def test_invalid_date_stays_unknown():
    assert observed_calendar_day("31 February 2026") is None
