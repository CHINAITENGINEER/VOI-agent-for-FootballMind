"""Pure HTTP/date/quota policies, independent of unpublished public contracts."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from email.utils import parsedate_to_datetime
import math
from zoneinfo import ZoneInfo


def retry_delay(attempt: int, *, status: int | None, retry_after: str | None = None,
                now: datetime | None = None, max_attempts: int = 3,
                max_inline_wait: float = 60.0, jitter: float = 0.0) -> float | None:
    """One-based attempt. None means stop/defer; never shorten Retry-After.

    A long Retry-After is returned to the caller as a deferred failure, not an
    instruction to send early. 401/403 require credentials/operator inspection,
    not blind retries; transient 408/429/5xx and transport failures are bounded.
    """
    if attempt < 1 or max_attempts < 1 or max_inline_wait < 0 or not 0 <= jitter <= 1:
        raise ValueError('Invalid retry policy')
    if attempt >= max_attempts or (status is not None and status not in (408, 429, 500, 502, 503, 504)):
        return None
    delay = min(2 ** min(attempt - 1, 10), 30) + jitter
    if retry_after:
        try:
            value = float(retry_after)
            if not math.isfinite(value) or value < 0:
                raise ValueError('Invalid Retry-After seconds')
        except ValueError:
            try:
                parsed = parsedate_to_datetime(retry_after)
                if parsed.tzinfo is None:
                    raise ValueError('Missing Retry-After timezone')
                reference = now or datetime.now(timezone.utc)
                if reference.tzinfo is None:
                    raise ValueError('Missing clock timezone')
                value = max(0.0, (parsed - reference).total_seconds())
            except (ValueError, TypeError, OverflowError):
                value = 0.0
        delay = max(delay, value)
    return delay if delay <= max_inline_wait else None


def football_data_windows(start: date, end: date, zone: str = 'Asia/Shanghai',
                          days_per_request: int = 10) -> list[tuple[str, str]]:
    """Inclusive LOCAL dates -> UTC whole-day [dateFrom, dateTo) filters.

    This is partitioning, not invented server pagination. Final event filtering
    belongs to normalization; extra UTC-edge events remain in the raw archive.
    """
    if end < start or days_per_request < 1:
        raise ValueError('Invalid date window')
    tz = ZoneInfo(zone)
    left = datetime.combine(start, time.min, tz).astimezone(timezone.utc).date()
    edge = datetime.combine(end + timedelta(days=1), time.min, tz).astimezone(timezone.utc)
    stop = edge.date() + (timedelta(days=1) if edge.time() != time.min else timedelta())
    windows = []
    while left < stop:
        right = min(stop, left + timedelta(days=days_per_request))
        windows.append((left.isoformat(), right.isoformat()))
        left = right
    return windows


def odds_credit_estimate(markets: str, regions: str, *, historical: bool = False) -> int:
    """Conservative request reservation, not the provider's billed actual cost."""
    def items(raw):
        values = {v.strip() for v in raw.split(',') if v.strip()}
        if not values:
            raise ValueError('Empty market/region configuration')
        return values
    market_values, region_values = items(markets), items(regions)
    if not market_values <= {'h2h', 'spreads', 'totals'}:
        raise ValueError('Only verified featured football markets are supported')
    if not region_values <= {'us', 'us2', 'uk', 'eu', 'au'}:
        raise ValueError('Unknown region requires current documentation verification')
    return len(market_values) * len(region_values) * (10 if historical else 1)


def api_football_next_page(payload: dict, requested_page: int, *, max_pages: int = 100) -> int | None:
    """Only for API-Football endpoints documented to return paging metadata.

    No collector for that provider is enabled yet. A bounded helper is ready for
    the actual /odds and /players paging contract, tested with synthetic data.
    """
    if payload.get('errors'):
        raise ValueError('Provider errors must be handled before pagination')
    paging = payload.get('paging')
    if not isinstance(paging, dict):
        raise ValueError('Missing paging metadata')
    current, total = paging.get('current'), paging.get('total')
    if type(current) is not int or type(total) is not int or not 1 <= current <= total:
        raise ValueError('Invalid paging metadata')
    if current != requested_page:
        raise ValueError('Repeated or unexpected page')
    if current >= total:
        return None
    if current >= max_pages:
        raise ValueError('Page budget exhausted; result is partial')
    return current + 1
