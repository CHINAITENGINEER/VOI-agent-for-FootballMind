"""HTTP-only building block; not a second Collector/RawObservation protocol.

The mandatory on_attempt hook is a private transport seam. Production adapters
in football_spider bind it to the shared Observation / RawArchive before parsing.
This component remains independent of business normalization and persistence.
"""
from __future__ import annotations

from datetime import datetime, timezone
from http.cookiejar import LoadError, MozillaCookieJar
from pathlib import Path
import os
import random
import re
import time
from urllib.parse import urlsplit
from uuid import uuid4

import requests

from .policy import retry_delay

_SECRET = re.compile(r'authorization|cookie|token|api.?key|rkey|password|secret', re.I)
_RESPONSE_HEADERS = {'content-type', 'retry-after', 'date', 'etag', 'last-modified', 'location',
                     'x-requests-remaining', 'x-requests-used', 'x-requests-last',
                     'x-requestcounter-reset', 'x-requests-available-minute',
                     'x-ratelimit-requests-remaining', 'x-ratelimit-remaining'}


class TransportError(RuntimeError):
    def __init__(self, category: str, observation_id: str | None = None):
        super().__init__(category)
        self.category = category
        self.observation_id = observation_id


class RequestTransport:
    def __init__(self, source, *, on_attempt, session=None, cookie_path=None,
                 max_attempts=3, request_budget=30, credit_budget=30, interval=1.0,
                 timeout=(10, 35), reserve=None, user_agents=(), sleep=time.sleep,
                 monotonic=time.monotonic, clock=lambda: datetime.now(timezone.utc),deadline=None):
        if not callable(on_attempt):
            raise ValueError('Archive callback is mandatory')
        if max_attempts < 1 or request_budget < 0 or credit_budget < 0 or interval < 0:
            raise ValueError('Invalid transport limits')
        self.source, self.on_attempt = source, on_attempt
        self.session = session if session is not None else requests.Session()
        self.cookie_path = Path(cookie_path) if cookie_path else None
        self.max_attempts, self.request_budget, self.credit_budget = max_attempts, request_budget, credit_budget
        self.interval, self.timeout, self.reserve = interval, timeout, reserve
        self.user_agents = tuple(user_agents)
        self.sleep, self.monotonic, self.clock = sleep, monotonic, clock
        self.deadline = deadline
        self.last_start = None
        self.provider_remaining = None
        self.stats = dict(logical_requests=0, attempts=0, archived_attempts=0, retries=0,
                          transport_errors=0, http_errors=0, credits_reserved=0,
                          archive_errors=0, cookie_errors=0)
        self.stats['cookie_errors']+=getattr(self.session,'cookie_errors',0)
        if self.cookie_path and session is None:
            jar = MozillaCookieJar(str(self.cookie_path))
            if self.cookie_path.exists():
                try:
                    jar.load(ignore_discard=True, ignore_expires=False)
                except (LoadError, OSError):
                    # Keep the original file; use a clean in-memory jar this run.
                    self.stats['cookie_errors'] += 1
            self.session.cookies = jar

    def get(self, url, params=None, *, headers=None, estimated_credits=0):
        """Return archived response bytes. Status errors raise AFTER the hook.

        The body is intentionally not parsed here. Non-JSON and 200 provider
        error envelopes remain auditable; callers apply envelopes only afterward.
        """
        parts = urlsplit(url)
        if parts.scheme not in ('http', 'https') or not parts.netloc or parts.query or parts.username or parts.password or parts.fragment:
            raise ValueError('Use a credential-free endpoint and separate params/headers')
        if type(estimated_credits) is not int or estimated_credits < 0:
            raise ValueError('Invalid estimated credits')
        params, headers = dict(params or {}), dict(headers or {})
        request_id = str(uuid4())
        self.stats['logical_requests'] += 1
        for attempt in range(1, self.max_attempts + 1):
            if self.deadline is not None and self.monotonic() >= self.deadline:
                raise TransportError('TIMEOUT')
            if (self.stats['attempts'] >= self.request_budget
                    or self.stats['credits_reserved'] + estimated_credits > self.credit_budget
                    or (self.provider_remaining is not None and self.provider_remaining < max(1, estimated_credits))):
                raise TransportError('RATE_LIMITED')
            # E can inject the existing atomic cross-process quota reservation.
            if self.reserve is not None and not self.reserve(self.source, estimated_credits):
                raise TransportError('RATE_LIMITED')
            self.stats['credits_reserved'] += estimated_credits
            if self.last_start is not None:
                pacing=max(0,self.interval-(self.monotonic()-self.last_start))+random.uniform(0,0.2)
                if self.deadline is not None and self.monotonic()+pacing>=self.deadline:
                    raise TransportError('TIMEOUT')
                self.sleep(pacing)
            request_timeout=self.timeout
            if self.deadline is not None:
                left=self.deadline-self.monotonic()
                if left<=0:raise TransportError('TIMEOUT')
                request_timeout=(min(self.timeout[0],left/2),min(self.timeout[1],left/2))
            self.last_start = self.monotonic()
            self.stats['attempts'] += 1
            self.stats['retries'] += int(attempt > 1)
            started = self.clock()
            observation_id = str(uuid4())
            actual_headers = dict(headers)
            if self.user_agents:
                actual_headers.setdefault('User-Agent', random.choice(self.user_agents))
            response, body, status, error_class = None, b'', None, None
            response_headers = {}
            try:
                response = self.session.get(url, params=params, headers=actual_headers,
                                            timeout=request_timeout, allow_redirects=False)
                body, status = response.content, response.status_code
                response_headers = {k.lower(): v for k, v in response.headers.items() if k.lower() in _RESPONSE_HEADERS}
            except requests.RequestException as exc:
                error_class = type(exc).__name__
                self.stats['transport_errors'] += 1
            finally:
                if response is not None:
                    response.close()
            if status is not None and not 200 <= status < 300:
                self.stats['http_errors'] += 1
            redacted_body = body
            for key, value in {**params, **actual_headers}.items():
                if _SECRET.search(key) and value and str(value) != 'null':
                    redacted_body = redacted_body.replace(str(value).encode(), b'[REDACTED]')
            facts = dict(source=self.source, request_id=request_id, observation_id=observation_id,
                         attempt=attempt, endpoint=url, method='GET',
                         params={k: '[REDACTED]' if _SECRET.search(k) else v for k,v in params.items()},
                         request_headers={k: '[REDACTED]' if _SECRET.search(k) else v for k,v in actual_headers.items()},
                         requested_at=started, observed_at=self.clock(), http_status=status,
                         response_headers=response_headers, body=redacted_body, error_class=error_class,
                         credential_echo_redacted=(redacted_body != body))
            try:
                self.on_attempt(**facts)
                self.stats['archived_attempts'] += 1
            except Exception:
                self.stats['archive_errors'] += 1
                raise TransportError('ARCHIVE_FAILED', observation_id) from None
            for quota_header in ('x-requests-remaining', 'x-ratelimit-requests-remaining',
                                 'x-ratelimit-remaining', 'x-requests-available-minute'):
                remaining = response_headers.get(quota_header)
                if remaining is None:
                    continue
                try:
                    value = max(0, int(remaining))
                    self.provider_remaining = value if self.provider_remaining is None else min(value, self.provider_remaining)
                except (TypeError, ValueError):
                    pass
            if status is not None and 200 <= status < 300:
                return redacted_body
            category = error_class or ('RATE_LIMITED' if status == 429 else f'HTTP_{status}')
            delay = retry_delay(attempt, status=status, retry_after=response_headers.get('retry-after'),
                                now=self.clock(), max_attempts=self.max_attempts, jitter=random.uniform(0, 0.2))
            if delay is None:
                raise TransportError(category, observation_id)
            if self.deadline is not None and self.monotonic()+delay>=self.deadline:
                raise TransportError('TIMEOUT',observation_id)
            self.sleep(delay)
        raise TransportError('RETRIES_EXHAUSTED')

    def close(self):
        try:
            if self.cookie_path and isinstance(self.session.cookies, MozillaCookieJar):
                # Preserve an unreadable original for operator investigation.
                if not self.stats['cookie_errors']:
                    self.cookie_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = self.cookie_path.with_name(self.cookie_path.name + '.' + str(uuid4()) + '.tmp')
                    try:
                        self.session.cookies.save(str(temporary), ignore_discard=True)
                        os.replace(temporary, self.cookie_path)
                    finally:
                        if temporary.exists():
                            temporary.unlink()
        except OSError:
            self.stats['cookie_errors'] += 1
        finally:
            self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
