from __future__ import annotations
import json
import random
import time
from email.utils import parsedate_to_datetime
from http.cookiejar import MozillaCookieJar
from uuid import uuid4
import requests
from football_spider.models.contracts import now
from football_spider.storage.raw_archive import RawArchive
from football_spider.collectors.quota import request_slot


class FetchError(RuntimeError):
    def __init__(self, category, observation=None):
        super().__init__(category)
        self.category = category
        self.observation = observation


class ArchiveHttp:
    """Every attempt is archived before status checking or JSON parsing.

    Query/header credentials never enter summaries/exceptions. Cookie jars are
    source-scoped. No CAPTCHA evasion, unlimited retries or rate-limit bypass.
    """
    def __init__(self, archive: RawArchive, source, *, max_requests=80, interval=1.0,
                 retries=3, session=None, sleep=time.sleep):
        self.archive, self.source = archive, source
        self.max_requests, self.interval, self.retries = max_requests, interval, retries
        self.session = session or requests.Session()
        self.sleep = sleep
        self.observations = []
        self.last_request = 0.0
        self.quota_remaining = None
        self.cookie_path = archive.root / 'cookies' / (source + '.txt')
        if session is None:
            jar = MozillaCookieJar(str(self.cookie_path))
            if self.cookie_path.exists():
                jar.load(ignore_discard=True, ignore_expires=False)
            self.session.cookies = jar

    def close(self):
        self.cookie_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(self.session.cookies, MozillaCookieJar):
            self.session.cookies.save(ignore_discard=True)
        self.session.close()

    def get(self, url, params=None, *, headers=None, context=None):
        request_id = str(uuid4())
        params = params or {}
        for attempt in range(self.retries):
            if len(self.observations) >= self.max_requests or self.quota_remaining == 0:
                raise FetchError('RATE_LIMITED')
            delay = request_slot(self.archive.root, self.source, self.interval)
            if delay is None:
                raise FetchError('RATE_LIMITED')
            self.sleep(delay + random.uniform(0.05, 0.2))
            self.last_request = time.monotonic()
            try:
                response = self.session.get(url, params=params, headers=headers, timeout=(10, 35))
            except requests.RequestException as exc:
                obs = self.archive.save(self.source, url, params, b'', request_id=request_id,
                                        context=context, error_class=type(exc).__name__)
                self.observations.append(obs)
                if attempt + 1 == self.retries:
                    raise FetchError(type(exc).__name__, obs) from None
                self.sleep(min(2 ** attempt, 30))
                continue
            # Error responses can echo API keys; redact known supplied secrets
            # before archiving, and record that redaction happened.
            body = response.content
            secrets = [str(v) for k, v in {**params, **(headers or {})}.items()
                       if any(s in k.lower() for s in ('token', 'apikey', 'api-key', 'authorization')) and v]
            changed = False
            for secret in secrets:
                if secret not in ('null',) and secret.encode() in body:
                    body = body.replace(secret.encode(), b'[REDACTED]')
                    changed = True
            obs = self.archive.save(self.source, url, params, body, status=response.status_code,
                                    headers=response.headers, request_id=request_id,
                                    context={**(context or {}), 'credential_echo_redacted': changed})
            self.observations.append(obs)
            remaining = response.headers.get('x-requests-remaining')
            if remaining is not None:
                try:
                    self.quota_remaining = max(0, int(remaining))
                except ValueError:
                    pass
            if response.status_code in (403, 429, 500, 502, 503, 504):
                category = 'RATE_LIMITED' if response.status_code == 429 else f'HTTP_{response.status_code}'
                if attempt + 1 == self.retries:
                    raise FetchError(category, obs)
                retry_after = response.headers.get('Retry-After')
                wait = min(2 ** attempt, 30)
                if retry_after:
                    try:
                        wait = max(0, float(retry_after))
                    except ValueError:
                        try:
                            wait = max(0, (parsedate_to_datetime(retry_after) - now()).total_seconds())
                        except (ValueError, TypeError):
                            pass
                if wait > 60:
                    # Defer to a later invocation rather than ignore Retry-After.
                    raise FetchError(category, obs)
                self.sleep(wait)
                continue
            if not 200 <= response.status_code < 300:
                raise FetchError(f'HTTP_{response.status_code}', obs)
            try:
                payload = json.loads(body)
            except (ValueError, UnicodeError):
                raise FetchError('PARSE_JSON', obs) from None
            return obs, payload
        raise FetchError('RETRIES_EXHAUSTED')
