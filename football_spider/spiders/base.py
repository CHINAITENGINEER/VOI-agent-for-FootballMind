"""Reusable HTTP clients with retry, pacing and persistent cookies."""

from __future__ import annotations

import asyncio
import json
import random
import time
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import aiohttp
import requests
from loguru import logger

from football_spider.config import Settings
from football_spider.storage.raw_archive import RawArchive
from football_spider.platform_config import PlatformSettings


class RetryableStatusError(RuntimeError):
    def __init__(self, status: int, url: str, retry_after: str | None = None):
        super().__init__(f"HTTP {status}: {url}")
        self.status = status
        self.retry_after = retry_after


class CookieStore:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> dict[str, str]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return {str(k): str(v) for k, v in data.items()}
        except FileNotFoundError:
            return {}
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("Cookie 文件读取失败，将使用空 Cookie: {}", exc)
            return {}

    def save(self, cookies: dict[str, str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(".tmp")
        temp.write_text(
            json.dumps(cookies, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temp.replace(self.path)


def retry_delay(
    attempt: int, settings: Settings, retry_after: str | None = None
) -> float:
    if retry_after:
        try:
            return max(0.0, min(float(retry_after), 120.0))
        except ValueError:
            try:
                target = parsedate_to_datetime(retry_after)
                return max(0.0, min(target.timestamp() - time.time(), 120.0))
            except (TypeError, ValueError):
                pass
    jitter = random.uniform(settings.min_delay, settings.max_delay)
    return min(settings.retry_backoff ** attempt + jitter, 120.0)


class RequestsJsonClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cookie_store = CookieStore(settings.cookie_path)
        self.session = requests.Session()
        self.session.cookies.update(self.cookie_store.load())
        self.archive = RawArchive(PlatformSettings().data_dir)

    def close(self) -> None:
        self.cookie_store.save(self.session.cookies.get_dict())
        self.session.close()

    def __enter__(self) -> "RequestsJsonClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def get_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.settings.retries):
            archived = False
            if attempt == 0:
                time.sleep(random.uniform(self.settings.min_delay, self.settings.max_delay))
            headers = self.settings.browser_headers(random.choice(self.settings.user_agents))
            try:
                response = self.session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=(
                        self.settings.connect_timeout,
                        self.settings.read_timeout,
                    ),
                )
                self.archive.save('spftll', url, params, response.content,
                                  status=response.status_code, headers=dict(response.headers),
                                  context={'entrypoint': 'legacy'})
                archived = True
                if response.status_code in self.settings.retry_statuses:
                    raise RetryableStatusError(
                        response.status_code,
                        response.url,
                        response.headers.get("Retry-After"),
                    )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError(f"接口返回的不是 JSON 对象: {response.url}")
                return payload
            except (requests.RequestException, ValueError, RetryableStatusError) as exc:
                if not archived:
                    self.archive.save('spftll', url, params, b'', error_class=type(exc).__name__,
                                      context={'entrypoint': 'legacy'})
                last_error = exc
                if attempt + 1 >= self.settings.retries:
                    break
                delay = retry_delay(
                    attempt,
                    self.settings,
                    exc.retry_after if isinstance(exc, RetryableStatusError) else None,
                )
                logger.warning(
                    "请求失败，第 {}/{} 次重试将在 {:.1f}s 后进行: {}",
                    attempt + 1,
                    self.settings.retries,
                    delay,
                    exc,
                )
                time.sleep(delay)
        raise RuntimeError(f"请求在重试后仍失败: {url}") from last_error


class AiohttpJsonClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cookie_store = CookieStore(settings.cookie_path)
        self.session: aiohttp.ClientSession | None = None
        self.semaphore = asyncio.Semaphore(settings.concurrency)
        self.archive = RawArchive(PlatformSettings().data_dir)

    async def __aenter__(self) -> "AiohttpJsonClient":
        timeout = aiohttp.ClientTimeout(
            total=None,
            connect=self.settings.connect_timeout,
            sock_read=self.settings.read_timeout,
        )
        jar = aiohttp.CookieJar(unsafe=True)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            cookie_jar=jar,
            cookies=self.cookie_store.load(),
        )
        return self

    async def __aexit__(self, *_: object) -> None:
        assert self.session is not None
        cookie_map: dict[str, str] = {}
        for cookie in self.session.cookie_jar:
            cookie_map[cookie.key] = cookie.value
        self.cookie_store.save(cookie_map)
        await self.session.close()

    async def get_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        assert self.session is not None
        last_error: Exception | None = None
        for attempt in range(self.settings.retries):
            headers = self.settings.browser_headers(random.choice(self.settings.user_agents))
            archived = False
            try:
                async with self.semaphore:
                    await asyncio.sleep(
                        random.uniform(self.settings.min_delay, self.settings.max_delay)
                    )
                    async with self.session.get(
                        url, params=params, headers=headers
                    ) as response:
                        body = await response.read()
                        self.archive.save('spftll', url, params, body,
                                          status=response.status, headers=dict(response.headers),
                                          context={'entrypoint': 'legacy'})
                        archived = True
                        if response.status in self.settings.retry_statuses:
                            raise RetryableStatusError(
                                response.status,
                                str(response.url),
                                response.headers.get("Retry-After"),
                            )
                        response.raise_for_status()
                        payload = await response.json(content_type=None)
                        if not isinstance(payload, dict):
                            raise ValueError(f"接口返回的不是 JSON 对象: {response.url}")
                        return payload
            except (
                aiohttp.ClientError,
                asyncio.TimeoutError,
                ValueError,
                RetryableStatusError,
            ) as exc:
                if not archived:
                    self.archive.save('spftll', url, params, b'', error_class=type(exc).__name__,
                                      context={'entrypoint': 'legacy'})
                last_error = exc
                if attempt + 1 >= self.settings.retries:
                    break
                delay = retry_delay(
                    attempt,
                    self.settings,
                    exc.retry_after if isinstance(exc, RetryableStatusError) else None,
                )
                logger.warning(
                    "异步请求失败，第 {}/{} 次重试将在 {:.1f}s 后进行: {}",
                    attempt + 1,
                    self.settings.retries,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)
        raise RuntimeError(f"请求在重试后仍失败: {url}") from last_error
