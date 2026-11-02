"""Synchronous facade over one persistent aiohttp session for legacy hosts.

Uses the same TLS-verifying transport as the verified legacy detail collector.
No certificate verification downgrade; a private event loop lives for one batch.
"""
import asyncio
import json
from http.cookies import SimpleCookie
from types import SimpleNamespace
import aiohttp
import requests
from yarl import URL
from football_spider.storage.raw_archive import atomic_write, json_bytes


class AioSession:
    cookies = None

    def __init__(self, cookie_path):
        self.cookie_path = cookie_path
        self.cookie_errors = 0
        self.loop = asyncio.new_event_loop()
        self.session = self.loop.run_until_complete(self._open())

    async def _open(self):
        jar = aiohttp.CookieJar()
        if self.cookie_path.exists():
            try:
                items=json.loads(self.cookie_path.read_bytes())
                if not isinstance(items,list) or any(not isinstance(i,dict) or 'name' not in i or 'value' not in i for i in items):
                    raise ValueError('Invalid cookie archive')
            except (ValueError,OSError):
                items=[];self.cookie_errors+=1
            for item in items:
                cookie = SimpleCookie()
                cookie[item['name']] = item['value']
                for key in ('domain', 'path', 'expires', 'secure', 'httponly', 'max-age'):
                    if item.get(key):
                        cookie[item['name']][key] = item[key]
                host = item.get('domain', '').lstrip('.')
                if host:
                    jar.update_cookies(cookie, URL('https://' + host + (item.get('path') or '/')))
        return aiohttp.ClientSession(cookie_jar=jar, timeout=aiohttp.ClientTimeout(total=45, connect=10, sock_read=35))

    def get(self, url, *, params, headers, timeout, allow_redirects=False):
        async def fetch():
            limits=aiohttp.ClientTimeout(total=sum(timeout),connect=timeout[0],sock_read=timeout[1])
            async with self.session.get(url, params=params, headers=headers,timeout=limits,allow_redirects=allow_redirects) as response:
                return SimpleNamespace(status_code=response.status, content=await response.read(), headers=dict(response.headers),close=lambda:None)
        try:
            return self.loop.run_until_complete(fetch())
        except asyncio.TimeoutError:
            raise requests.Timeout('aiohttp timeout') from None
        except aiohttp.ClientError:
            raise requests.ConnectionError('aiohttp connection error') from None

    def close(self):
        cookies = [{'name': m.key, 'value': m.value, **{k: m[k] for k in ('domain', 'path', 'expires', 'secure', 'httponly', 'max-age')}} for m in self.session.cookie_jar]
        try:
            # Preserve corrupt original for diagnosis; a later run can inspect it.
            target=self.cookie_path.with_suffix('.recovered.json') if self.cookie_errors else self.cookie_path
            atomic_write(target, json_bytes(cookies))
        finally:
            self.loop.run_until_complete(self.session.close())
            self.loop.run_until_complete(asyncio.sleep(0.25))
            self.loop.close()
