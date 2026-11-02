"""Recoverable immutable response archive: payload first, metadata commit last."""
from __future__ import annotations
import gzip
import hashlib
import json
import os
import re
import tempfile
from datetime import timezone
from filelock import FileLock
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from football_spider.models.contracts import Observation, now

SECRET = re.compile(r'key|token|authorization|cookie|password|secret', re.I)


def redact(value):
    if isinstance(value, dict):
        return {str(k): '[REDACTED]' if SECRET.search(str(k)) else redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(x) for x in value]
    return value


def atomic_write(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def json_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), default=str, allow_nan=False).encode('utf-8')


def digest(value) -> str:
    return hashlib.sha256(json_bytes(value)).hexdigest()


class RawArchive:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def save(self, source, endpoint, params, body: bytes, *, status=None, headers=None,
             request_id=None, context=None, error_class=None, observed_at=None, observation_id=None) -> Observation:
        timestamp = observed_at or now()
        if timestamp.tzinfo is None:
            raise ValueError('Observation time requires timezone')
        timestamp=timestamp.astimezone(timezone.utc)
        if not re.fullmatch(r'[a-zA-Z0-9_-]+', source):
            raise ValueError('Invalid source name')
        parts = urlsplit(endpoint)
        # Endpoint never includes authentication query strings or userinfo.
        endpoint = urlunsplit((parts.scheme, parts.hostname or '', parts.path, '', ''))
        fmt = 'bytes'
        try:
            json.loads(body)
            fmt = 'json'
        except (ValueError, UnicodeError):
            if b'<html' in body[:1000].lower() or b'<!doctype' in body[:1000].lower():
                fmt = 'html'
        kwargs = {'request_id': request_id} if request_id else {}
        if observation_id is not None:
            from uuid import UUID
            kwargs['id'] = str(UUID(str(observation_id)))
        obs = Observation(source=source, endpoint=endpoint, params=redact(params), observed_at=timestamp,
                          http_status=status, payload_path='', content_hash=hashlib.sha256(body).hexdigest(),
                          payload_format=fmt, context=redact(context or {}), error_class=error_class,
                          response_headers={k.lower(): str(v) for k, v in (headers or {}).items()
                                            if k.lower() in {'content-type', 'retry-after', 'x-requests-remaining', 'x-requests-used', 'x-requests-last'}}, **kwargs)
        folder = self.root / 'raw' / source / timestamp.strftime('%Y-%m-%d')
        stem = timestamp.strftime('%H%M%S_%f') + '_' + obs.id
        payload = folder / (stem + '.body.gz')
        obs.payload_path = str(payload.relative_to(self.root))
        metadata = folder / (stem + '.observation.json')
        self.root.mkdir(parents=True,exist_ok=True)
        with FileLock(str(self.root/'raw-write.lock')):
            known=list((self.root/'raw').glob(f'*/*/*_{obs.id}.observation.json')) if observation_id else ([metadata] if metadata.exists() else [])
            if known:
                previous = Observation.model_validate_json(known[0].read_bytes())
                if len(known)!=1 or previous.model_dump() != obs.model_dump():
                    raise ValueError('Immutable observation ID collision')
                self.read_body(previous)
                return previous
            atomic_write(payload, gzip.compress(body, mtime=0))
            atomic_write(metadata, obs.model_dump_json(indent=2).encode())
        return obs

    def read_body(self, obs: Observation) -> bytes:
        path = (self.root / obs.payload_path).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Archive path escapes configured root')
        body = gzip.decompress(path.read_bytes())
        if hashlib.sha256(body).hexdigest() != obs.content_hash:
            raise ValueError('Raw archive checksum mismatch')
        return body

    def observations(self, source=None):
        items = []
        for path in (self.root / 'raw').glob('*/*/*.observation.json'):
            obs = Observation.model_validate_json(path.read_bytes())
            if source is None or obs.source == source:
                items.append(obs)
        return sorted(items, key=lambda x: (x.observed_at, x.id))
