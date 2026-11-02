"""Validate response containers only; no business normalization or coercion.

Archive first. These checks classify empty data separately from shape drift;
B owns row validation, score semantics, entities and canonical conversion.
"""
from __future__ import annotations
import json


class SourceEnvelopeError(ValueError):
    def __init__(self, category: str):
        super().__init__(category)
        self.category = category


def response_count(source: str, endpoint_kind: str, body: bytes) -> int:
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeError):
        raise SourceEnvelopeError('PARSE_JSON') from None
    def require(condition):
        if not condition:
            raise SourceEnvelopeError('SCHEMA_DRIFT')
    if source == 'spftll':
        require(isinstance(payload, dict) and 'code' in payload)
        if payload['code'] != 0:
            raise SourceEnvelopeError('PROVIDER_ERROR')
        require('data' in payload)
        data = payload['data']
        if data is None:
            return 0
        if endpoint_kind in ('list', 'recent'):
            require(isinstance(data, list))
            key = 'list' if endpoint_kind == 'list' else 'matchList'
            require(all(isinstance(group, dict) and isinstance(group.get(key), list) for group in data))
            return sum(len(group[key]) for group in data)
        require(isinstance(data, dict))
        if endpoint_kind == 'odds':
            require('footballMoreSpInfo' in data)
            require(data['footballMoreSpInfo'] is None or isinstance(data['footballMoreSpInfo'], dict))
            return int(data['footballMoreSpInfo'] is not None)
        require(endpoint_kind == 'detail' and 'matchId' in data)
        return 1
    if source == 'football_data':
        require(isinstance(payload, dict))
        if payload.get('errorCode') or ('message' in payload and 'matches' not in payload and 'competitions' not in payload and 'id' not in payload):
            raise SourceEnvelopeError('PROVIDER_ERROR')
        if endpoint_kind == 'detail':
            require('id' in payload and 'homeTeam' in payload and 'awayTeam' in payload)
            return 1
        key = 'competitions' if endpoint_kind == 'catalog' else 'matches'
        require(isinstance(payload.get(key), list))
        return len(payload[key])
    if source == 'odds_api':
        if isinstance(payload, dict) and ('error_code' in payload or 'message' in payload):
            raise SourceEnvelopeError('PROVIDER_ERROR')
        if endpoint_kind == 'historical':
            require(isinstance(payload, dict) and 'timestamp' in payload and isinstance(payload.get('data'), list))
            payload = payload['data']
        require(isinstance(payload, list))
        return len(payload)
    if source == 'understat':
        require(isinstance(payload, dict) and isinstance(payload.get('dates'), list)
                and isinstance(payload.get('teams'), dict) and isinstance(payload.get('players'), list))
        return len(payload['dates'])
    raise ValueError('Unsupported source envelope')
