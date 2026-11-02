'''Registered public endpoints and pure platform projections; never fetch eagerly.'''
from __future__ import annotations
from copy import deepcopy
import json
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import UUID
from football_spider.config import settings
from football_spider.models.contracts import PARSER_VERSION, SCHEMA_VERSION, Observation
from football_spider.normalization.sources import normalize_spftll

VERSION = 'public-platform-catalog-v1'
PROJECTION_VERSION = 'spftll-agent-projection-v1'
MID = re.compile(r'^[0-9]{1,20}$')


def source_catalog():
    '''Capability metadata only. No match facts or mandatory query order.'''
    return {'version': VERSION, 'projection_version': PROJECTION_VERSION,
            'parser_version': PARSER_VERSION, 'schema_version': SCHEMA_VERSION,
            'followup_endpoints': [settings.detail_api_url, settings.more_odds_api_url],
            'entries': [
                {'id': 'spftll_recent', 'source': 'spftll', 'action': 'read_url',
                 'url': settings.generic_match_list_api_url,
                 'description': '来源内近期赛程，可获取源比赛 ID、球队和时间；覆盖并不完整。'},
                {'id': 'spftll_lottery', 'source': 'spftll', 'action': 'read_url',
                 'url': settings.list_api_url + '?' + urlencode(
                     {'dateFormat': '', 'notSingle': 'false', 'mKind': 0, 'reqtype': 0}),
                 'description': '来源内当前足彩选场与报价；不代表全部赛事或历史赔率。'},
            ],
            'selection_rule': 'Optional candidates, not a required checklist or priority order.',
            'detail_rule': 'Only follow detail/odds links returned from observed source match IDs.',
            'time_basis': 'Present-time observation; source publication time remains unknown.',
            'coverage': 'selected_recent_matches_not_complete_league_or_historical_backfill',
            'cost': {'http_request_lower_bound': 1, 'money_estimate': None,
                     'latency_seconds_estimate': None}, 'semantic_relevance': 'unassessed'}


def registered_request(url):
    '''Exact endpoints and public parameters; headers are never supplied by candidates.'''
    if not isinstance(url, str):
        return None
    try:
        parts = urlsplit(url)
        if parts.scheme != 'https' or parts.username or parts.password or parts.fragment or parts.port not in (None, 443):
            return None
        pairs = parse_qsl(parts.query, keep_blank_values=True)
        params = dict(pairs)
        if len(params) != len(pairs):
            return None
        endpoint = urlunsplit((parts.scheme, parts.netloc, parts.path, '', ''))
        if endpoint == settings.generic_match_list_api_url and not params:
            kind = 'recent'
        elif endpoint == settings.list_api_url and params == {
                'dateFormat': '', 'notSingle': 'false', 'mKind': '0', 'reqtype': '0'}:
            kind = 'lottery'
        elif (endpoint == settings.detail_api_url and set(params) == {'gameKind', 'matchId'}
              and params['gameKind'] == '0' and MID.fullmatch(params['matchId'])):
            kind = 'detail'
        elif (endpoint == settings.more_odds_api_url and set(params) == {'mKind', 'matchId'}
              and params['mKind'] == '0' and MID.fullmatch(params['matchId'])):
            kind = 'odds'
        else:
            return None
    except ValueError:
        return None
    return {'source': 'spftll', 'kind': kind, 'endpoint': endpoint, 'params': params}


def detail_links(match_id):
    if not MID.fullmatch(str(match_id)):
        return []
    return [
        {'url': settings.detail_api_url + '?' + urlencode({'gameKind': 0, 'matchId': match_id}),
         'title': 'spftll 比赛 ' + str(match_id) + ' 详情与来源内近期交锋'},
        {'url': settings.more_odds_api_url + '?' + urlencode({'mKind': 0, 'matchId': match_id}),
         'title': 'spftll 比赛 ' + str(match_id) + ' 当前报价'},
    ]


def project_platform(observation, body, document, *, record_sink=None):
    '''Adapt an already archived response without network or database writes.'''
    spec = registered_request(document['url'])
    if spec is None:
        return document
    doc = deepcopy(document)
    doc['platform'] = {'catalog_version': VERSION, 'projection_version': PROJECTION_VERSION,
                       'parser_version': PARSER_VERSION, 'schema_version': SCHEMA_VERSION,
                       'source': 'spftll', 'endpoint_kind': spec['kind'],
                       'observation_id': observation.id, 'quality': 'unavailable',
                       'coverage': 'selected_matches_incomplete',
                       'availability_basis': 'observed_at_only_not_verified_publication',
                       'matches': 0, 'quotes': 0, 'issues': 0, 'records_displayed': 0}
    if observation.error_class or observation.http_status != 200:
        return doc
    doc.update(content_kind='platform_structured', title='spftll ' + spec['kind'],
               text='', links=[], text_truncated=False)
    try:
        payload = json.loads(body)
        if not isinstance(payload, dict) or 'data' not in payload or payload.get('code') != 0:
            raise ValueError('PROVIDER_ENVELOPE_INVALID')
        data = payload['data']
        if data is not None:
            if spec['kind'] in {'recent', 'lottery'}:
                child = 'matchList' if spec['kind'] == 'recent' else 'list'
                if (not isinstance(data, list) or any(not isinstance(g, dict) or child not in g
                        or g[child] is not None and not isinstance(g[child], list) for g in data)):
                    raise ValueError('SCHEDULE_SHAPE_INVALID')
            elif not isinstance(data, dict):
                raise ValueError('DETAIL_SHAPE_INVALID')
            elif spec['kind'] == 'detail' and str(data.get('matchId')) != spec['params']['matchId']:
                raise ValueError('DETAIL_ID_MISMATCH')
            elif spec['kind'] == 'odds' and data.get('footballMoreSpInfo'):
                if str(data['footballMoreSpInfo'].get('matchId')) != spec['params']['matchId']:
                    raise ValueError('ODDS_ID_MISMATCH')
        batch = normalize_spftll(observation, payload)
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
        doc.update(status='invalid_content', signals=['platform_payload_invalid'])
        doc['platform']['quality'] = 'invalid'
        return doc
    meta = doc['platform']
    meta.update(quotes=len(batch.quotes), issues=len(batch.issues))
    records = []
    for match in batch.matches:
        if not match.home.name.strip() or not match.away.name.strip() or not MID.fullmatch(match.source_id):
            meta['issues'] += 1
            continue
        row = match.model_dump(mode='json')
        records.append({'kind': 'fixture', 'source': 'spftll',
                        **{k: row[k] for k in ('source_id', 'competition', 'home', 'away', 'kickoff',
                            'status', 'score', 'half_score', 'score_period', 'missing', 'coverage')}})
    meta['matches'] = len(records)
    for quote in batch.quotes:
        records.append({'kind': 'odds', **quote.model_dump(mode='json')})
    if record_sink is not None:
        record_sink.extend(deepcopy(records))
    if not records:
        doc.update(status='invalid_content' if meta['issues'] else 'empty_structured',
                   signals=['platform_records_invalid' if meta['issues'] else 'platform_no_data_not_no_match'])
        meta['quality'] = 'invalid' if meta['issues'] else 'no_data'
        return doc
    meta['quality'] = 'partial' if meta['issues'] else 'normalized'
    doc.update(status='readable', signals=['platform_normalized_not_independently_verified',
                                          'source_coverage_incomplete', 'current_observation_not_historical_snapshot'])
    if meta['issues']:
        doc['signals'].append('platform_validation_issues')
    lines, length = [], 0
    for row in records:
        line = json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
        if length + len(line) + 1 > 12000:
            break
        lines.append(line)
        length += len(line) + 1
        if row['kind'] == 'fixture':
            for link in detail_links(row['source_id']):
                if link['url'] != doc['url'] and link not in doc['links'] and len(doc['links']) < 20:
                    doc['links'].append(link)
    doc['text'] = chr(10).join(lines)
    doc['text_truncated'] = len(lines) < len(records)
    meta['records_displayed'] = len(lines)
    if doc['text_truncated']:
        doc['signals'].append('bounded_source_order_projection_not_complete_result_set')
    if not lines:
        doc['status'] = 'unsupported_content'
        meta['quality'] = 'projection_window_exceeded'
    return doc


def verify_platform_document(archive, document, *, record_sink=None):
    '''Reconstruct a structured document from its immutable raw observation.'''
    from .experience_sources import project_body

    meta = document['platform']
    if meta['projection_version'] != PROJECTION_VERSION:
        raise ValueError('PLATFORM_PROJECTION_VERSION_UNSUPPORTED')
    observation_id = str(UUID(meta['observation_id']))
    paths = list((archive.root / 'raw' / 'spftll').glob('*/*_' + observation_id + '.observation.json'))
    if len(paths) != 1:
        raise ValueError('PLATFORM_RAW_OBSERVATION_MISSING')
    observation = Observation.model_validate_json(paths[0].read_bytes())
    spec = registered_request(document['url'])
    if (not spec or observation.source != 'spftll' or observation.endpoint != spec['endpoint']
            or observation.params != spec['params'] or observation.request_id != document['request_id']
            or observation.content_hash != document['raw_sha256']
            or document['document_id'] != 'doc-' + observation.id
            or document['observed_at'] != observation.observed_at.isoformat()
            or document['source_published_at'] is not None):
        raise ValueError('PLATFORM_OBSERVATION_BINDING_MISMATCH')
    body = archive.read_body(observation)
    base = project_body(body, document['url'], observation.http_status,
                        observation.response_headers.get('content-type', ''), observation.error_class)
    expected = project_platform(observation, body, base, record_sink=record_sink)
    if any(document.get(key) != value for key, value in expected.items()):
        raise ValueError('PLATFORM_PROJECTION_REPLAY_MISMATCH')
    return {'observation_id': observation.id, 'raw_sha256': observation.content_hash,
            'projection_version': PROJECTION_VERSION}
