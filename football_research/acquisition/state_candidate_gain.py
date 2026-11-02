"""Research-only g with query-time evidence/candidate relations.

The builder accepts an explicit already-observed snapshot, never a target body
or a labelled transition. Topic indicators are features, not value judgements.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import re
import numpy as np
from sklearn.linear_model import Ridge
from . import historical_gain_head as base
from .context_predictor import _digest

VERSION = 'historical-state-candidate-gain-v1'
QUERY_SCHEMA = 'historical-state-candidate-query-v1'
SNAPSHOT_FIELDS = {'physical_match_id', 'used_input_ids', 'claims'}
CLAIM_FIELDS = {'input_id', 'channel', 'entity', 'team', 'facet', 'value', 'pending', 'text'}
FEATURES = ('known_entities', 'pending_entities', 'conflicting_entity_facets',
            'known_body_claims', 'entity_title_overlap', 'pending_entity_title_overlap',
            'known_body_title_jaccard', 'title_injury', 'title_lineup', 'title_team_news',
            'pending_x_injury', 'pending_x_lineup', 'pending_x_team_news',
            'known_availability_x_injury', 'known_selection_x_lineup')
STOPWORDS = frozenset('the a an and of to in for on at is are was with from as will this that be have has'.split())


def tokens(text):
    return set(re.findall(r"[^\W_]+", text.casefold())) - STOPWORDS


def entity_in_title(entity, title):
    # Exact whole phrase, including multiword names: no surname/alias guessing.
    words = re.findall(r"[^\W_]+", entity.casefold())
    target = re.findall(r"[^\W_]+", title.casefold())
    return bool(words) and any(target[i:i+len(words)] == words for i in range(len(target)-len(words)+1))


def build_query(public_query, snapshot):
    base.numeric(public_query)
    if set(snapshot) != SNAPSHOT_FIELDS or not isinstance(snapshot['physical_match_id'], str):
        raise ValueError('Exact observed snapshot contract required')
    ids = snapshot['used_input_ids']
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids) or len(ids) != len(set(ids)):
        raise ValueError('Unique observed input IDs required')
    claims = snapshot['claims']
    if not isinstance(claims, list):
        raise ValueError('Observed claims required')
    for c in claims:
        if set(c) != CLAIM_FIELDS or c['input_id'] not in ids:
            raise ValueError('Claim must bind to an already observed input')
        if (c['channel'] not in ('title', 'body') or c['team'] not in ('home', 'away', 'unknown')
                or type(c['pending']) is not bool
                or not all(isinstance(c[k], str) for k in ('entity', 'facet', 'value', 'text'))):
            raise ValueError('Invalid observed claim')
    body = [c for c in claims if c['channel'] == 'body']
    # Duplicate quotations do not increase evidence counts.
    body = list({_digest(c): c for c in body}.values())
    entities = {(c['team'], c['entity'].casefold()) for c in body if c['entity'].strip()}
    pending = {(c['team'], c['entity'].casefold()) for c in body if c['pending'] and c['entity'].strip()}
    grouped = defaultdict(set)
    for c in body:
        if c['entity'].strip() and c['facet'] in ('availability', 'selection'):
            grouped[(c['team'], c['entity'].casefold(), c['facet'])].add(c['value'])
    title = public_query['title']; title_tokens = tokens(title)
    injury = float(bool(title_tokens & {'injury', 'injuries', 'fitness', 'availability', 'suspension', 'suspensions'}))
    lineup = float(bool(title_tokens & {'lineup', 'lineups', 'xi'}) or 'line-up' in title.casefold()
                   or 'line up' in title.casefold() or 'starting' in title_tokens)
    news = float('team news' in title.casefold())
    similarities = [len(title_tokens & tokens(c['text'])) / max(1, len(title_tokens | tokens(c['text']))) for c in body]
    pending_count = min(len(pending), 5)/5
    availability = min(sum(k[2] == 'availability' for k in grouped), 5)/5
    selection = min(sum(k[2] == 'selection' for k in grouped), 5)/5
    values = [min(len(entities), 10)/10, pending_count,
              min(sum(len(v) > 1 for v in grouped.values()), 5)/5,
              min(len(body), 10)/10,
              sum(entity_in_title(e, title) for _, e in entities)/max(1, len(entities)),
              sum(entity_in_title(e, title) for _, e in pending)/max(1, len(pending)),
              max(similarities, default=0.), injury, lineup, news,
              pending_count*injury, pending_count*lineup, pending_count*news,
              availability*injury, selection*lineup]
    return {'schema': QUERY_SCHEMA, 'public': deepcopy(public_query),
            'relations': dict(zip(FEATURES, values))}


def vector(model, query):
    if set(query) != {'schema', 'public', 'relations'} or query['schema'] != QUERY_SCHEMA:
        raise ValueError('Exact state-candidate query contract required')
    relations = query['relations']
    if set(relations) != set(FEATURES):
        raise ValueError('State-candidate relation contract differs')
    values = [relations[k] for k in FEATURES]
    if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not np.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError('Finite bounded relation values required')
    return np.r_[base.numeric(query['public']), base.text_vector(model['encoder'], query['public']['title']), values]


def fit(rows, alpha=.1):
    # Reuse the train-only title vocabulary and equal-match weighting.
    seed = base.fit([{**r, 'query': r['query']['public']} for r in rows], alpha=alpha)
    model = {k: deepcopy(v) for k, v in seed.items() if k not in ('coefficients', 'intercept')}
    model.update(schema=VERSION, query_schema=QUERY_SCHEMA, relation_features=list(FEATURES))
    x = np.asarray([vector(model, r['query']) for r in rows]); y = np.asarray([r['signed_gain'] for r in rows])
    counts = Counter(r['physical_match_id'] for r in rows)
    weights = [1/(len(counts)*counts[r['physical_match_id']]) for r in rows]
    ridge = Ridge(alpha=alpha, solver='svd').fit(x, y, sample_weight=weights)
    model.update(coefficients=ridge.coef_.tolist(), intercept=float(ridge.intercept_))
    return model


def estimate(model, query):
    if model.get('schema') != VERSION or model.get('query_schema') != QUERY_SCHEMA or model.get('relation_features') != list(FEATURES):
        raise ValueError('State-candidate model contract differs')
    x = vector(model, query); w = np.asarray(model['coefficients'], float)
    if w.shape != x.shape or not np.all(np.isfinite(w)) or not np.isfinite(model['intercept']):
        raise ValueError('Invalid state-candidate parameters')
    return float(x@w+model['intercept'])


def save(model, path):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump({**model, 'model_sha256': _digest(model)}, f, ensure_ascii=False, indent=2, allow_nan=False)


def load(path):
    model = json.loads(Path(path).read_text(encoding='utf-8')); digest = model.pop('model_sha256')
    if model.get('schema') != VERSION or _digest(model) != digest:
        raise ValueError('State-candidate model binding differs')
    return model
