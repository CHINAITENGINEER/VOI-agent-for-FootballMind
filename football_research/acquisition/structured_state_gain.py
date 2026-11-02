"""Offline g ablation: typed observed state, visible-title routing, learned value.

Title routes are lexical capability hints, not assertions about an unread body.
No rule assigns positive gain to a topic, absence, pending state or novelty.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from . import state_candidate_gain as base
from . import historical_gain_head as numeric
from .context_predictor import _digest

VERSION = 'historical-structured-state-gain-v1'
QUERY_SCHEMA = 'historical-structured-state-query-v1'
TOPICS = ('availability', 'selection', 'tactics', 'preview', 'training', 'other')
FACETS = TOPICS[:3]
FEATURES = tuple(f'{team}_{value}' for team in ('home', 'away')
                 for value in ('unavailable', 'doubtful', 'available', 'recovering')) + tuple(
    f'title_{t}' for t in TOPICS) + tuple(
    f'{facet}_{relation}' for facet in FACETS
    for relation in ('known', 'pending', 'mixed_values', 'candidate_uncovered')) + (
    'named_known_entity', 'named_pending_entity', 'named_mixed_entity',
    'known_training_participation', 'preview_x_availability',
    'preview_x_pending_availability', 'unmodeled_body_fraction')


def title_routes(title):
    words = base.tokens(title); lower = title.casefold()
    topics = {
        'availability': bool(words & {'injury', 'injuries', 'fitness', 'availability', 'suspension', 'suspensions'}),
        'selection': bool(words & {'lineup', 'lineups', 'xi'}) or 'line-up' in lower or 'line up' in lower,
        'tactics': bool(words & {'tactical', 'tactics', 'formation', 'attack', 'attacking', 'defending', 'defensive'}),
        'preview': 'preview' in words,
        'training': bool(words & {'training', 'cobham'}),
    }
    topics['other'] = not any(topics.values())
    return {k: float(v) for k, v in topics.items()}


def build_query(public, snapshot):
    # Existing strict snapshot/visibility validation; no runtime contract change.
    old = base.build_query(public, snapshot)
    claims = [c for c in snapshot['claims'] if c['channel'] == 'body']
    # Distinct assertions, not number of repeated documents/quotations.
    facts = {(c['team'], c['entity'].casefold(), c['facet'], c['value'], c['pending']) for c in claims}
    routes = title_routes(public['title']); result = {}
    scale = lambda values: min(len(values), 5) / 5
    for team in ('home', 'away'):
        for value in ('unavailable', 'doubtful', 'available', 'recovering'):
            result[f'{team}_{value}'] = scale({e for t,e,f,v,p in facts if t == team and f == 'availability' and v == value and e})
    result.update({f'title_{t}': routes[t] for t in TOPICS})
    grouped = defaultdict(set); pending = set()
    for t,e,f,v,p in facts:
        facet = 'tactics' if f.startswith('context:tactical_plan:') else f
        if e and facet in FACETS:
            grouped[t,e,facet].add(v)
            if p: pending.add((t,e,facet))
    for facet in FACETS:
        keys = {k for k in grouped if k[2] == facet}
        result[f'{facet}_known'] = scale(keys)
        result[f'{facet}_pending'] = scale(keys & pending)
        # Mixed source assertions are a feature, not adjudicated contradictions.
        result[f'{facet}_mixed_values'] = scale({k for k in keys if len(grouped[k]) > 1})
        result[f'{facet}_candidate_uncovered'] = routes[facet] * float(not keys)
    named = {k for k in grouped if base.entity_in_title(k[1], public['title'])}
    result['named_known_entity'] = scale(named)
    result['named_pending_entity'] = scale(named & pending)
    result['named_mixed_entity'] = scale({k for k in named if len(grouped[k]) > 1})
    result['known_training_participation'] = scale({(t,e) for t,e,f,v,p in facts if 'training_participation' in f and e})
    result['preview_x_availability'] = routes['preview'] * result['availability_known']
    result['preview_x_pending_availability'] = routes['preview'] * result['availability_pending']
    result['unmodeled_body_fraction'] = sum(f == 'context:other:unmodeled' for t,e,f,v,p in facts) / max(1,len(facts))
    assert set(result) == set(FEATURES)
    return {'schema': QUERY_SCHEMA, 'base': old, 'structured': result}


def vector(model, query):
    if set(query) != {'schema', 'base', 'structured'} or query['schema'] != QUERY_SCHEMA:
        raise ValueError('Exact structured prequery contract required')
    original = base.vector(model, query['base'])
    values = query['structured']
    if set(values) != set(FEATURES): raise ValueError('Structured feature contract differs')
    x = [values[k] for k in FEATURES]
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or not 0 <= v <= 1 for v in x):
        raise ValueError('Finite bounded structured features required')
    mode = model['representation']
    if mode == 'baseline': return original
    if mode == 'structured': return np.r_[numeric.numeric(query['base']['public']), x]
    if mode == 'combined': return np.r_[original, x]
    raise ValueError('Unknown representation')


def fit(rows, representation='structured', alpha=.1):
    if representation not in ('baseline','structured','combined'): raise ValueError('Unknown representation')
    seed = base.fit([{**r,'query':r['query']['base']} for r in rows],alpha=alpha)
    model = {**deepcopy(seed), 'schema':VERSION, 'query_schema':QUERY_SCHEMA,
             'representation':representation, 'structured_features':list(FEATURES)}
    if representation == 'baseline': return model
    if representation == 'structured': model['encoder'] = {'terms':[], 'idf':[]}
    x = np.asarray([vector(model,r['query']) for r in rows])
    counts = Counter(r['physical_match_id'] for r in rows)
    weights = [1/(len(counts)*counts[r['physical_match_id']]) for r in rows]
    ridge = Ridge(alpha=alpha,solver='svd').fit(x,[r['signed_gain'] for r in rows],sample_weight=weights)
    model.update(coefficients=ridge.coef_.tolist(),intercept=float(ridge.intercept_))
    return model


def estimate(model, query):
    if model.get('schema') != VERSION or model.get('query_schema') != QUERY_SCHEMA or model.get('structured_features') != list(FEATURES):
        raise ValueError('Structured model contract differs')
    x = vector(model,query); w = np.asarray(model['coefficients'])
    if x.shape != w.shape or not np.all(np.isfinite(w)) or not np.isfinite(model['intercept']):
        raise ValueError('Invalid model parameters')
    return float(x @ w + model['intercept'])


def save(model,path):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump({**model,'model_sha256':_digest(model)},f,ensure_ascii=False,indent=2,allow_nan=False)


def load(path):
    model = json.loads(Path(path).read_text('utf-8')); digest=model.pop('model_sha256')
    if model.get('schema') != VERSION or _digest(model) != digest: raise ValueError('Model binding differs')
    return model
