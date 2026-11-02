"""Small research g estimating fixed-f realized loss reduction from prequery data.

Only the explicit public feature contract enters the regressor. Historical
reconstructed menus are research assumptions, not recorded online propensities.
"""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from .context_predictor import _digest

VERSION = 'historical-prequery-gain-head-v1'
QUERY_SCHEMA = 'historical-prequery-visible-features-v1'
FIELDS = {'schema', 'probabilities', 'acquired_documents', 'visible_candidates', 'article_age_hours', 'title'}


def numeric(query):
    if set(query) != FIELDS or query['schema'] != QUERY_SCHEMA:
        raise ValueError('Exact prequery feature contract required')
    p = np.asarray(query['probabilities'], float)
    if p.shape != (3,) or not np.all(np.isfinite(p)) or np.any(p <= 0) or abs(p.sum()-1) > 1e-10:
        raise ValueError('Valid prequery probabilities required')
    for key in ('acquired_documents', 'visible_candidates'):
        if type(query[key]) is not int or query[key] < 0:
            raise ValueError('Nonnegative prequery counts required')
    age = query['article_age_hours']
    if isinstance(age, bool) or not isinstance(age, (float, int)) or not np.isfinite(age) or age < 0:
        raise ValueError('Nonnegative finite article age required')
    if not isinstance(query['title'], str):
        raise ValueError('Visible title must be text')
    return np.r_[p, -np.sum(p*np.log(p))/np.log(3),
                 query['acquired_documents']/3, query['visible_candidates']/3, min(age, 168)/168]


def text_vector(encoder, title):
    analyzer = TfidfVectorizer(ngram_range=(1, 2)).build_analyzer()
    counts = Counter(analyzer(title))
    x = np.asarray([(1+np.log(counts[t]))*idf if counts[t] else 0.
                    for t, idf in zip(encoder['terms'], encoder['idf'])])
    norm = np.linalg.norm(x)
    return x/norm if norm else x


def vector(model, query):
    return np.r_[numeric(query), text_vector(model['encoder'], query['title'])]


def fit(rows, *, use_title=True, alpha=0.1):
    if not rows or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('Training rows and positive ridge penalty required')
    for row in rows:
        numeric(row['query'])
        if not np.isfinite(row['signed_gain']):
            raise ValueError('Finite training gain required')
    # Avoid letting enumerated state counts change title document frequency.
    titles = sorted({(r['physical_match_id'], r['query']['title']) for r in rows})
    encoder = {'terms': [], 'idf': []}
    if use_title:
        tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=32, sublinear_tf=True)
        try:
            tfidf.fit([t for _, t in titles])
            encoder = {'terms': tfidf.get_feature_names_out().tolist(), 'idf': tfidf.idf_.tolist()}
        except ValueError as exc:
            if 'empty vocabulary' not in str(exc):
                raise
    model = {'schema': VERSION, 'query_schema': QUERY_SCHEMA, 'encoder': encoder,
             'fit_match_ids': sorted({r['physical_match_id'] for r in rows}), 'alpha': alpha,
             'provenance': 'historical_reconstructed_fixed_f_research',
             'production_ready': False, 'empirical_selection_allowed': False,
             'calibration_status': 'not_established'}
    counts = Counter(r['physical_match_id'] for r in rows)
    weights = [1/(len(counts)*counts[r['physical_match_id']]) for r in rows]
    x = np.asarray([vector(model, r['query']) for r in rows]); y = np.asarray([r['signed_gain'] for r in rows])
    ridge = Ridge(alpha=alpha, solver='svd').fit(x, y, sample_weight=weights)
    model.update(coefficients=ridge.coef_.tolist(), intercept=float(ridge.intercept_),
                 constant_baseline=float(np.average(y, weights=weights)))
    return model


def estimate(model, query):
    if model.get('schema') != VERSION or model.get('query_schema') != QUERY_SCHEMA:
        raise ValueError('Gain model contract differs')
    x = vector(model, query); w = np.asarray(model['coefficients'], float)
    if w.shape != x.shape or not np.all(np.isfinite(w)) or not np.isfinite(model['intercept']):
        raise ValueError('Invalid gain parameters')
    return float(x@w+model['intercept'])


def signed_gain(before, after, outcome):
    if type(outcome) is not int or outcome not in (0, 1, 2):
        raise ValueError('Valid outcome required')
    for p in (before, after):
        numeric({'schema': QUERY_SCHEMA, 'probabilities': p, 'acquired_documents': 0,
                 'visible_candidates': 1, 'article_age_hours': 0., 'title': ''})
    return float(np.log(after[outcome])-np.log(before[outcome]))


def save(model, path):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump({**deepcopy(model), 'model_sha256': _digest(model)}, f, ensure_ascii=False, indent=2, allow_nan=False)


def load(path):
    model = json.loads(Path(path).read_text(encoding='utf-8')); digest = model.pop('model_sha256')
    if model.get('schema') != VERSION or _digest(model) != digest:
        raise ValueError('Gain model binding differs')
    return model
