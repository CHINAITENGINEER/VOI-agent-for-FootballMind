"""Historical corpus experiment: frozen numeric f plus a gated text residual.

This module never fetches data and has no production policy registration. The
caller supplies only evidence already read; outcome labels are fit-only inputs.
"""
from __future__ import annotations

import re
import json
import hashlib
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax, logsumexp

SCHEMA = 'footballmind-corpus-text-residual-f-v1'


def scope_exclusion(article):
    text = ' '.join((article['title'], str(article.get('source_category') or ''),
                     ' '.join(article.get('source_keywords') or [])))
    if re.search(r"\bwomen(?:['’]?s)?\b|\bu-?\d{2}s?\b|\bunder[- ]\d{2}s?\b|\bacademy\b|\bwsl\b|\byouth\b", text, re.I):
        return 'explicit_other_age_or_gender_scope'
    return None


def split_name(date):
    return 'train' if date < '2025-07-01' else 'validation' if date < '2026-01-01' else 'test'


def make_panel(matches, links, articles):
    """Selection and split use identities/dates only, never result labels.

    Cross-split shared body clusters are removed from *all* affected splits.
    Same-date bodies are excluded because kickoff ordering is not established.
    """
    accepted, rejected = [], []
    for link in links:
        a, m = articles[link['article_id']], matches[link['match_id']]
        reason = scope_exclusion(a)
        if link['days_before_match_date'] < 1:
            reason = 'same_day_kickoff_order_unknown'
        if a.get('modified_at') and a['modified_at'][:10] >= m['match_date']:
            reason = 'source_declared_modification_on_or_after_match_date'
        item = {**link, 'date': m['match_date'], 'split': split_name(m['match_date']),
                'body_sha256': a['body_sha256']}
        (rejected if reason else accepted).append({**item, 'reason': reason})
    clusters = defaultdict(set)
    for item in accepted:
        clusters[item['body_sha256']].add(item['split'])
    bags = defaultdict(list)
    for item in accepted:
        if len(clusters[item['body_sha256']]) > 1:
            rejected.append({**item, 'reason': 'shared_body_cluster_crosses_split'})
        else:
            bags[item['match_id']].append(item)
    panel = []
    for mid in sorted(bags, key=lambda k: (matches[k]['match_date'], k)):
        candidates = sorted(bags[mid], key=lambda x: (articles[x['article_id']]['published_at'], x['article_id']))
        panel.append({'match_id': mid, 'date': matches[mid]['match_date'],
                      'split': split_name(matches[mid]['match_date']), 'candidates': candidates})
    return panel, rejected


def loss_each(probabilities, targets):
    p, y = np.asarray(probabilities), np.asarray(targets)
    if p.shape != (len(y), 3) or np.any(y < 0) or np.any(y > 2):
        raise ValueError('Invalid probability/target dimensions')
    return -np.log(np.maximum(p[np.arange(len(y)), y], 1e-15))


def _unpack(theta, d, hidden):
    p = d * hidden
    w1, b1 = theta[:p].reshape(d, hidden), theta[p:p+hidden]
    q = p+hidden
    w2, b2 = theta[q:q+hidden*3].reshape(hidden, 3), theta[q+hidden*3:]
    if b2.shape != (3,):
        raise ValueError('Invalid parameter length')
    return w1, b1, w2, b2


def objective(theta, x, base, y, weights, gate, hidden, alpha):
    w1, b1, w2, b2 = _unpack(theta, x.shape[1], hidden)
    h = np.tanh(x @ w1 + b1)
    logits = np.log(base) + gate[:, None] * (h @ w2 + b2)
    loss = weights @ (logsumexp(logits, axis=1) - logits[np.arange(len(y)), y])
    loss += alpha/2 * (np.sum(w1*w1) + np.sum(w2*w2) + np.sum(b1*b1) + np.sum(b2*b2))
    grad = softmax(logits, axis=1)
    grad[np.arange(len(y)), y] -= 1
    grad *= (weights * gate)[:, None]
    gh = (grad @ w2.T) * (1-h*h)
    arrays = [x.T @ gh + alpha*w1, gh.sum(0)+alpha*b1,
              h.T @ grad + alpha*w2, grad.sum(0)+alpha*b2]
    return float(loss), np.concatenate([a.ravel() for a in arrays])


def fit_residual(x, base, y, weights, *, alpha, seed=20261020, hidden=16):
    x, base, y = np.asarray(x, dtype=float), np.asarray(base, dtype=float), np.asarray(y, dtype=int)
    weights = np.asarray(weights, dtype=float)
    if (not np.isfinite(x).all() or not np.isfinite(base).all() or np.any(base <= 0)
            or not np.allclose(base.sum(1), 1) or not np.isclose(weights.sum(), 1)):
        raise ValueError('Invalid training inputs')
    # Fitted on training evidence subsets only, never validation/test vectors.
    mean = weights @ x
    scale = np.maximum(np.sqrt(weights @ ((x-mean)**2)), .05)
    z = (x-mean)/scale
    rng = np.random.default_rng(seed)
    theta = np.r_[rng.normal(0, .05/np.sqrt(x.shape[1]), (x.shape[1], hidden)).ravel(),
                  np.zeros(hidden), np.zeros(hidden*3), np.zeros(3)]
    result = minimize(objective, theta, args=(z, base, y, weights, np.ones(len(y)), hidden, alpha),
                      jac=True, method='L-BFGS-B', options={'maxiter': 1500, 'ftol': 1e-11, 'gtol': 1e-6})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError('Residual optimization did not converge: '+str(result.message))
    return {'schema': SCHEMA, 'input_dimension': x.shape[1], 'hidden': hidden,
            'parameters': result.x.tolist(), 'mean': mean.tolist(), 'scale': scale.tolist(),
            'parameter_count': len(result.x), 'alpha': alpha, 'seed': seed,
            'optimizer': {'success': bool(result.success), 'iterations': int(result.nit), 'objective': float(result.fun)},
            'production_ready': False, 'historical_version_unknown': True}


def predict_residual(model, x, base, read_counts):
    if model['schema'] != SCHEMA:
        raise ValueError('Unknown model schema')
    x, base = np.asarray(x, dtype=float), np.asarray(base, dtype=float)
    counts = np.asarray(read_counts)
    if (x.shape != (len(base), model['input_dimension']) or base.shape[1:] != (3,)
            or counts.shape != (len(base),) or np.any(counts < 0)
            or not np.isfinite(x).all() or not np.isfinite(base).all()
            or np.any(base <= 0) or not np.allclose(base.sum(1), 1)):
        raise ValueError('Invalid inference inputs')
    w1, b1, w2, b2 = _unpack(np.array(model['parameters']), x.shape[1], model['hidden'])
    h = np.tanh(((x-np.array(model['mean']))/np.array(model['scale'])) @ w1 + b1)
    p = softmax(np.log(base) + (counts > 0)[:, None]*(h @ w2 + b2), axis=1)
    p[counts == 0] = base[counts == 0]  # Exact identity, including serialization/replay.
    return p


class FrozenCorpusTextPredictor:
    """Load a research candidate for explicit read-subset f evaluations.

    Does not load labels, optimize, fetch, or change production selection gates.
    Restricted to the archived panel whose match/article associations were frozen.
    """
    def __init__(self, corpus_root, experiment_dir, branch='body'):
        from football_research.data_adapters.e_drive_corpus import ExternalFootballCorpus
        self.root, self.directory = Path(corpus_root), Path(experiment_dir)
        if branch not in ('metadata', 'title', 'body'):
            raise ValueError('Unknown evidence branch')
        self.branch = branch
        base_bytes = (self.directory/'f-base.json').read_bytes()
        model_bytes = (self.directory/('f-'+branch+'.json')).read_bytes()
        self.base_model, self.model = json.loads(base_bytes), json.loads(model_bytes)
        self.model_sha256 = hashlib.sha256(model_bytes).hexdigest()
        if self.model['base_model_sha256'] != hashlib.sha256(base_bytes).hexdigest():
            raise ValueError('Frozen base model digest mismatch')
        self.corpus = ExternalFootballCorpus(corpus_root)
        self.matches = {m['match_id']: m for m in self.corpus.rows('structured/matches.jsonl')}
        self.positions = {r['match_id']: r['row'] for r in self.corpus.rows('vectors/prematch_numeric.rows.jsonl')}
        self.numeric = np.load(self.root/'vectors/prematch_numeric.npy', mmap_mode='r')
        self.panel = {p['match_id']: p for p in
                      (json.loads(line) for line in (self.directory/'panel.jsonl').read_text(encoding='utf-8').splitlines())}

    def predict(self, match_id, read_article_ids):
        panel = self.panel[match_id]
        ids = list(read_article_ids)
        allowed = {c['article_id'] for c in panel['candidates']}
        if len(ids) != len(set(ids)) or not set(ids) <= allowed:
            raise ValueError('Duplicate or out-of-panel article')
        match = self.matches[match_id]
        base = self.base_model
        numeric = (self.numeric[self.positions[match_id]].astype(float)-np.array(base['mean']))/np.array(base['scale'])
        prior = softmax(numeric@np.array(base['coef']).T+np.array(base['intercept']))
        selected = [c for c in panel['candidates'] if c['article_id'] in ids]
        sides = [[c for c in selected if self.corpus.articles[c['article_id']]['source_team'] == match[side]]
                 for side in ('home_team', 'away_team')]
        vec = list(numeric)+[float(len(side)) for side in sides]
        vec += [float(np.mean([c['days_before_match_date'] for c in side])) if side else 0. for side in sides]
        vec += [float(sum(self.corpus.articles[c['article_id']]['host'] == host for c in selected))
                for host in self.model['host_order']]
        if self.branch != 'metadata':
            for side in sides:
                vs = []
                for candidate in side:
                    aid = candidate['article_id']
                    vs.append(self.corpus.read_evidence(aid, already_read_ids=set(ids))['body_vector']
                              if self.branch == 'body' else self.corpus.unread_candidate(aid)['title_vector'])
                vec.extend(np.mean(vs, axis=0).tolist() if vs else [0.]*384)
        after = predict_residual(self.model, np.array(vec)[None, :], prior[None, :], [len(ids)])[0]
        return {'match_id': match_id, 'base_probabilities': prior.tolist(), 'probabilities': after.tolist(),
                'used_input_ids': ids, 'f_model_sha256': self.model_sha256,
                'branch': self.branch, 'production_ready': False, 'provenance': 'historical_reconstruction'}
