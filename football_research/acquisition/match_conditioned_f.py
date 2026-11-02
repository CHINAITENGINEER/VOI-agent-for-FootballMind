"""Offline, match-conditioned f; no query-value estimation or live promotion.

Reuse the existing Poisson and multinomial optimization kernels. Match priors
are fitted strictly before a reconstructed prediction time. Text residuals
separate observed titles from acquired bodies and are selected by forward
physical-match validation, never by the number of enumerated reveal states.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer

from football_research.baselines.models import PoissonModel
from .context_predictor import _digest, _grams
from .fixture_roles import ROLES, _mentions
from .historical_context_training import _project
from .historical_predictor_selection import _optimize
from .news_link_projection import VERSION as NEWS_PROJECTION
from .state import parse_time

VERSION = 'match-conditioned-separate-reveal-f-v1'
PRIOR_VERSION = 'reconstructed-asof-poisson-prior-v1'
PARAMETERS = {'regularizations': [None, 10., 1., .1, .01],
              'max_features': 128, 'warmup_matches': 6,
              'minimum_fold_matches': 3, 'validation_block_matches': 3,
              'selection_rule': 'simplest_within_one_paired_standard_error_of_best',
              'prior_regularization': 5., 'text_projection': NEWS_PROJECTION}


class ReconstructedMatchPrior:
    """An explicitly retrospective source; never a historical live snapshot."""
    def __init__(self, rows, *, source_binding):
        self.rows, self.source_binding = deepcopy(rows), deepcopy(source_binding)
        self._cache = {}
        ids = [str(r['source'])+':'+str(r['source_id']) for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate physical match in structured history')
        for row in rows:
            if (row.get('information_mode') != 'historical_reconstruction'
                    or row.get('score_period') != 'regulation'
                    or row.get('status') != 'finished'
                    or any(type(row.get(k)) is not int or row[k] < 0
                           for k in ('home_goals', 'away_goals'))
                    or parse_time(row['result_available_at']) < parse_time(row['kickoff'])):
                raise ValueError('Validated reconstructed regulation results required')
        self.identities = dict(zip(ids, self.rows))

    @classmethod
    def from_export(cls, folder):
        folder = Path(folder)
        manifest_path = folder/'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('information_mode') != 'historical_reconstruction':
            raise ValueError('Explicit historical reconstruction export required')
        entry = manifest['files']['matches']
        path = (folder/entry['path']).resolve()
        if not path.is_relative_to(folder.resolve()):
            raise ValueError('Result archive must be inside the declared export')
        raw = path.read_bytes()
        if sha256(raw).hexdigest() != entry['sha256']:
            raise ValueError('Structured result archive digest mismatch')
        rows = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
        if len(rows) != entry['row_count']:
            raise ValueError('Structured result archive count mismatch')
        return cls(rows, source_binding={'path': str(path), 'sha256': entry['sha256'],
            'manifest_path': str(manifest_path.resolve()),
            'manifest_sha256': sha256(manifest_path.read_bytes()).hexdigest(),
            'assumptions': manifest.get('assumptions'),
            'actual_source_observed_at': manifest.get('source_observed_at')})

    def snapshot(self, fixture, as_of):
        cutoff = parse_time(as_of)
        mid = fixture['physical_match_id']
        if cutoff >= parse_time(fixture['kickoff']):
            raise ValueError('Match prior must precede kickoff')
        target = self.identities.get(mid)
        if target is None:
            raise ValueError('An exact source identity is required; no fuzzy fixture matching')
        for role in ('home', 'away'):
            accepted = {fixture[role+'_name'], *fixture.get(role+'_aliases', [])}
            if target[role+'_team_name'] not in accepted:
                raise ValueError('Ordered team identity differs from structured archive')
        if (parse_time(target['kickoff']) != parse_time(fixture['kickoff'])
                or fixture.get('competition', target['competition_name']) != target['competition_name']):
            raise ValueError('Fixture clock or competition differs from structured archive')
        if cutoff.isoformat() not in self._cache:
            # Availability is strict; the current result never enters a fit.
            history = sorted([r for r in self.rows
                if parse_time(r['result_available_at']) < cutoff and parse_time(r['kickoff']) < cutoff],
                key=lambda r: (parse_time(r['kickoff']), str(r['source_id'])))
            if not history:
                raise ValueError('No prior matches; do not invent a fitted starting probability')
            kernel = PoissonModel(regularization=PARAMETERS['prior_regularization']).fit(history, as_of=cutoff)
            self._cache[cutoff.isoformat()] = kernel, history
        kernel, history = self._cache[cutoff.isoformat()]
        # Only identity is passed into predict_proba, never target goals.
        features = {k: target[k] for k in ('competition_id', 'home_team_id', 'away_team_id')}
        known = {r[k] for r in history for k in ('home_team_id', 'away_team_id')}
        payload = {'schema': PRIOR_VERSION, 'physical_match_id': mid,
            'fixture': deepcopy(fixture), 'as_of': cutoff.isoformat(),
            'probabilities': list(kernel.predict_proba(features)),
            'prediction_features': features, 'kernel': kernel.to_state(),
            'training_match_ids': [str(r['source'])+':'+str(r['source_id']) for r in history],
            'training_rows_sha256': _digest(history),
            'latest_result_available_at': max(parse_time(r['result_available_at']) for r in history).isoformat(),
            'unknown_team_roles': [role for role in ('home', 'away') if features[role+'_team_id'] not in known],
            'source_binding': self.source_binding, 'information_mode': 'historical_reconstruction',
            'availability_basis': 'source_reconstruction_not_historical_capture',
            'operational_eligible': False}
        return {**payload, 'snapshot_sha256': _digest(payload)}


def bind_prior(state, snapshot):
    """Bind one immutable per-match baseline throughout a reveal episode."""
    payload = {k: v for k, v in snapshot.items() if k != 'snapshot_sha256'}
    if (_digest(payload) != snapshot.get('snapshot_sha256')
            or snapshot.get('schema') != PRIOR_VERSION
            or snapshot['fixture'] != state['fixture']
            or snapshot['physical_match_id'] != state['fixture']['physical_match_id']
            or state.get('information_mode') != 'historical_reconstruction'
            or parse_time(snapshot['as_of']) > parse_time(state['decision_at'])
            or parse_time(snapshot['latest_result_available_at']) >= parse_time(snapshot['as_of'])
            or snapshot['physical_match_id'] in snapshot['training_match_ids']):
        raise ValueError('Invalid, future, or mismatched reconstructed prior snapshot')
    expected = PoissonModel.from_state(snapshot['kernel']).predict_proba(snapshot['prediction_features'])
    if not np.allclose(expected, snapshot['probabilities'], rtol=0., atol=1e-14):
        raise ValueError('Prior probabilities differ from the frozen structured model')
    return {**deepcopy(state), 'prior_probabilities': list(expected),
            'prior_basis': snapshot['schema']+':'+snapshot['snapshot_sha256']}


def project(state):
    """Validate clocks first; form title/body channels only from visible inputs.

    No source, time, document-count, or failed-query feature can change logits.
    Every document is normalized locally, so an OOV new document cannot dilute
    an earlier one. No semantic relevance or per-player effect is inferred.
    """
    checked = _project(state, text_projection=NEWS_PROJECTION)
    if checked['prior'] is None:
        raise ValueError('A match-conditioned f requires an explicit frozen prior')
    aliases = {role: checked['fixture'][role+'_aliases'] for role in ('home', 'away')}
    titles = {link['title'] for link in checked['observed_links']}
    titles.update(doc['title'] for doc in checked['documents'] if doc['title'].strip())
    channels = {'title': [], 'body': []}
    def roles(paragraphs):
        result = {r: [] for r in ROLES}
        for text in sorted(set(paragraphs)):
            if text.strip():
                result[_mentions(text, aliases)[0]].append(text)
        return result
    for title in sorted(titles):
        channels['title'].append(roles([title]))
    seen = set()
    for doc in checked['documents']:
        # Identical normalized body under a second URL is not extra evidence.
        paragraphs = doc.get('paragraphs') or doc['text'].splitlines()
        paragraphs = sorted(set(' '.join(p.split()) for p in paragraphs if p.strip()))
        key = tuple(paragraphs)
        if key not in seen:
            seen.add(key)
            channels['body'].append(roles(paragraphs))
    return {'physical_match_id': checked['physical_match_id'],
            'prior': checked['prior'], 'channels': channels,
            'decision_at': checked['decision_at']}


def _encoder(rows):
    matches = defaultdict(set)
    for row in rows:
        for blocks in row['state']['channels'].values():
            for block in blocks:
                for texts in block.values():
                    matches[row['physical_match_id']].update(texts)
        matches[row['physical_match_id']]  # Preserve empty matches in IDF.
    corpus = [sorted(matches[mid]) for mid in sorted(matches)]
    if not any(any(_grams(texts)) for texts in corpus):
        return {'vocabulary': [], 'idf': []}
    vectorizer = TfidfVectorizer(analyzer=_grams, max_features=PARAMETERS['max_features'], sublinear_tf=True)
    vectorizer.fit(corpus)
    return {'vocabulary': vectorizer.get_feature_names_out().tolist(), 'idf': vectorizer.idf_.tolist()}


def _vector(encoder, state):
    vocab, idf = encoder['vocabulary'], np.asarray(encoder['idf'])
    width = len(ROLES)*len(vocab)
    pieces = []
    for channel in ('title', 'body'):
        total = np.zeros(width)
        for block in state['channels'][channel]:
            parts = []
            for role in ROLES:
                counts = Counter(_grams(block[role]))
                parts.append(np.asarray([1.+np.log(counts[t]) if counts[t] else 0. for t in vocab])*idf)
            local = np.concatenate(parts)
            norm = np.linalg.norm(local)
            if norm:
                total += local/norm
        pieces.append(total)
    basic = np.concatenate(pieces)
    p = state['prior']
    # Conditional interaction; effect of the same acquired text may depend on
    # the match's starting balance. The baseline itself remains a log offset.
    return np.concatenate([basic, (p[0]-p[2])*basic, p[1]*basic])


def _fit(rows, regularization):
    encoder = _encoder(rows)
    x = np.asarray([_vector(encoder, row['state']) for row in rows])
    counts = Counter(r['physical_match_id'] for r in rows)
    weights = np.asarray([1./(len(counts)*counts[r['physical_match_id']]) for r in rows])
    design = (type('EncoderHolder', (), {'encoder': encoder})(), x,
              np.asarray([r['outcome'] for r in rows]),
              np.log(np.asarray([r['state']['prior'] for r in rows])), weights)
    if not x.shape[1]:
        return encoder, [], {'success': True, 'method': 'empty_vocabulary_prior_only'}
    model, fit = _optimize(design, regularization)
    return encoder, model.coefficients, fit


def _predict(encoder, coefficients, state):
    vector = _vector(encoder, state)
    residual = vector@np.asarray(coefficients) if len(vector) else np.zeros(3)
    return softmax(np.log(state['prior'])+residual)


class MatchConditionedF:
    """Versioned research candidate; intentionally not a HistoricalContextModel."""
    def __init__(self, encoder, coefficients, metadata):
        self.encoder, self.coefficients, self.metadata = deepcopy((encoder, coefficients, metadata))
        self.model_version = VERSION+':'+_digest(self._payload())[:24]

    def _payload(self):
        return {'schema': VERSION, 'parameters': PARAMETERS, 'encoder': self.encoder,
                'coefficients': self.coefficients, 'metadata': self.metadata}

    def predict(self, state, *, prior_snapshot):
        if self.model_version != VERSION+':'+_digest(self._payload())[:24]:
            raise ValueError('Frozen f was modified')
        bound = bind_prior(state, prior_snapshot)
        visible = project(bound)
        return {'probabilities': list(_predict(self.encoder, self.coefficients, visible)),
            'initial_probabilities': visible['prior'], 'model_version': self.model_version,
            'prior_snapshot_sha256': prior_snapshot['snapshot_sha256'],
            'probability_basis': VERSION, 'evidence_origin': 'historical_reconstruction',
            'gain_model_enabled': False, 'production_ready': False}

    def save(self, path):
        path = Path(path)
        with path.open('x', encoding='utf-8') as handle:
            json.dump({**self._payload(), 'model_version': self.model_version}, handle, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        raw = json.loads(Path(path).read_text(encoding='utf-8'))
        if raw.get('schema') != VERSION or raw.get('parameters') != PARAMETERS:
            raise ValueError('Unknown conditioned-f protocol')
        model = cls(raw['encoder'], raw['coefficients'], raw['metadata'])
        if model.model_version != raw.get('model_version'):
            raise ValueError('Frozen f digest mismatch')
        width = 6*len(ROLES)*len(model.encoder['vocabulary'])
        if (len(model.encoder['idf']) != len(model.encoder['vocabulary'])
                or np.asarray(model.coefficients).shape != ((width, 3) if width else (0,))
                or not np.all(np.isfinite(model.coefficients))):
            raise ValueError('Invalid conditioned-f dimensions or coefficients')
        return model


def fit_forward_f(rows, *, dataset_binding):
    """A only. Input rows carry visible projection, outcome and result clock."""
    rows = deepcopy(rows)
    facts = {}
    for row in rows:
        mid = row['physical_match_id']
        fact = (row['outcome'], row['result_available_at'], row['state']['decision_at'])
        if (type(fact[0]) is not int or fact[0] not in (0, 1, 2)
                or row['state']['physical_match_id'] != mid
                or parse_time(fact[1]) <= parse_time(fact[2])
                or (mid in facts and facts[mid] != fact)):
            raise ValueError('One consistent physical match label/decision clock required')
        facts[mid] = fact
    ids = sorted(facts, key=lambda mid: (parse_time(facts[mid][2]), mid))
    if len(ids) < PARAMETERS['warmup_matches']+PARAMETERS['validation_block_matches']:
        raise ValueError('Insufficient independent A matches for forward selection')
    folds, scores = [], [[] for _ in PARAMETERS['regularizations']]
    for start in range(PARAMETERS['warmup_matches'], len(ids), PARAMETERS['validation_block_matches']):
        valid_ids = ids[start:start+PARAMETERS['validation_block_matches']]
        first_decision = min(parse_time(facts[mid][2]) for mid in valid_ids)
        train_ids = [mid for mid in ids[:start] if parse_time(facts[mid][1]) < first_decision]
        if len(train_ids) < PARAMETERS['minimum_fold_matches'] or len({facts[mid][0] for mid in train_ids}) < 2:
            raise ValueError('Insufficient purged forward training fold')
        training = [r for r in rows if r['physical_match_id'] in train_ids]
        hypotheses = []
        for i, lam in enumerate(PARAMETERS['regularizations']):
            encoder, coef, optimizer = _fit(training, lam)
            losses = []
            for mid in valid_ids:
                test = [r for r in rows if r['physical_match_id'] == mid]
                loss = float(np.mean([-np.log(_predict(encoder, coef, r['state'])[r['outcome']]) for r in test]))
                scores[i].append(loss)
                losses.append({'physical_match_id': mid, 'log_loss': loss})
            hypotheses.append({'regularization': lam, 'match_losses': losses,
                'encoder_sha256': _digest(encoder), 'coefficients_sha256': _digest(coef), 'optimizer': optimizer})
        folds.append({'training_ids': train_ids, 'validation_ids': valid_ids,
            'first_validation_decision': first_decision.isoformat(),
            'latest_training_result': max(parse_time(facts[mid][1]) for mid in train_ids).isoformat(),
            'hypotheses': hypotheses})
    matrix = np.asarray(scores)
    means = matrix.mean(axis=1)
    best = int(np.argmin(means))
    differences = matrix-matrix[best]
    se = differences.std(axis=1, ddof=1)/np.sqrt(matrix.shape[1])
    selected = next(i for i in range(len(means)) if means[i] <= means[best]+se[i]+1e-12)
    lam = PARAMETERS['regularizations'][selected]
    encoder, coef, optimizer = _fit(rows, lam)
    selection = {'folds': folds, 'independent_validation_matches': matrix.shape[1],
        'candidates': [{'regularization': value, 'mean_match_log_loss': float(means[i]),
            'paired_standard_error_to_best': float(se[i])} for i, value in enumerate(PARAMETERS['regularizations'])],
        'best_mean_index': best, 'selected_index': selected, 'selected_regularization': lam,
        'base_only_selected': lam is None, 'selection_rule': PARAMETERS['selection_rule'],
        'standard_error_scope': 'small_dependent_time_series_diagnostic_not_confidence_bound'}
    metadata = {'fit_execution_at': datetime.now(timezone.utc).isoformat(),
        'training_match_ids': ids, 'training_states': len(rows),
        'training_rows_sha256': _digest(rows), 'dataset_binding': dataset_binding,
        'selection': selection, 'optimizer': optimizer, 'evidence_origin': 'historical_reconstruction',
        'fit_scope': 'A_only_forward_purged_selection_equal_match_weights',
        'body_features': 'additive_per_document_normalized_role_tfidf_separate_from_observed_titles',
        'uncertainty': 'not_calibrated', 'production_ready': False, 'gain_model_enabled': False}
    return MatchConditionedF(encoder, coef, metadata)
