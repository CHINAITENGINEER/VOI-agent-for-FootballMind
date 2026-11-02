"""Regularized conditional signed gain on verified, pre-query context inputs.

This reuses the earlier ridge estimator's regression idea, not its exact
structured-action cells or descriptive confidence intervals. No Bayes EVSI,
unread text, automatic topic utility, or zero-for-unknown convention is used.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
from hashlib import sha256
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .context_gain import VERSION as GAIN_VERSION, _digest, _read, _write
from .context_predictor import ContextPredictor, _grams, NUMERIC
from .state import parse_time
from .text_returns import _p

VERSION = 'visible-context-signed-ridge-value-v1'
PARAMETERS = {'regularization': 20., 'max_features': 256, 'min_physical_matches': 12,
              'min_dates': 2, 'min_text_cosine': .2,
              'interaction_size': 128, 'interaction_seed': 'footballmind-context-query-196-v1'}
UNITS = ('money', 'requests', 'latency_ms', 'tokens')


@lru_cache(maxsize=16)
def _sketch_coordinates(role, size):
    """Fixed coordinate hashes; no labels, fitting, or Python hash randomization."""
    raw = [sha256((PARAMETERS['interaction_seed']+':'+role+':'+str(i)).encode()).digest()
           for i in range(size)]
    return (np.asarray([int.from_bytes(h[:8], 'big') % PARAMETERS['interaction_size'] for h in raw]),
            np.asarray([1. if h[8] & 1 else -1. for h in raw]))


def _interaction(state, candidate):
    """128 fixed bilinear coordinates via CountSketch circular convolution.

    Separate L2 normalization bounds the two inputs before their outer product.
    The hash is fixed, shared across fixtures, and frozen in PARAMETERS. This
    gives state/candidate interactions without fitting a full quadratic layer.
    """
    sketches = []
    for role, vector in (('state', state), ('candidate', candidate)):
        norm = np.linalg.norm(vector)
        normalized = vector/norm if norm else vector
        buckets, signs = _sketch_coordinates(role, len(vector))
        sketches.append(np.bincount(buckets, weights=normalized*signs,
                                     minlength=PARAMETERS['interaction_size']))
    return np.fft.ifft(np.fft.fft(sketches[0])*np.fft.fft(sketches[1])).real


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('A finite numerical value is required')
    return float(value)


def _bounded(value):
    value = _finite(value)
    if value < 0:
        raise ValueError('Negative time or cost is invalid')
    return math.log1p(value)/(1.+math.log1p(value))


def _public(inputs):
    """Whitelist solely the R195 pre-query input; extra fields are not read."""
    if inputs.get('schema') != GAIN_VERSION or inputs.get('numeric_names') != list(NUMERIC):
        raise ValueError('A versioned context-gain query input is required')
    state, candidate = inputs['state'], inputs['candidate']
    decision, cutoff, kickoff = (parse_time(state[k]) for k in ('decision_at', 'prediction_cutoff', 'kickoff_at'))
    if not decision <= cutoff < kickoff:
        raise ValueError('Invalid pre-query decision clocks')
    texts = state['texts']
    sources = state['sources']
    if (not isinstance(texts, list) or any(not isinstance(t, str) for t in texts)
            or not isinstance(sources, list) or any(not isinstance(s, str) or not s for s in sources)):
        raise ValueError('Already-visible text and sources are required')
    numeric = [_finite(x) for x in state['numeric']]
    if len(numeric) != len(NUMERIC) or any(not 0 <= x <= 1 for x in numeric):
        raise ValueError('Visible state summaries must match their bounded schema')
    p, prior = _p(inputs['prediction_before']), _p(state['prior'])
    request = candidate['request']
    if request['tool'] == 'read_url':
        host = urlsplit(request['url']).hostname
        if not host or urlsplit(request['url']).scheme not in {'http', 'https'}:
            raise ValueError('An observed HTTP capability is required')
        query = []
    elif request['tool'] == 'search_web':
        host = request.get('source', 'open_web')
        if not isinstance(request['query'], str) or not request['query'].strip() or not isinstance(host, str):
            raise ValueError('A visible search request is required')
        query = [request['query']]
    else:
        raise ValueError('Unsupported query tool')
    for observed in candidate['observed_parents']:
        parent = observed['parent']
        if not (parse_time(parent['observed_at']) <= parse_time(parent['available_at']) <= decision):
            raise ValueError('Parent context was not available before the query')
        if not parent.get('document_id') or len(parent.get('raw_sha256', '')) != 64:
            raise ValueError('Parent text requires its observed source binding')
        for label in observed['labels']:
            if not isinstance(label['text'], str):
                raise ValueError('Candidate labels must be already-observed text')
            query.append(label['text'])
    cost = candidate['declared_cost']
    if candidate['cost_status'] not in {'known', 'estimated', 'simulated'} or not {'money', 'requests'} <= set(cost):
        raise ValueError('Declared acquisition cost required')
    numeric += p+prior+[_bounded((kickoff-decision).total_seconds()/3600.)]
    numeric += [_bounded(cost.get(k, 0)) for k in UNITS]
    texts, query = sorted(set(t for t in texts if t.strip())), sorted(set(t for t in query if t.strip()))
    return {'texts': texts, 'labels': query, 'numeric': numeric, 'sources': sorted(set(sources)),
        'channel': request['tool']+'|'+host.lower(), 'mask': [bool(texts), bool(query)],
        'decision_at': decision.isoformat(), 'kickoff_at': kickoff.isoformat(), 'date': kickoff.date().isoformat()}


def _bundles(paths, *, predictor_path, as_of):
    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    at = parse_time(as_of)
    model = ContextPredictor.load(predictor_path) if predictor_path is not None else None
    model_sha = sha256(Path(predictor_path).read_bytes()).hexdigest() if model else None
    rows, origins, bindings = {}, set(), []
    for raw in paths:
        folder = Path(raw)
        report = _read(folder/'report.json')
        if report.get('schema') != GAIN_VERSION or parse_time(report['as_of']) > at:
            raise ValueError('A completed context-gain supervision artifact is required')
        origins.add(report['evidence_origin'])
        features, targets = _read(folder/'features.json'), _read(folder/'targets.json')
        source = _read(folder/'source-inputs.json')
        registrations = _read(folder/'calibration-registrations.json')
        if (_digest(features) != report['features_sha256'] or _digest(targets) != report['targets_sha256']
                or sha256((folder/'source-inputs.json').read_bytes()).hexdigest() != report['source_inputs_file_sha256']
                or sha256((folder/'calibration-registrations.json').read_bytes()).hexdigest() != report['registrations_file_sha256']):
            raise ValueError('Supervision input/target/source binding differs')
        for path, expected in source.get('input_bindings', {}).items():
            if sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise ValueError('Original supervision source bytes changed')
        if len(features) != len(targets) or len({r['sample_id'] for r in features}) != len(features):
            raise ValueError('Query inputs and signed targets must have distinct paired identities')
        by_id = {r['sample_id']: r for r in features}
        if set(by_id) != {r['sample_id'] for r in targets}:
            raise ValueError('Query inputs do not pair with all signed targets')
        if targets and (model is None or report['predictor_sha256'] != model_sha
                or report['predictor_model_version'] != model.model_version
                or report['evidence_origin'] != model.metadata['evidence_origin']):
            raise ValueError('Every signed target must use the same actual frozen f artifact and origin')
        if targets and report['evidence_origin'] == 'real_observational':
            # Never accept a caller re-signing labels, gains or after forecasts.
            from .context_gain import build_context_gain, build_context_gain_from_runs
            with TemporaryDirectory(prefix='fmgfit-') as tmp:
                destination = Path(tmp)/'checked'
                if source['schema'] == 'raw-context-gain-inputs-v1':
                    build_context_gain_from_runs(source['run_folders'], source['result_archive'], destination,
                        predictor_path=predictor_path, registrations=registrations,
                        previous_plans=source['previous_plans'], as_of=report['as_of'])
                elif source.get('source_corpus_directory'):
                    build_context_gain(source['source_corpus_directory'], destination, predictor_path=predictor_path,
                                       registrations=registrations, as_of=report['as_of'])
                else:
                    raise ValueError('Real supervision needs its reconstructable raw gateway')
                if features != _read(destination/'features.json') or targets != _read(destination/'targets.json'):
                    raise ValueError('Signed targets differ from same-f raw replay')
        for target in targets:
            ident, physical, y = target['sample_id'], target['physical_match_id'], target['outcome']
            clean = _public(by_id[ident]['inputs'])
            p, q = _p(target['prediction_before']), _p(target['prediction_after'])
            if (type(y) is not int or y not in (0, 1, 2) or p != _p(by_id[ident]['inputs']['prediction_before'])
                    or abs(_finite(target['signed_gain'])-(math.log(q[y])-math.log(p[y]))) > 1e-12
                    or not parse_time(clean['kickoff_at']) <= parse_time(target['result_available_at']) <= at
                    or parse_time(clean['decision_at']) <= parse_time(model.metadata['training_as_of'])
                    or physical in model.metadata['training_physical_match_ids']):
                raise ValueError('Signed gain, label clock or held-out physical identity is invalid')
            row = {'sample_id': ident, 'physical_match_id': physical, 'gain': target['signed_gain'],
                   'outcome': y, 'result_available_at': target['result_available_at'], 'context': clean}
            if ident in rows and rows[ident] != row:
                raise ValueError('Conflicting repeated supervision sample')
            rows[ident] = row
        bindings.append({'folder': str(folder.resolve()), 'report_sha256': sha256((folder/'report.json').read_bytes()).hexdigest(),
                         'features_sha256': report['features_sha256'], 'targets_sha256': report['targets_sha256']})
    if len(origins) > 1 or origins-set({'real_observational', 'constructed_mechanism'}):
        raise ValueError('One explicit supervision origin is required')
    records = [rows[k] for k in sorted(rows)]
    facts = {}
    for row in records:
        fact = (row['outcome'], row['context']['kickoff_at'])
        if row['physical_match_id'] in facts and facts[row['physical_match_id']] != fact:
            raise ValueError('Conflicting same-physical-match supervision facts')
        facts[row['physical_match_id']] = fact
    dates = {r['context']['date'] for r in records}
    reasons = []
    if len(facts) < PARAMETERS['min_physical_matches']:
        reasons.append('fewer_than_12_independent_B_matches')
    if len(dates) < PARAMETERS['min_dates']:
        reasons.append('fewer_than_2_B_kickoff_dates')
    if model is None:
        reasons.append('frozen_context_predictor_missing')
    ready = {'ready_to_fit': not reasons, 'reasons': reasons, 'physical_matches': len(facts),
        'kickoff_dates': len(dates), 'transitions': len(records),
        'evidence_origin': next(iter(origins), None),
        'gate_scope': 'necessary_training_gate_local_support_checked_separately_no_effect_claim'}
    return records, model, model_sha, bindings, ready


def training_readiness(gain_dirs, *, predictor_path=None, as_of):
    return _bundles(gain_dirs, predictor_path=predictor_path, as_of=as_of)[-1]


class ContextValueModel:
    schema_version = VERSION
    readiness = staticmethod(training_readiness)

    def __init__(self):
        self.metadata, self.encoder, self.coefficients, self.support_records = {}, {}, [], []
        self.model_version = ''

    def _lexical(self, texts):
        counts = Counter(_grams(texts))
        vector = np.asarray([(1.+math.log(counts[token])) if counts[token] else 0. for token in self.encoder['vocabulary']])
        vector *= np.asarray(self.encoder['idf'])
        norm = np.linalg.norm(vector)
        return vector/norm if norm else vector

    def _vector(self, context):
        a, b = self._lexical(context['texts']), self._lexical(context['labels'])
        numeric = np.asarray(context['numeric'])/math.sqrt(len(context['numeric']))
        channel = np.asarray([float(context['channel'] == key) for key in self.encoder['channels']])
        sources = np.asarray([float(key in context['sources']) for key in self.encoder['sources']])
        if np.linalg.norm(sources):
            sources /= np.linalg.norm(sources)
        # The original additive model cancels a common state when comparing
        # two actions. These fixed interactions permit the preferred action
        # to change with the already-visible state, including within a host.
        state = np.concatenate([a, numeric[:-len(UNITS)], sources])
        candidate = np.concatenate([b, channel, numeric[-len(UNITS):]])
        interactions = _interaction(state, candidate)
        return np.concatenate([[1.], a, b, numeric, channel, sources, interactions]), np.concatenate([a, b])

    def _fit_regression(self, rows):
        """Shared fixed ridge kernel; input contracts stay in each model version."""
        grouped = defaultdict(set)
        for row in rows:
            grouped[row['physical_match_id']].update(row['context']['texts']+row['context']['labels'])
        corpus = [sorted(grouped[mid]) for mid in sorted(grouped)]
        vocabulary, idf = [], []
        if any(any(_grams(text)) for text in corpus):
            vectorizer = TfidfVectorizer(analyzer=_grams, max_features=PARAMETERS['max_features'], sublinear_tf=True)
            vectorizer.fit(corpus)
            vocabulary, idf = vectorizer.get_feature_names_out().tolist(), vectorizer.idf_.tolist()
        self.encoder = {'vocabulary': vocabulary, 'idf': idf, 'numeric_size': len(rows[0]['context']['numeric']),
            'channels': sorted({r['context']['channel'] for r in rows}),
            'sources': sorted({s for r in rows for s in r['context']['sources']})}
        counts = Counter(r['physical_match_id'] for r in rows)
        weights = np.asarray([1./(len(counts)*counts[r['physical_match_id']]) for r in rows])
        vectors = [self._vector(row['context']) for row in rows]
        x, y = np.asarray([r[0] for r in vectors]), np.asarray([r['gain'] for r in rows])
        penalty = np.eye(x.shape[1])*PARAMETERS['regularization']
        penalty[0, 0] = 0.
        beta = np.linalg.solve(x.T@(x*weights[:, None])+penalty, x.T@(y*weights))
        self.coefficients = beta.tolist()
        self.support_records = [{'physical_match_id': row['physical_match_id'], 'date': row['context']['date'],
            'channel': row['context']['channel'], 'mask': row['context']['mask'], 'lexical': vector[1].tolist()}
            for row, vector in zip(rows, vectors)]
        return counts, float(np.sum(weights*(y-x@beta)**2)+beta@penalty@beta)

    def fit(self, gain_dirs, *, predictor_path, as_of, dataset_version):
        rows, model, digest, bindings, ready = _bundles(gain_dirs, predictor_path=predictor_path, as_of=as_of)
        if not ready['ready_to_fit']:
            raise ValueError('Context value not_ready: '+', '.join(ready['reasons']))
        if not isinstance(dataset_version, str) or not dataset_version.strip():
            raise ValueError('Explicit supervised dataset version required')
        execution_at = datetime.now(timezone.utc)
        if parse_time(as_of) > execution_at:
            raise ValueError('The calibration cutoff cannot follow actual fitting execution')
        counts, objective = self._fit_regression(rows)
        self.metadata = {'training_as_of': parse_time(as_of).isoformat(),
            'fit_execution_at': execution_at.isoformat(), 'dataset_version': dataset_version,
            'training_physical_match_ids': sorted(counts), 'predictor_sha256': digest,
            'predictor_model_version': model.model_version,
            'predictor_training_physical_match_ids': model.metadata['training_physical_match_ids'],
            'evidence_origin': ready['evidence_origin'], 'readiness': ready, 'input_bindings': bindings,
            'training_rows_sha256': _digest(rows), 'negative_targets': sum(r['gain'] < -1e-12 for r in rows),
            'target': 'expected_signed_fixed_f_log_loss_reduction_before_cost',
            'weighting': 'equal_physical_match_then_equal_distinct_actual_transition',
            'encoding': 'B_only_char_3_5_TFIDF_state_and_candidate_labels_separate_no_URL_identity',
            'interaction': 'fixed_128_signed_TensorSketch_normalized_state_by_candidate_outer_product',
            'support_scope': 'same_tool_source_and_text_presence_with_shared_text_neighborhood_not_calibration',
            'uncertainty_kind': 'not_estimated_no_confidence_interval_or_coverage_guarantee',
            'production_ready': False, 'calibration_status': 'not_established',
            'objective': objective}
        self.model_version = VERSION+':'+_digest(self._payload())[:24]
        return self

    def predict(self, inputs, *, physical_match_id, predictor_sha256):
        self._check()
        if predictor_sha256 != self.metadata['predictor_sha256']:
            raise ValueError('Conditional value belongs to a different frozen f artifact')
        if not isinstance(physical_match_id, str) or not physical_match_id:
            raise ValueError('Explicit canonical physical prediction identity required')
        if physical_match_id in self.metadata['training_physical_match_ids']+self.metadata['predictor_training_physical_match_ids']:
            raise ValueError('Prediction fixture belongs to f/g training identities')
        context = _public(inputs)
        if parse_time(context['decision_at']) <= parse_time(self.metadata['training_as_of']):
            raise ValueError('Value prediction must follow calibration training cutoff')
        if parse_time(context['decision_at']) < parse_time(self.metadata['fit_execution_at']):
            raise ValueError('Value prediction precedes actual fitting execution')
        vector, reason, nearby = self._support(context)
        supported = reason == 'supported_empirical_context'
        gain = float(vector@np.asarray(self.coefficients)) if supported else None
        return {'schema': VERSION, 'model_version': self.model_version,
            'expected_signed_gain': gain, 'expected_gain': gain, 'support_status': 'supported' if supported else 'UNKNOWN',
            'reason': reason, 'evidence_origin': self.metadata['evidence_origin'],
            'support': {'physical_matches': len({r['physical_match_id'] for r in nearby}),
                'kickoff_dates': len({r['date'] for r in nearby}), 'channel': context['channel'],
                'minimum_matches': PARAMETERS['min_physical_matches'], 'minimum_dates': PARAMETERS['min_dates'],
                'text_cosine_floor': PARAMETERS['min_text_cosine'], 'scope': self.metadata['support_scope']},
            'value_kind': 'empirical_signed_fixed_predictor_gain_not_Bayes_EVSI',
            'cost_deducted': False, 'uncertainty': None, 'stop_certificate': None,
            'calibration_status': 'not_established', 'production_ready': False}

    def _support(self, context):
        """Shared source/mask/text neighborhood with unchanged absolute gates."""
        vector, lexical = self._vector(context)
        reason, nearby = 'supported_empirical_context', []
        if context['channel'] not in self.encoder['channels']:
            reason = 'unseen_tool_or_request_source'
        elif set(context['sources'])-set(self.encoder['sources']):
            reason = 'unseen_visible_source'
        elif any(present and not np.linalg.norm(block) for present, block in
                 zip(context['mask'], np.split(lexical, 2))):
            reason = 'visible_text_out_of_vocabulary'
        else:
            for row in self.support_records:
                if row['channel'] != context['channel'] or row['mask'] != context['mask']:
                    continue
                historical = np.asarray(row['lexical'])
                similarity = float(lexical@historical/(np.linalg.norm(lexical)*np.linalg.norm(historical))) if np.linalg.norm(lexical) and np.linalg.norm(historical) else 1.
                if similarity >= PARAMETERS['min_text_cosine']:
                    nearby.append(row)
            if len({r['physical_match_id'] for r in nearby}) < PARAMETERS['min_physical_matches']:
                reason = 'insufficient_independent_local_matches'
            elif len({r['date'] for r in nearby}) < PARAMETERS['min_dates']:
                reason = 'insufficient_local_kickoff_dates'
        return vector, reason, nearby

    def _payload(self):
        return {'schema': VERSION, 'parameters': PARAMETERS, 'metadata': self.metadata, 'encoder': self.encoder,
                'coefficients': self.coefficients, 'support_records': self.support_records}

    def _check(self):
        if not self.model_version or self.model_version != VERSION+':'+_digest(self._payload())[:24]:
            raise ValueError('Frozen context value content/version mismatch')

    def save(self, path):
        self._check()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        _write(path, {**self._payload(), 'model_version': self.model_version})

    @classmethod
    def load(cls, path, *, predictor_path):
        payload = _read(path)
        if payload.get('schema') != VERSION or payload.get('parameters') != PARAMETERS:
            raise ValueError('Unsupported context value schema or fixed parameters')
        model = cls()
        for name in ('metadata', 'encoder', 'coefficients', 'support_records', 'model_version'):
            setattr(model, name, payload[name])
        model._check()
        f = ContextPredictor.load(predictor_path)
        if (sha256(Path(predictor_path).read_bytes()).hexdigest() != model.metadata['predictor_sha256']
                or f.model_version != model.metadata['predictor_model_version']
                or f.metadata['evidence_origin'] != model.metadata['evidence_origin']):
            raise ValueError('Frozen context value belongs to another f artifact or origin')
        enc = model.encoder
        size = (1+2*len(enc['vocabulary'])+enc['numeric_size']+len(enc['channels'])+len(enc['sources'])
                +PARAMETERS['interaction_size'])
        if (len(model.coefficients) != size or not np.all(np.isfinite(model.coefficients))
                or enc['vocabulary'] != sorted(set(enc['vocabulary'])) or len(enc['vocabulary']) > 256
                or len(enc['idf']) != len(enc['vocabulary']) or any(not math.isfinite(x) or x < 1 for x in enc['idf'])
                or len(set(model.metadata['training_physical_match_ids'])) < 12
                or any(len(row['lexical']) != 2*len(enc['vocabulary']) for row in model.support_records)):
            raise ValueError('Invalid frozen shared encoder or independent support')
        return model


__all__ = ['ContextValueModel', 'training_readiness', 'VERSION', 'PARAMETERS']
