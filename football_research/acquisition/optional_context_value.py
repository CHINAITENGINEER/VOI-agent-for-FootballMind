"""Signed query gain with an explicit absent prior and frozen learned base.

The numerical kernel is the existing context ridge model. This version adds
only the optional-prior input/provenance contract and independent raw B gate.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .context_gain import _digest, _read, _write, VERSION as LEGACY_GAIN_VERSION
from .context_predictor import NUMERIC
from .context_value import ContextValueModel, PARAMETERS, UNITS, _public as _legacy_public, _finite
from .optional_context_query import GAIN_VERSION, validate_forecast
from .optional_prior_predictor import OptionalPriorContextPredictor
from .state import parse_time
from .text_returns import _p

VERSION = 'official-optional-context-signed-ridge-value-v1'


def _public(inputs, *, predictor=None, gain_version=GAIN_VERSION, forecast_validator=validate_forecast):
    if inputs.get('schema') != gain_version or inputs.get('numeric_names') != list(NUMERIC):
        raise ValueError('An optional-prior pre-query input is required')
    state = inputs['state']
    forecast = forecast_validator(inputs['forecast'], state=state, predictor=predictor)
    missing = state['prior'] is None
    if (type(inputs.get('prior_missing')) is not bool or inputs['prior_missing'] != missing
            or _p(inputs['prediction_before']) != _p(forecast['probabilities'])):
        raise ValueError('Query prediction or missing-prior mask differs from the original forecast')
    # Reuse strict clocks, source, candidate-label and cost validation. The
    # numeric feature slot explicitly means forecast BASE in this new schema;
    # the original input/Actor state is neither changed nor called a prior.
    projection = deepcopy(inputs)
    projection['schema'] = LEGACY_GAIN_VERSION
    projection['state']['prior'] = list(forecast['base_probabilities'])
    context = _legacy_public(projection)
    context['numeric'] = context['numeric'][:-len(UNITS)]+[float(missing)]+context['numeric'][-len(UNITS):]
    context['mask'].append(missing)
    context.update(prior_domain=forecast['prior_domain'],
        predictor_context_supported=forecast['prior_domain_context_supported'])
    return context


def _bundles(paths, *, predictor_path, as_of, gain_version=GAIN_VERSION,
             predictor_type=OptionalPriorContextPredictor, public_validator=_public,
             source_version='raw-official-context-gain-inputs-v1', raw_builder=None,
             after_support_statuses=('base_only','shared_feature_projection','source_extrapolation')):
    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    at = parse_time(as_of)
    model = predictor_type.load(predictor_path) if predictor_path is not None else None
    model_sha = sha256(Path(predictor_path).read_bytes()).hexdigest() if model is not None else None
    rows, origins, bindings, aliases, actual_keys = {}, set(), [], {}, {}
    for raw in paths:
        folder = Path(raw)
        report = _read(folder/'report.json')
        if report.get('schema') != gain_version or parse_time(report['as_of']) > at:
            raise ValueError('A completed optional-prior gain supervision bundle is required')
        origin = report.get('evidence_origin')
        origins.add(origin)
        features, targets = _read(folder/'features.json'), _read(folder/'targets.json')
        source, registrations = _read(folder/'source-inputs.json'), _read(folder/'calibration-registrations.json')
        if (_digest(features) != report['features_sha256'] or _digest(targets) != report['targets_sha256']
                or sha256((folder/'source-inputs.json').read_bytes()).hexdigest() != report['source_inputs_file_sha256']
                or sha256((folder/'calibration-registrations.json').read_bytes()).hexdigest() != report['registrations_file_sha256']):
            raise ValueError('Optional gain input/target/source binding differs')
        for path, expected in source.get('input_bindings', {}).items():
            if sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise ValueError('Original optional gain source bytes changed')
        if len(features) != len(targets) or len({r['sample_id'] for r in features}) != len(features):
            raise ValueError('Gain inputs and targets need distinct paired sample identities')
        by_id = {row['sample_id']:row for row in features}
        if set(by_id) != {row['sample_id'] for row in targets}:
            raise ValueError('Every optional query must pair with its signed target')
        if targets and (model is None or report['predictor_sha256'] != model_sha
                or report['predictor_model_version'] != model.model_version or origin != model.metadata['evidence_origin']):
            raise ValueError('All optional targets must use the same frozen predictor and evidence origin')
        if targets and origin == 'real_observational':
            if source.get('schema') != source_version:
                raise ValueError('Real optional gains require the reconstructable original official raw gateway')
            if raw_builder is None:
                from football_research.data_adapters.official_context_gain import build_official_context_gain_from_runs
                builder = build_official_context_gain_from_runs
            else:
                builder = raw_builder
            with TemporaryDirectory(prefix='optional-gain-') as tmp:
                rebuilt = Path(tmp)/'checked'
                builder(source['run_folders'], source['result_archives'], rebuilt,
                    predictor_path=predictor_path, as_of=report['as_of'])
                if features != _read(rebuilt/'features.json') or targets != _read(rebuilt/'targets.json'):
                    raise ValueError('Optional signed targets differ from same-f original raw replay')
        for target in targets:
            ident, physical, y = target['sample_id'], target['physical_match_id'], target['outcome']
            inputs = by_id[ident]['inputs']
            context = public_validator(inputs, predictor=model)
            forecast = inputs['forecast']
            p, q = _p(target['prediction_before']), _p(target['prediction_after'])
            if not context['predictor_context_supported']:
                raise ValueError('Unsupported prior domain cannot become supported zero-gain supervision')
            if (target.get('before_domain_context_supported', True) is not True
                    or target.get('after_domain_context_supported', True) is not True
                    or target.get('before_support') != forecast['support_status']
                    or target.get('after_support') not in after_support_statuses):
                raise ValueError('Both actual same-domain forecasts require frozen context support')
            if (type(y) is not int or y not in (0,1,2) or physical != forecast['physical_match_id']
                    or p != _p(inputs['prediction_before'])
                    or abs(_finite(target['signed_gain'])-(math.log(q[y])-math.log(p[y]))) > 1e-12
                    or not parse_time(context['kickoff_at']) <= parse_time(target['result_available_at']) <= at
                    or physical in model.metadata['training_physical_match_ids']):
                raise ValueError('Optional signed gain, independent label clock or physical identity differs')
            alias = forecast['match_id']
            if alias in aliases and aliases[alias] != physical:
                raise ValueError('A provider alias cannot represent two physical B fixtures')
            aliases[alias] = physical
            row = {'sample_id':ident, 'physical_match_id':physical, 'provider_match_id':alias,
                'gain':target['signed_gain'], 'outcome':y, 'result_available_at':target['result_available_at'],
                'context':context}
            # Repeating an artifact does not increase support or reweight it.
            key = (target.get('run_id'), target.get('before_prefix_id'), target.get('after_prefix_id'))
            if None in key:
                raise ValueError('Actual run and adjacent prefix identities are required')
            fingerprint = _digest({k:v for k,v in row.items() if k != 'sample_id'})
            if key in actual_keys:
                if actual_keys[key] != fingerprint:
                    raise ValueError('Conflicting duplicate actual optional-gain transition')
                continue
            actual_keys[key] = fingerprint
            if ident in rows and rows[ident] != row:
                raise ValueError('Conflicting repeated optional gain sample')
            rows[ident] = row
        bindings.append({'folder':str(folder.resolve()), 'report_sha256':sha256((folder/'report.json').read_bytes()).hexdigest(),
            'features_sha256':report['features_sha256'], 'targets_sha256':report['targets_sha256']})
    if len(origins)>1 or origins-{'real_observational','constructed_mechanism'}:
        raise ValueError('A single explicit real or constructed supervision origin is required')
    records, facts = [rows[k] for k in sorted(rows)], {}
    for row in records:
        fact = (row['outcome'],row['context']['kickoff_at'])
        if row['physical_match_id'] in facts and facts[row['physical_match_id']] != fact:
            raise ValueError('Conflicting physical B match outcome or kickoff')
        facts[row['physical_match_id']] = fact
    dates = {row['context']['date'] for row in records}
    reasons = []
    if len(facts)<PARAMETERS['min_physical_matches']:
        reasons.append('fewer_than_12_independent_B_matches')
    if len(dates)<PARAMETERS['min_dates']:
        reasons.append('fewer_than_2_B_kickoff_dates')
    if model is None:
        reasons.append('frozen_optional_predictor_missing')
    ready = {'ready_to_fit':not reasons,'reasons':reasons,'physical_matches':len(facts),
        'kickoff_dates':len(dates),'transitions':len(records),'evidence_origin':next(iter(origins),None),
        'gate_scope':'necessary_B_gate_not_local_calibration_or_real_VoI_effect'}
    return records, model, model_sha, bindings, ready, aliases


def training_readiness(gain_dirs, *, predictor_path=None, as_of):
    return _bundles(gain_dirs,predictor_path=predictor_path,as_of=as_of)[4]


class OptionalContextValueModel(ContextValueModel):
    schema_version = VERSION
    readiness = staticmethod(training_readiness)
    predictor_type = OptionalPriorContextPredictor
    public_validator = staticmethod(_public)
    lexical_blocks = 2
    mask_size = 3
    encoding_description = 'existing_char_3_5_TFIDF_with_explicit_forecast_base_and_missing_prior_mask'
    support_description = 'same_actual_prior_domain_tool_source_text_mask_and_neighborhood_not_calibration'

    @staticmethod
    def _training_rows(gain_dirs, *, predictor_path, as_of):
        return _bundles(gain_dirs, predictor_path=predictor_path, as_of=as_of)

    @staticmethod
    def _fit_clock():
        return datetime.now(timezone.utc)

    def fit(self, gain_dirs, *, predictor_path, as_of, dataset_version):
        rows,f,fsha,bindings,ready,aliases = self._training_rows(gain_dirs,predictor_path=predictor_path,as_of=as_of)
        if not ready['ready_to_fit']:
            raise ValueError('Optional context value not_ready: '+', '.join(ready['reasons']))
        if not isinstance(dataset_version,str) or not dataset_version.strip():
            raise ValueError('An explicit supervised dataset version is required')
        executed = self._fit_clock()
        if parse_time(as_of)>executed:
            raise ValueError('Calibration cutoff cannot follow actual fitting execution')
        counts,objective = self._fit_regression(rows)
        self.predictor = f
        self.metadata = {'training_as_of':parse_time(as_of).isoformat(),'fit_execution_at':executed.isoformat(),
            'dataset_version':dataset_version,'training_physical_match_ids':sorted(counts),
            'training_match_ids':sorted(aliases),'training_aliases':aliases,
            'predictor_sha256':fsha,'predictor_model_version':f.model_version,
            'predictor_training_physical_match_ids':f.metadata['training_physical_match_ids'],
            'evidence_origin':ready['evidence_origin'],'readiness':ready,'input_bindings':bindings,
            'training_rows_sha256':_digest(rows),'negative_targets':sum(row['gain'] < -1e-12 for row in rows),
            'target':'expected_signed_fixed_f_log_loss_reduction_before_cost',
            'weighting':'equal_physical_match_then_equal_distinct_actual_transition',
            'encoding':self.encoding_description,
            'numeric_contract':'visible_masks_then_forecast_probabilities_then_forecast_base_then_time_then_prior_missing_then_costs',
            'interaction':'fixed_128_signed_TensorSketch_normalized_state_by_candidate_outer_product',
            'support_scope':self.support_description,
            'unsupported_domain_behavior':'UNKNOWN_never_zero_value_supervision',
            'uncertainty_kind':'not_estimated_no_confidence_interval_or_coverage_guarantee',
            'production_ready':False,'calibration_status':'not_established','objective':objective}
        self.model_version = self.schema_version+':'+_digest(self._payload())[:24]
        return self

    def predict(self, inputs, *, physical_match_id, predictor_sha256):
        self._check()
        if predictor_sha256 != self.metadata['predictor_sha256']:
            raise ValueError('Optional value requires the same frozen predictor bytes')
        if (not isinstance(physical_match_id,str) or not physical_match_id
                or physical_match_id in self.metadata['training_physical_match_ids']+self.metadata['predictor_training_physical_match_ids']):
            raise ValueError('An independent held-out physical fixture is required')
        context = self.public_validator(inputs,predictor=self.predictor)
        forecast = inputs['forecast']
        if (forecast['physical_match_id'] != physical_match_id or forecast['match_id'] in self.metadata['training_match_ids']
                or parse_time(context['decision_at']) <= parse_time(self.metadata['training_as_of'])
                or parse_time(context['decision_at']) < parse_time(self.metadata['fit_execution_at'])):
            raise ValueError('Value query identity or actual post-fit clock differs')
        if context['predictor_context_supported']:
            vector,reason,nearby = self._support(context)
        else:
            vector,reason,nearby = None,'predictor_prior_domain_context_unsupported',[]
        supported = reason == 'supported_empirical_context'
        gain = float(vector@np.asarray(self.coefficients)) if supported else None
        return {'schema':self.schema_version,'model_version':self.model_version,'expected_signed_gain':gain,'expected_gain':gain,
            'support_status':'supported' if supported else 'UNKNOWN','reason':reason,
            'prior_domain':context['prior_domain'],'evidence_origin':self.metadata['evidence_origin'],
            'support':{'physical_matches':len({r['physical_match_id'] for r in nearby}),
                'kickoff_dates':len({r['date'] for r in nearby}),'channel':context['channel'],
                'minimum_matches':PARAMETERS['min_physical_matches'],'minimum_dates':PARAMETERS['min_dates'],
                'text_cosine_floor':PARAMETERS['min_text_cosine'],'scope':self.metadata['support_scope']},
            'value_kind':'empirical_signed_fixed_predictor_gain_not_Bayes_EVSI','cost_deducted':False,
            'uncertainty':None,'stop_certificate':None,'calibration_status':'not_established','production_ready':False}

    def _payload(self):
        return {'schema':self.schema_version,'parameters':PARAMETERS,'metadata':self.metadata,'encoder':self.encoder,
            'coefficients':self.coefficients,'support_records':self.support_records}

    def _check(self):
        if not self.model_version or self.model_version != self.schema_version+':'+_digest(self._payload())[:24]:
            raise ValueError('Frozen optional context value content/version mismatch')

    @classmethod
    def load(cls, path, *, predictor_path):
        payload = _read(path)
        if payload.get('schema') != cls.schema_version or payload.get('parameters') != PARAMETERS:
            raise ValueError('Unsupported optional context value schema or fixed parameters')
        model = cls()
        for name in ('metadata','encoder','coefficients','support_records','model_version'):
            setattr(model,name,payload[name])
        model._check()
        f = cls.predictor_type.load(predictor_path)
        meta,enc = model.metadata,model.encoder
        if (sha256(Path(predictor_path).read_bytes()).hexdigest() != meta['predictor_sha256']
                or f.model_version != meta['predictor_model_version'] or f.metadata['evidence_origin'] != meta['evidence_origin']
                or f.metadata['training_physical_match_ids'] != meta['predictor_training_physical_match_ids']):
            raise ValueError('Optional context value belongs to a different frozen predictor or origin')
        n = 1+cls.lexical_blocks*len(enc['vocabulary'])+enc['numeric_size']+len(enc['channels'])+len(enc['sources'])+PARAMETERS['interaction_size']
        if (len(model.coefficients)!=n or not np.all(np.isfinite(model.coefficients))
                or enc['numeric_size']!=len(NUMERIC)+6+1+1+len(UNITS)
                or enc['vocabulary']!=sorted(set(enc['vocabulary'])) or len(enc['vocabulary'])>PARAMETERS['max_features']
                or len(enc['idf'])!=len(enc['vocabulary']) or any(not math.isfinite(x) or x<1 for x in enc['idf'])
                or len(set(meta['training_physical_match_ids']))<12
                or set(meta['training_physical_match_ids']) & set(meta['predictor_training_physical_match_ids'])
                or set(meta['training_aliases'])!=set(meta['training_match_ids'])
                or set(meta['training_aliases'].values())!=set(meta['training_physical_match_ids'])
                or not parse_time(meta['training_as_of'])<=parse_time(meta['fit_execution_at'])
                or any(len(row['lexical'])!=cls.lexical_blocks*len(enc['vocabulary']) or len(row['mask'])!=cls.mask_size
                    or any(type(v) is not bool for v in row['mask']) for row in model.support_records)):
            raise ValueError('Invalid frozen optional value encoder or independent support')
        model.predictor = f
        return model


__all__ = ['OptionalContextValueModel','training_readiness','VERSION','GAIN_VERSION','PARAMETERS']
