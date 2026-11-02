"""Deterministic adapter from a registered candidate value model to QueryValueAssessment."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path

from .query_value_contract import QueryValueAssessment
from . import match_return_assessment as return_review

VERSION = 'candidate-value-model-adapter-v1'
MAX_RULES = 64
MAX_CONTEXT_DEPTH = 16


def _packed(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(',', ':'))


def _context_value(value, depth=0):
    '''Copy bounded JSON conditions without coercing missing or typed values.'''
    if depth > MAX_CONTEXT_DEPTH:
        raise ValueError('VALUE_MODEL_WHEN_DEPTH')
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError('VALUE_MODEL_WHEN_KEY')
        return {key: _context_value(item, depth + 1) for key, item in value.items()}
    if isinstance(value, list):
        return [_context_value(item, depth + 1) for item in value]
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError('VALUE_MODEL_WHEN_JSON')


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return float(value)


class CandidateValueModel:
    version = VERSION

    def __init__(self, config):
        if not isinstance(config, Mapping):
            raise ValueError('value model must be an object')
        if config.get('version') != VERSION:
            raise ValueError('VALUE_MODEL_VERSION')
        raw_rules = config.get('rules', [])
        if not isinstance(raw_rules, list) or len(raw_rules) > MAX_RULES:
            raise ValueError('VALUE_MODEL_RULES')
        rules = []
        seen = set()
        for raw in raw_rules:
            if not isinstance(raw, Mapping):
                raise ValueError('VALUE_MODEL_RULE')
            rule_id = raw.get('rule_id')
            action = raw.get('action')
            if not isinstance(rule_id, str) or not rule_id.strip() or rule_id in seen:
                raise ValueError('VALUE_MODEL_RULE_ID')
            if not isinstance(action, str) or not action.strip():
                raise ValueError('VALUE_MODEL_ACTION')
            seen.add(rule_id)
            rule = {'rule_id': rule_id.strip(), 'action': action.strip()}
            for key in ('engine', 'query_contains', 'url_contains'):
                value = raw.get(key)
                if value is not None:
                    if not isinstance(value, str) or not value.strip():
                        raise ValueError(f'VALUE_MODEL_{key.upper()}')
                    rule[key] = value
            when = raw.get('when', {})
            if not isinstance(when, Mapping):
                raise ValueError('VALUE_MODEL_WHEN')
            when = _context_value(when)
            if when:
                rule['when'] = when
            if 'when_return' in raw:
                if not isinstance(raw['when_return'], Mapping):
                    raise ValueError('VALUE_MODEL_WHEN_RETURN')
                # An empty pattern still requires a bound, non-null review.
                rule['when_return'] = _context_value(raw['when_return'])
            status = raw.get('status', 'modeled')
            if status not in {'modeled', 'unmeasured'}:
                raise ValueError('VALUE_MODEL_STATUS')
            rule['status'] = status
            if raw.get('cost') is not None:
                rule['cost'] = _number(raw['cost'], 'cost')
            if status == 'modeled':
                if raw.get('expected_value') is None or raw.get('cost') is None:
                    raise ValueError('MODELED_RULE_NEEDS_VALUE_AND_COST')
                rule['expected_value'] = _number(raw['expected_value'], 'expected_value')
            elif raw.get('expected_value') is not None:
                raise ValueError('UNMEASURED_RULE_CANNOT_HAVE_VALUE')
            rule['reason'] = str(raw.get('reason') or 'registered_candidate_value_model')
            rules.append(rule)
        default = config.get('default', {'status': 'unmeasured'})
        if not isinstance(default, Mapping) or default.get('status', 'unmeasured') not in {'modeled', 'unmeasured'}:
            raise ValueError('VALUE_MODEL_DEFAULT')
        default = dict(default)
        if default.get('cost') is not None:
            default['cost'] = _number(default['cost'], 'cost')
        if default.get('status', 'unmeasured') == 'modeled':
            if default.get('expected_value') is None or default.get('cost') is None:
                raise ValueError('MODELED_DEFAULT_NEEDS_VALUE_AND_COST')
            default['expected_value'] = _number(default['expected_value'], 'expected_value')
        elif default.get('expected_value') is not None:
            raise ValueError('UNMEASURED_DEFAULT_CANNOT_HAVE_VALUE')
        self._config = {'version': VERSION, 'rules': rules, 'default': deepcopy(dict(default))}
        self._rules = tuple(deepcopy(rules))
        self._uses_current_return = any('when_return' in rule for rule in rules)
        self._default = deepcopy(dict(default))
        self._hash = hashlib.sha256(_packed(self._config).encode('utf-8')).hexdigest()

    @classmethod
    def from_path(cls, path):
        source = Path(path)
        return cls(json.loads(source.read_text(encoding='utf-8')))

    def configuration(self):
        descriptor = {'adapter_version': VERSION, 'model_hash': self._hash,
                'rule_count': len(self._rules), 'match_priority': 'first_match',
                'numeric_voi_calibrated': False}
        # Unconditional Round123 models keep their persisted identity on resume.
        if any(rule.get('when') for rule in self._rules):
            descriptor['context_match_mode'] = 'exact_recursive'
        if self._uses_current_return:
            descriptor['current_return_evidence_version'] = return_review.SELECTION_EVIDENCE_VERSION
            descriptor['current_return_judgement'] = 'ACTOR_PROVISIONAL_NOT_GOLD'
        return descriptor

    @staticmethod
    def _context_matches(state, expected):
        if not isinstance(state, Mapping):
            return False
        for key, value in expected.items():
            if key not in state or not CandidateValueModel._value_matches(state[key], value):
                return False
        return True

    @staticmethod
    def _value_matches(actual, expected):
        if isinstance(expected, Mapping):
            return CandidateValueModel._context_matches(actual, expected)
        if isinstance(expected, list):
            return (isinstance(actual, list) and len(actual) == len(expected)
                    and all(CandidateValueModel._value_matches(a, e)
                            for a, e in zip(actual, expected)))
        if type(expected) in (int, float):
            return type(actual) in (int, float) and actual == expected
        return type(actual) is type(expected) and actual == expected

    @staticmethod
    def _matches(rule, response, state, current_return=None):
        if not CandidateValueModel._context_matches(state, rule.get('when', {})):
            return False
        if 'when_return' in rule and not CandidateValueModel._context_matches(current_return, rule['when_return']):
            return False
        if response.get('action') != rule['action']:
            return False
        for key in ('engine',):
            if key in rule and response.get(key) != rule[key]:
                return False
        if 'query_contains' in rule and rule['query_contains'] not in (response.get('query') or ''):
            return False
        if 'url_contains' in rule and rule['url_contains'] not in (response.get('url') or ''):
            return False
        return True

    @staticmethod
    def _assessment(candidate_id, row):
        if row.get('status', 'unmeasured') != 'modeled':
            return QueryValueAssessment.unmeasured(candidate_id, cost=row.get('cost'), reason=row.get('reason', 'value_model_unmeasured'))
        return QueryValueAssessment.modeled(candidate_id, expected_value=row['expected_value'], cost=row['cost'], reason=row.get('reason', 'registered_candidate_value_model'))

    def assess(self, state, candidate):
        response = candidate.response
        current_return = (return_review.current_return_evidence(state, response)
                          if self._uses_current_return else None)
        for rule in self._rules:
            if self._matches(rule, response, state, current_return):
                return self._assessment(candidate.candidate_id, rule)
        return self._assessment(candidate.candidate_id, self._default)

    __call__ = assess


__all__ = ['VERSION', 'CandidateValueModel']
