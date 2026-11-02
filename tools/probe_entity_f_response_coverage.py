"""Offline numerical capability probe; never a match forecast or g label.

Constructed projected states exercise frozen predictor coefficients, including
prior interactions and a second distinct entity. They do not assert that any
football event occurred and do not test extraction or source relevance.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from football_research.acquisition.entity_semantic_f import (
    EntitySemanticF, REPRESENTATION_SHA256, _predict, vector,
)
from football_research.acquisition.semantic_f import FEATURES

PRIORS = ((1 / 3, 1 / 3, 1 / 3), (.6, .25, .15), (.15, .25, .6))


def probe_model(model: EntitySemanticF) -> dict:
    rows = []
    n = len(FEATURES)
    for prior in PRIORS:
        for index, feature in enumerate(FEATURES):
            before = {
                'prior': list(prior), 'representation_sha256': REPRESENTATION_SHA256,
                'title': [0.] * n, 'title_entity_counts': [0.] * n,
                'novel_body': [0.] * n, 'novel_body_entity_counts': [0.] * n,
            }
            for entity_count in (1, 2):
                after = deepcopy(before)
                after['novel_body'][index] = 1.
                after['novel_body_entity_counts'][index] = float(entity_count)
                p = _predict(model.coefficients, before)
                q = _predict(model.coefficients, after)
                rows.append({
                    'feature': feature, 'constructed_prior': list(prior),
                    'entity_count_before': entity_count - 1,
                    'entity_count_after': entity_count,
                    'before': p.tolist(), 'after': q.tolist(),
                    'feature_vector_changed': bool(np.any(vector(after) != vector(before))),
                    'max_probability_change': float(np.max(np.abs(q - p))),
                    'probability_l1_change': float(np.sum(np.abs(q - p))),
                })
                before = after
    coverage = []
    coefficients = np.asarray(model.coefficients)
    for index, feature in enumerate(FEATURES):
        selected = [r for r in rows if r['feature'] == feature]
        # Softmax ignores a logit shift common to all three outcome classes.
        block = coefficients[[2*n + index, 3*n + index,
                              6*n + index, 7*n + index]]
        class_contrast = float(np.max(np.ptp(block, axis=1)))
        coverage.append({
            'feature': feature,
            'body_class_contrast_max': class_contrast,
            'body_has_class_contrast': class_contrast > 1e-12,
            'max_probe_probability_change': max(r['max_probability_change'] for r in selected),
            'responsive_probe_count': sum(r['max_probability_change'] > 1e-12 for r in selected),
            'probe_count': len(selected),
        })
    return {
        'model_version': model.model_version,
        'metadata_scope': model.metadata.get('scope'),
        'optimizer': model.metadata.get('optimizer'),
        'feature_count': n, 'constructed_probe_count': len(rows),
        'body_features_with_class_contrast': sum(r['body_has_class_contrast'] for r in coverage),
        'body_features_responding_in_probes': sum(r['responsive_probe_count'] > 0 for r in coverage),
        'coverage': coverage, 'probes': rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    reports = []
    for path in args.model:
        raw = path.read_bytes()
        model = EntitySemanticF.load(path)
        if path.read_bytes() != raw:
            raise ValueError('Model changed while loading')
        reports.append({**probe_model(model), 'model_path': str(path.resolve()),
                        'model_raw_sha256': sha256(raw).hexdigest()})
    result = {
        'schema': 'entity-f-constructed-response-coverage-v1',
        'executed_at': datetime.now(timezone.utc).isoformat(),
        'purpose': 'numerical_response_diagnosis_not_information_value',
        'input_origin': 'constructed_projected_states_not_source_assertions',
        'network_requests': 0, 'external_model_calls': 0, 'training': False,
        'g_labels_emitted': 0, 'runtime_changed': False, 'models': reports,
        'limitations': [
            'Projected-state probe does not validate body extraction or end-to-end runtime consumption.',
            'Numerical response is not accuracy, calibrated gain or causal information value.',
            'Zero response is a limitation of this frozen f, not proof that a source is useless.',
            'Keep zero-gain rows; do not filter training or evaluation by realized gain sign.',
            'Nonzero response does not authorize promotion of a research-only model.',
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps([{'model': r['model_version'],
                      'responsive_features': r['body_features_responding_in_probes'],
                      'probes': r['constructed_probe_count']} for r in reports]))


if __name__ == '__main__':
    main()
