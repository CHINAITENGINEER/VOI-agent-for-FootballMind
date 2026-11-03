"""Compare replay outputs with a small, versioned numerical reference."""
from __future__ import annotations

import json
import math
from pathlib import Path

REFERENCE = Path(__file__).resolve().parents[1] / 'docs/reproduction/reference-predictions-v1.json'


def verify_predictions(predictions_path, reference_path=None):
    reference = json.loads(Path(reference_path or REFERENCE).read_text(encoding='utf-8-sig'))
    if reference.get('schema') != 'footballmind-reference-predictions-v1':
        raise ValueError('Unsupported prediction reference')
    expected = {(r['case_id'], r['head']): r['prediction'] for r in reference['predictions']}
    if not expected or len(expected) != len(reference['predictions']):
        raise ValueError('Empty or duplicate prediction reference')
    found = {}
    total = 0
    with Path(predictions_path).open(encoding='utf-8-sig') as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            key = (row['case_id'], row['head'])
            if key in found:
                raise ValueError('Duplicate case/head in replay output')
            found[key] = row['prediction']
    differences = []
    max_difference = 0.0
    for key, saved in expected.items():
        current = found.get(key)
        if current is None:
            differences.append({'case_id': key[0], 'head': key[1], 'field': 'missing_prediction'})
            continue
        for field in ('p_before', 'p_after'):
            values = current.get(field)
            if (not isinstance(values, list) or len(values) != 3
                    or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values)
                    or not math.isclose(sum(values), 1, abs_tol=1e-6)):
                differences.append({'case_id': key[0], 'head': key[1], 'field': field, 'reason': 'invalid_probability'})
                continue
            error = max(abs(a - b) for a, b in zip(values, saved[field]))
            max_difference = max(max_difference, error)
            if error > 1e-6:
                differences.append({'case_id': key[0], 'head': key[1], 'field': field, 'max_abs_error': error})
        for field in ('selected_ids', 'unselected_ids', 'stop_reason', 'budget', 'threshold',
                      'body_adoptions', 'physical_match_id', 'f_used_ids'):
            if current.get(field) != saved[field]:
                differences.append({'case_id': key[0], 'head': key[1], 'field': field})
        for field in ('f_sha256', 'g_archive_sha256'):
            if current.get('model_bindings', {}).get(field) != saved['model_bindings'][field]:
                differences.append({'case_id': key[0], 'head': key[1], 'field': 'model_bindings.' + field})
    return {'schema': 'footballmind-reproduction-check-v1',
            'status': 'PASS' if not differences else 'FAIL',
            'reference_predictions': len(expected), 'matched_predictions': sum(k in found for k in expected),
            'input_rows': total, 'additional_rows_not_compared': sum(k not in expected for k in found),
            'probability_absolute_tolerance': 1e-6, 'max_probability_difference': max_difference,
            'differences': differences, 'scope': 'frozen_numerical_regression_not_effectiveness_evaluation'}
