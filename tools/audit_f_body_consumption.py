"""Offline audit of an explicit entity-f before/after pair; never a g label.

Run from the repository root with PYTHONPATH=. No provider or HTTP dependency.
Inputs contain actual states and separately reviewed, quote-bound extractions.
This checks representation consumption, not extraction correctness or timing
provenance. It does not infer that an unchanged prediction is optimal.
"""
from copy import deepcopy
from hashlib import sha256
import argparse
import json
from pathlib import Path

import numpy as np

from football_research.acquisition.entity_semantic_f import (
    EntityEvidenceBank, EntitySemanticF, _predict, vector,
)
from football_research.acquisition.historical_context_training import _project
from football_research.acquisition.news_link_projection import VERSION
from football_research.acquisition.semantic_f import public_input, validate_response
from football_research.acquisition.state import parse_time


def audit_pair(model, bank, before, after):
    """Use the same frozen coefficient kernel as the entity market shadow."""
    b, a = (_project(s, text_projection=VERSION) for s in (before, after))
    if b['fixture'] != a['fixture'] or b['prior'] != a['prior']:
        raise ValueError('A same-fixture, fixed-prior comparison is required')
    if before.get('prior_basis') != after.get('prior_basis'):
        raise ValueError('The prior basis must remain fixed')
    if parse_time(a['decision_at']) < parse_time(b['decision_at']):
        raise ValueError('After decision must not precede before')
    pb, pa = bank.project(before), bank.project(after)
    xb, xa = vector(pb), vector(pa)
    yb, ya = _predict(model.coefficients, pb), _predict(model.coefficients, pa)
    before_body_ids = {public_input(b['fixture'], d, 'body')['input_id']
                       for d in b['documents']}
    body_rows = []
    for doc in a['documents']:
        public = public_input(a['fixture'], doc, 'body')
        key = public['input_id']
        claims = validate_response(public, bank.records[key]['response'])
        ignored = [i for i in pa['unmodeled_claims'] if i['input_id'] == key]
        body_rows.append({
            'url': doc['url'], 'input_id': key,
            'text_sha256_utf8': sha256(doc['text'].encode('utf-8')).hexdigest(),
            'text_characters': len(doc['text']), 'new_to_before': key not in before_body_ids,
            'semantic_input_consumed': key in pa['used_input_ids'],
            'claim_count': len(claims), 'supported_claim_count': len(claims)-len(ignored),
            'unsupported_claims': ignored,
        })
    changed = np.flatnonzero(xa != xb).tolist()
    new_bodies = [row for row in body_rows if row['new_to_before']]
    if not new_bodies:
        reason = 'no_new_body'
    elif not changed and all(row['claim_count'] == 0 for row in new_bodies):
        reason = 'body_consumed_no_extracted_claims'
    elif not changed and all(row['supported_claim_count'] == 0 for row in new_bodies):
        reason = 'body_consumed_but_outside_f_feature_support'
    elif not changed:
        reason = 'supported_claims_add_no_novel_features'
    elif np.allclose(yb, ya, rtol=0, atol=1e-14):
        reason = 'features_changed_but_model_response_negligible'
    else:
        reason = 'body_changed_features_and_prediction'
    return {
        'schema': 'footballmind-body-consumption-diagnostic-v1',
        'scope': 'offline_fixed_f_representation_audit',
        'model_version': model.model_version,
        'physical_match_id': a['physical_match_id'],
        'prior_basis': before['prior_basis'],
        'prediction_before': yb.tolist(), 'prediction_after': ya.tolist(),
        'probability_l1_change': float(np.abs(ya-yb).sum()),
        'changed_feature_indices': changed, 'reason': reason,
        'before_projection': pb, 'after_projection': pa, 'bodies': body_rows,
        'extraction_provenance': {row['input_id']: deepcopy(bank.records[row['input_id']]['provenance'])
                                  for row in body_rows},
        'loss_before': None, 'loss_after': None, 'gain': None,
        'g_training_eligibility': 'not_assessed', 'g_training_rows_emitted': 0,
        'optimality_established': False, 'pipeline_timing_provenance_verified': False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding='utf-8'))
    report = audit_pair(EntitySemanticF.load(args.model),
                        EntityEvidenceBank(payload['extractions']),
                        payload['before'], payload['after'])
    report['model_file_sha256'] = sha256(args.model.read_bytes()).hexdigest()
    report['input_file_sha256'] = sha256(args.input.read_bytes()).hexdigest()
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
