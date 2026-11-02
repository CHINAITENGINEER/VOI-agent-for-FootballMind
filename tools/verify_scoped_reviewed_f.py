"""Materialize behavioral checks on saved research models without retraining."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from football_research.acquisition import scoped_reviewed_f as f
from tools.train_reviewed_rich_f import write


def main():
    source = ROOT / 'var/reviews/fm-reviewed-rich-f-training-20261014-v1'
    out = ROOT / 'var/reviews/fm-scoped-reviewed-f-training-20261014-v1'
    rows = json.loads((source / 'training-rows.json').read_text(encoding='utf-8'))
    previous = json.loads((source / 'plan.json').read_text(encoding='utf-8'))
    report = json.loads((out / 'report.json').read_text(encoding='utf-8'))
    model = f.load(out / 'typed_context-candidate.json')
    ticket = next(e for r in rows for e in r['evidence'] if 'concept:ticket_sales' in e['atoms'])
    no_evidence = []; full_fit = []; relevance_checks = 0
    for r in rows:
        p = f.predict(model, r)
        extended = deepcopy(r); extended['evidence'].append(ticket)
        assert np.array_equal(p, f.predict(model, extended))
        relevance_checks += 1
        if not r['evidence']:
            assert np.allclose(p, r['prior'], atol=1e-14, rtol=0)
            no_evidence.append(r['physical_match_id'])
        if r['physical_match_id'] in previous['development_ids'] and r['evidence']:
            full_fit.append({'physical_match_id': r['physical_match_id'], 'used_input_ids': r['used_input_ids'],
                             'prior': r['prior'], 'prediction': p.tolist(),
                             'l1_shift': float(np.abs(p - r['prior']).sum()),
                             'scope': 'in_sample_response_not_predictive_validation', 'gain': None})
    scores = report['families']['typed_context']['candidates']
    base = {x['physical_match_id']: x for x in scores[0]['per_match']}
    candidate = next(c for c in scores if c['regularization'] == report['families']['typed_context']['candidate_regularization'])
    paired = [{'physical_match_id': x['physical_match_id'], 'baseline_log_loss': base[x['physical_match_id']]['log_loss'],
               'candidate_log_loss': x['log_loss'],
               'baseline_minus_candidate': base[x['physical_match_id']]['log_loss'] - x['log_loss']}
              for x in candidate['per_match']]
    write(out / 'response-diagnostics.json', {'training_responses': full_fit, 'forward_validation_pairs': paired})
    write(out / 'behavior-verification.json', {
        'ticket_injection_exact_invariance_rows': relevance_checks,
        'empty_evidence_prior_preservation_rows': len(no_evidence),
        'nonzero_training_responses': sum(x['l1_shift'] > 1e-12 for x in full_fit),
        'validation_better': sum(x['baseline_minus_candidate'] > 1e-12 for x in paired),
        'validation_worse': sum(x['baseline_minus_candidate'] < -1e-12 for x in paired),
        'validation_equal': sum(abs(x['baseline_minus_candidate']) <= 1e-12 for x in paired),
        'ticket_result_is_designed_invariance_not_learned_usefulness': True,
        'does_not_assess_g_information_value': True, 'runtime_integration': False,
        'network_requests': 0, 'model_calls': 'local_f_only',
        'source_model_sha256': sha256((out / 'typed_context-candidate.json').read_bytes()).hexdigest()})
    print(json.dumps({'ticket_invariance_rows': relevance_checks, 'empty_evidence_rows': len(no_evidence),
                      'nonzero_training_responses': sum(x['l1_shift'] > 1e-12 for x in full_fit)}))


if __name__ == '__main__':
    main()
