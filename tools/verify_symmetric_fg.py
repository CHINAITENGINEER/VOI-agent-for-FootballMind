"""Read back one paired experiment; report missed opportunities without promotion."""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from football_research.acquisition import symmetric_semantic_f as f
from football_research.acquisition import state_candidate_gain as g
from football_research.acquisition import policy_return_continuation_g as h
from football_research.acquisition.historical_gain_head import signed_gain
from tools.train_symmetric_fg import public_row


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, data):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)


def verify(directory):
    plan, report = (read(directory/name) for name in ('plan.json', 'report.json'))
    for path, digest in plan['source_bindings'].items():
        assert sha256((ROOT/path).read_bytes()).hexdigest() == digest, path
    labeler = f.load(directory/'f-labeler.json')
    immediate = g.load(directory/'g-immediate.json')
    continuation = h.load(directory/'g-policy-return.json')
    f_hash = sha256((directory/'f-labeler.json').read_bytes()).hexdigest()
    assert immediate['f_labeler_sha256'] == continuation['f_labeler_sha256'] == f_hash
    assert continuation['immediate_model'] == immediate
    assert set(labeler['fit_match_ids']) == set(plan['f_train_ids'])
    assert set(immediate['fit_match_ids']) == set(plan['g_train_ids'])
    assert set(continuation['fit_match_ids']) == set(plan['g_train_ids'])
    assert not set(plan['development_holdout_ids']) & set(plan['f_train_ids']+plan['g_train_ids'])
    from tools.train_state_candidate_g import EVENTS
    from tools.train_learned_continuation_g import SOURCE
    rows = read(EVENTS)
    keyed = {(r['physical_match_id'], tuple(sorted(r['used_input_ids']))): r for r in rows}
    states = read(SOURCE/'states.json')
    stored = read(directory/'state-predictions.json')
    for sid, p in stored.items():
        state = states[sid]
        row = keyed[state['physical_match_id'], tuple(sorted(state['used_input_ids']))]
        assert np.allclose(f.predict(labeler, public_row(row)), p, rtol=0, atol=1e-14)
    transitions = read(directory/'transitions.json')
    graph = defaultdict(list)
    for t in transitions:
        graph[t['state_id']].append(t)
        assert t['f_labeler_sha256'] == f_hash
        assert t['prediction_before'] == stored[t['state_id']]
        assert t['prediction_after'] == stored[t['next_state_id']]
        assert t['query']['public']['probabilities'] == t['prediction_before']
        assert abs(t['signed_gain']-signed_gain(t['prediction_before'], t['prediction_after'], t['outcome'])) < 1e-14
        assert t['body_input_id'] not in states[t['state_id']]['used_input_ids']
    targets = read(directory/'continuation-targets.json')
    for mid in plan['g_train_ids']:
        teacher = g.load(directory/'crossfit-teachers'/f"without-{mid.replace(':', '-')}.json")
        assert set(teacher['fit_match_ids']) == set(plan['g_train_ids'])-{mid}
        for row in (r for r in targets if r['physical_match_id'] == mid):
            edge = next(t for t in graph[row['state_id']] if t['action_id'] == row['action_id'])
            future = graph.get(edge['next_state_id'], [])
            value = h.successor_target(teacher, future, row['packet']['cost_loss_units'])
            assert value == row['successor_policy_label']
            assert value['continuation_target'] == row['continuation_target']
    replay = read(directory/'policy-replays.json')
    assert len(replay) == 450
    # Attribution diagnostic only: observed outcomes never enter the chooser.
    first_menus = []
    for mid in plan['development_holdout_ids']:
        start = max((sid for sid, state in states.items()
                     if state['physical_match_id'] == mid and state['read_count'] == 0),
                    key=lambda sid: states[sid]['title_count'])
        menu = [{'action_id': t['action_id'], 'query': t['query']} for t in graph[start]]
        selected = h.choose(continuation, menu, 2, .001)
        first_menus.append({'physical_match_id': mid, 'state_id': start,
            'policy_decision': selected,
            'candidates': [{'action_id': t['action_id'], 'title': t['query']['public']['title'],
                'predicted_immediate_gain': g.estimate(immediate, t['query']),
                'observed_gain_for_posthoc_diagnosis_only': t['signed_gain']} for t in graph[start]]})
    primary = {r['policy']: r for r in report['primary_comparison'] if r['family'] == 'symmetric_pair'}
    evals = report['f_evaluation']['development_holdout']
    wins = sum(evals['symmetric_f']['per_match'][m]['loss'] < v['loss']
               for m, v in evals['old_f']['per_match'].items())
    result = {'schema': 'symmetric-fg-readback-validation-v1',
        'source_bindings_unchanged': True, 'model_reload_predictions_reproduced': len(stored),
        'paired_gain_labels_reproduced': len(transitions),
        'crossfit_continuation_targets_reproduced': len(targets),
        'whole_match_teacher_exclusion_verified': True,
        'f_g_h_binding_verified': True, 'policy_trajectories': len(replay),
        'evaluation_matches_previously_used': True, 'f_development_improved_matches': wins,
        'f_development_total_matches': len(evals['old_f']['per_match']),
        'new_g_primary_average_reads': primary['two_step']['expected_reads'],
        'new_g_loss_plus_cost': primary['two_step']['loss_plus_cost'],
        'fixed_loss_plus_cost': primary['fixed']['loss_plus_cost'],
        'g_minus_fixed_objective': primary['two_step']['loss_plus_cost']-primary['fixed']['loss_plus_cost'],
        'g_policy_improvement_established': False, 'promotion': 'retain_existing_runtime',
        'network_calls': 0, 'training_in_this_verification': False}
    write(directory/'readback-validation.json', result)
    write(directory/'g-first-menu-diagnosis.json', first_menus)
    write(directory/'stage-decision.json', {
        'schema': 'symmetric-fg-stage-decision-v1',
        'status': 'training_completed_policy_candidate_rejected',
        'f': 'Keep as research candidate: mean improves on reused development set; only 2/5 matches improve.',
        'g': 'Do not promote: primary development policy stops all five despite beneficial fixed reads.',
        'causal_explanation_established': False,
        'supported_observation': 'Training has 7 positive, 46 zero, 53 negative transitions; development has 30 positive, 63 zero, 21 negative. Eight g matches cannot establish conditional value generalization.',
        'next': 'Broaden independent match and state coverage with public pre-read menus and whole-match splits; keep positive/zero/negative labels. Compare within-menu ranking and STOP calibration on train-only cross-fitting before reusing a development set.',
        'do_not': ['Drop negative rows', 'Force g to read to obtain a visually better demonstration',
                   'Increase depth or model parameters using repeated holdout selection',
                   'Treat an MSE-winning zero model as proof that information is worthless'],
        'gpu_required_for_current_models': False})
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    verify(parser.parse_args().directory)
