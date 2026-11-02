"""One predeclared g comparison using existing fixed-f labels and splits."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import socket
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from football_research.acquisition import historical_gain_head as old_g
from football_research.acquisition import state_candidate_gain as new_g
from football_research.acquisition import entity_evidence_updates as f
from football_research.acquisition.historical_policy_comparison import rollout
from tools.train_reviewed_rich_f import write
from tools.reconstruct_panel_priors import no_network

SOURCE = ROOT/'var/reviews/fm-historical-transition-g-training-20261014-v2'
EVENTS = ROOT/'var/reviews/fm-entity-updates-f-training-20261014-v1/event-training-rows.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def before_snapshot(row):
    # Drop outcome, after and result clocks before constructing any relation.
    safe = {k: row[k] for k in ('physical_match_id', 'decision_at', 'used_input_ids', 'evidence_events')}
    for event in safe['evidence_events']:
        if event['input_id'] not in safe['used_input_ids']:
            raise ValueError('Evidence event is not visible in the before state')
    _, audit = f.resolve(safe)
    active = {e['event_id'] for e in audit['events'] if e['status'] == 'retained'}
    claims = []
    for e in safe['evidence_events']:
        if e['event_id'] not in active:
            continue
        facet, value, pending, _ = f.status(e)
        claims.append({'input_id': e['input_id'], 'channel': e['evidence']['channel'],
                       'entity': e['entity'], 'team': e['evidence']['team'],
                       'facet': facet, 'value': value or '', 'pending': pending,
                       'text': e['evidence']['text']})
    return {'physical_match_id': safe['physical_match_id'], 'used_input_ids': list(safe['used_input_ids']), 'claims': claims}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True); args = ap.parse_args()
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    socket.socket.connect = no_network; socket.getaddrinfo = no_network
    old_plan = read(SOURCE/'plan.json')
    bindings = {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest()
                for p in [SOURCE/'plan.json', SOURCE/'transitions.json', SOURCE/'states.json',
                          SOURCE/'f-labeler.json', SOURCE/'g-visible_title.json', EVENTS]}
    write(out/'plan.json', {'schema': 'state-candidate-policy-comparison-plan-v1',
        'frozen_at': datetime.now(timezone.utc).isoformat(), 'source_bindings': bindings,
        'f_refit': False, 'new_g_alpha': .1, 'new_g_relation_features': list(new_g.FEATURES),
        'g_training_ids': old_plan['g_training_ids'], 'g_holdout_ids': old_plan['g_holdout_ids'],
        'budgets': [1, 2, 3], 'cost_loss_units': [0., .001, .01],
        'policies': ['stop', 'fixed', 'random_exact', 'old_g', 'state_g', 'old_g_forced', 'state_g_forced'],
        'fixed_order': 'lexicographically ascending observed action ID',
        'random': 'exact expectation over uniform choices without replacement on reconstructed graph',
        'all_read': 'separate uncapped fixed-order reference',
        'forced_g': 'ranking diagnostic ignoring stopping threshold, not main deployment policy',
        'no_holdout_tuning': True, 'new_independent_test': False,
        'previously_used_development_holdout': True, 'network': 0, 'online_selection_allowed': False})
    transitions = read(SOURCE/'transitions.json'); states = read(SOURCE/'states.json')
    rows = read(EVENTS)
    keyed = {(r['physical_match_id'], tuple(sorted(r['used_input_ids']))): r for r in rows}
    snapshots = {sid: before_snapshot(keyed[s['physical_match_id'], tuple(s['used_input_ids'])]) for sid, s in states.items()}
    enriched = []
    for t in transitions:
        s = states[t['state_id']]; after = states[t['next_state_id']]
        assert set(after['used_input_ids'])-set(s['used_input_ids']) == {t['body_input_id']}
        assert set(s['used_input_ids']).issubset(after['used_input_ids'])
        assert t['body_input_id'] not in snapshots[t['state_id']]['used_input_ids']
        enriched.append({**t, 'query': new_g.build_query(t['query'], snapshots[t['state_id']])})
    write(out/'prequery-feature-audit.json', {'snapshots': snapshots, 'queries': [
        {'state_id': t['state_id'], 'action_id': t['action_id'], 'query': t['query']} for t in enriched],
        'candidate_unread_body_excluded': True, 'outcome_after_gain_not_feature_inputs': True})
    train = [t for t in enriched if t['g_split'] == 'train']; test = [t for t in enriched if t['g_split'] == 'holdout']
    assert set(t['physical_match_id'] for t in train) == set(old_plan['g_training_ids'])
    assert set(t['physical_match_id'] for t in test) == set(old_plan['g_holdout_ids'])
    model = new_g.fit(train, alpha=.1)
    model['f_labeler_sha256'] = bindings[str((SOURCE/'f-labeler.json').relative_to(ROOT))]
    new_g.save(model, out/'g-state-candidate.json'); model = new_g.load(out/'g-state-candidate.json')
    old = old_g.load(SOURCE/'g-visible_title.json')
    scores = {'state_g': lambda public: new_g.estimate(model, public['query']),
              'old_g': lambda public: old_g.estimate(old, public['query']['public'])}
    evaluation = {}; predictions = []
    for name in ('zero', 'constant', 'old_g', 'state_g'):
        grouped = defaultdict(list)
        for t in test:
            predicted = (0. if name == 'zero' else old['constant_baseline'] if name == 'constant'
                         else scores[name]({'action_id': t['action_id'], 'query': t['query']}))
            grouped[t['physical_match_id']].append((predicted-t['signed_gain'])**2)
            predictions.append({'model': name, 'state_id': t['state_id'], 'action_id': t['action_id'],
                                'prediction': predicted, 'signed_gain': t['signed_gain']})
        evaluation[name] = {'match_weighted_mse': float(np.mean([np.mean(v) for v in grouped.values()])),
                            'per_match_mse': {k: float(np.mean(v)) for k, v in grouped.items()}}
    graph = defaultdict(list)
    for t in enriched:
        graph[t['state_id']].append(t)
    all_runs = []; references = []
    for mid in old_plan['g_holdout_ids']:
        choices = [(sid, s) for sid, s in states.items() if s['physical_match_id'] == mid and s['read_count'] == 0]
        start = max(choices, key=lambda x: x[1]['title_count'])[0]
        for cost in (0., .001, .01):
            ref = rollout(graph, start, policy='fixed', budget=len({t['body_input_id'] for t in test if t['physical_match_id'] == mid}), cost=cost)
            references.append({**ref, 'physical_match_id': mid, 'comparison_policy': 'read_all'})
            for budget in (1, 2, 3):
                for name in ('stop', 'fixed', 'random_exact', 'old_g', 'state_g', 'old_g_forced', 'state_g_forced'):
                    policy = ('g_forced' if name.endswith('_forced') else 'g' if name in scores
                              else 'random' if name == 'random_exact' else name)
                    scorer = scores.get(name.removesuffix('_forced'))
                    run = rollout(graph, start, policy=policy, budget=budget, cost=cost, scorer=scorer)
                    # Every path's additive gain must equal final minus initial log-probability.
                    for path in run['paths']:
                        if not path['steps']:
                            continue
                        selected = [next(t for t in graph[s['state_id']] if t['action_id'] == s['action_id']) for s in path['steps']]
                        exact = old_g.signed_gain(selected[0]['prediction_before'], selected[-1]['prediction_after'], selected[0]['outcome'])
                        assert abs(exact-path['gain']) < 1e-12
                        assert path['reads'] <= budget
                    all_runs.append({**run, 'physical_match_id': mid, 'comparison_policy': name})
    summary = []
    for budget in (1, 2, 3):
        for cost in (0., .001, .01):
            group = [r for r in all_runs if r['budget'] == budget and r['cost_loss_units'] == cost]
            random_by_match = {r['physical_match_id']: r for r in group if r['comparison_policy'] == 'random_exact'}
            for name in ('stop', 'fixed', 'random_exact', 'old_g', 'state_g', 'old_g_forced', 'state_g_forced'):
                selected = [r for r in group if r['comparison_policy'] == name]
                assert len(selected) == 5
                diffs = [r['expected_net']-random_by_match[r['physical_match_id']]['expected_net'] for r in selected]
                summary.append({'budget': budget, 'cost_loss_units': cost, 'policy': name,
                    **{k: float(np.mean([r['expected_'+k] for r in selected])) for k in ('reads', 'gain', 'net')},
                    'paired_net_vs_random': diffs, 'mean_net_vs_random': float(np.mean(diffs)),
                    'positive_vs_random_matches': sum(d > 1e-12 for d in diffs),
                    'negative_vs_random_matches': sum(d < -1e-12 for d in diffs)})
    starts = {r['physical_match_id']: r['paths'][0]['end_state_id'] for r in all_runs if r['comparison_policy'] == 'stop'}
    coverage = {name: {'nonzero_all_queries': sum(t['query']['relations'][name] != 0 for t in enriched),
                       'nonzero_start_candidates': sum(t['query']['relations'][name] != 0 for t in test if t['state_id'] == starts[t['physical_match_id']])}
                for name in new_g.FEATURES}
    report = {'schema': 'state-candidate-g-policy-comparison-v1', 'trained': True,
        'train_matches': 8, 'holdout_matches': 5, 'train_rows': len(train), 'holdout_rows': len(test),
        'evaluation': evaluation, 'policy_summary': summary, 'relation_feature_coverage': coverage,
        'new_independent_test': False, 'production_ready': False, 'online_selection_allowed': False,
        'f_retrained': False, 'network': 0, 'provider_calls': 0,
        'limitations': ['Previously used development holdout; no new blind validation.',
        'All 13 g matches involve Leeds; historical titles/visibility are reconstructed.',
        'Same fixed research f; gain is updater-specific, not intrinsic article value.',
        'Costs are hypothetical log-loss units, not dollars; action space is archived body reads.',
        'Exact random path probabilities are simulation quantities, not observed logging propensities.']}
    write(out/'report.json', report); write(out/'policy-replays.json', all_runs)
    write(out/'all-read-reference.json', references); write(out/'holdout-predictions.json', predictions)
    print(json.dumps({'evaluation': evaluation, 'cost_zero_policy_summary': [r for r in summary if r['cost_loss_units'] == 0],
                      'feature_coverage': coverage}, ensure_ascii=False))


if __name__ == '__main__':
    main()
