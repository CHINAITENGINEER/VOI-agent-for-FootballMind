"""Train a bounded g from one-body historical transitions of a frozen f.

The f parameter fit, g parameter fit and holdout matches are disjoint. This is
still reused development material with reconstructed visibility, not a blind
or prospective study and not an online policy-value estimate.
"""
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
from football_research.acquisition import entity_evidence_updates as f
from football_research.acquisition import historical_gain_head as g
from football_research.acquisition.context_predictor import _digest
from tools.reconstruct_panel_priors import no_network
from tools.train_reviewed_rich_f import write

SOURCE = ROOT/'var/reviews/fm-entity-updates-f-training-20261014-v1'
RICH = ROOT/'var/reviews/fm-reviewed-rich-f-training-20261014-v1'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True); args = ap.parse_args()
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    socket.socket.connect = no_network; socket.getaddrinfo = no_network
    rows = read(SOURCE/'event-training-rows.json'); previous = read(RICH/'report.json')
    f_ids = previous['folds'][0]['training_ids']
    later = [mid for fold in previous['folds'] for mid in fold['validation_ids']]
    train_ids, test_ids = later[:8], later[8:]
    assert not set(f_ids)&set(later) and not set(train_ids)&set(test_ids)
    facts = {r['physical_match_id']: r for r in rows}; dt = f.timestamp
    assert max(dt(facts[mid]['result_available_at']) for mid in f_ids) < min(dt(facts[mid]['decision_at']) for mid in later)
    assert max(dt(facts[mid]['result_available_at']) for mid in train_ids) < min(dt(facts[mid]['decision_at']) for mid in test_ids)
    write(out/'plan.json', {'schema': 'historical-fixed-f-g-training-plan-v1',
        'frozen_at': datetime.now(timezone.utc).isoformat(), 'f_training_ids': f_ids,
        'g_training_ids': train_ids, 'g_holdout_ids': test_ids,
        'f_regularization': 10., 'g_alpha': 0.1,
        'g_models': ['numeric', 'visible_title'], 'main_candidate': 'visible_title',
        'cost_sensitivity_loss_units': [0., .001, .01],
        'no_hyperparameter_selection_on_g_holdout': True,
        'previously_used_development_data': True, 'new_independent_holdout': False,
        'time_basis': 'historical reconstruction; publication assumed menu visibility, results kickoff+3h assumption',
        'transition': 'one actually archived body added to same discovered-title state; simulated enumeration',
        'zero_negative_gains_retained': True, 'propensity': None,
        'online_selection_allowed': False, 'network': 0,
        'source_rows_sha256': sha256((SOURCE/'event-training-rows.json').read_bytes()).hexdigest()})
    frozen_f = f.fit([r for r in rows if r['physical_match_id'] in f_ids], 10.)
    f.save(frozen_f, out/'f-labeler.json'); labeler_sha = sha256((out/'f-labeler.json').read_bytes()).hexdigest()
    assert set(frozen_f['head']['head']['encoder']['fit_match_ids']) == set(f_ids)
    protocol = read(ROOT/'var/f-entity-semantic-20261011-v2/protocol.json')
    records = {r['input']['input_id']: r for r in read(protocol['source_bindings']['records']['path'])}
    clocks = {(c['fixture']['physical_match_id'], d['url']): d['source_published_at']
              for c in read(protocol['source_bindings']['cases']['path']) for d in c['documents']}
    by_match = defaultdict(list)
    for r in rows:
        if r['physical_match_id'] in later: by_match[r['physical_match_id']].append(r)
    transitions = []; state_registry = {}; predictions = {}
    for mid in later:
        states = by_match[mid]; keyed = {frozenset(r['used_input_ids']): r for r in states}
        assert len(keyed) == len(states)
        for keys, row in keyed.items():
            sid = _digest({'match': mid, 'inputs': sorted(keys)})
            state_registry[sid] = {'physical_match_id': mid, 'used_input_ids': sorted(keys),
                'read_count': sum(records[k]['input']['channel']=='body' for k in keys),
                'title_count': sum(records[k]['input']['channel']=='title' for k in keys)}
            predictions[sid] = f.predict(frozen_f, row).tolist()
        for keys, before in keyed.items():
            sid = _digest({'match': mid, 'inputs': sorted(keys)})
            for target_keys, after in keyed.items():
                added = target_keys-keys
                if not keys.issubset(target_keys) or len(added) != 1: continue
                body_id = next(iter(added)); public = records[body_id]['input']
                if public['channel'] != 'body': continue
                # Title of this exact candidate must already be exposed.
                titles = [records[k]['input'] for k in keys if records[k]['input']['channel']=='title' and records[k]['input']['url']==public['url']]
                if len(titles) != 1: continue
                title = titles[0]; assert title['fixture']['physical_match_id'] == mid
                tid = _digest({'match': mid, 'inputs': sorted(target_keys)})
                p, q = predictions[sid], predictions[tid]
                # Only after the fixed-f predictions are complete do we read y for scoring.
                y = before['outcome']; assert y == after['outcome']
                age = (dt(before['decision_at'])-dt(clocks[mid, public['url']])).total_seconds()/3600
                resolved_before, _ = f.resolve(before); resolved_after, audit = f.resolve(after)
                from football_research.acquisition.scoped_reviewed_f import route
                projected_before = route(resolved_before)[0]['evidence']; projected_after = route(resolved_after)[0]['evidence']
                effect = ('representation_unchanged' if projected_before == projected_after else
                          'representation_changed_probability_unchanged' if np.allclose(p,q,atol=1e-12,rtol=0) else 'probability_changed')
                transitions.append({'physical_match_id': mid, 'state_id': sid, 'next_state_id': tid,
                    'action_id': 'read-'+_digest({'url': public['url']})[:20], 'body_input_id': body_id,
                    'body_url': public['url'], 'query': {'schema': g.QUERY_SCHEMA, 'probabilities': p,
                        'acquired_documents': state_registry[sid]['read_count'], 'visible_candidates': 0,
                        'article_age_hours': age, 'title': title['text']},
                    'prediction_before': p, 'prediction_after': q, 'outcome': y,
                    'signed_gain': g.signed_gain(p,q,y), 'gain_units': 'natural_log_loss_reduction',
                    'loss_before': float(-np.log(p[y])), 'loss_after': float(-np.log(q[y])),
                    'f_labeler_sha256': labeler_sha, 'effect_class': effect,
                    'unresolved_conflict_groups_after': len(audit['conflicts']),
                    'g_split': 'train' if mid in train_ids else 'holdout',
                    'provenance': 'enumerated_historical_reconstruction', 'propensity': None,
                    'prospective_training_eligible': False, 'offline_research_eligible': True})
    by_state = defaultdict(list)
    for t in transitions: by_state[t['state_id']].append(t)
    for menu in by_state.values():
        for t in menu: t['query']['visible_candidates'] = len(menu)
    assert {t['physical_match_id'] for t in transitions} == set(later)
    write(out/'transitions.json', transitions); write(out/'states.json', state_registry)
    train = [t for t in transitions if t['g_split']=='train']; test = [t for t in transitions if t['g_split']=='holdout']
    models = {}
    for name, title in [('numeric',False),('visible_title',True)]:
        model = g.fit(train,use_title=title,alpha=.1); model['f_labeler_sha256'] = labeler_sha
        g.save(model,out/f'g-{name}.json'); restored = g.load(out/f'g-{name}.json')
        assert g.estimate(model,test[0]['query']) == g.estimate(restored,test[0]['query'])
        models[name] = model
    constant = models['visible_title']['constant_baseline']
    predictions_g = {name: [g.estimate(m,t['query']) for t in test] for name,m in models.items()}
    predictions_g.update(constant=[constant]*len(test),zero=[0.]*len(test))
    evaluation = {}
    for name, preds in predictions_g.items():
        grouped = defaultdict(list)
        for t,pred in zip(test,preds): grouped[t['physical_match_id']].append((pred-t['signed_gain'])**2)
        evaluation[name] = {'match_weighted_mse': float(np.mean([np.mean(v) for v in grouped.values()])),
                            'per_match_mse': {k:float(np.mean(v)) for k,v in grouped.items()}}
    # Historical graph replay: all archived titles visible initially; no search
    # simulation, no acquisition propensities or claims of real policy evaluation.
    rollouts = []
    for mid in test_ids:
        candidates = [(sid,v) for sid,v in state_registry.items() if v['physical_match_id']==mid and v['read_count']==0]
        start = max(candidates,key=lambda pair:pair[1]['title_count'])[0]
        for cost in (0.,.001,.01):
            for policy in ('visible_title','constant','read_all','stop'):
                current = start; steps = []
                while by_state.get(current):
                    menu = sorted(by_state[current],key=lambda t:t['action_id'])
                    if policy == 'stop': break
                    if policy in ('visible_title','constant'):
                        values = [g.estimate(models['visible_title'],t['query']) if policy=='visible_title' else constant for t in menu]
                        best = int(np.argmax(values))
                        if values[best] <= cost: break
                    else: best = 0
                    chosen = menu[best]; steps.append({'action_id':chosen['action_id'], 'signed_gain':chosen['signed_gain']})
                    current = chosen['next_state_id']
                total = sum(t['signed_gain'] for t in steps)
                rollouts.append({'physical_match_id':mid,'policy':policy,'hypothetical_cost_loss_units':cost,
                    'reads':len(steps),'realized_reconstructed_gain':total,'net':total-cost*len(steps),'steps':steps,
                    'scope':'reconstructed_graph_rollout_not_live_or_causal_evaluation'})
    write(out/'rollouts.json',rollouts)
    write(out/'holdout-predictions.json',[{'transition':t,'predictions':{k:float(v[i]) for k,v in predictions_g.items()}}
                                         for i,t in enumerate(test)])
    gains = np.array([t['signed_gain'] for t in transitions])
    report={'schema':'historical-fixed-f-g-result-v1','f_training_matches':len(f_ids),
        'g_training_matches':len(train_ids),'g_holdout_matches':len(test_ids),
        'transitions':len(transitions),'train_rows':len(train),'holdout_rows':len(test),
        'positive':int(sum(gains>1e-12)),'negative':int(sum(gains < -1e-12)),'zero_tolerance_1e12':int(sum(abs(gains)<=1e-12)),
        'constant_baseline':constant,'evaluation':evaluation,'g_trained':True,
        'empirical_prospective_g_trained':False,'online_selection_allowed':False,
        'f_model_fit_excludes_all_g_targets':True,'production_ready':False,'network':0,'provider_calls':0,
        'limitations':['only 8 train / 5 holdout matches; all g targets Leeds',
        'reused development corpus and model design; no new blind test',
        'historical version and visibility assumptions; no observed selection propensity',
        'target is gain of this particular research f, not intrinsic information value',
        'no data on real tool failure probability, latency or dollar cost']}
    write(out/'report.json', report)
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__': main()
