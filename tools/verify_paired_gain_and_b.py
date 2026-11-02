"""Read back paired training and frozen B evaluation without any network."""
import argparse
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import socket
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.train_paired_state_gain import read,write,errors,PREVIOUS,SOURCE
from tools.reconstruct_panel_priors import no_network
from football_research.acquisition import paired_state_gain as paired
from football_research.acquisition import learned_continuation_g as g2
from football_research.acquisition import entity_evidence_updates as f
from football_research.acquisition import historical_gain_head as gain
from football_research.acquisition.context_predictor import _digest
from football_research.acquisition.historical_policy_comparison import rollout


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--training',type=Path,required=True);ap.add_argument('--evaluation',type=Path,required=True)
    args=ap.parse_args();train_dir=args.training;ev=args.evaluation
    socket.socket.connect=no_network;socket.getaddrinfo=no_network
    plan=read(train_dir/'plan.json')
    for path,digest in plan['source_bindings'].items():assert sha256((ROOT/path).read_bytes()).hexdigest()==digest
    queries={(q['state_id'],q['action_id']):q['query'] for q in read(PREVIOUS/'prequery-feature-audit.json')['queries']}
    source=[{**t,'query':queries[t['state_id'],t['action_id']]} for t in read(SOURCE/'transitions.json')]
    folds=read(train_dir/'cross-validation.json')
    for fold in folds:
        model=paired.load(train_dir/fold['model_file'])
        valid=[t for t in source if t['physical_match_id']==fold['validation_match']]
        assert set(model['fit_match_ids'])==set(plan['train_ids'])-{fold['validation_match']}
        assert not set(model['fit_match_ids'])&set(plan['holdout_ids'])
        assert errors(model,valid)==fold['metrics']
    selection=read(train_dir/'training-only-selection.json')
    cv=[{'weight':w,'mse':float(np.mean([r['metrics']['absolute_mse'] for r in folds if r['pair_weight']==w]))} for w in (0.,1.)]
    assert min(cv,key=lambda r:(r['mse'],r['weight']))['weight']==selection['selected_pair_weight']
    models={w:paired.load(train_dir/f'g-pair-{int(w)}.json') for w in (0.,1.)}
    train=[t for t in source if t['g_split']=='train']
    for w,model in models.items():
        fitted=paired.fit(train,pair_weight=w)
        assert np.allclose(fitted['coefficients'],model['coefficients'],atol=1e-15,rtol=1e-12)
        assert abs(fitted['intercept']-model['intercept'])<1e-15
    graph=defaultdict(list)
    for t in source:graph[t['state_id']].append(t)
    states=read(SOURCE/'states.json')
    starts={mid:max(((sid,s) for sid,s in states.items() if s['physical_match_id']==mid and s['read_count']==0),
                   key=lambda p:p[1]['title_count'])[0] for mid in plan['holdout_ids']}
    training_runs=read(train_dir/'policy-replays.json')
    for r in training_runs:
        name=r['comparison_policy'];weight=float(name[-1]) if name.startswith('pair_') else None
        again=rollout(graph,starts[r['physical_match_id']],policy=r['policy'],budget=r['budget'],cost=r['cost_loss_units'],
                      scorer=(lambda q,m=models[weight]:paired.estimate(m,q['query'])) if weight is not None else None)
        assert all(again[k]==r[k] for k in again)
    ep=read(ev/'plan.json')
    for path,digest in ep['source_bindings'].items():assert sha256(Path(path).read_bytes()).hexdigest()==digest
    labeler=f.load(SOURCE/'f-labeler.json');predictions=read(ev/'fixed-f-predictions.json')
    rows=read(ev/'prediction-input-rows.json');b_ids=set(ep['fixture_ids'])
    assert not b_ids&(set(plan['train_ids'])|set(plan['holdout_ids'])|set(labeler['head']['head']['encoder']['fit_match_ids']))
    for row in rows:
        assert row['outcome'] is None
        sid=_digest({'match':row['physical_match_id'],'inputs':sorted(row['used_input_ids'])})
        assert np.array_equal(f.predict(labeler,row),predictions[sid])
    priors=read(ev/'prior-snapshots.json')
    for snapshot in priors.values():assert not set(snapshot['training_match_ids'])&set(ep['base_prior_excludes'])
    bg=defaultdict(list);transitions=read(ev/'transitions.json');bs=read(ev/'states.json')
    for t in transitions:
        assert gain.signed_gain(predictions[t['state_id']],predictions[t['next_state_id']],t['outcome'])==t['signed_gain']
        assert set(bs[t['next_state_id']]['used_input_ids'])-set(bs[t['state_id']]['used_input_ids'])=={t['body_input_id']}
        assert t['body_input_id'] not in bs[t['state_id']]['used_input_ids']
        bg[t['state_id']].append(t)
    starts={mid:max(((sid,s) for sid,s in bs.items() if s['physical_match_id']==mid and s['read_count']==0),
                   key=lambda p:p[1]['title_count'])[0] for mid in b_ids}
    two=g2.load(ROOT/'var/reviews/fm-g-learned-two-step-20261014-v1/g-two-step.json')
    selected=models[selection['selected_pair_weight']];bruns=read(ev/'policy-replays.json')
    for r in bruns:
        again=rollout(bg,starts[r['physical_match_id']],policy=r['policy'],budget=r['budget'],cost=r['cost_loss_units'],
                      scorer=lambda q:paired.estimate(selected,q['query']),menu_selector=lambda m,l,c:g2.choose(two,m,l,c))
        assert all(again[k]==r[k] for k in again)
    fixtures={c['fixture']['physical_match_id']:c['fixture'] for c in read(ROOT/'var/submission/p8-g-stream-v3/cases.json')}
    by_match=[]
    for mid in ep['fixture_ids']:
        rr={r['comparison_policy']:r for r in bruns if r['physical_match_id']==mid and r['budget']==2 and r['cost_loss_units']==.001}
        by_match.append({'physical_match_id':mid,'home':fixtures[mid]['home_name'],'away':fixtures[mid]['away_name'],
            'selected_g':{k:rr['selected_g']['expected_'+k] for k in ('reads','gain','net')},
            'random':{k:rr['random']['expected_'+k] for k in ('reads','gain','net')},
            'g_actions':[{'title':next(t for t in bg[s['state_id']] if t['action_id']==s['action_id'])['query']['public']['title'],
                          'estimated_gain':s['estimated_gain'],'actual_gain':s['signed_gain']} for s in rr['selected_g']['paths'][0]['steps']]})
    write(ev/'per-match-decisions.json',by_match)
    verification={'cv_folds_recomputed':len(folds),'training_only_selection_recomputed':True,'g_refits_match_saved':2,
        'old_development_runs_exact_replayed':len(training_runs),'B_predictions_recomputed':len(rows),
        'B_signed_gains_recomputed':len(transitions),'B_runs_exact_replayed':len(bruns),
        'B_overlap_current_g_fit_and_development_and_f_fit':0,'prior_target_exclusion_checks':len(priors),
        'network':0,'api_calls':0,'C_scored':False}
    write(ev/'readback-verification.json',verification);write(train_dir/'readback-verification.json',verification)
    import json
    print(json.dumps(verification))


if __name__=='__main__':main()
