"""Read back learned heads, recompute features/folds and exact policy paths."""
import argparse
from hashlib import sha256
from pathlib import Path
import socket
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.train_structured_state_g import B,MODES,PREVIOUS,SOURCE,read,write,metrics,run_evaluation
from tools.train_state_candidate_g import before_snapshot
from tools.reconstruct_panel_priors import no_network
from football_research.acquisition import structured_state_gain as g


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,required=True);out=ap.parse_args().directory
    socket.socket.connect=no_network;socket.getaddrinfo=no_network
    plan=read(out/'plan.json');report=read(out/'report.json');audit=read(PREVIOUS/'prequery-feature-audit.json')
    for path,digest in plan['source_bindings'].items():assert sha256(Path(path).read_bytes()).hexdigest()==digest
    rows=[{**r,'query':g.build_query(r['query'],audit['snapshots'][r['state_id']])} for r in read(SOURCE/'transitions.json')]
    saved=read(out/'prequery-features.json')
    assert saved==[{'state_id':r['state_id'],'action_id':r['action_id'],'query':r['query']} for r in rows]
    train=[r for r in rows if r['g_split']=='train'];dev=[r for r in rows if r['g_split']=='holdout']
    for fold in read(out/'cross-validation.json'):
        m=g.load(out/'fold-models'/fold['model_file']);mid=fold['validation_match']
        assert mid not in m['fit_match_ids']
        assert metrics(m,[r for r in train if r['physical_match_id']==mid])==fold['metrics']
    cv=[{'representation':mode,**{key:float(np.mean([f['metrics'][key] for f in read(out/'cross-validation.json')
            if f['representation']==mode and f['metrics'][key] is not None])) for key in ('absolute_mse','delta_mse')}} for mode in MODES]
    assert cv==report['cv']
    selected=min(cv,key=lambda x:(x['absolute_mse'],MODES.index(x['representation'])))['representation']
    assert selected==read(out/'training-only-selection.json')['selected']==report['selected']
    models={mode:g.load(out/f'g-{mode}.json') for mode in MODES}
    for mode,m in models.items():
        refit=g.fit(train,mode)
        assert refit==m
        assert m['production_ready'] is False and m['empirical_selection_allowed'] is False
    bstates=read(B/'states.json');keyed={(r['physical_match_id'],tuple(sorted(r['used_input_ids']))):r for r in read(B/'prediction-input-rows.json')}
    brows=[]
    for r in read(B/'transitions.json'):
        state=bstates[r['state_id']];snap=before_snapshot(keyed[r['physical_match_id'],tuple(state['used_input_ids'])])
        assert r['body_input_id'] not in snap['used_input_ids']
        brows.append({**r,'query':g.build_query(r['query']['public'],snap)})
    assert read(out/'B-prequery-features.json')==[{'state_id':r['state_id'],'action_id':r['action_id'],'query':r['query']} for r in brows]
    total=0;detailed=[]
    for label,panel,states in [('development',dev,read(SOURCE/'states.json')),('B_reused',brows,bstates)]:
        for mode,m in models.items():assert metrics(m,panel)==report['evaluations'][label][mode]
        runs,summary=run_evaluation(models,panel,states)
        assert runs==read(out/f'{label}-policy-replays.json') and summary==report['policy_summary'][label]
        total+=len(runs)
        for mid in sorted({r['physical_match_id'] for r in runs}):
            subset=[r for r in runs if r['physical_match_id']==mid and r['budget']==2 and r['cost_loss_units']==.001]
            detailed.append({'panel':label,'physical_match_id':mid,'budget':2,'cost_loss_units':.001,
                'policies':{r['comparison_policy']:{'reads':r['expected_reads'],'gain':r['expected_gain'],'net':r['expected_net'],
                 'paths':r['paths']} for r in subset}})
    write(out/'per-match-decisions.json',detailed)
    write(out/'readback-verification.json',{'pass':True,'query_features_recomputed':len(rows)+len(brows),
        'fold_metrics_recomputed':24,'final_model_refits_match':3,'training_selection_recomputed':True,
        'exact_policy_replays':total,'input_bindings_unchanged':True,'network':0,'api':0})
    print({'queries':len(rows)+len(brows),'folds':24,'models':3,'policy_replays':total,'pass':True})


if __name__=='__main__':main()
