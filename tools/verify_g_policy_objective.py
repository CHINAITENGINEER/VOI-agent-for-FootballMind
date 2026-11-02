"""Recompute nested choices and returns from saved fold models, offline."""
import argparse
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import socket
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.evaluate_g_policy_objective import (PREVIOUS,SOURCE,B,MODES,read,write,attach,make_graph,
    start_for,runs_for,records_for,summary,grid_mean)
from tools.reconstruct_panel_priors import no_network
from football_research.acquisition import structured_state_gain as g
from football_research.acquisition import policy_objective_selection as selector


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,required=True);out=ap.parse_args().directory
    socket.socket.connect=no_network;socket.getaddrinfo=no_network
    plan=read(out/'plan.json');report=read(out/'report.json');ids=plan['training_ids']
    for path,digest in plan['source_bindings'].items():assert sha256(Path(path).read_bytes()).hexdigest()==digest
    rows=attach(read(SOURCE/'transitions.json'),read(PREVIOUS/'prequery-features.json'))
    train=[r for r in rows if r['g_split']=='train'];states=read(SOURCE/'states.json');graph=make_graph(rows)
    models={}
    for item in read(out/'fit-index.json'):
        model=g.load(out/'fold-models'/item['file']);fitids=item['fit_match_ids'];mode=item['representation']
        assert model['fit_match_ids']==fitids
        assert model==g.fit([r for r in train if r['physical_match_id'] in fitids],mode,alpha=.1)
        models[mode,tuple(fitids)]=model
    def get(mode,fitids):return models[mode,tuple(sorted(fitids))]
    records=[];outer_runs=[]
    for mid in ids:
        for mode in MODES:
            model=get(mode,set(ids)-{mid});runs=runs_for(graph,start_for(states,mid),mid,model)
            records+=records_for(mode,model,runs)
            outer_runs += [{**r,'comparison_policy':mode,'model_representation':mode} for r in runs]
    assert records==read(out/'training-oof-policy-records.json')
    assert outer_runs==read(out/'training-oof-policy-replays.json')
    final=selector.select(records,ids);assert final==read(out/'training-only-selection.json')
    nested_runs=[];inner_cells=0
    for fold in read(out/'nested-selections.json'):
        outer=fold['outer_match'];inner_ids=[mid for mid in ids if mid!=outer];inner=[]
        for mid in inner_ids:
            for mode in MODES:
                model=get(mode,set(inner_ids)-{mid})
                assert outer not in model['fit_match_ids'] and mid not in model['fit_match_ids']
                inner+=records_for(mode,model,runs_for(graph,start_for(states,mid),mid,model))
        assert inner==fold['inner_records'];inner_cells+=len(inner)
        choice=selector.select(inner,inner_ids);assert choice==fold['inner_selection']
        for name,mode,policy in [('nested_selected',choice['selected'],'g'),('baseline','baseline','g'),
            ('selected_forced',choice['selected'],'g_forced'),('stop',None,'stop'),('fixed',None,'fixed'),('random',None,'random')]:
            model=get(mode,inner_ids) if mode else None
            nested_runs += [{**r,'comparison_policy':name,'model_representation':mode}
                           for r in runs_for(graph,start_for(states,outer),outer,model,policy)]
    assert nested_runs==read(out/'nested-outer-replays.json')
    assert grid_mean(nested_runs)==report['nested_outer_grid_mean_net']
    assert summary(nested_runs)==report['nested_outer_scenarios']
    final_models={mode:g.load(PREVIOUS/f'g-{mode}.json') for mode in MODES};selected=final['selected'];eval_count=0
    for label,panel,ss in [('development',[r for r in rows if r['g_split']=='holdout'],states),
        ('B_reused',attach(read(B/'transitions.json'),read(PREVIOUS/'B-prequery-features.json')),read(B/'states.json'))]:
        gg=make_graph(panel);runs=[]
        for mid in sorted({r['physical_match_id'] for r in panel}):
            assert mid not in ids
            for name,mode,policy in [('policy_selected',selected,'g'),('baseline','baseline','g'),
                ('selected_forced',selected,'g_forced'),('stop',None,'stop'),('fixed',None,'fixed'),('random',None,'random')]:
                runs += [{**r,'comparison_policy':name,'model_representation':mode}
                    for r in runs_for(gg,start_for(ss,mid),mid,final_models[mode] if mode else None,policy)]
        assert runs==read(out/f'{label}-policy-replays.json');eval_count+=len(runs)
        assert summary(runs)==report['reused_evaluations'][label]['scenarios']
        assert grid_mean(runs)==report['reused_evaluations'][label]['grid_mean_net']
    target_profile={}
    for name,panel in [('training',train),('development',[r for r in rows if r['g_split']=='holdout']),('B_reused',read(B/'transitions.json'))]:
        groups=defaultdict(list)
        for row in panel:groups[row['physical_match_id']].append(row['signed_gain'])
        target_profile[name]={'matches':len(groups),'transitions':len(panel),
            'match_weighted_mean_signed_gain':float(np.mean([np.mean(v) for v in groups.values()])),
            'per_match_mean_signed_gain':{k:float(np.mean(v)) for k,v in groups.items()},
            'positive_rows':sum(r['signed_gain']>1e-10 for r in panel),
            'negative_rows':sum(r['signed_gain']< -1e-10 for r in panel),
            'zero_rows':sum(abs(r['signed_gain'])<=1e-10 for r in panel),
            'interpretation':'enumerated fixed-f historical labels, not independent randomized draws or proof of population shift'}
    write(out/'label-distribution-diagnostic.json',target_profile)
    result={'pass':True,'fold_models_refitted_identically':len(models),'inner_policy_cells_replayed':inner_cells,
        'outer_candidate_runs_replayed':len(outer_runs),'nested_outer_runs_replayed':len(nested_runs),
        'reused_evaluation_runs_replayed':eval_count,'all_selection_records_recomputed':True,
        'outer_match_excluded_from_inner_fits_and_selection':True,'source_inputs_unchanged':True,'network':0,'api':0}
    write(out/'readback-verification.json',result)
    print(result)


if __name__=='__main__':main()
