"""Predeclared three-representation g ablation; no new network or f/h fit."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
from hashlib import sha256
from pathlib import Path
import socket
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.train_learned_continuation_g import PREVIOUS,SOURCE,read,write,change
from tools.train_state_candidate_g import before_snapshot
from tools.reconstruct_panel_priors import no_network
from football_research.acquisition import structured_state_gain as g
from football_research.acquisition import state_candidate_gain as old
from football_research.acquisition import paired_state_gain as pairs
from football_research.acquisition.historical_policy_comparison import rollout

B=ROOT/'var/reviews/fm-g-frozen-historical-b-20261014-v1'
MODES=('baseline','structured','combined')


def metrics(model,rows):
    predictions=[g.estimate(model,r['query']) if model is not None else 0. for r in rows]
    errors=defaultdict(list);delta=defaultdict(list)
    for r,p in zip(rows,predictions):errors[r['physical_match_id']].append((p-r['signed_gain'])**2)
    for pair in pairs.build_pairs(rows):
        value=predictions[pair['after_index']]-predictions[pair['before_index']]-pair['gain_delta']
        delta[pair['physical_match_id']].append(value**2)
    return {'absolute_mse':float(np.mean([np.mean(x) for x in errors.values()])),
            'delta_mse':float(np.mean([np.mean(x) for x in delta.values()])) if delta else None}


def run_evaluation(models,rows,states):
    graph=defaultdict(list)
    for r in rows:graph[r['state_id']].append(r)
    starts={mid:max(((sid,s) for sid,s in states.items() if s['physical_match_id']==mid and s['read_count']==0),
                    key=lambda item:item[1]['title_count'])[0] for mid in sorted({r['physical_match_id'] for r in rows})}
    runs=[]
    for mid,start in starts.items():
        for budget in (1,2,3):
            for cost in (0.,.001,.01):
                for name in ('stop','fixed','random',*MODES):
                    result=rollout(graph,start,policy='g' if name in models else name,budget=budget,cost=cost,
                                  scorer=(lambda a,m=models[name]:g.estimate(m,a['query'])) if name in models else None)
                    runs.append({**result,'physical_match_id':mid,'comparison_policy':name})
    summary=[]
    for budget in (1,2,3):
        for cost in (0.,.001,.01):
            for name in ('stop','fixed','random',*MODES):
                cells=[r for r in runs if r['budget']==budget and r['cost_loss_units']==cost and r['comparison_policy']==name]
                summary.append({'budget':budget,'cost':cost,'policy':name,
                    **{key:float(np.mean([r['expected_'+key] for r in cells])) for key in ('reads','gain','net')}})
    return runs,summary


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);out=ap.parse_args().output
    out.mkdir(parents=True,exist_ok=False);socket.socket.connect=no_network;socket.getaddrinfo=no_network
    original=read(PREVIOUS/'plan.json');train_ids=original['g_training_ids'];dev_ids=original['g_holdout_ids']
    paths=[SOURCE/'transitions.json',SOURCE/'states.json',PREVIOUS/'prequery-feature-audit.json',
           PREVIOUS/'plan.json',PREVIOUS/'g-state-candidate.json',B/'transitions.json',B/'states.json',B/'prediction-input-rows.json',
           ROOT/'football_research/acquisition/structured_state_gain.py',Path(__file__)]
    bindings={str(p):sha256(p.read_bytes()).hexdigest() for p in paths}
    write(out/'plan.json',{'schema':'structured-state-g-ablation-v1','created_at':datetime.now(timezone.utc).isoformat(),
        'train_ids':train_ids,'development_ids':dev_ids,'source_bindings':bindings,
        'representations':list(MODES),'alpha':.1,'structured_features':list(g.FEATURES),
        'selection':'minimum eight-match leave-one-match-out absolute gain MSE; exact tie favors baseline then structured then combined',
        'lexical_title_routes_not_unread_body_semantics':True,'all_value_coefficients_learned':True,
        'historical_development_and_B_previously_examined':True,'new_independent_matches':0,
        'train_selection_not_blind_prospective_validation':True,'f_h_fit':False,'network':0,'online_selection_allowed':False})
    audit=read(PREVIOUS/'prequery-feature-audit.json');states=read(SOURCE/'states.json')
    rows=[]
    for r in read(SOURCE/'transitions.json'):
        snap=audit['snapshots'][r['state_id']]
        assert r['body_input_id'] not in snap['used_input_ids']
        rows.append({**r,'query':g.build_query(r['query'],snap)})
    train=[r for r in rows if r['g_split']=='train'];dev=[r for r in rows if r['g_split']=='holdout']
    assert {r['physical_match_id'] for r in train}==set(train_ids)
    assert {r['physical_match_id'] for r in dev}==set(dev_ids)
    write(out/'prequery-features.json',[{'state_id':r['state_id'],'action_id':r['action_id'],'query':r['query']} for r in rows])
    (out/'fold-models').mkdir();folds=[]
    for mode in MODES:
        for mid in train_ids:
            fitting=[r for r in train if r['physical_match_id']!=mid];validation=[r for r in train if r['physical_match_id']==mid]
            m=g.fit(fitting,mode);name=f'{mode}-without-{mid.replace(":","-")}.json'
            g.save(m,out/'fold-models'/name);m=g.load(out/'fold-models'/name)
            assert set(m['fit_match_ids'])==set(train_ids)-{mid}
            folds.append({'representation':mode,'validation_match':mid,'model_file':name,'metrics':metrics(m,validation),
                          'zero_reference':metrics(None,validation)})
    write(out/'cross-validation.json',folds)
    cv=[{'representation':mode,**{key:float(np.mean([f['metrics'][key] for f in folds if f['representation']==mode and f['metrics'][key] is not None]))
          for key in ('absolute_mse','delta_mse')}} for mode in MODES]
    selected=min(cv,key=lambda x:(x['absolute_mse'],MODES.index(x['representation'])))['representation']
    write(out/'training-only-selection.json',{'cv':cv,'selected':selected,'evaluation_used_for_selection':False})
    models={}
    for mode in MODES:
        m=g.fit(train,mode);g.save(m,out/f'g-{mode}.json');models[mode]=g.load(out/f'g-{mode}.json')
    baseline=old.load(PREVIOUS/'g-state-candidate.json')
    assert models['baseline']['coefficients']==baseline['coefficients'] and models['baseline']['intercept']==baseline['intercept']
    # Only now build/evaluate the second reused panel; no fit on either panel.
    bstates=read(B/'states.json');inputs=read(B/'prediction-input-rows.json')
    keyed={(r['physical_match_id'],tuple(sorted(r['used_input_ids']))):r for r in inputs}
    brows=[]
    for r in read(B/'transitions.json'):
        state=bstates[r['state_id']];snapshot=before_snapshot(keyed[r['physical_match_id'],tuple(state['used_input_ids'])])
        assert r['body_input_id'] not in snapshot['used_input_ids']
        brows.append({**r,'query':g.build_query(r['query']['public'],snapshot)})
    assert not {r['physical_match_id'] for r in brows}&set(train_ids+dev_ids)
    write(out/'B-prequery-features.json',[{'state_id':r['state_id'],'action_id':r['action_id'],'query':r['query']} for r in brows])
    evaluations={};summary={};details=[];total=0
    for label,panel,ss in [('development',dev,states),('B_reused',brows,bstates)]:
        evaluations[label]={mode:metrics(m,panel) for mode,m in models.items()};evaluations[label]['zero']=metrics(None,panel)
        runs,summ=run_evaluation(models,panel,ss);summary[label]=summ;total+=len(runs)
        write(out/f'{label}-policy-replays.json',runs)
        for r in panel:
            details.append({'panel':label,'physical_match_id':r['physical_match_id'],'state_id':r['state_id'],'action_id':r['action_id'],
                'visible_title':r['query']['base']['public']['title'],'realized_gain':r['signed_gain'],
                'predictions':{mode:g.estimate(m,r['query']) for mode,m in models.items()}})
    write(out/'gain-predictions.json',details)
    coverage={}
    for label,panel in [('train',train),('development',dev),('B_reused',brows)]:
        coverage[label]={key:sum(r['query']['structured'][key]!=0 for r in panel) for key in g.FEATURES}
    report={'schema':'structured-state-g-ablation-result-v1','selected':selected,'cv':cv,'evaluations':evaluations,'policy_summary':summary,
        'parameters':{mode:len(m['coefficients'])+1 for mode,m in models.items()},'feature_coverage':coverage,
        'train_matches':8,'development_matches':5,'B_matches':5,'new_independent_matches':0,'f_h_fit':False,
        'network':0,'api_calls':0,'production_ready':False,'online_selection_allowed':False,'policy_runs':total,
        'limitations':['All matches involve Leeds; no cross-team confirmation.',
        'Development and B results have been examined before; neither is a new blind test.',
        'Structured claims inherit existing reviewed extraction and historical version limitations.',
        'Candidate topic uses visible-title lexical routes, not an LLM or unread-body understanding.',
        'Enumerated historical states and costs in log-loss units; no new prospective trajectories.',
        'Zero reference is reported; representation selection alone does not validate decision usefulness.']}
    write(out/'report.json',report)
    for path,digest in bindings.items():assert sha256(Path(path).read_bytes()).hexdigest()==digest
    lines=['g 查询前结构化状态实验','',f'训练内部选择：{selected}',
           '仅改变 g 的输入表示；f、h、训练标签、查询成本保持不变。',
           '结构化状态区别主客队与可用性状态；标题路由仅表示可能涉及的主题，不保证正文包含该事实。',
           '训练8场内部整场留出选择；旧开发5场和已看过的B5场仅复用评估。不是新增盲测。','',str(cv),'',str(evaluations)]
    for label,summ in summary.items():
        lines+=['',label,'预算 | 成本 | 策略 | 平均读取 | 不计成本 | 计入成本']
        for s in summ:lines.append(f"{s['budget']} | {s['cost']:.3f} | {s['policy']} | {s['reads']:.2f} | {change(s['gain'])} | {change(s['net'])}")
    (out/'研究结果.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import json
    print(json.dumps({'selected':selected,'cv':cv,'evaluations':evaluations,'parameters':report['parameters'],
        'budget2_cost001':{k:[r for r in v if r['budget']==2 and r['cost']==.001] for k,v in summary.items()}},ensure_ascii=False))


if __name__=='__main__':main()
