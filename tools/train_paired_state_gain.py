"""Bounded training-only selection of paired supervision for fixed-f g."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import socket
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.train_learned_continuation_g import PREVIOUS, SOURCE, read, write, change
from tools.reconstruct_panel_priors import no_network
from football_research.acquisition import paired_state_gain as candidate
from football_research.acquisition import state_candidate_gain as base
from football_research.acquisition.historical_policy_comparison import rollout


def errors(model, rows):
    pred = [candidate.estimate(model, r['query']) for r in rows]
    groups = defaultdict(list)
    for r, p in zip(rows, pred): groups[r['physical_match_id']].append((p-r['signed_gain'])**2)
    pairs = candidate.build_pairs(rows); deltas = defaultdict(list)
    for p in pairs:
        error = pred[p['after_index']]-pred[p['before_index']]-p['gain_delta']
        deltas[p['physical_match_id']].append(error**2)
    return {'absolute_mse': float(np.mean([np.mean(v) for v in groups.values()])),
            'delta_mse': float(np.mean([np.mean(v) for v in deltas.values()])) if deltas else None,
            'rows': len(rows), 'directed_pairs': len(pairs)}


def coverage(rows):
    pairs = candidate.build_pairs(rows)
    return {'matches':len({r['physical_match_id'] for r in rows}), 'directed_pairs':len(pairs),
        'changed':sum(abs(p['gain_delta'])>1e-10 for p in pairs),
        'positive_to_negative_or_reverse':sum(rows[p['before_index']]['signed_gain']*rows[p['after_index']]['signed_gain'] < -1e-14 for p in pairs),
        'maximum_absolute_delta':max((abs(p['gain_delta']) for p in pairs),default=0.),
        'independent_examples':False}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); out=ap.parse_args().output
    out.mkdir(parents=True,exist_ok=False)
    socket.socket.connect=no_network; socket.getaddrinfo=no_network
    original=read(PREVIOUS/'plan.json'); train_ids=original['g_training_ids']; hold_ids=original['g_holdout_ids']
    sources=[SOURCE/'transitions.json',SOURCE/'states.json',PREVIOUS/'plan.json',
             PREVIOUS/'prequery-feature-audit.json',PREVIOUS/'g-state-candidate.json']
    bindings={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in sources}
    write(out/'plan.json',{'schema':'paired-state-g-experiment-v1','created_at':datetime.now(timezone.utc).isoformat(),
        'train_ids':train_ids,'holdout_ids':hold_ids,'source_bindings':bindings,'alpha':.1,
        'pair_weight_candidates':[0.,1.], 'selection':'lowest eight-match leave-one-match-out mean absolute gain MSE; exact tie prefers zero',
        'no_feature_search':True,'features':'unchanged 54-dimensional state-candidate vector',
        'pair_objective':'match-weighted same-action gain differences after another acquisition',
        'pairs_correlated_not_new_independent_samples':True,'historical_development_holdout_reused':True,
        'no_holdout_selection':True,'online_selection_allowed':False})
    queries={(r['state_id'],r['action_id']):r['query'] for r in read(PREVIOUS/'prequery-feature-audit.json')['queries']}
    rows=[{**r,'query':queries[r['state_id'],r['action_id']]} for r in read(SOURCE/'transitions.json')]
    train=[r for r in rows if r['g_split']=='train']; held=[r for r in rows if r['g_split']=='holdout']
    assert {r['physical_match_id'] for r in train}==set(train_ids)
    assert {r['physical_match_id'] for r in held}==set(hold_ids)
    folds=[]; (out/'fold-models').mkdir()
    for weight in (0.,1.):
        for mid in train_ids:
            fitting=[r for r in train if r['physical_match_id']!=mid]; valid=[r for r in train if r['physical_match_id']==mid]
            model=candidate.fit(fitting,pair_weight=weight,alpha=.1)
            assert set(model['fit_match_ids'])==set(train_ids)-{mid}
            file=out/'fold-models'/f"pair-{int(weight)}-without-{mid.replace(':','-')}.json"
            candidate.save(model,file); model=candidate.load(file)
            folds.append({'pair_weight':weight,'validation_match':mid,'fit_match_ids':model['fit_match_ids'],
                          'metrics':errors(model,valid),'model_file':str(file.relative_to(out))})
    write(out/'cross-validation.json',folds)
    cv=[{'pair_weight':w,'mean_absolute_mse':float(np.mean([f['metrics']['absolute_mse'] for f in folds if f['pair_weight']==w])),
         'mean_delta_mse':float(np.mean([f['metrics']['delta_mse'] for f in folds if f['pair_weight']==w and f['metrics']['delta_mse'] is not None]))}
        for w in (0.,1.)]
    chosen=min(cv,key=lambda c:(c['mean_absolute_mse'],c['pair_weight']))['pair_weight']
    write(out/'training-only-selection.json',{'cv':cv,'selected_pair_weight':chosen,'holdout_used':False})
    models={}
    for w in (0.,1.):
        m=candidate.fit(train,pair_weight=w,alpha=.1)
        m['f_labeler_sha256']=base.load(PREVIOUS/'g-state-candidate.json')['f_labeler_sha256']
        path=out/f'g-pair-{int(w)}.json'; candidate.save(m,path); models[w]=candidate.load(path)
    old=base.load(PREVIOUS/'g-state-candidate.json')
    assert models[0.]['coefficients']==old['coefficients'] and models[0.]['intercept']==old['intercept']
    evaluation={str(w):errors(m,held) for w,m in models.items()}
    graph=defaultdict(list)
    for r in rows: graph[r['state_id']].append(r)
    states=read(SOURCE/'states.json'); runs=[]
    starts={mid:max(((sid,s) for sid,s in states.items() if s['physical_match_id']==mid and s['read_count']==0),
                    key=lambda x:x[1]['title_count'])[0] for mid in hold_ids}
    for mid,start in starts.items():
        for budget in (1,2,3):
            for cost in (0.,.001,.01):
                for name in ('stop','fixed','random','pair_0','pair_1'):
                    weight=float(name[-1]) if name.startswith('pair_') else None
                    run=rollout(graph,start,policy='g' if weight is not None else name,budget=budget,cost=cost,
                                scorer=(lambda q,m=models[weight]:candidate.estimate(m,q['query'])) if weight is not None else None)
                    assert all(p['reads']<=budget for p in run['paths'])
                    runs.append({**run,'physical_match_id':mid,'comparison_policy':name})
    write(out/'policy-replays.json',runs)
    summary=[]
    for budget in (1,2,3):
        for cost in (0.,.001,.01):
            for name in ('stop','fixed','random','pair_0','pair_1'):
                selected=[r for r in runs if r['budget']==budget and r['cost_loss_units']==cost and r['comparison_policy']==name]
                summary.append({'budget':budget,'cost':cost,'policy':name,
                    **{k:float(np.mean([r['expected_'+k] for r in selected])) for k in ('reads','gain','net')}})
    details=[]
    for split,rs in [('train',train),('holdout',held)]:
        for pair in candidate.build_pairs(rs):
            a,b=rs[pair['before_index']],rs[pair['after_index']]
            acquired=next(r for r in graph[a['state_id']] if r['action_id']==pair['acquired_action_id'])
            details.append({**pair,'split':split,'candidate_title':a['query']['public']['title'],
                'acquired_title':acquired['query']['public']['title'],'gain_before':a['signed_gain'],'gain_after':b['signed_gain'],
                'base_predicted_delta':candidate.estimate(models[0.],b['query'])-candidate.estimate(models[0.],a['query']),
                'paired_predicted_delta':candidate.estimate(models[1.],b['query'])-candidate.estimate(models[1.],a['query']),
                'heldout_diagnostics_not_selection':split=='holdout'})
    write(out/'state-change-comparisons.json',details)
    report={'schema':'paired-state-g-result-v1','train_coverage':coverage(train),'holdout_coverage':coverage(held),
        'training_only_cv':cv,'selected_pair_weight':chosen,'holdout_evaluation':evaluation,'policy_summary':summary,
        'fit_parameters':len(models[1.]['coefficients'])+1,'new_independent_matches':0,'network':0,'api_calls':0,
        'production_ready':False,'online_selection_allowed':False,'training_completed':True,
        'limitations':['Old development holdout reused; no independent confirmation.',
                       'All g training/holdout matches involve Leeds.',
                       'Pair targets are outcome-conditioned fixed-f interactions, not causal intrinsic information value.',
                       'Pairs include mirrored directions and overlapping states; no sample-count independence claim.',
                       'Known archived body menus, no discovery, no newly exposed actions.']}
    write(out/'report.json',report)
    for p,digest in bindings.items():assert sha256((ROOT/p).read_bytes()).hexdigest()==digest
    lines=['g 状态变化监督实验','',
           '目标：同一候选文章在读过其他信息后价值变化时，g也应学习对应变化。',
           '特征54维、参数55个、alpha0.1均不变；比较pair_weight=0和1，按训练8场内部整场留出选择。',
           f'训练集内部选择pair_weight={chosen}；5场开发留出未参与选择。',
           f'训练对照：{report["train_coverage"]}',f'留出对照：{report["holdout_coverage"]}',
           '', '训练集内部交叉验证：']
    for c in cv:lines.append(str(c))
    lines+=['','旧开发留出误差：',str(evaluation),'','预算 | 成本 | 策略 | 平均读取 | 不计成本 | 计入成本']
    for s in summary:lines.append(f"{s['budget']} | {s['cost']:.3f} | {s['policy']} | {s['reads']:.2f} | {change(s['gain'])} | {change(s['net'])}")
    lines+=['','有向对照和镜像对照不是独立比赛；没有新增数据或实时采集。',
            '实际收益差用于训练标签/离线诊断，不进入候选查询前特征。未改f或h，未启用正式模型。']
    (out/'研究结果.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import json
    print(json.dumps({'cv':cv,'selected_pair_weight':chosen,'evaluation':evaluation,
                      'budget2': [s for s in summary if s['budget']==2 and s['cost']==.001]},ensure_ascii=False))


if __name__=='__main__':main()
