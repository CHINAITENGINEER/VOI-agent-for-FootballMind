"""Nested whole-match CV for g model selection by decision return, offline."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import socket
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.train_structured_state_g import SOURCE,B,read,write
from tools.train_learned_continuation_g import change
from tools.reconstruct_panel_priors import no_network
from football_research.acquisition import structured_state_gain as g
from football_research.acquisition import policy_objective_selection as selector
from football_research.acquisition.historical_policy_comparison import rollout

PREVIOUS=ROOT/'var/reviews/fm-g-structured-state-20261014-v1'
MODES=selector.MODES


def attach(rows,queries):
    keyed={(q['state_id'],q['action_id']):q['query'] for q in queries}
    return [{**r,'query':keyed[r['state_id'],r['action_id']]} for r in rows]


def make_graph(rows):
    graph=defaultdict(list)
    for r in rows:graph[r['state_id']].append(r)
    return graph


def start_for(states,mid):
    return max(((sid,s) for sid,s in states.items() if s['physical_match_id']==mid and s['read_count']==0),
               key=lambda item:item[1]['title_count'])[0]


def runs_for(graph,start,mid,model,policy='g'):
    return [{**rollout(graph,start,policy=policy,budget=budget,cost=cost,
                       scorer=(lambda a:g.estimate(model,a['query'])) if model is not None else None),
             'physical_match_id':mid}
            for budget in selector.BUDGETS for cost in selector.COSTS]


def records_for(mode,model,runs):
    return [{'representation':mode,'physical_match_id':r['physical_match_id'],'budget':r['budget'],
             'cost':r['cost_loss_units'],'expected_net':r['expected_net'],'fit_match_ids':model['fit_match_ids']} for r in runs]


def summary(runs):
    result=[]
    for name in sorted({r['comparison_policy'] for r in runs}):
        for budget in selector.BUDGETS:
            for cost in selector.COSTS:
                selected=[r for r in runs if r['comparison_policy']==name and r['budget']==budget and r['cost_loss_units']==cost]
                result.append({'policy':name,'budget':budget,'cost':cost,
                    **{key:float(np.mean([r['expected_'+key] for r in selected])) for key in ('reads','gain','net')}})
    return result


def grid_mean(runs):
    return {name:float(np.mean([r['expected_net'] for r in runs if r['comparison_policy']==name]))
            for name in sorted({r['comparison_policy'] for r in runs})}


def decision_diagnostics(graph,runs,models):
    """Post-decision labels only; never exposed to the scorer or selection rule."""
    result=[]
    for run in runs:
        if run['budget']!=2 or run['cost_loss_units']!=.001 or run['comparison_policy'] not in ('baseline','policy_selected'):
            continue
        mode=run['model_representation'];model=models[mode]
        for path in run['paths']:
            for step in path['steps']:
                edges=graph[step['state_id']];chosen=next(e for e in edges if e['action_id']==step['action_id'])
                result.append({'physical_match_id':run['physical_match_id'],'policy':run['comparison_policy'],'decision':'read',
                    'state_id':step['state_id'],'visible_title':chosen['query']['base']['public']['title'],
                    'predicted_gain':step['estimated_gain'],'realized_gain':chosen['signed_gain'],
                    'realized_immediate_net':chosen['signed_gain']-.001,
                    'scope':'retrospective fixed-f immediate diagnostic; not intrinsic or causal information value'})
            if path['stop_reason']=='estimated_net_gain_nonpositive':
                edges=graph[path['end_state_id']]
                result.append({'physical_match_id':run['physical_match_id'],'policy':run['comparison_policy'],'decision':'stop',
                    'state_id':path['end_state_id'],'remaining_candidates':[
                        {'visible_title':e['query']['base']['public']['title'],'predicted_gain':g.estimate(model,e['query']),
                         'realized_gain':e['signed_gain'],'realized_immediate_net':e['signed_gain']-.001} for e in edges],
                    'scope':'retrospective remaining one-read outcomes, excluded from prequery policy'})
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);out=ap.parse_args().output
    out.mkdir(parents=True,exist_ok=False);socket.socket.connect=no_network;socket.getaddrinfo=no_network
    oldplan=read(PREVIOUS/'plan.json');ids=oldplan['train_ids'];dev_ids=oldplan['development_ids']
    paths=[SOURCE/'transitions.json',SOURCE/'states.json',PREVIOUS/'prequery-features.json',PREVIOUS/'B-prequery-features.json',
           PREVIOUS/'plan.json',B/'transitions.json',B/'states.json',
           ROOT/'football_research/acquisition/policy_objective_selection.py',Path(__file__)]
    paths += [PREVIOUS/f'g-{mode}.json' for mode in MODES]
    bindings={str(p):sha256(p.read_bytes()).hexdigest() for p in paths}
    write(out/'plan.json',{'schema':'nested-policy-objective-experiment-v1','created_at':datetime.now(timezone.utc).isoformat(),
        'training_ids':ids,'development_ids':dev_ids,'source_bindings':bindings,'representations':list(MODES),'alpha':.1,
        'costs':list(selector.COSTS),'budgets':list(selector.BUDGETS),
        'primary_objective':'equal-match average net loss reduction across all nine predeclared scenarios',
        'scenario_weights':'equal, illustrative sensitivity grid, not empirically measured user preferences',
        'secondary_display':{'budget':2,'cost':.001,'not_used_to_pick_a_winning_cell':True},
        'tie_order':list(MODES),'threshold_unchanged':'read iff predicted signed gain exceeds supplied cost',
        'nested_cv':'outer one-match exclusion; inner one-match exclusion from the other seven; 6 fit / 1 select / 1 outer test',
        'final_choice':'eight-match out-of-fold net return, distinct from unbiased assessment of its selected maximum',
        'forced_g':'diagnostic only, same ranking without learned stop; not eligible for representation selection',
        'f_h_retrained':False,'no_new_features':True,'no_dev_B_selection':True,
        'prior_exposure':'researcher has seen these historical datasets; nested computation is not a new blind experiment',
        'online_selection_allowed':False,'network':0})
    rows=attach(read(SOURCE/'transitions.json'),read(PREVIOUS/'prequery-features.json'))
    train=[r for r in rows if r['g_split']=='train'];states=read(SOURCE/'states.json');graph=make_graph(rows)
    cache={};(out/'fold-models').mkdir();fit_index=[]
    def trained(mode,fitting_ids):
        key=mode,tuple(sorted(fitting_ids))
        if key not in cache:
            m=g.fit([r for r in train if r['physical_match_id'] in fitting_ids],mode,alpha=.1)
            assert set(m['fit_match_ids'])==set(fitting_ids)
            file=f'{mode}-{sha256("|".join(key[1]).encode()).hexdigest()[:16]}.json'
            g.save(m,out/'fold-models'/file);cache[key]=g.load(out/'fold-models'/file)
            fit_index.append({'representation':mode,'fit_match_ids':list(key[1]),'file':file})
        return cache[key]
    outer_records=[];outer_runs=[]
    for mid in ids:
        for mode in MODES:
            m=trained(mode,set(ids)-{mid});rs=runs_for(graph,start_for(states,mid),mid,m)
            outer_records+=records_for(mode,m,rs)
            outer_runs += [{**r,'comparison_policy':mode,'model_representation':mode} for r in rs]
    final_choice=selector.select(outer_records,ids)
    write(out/'training-oof-policy-records.json',outer_records)
    write(out/'training-only-selection.json',final_choice)
    write(out/'training-oof-policy-replays.json',outer_runs)
    nested=[];nested_runs=[]
    for outer in ids:
        inner_ids=[mid for mid in ids if mid!=outer];records=[]
        for inner in inner_ids:
            for mode in MODES:
                m=trained(mode,set(inner_ids)-{inner})
                records+=records_for(mode,m,runs_for(graph,start_for(states,inner),inner,m))
        choice=selector.select(records,inner_ids)
        assert all(outer not in r['fit_match_ids'] and outer!=r['physical_match_id'] for r in records)
        nested.append({'outer_match':outer,'inner_selection':choice,'inner_records':records})
        selected=choice['selected'];model=trained(selected,set(inner_ids))
        for name,mode,policy in [('nested_selected',selected,'g'),('baseline','baseline','g'),
                                 ('selected_forced',selected,'g_forced'),('stop',None,'stop'),('fixed',None,'fixed'),('random',None,'random')]:
            m=trained(mode,set(inner_ids)) if mode else None
            nested_runs += [{**r,'comparison_policy':name,'model_representation':mode} for r in runs_for(graph,start_for(states,outer),outer,m,policy)]
    write(out/'nested-selections.json',nested);write(out/'nested-outer-replays.json',nested_runs)
    write(out/'fit-index.json',fit_index)
    selected=final_choice['selected'];models={mode:g.load(PREVIOUS/f'g-{mode}.json') for mode in MODES}
    for m in models.values():assert set(m['fit_match_ids'])==set(ids)
    write(out/'selected-policy.json',{'schema':'historical-g-policy-selection-config-v1','selected_representation':selected,
        'model_path':str(PREVIOUS/f'g-{selected}.json'),'model_source_sha256':bindings[str(PREVIOUS/f'g-{selected}.json')],
        'selection_objective':final_choice['criterion'],'training_match_ids':ids,'threshold':'predicted_gain > supplied_cost',
        'production_ready':False,'online_selection_allowed':False})
    evaluation={};diags=[]
    for label,panel,ss in [('development',[r for r in rows if r['g_split']=='holdout'],states),
                         ('B_reused',attach(read(B/'transitions.json'),read(PREVIOUS/'B-prequery-features.json')),read(B/'states.json'))]:
        gg=make_graph(panel);runs=[]
        for mid in sorted({r['physical_match_id'] for r in panel}):
            assert mid not in ids
            for name,mode,policy in [('policy_selected',selected,'g'),('baseline','baseline','g'),
                                    ('selected_forced',selected,'g_forced'),('stop',None,'stop'),('fixed',None,'fixed'),('random',None,'random')]:
                runs += [{**r,'comparison_policy':name,'model_representation':mode} for r in runs_for(gg,start_for(ss,mid),mid,models[mode] if mode else None,policy)]
        write(out/f'{label}-policy-replays.json',runs)
        evaluation[label]={'grid_mean_net':grid_mean(runs),'scenarios':summary(runs)}
        diags += [{'panel':label,**d} for d in decision_diagnostics(gg,runs,models)]
    write(out/'decision-diagnostics.json',diags)
    nested_summary=summary(nested_runs);nested_grid=grid_mean(nested_runs)
    report={'schema':'nested-policy-objective-result-v1','training_selection':final_choice,
        'nested_outer_grid_mean_net':nested_grid,'nested_outer_scenarios':nested_summary,
        'nested_selected_modes':{r['outer_match']:r['inner_selection']['selected'] for r in nested},
        'reused_evaluations':evaluation,'trained_fold_models':len(fit_index),'network':0,'api':0,
        'new_independent_matches':0,'f_h_retrained':False,'production_ready':False,'online_selection_allowed':False,
        'limitations':['Only eight training matches, all Leeds; scenario repetitions are not new independent samples.',
            'Same researcher has inspected prior results; computational nesting does not erase prior exposure.',
            'This changes representation selection objective; each gain regressor still minimizes ridge MSE.',
            'Forced-read ablation separates stopping effect from ranking on this finite graph; not a recommendation to always read.',
            'Labels depend on a fixed research f and reconstructed historical menus; no causal or prospective performance claim.']}
    write(out/'report.json',report)
    for path,digest in bindings.items():assert sha256(Path(path).read_bytes()).hexdigest()==digest
    lines=['g 决策目标与停止诊断','',f'训练8场交叉预测按策略收益选择：{selected}',
        f'训练交叉预测网格均值：{final_choice["mean_net"]}',
        '该最大值用于选择，不能当成选择后的无偏性能；另做外层整场留出。',
        f'嵌套外层均值：{nested_grid}',
        '这里改的是模型选择目标，g内部仍是岭回归MSE；没有声称已实现端到端策略梯度。',
        '固定比较预算1/2/3、成本0/0.001/0.01；每场等权，9种设置不算9个独立样本。']
    for label,info in [('nested_training',{'scenarios':nested_summary}),*evaluation.items()]:
        lines+=['',label,'预算 | 成本 | 策略 | 平均读取 | 不计成本 | 计入成本']
        for s in info['scenarios']:
            lines.append(f"{s['budget']} | {s['cost']:.3f} | {s['policy']} | {s['reads']:.2f} | {change(s['gain'])} | {change(s['net'])}")
    (out/'研究结果.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import json
    print(json.dumps({'selected':selected,'oof':final_choice['mean_net'],'nested':nested_grid,
        'nested_modes':report['nested_selected_modes'],'trained_fold_models':len(fit_index),
        'evaluation':{k:v['grid_mean_net'] for k,v in evaluation.items()},
        'budget2cost001':{k:[s for s in v['scenarios'] if s['budget']==2 and s['cost']==.001] for k,v in evaluation.items()}},ensure_ascii=False))


if __name__=='__main__':main()
