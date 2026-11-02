"""Read-back verification and explicitly post-result train-mean control."""
from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
import argparse
import json
from pathlib import Path
import socket
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.train_learned_continuation_g import PREVIOUS, SOURCE, read, write, public_menu, change
from tools.reconstruct_panel_priors import no_network
from football_research.acquisition import learned_continuation_g as g2
from football_research.acquisition import state_candidate_gain as g1
from football_research.acquisition.historical_policy_comparison import rollout


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();out=args.output
    socket.socket.connect=no_network;socket.getaddrinfo=no_network
    plan=read(out/'plan.json');model=g2.load(out/'g-two-step.json');base=model['immediate_model']
    for path,digest in plan['source_bindings'].items():assert sha256((ROOT/path).read_bytes()).hexdigest()==digest
    queries={(q['state_id'],q['action_id']):q['query'] for q in read(PREVIOUS/'prequery-feature-audit.json')['queries']}
    graph=defaultdict(list);states=read(SOURCE/'states.json')
    for t in read(SOURCE/'transitions.json'):graph[t['state_id']].append({**t,'query':queries[t['state_id'],t['action_id']]})
    teachers={mid:g1.load(out/'crossfit-teachers'/f"leave-out-{mid.replace(':','-')}.json") for mid in plan['train_ids']}
    targets=read(out/'continuation-training-targets.json')
    for row in targets:
        teacher=teachers[row['physical_match_id']]
        assert set(teacher['fit_match_ids'])==set(plan['train_ids'])-{row['physical_match_id']}
        assert not set(teacher['fit_match_ids'])&set(plan['holdout_ids'])
        assert row['packet']['menu']==public_menu(graph[row['state_id']])
        following=graph.get(row['next_state_id'],[])
        assert all(e['physical_match_id']==row['physical_match_id'] for e in following)
        expected=max([0.]+[g1.estimate(teacher,e['query'])-row['packet']['cost_loss_units'] for e in following])
        assert expected==row['continuation_target']
    # Prespecify the diagnostic control here; no fitting/tuning to holdout returns.
    grouped=defaultdict(lambda:defaultdict(list))
    for row in targets:grouped[row['packet']['cost_loss_units']][row['physical_match_id']].append(row['continuation_target'])
    means={cost:float(np.mean([np.mean(values) for values in per_match.values()])) for cost,per_match in grouped.items()}
    write(out/'post-result-control-plan.json',{'status':'post_result_diagnostic_not_prespecified_main_candidate',
        'rule':'equal-match mean training continuation target at each declared cost',
        'cost_to_training_mean':means,'holdout_outcomes_used_to_set_bonus':False,
        'threshold_or_hyperparameter_search':False})
    constant_models={}
    for cost,mean in means.items():
        control=deepcopy(model);control['coefficients']=[0.]*len(control['coefficients']);control['intercept']=mean
        constant_models[cost]=control
    starts={mid:max(((sid,s) for sid,s in states.items() if s['physical_match_id']==mid and s['read_count']==0),key=lambda p:p[1]['title_count'])[0]
            for mid in plan['holdout_ids']}
    runs=read(out/'policy-replays.json');control_runs=[]
    for r in runs:
        selector=lambda menu,left,cost:g2.choose(model,menu,left,cost)
        replay=rollout(graph,starts[r['physical_match_id']],policy=r['policy'],budget=r['budget'],cost=r['cost_loss_units'],
            scorer=lambda p:g1.estimate(base,p['query']),menu_selector=selector)
        assert all(replay[k]==r[k] for k in replay)
        if r['comparison_policy']=='two_step':
            control=constant_models[r['cost_loss_units']]
            c=rollout(graph,starts[r['physical_match_id']],policy='menu',budget=r['budget'],cost=r['cost_loss_units'],
                menu_selector=lambda menu,left,cost:g2.choose(control,menu,left,cost))
            control_runs.append({**c,'physical_match_id':r['physical_match_id'],'comparison_policy':'train_mean_continuation'})
    summary=[]
    for budget in (1,2,3):
        for cost in (0.,.001,.01):
            cell=[r for r in control_runs if r['budget']==budget and r['cost_loss_units']==cost]
            summary.append({'budget':budget,'cost_loss_units':cost,**{k:float(np.mean([r['expected_'+k] for r in cell])) for k in ('reads','gain','net')}})
    write(out/'train-mean-control-replays.json',control_runs);write(out/'train-mean-control-summary.json',summary)
    changes=[]
    for r in runs:
        if r['comparison_policy']!='two_step':continue
        one=next(x for x in runs if x['comparison_policy']=='one_step' and all(x[k]==r[k] for k in ('physical_match_id','budget','cost_loss_units')))
        one_actions=[s['action_id'] for s in one['paths'][0]['steps']];two_actions=[s['action_id'] for s in r['paths'][0]['steps']]
        if one_actions==two_actions:continue
        detailed=[]
        for step in r['paths'][0]['steps']:
            edge=next(t for t in graph[step['state_id']] if t['action_id']==step['action_id'])
            choice=next(a for a in step['menu_decision']['candidates'] if a['action_id']==step['action_id'])
            detailed.append({'title':edge['query']['public']['title'],'body_url':edge['body_url'],
                'actual_gain':step['signed_gain'],'value_decomposition':choice})
        changes.append({'physical_match_id':r['physical_match_id'],'budget':r['budget'],'cost_loss_units':r['cost_loss_units'],
            'one_step_actions':one_actions,'two_step_actions':two_actions,'net_difference':r['expected_net']-one['expected_net'],
            'two_step_reads':detailed})
    write(out/'changed-decisions.json',changes)
    lines=['二步模型机制验收与训练均值对照','',
        '这份对照在看到主实验结果后增加；固定奖励只由训练目标按比赛等权平均计算，未按留出收益调参。',
        '预算 | 成本 | 一步 | 固定后续奖励 | 学习式二步']
    report=read(out/'report.json')
    for s in summary:
        def row(policy):return next(r for r in report['policy_summary'] if r['policy']==policy and r['budget']==s['budget'] and r['cost_loss_units']==s['cost_loss_units'])
        lines.append(f"{s['budget']} | {s['cost_loss_units']:.3f} | {change(row('one_step')['net'])} | {change(s['net'])} | {change(row('two_step')['net'])}")
    lines+=['','发生动作变化的场景：']
    for case in changes:
        lines.append(f"{case['physical_match_id']}，预算{case['budget']}，成本{case['cost_loss_units']}；相对一步策略{change(case['net_difference'])}")
        for r in case['two_step_reads']:
            d=r['value_decomposition']
            lines.append(f"  {r['title']}；即时预计{change(d['immediate_gain'])}；预计后续净价值 {d['continuation_net_value']:.9f}；实际{change(r['actual_gain'])}")
    lines+=['','解释：当前收益改变来自额外读取一次；不是已证实的“先无收益、后获得有价值的新线索”链条。',
        '后续价值头改变了停止门槛，但本数据未证明其带来更好的第一动作排序或真正的间接信息价值。',
        '不能把高于一个固定均值对照解释为优于所有阈值策略；未进行阈值搜索。',
        '仍需独立的多球队序列验证，保留失败、零收益及负收益，不按已知收益挑样。']
    with (out/'机制验收.txt').open('x',encoding='utf-8') as handle:handle.write('\n'.join(lines)+'\n')
    write(out/'readback-verification.json',{'source_bindings_unchanged':len(plan['source_bindings']),
        'continuation_targets_recomputed':len(targets),'whole_match_exclusion_teachers_checked':len(teachers),
        'holdout_training_overlap':0,'policy_runs_exact_replay':len(runs),'control_runs':len(control_runs),
        'changed_decision_cells':len(changes),'network':0,'api_calls':0})
    print(json.dumps({'readback':read(out/'readback-verification.json'),'constant_control':summary,
                      'changed_decisions':[(r['physical_match_id'],r['budget'],r['cost_loss_units'],r['net_difference']) for r in changes]},ensure_ascii=False))


if __name__=='__main__':main()
