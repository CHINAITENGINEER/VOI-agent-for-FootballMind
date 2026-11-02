"""Forward validation of state-depth weighting, without tuning against external results."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.train_postread_stop_policy import temporal_folds, run_policy
from tools.train_e_drive_postread_g import ResearchData, SelectorInputs, evaluate
from tools.train_e_drive_rolling_g import transitions, dump, write_rows, digest
from football_research.acquisition.postread_g_use import fit_selector, score
from football_research.acquisition.postread_state_weights import MODES, state_weights


def rows(path):return [json.loads(l) for l in path.read_text(encoding='utf8').splitlines()]


def main():
    ap=argparse.ArgumentParser()
    for key in ('root','previous','output'):ap.add_argument('--'+key,type=Path,required=True)
    a=ap.parse_args(); start=time.perf_counter(); out=a.output;out.mkdir(parents=True,exist_ok=False)
    panel=rows(a.previous/'panel.jsonl'); prior=json.loads((a.previous/'protocol.json').read_text(encoding='utf8'))
    train=[p for p in panel if p['match_id'] in set(prior['g_fit_ids'])];folds=temporal_folds(train)
    protocol={'schema':'postread-depth-balance-forward-v1','modes':MODES,
        'alpha':10.,'learner':'ridge','mode':'body_state','threshold':0.,'adoption_cap':3,
        'selection':'minimum_forward_mean_loss; change only if paired reduction exceeds one paired SE',
        'same_match_total_weight_all_modes':True,'outcome_based_row_filter':False,
        'reachable_mode_zero_weights_only_depth_gte_cap':True,'all_rows_retained':True,
        'f_sha256':digest(a.previous/'f-body.json'),'g_fit_ids':prior['g_fit_ids'],
        'external_evaluation_previously_examined':True,'external_outcomes_for_selection':False,
        'alpha_inherited_from_prior_research':True,'production_promoted':False,
        'folds':[{'fit_ids':[p['match_id'] for p in h],'holdout_ids':[p['match_id'] for p in t]} for h,t in folds]}
    dump(out/'protocol.json',protocol)
    data=ResearchData(a.root,a.previous,panel);inputs=SelectorInputs(data)
    f=json.loads((a.previous/'f-body.json').read_text(encoding='utf8'))
    tr,_,states=transitions(data,train,f,protocol['f_sha256'],'train')
    x=inputs.matrix('body_state',tr); y=np.array([r['label']['signed_gain'] for r in tr])
    mids=[r['match_id'] for r in tr];depths=[len(r['query']['already_read_ids']) for r in tr]
    def fit(idx,mode):
        mm=[mids[i] for i in idx];dd=[depths[i] for i in idx]
        weights=state_weights(mm,dd,mode=mode,adoption_cap=3)
        model=fit_selector(x[idx],y[idx],mm,kind='ridge',alpha=10.,mode='body_state',sample_weights=weights)
        model.update(training_weight_mode=mode,f_model_sha256=protocol['f_sha256'])
        return model
    oof=[];trace=[];fold_summary=[]
    for k,(history,target) in enumerate(folds,1):
        fitids={p['match_id'] for p in history};idx=[i for i,m in enumerate(mids) if m in fitids]
        for mode in MODES:
            model=fit(idx,mode);dump(out/f'fold-{k}-{mode}.json',model)
            runs=run_policy(inputs,target,states,model,0.,3)
            stats,detail=evaluate(data,target,states,runs,3)
            oof.extend({'fold':k,'weight_mode':mode,**d} for d in detail)
            trace.extend({'fold':k,'weight_mode':mode,**r} for r in runs)
            fold_summary.append({'fold':k,'weight_mode':mode,'fit_matches':len(history),
                'holdout_matches':len(target),'training_rows':len(idx),**stats})
        print('FORWARD_FOLD',k,'fit',len(history),'holdout',len(target),flush=True)
    write_rows(out/'forward-per-match.jsonl',oof);write_rows(out/'forward-traces.jsonl',trace)
    loss={m:np.array([r['loss']['g'] for r in oof if r['weight_mode']==m]) for m in MODES}
    base=MODES[0];grid=[]
    for mode in MODES:
        delta=loss[base]-loss[mode]
        grid.append({'weight_mode':mode,'mean_loss':float(loss[mode].mean()),
            'paired_loss_reduction_vs_original':float(delta.mean()),
            'paired_standard_error':float(delta.std(ddof=1)/np.sqrt(len(delta))),
            'improved':int(sum(delta>1e-12)),'worsened':int(sum(delta < -1e-12)),
            'unchanged':int(sum(abs(delta)<=1e-12))})
    best=min(grid,key=lambda r:r['mean_loss'])
    selected=best['weight_mode'] if best['paired_loss_reduction_vs_original']>best['paired_standard_error'] else base
    selection={'selected':selected,'grid':grid,'oof_matches':len(loss[base]),
        'external_outcomes_used':False,'one_se_rule_not_significance_test':True}
    dump(out/'selection.json',selection) # Selection frozen before external evaluation.
    models={}
    for mode in MODES:
        models[mode]=fit(list(range(len(tr))),mode);dump(out/f'g-{mode}.json',models[mode])
    original=json.loads((a.previous/'g-body_state-ridge-10.0.json').read_text(encoding='utf8'))
    np.testing.assert_allclose(score(original,x),score(models[base],x),atol=1e-12,rtol=0)
    external={};replay_count=0
    for split,key in [('validation','g_validation_ids'),('test','evaluation_ids')]:
        group=[p for p in panel if p['match_id'] in set(prior[key])]
        _,_,sp=transitions(data,group,f,protocol['f_sha256'],split);external[split]={}
        for mode in MODES:
            reloaded=json.loads((out/f'g-{mode}.json').read_text(encoding='utf8'))
            runs=run_policy(inputs,group,sp,reloaded,0.,3)
            assert runs==run_policy(inputs,group,sp,models[mode],0.,3)
            stats,detail=evaluate(data,group,sp,runs,3);external[split][mode]=stats
            write_rows(out/f'{split}-{mode}-per-match.jsonl',detail)
            write_rows(out/f'{split}-{mode}-traces.jsonl',runs)
            replay_count+=sum(len(r['trace']) for r in runs)
    weights={}
    for mode in MODES:
        w=state_weights(mids,depths,mode=mode)
        weights[mode]={str(d):float(sum(w[i] for i,v in enumerate(depths) if v==d)) for d in sorted(set(depths))}
    report={'selection':selection,'folds':fold_summary,'training_weight_mass_by_depth':weights,
        'evaluation':external,'original_ridge_reproduced':True,'saved_model_exact_replay_decisions':replay_count,
        'training_rows':len(tr),'training_gain_signs':dict(Counter('positive' if v>1e-12 else 'negative' if v < -1e-12 else 'zero' for v in y)),
        'elapsed_seconds':time.perf_counter()-start,'network':0,'paid_api':0,'gpu':False,
        'production_changed':False,'new_independent_matches':0}
    dump(out/'report.json',report)
    text=['g 状态深度训练权重对照','',
        '假设：同场全子集枚举使中间状态样本偏多；本轮只改训练权重，不改f、文本向量、g结构或停止阈值。',
        '三种方式：原每场等权/场内边等权；每场等权/各深度等权；每场等权/三篇预算下可达深度等权。',
        '第三种将已经采用三篇后的动作权重设0，因为这次策略三篇预算已用尽；不按收益符号删样本。',
        '59场训练中按日期前向三折，严格用过去训练并剔除跨折同正文，覆盖45场折外比赛。',
        '所有超参数固定，只有权重方式用折外完整策略损失选择。外部验证/测试以前已看过，不称新盲测。','',
        '折外结果（log loss越小越好）：']
    for r in grid:text.append(f"{r['weight_mode']}: {r['mean_loss']:.6f}；相对原权重"+('损失减少' if r['paired_loss_reduction_vs_original']>=0 else '损失增加')+f"{abs(r['paired_loss_reduction_vs_original']):.6f}；配对标准误{r['paired_standard_error']:.6f}")
    text.extend(['','预声明规则选中：'+selected,'', '外部评估（只报告，不反选模型）：'])
    for split in external:
        for mode,stats in external[split].items():text.append(f"{split}/{mode}: {stats['mean_log_loss']['g']:.6f}")
    text.extend(['','未上线替换；样本权重实验不能证明信息选择优于合理基线。',
        '原ridge重现通过，保存模型决策回放一致。完整分折、逐场结果和权重质量分布见report.json。'])
    (out/'训练权重对照说明.txt').write_text('\n'.join(text)+'\n',encoding='utf8')
    dump(out/'manifest.json',{p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'})
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
