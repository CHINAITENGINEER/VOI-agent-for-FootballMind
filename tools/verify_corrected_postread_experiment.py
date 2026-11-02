"""Read-only verification of corrected diagnostic models; writes new reports only."""
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parent))
import correct_postread_reversal_experiment as experiment

b=experiment.base


def effective_model(name,selection):
    spec=selection['shrinkage'].get(name)
    if spec is None:return dict(named_model=name,effective_model=name,text_weight=None,baseline_only=name in ('zero','source_mean','train_mean'))
    return dict(named_model=name,effective_model=spec['baseline'] if spec['weight']==0 else (spec['text_model'] if spec['weight']==1 else name),
        text_weight=spec['weight'],baseline_only=spec['weight']==0)


def paired_interval(a,before):
    mids=sorted(set(a)&set(before))
    delta=np.array([a[m]-before[m] for m in mids])
    draws=np.random.default_rng(20261025).choice(delta,size=(4000,len(delta)),replace=True).mean(axis=1)
    return dict(matches=len(mids),mean_loss_difference=float(delta.mean()),
        percentile_95_interval=np.quantile(draws,[.025,.975]).tolist(),
        improved_matches=int((delta < -1e-12).sum()),worsened_matches=int((delta > 1e-12).sum()),
        tied_matches=int((abs(delta)<=1e-12).sum()),seed=20261025,resamples=4000,
        inference='exploratory previously-viewed data; no multiplicity correction')


def main():
    output=experiment.DEFAULT_OUTPUT
    if (output/'independent-verification.json').exists():raise FileExistsError('Refuse to overwrite existing verification')
    manifest=b.read(output/'manifest.json');plan=b.read(output/'experiment-plan.json')
    assert all(b.sha(output/name)==value for name,value in manifest['files'].items())
    assert all(b.sha(Path(name))==value for name,value in plan['source_sha256'].items())
    original_features=b.rows(b.DATA/'features.jsonl');samples=b.rows(b.DATA/'samples.jsonl')
    sample_by={s['sample_id']:s for s in samples};labels={r['sample_id']:r for r in b.rows(b.DATA/'labels.jsonl')}
    checks={}; effective={};ci={};numeric=[]
    with threadpool_limits(limits=1):
        for name,assignment in plan['named_splits'].items():
            folder=output/name
            features=[{**r,'split':assignment[r['physical_match_id']]} for r in original_features if assignment[r['physical_match_id']]!='unused']
            train_indices=np.array([i for i,r in enumerate(features) if r['split']=='train'])
            mids=np.array([r['physical_match_id'] for r in features]);y=np.array([labels[r['sample_id']]['realized_gain'] for r in features])
            model=joblib.load(folder/'models.joblib');design=experiment.transform_design(features,model['encoders'])
            expected={(r['model'],r['sample_id']):r['predicted_gain'] for r in b.rows(folder/'predictions.jsonl')}
            pred=dict(zero=np.zeros(len(features)),train_mean=np.full(len(features),model['target_mean']))
            source=model['source_mean']
            pred['source_mean']=np.array([source['domain_means'].get(sample_by[r['sample_id']]['source_domain'],source['fallback_match_equal_training_mean']) for r in features])
            for key,spec in model['fitted'].items():
                scaler=spec['scaler'];pred[key]=spec['estimator'].predict(scaler.transform(design[spec['input_key']]))*model['target_scale']+model['target_mean']
                assert np.isfinite(scaler.scale_).all() and (scaler.scale_>0).all() and np.isfinite(pred[key]).all()
                mean_expected=np.average(design[spec['input_key']][train_indices],axis=0,weights=b.match_weights(mids[train_indices]))
                assert np.max(abs(mean_expected-scaler.mean_))<1e-12
                if np.any(scaler.var_<0):
                    numeric.append(dict(fold=name,model=key,minimum_variance=float(scaler.var_.min()),
                        all_scales_and_predictions_finite=True,interpretation='tiny negative floating variance in constant column; sklearn constant scale remains finite'))
            for key,spec in model['shrinkage'].items():pred[key]=spec['weight']*pred[spec['text_model']]+(1-spec['weight'])*pred[spec['baseline']]
            pred['validation_selected']=pred[model['selection']['selected_model']]
            pred['validation_selected_fulltext']=pred[model['selection']['selected_fulltext_family']]
            error=max(abs(float(p[i])-expected[key,r['sample_id']]) for key,p in pred.items() for i,r in enumerate(features))
            assert error<1e-12
            recomputed=[dict(model=key,split=s,mse=b.macro((p[ii]-y[ii])**2,mids[ii])) for key,p in pred.items()
                if key not in ('validation_selected','validation_selected_fulltext')
                for s in b.SPLITS for ii in [np.array([i for i,r in enumerate(features) if r['split']==s])]]
            saved={(r['model'],r['split']):r['mse'] for r in b.read(folder/'metrics.json')}
            assert max(abs(r['mse']-saved[r['model'],r['split']]) for r in recomputed)<1e-12
            checks[name]=dict(saved_encoder_model_prediction_max_error=error,metrics_recomputed=len(recomputed),model_sha256=b.sha(folder/'models.joblib'))
            sel=b.read(folder/'selection-frozen.json')
            effective[name]=dict(overall=effective_model(sel['selected_model'],sel),fulltext_family=effective_model(sel['selected_fulltext_family'],sel))
            paths=b.rows(folder/'policy-paths.jsonl')
            policy={strategy:{r['physical_match_id']:r['final_loss'] for r in paths if r['budget']==2 and r['strategy']==strategy}
                for strategy in ('no_information','random_exact','validation_selected_fulltext')}
            ci[name]={baseline:paired_interval(policy['validation_selected_fulltext'],policy[baseline]) for baseline in ('no_information','random_exact')}
    random=b.read(output/'random-resplits.json');effective_random=Counter()
    for row in random:
        sel={'shrinkage':row['shrinkage']};effective_random[effective_model(row['selected'],sel)['effective_model']]+=1
    result=dict(status='passed',manifest_files_checked=len(manifest['files']),source_files_unchanged=True,
        folds=checks,effective_selections=effective,effective_random_winners=dict(effective_random),
        paired_final_policy_intervals_budget2=ci,numerical_diagnostics=numeric,
        training_calls=0,network_requests=0,model_parameter_changes=0,
        note='This verifier transforms and predicts from saved model bytes; it does not refit encoders or estimators.')
    b.save(output/'independent-verification.json',result)
    lines=['修正实验的独立验收与解释补充','',
        f'源数据、旧实验和冻结f摘要一致；本包最初{len(manifest["files"])}份产物摘要通过。',
        '从序列化编码器和模型重新生成每条预测，重新计算全部MSE；各命名切分误差均小于1e-12。没有重新训练。','',
        '特别注意：名称属于全文家族，不代表最终仍在使用正文；当权重为0时，实际退化为基线。']
    for name,value in effective.items():
        x=value['fulltext_family'];lines.append(f'{name}: 全文权重={x["text_weight"]}，实际模型={x["effective_model"]}，只用基线={x["baseline_only"]}。')
    lines.extend(['',f'30次重排按实际有效模型统计：{dict(effective_random)}。','',
        '原切分的6场测试，最多采用2篇：全文方案相对基线的逐场配对比较（差值为模型loss减基线loss，负值表示降低）：'])
    for baseline,value in ci['P0_original'].items():
        lines.append(f'{baseline}: 平均差={value["mean_loss_difference"]:.9f}；95%区间={value["percentile_95_interval"]}；改善{value["improved_matches"]}场、变差{value["worsened_matches"]}场、相同{value["tied_matches"]}场。')
    lines.extend(['','上述区间为已查看小样本的探索性bootstrap，不能据此宣布新模型独立验证成功。',
        '当前结果暴露两个不同问题：回归MSE最优不一定对应策略loss最优；收缩可能帮助，也可能过早停止并错过有用证据。',
        '来源识别：新版遮盖标题测试准确率33.33%来自独立6场/12篇；旧81.3%来自全部289状态行上的样本内计算，两者不是同一统计量。',
        '训练多数类基线测试准确率为0，是训练多数来源Leeds在该测试组缺席；这反映来源分布变化，不是把0作为普遍随机准确率。',
        f'数值诊断：{len(numeric)}个标准化器存在极小的负方差浮点残差；已逐一核对scale、模型参数输出与全部预测为有限数，未改写权重。详见JSON。',
        '最终判断：评估链路修正完成，模型泛化仍不稳定。下一步应在开发集内将模型选择与完整策略损失对齐，并保留独立旧归档评估集。正式f/g不替换。'])
    (output/'acceptance-notes.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    shutil.copy2(__file__,output/'verification-script.py')
    shutil.copy2(experiment.__file__,output/'current-rerun-script.py')
    shutil.copy2(b.__file__,output/'shared-diagnostic-helpers.py')
    print(json.dumps(dict(status='passed',folds=len(checks),max_prediction_error=max(r['saved_encoder_model_prediction_max_error'] for r in checks.values()),
        effective_selections=effective,numerical_notes=len(numeric)),ensure_ascii=True))


if __name__=='__main__':main()
