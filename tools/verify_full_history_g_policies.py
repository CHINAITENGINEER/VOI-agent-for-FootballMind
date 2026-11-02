"""Independently verify saved frozen policies and bootstrap paired match effects."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def read_json(p):return json.loads(p.read_text(encoding="utf8"))
def rows(p):return [json.loads(s) for s in p.read_text(encoding="utf8").split("\n") if s]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)


def verify(directory):
    report=read_json(directory/'strategy-comparison.json')
    manifest=read_json(directory/'manifest.json')
    require(all(sha(directory/name)==digest for name,digest in manifest.items()),'Output hash mismatch')
    paths=rows(directory/'per-match-paths.jsonl')
    failures=rows(directory/'failures.jsonl')
    predictions=rows(directory/'all-frozen-head-predictions.jsonl')
    labels=rows(directory/'labels.jsonl')
    ll_diff=0.;gain_diff=0.
    groups=defaultdict(list);roles=defaultdict(set);pools={};identity={}
    for r in paths:
        mid=r['physical_match_id'];role=r['split_role'];roles[role].add(mid)
        require(role in ('validation','test'),'Unexpected heldout role')
        before=np.asarray(r['p_before']);after=np.asarray(r['p_after']);y=r['outcome_class']
        require(np.isfinite(before).all() and np.isfinite(after).all() and min(before)>0 and min(after)>0,'Invalid probability')
        ll=float(-np.log(after[y]));gain=float(np.log(after[y])-np.log(before[y]))
        ll_diff=max(ll_diff,abs(ll-r['final_log_loss']));gain_diff=max(gain_diff,abs(gain-r['realized_gain']))
        require(len(r['selected_ids'])==len(set(r['selected_ids']))==r['f_updates'],'Duplicate adoption/count mismatch')
        pool=set(r['selected_ids'])|set(r['unselected_ids'])
        require(not set(r['selected_ids'])&set(r['unselected_ids']) and len(pool)==r['candidate_count'],'Pool partition mismatch')
        if mid in pools:require(pool==pools[mid],'Strategy pool changed')
        else:pools[mid]=pool
        key=(tuple(r['p_before']),y,r['previous_development_exposure'],r['f_model_sha256'])
        if mid in identity:require(identity[mid]==key,'Baseline/outcome/model differs between policies')
        else:identity[mid]=key
        if r['strategy']=='all':require(r['f_updates']==r['candidate_count'],'All strategy truncated pool')
        elif r['strategy']=='none':require(r['f_updates']==0 and np.array_equal(before,after),'None strategy updated f')
        else:require(r['f_updates']<=r['budget'],'Exceeded f update budget')
        groups[(role,r['strategy'],r['budget'])].append(r)
    require(ll_diff<1e-12 and gain_diff<1e-12,'Loss/gain formula mismatch')
    require(len(roles['validation'])==193 and len(roles['test'])==320 and not roles['validation']&roles['test'],'Heldout match coverage mismatch')
    for s in report['summary']:
        rr=groups[(s['split_role'],s['strategy'],s['budget'])]
        expected=10 if s['strategy']=='random' else 1
        counts=Counter(r['physical_match_id'] for r in rr)
        require(set(counts)==roles[s['split_role']] and set(counts.values())=={expected},'Incomplete strategy/match coverage')
        require(abs(np.mean([r['final_log_loss'] for r in rr])-s['mean_final_log_loss'])<1e-12,'Strategy summary mismatch')
    labelmap={r['sample_id']:r for r in labels}
    require(len(labelmap)==len(labels),'Duplicate gain sample IDs')
    byhead=defaultdict(list)
    for r in predictions:
        target=labelmap[r['sample_id']]
        require(r['physical_match_id']==target['physical_match_id'] and r['realized_gain']==target['realized_gain'],'Prediction label ID mismatch')
        byhead[(r['model'],r['split_role'])].append(r)
    for m in read_json(directory/'gain-metrics.json'):
        rr=byhead[(m['model'],m['split_role'])]
        e=np.array([r['predicted_gain']-r['realized_gain'] for r in rr])
        per=defaultdict(list)
        for r,err in zip(rr,e):per[r['physical_match_id']].append(err*err)
        require(abs(float(np.mean(e*e))-m['mse'])<1e-12 and abs(float(np.mean([np.mean(v) for v in per.values()]))-m['match_macro_mse'])<1e-12,'Gain metric mismatch')
    oracle=rows(directory/'oracle.jsonl');oraclemap={(r['physical_match_id'],r['budget']):r for r in oracle}
    require(len(oraclemap)==513*4,'Oracle coverage missing')
    for r in paths:
        o=oraclemap[(r['physical_match_id'],r['budget'])]
        if o.get('oracle_upper_bound') and not (r['strategy']=='all' and r['candidate_count']>r['budget']):
            require(o['final_log_loss']<=r['final_log_loss']+1e-6,'Oracle is worse than feasible policy')
    boot=[];rng=np.random.default_rng(20261025)
    for role in ('validation','test'):
        for exposure in ('all',False,True):
            mids=sorted(mid for mid in roles[role] if exposure=='all' or identity[mid][2] is exposure)
            selection={strategy:{mid:float(np.mean([r['final_log_loss'] for r in groups[(role,strategy,3)] if r['physical_match_id']==mid])) for mid in mids} for strategy in ('new_g','none','all','fixed','random')}
            for comparator in ('none','all','fixed','random'):
                delta=np.array([selection[comparator][mid]-selection['new_g'][mid] for mid in mids])
                resamples=np.mean(delta[rng.integers(0,len(delta),size=(2000,len(delta)))],axis=1)
                boot.append(dict(split_role=role,previous_development_exposure=exposure,comparison='new_g_vs_'+comparator,budget=3,
                    matches=len(mids),mean_log_loss_reduction=float(delta.mean()),ci95=[float(v) for v in np.quantile(resamples,[.025,.975])],
                    improved=int((delta>1e-12).sum()),equal=int((abs(delta)<=1e-12).sum()),worse=int((delta < -1e-12).sum()),
                    bootstrap_unit='physical_match_id',bootstrap_draws=2000,seed=20261025,
                    all_comparator_is_uncapped=comparator=='all'))
    return dict(status='passed',directory=str(directory),matches={r:len(v) for r,v in roles.items()},paths=len(paths),labels=len(labels),
        exact_gain_recalculation_max_difference=gain_diff,loss_recalculation_max_difference=ll_diff,
        failures_by_reason=dict(Counter(r.get('reason','UNKNOWN') for r in failures)),oracle_status=dict(Counter(r['status'] for r in oracle)),
        exposure_counts={r:{str(e):sum(identity[mid][2] is e for mid in mids) for e in (False,True,None)} for r,mids in roles.items()},
        budget3_summary=[r for r in report['summary'] if r['budget']==3],paired_match_bootstrap=boot,
        new_training=False,threshold_changed=False,source_requests=0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluations',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=dict(status='passed',evaluations=[verify(d) for d in a.evaluations])
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(status='passed',output=str(a.output))))


if __name__=='__main__':main()
