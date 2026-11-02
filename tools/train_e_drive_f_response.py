"""Earlier-only f model selection and body-specific response controls, no test evaluation."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from football_research.acquisition import convex_text_f
from football_research.acquisition.corpus_text_f import loss_each
from football_research.acquisition.corpus_rolling_g import subsets
from tools.train_e_drive_rolling_g import Data,purge,dump,write_rows,digest


def fit(data,panels,branch,alpha):
    x,b,y,w=[],[],[],[]
    for p in panels:
        mid=p['match_id']; ss=subsets([c['article_id'] for c in p['candidates']])[1:]
        for s in ss:
            x.append(data.feature(mid,s)); b.append(data.prior(mid))
            y.append(data.y[data.positions[mid]]); w.append(1/(len(panels)*len(ss)))
    m=convex_text_f.fit(x,b,y,w,alpha=alpha,branch=branch,metadata_dimension=15+len(data.hosts))
    m.update(fit_match_ids=[p['match_id'] for p in panels],host_order=data.hosts,
        fit_max_date=max(p['date'] for p in panels))
    return m


def prediction(data,model,mid,ids):
    if model is None:
        return data.prior(mid)
    return convex_text_f.predict(model,data.feature(mid,ids)[None],data.prior(mid)[None],[len(ids)])[0]


def evaluate(data,model,panels):
    probabilities=[]; rows=[]
    for p in panels:
        mid=p['match_id']; ids=[c['article_id'] for c in p['candidates']]
        pred=prediction(data,model,mid,ids); probabilities.append(pred)
        y=int(data.y[data.positions[mid]])
        rows.append(dict(match_id=mid,probabilities=pred.tolist(),outcome_index=y,
            log_loss=float(-np.log(pred[y]))))
    return float(np.mean([r['log_loss'] for r in rows])),rows


def run_stage(data,history,targets,out,name):
    history=purge(history,targets)
    boundary=history[int(len(history)*.75)]['date']
    inner_val=[p for p in history if p['date']>=boundary]
    inner_train=purge([p for p in history if p['date']<boundary],inner_val)
    assert max(p['date'] for p in history)<min(p['date'] for p in targets)
    summaries={}; models={}
    base_inner,_=evaluate(data,None,inner_val)
    for branch in ['bias','metadata','body']:
        grid=[]
        for alpha in [.1,1.,10.,100.]:
            model=fit(data,inner_train,branch,alpha)
            ll,_=evaluate(data,model,inner_val)
            grid.append(dict(alpha=alpha,inner_log_loss=ll))
        selected=min(grid,key=lambda r:r['inner_log_loss'])
        model=fit(data,history,branch,selected['alpha'])
        model.update(inner_train_ids=[p['match_id'] for p in inner_train],
            inner_validation_ids=[p['match_id'] for p in inner_val],grid=grid,
            base_model_sha256=digest(out/'f-base.json'))
        dump(out/f'{name}-f-{branch}.json',model); models[branch]=model
        summaries[branch]=dict(grid=grid,selected=selected,parameter_count=model['parameter_count'])
    candidates={'no_information':base_inner,**{b:s['selected']['inner_log_loss'] for b,s in summaries.items()}}
    winner=min(candidates,key=candidates.get)
    # Save selection before consulting any evaluation outcomes.
    selection=dict(selected=winner,inner_scores=candidates,fit_ids=[p['match_id'] for p in history],
        target_ids=[p['match_id'] for p in targets],selection_uses_target_results=False)
    dump(out/f'{name}-selection.json',selection)
    evaluated={}; details=[]
    for branch,model in {'no_information':None,**models}.items():
        ll,rs=evaluate(data,model,targets)
        evaluated[branch]=ll
        details.extend(dict(branch=branch,**r) for r in rs)
    # Same metadata, actual candidate embeddings varied: quantify content response.
    shifts=[]; gain_rows=[]
    for p in targets:
        mid=p['match_id']; ids=[c['article_id'] for c in p['candidates']]
        model=models['body']; states=subsets(ids)
        cache={s:prediction(data,model,mid,s) for s in states}
        for s in states:
            for aid in ids:
                if aid in s: continue
                nxt=tuple(i for i in ids if i in set(s)|{aid}); before,after=cache[s],cache[nxt]
                y=int(data.y[data.positions[mid]])
                gain_rows.append(dict(match_id=mid,adopted=list(s),candidate_id=aid,
                    probabilities_before=before.tolist(),probabilities_after=after.tolist(),
                    signed_gain=float(np.log(after[y])-np.log(before[y])),outcome_index=y,
                    f_model_sha256=digest(out/f'{name}-f-body.json'),research_only=True))
                if s: shifts.append(float(np.max(abs(after-before))))
    # Text ablation: zero standardized text, preserve metadata and intercept exactly.
    m=models['body']; text_rows=[j for j,c in enumerate(m['columns']) if c>=15+len(data.hosts)]
    ablated=json.loads(json.dumps(m))
    for j in text_rows: ablated['coef'][j]=[0.,0.,0.]
    ablation_ll,_=evaluate(data,ablated,targets)
    write_rows(out/f'{name}-predictions.jsonl',details)
    write_rows(out/f'{name}-body-transitions.jsonl',gain_rows)
    source_counts=Counter(data.corpus.articles[c['article_id']]['source_team'] for p in history for c in p['candidates'])
    report=dict(fit_matches=len(history),target_matches=len(targets),source_team_links=dict(source_counts),
        inner_base_loss=base_inner,selection=selection,candidates=summaries,evaluation_log_loss=evaluated,
        body_zeroed_text_log_loss=ablation_ll,body_specific_loss_reduction=ablation_ll-evaluated['body'],
        nonempty_transition_count=len(shifts),nonempty_probability_changes_above_1e_10=sum(s>1e-10 for s in shifts),
        max_nonempty_probability_change=max(shifts,default=0.),
        body_gain_signs=dict(Counter('positive' if r['signed_gain']>1e-12 else 'negative' if r['signed_gain']< -1e-12 else 'near_zero' for r in gain_rows)),
        g_labels_exported_diagnostic_only=True)
    print(name,json.dumps(report,ensure_ascii=False),flush=True)
    return report


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--previous',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(); out=a.output; out.mkdir(parents=True,exist_ok=False); started=time.perf_counter()
    panel=[json.loads(l) for l in (a.previous/'panel.jsonl').read_text(encoding='utf-8').splitlines()]
    train=[p for p in panel if p['split']=='train']; val=[p for p in panel if p['split']=='validation']
    boundary=train[24]['date']; early=[p for p in train if p['date']<boundary]; later=[p for p in train if p['date']>=boundary]
    dump(out/'protocol.json',dict(schema='footballmind-f-response-repair-study-v1',
        models=['no_information','convex_bias','convex_metadata','convex_body'],alphas=[.1,1.,10.,100.],
        selection='inner_last_quarter_dates_log_loss_then_refit_history',
        stages={'fixed24':dict(history=[p['match_id'] for p in early],evaluation=[p['match_id'] for p in later]),
                'expanded74':dict(history=[p['match_id'] for p in train],evaluation=[p['match_id'] for p in val])},
        no_gain_or_probability_change_selection=True,test46_evaluated=False,
        same_data_previously_examined=True,prospective=False,production_promotion=False,
        cost_penalty=0.,body_ablation='zero_text_coefficients_keep_metadata_and_intercept'))
    (out/'f-base.json').write_bytes((a.previous/'f-base.json').read_bytes())
    write_rows(out/'panel.jsonl',panel)
    data=Data(a.root,a.previous,panel)
    report={name:run_stage(data,h,t,out,name) for name,h,t in [('fixed24',early,later),('expanded74',train,val)]}
    report.update(duration_seconds=time.perf_counter()-started,network_calls=0,api_calls=0,gpu=False,
        test46_evaluated=False,production_modified=False)
    dump(out/'report.json',report)
    dump(out/'manifest.json',{p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'})


if __name__=='__main__':
    with threadpool_limits(limits=1): main()
