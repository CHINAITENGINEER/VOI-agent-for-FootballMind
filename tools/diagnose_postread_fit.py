"""Measure fitted training error and held-out error; no model selection/promotion."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.train_e_drive_postread_g import ResearchData, SelectorInputs
from football_research.acquisition.postread_g_use import score, neural_objective, fit_selector
from football_research.acquisition.corpus_rolling_g import match_weights


def read(path):return json.loads(path.read_text(encoding='utf8'))
def rows(path):return [json.loads(l) for l in path.read_text(encoding='utf8').splitlines()]


def main():
    ap=argparse.ArgumentParser()
    for k in ('root','experiment','output'):ap.add_argument('--'+k,type=Path,required=True)
    ap.add_argument('--skip-capacity-probes',action='store_true')
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    panel=rows(a.experiment/'panel.jsonl');data=ResearchData(a.root,a.experiment,panel);inputs=SelectorInputs(data)
    tr=rows(a.experiment/'training-transitions.jsonl');vr=rows(a.experiment/'validation-transitions.jsonl')
    y=np.array([r['label']['signed_gain'] for r in tr]);yv=np.array([r['label']['signed_gain'] for r in vr])
    mids=[r['match_id'] for r in tr];w=match_weights(mids);wv=match_weights([r['match_id'] for r in vr])
    center=float(w@y);variance=float(w@((y-center)**2))
    def metrics(pred,target,weight):
        mse=float(weight@((pred-target)**2));base=float(weight@((target-center)**2))
        return {'weighted_mse':mse,'rmse':float(np.sqrt(mse)),
                'r2_vs_training_mean':1-mse/base,'prediction_mean':float(weight@pred),
                'prediction_sd':float(np.sqrt(weight@((pred-weight@pred)**2))),
                'weighted_sign_accuracy':float(weight@((pred>0)==(target>0)))}
    results=[];matrices={};prediction_rows=[]
    for mode in ('title_local','title_state','body_only','body_state'):
        x=np.load(a.experiment/f'{mode}-train-vectors.npy')
        regenerated=np.array([inputs.vector(mode,r['match_id'],r['selector_state']['f_adopted_ids'],
            r['selector_state']['candidate_id'],r['selector_state']['probabilities_before']) for r in tr])
        np.testing.assert_allclose(x,regenerated,atol=1e-12,rtol=0)
        xv=np.array([inputs.vector(mode,r['match_id'],r['selector_state']['f_adopted_ids'],
            r['selector_state']['candidate_id'],r['selector_state']['probabilities_before']) for r in vr])
        matrices[mode]=(x,xv)
        selected=read(a.experiment/f'g-{mode}-selected.json')
        for path in sorted(a.experiment.glob(f'g-{mode}-*.json')):
            if path.name.endswith('-selected.json'):continue
            m=read(path);item={'file':path.name,'mode':mode,'kind':m['kind'],'alpha':m['alpha'],
                'parameters':m['parameter_count'],'selected':m['kind']==selected['kind'] and m['alpha']==selected['alpha'],
                'train':metrics(score(m,x),y,w),'validation':metrics(score(m,xv),yv,wv),
                'optimizer':m.get('optimizer')}
            if m['kind']=='mlp':
                theta=np.asarray(m['parameters']);z=(x-m['mean'])/m['scale'];yn=(y-m['target_center'])/m['target_scale']
                val,grad=neural_objective(theta,z,yn,w,m['hidden'],m['alpha'])
                item.update(penalized_objective=val,gradient_inf_norm=float(np.max(abs(grad))),
                            penalty=.5*m['alpha']*float(theta@theta))
            results.append(item)
            if mode=='body_state' and item['selected']:
                for split, matrix, target, weight, records in [('train',x,y,w,tr),('validation',xv,yv,wv,vr)]:
                    for r,t,p,wt in zip(records,target,score(m,matrix),weight):
                        prediction_rows.append({'split':split,'match_id':r['match_id'],'actual_gain':float(t),
                            'predicted_gain':float(p),'weight':float(wt)})
    # Train-only capacity probes; no validation/test scores or model promotion.
    probes=[];x=matrices['body_state'][0]
    for alpha in (() if a.skip_capacity_probes else (1e-4,1e-6)):
        m=fit_selector(x,y,mids,kind='ridge',alpha=alpha,mode='body_state')
        probes.append({'kind':'ridge','alpha':alpha,'train':metrics(score(m,x),y,w),
                       'purpose':'training_only_fit_capacity_probe','heldout_evaluated':False})
        print('CAPACITY',alpha,probes[-1]['train']['weighted_mse'],flush=True)
    report={'training_matches':len(set(mids)),'training_rows':len(y),'validation_rows':len(yv),
        'weights':'equal total per physical match','training_target_mean':center,'training_target_sd':float(np.sqrt(variance)),
        'baselines':{'train_zero_mse':float(w@(y*y)),'train_mean_mse':variance,
                     'validation_zero_mse':float(wv@(yv*yv)),'validation_training_mean_mse':float(wv@((yv-center)**2))},
        'models':results,'capacity_probes':probes,'test_evaluated':False,'model_promoted':False,
        'saved_training_vectors_independently_regenerated':True,
        'interpretation':'Convergence of penalized objective does not establish unpenalized training fit or generalization.'}
    (a.output/'fit-diagnostics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    with (a.output/'selected-predictions.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['split','match_id','actual_gain','predicted_gain','weight'])
        writer.writeheader();writer.writerows(prediction_rows)
    print(json.dumps({**{k:v for k,v in report.items() if k!='models'},'models':[r for r in results if r['mode']=='body_state']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
