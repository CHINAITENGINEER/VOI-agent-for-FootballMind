"""Candidate g diagnostic after the 15-Oct f expansion.

The feature set is intentionally small and auditable. It is not the frozen
runtime g contract and cannot acquire selection authority.
"""
from pathlib import Path
from collections import defaultdict
import json, math, hashlib
import numpy as np
from sklearn.linear_model import Ridge

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'var/reviews/fm-historical-transition-g-training-20261014-v2/transitions.json'
NEW=ROOT/'var/reviews/fm-g-nonleeds-expansion-20261015-v1/evaluation/prediction-input-rows.json'
OUT=ROOT/'var/reviews/fm-g-nonleeds-expansion-20261015-v1/g-diagnostic'
FEATURES=['p_home','p_draw','p_away','entropy','acquired_documents','visible_candidates','title_availability','title_selection','title_training','title_preview','title_other','l1_source_delta']
def H(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def entropy(p): return float(-sum(x*math.log(max(x,1e-12)) for x in p))
def title_bits(title):
    t=(title or '').casefold()
    return [float(any(x in t for x in ('injury','injuries','availability','fitness','suspension'))),float(any(x in t for x in ('line-up','lineup','line up','xi'))),float('training' in t or 'cobham' in t),float('preview' in t),0.0]
def vec(p,docs,title,l1=0.):
    bits=title_bits(title); return [*p,entropy(p),float(docs),1.,*bits[:4],float(not any(bits[:4])),float(l1)]
def fit(rows,hold=None):
    train=[r for r in rows if r['physical_match_id'] not in (hold or set())]
    x=np.asarray([r['x'] for r in train]); y=np.asarray([r['gain'] for r in train]); m=Ridge(alpha=.1).fit(x,y); return m
def main():
    old=H(OLD); new=H(NEW); rows=[]
    for t in old:
        q=t['query']; p=t['prediction_before']; rows.append({'physical_match_id':t['physical_match_id'],'gain':float(t['signed_gain']),'x':vec(p,q.get('acquired_documents',0),q.get('title',''),0.)})
    grouped=defaultdict(list)
    for r in new: grouped[r['physical_match_id']].append(r)
    names={'understat:28891':'Burnley vs Chelsea statistics/team news','understat:28907':'Confirmed Liverpool line-up v West Ham United'}
    outcomes={'understat:28891':2,'understat:28907':2}
    new_transitions=[]
    for mid, rs in grouped.items():
        # Create every one-document transition from a smaller visible state.
        for a in rs:
            sa=set(a['used_input_ids'])
            for b in rs:
                sb=set(b['used_input_ids'])
                if len(sb)==len(sa)+1 and sa < sb:
                    y=outcomes[mid]; gain=math.log(max(b['prediction_after'][y],1e-12))-math.log(max(a['prediction_after'][y],1e-12))
                    x=vec(a['prediction_after'],len(sa),names[mid],float(a['l1_shift']))
                    new_transitions.append({'physical_match_id':mid,'gain':gain,'x':x,'from_state':a['used_input_ids'],'to_state':b['used_input_ids']})
    # Deduplicate identical set transitions.
    uniq={ (r['physical_match_id'],tuple(sorted(r['from_state'])),tuple(sorted(r['to_state']))):r for r in new_transitions }
    new_transitions=list(uniq.values()); allrows=rows+new_transitions; OUT.mkdir(parents=True,exist_ok=True)
    split={}
    for mid in ['understat:28891','understat:28907']:
        test=[r for r in new_transitions if r['physical_match_id']==mid]; m=fit(allrows,{mid}); pred=[float(m.predict([r['x']])[0]) for r in test]; truth=[r['gain'] for r in test]
        split[mid]={'n':len(test),'mse_model':float(np.mean((np.asarray(pred)-truth)**2)) if test else None,'mse_zero':float(np.mean(np.asarray(truth)**2)) if test else None,'gain_truth':truth,'prediction':pred}
    m=fit(allrows); allpred=np.asarray(m.predict([r['x'] for r in allrows])); yy=np.asarray([r['gain'] for r in allrows])
    report={'schema':'fm-g-expansion-diagnostic-v1','feature_schema':'fm-g-expansion-features-v1','features':FEATURES,'n_old_rows':len(rows),'n_new_transition_rows':len(new_transitions),'physical_match_groups':sorted({r['physical_match_id'] for r in allrows}),'model':'Ridge(alpha=0.1)','overall_mse':float(np.mean((allpred-yy)**2)),'overall_zero_mse':float(np.mean(yy**2)),'new_match_grouped_holdout':split,'selection_allowed':False,'empirical_support_status':'UNKNOWN','production_ready':False,'source_overlap':['understat:28891 reuses earlier Chelsea diagnostic body B01'],'interpretation':'diagnostic only; new rows show f-induced signed gains including harmful updates, but two historical non-Leeds matches are insufficient for g promotion'}
    (OUT/'model.json').write_text(json.dumps({'schema':report['schema'],'feature_schema':report['feature_schema'],'features':FEATURES,'coefficients':m.coef_.tolist(),'intercept':float(m.intercept_),'alpha':.1,'model_sha256':hashlib.sha256(json.dumps({'coef':m.coef_.tolist(),'intercept':float(m.intercept_)},sort_keys=True).encode()).hexdigest()},ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'); (OUT/'rows.json').write_text(json.dumps(new_transitions,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
