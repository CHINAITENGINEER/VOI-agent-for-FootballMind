from pathlib import Path
import json, math, numpy as np
from sklearn.linear_model import Ridge
ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'var/reviews/fm-g-nonleeds-expansion-20261015-v1'
diag=base/'g-diagnostic'
rows=json.loads((diag/'rows.json').read_text(encoding='utf-8'))
old=json.loads((ROOT/'var/reviews/fm-historical-transition-g-training-20261014-v2/transitions.json').read_text(encoding='utf-8-sig'))
def xold(t):
 q=t['query']; p=t['prediction_before']; ent=-sum(v*math.log(max(v,1e-12)) for v in p); title=q.get('title','').casefold()
 bits=[float(any(s in title for s in ('injury','injuries','availability','fitness','suspension'))),float(any(s in title for s in ('line-up','lineup','line up','xi'))),float('training' in title or 'cobham' in title),float('preview' in title),0.0]
 return [*p,ent,float(q.get('acquired_documents',0)),1.,*bits[:4],float(not any(bits[:4])),0.]
oldrows=[{'physical_match_id':t['physical_match_id'],'gain':t['signed_gain'],'x':xold(t)} for t in old]
def fit(exclude):
 tr=[r for r in oldrows+rows if r['physical_match_id'] not in exclude]
 return Ridge(alpha=.1).fit([r['x'] for r in tr],[r['gain'] for r in tr])
report={}
for mid in sorted(set(r['physical_match_id'] for r in rows)):
 m=fit({mid}); initial=[r for r in rows if r['physical_match_id']==mid and not r['from_state']]; scores=[float(m.predict([r['x']])[0]) for r in initial]; cost=.001
 report[mid]={'initial_actions':len(initial),'predicted_signed_gain':scores,'maximum_predicted_gain':max(scores) if scores else None,'query_cost':cost,'proposal':'STOP' if not scores or max(scores)<=cost else 'QUERY','selection_allowed':False,'actual_initial_signed_gain':[r['gain'] for r in initial],'exclusion':mid}
payload={'schema':'fm-g-expansion-policy-diagnostic-v1','cost':.001,'matches':report,'baselines':['stop','fixed_order'],'formal_runtime':False,'selection_allowed':False,'interpretation':'initial g proposal is evaluated with grouped holdout; a stop/query result is not empirical deployment evidence'}
(diag/'policy-replay.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
