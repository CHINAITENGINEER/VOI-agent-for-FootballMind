"""Fit a read-only diagnostic g on the existing post-read bridge.

The bridge has real historical fixed-f gains but no action_id/propensity, so this
script creates a batch diagnostic artifact only. It never promotes the model.
"""
from __future__ import annotations
import csv, hashlib, json, math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

ROOT=Path(r'D:/a1-足彩')
SRC=ROOT/r'var/work/fm-postread-g-label-bridge-20261021-v2'
OUT=ROOT/r'var/work/fm-g-batch-diagnostic-training-20261024-v1'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def weighted_mse(y,p,w=None):
 y=np.asarray(y,float);p=np.asarray(p,float)
 if w is None:return float(np.mean((y-p)**2))
 return float(np.sum(w*(y-p)**2)/np.sum(w))
def weighted_mae(y,p,w=None):
 y=np.asarray(y,float);p=np.asarray(p,float)
 if w is None:return float(np.mean(np.abs(y-p)))
 return float(np.sum(w*np.abs(y-p))/np.sum(w))
def spearman(y,p):
 from scipy.stats import spearmanr
 if len(set(y))<2 or len(set(p))<2:return None
 return float(spearmanr(y,p).statistic)
def main():
 OUT.mkdir(parents=True,exist_ok=False)
 rows=[json.loads(x) for x in (SRC/'training-rows.jsonl').read_text(encoding='utf8').splitlines() if x.strip()]
 x=np.load(SRC/'body_state_vectors.npy'); body=np.load(SRC/'body_vectors.npy')
 assert x.shape==(len(rows),393) and body.shape==(len(rows),384) and np.isfinite(x).all()
 mids=[r['physical_match_id'] for r in rows]; groups=sorted(set(mids)); y=np.asarray([r['realized_signed_gain'] for r in rows],float)
 assert set(Counter(mids))==set(groups)
 # Match-balanced weights: each independent match contributes total 1.
 counts=Counter(mids); weights=np.asarray([1.0/counts[m] for m in mids])
 alphas=[0.1,1.0,10.0,100.0]
 model_grid=[]
 lomo=[]
 for alpha in alphas:
  pred=np.full(len(rows),np.nan)
  for hold in groups:
   train=np.asarray([m!=hold for m in mids]); test=~train
   fit=Ridge(alpha=alpha).fit(x[train],y[train],sample_weight=weights[train])
   pred[test]=fit.predict(x[test])
  model_grid.append({'alpha':alpha,'row_mse':weighted_mse(y,pred,weights),'row_mae':weighted_mae(y,pred,weights),'spearman':spearman(y,pred),'predictions':pred.tolist()})
 # Choose alpha=10 because it is the frozen existing diagnostic setting, not by test search.
 selected_alpha=10.0; selected=next(z for z in model_grid if z['alpha']==selected_alpha)
 full=Ridge(alpha=selected_alpha).fit(x,y,sample_weight=weights)
 model={'schema':'footballmind-g-batch-diagnostic-model-v1','model_kind':'sklearn_ridge','input_layout':{'body_state_vector_dim':int(x.shape[1]),'body_tfidf_dim':384,'state_dim':9},'alpha':selected_alpha,'coefficients':full.coef_.tolist(),'intercept':float(full.intercept_),'fit_match_ids':groups,'fit_rows':len(rows),'weighting':'each physical_match_id total weight=1','f_model_sha256':sorted(set(r['f_model_sha256'] for r in rows)),'selection_allowed':False,'production_ready':False,'empirical_support_status':'UNKNOWN','formal_training_eligible_rows':0,'provenance':'historical_reconstruction_fixed_f_postread_bridge'}
 dump(OUT/'model.json',model)
 # Baselines: zero and training-only match-weighted mean per LOMO fold.
 pred= np.asarray(selected['predictions'],float); zero=np.zeros_like(y)
 mean_pred=np.full(len(rows),np.nan)
 for hold in groups:
  train=np.asarray([m!=hold for m in mids]); test=~train
  mean=float(np.sum(weights[train]*y[train])/np.sum(weights[train])); mean_pred[test]=mean
 metrics=[]
 for name,p in [('zero',zero),('constant_lomo',mean_pred),('ridge_lomo_alpha10',pred)]:
  metrics.append({'model':name,'row_mse':weighted_mse(y,p,weights),'row_mae':weighted_mae(y,p,weights),'spearman':spearman(y,p),'positive_direction_accuracy':(sum((a>0)==(b>0) for a,b in zip(y,p) if abs(a)>1e-12)/sum(abs(a)>1e-12 for a in y))})
 with (OUT/'metrics.csv').open('w',encoding='utf8-sig',newline='') as h:
  w=csv.DictWriter(h,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
 # Per-row and per-match diagnostics.
 outrows=[]
 for i,r in enumerate(rows): outrows.append({'sample_id':r['sample_id'],'physical_match_id':mids[i],'realized_signed_gain':float(y[i]),'zero_prediction':0.0,'constant_lomo_prediction':float(mean_pred[i]),'ridge_lomo_prediction':float(pred[i]),'body_sha256':r['body_sha256'],'f_model_sha256':r['f_model_sha256'],'body_visible_to_g':r['body_visible_to_g'],'body_adopted_by_f':r['body_adopted_by_f'],'modeled_claim_count':r['modeled_claim_count'],'action_id':r['action_id'],'propensity':r['propensity'],'prospective_training_eligible':r['prospective_training_eligible']})
 (OUT/'predictions.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in outrows)+'\n',encoding='utf8')
 per=[]
 for hold in groups:
  idx=[i for i,m in enumerate(mids) if m==hold]
  per.append({'physical_match_id':hold,'rows':len(idx),'zero_mse':weighted_mse(y[idx],zero[idx]),'constant_lomo_mse':weighted_mse(y[idx],mean_pred[idx]),'ridge_lomo_mse':weighted_mse(y[idx],pred[idx]),'gain_signs':dict(Counter('positive' if y[i]>1e-12 else 'negative' if y[i]<-1e-12 else 'zero' for i in idx))})
 # Plot
 plt.figure(figsize=(8,5.5));
 colors=['#228B8D' if v>1e-12 else '#C65D3B' if v< -1e-12 else '#7D8790' for v in y]
 plt.scatter(y,pred,c=colors,s=70,alpha=.85,edgecolor='white',linewidth=.6)
 lo=float(min(y.min(),pred.min()))-.003; hi=float(max(y.max(),pred.max()))+.003
 plt.plot([lo,hi],[lo,hi],'k--',lw=1,label='perfect prediction')
 for i,r in enumerate(rows): plt.annotate(r['sample_id'],(y[i],pred[i]),xytext=(4,4),textcoords='offset points',fontsize=8)
 plt.xlabel('Realized signed gain (log-loss units)');plt.ylabel('LOMO predicted gain');plt.title('Diagnostic g: fixed f, full body + state, leave-one-match-out');plt.grid(alpha=.22);plt.legend();plt.tight_layout();plt.savefig(OUT/'predicted-vs-realized.png',dpi=180);plt.close()
 report={'schema':'footballmind-g-batch-diagnostic-training-report-v1','status':'DIAGNOSTIC_BATCH_FIT_COMPLETE_NOT_FORMAL_G','source_dir':str(SRC),'source_training_rows_sha256':sha(SRC/'training-rows.jsonl'),'source_body_state_vectors_sha256':sha(SRC/'body_state_vectors.npy'),'rows':len(rows),'independent_matches':len(groups),'match_ids':groups,'gain_class_counts':dict(Counter('positive' if v>1e-12 else 'negative' if v< -1e-12 else 'zero' for v in y)),'f_model_sha256':sorted(set(r['f_model_sha256'] for r in rows)),'feature_shape':list(x.shape),'model_grid':[{k:v for k,v in g.items() if k!='predictions'} for g in model_grid],'selected_alpha':selected_alpha,'metrics':metrics,'per_match':per,'formal_training_eligible_rows':0,'selection_allowed':False,'empirical_support_status':'UNKNOWN','production_ready':False,'why_not_formal':['action_id missing in all rows','propensity missing in all rows','complete pre-query menu unavailable','historical source_available_at not fully proven','rows are historical reconstruction and transitions within a match are correlated'],'network_requests':0,'api_calls':0,'gpu_used':False,'next_required_for_formal':['preserve this feature/label contract','add action_id and propensity at collection time','retain full menu and pre-query state','obtain independent outcome linkage and clocks','repeat with held-out physical matches']}
 dump(OUT/'report.json',report); dump(OUT/'per-match.json',per)
 readme=f'''# Diagnostic batch g fit\n\nThis folder fits a small research-only g on the existing post-read bridge. It retains all 7 rows: 2 zero, 2 positive, and 3 negative realized signed gains from 5 physical matches.\n\nThe fitted input is the existing 393-dimensional body+state vector. Alpha=10 is fixed to match the existing bridge diagnostic. Leave-one-physical-match-out predictions are reported alongside zero and constant baselines.\n\nThis model cannot be used as empirical g: all rows lack action_id and propensity, the complete pre-query menu is unavailable, and historical source availability is not fully proven. `selection_allowed=false`, `production_ready=false`, and formal training rows remain 0.\n'''
 (OUT/'README.md').write_text(readme,encoding='utf8')
 files=[]
 for p in sorted(OUT.iterdir()):
  if p.is_file() and p.name!='manifest.json':files.append({'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)})
 dump(OUT/'manifest.json',{'schema':'footballmind-g-batch-diagnostic-training-manifest-v1','files':files})
 print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
