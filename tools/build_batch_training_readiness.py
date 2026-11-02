"""Materialize a research-only batch-training package from existing artifacts.

Copies the already-built 7-row historical bridge and small vector arrays, and
builds an index over the large archived corpus without copying raw bodies.
No model fit, provider call, runtime write, or source mutation is performed.
"""
from __future__ import annotations
import csv,hashlib,json,shutil
from pathlib import Path
from collections import Counter,defaultdict
ROOT=Path(r'D:/a1-足彩')
BRIDGE=ROOT/r'var/work/fm-postread-g-label-bridge-20261021-v2'
ART=ROOT/r'var/repairs/batch-training-readiness-20261024-v1'
CORPUS=ROOT/r'var/corpora/football-news-pipeline-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def wj(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def main():
 ART.mkdir(parents=True,exist_ok=True)
 # copy small existing supervised artifacts byte-for-byte
 for name in ['training-rows.jsonl','body_vectors.npy','body_state_vectors.npy','vectorizer-vocabulary.json']:
  shutil.copy2(BRIDGE/name,ART/name)
 # Load and normalize the seven training rows only by preserving original JSON.
 rows=[json.loads(x) for x in (BRIDGE/'training-rows.jsonl').read_text(encoding='utf8').splitlines() if x.strip()]
 mids=sorted({r['physical_match_id'] for r in rows})
 split={'group_key':'physical_match_id','train_matches':['understat:28778','understat:28779','understat:28942'],'validation_matches':['understat:28780'],'test_matches':['understat:28782'],'warning':'five independent matches and seven rows are diagnostic only'}
 wj(ART/'split-manifest.json',split)
 # index auto-linked corpus without treating links as verified labels
 links=defaultdict(list)
 for line in (CORPUS/'article_match_links.jsonl').read_text(encoding='utf8',errors='replace').splitlines():
  if not line.strip():continue
  x=json.loads(line);links[x.get('article_id')].append({k:x.get(k) for k in ('physical_match_id','binding_method','review_status','verified_link','g_training_eligible','days_before_match_date')})
 out=ART/'corpus-index.jsonl'; counts=Counter();
 with out.open('w',encoding='utf8') as h:
  for line in (CORPUS/'articles.jsonl').read_text(encoding='utf8',errors='replace').splitlines():
   if not line.strip():continue
   x=json.loads(line); bp=x.get('body_path');rp=x.get('raw_path'); body_exists=bool(bp and Path(bp).exists()); raw_exists=bool(rp and Path(rp).exists()); ls=links.get(x.get('article_id'),[])
   rec={'article_id':x.get('article_id'),'url':x.get('url'),'normalized_url':x.get('normalized_url'),'title':x.get('title'),'source_domain':x.get('source_domain'),'source_type':x.get('source_type'),'published_at':x.get('published_at'),'updated_at':x.get('updated_at'),'retrieved_at':x.get('retrieved_at'),'body_status':x.get('body_status'),'body_characters':x.get('body_characters'),'recorded_body_sha256':x.get('recorded_body_sha256'),'normalized_body_sha256':x.get('normalized_body_sha256'),'body_path':bp,'body_path_exists':body_exists,'raw_path':rp,'raw_path_exists':raw_exists,'match_links':ls,'link_status':'verified' if any(z.get('verified_link') is True for z in ls) else ('automated_candidate' if ls else 'unlinked'),'label_status':'unlabeled_corpus'}
   h.write(json.dumps(rec,ensure_ascii=False)+'\n')
   counts['articles']+=1;counts['body_path_exists' if body_exists else 'body_path_missing']+=1;counts['raw_path_exists' if raw_exists else 'raw_path_missing']+=1;counts['full_text' if x.get('body_status')=='full_text' else (x.get('body_status') or 'status_unknown')]+=1;counts['linked' if ls else 'unlinked']+=1;counts['verified_link' if any(z.get('verified_link') is True for z in ls) else 'not_verified']+=1
 # research readiness summary
 readiness={'schema':'footballmind-batch-training-readiness-v1','status':'batch_artifacts_materialized_research_only','supervised_bridge':{'path':str(ART/'training-rows.jsonl'),'rows':len(rows),'physical_matches':len(mids),'positive_gain_rows':sum(r['realized_signed_gain']>0 for r in rows),'zero_gain_rows':sum(r['realized_signed_gain']==0 for r in rows),'negative_gain_rows':sum(r['realized_signed_gain']<0 for r in rows),'action_id_present':sum(bool(r.get('action_id')) for r in rows),'propensity_present':sum(r.get('propensity') is not None for r in rows),'formal_g_training_eligible_rows':0,'label_origin':'historical_reconstruction_joined_to_independent_understat_result'},'vector_artifacts':{'body_vectors_shape':[7,384],'body_state_vectors_shape':[7,393],'vectorizer':'character TF-IDF 3-5 grams max_features=384','source':str(BRIDGE)},'unlabeled_corpus':dict(counts),'split':split,'rules':['Do not treat auto match links as verified labels.','Do not mix unlabeled corpus with realized-gain rows without an explicit objective.','Group all supervised evaluation by physical_match_id.','Current retrieval/prospective availability clocks remain separate from historical rows.'],'training_status':'research_batch_can_run_on_7_historical_rows_but_not_empirical_or_production_ready'}
 wj(ART/'readiness.json',readiness)
 (ART/'README.md').write_text('''# Batch training readiness package\n\nThis package materializes existing research artifacts for a batch experiment. It does not train or promote a model.\n\n## Supervised rows\n\n`training-rows.jsonl` contains the exact seven rows from the frozen post-read bridge: five independent historical matches, with positive, zero and negative realized signed gains. The rows have no action IDs or propensities, so they are suitable for diagnostics or offline regression only. They are not formal empirical-g rows.\n\n`body_vectors.npy` and `body_state_vectors.npy` are the existing 384-dimensional character TF-IDF and 393-dimensional body-plus-state arrays. They were copied byte-for-byte from the bridge.\n\n## Large corpus\n\n`corpus-index.jsonl` indexes the existing 13,752-article corpus without copying the raw bodies. Only path existence and metadata are recorded. Automatic article-match links remain unverified candidates; they are not labels.\n\n## Split\n\nThe split is grouped by `physical_match_id`: three development matches, one validation match and one test match. This is a diagnostic split because it contains only five independent matches.\n''',encoding='utf8')
 # manifest
 files=[]
 for p in sorted(ART.iterdir()):
  if p.name!='manifest.json':files.append({'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)})
 wj(ART/'manifest.json',{'schema':'footballmind-batch-training-readiness-manifest-v1','files':files})
 print(json.dumps(readiness,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
