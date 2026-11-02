"""Complete existing post-read rows with only locally evidenced metadata.

Adds explicit field-level status and trajectory evidence. It never infers an
action_id/propensity/menu from article order or sample IDs and never emits new
empirical g labels.
"""
from __future__ import annotations
import csv, hashlib, json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(r'D:/a1-足彩')
OUT=ROOT/r'var/repairs/fm-batch-data-completion-20261024-v2'
STAGE2=ROOT/r'var/work/fm-postread-g-dataset-v1-20261021/postread-g-dataset-v1.jsonl'
BRIDGE=ROOT/r'var/work/fm-postread-g-label-bridge-20261021-v2/training-rows.jsonl'
PROV=ROOT/r'var/work/fm-f-evidence-training-panel-20261014-01/provenance'
AUDIT=ROOT/r'var/work/fm-g-f-v1-20261021/training-data-audit.json'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def wj(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def wjl(p,rows):p.write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in rows)+'\n',encoding='utf8')
def load_jsonl(p):return [json.loads(x) for x in p.read_text(encoding='utf8').splitlines() if x.strip()]
def status(value,source=None,reason=None):return {'value':value,'status':'observed' if value is not None else 'UNKNOWN','source':source,'reason':reason}
def main():
 if OUT.exists():raise FileExistsError(str(OUT))
 OUT.mkdir(parents=True)
 stage={r['sample_id']:r for r in load_jsonl(STAGE2)}
 bridge={r['sample_id']:r for r in load_jsonl(BRIDGE)}
 # Evidence from the original panel provenance; S08 is the only row with an exact action id.
 prov={}
 for p in sorted(PROV.glob('S*.json')):
  prov[p.stem]=json.loads(p.read_text(encoding='utf8'))
 rows=[]
 for sid in sorted(stage):
  s=stage[sid]; b=bridge.get(sid); p=prov.get(sid,{})
  pre=p.get('pre_read_metadata') or {}
  action_id=p.get('action_id')
  candidate_id=p.get('candidate_id') or pre.get('candidate_id')
  source_available=p.get('source_available_at') or s['source'].get('source_available_at')
  propensity=p.get('propensity')
  # The research candidate pool is a post-read panel grouping, never a recovered original menu.
  same=[x['sample_id'] for x in stage.values() if x['physical_match_id']==s['physical_match_id']]
  body_path=Path(s['body']['body_path'])
  body_sha=sha(body_path) if body_path.exists() else None
  body_match=body_sha==s['body']['body_sha256']
  outcome=(b or {}).get('outcome_index')
  gain=(b or {}).get('realized_signed_gain')
  before=(b or {}).get('prediction_before',s['current_state'].get('prediction_before'))
  after=(b or {}).get('prediction_after',s['post_f_audit'].get('prediction_after'))
  row={
   'schema':'footballmind-batch-completed-row-v1','sample_id':sid,'physical_match_id':s['physical_match_id'],'fixture':s['fixture'],
   'split':s.get('split'),'body':{'path':s['body']['body_path'],'sha256':s['body']['body_sha256'],'observed_sha256':body_sha,'sha_match':body_match,'characters':s['body']['body_characters'],'raw_sha256':s['body'].get('raw_sha256')},
   'source':{'url':s['source']['source_url'],'domain':s['source']['source_domain'],'published_at':s['source']['source_published_at'],'captured_at':s['source']['captured_at'],'available_at':source_available,'availability_status':s['source'].get('availability_status'),'historical_version_unknown':s['source'].get('historical_version_unknown'),'information_mode':s['source'].get('information_mode')},
   'clocks':{'decision_at':s['current_state']['cutoff'],'kickoff':s['fixture']['kickoff'],'source_published_at':s['source']['source_published_at'],'source_available_at':source_available,'captured_at':s['source']['captured_at'],'pre_kickoff_published':True,'pre_kickoff_availability_proven':source_available is not None},
   'g_read_state':s['g_read_state'],'f_state':{'prediction_before':before,'prediction_after':after,'probability_l1_shift':s['post_f_audit'].get('probability_l1_shift'),'feature_delta':s['post_f_audit'].get('feature_delta'),'f_received_body':s['post_f_audit'].get('f_body_received'),'f_modeled_claim_count':s['post_f_audit'].get('f_modeled_claim_count'),'f_unmodeled_claim_count':s['post_f_audit'].get('f_unmodeled_claim_count'),'f_model_version':(b or {}).get('f_model_version'),'f_model_sha256':(b or {}).get('f_model_sha256')},
   'outcome':{'outcome_index':outcome,'score':(b or {}).get('outcome_score'),'loss_before':(b or {}).get('loss_before'),'loss_after':(b or {}).get('loss_after'),'realized_signed_gain':gain,'independent_result_linked':outcome is not None},
   'action_evidence':{'action_id':status(action_id,'provenance/'+sid+'.json' if action_id else None,'exact original action field absent' if action_id is None else None),'candidate_id':status(candidate_id,'provenance/'+sid+'.json' if candidate_id else None,'S07 candidate id comes from pre_read_metadata; not an action id' if sid=='S07' else None),'propensity':status(propensity,None,'no recorded propensity; do not infer from fixed/random strategy simulations'),'source_available_at':status(source_available,'provenance/'+sid+'.json' if source_available else None,'historical source availability not proven' if source_available is None else None)},
   'menu_evidence':{'research_candidate_ids':same,'menu_source':'stage2_postread_panel','complete_original_menu':False,'menu_status':'postread_research_pool_only','reason':'archived replay does not expose the original pre-query menu'},
   'eligibility':{'fixed_f_before_after_outcome_complete':bool(b and before is not None and after is not None and outcome is not None and gain is not None),'formal_g_training_eligible':False,'formal_exclusion_reasons':['action_id_missing' if action_id is None else None,'propensity_missing' if propensity is None else None,'complete_original_menu_missing','historical_source_available_at_unknown' if source_available is None else None,'prospective_trajectory=false'],'label_origin':(b or {}).get('label_origin','no independent settled label')},
   'completion_note':'Only fields observed in local artifacts are filled; UNKNOWN fields remain explicit.'
  }
  row['eligibility']['formal_exclusion_reasons']=[x for x in row['eligibility']['formal_exclusion_reasons'] if x]
  rows.append(row)
 wjl(OUT/'completed-rows.jsonl',rows)
 menu=[]
 seen=set()
 for r in rows:
  mid=r['physical_match_id']
  if mid in seen:continue
  seen.add(mid);menu.append({'physical_match_id':mid,'candidate_ids':r['menu_evidence']['research_candidate_ids'],'menu_source':'stage2_postread_panel','complete_original_menu':False,'action_ids':None,'propensities':None,'status':'research_pool_only'})
 wjl(OUT/'candidate-menu-audit.jsonl',menu)
 fields=['sample_id','physical_match_id','body_sha_match','source_published_at','source_available_at','captured_at','action_id','candidate_id','propensity','complete_original_menu','prediction_before_present','prediction_after_present','outcome_present','realized_gain_present','fixed_f_before_after_outcome_complete','formal_g_training_eligible']
 with (OUT/'field-completion.csv').open('w',encoding='utf-8-sig',newline='') as h:
  w=csv.DictWriter(h,fieldnames=fields);w.writeheader()
  for r in rows:
   w.writerow({'sample_id':r['sample_id'],'physical_match_id':r['physical_match_id'],'body_sha_match':r['body']['sha_match'],'source_published_at':r['source']['published_at'],'source_available_at':r['source']['available_at'],'captured_at':r['source']['captured_at'],'action_id':r['action_evidence']['action_id']['value'],'candidate_id':r['action_evidence']['candidate_id']['value'],'propensity':r['action_evidence']['propensity']['value'],'complete_original_menu':r['menu_evidence']['complete_original_menu'],'prediction_before_present':r['f_state']['prediction_before'] is not None,'prediction_after_present':r['f_state']['prediction_after'] is not None,'outcome_present':r['outcome']['outcome_index'] is not None,'realized_gain_present':r['outcome']['realized_signed_gain'] is not None,'fixed_f_before_after_outcome_complete':r['eligibility']['fixed_f_before_after_outcome_complete'],'formal_g_training_eligible':False})
 counts=Counter()
 for r in rows:
  for k in ['body_sha_match','source_available_at','action_id','candidate_id','propensity','complete_original_menu','fixed_f_before_after_outcome_complete','formal_g_training_eligible']:
   val=r['body']['sha_match'] if k=='body_sha_match' else r['source']['available_at'] if k=='source_available_at' else r['action_evidence'][k]['value'] if k in r['action_evidence'] else r['menu_evidence']['complete_original_menu'] if k=='complete_original_menu' else r['eligibility'][k]
   if k=='source_available_at': counts['source_available_at_known' if val is not None else 'source_available_at_unknown']+=1
   elif k in ('action_id','candidate_id','propensity'): counts[f'{k}_known' if val is not None else f'{k}_unknown']+=1
   elif k=='body_sha_match':counts['body_sha_match' if val else 'body_sha_mismatch']+=1
   elif k=='complete_original_menu':counts['complete_original_menu_true' if val else 'complete_original_menu_false']+=1
   elif k=='fixed_f_before_after_outcome_complete':counts['fixed_f_before_after_outcome_complete' if val else 'fixed_f_before_after_outcome_incomplete']+=1
   elif k=='formal_g_training_eligible':counts['formal_g_training_eligible_true' if val else 'formal_g_training_eligible_false']+=1
 report={'schema':'footballmind-batch-data-completion-report-v1','status':'metadata_completed_where_evidenced','generated_at':datetime.now(timezone.utc).isoformat(),'rows':len(rows),'physical_matches':len(set(r['physical_match_id'] for r in rows)),'counts':dict(counts),'trajectory_candidates_with_exact_action_id':[r['sample_id'] for r in rows if r['action_evidence']['action_id']['value'] is not None],'settled_fixed_f_rows':[r['sample_id'] for r in rows if r['eligibility']['fixed_f_before_after_outcome_complete']],'formal_g_training_eligible_rows':0,'no_inference_rules':['Do not convert candidate_id into action_id.','Do not infer propensity from strategy simulation order.','Do not call postread research pools original pre-query menus.','Do not treat published_at as source_available_at.'],'source_artifacts':[str(STAGE2),str(BRIDGE),str(PROV)]}
 wj(OUT/'completion-report.json',report)
 (OUT/'README.md').write_text('''# Existing batch data completion\n\nThis package fills only metadata evidenced by existing local records. It joins the stage-2 post-read dataset, the fixed-f bridge, and per-sample provenance. The seven historical rows retain before/after/outcome/gain; S08 retains an exact read action and source-availability receipt but has no settled outcome label.\n\nOriginal action IDs, propensities, and complete pre-query menus remain UNKNOWN where absent. The candidate lists are post-read research pools, not recovered original menus. Formal empirical g eligibility remains zero.\n''',encoding='utf8')
 files=[]
 for p in sorted(OUT.iterdir()):
  if p.is_file() and p.name!='manifest.json':files.append({'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)})
 wj(OUT/'manifest.json',{'schema':'footballmind-batch-data-completion-manifest-v1','files':files})
 print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

