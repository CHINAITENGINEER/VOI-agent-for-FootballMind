from __future__ import annotations
import csv,json,hashlib
from pathlib import Path
ROOT=Path(r'D:/a1-足彩'); OUT=ROOT/r'var/repairs/preannotation-v3-canonical-candidate-audit-20261024-v1'
def wj(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def main():
 base=json.loads((OUT/'canonical-candidate-audit.json').read_text(encoding='utf8'))
 recs={r['record_id']:r for r in base['records']}
 recovery=json.loads((OUT/'body-recovery/recovery-audit.json').read_text(encoding='utf8'))
 for rr in recovery['recovered']:
  x=recs[rr['record_id']]
  x.update({'current_recovery_status':'complete_body_recovered_current_retrieval','current_recovered_body_path':rr['recovered_body_path'],'current_recovered_body_chars':rr['recovered_body_chars'],'current_recovered_body_sha256':rr['recovered_body_sha256'],'current_raw_response_sha256':rr['raw_response_sha256'],'current_retrieved_at_utc':rr['retrieved_at_utc'],'current_retrieval_proves_old_availability':False,'current_body_historical_research_eligible':True,'current_body_f_admission':'research_only_until_manual_semantic_and_time_review'})
  x['f_candidate_status']='current_body_recovered_research_only_pending_semantic_and_time_review'
  x['exclusion_or_next_step']='Current full body is available for historical research, but current retrieval does not prove source availability before the historical cutoff; run frozen semantic protocol separately before any f call. Need fixed-f before/after plus independent result for g labels.'
 summary=dict(base['summary'])
 summary.update({'schema':'preannotation-v3-canonical-candidate-audit-v2','recovery_integrated':True,'current_recovered_body_records':len(recovery['recovered']),'current_historical_research_candidates':sum(bool(x.get('historical_research_eligible') or x.get('current_body_historical_research_eligible')) for x in recs.values()),'prospective_eligible':0,'g_label_eligible':0,'current_retrieval_is_not_old_availability_proof':True})
 summary['clusters']=base['summary']['clusters']
 out={'summary':summary,'records':[recs[k] for k in sorted(recs)]}
 wj(OUT/'canonical-candidate-audit-v2.json',out)
 fields=['record_id','article_title','canonical_match_id','canonical_name_id','same_match_cluster','identity_status','body_chars','body_completeness','current_recovery_status','current_recovered_body_chars','source_published_at','system_available_at','current_retrieved_at_utc','published_before_kickoff','prospective_eligible','current_body_historical_research_eligible','g_label_eligible','f_candidate_status']
 with (OUT/'canonical-candidate-table-v2.csv').open('w',encoding='utf-8-sig',newline='') as h:
  w=csv.DictWriter(h,fieldnames=fields); w.writeheader();
  for x in out['records']:w.writerow({k:x.get(k) for k in fields})
 (OUT/'README-v2.md').write_text(f'''# Canonical candidate audit v2 (recovery integrated)\n\nThis addendum joins the original archived candidate audit with two current body recoveries. It preserves the v1 files and does not write Argilla, call f/g, or emit training labels.\n\n- reviewed records: {summary["reviewed_records"]}\n- unique physical match clusters: {summary["unique_identity_clusters"]}\n- current complete body recoveries: {summary["current_recovered_body_records"]}\n- current historical research candidates: {summary["current_historical_research_candidates"]}\n- prospective eligible: 0\n- empirical g labels: 0\n\nA current retrieval on 2026-10-24 is not evidence that the complete body was operationally available before the 2026-10-18 cutoff. Both recovered bodies remain research-only pending semantic review and a separate f run. The two Palace records share one physical match cluster and must not be counted as two independent matches.\n''',encoding='utf8')
 files=[]
 for p in sorted(OUT.rglob('*')):
  if p.is_file() and p.name!='manifest.json':files.append({'path':str(p.relative_to(OUT)).replace('\\','/'),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 wj(OUT/'manifest.json',{'schema':'preannotation-v3-canonical-candidate-audit-manifest-v1','files':files})
 print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
