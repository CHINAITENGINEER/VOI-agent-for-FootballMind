"""Append a second locally recovered body to the read-only audit package."""
from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path
from bs4 import BeautifulSoup
ROOT=Path(r'D:/a1-足彩')
OUT=ROOT/r'var/repairs/preannotation-v3-canonical-candidate-audit-20261024-v1'
RAW=OUT/r'body-recovery/raw/ur1-efe647982f41ba4a345be862.html'
REVIEW=ROOT/r'var/repairs/preannotation-v3-second-review-20261024-v1/second-review.jsonl'
RID='ur1-efe647982f41ba4a345be862'
URL='https://www.cpfc.co.uk/news/match-previews/the-opposite-view-hurzeler-on-sage-palace-rivalry/'
CANON='footballmind:premier-league:2026-27:2026-10-18:brighton:crystal-palace'
KICK='2026-10-18T13:00:00+00:00'
RETRIEVED='2026-10-24T10:22:06Z'
def sha(b): return hashlib.sha256(b).hexdigest()
def compact(s): return ' '.join(s.split())
def row():
 for line in REVIEW.read_text(encoding='utf8').splitlines():
  if line.strip():
   x=json.loads(line)
   if x.get('record_id')==RID:return x
 raise RuntimeError(RID)
def extract(html):
 s=BeautifulSoup(html,'html.parser'); secs=s.find_all('section',class_='overflow-hidden')
 if len(secs)<2: raise RuntimeError('article section missing')
 sec=secs[1]; blocks=[]
 for ci,ch in enumerate(sec.find_all(recursive=False)):
  txt=compact(ch.get_text(' ',strip=True))
  if txt.startswith('Related News'): break
  # Child 7 is the event metadata card; omit it. Empty/image blocks are omitted.
  if ci==7: continue
  for e in ch.find_all(['p','blockquote','h3','h4','li'], recursive=True):
   if e.name=='p' and e.find_parent('blockquote') is not None: continue
   # do not include navigation or nested category tags
   if e.name=='li' and ci not in (5,): continue
   value=compact(e.get_text(' ',strip=True))
   if not value: continue
   if e.name in ('h3','h4') and value in ('Related News','Brighton & Hove Albion v Crystal Palace'): continue
   blocks.append({'ordinal':len(blocks),'kind':('blockquote' if e.name=='blockquote' else ('heading' if e.name in ('h3','h4') else 'paragraph')),'text':value,'html_selector':f'section.overflow-hidden:nth-of-type(2) > child[{ci}] {e.name}'})
 body='\n\n'.join(x['text'] for x in blocks)+'\n'
 return body,blocks

def wj(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def main():
 raw=RAW.read_bytes(); rr=sha(raw); b,blocks=extract(raw.decode('utf8','replace')); old=row(); recdir=OUT/'body-recovery/recovered'; recdir.mkdir(parents=True,exist_ok=True)
 bp=recdir/f'{RID}.txt'; bp.write_text(b,encoding='utf8',newline='\n')
 rec={'schema':'preannotation-v3-body-recovery-record-v1','status':'current_retrieval_recovered_for_historical_research_only','record_id':RID,'source_url':URL,'source_domain':'www.cpfc.co.uk','canonical_name_id':CANON,'home_team':'Brighton & Hove Albion','away_team':'Crystal Palace','kickoff':KICK,'source_published_at':old.get('times',{}).get('source_published_at'),'source_updated_at':old.get('times',{}).get('source_updated_at'),'historical_archived_body_chars':len(old.get('body') or ''),'historical_archived_body_sha256':sha((old.get('body') or '').encode('utf8')),'historical_archived_body_status':'suspected_1600_char_cap_mid_sentence','raw_response_path':str(RAW),'raw_response_bytes':len(raw),'raw_response_sha256':rr,'retrieved_at_utc':RETRIEVED,'retrieval_kind':'current_retrieval_after_original_archive','tls_verification':'certificate_verification_bypassed_for_recovery_probe','source_available_at_at_old_cutoff':'UNKNOWN','current_retrieval_proves_old_availability':False,'recovered_body_path':str(bp),'recovered_body_chars':len(b),'recovered_body_sha256':sha(b.encode('utf8')),'paragraph_count':len(blocks),'blocks':blocks,'parser':{'section_selector':'section.overflow-hidden[article_body_index=1]','stop_rule':'stop before first direct child whose normalized text starts with Related News','included_elements':['p','blockquote','h3','h4','selected li'],'excluded_content':['site navigation','image-only blocks','Related News','match metadata card (child 7)']},'body_used_for_f_or_g':False,'f_admission':'research_only_until_manual_semantic_and_time_review','g_label_eligible':False,'exclusion_reason':'No pre-kickoff source_available_at clock; current retrieval cannot establish prospective availability or empirical gain label.','receipt':{'status':'200','bytes':len(raw),'sha256':rr,'retrieved_at_utc':RETRIEVED,'url':URL,'tls_verification':'certificate_verification_bypassed_for_recovery_probe'}}
 wj(recdir/f'{RID}.json',rec)
 ap=OUT/'body-recovery/recovery-audit.json'; audit=json.loads(ap.read_text(encoding='utf8')); audit['status']='complete_recovery_for_two_capped_bodies_current_retrieval_only'; audit['recovered'].append(rec); audit['unrecovered']=[]; audit['recovery_run_at_utc']=datetime.now(timezone.utc).isoformat().replace('+00:00','Z'); audit['recovery_method_note']='Second response saved with curl -k because normal schannel verification failed; this is recorded as a retrieval transport detail, not as a prospective availability proof.'; wj(ap,audit)
 np=OUT/'network-probe.json'; probe=json.loads(np.read_text(encoding='utf8')); probe['status']='probe_and_partial_raw_recovery_no_model_call'; probe.setdefault('attempts',[]).append({'source_url':URL,'method':'GET','observed_status':200,'response_saved':True,'response_path':str(RAW),'response_bytes':len(raw),'raw_sha256':rr,'body_used':False,'retrieved_at_utc':RETRIEVED,'tls_verification':'certificate_verification_bypassed_with_curl_k','note':'Saved after normal schannel TLS failures; this is a current retrieval and does not establish old cutoff availability.'}); probe['recovery_summary']={'recovered_records':2,'unrecovered_records':0,'f_or_g_calls':0,'training_rows':0}; wj(np,probe)
 readme=OUT/'README.md'; txt=readme.read_text(encoding='utf8'); txt=txt.replace('The second Palace article remains unrecovered after a TLS failure.','The second Palace article was subsequently recovered with a recorded TLS-verification bypass (`curl -k`); the raw response and normalized body are retained. This does not establish the old cutoff availability.')
 readme.write_text(txt,encoding='utf8')
 files=[]
 for p in sorted(OUT.rglob('*')):
  if p.is_file() and p.name!='manifest.json': files.append({'path':str(p.relative_to(OUT)).replace('\\','/'),'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())})
 wj(OUT/'manifest.json',{'schema':'preannotation-v3-canonical-candidate-audit-manifest-v1','files':files})
 print(json.dumps({'record_id':RID,'raw_sha256':rr,'body_sha256':sha(b.encode('utf8')),'body_chars':len(b),'blocks':len(blocks),'manifest_files':len(files)},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
