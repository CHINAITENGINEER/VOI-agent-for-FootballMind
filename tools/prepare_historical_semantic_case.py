from __future__ import annotations
import hashlib,json,re
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CASE_PATH=ROOT/'var/submission/p1-single-match/corpus-probe/cases-with-interviews-routed.json'
OUT=ROOT/'var/submission/p7-historical-semantic/case-selection'
OUT.mkdir(parents=True,exist_ok=True)
CASES=json.loads(CASE_PATH.read_text(encoding='utf-8-sig'))
# Rule is deliberately metadata-only and is materialized before body access below.
rule={
 'version':'historical-case-selection-rule-v1',
 'case_pool':'cases-with-interviews-routed.json',
 'split_resolution':'No split field exists in this object; use known chronological A15/B5/C6 partition, so C is indices 20..25.',
 'candidate_order':'fixture kickoff ascending, then physical_match_id',
 'eligibility':'at least two distinct official prematch article documents with source_published_at before kickoff, and archived raw bytes present with matching sha256',
 'selection_inputs':['fixture physical_match_id/match_id/teams/kickoff','document URL/title/document_kind/source_published_at/captured_at/raw_path/raw_sha256'],
 'selection_inputs_forbidden':['outcome','label_provenance','home_goals','away_goals','result_available_at','prior_probabilities','VoI','document body text','injury/support branch content'],
 'selection_purpose':'historical_reconstruction mechanism case only; not an independent effectiveness evaluation',
 'information_mode':'historical_reconstruction',
 'menu_policy':'Menu entries are only source title/URL pairs observed in archived case metadata. The resulting multi-page menu is a reconstructed directory, not an original site snapshot.',
}
# Enumerate C by metadata only.
probe_root=ROOT/'var/submission/p1-single-match/corpus-probe'
def resolve_raw(raw_path:str)->Path:
    name=Path(raw_path.replace('\\','/')).name
    hits=list((probe_root/'raw').glob(name))
    if len(hits)!=1: raise FileNotFoundError((name,hits))
    return hits[0]
def file_ok(d):
    try:
      p=resolve_raw(d['raw_path'])
      return p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==d['raw_sha256']
    except (KeyError,FileNotFoundError): return False

def metadata_doc_ok(d,kick):
    try:
      pub=datetime.fromisoformat(d['source_published_at'].replace('Z','+00:00'))
      ko=datetime.fromisoformat(kick.replace('Z','+00:00'))
    except Exception:return False
    return d.get('url') and d.get('title') and d.get('document_kind','').startswith('official_') and pub<ko and file_ok(d)

candidates=[]
for idx,c in enumerate(CASES[20:26],start=20):
  f=c['fixture']; docs=[d for d in c.get('documents',[]) if metadata_doc_ok(d,f['kickoff'])]
  candidates.append({'source_index':idx,'physical_match_id':f['physical_match_id'],'match_id':f['match_id'],'home_name':f['home_name'],'away_name':f['away_name'],'kickoff':f['kickoff'],'eligible_document_count':len(docs),'eligible_urls':[d['url'] for d in docs]})
candidates.sort(key=lambda x:(x['kickoff'],x['physical_match_id']))
chosen_summary=next(x for x in candidates if x['eligible_document_count']>=2)
chosen=CASES[chosen_summary['source_index']]
# Only now access selected document bodies.

def parse_clock(raw):
  return raw if raw else None

docs=[]
for i,d in enumerate(sorted(chosen['documents'],key=lambda x:(x['source_published_at'],x['url']))):
  if not metadata_doc_ok(d,chosen['fixture']['kickoff']): continue
  raw=resolve_raw(d['raw_path']); raw_bytes=raw.read_bytes(); digest=hashlib.sha256(raw_bytes).hexdigest()
  receipt_path=raw.with_suffix('.json'); receipt=json.loads(receipt_path.read_text(encoding='utf-8')) if receipt_path.exists() else {}
  # Preserve paragraphs exactly as extracted in the existing corpus. They are the only
  # structured article segments retained by cases-with-interviews-routed.
  paragraphs=list(d.get('paragraphs') or [])
  segments=[]
  for j,text in enumerate(paragraphs):
    segments.append({'text':text,'content_role':'paragraph','original_paragraph_index':j,'source_url':d['url'],'source_published_at':d.get('source_published_at')})
  structured={
    'version':'structured-news-discovery-v3',
    'role':'article',
    'segments':segments,
    'excluded':[],
    'identity_metadata':[],
    'segment_provenance':{
      'kind':'preserved_source_segments',
      'basis':'cases-with-interviews-routed.json document.paragraphs (segments emitted by the source projection during original collection)',
      'raw_html_replay':str(raw.relative_to(ROOT)).replace('\\','/'),
      'raw_projection_replayed':False,
    },
  }
  docs.append({
   'document_id':f"{chosen['fixture']['physical_match_id']}:doc-{i}",
   'title':d['title'],'url':d['url'],'text':d['text'],'paragraphs':paragraphs,'structured_news':structured,
   'source_published_at':d.get('source_published_at'),'published_precision':d.get('published_precision'),
   'captured_at':d.get('captured_at') or receipt.get('captured_at'),
   'available_at':None,'available_at_basis':'not recorded in this archive; do not infer availability from actual acquisition/capture time',
   'requested_at':receipt.get('requested_at'),
   'raw_path':str(raw.relative_to(ROOT)).replace('\\','/'),'raw_sha256':digest,
   'source_time_basis':d.get('source_time_basis'),'title_basis':d.get('title_basis'),
   'information_mode':'historical_reconstruction','historical_snapshot_proven':False,
   'source_modified_at':d.get('source_modified_at'),'later_body_revision_possible':d.get('later_body_revision_possible'),
   'document_kind':d.get('document_kind'),'fixture_binding_basis':d.get('fixture_binding_basis'),
   'body_review_scope':d.get('body_review_scope'),
   'discovery':d.get('discovery'),
  })
assert len(docs)>=2
selection={
 'schema_version':'historical-case-selection-v1','created_at':datetime.now(timezone.utc).isoformat(),
 'selection_rule':rule,'selected_source_index':chosen_summary['source_index'],'selected_case':chosen_summary,
 'candidate_table':candidates,
 'selected_documents':[{'url':d['url'],'title':d['title'],'source_published_at':d['source_published_at'],'raw_path':str(resolve_raw(d['raw_path']).relative_to(ROOT)).replace('\\','/'),'raw_sha256':d['raw_sha256']} for d in docs],
 'audit':{'body_inspection_after_rule_recorded':True,'selection_used_outcome_or_label':False,'selection_used_voi':False,'actual_acquisition_clock_backfilled':False,'case_is_independent_effectiveness_evaluation':False,'historical_snapshot_proven':False},
 'data_inventory':{
   'case_object':'fixture, prior_probabilities, outcome, result_available_at, label_provenance, documents, status, exclusion_reasons, information_mode, background_links, background_links_status (outcome/labels intentionally not copied to panel)',
   'selected_document_object':'url,title,text,paragraphs,source_published_at,published_precision,captured_at,raw_path,raw_sha256,source_time_basis,title_basis,discovery,information_mode,historical_snapshot_proven,source_modified_at,later_body_revision_possible,document_kind,fixture_binding_basis,body_review_scope',
   'archive_objects':['var/submission/p1-single-match/corpus-probe/raw/2b4a5296471e653a854e.html + .json','var/submission/p1-single-match/corpus-probe/raw/e9aad711bf4e1529bb24.html + .json'],
   'available_baseline_sources':['var/submission/p6-semantic-scenario/mechanism-01/run.json','var/submission/p6-semantic-scenario/live-01/run.json','football_agent/semantic_scenario_protocol.py','football_agent/semantic_scenario_model.py','football_agent/semantic_scenario_runtime.py','football_agent/semantic_scenario_report.py'],
   'baseline_note':'These sources are available for root runtime integration; no model run, result, label, or VoI is copied into this historical panel.'
 }
}
(OUT/'selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2),encoding='utf-8')
panel={
 'schema_version':'historical-semantic-panel-v1','information_mode':'historical_reconstruction',
 'case_status':'mechanism_case_not_independent_effectiveness_evaluation',
 'selection_path':str((OUT/'selection.json').relative_to(ROOT)).replace('\\','/'),
 'fixture':chosen['fixture'],
 'temporal_boundary':{'kickoff':chosen['fixture']['kickoff'],'actor_cutoff_clock':chosen['fixture']['kickoff'],'clock_basis':'fixture kickoff; no actual acquisition clock substituted'},
 'documents':docs,
 'menu':{
   'mode':'reconstructed_historical_directory','is_original_site_snapshot':False,
   'basis':'source title/url pairs observed in the selected case archive; assembled after capture for replay',
   'entries':[{'title':d['title'],'url':d['url'],'source_published_at':d['source_published_at'],'captured_at':d['captured_at'],'raw_sha256':d['raw_sha256'],'observed_in_archived_case':True} for d in docs]
 },
 'actor_visibility_contract':{
   'allowed':['fixture','kickoff','documents.title','documents.url','documents.text','documents.structured_news','documents.source_published_at','documents.captured_at','documents.available_at'],
   'excluded':['outcome','label_provenance','home_goals','away_goals','result_available_at','prior_probabilities','VoI','postmatch_target_score'],
   'note':'The panel contains explicit prematch article language and source clocks. historical_snapshot_proven=false and later_body_revision_possible are retained as provenance warnings.'
 }
}
(OUT/'panel.json').write_text(json.dumps(panel,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'selected':chosen_summary,'documents':len(docs),'panel':str(OUT/'panel.json'),'selection':str(OUT/'selection.json')},ensure_ascii=False))
