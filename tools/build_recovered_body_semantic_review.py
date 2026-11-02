"""Create a deterministic, read-only semantic review of recovered Palace bodies.

This is a Codex audit index, not provider output, f execution, or a label file.
"""
from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(r'D:/a1-足彩')
OUT=ROOT/r'var/repairs/preannotation-v3-canonical-candidate-audit-20261024-v1'
REC=OUT/'body-recovery/recovered'
CANON='footballmind:premier-league:2026-27:2026-10-18:brighton:crystal-palace'
KICK='2026-10-18T13:00:00+00:00'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(rid):
 p=REC/f'{rid}.txt'; return p,p.read_text(encoding='utf8')
def q(text, quote):
 start=text.find(quote)
 if start<0: raise ValueError(f'quote not found: {quote[:80]}')
 return {'quote':quote,'offset_start':start,'offset_end':start+len(quote)}
def claim(text, *,cid,entity,kind,value,scope,team,statement,quote,confidence,temporal_relation,f_mapping,reason):
 x={'claim_id':cid,'team':team,'entity':entity,'kind':kind,'value':value,'scope':scope,'statement':statement,'quote':q(text,quote),'confidence':confidence,'temporal_relation':temporal_relation,'f_mapping':f_mapping,'f_mapping_reason':reason}
 return x

def main():
 specs=[
  ('ur1-4cd71f9f33133a22c7448145',[
   dict(cid='sarr-01',entity='Ismaïla Sarr',kind='availability',value='recovering',scope='team_context',team='away',statement='Sarr is returning toward form after limited summer training and staged minutes; the article does not state target-match availability.',quote='The Senegal forward was withdrawn midway through the second-half, but Sage believes he could now be ready to complete a full 90 minutes.',confidence='medium',temporal_relation='recent_prior_match_to_target',f_mapping='unmodeled',reason='conditional readiness is not an explicit available/unavailable/doubtful target-match assertion under the frozen protocol.'),
   dict(cid='sarr-02',entity='Dean Henderson',kind='availability',value='unavailable',scope='this_match',team='away',statement='The article states that Palace captain Dean Henderson is out with injury and that Wharton will lead at Brighton.',quote='With captain Dean Henderson currently out with injury, Adam Wharton has been leading the team and is set to do so again at Brighton on Sunday.',confidence='high',temporal_relation='target_match',f_mapping='away:availability:unavailable',reason='Explicit current injury and target-match context.'),
   dict(cid='sarr-03',entity='Adam Wharton',kind='selection',value='unchanged_planned',scope='this_match',team='away',statement='Wharton is set to lead Palace again at Brighton.',quote='With captain Dean Henderson currently out with injury, Adam Wharton has been leading the team and is set to do so again at Brighton on Sunday.',confidence='high',temporal_relation='target_match',f_mapping='away:selection:unchanged_planned',reason='Explicit planned role for the target fixture.'),
   dict(cid='sarr-04',entity='Crystal Palace squad',kind='schedule',value='congested',scope='this_match',team='away',statement='The article describes Palace playing Thursday to Sunday with both games away.',quote="When asked whether he was feeling the impact of Palace's Thursday-Sunday schedule, with both of this week's games on the road, Sage responded in typically upbeat fashion.",confidence='high',temporal_relation='target_week',f_mapping='away:schedule:congested',reason='Explicit target-week schedule context; current frozen f has a schedule feature.'),
  ]),
  ('ur1-efe647982f41ba4a345be862',[
   dict(cid='brighton-01',entity='Mats Wieffer; Yankuba Minteh; Kaoru Mitoma',kind='availability',value='unavailable',scope='this_match',team='home',statement='The article says Brighton will be without the named trio for the weekend fixture.',quote="The Brighton boss is set to be without influential trio Mats Wieffer, Yankuba Minteh and Kaoru Mitoma this weekend, however.",confidence='high',temporal_relation='target_match',f_mapping='home:availability:unavailable',reason='Explicit target-week absence statement.'),
   dict(cid='brighton-02',entity='Pascal Struijk',kind='availability',value='doubtful',scope='this_match',team='home',statement='Struijk is described as a doubt pending recovery and a next-day decision.',quote='Summer signing Pascal Struijk is also a doubt, with Hurzeler stating on Friday: "We don\'t train today so he gets another day of recovery, and then we\'ll see how we\'ll be tomorrow and then we make the decision."',confidence='high',temporal_relation='target_match',f_mapping='home:availability:doubtful',reason='Explicit doubtful status and pending decision.'),
   dict(cid='brighton-03',entity='Jack Hinshelwood',kind='selection',value='unchanged_planned',scope='team_context',team='home',statement='Hinshelwood returned in a midweek European match; the article does not state he is selected for Brighton–Palace.',quote="Hurzeler welcomed back Jack Hinshelwood in Brighton's midweek European win over Lithuanian team FK Kauno Žalgiris.",confidence='medium',temporal_relation='recent_prior_match_to_target',f_mapping='unmodeled',reason='Return in a prior match does not establish target-match selection or availability.'),
   dict(cid='brighton-04',entity='Brighton team',kind='tactical_plan',value='attacking',scope='team_context',team='home',statement='The article describes Brighton as attack-minded in season background; it does not state a target-match tactical plan.',quote='In his first season in charge the German led his attack-minded Brighton team to eighth place in the Premier League and an FA Cup quarter-final',confidence='medium',temporal_relation='season_background',f_mapping='unmodeled',reason='Frozen f accepts tactical_plan only when tied to a current match; this is season background.'),
  ])]
 records=[]
 for rid, cs in specs:
  p,text=load(rid); meta=json.loads((REC/f'{rid}.json').read_text(encoding='utf8'))
  outclaims=[]
  for c in cs:
   outclaims.append(claim(text,**c))
  records.append({'record_id':rid,'source_url':meta['source_url'],'canonical_match_id':CANON,'kickoff':KICK,'source_published_at':meta['source_published_at'],'retrieved_at_utc':meta['retrieved_at_utc'],'body_path':str(p),'body_sha256':sha(p),'review_method':'codex_read_only_structured_review','provider_output':False,'used_for_f_or_g':False,'claims':outclaims,'f_feature_delta':sorted({c['f_mapping'] for c in outclaims if c['f_mapping']!='unmodeled'}),'f_run_status':'not_run','g_label_status':'ineligible_missing_pre_cutoff_source_available_at_and_before_after_outcome_chain'})
 summary={'schema':'preannotation-v3-recovered-body-semantic-review-v1','status':'research_review_only','reviewed_at_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),'physical_match_id':CANON,'body_records':len(records),'claims':sum(len(r['claims']) for r in records),'modelable_claims_if_protocol_run':sum(len(r['f_feature_delta']) for r in records),'provider_calls':0,'f_calls':0,'g_training_rows':0,'important_boundary':'Current retrieval body is not retroactive evidence of source availability before the original cutoff.'}
 payload={'summary':summary,'records':records}
 (OUT/'recovered-body-semantic-review.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 lines=['Recovered Palace–Brighton body semantic review (research only)','',f"Claims reviewed: {summary['claims']}",f"Potential frozen-f mappings: {summary['modelable_claims_if_protocol_run']}",'Provider/f/g calls: 0','', 'The two current bodies are retained with source and body SHA. Their current retrieval clocks do not prove pre-kickoff operational availability; this file does not emit f predictions or gain labels.']
 (OUT/'recovered-body-semantic-review.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
 # refresh manifest
 files=[]
 for x in sorted(OUT.rglob('*')):
  if x.is_file() and x.name!='manifest.json': files.append({'path':str(x.relative_to(OUT)).replace('\\','/'),'bytes':x.stat().st_size,'sha256':sha(x)})
 (OUT/'manifest.json').write_text(json.dumps({'schema':'preannotation-v3-canonical-candidate-audit-manifest-v1','files':files},ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

