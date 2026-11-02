"""Bounded offline f replay on two additional non-Leeds matches.

This is a research artifact: source reviews are explicit, historical versions
are unknown, and the resulting g fit is diagnostic only. No network is allowed
while this script runs.
"""
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json, socket, sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from tools.train_reviewed_rich_f import history, write
from tools.train_state_candidate_g import before_snapshot
from tools.reconstruct_panel_priors import TARGETS, no_network
from football_research.acquisition.match_conditioned_f import ReconstructedMatchPrior, bind_prior
from football_research.acquisition.historical_context_training import _discovery_panel, _project
from football_research.acquisition.news_link_projection import VERSION as PROJECTION
from football_research.acquisition.entity_semantic_f import EntityEvidenceBank
from football_research.acquisition.semantic_f import EXTRACTION_VERSION
from football_research.acquisition import entity_evidence_updates as ef
from football_research.acquisition import scoped_reviewed_f as scoped
from football_research.acquisition import reviewed_rich_f as rich
from football_research.acquisition import structured_state_gain as g
from tools.evaluate_nonleeds_g_probe import reviewed_claim

ROOTSRC=ROOT/'var/reviews/fm-g-nonleeds-expansion-20261015-v1'
F_PATH=ROOT/'var/reviews/fm-historical-transition-g-training-20261014-v2/f-labeler.json'
G_DIR=ROOT/'var/reviews/fm-g-structured-state-20261014-v1'

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def review(doc, text):
    p=doc['paragraphs']; ident=doc['id']; c=lambda *a,**k: reviewed_claim(text,*a,**k)
    if ident=='B01':
        body=[c('Cole Palmer','away','confirmed_absence',p[2],"Palmer is unavailable for Burnley",value='unavailable',context=p[1]),
              c('Benoit Badiashile','away','confirmed_available',p[9],"Badiashile is available for Burnley",value='available',context=p[6]),
              c('Enzo Fernandez','away','confirmed_available',p[10],"Fernandez and Neto are available for Burnley",value='available',context=p[7])]
    elif ident=='chelsea-stats':
        # Match-statistics prose is deliberately a negative/control document.
        body=[c('Chelsea','away','historical_match_stats',p[0],"Chelsea return to Premier League action",scope='team_context'),
              c('Burnley','home','historical_match_stats',p[1],"We have won eight of our nine Premier League away games against Burnley",scope='historical')]
    elif ident=='liverpool-lineup':
        body=[c('Alisson Becker and Florian Wirtz','away','confirmed_available',p[2],"return after recovering from respective fitness concerns",value='available'),
              c('Liverpool','away','confirmed_starter',p[4],"Liverpool starting line-up is supplied by the source",scope='this_match'),
              c('Liverpool','away','confirmed_bench',p[5],"Liverpool substitutes are supplied by the source",scope='this_match')]
    else: raise ValueError(ident)
    return {'body':body,'title':[],'reviewer':'current Codex session; exact serving model ID not exposed',
            'body_read_in_full':True,'historical_version_unknown':True}

def main():
    out=ROOTSRC/'evaluation'; out.mkdir(parents=True,exist_ok=False)
    socket.socket.connect=no_network; socket.getaddrinfo=no_network
    archive,sources=history(); by={f"{r['source']}:{r['source_id']}":r for r in archive}
    protocol=read(ROOT/'var/f-entity-semantic-20261011-v2/protocol.json')
    excluded=set(sum(protocol['splits'].values(),[])+TARGETS+['understat:28891','understat:28907'])
    labeler=read(F_PATH); models={k:g.load(G_DIR/f'g-{k}.json') for k in ('baseline','combined')}
    fixtures={'understat:28891':('BURNLEY-CHELSEA',['B01','chelsea-stats']),
              'understat:28907':('WESTHAM-LIVERPOOL',['liverpool-lineup'])}
    all_rows=[]; reports=[]
    for mid,(name,docids) in fixtures.items():
        target=by[mid]; fixture={'physical_match_id':mid,'match_id':mid,'home_team_id':target['home_team_id'],'away_team_id':target['away_team_id'],'home_name':target['home_team_name'],'away_name':target['away_team_name'],'competition_id':target['competition_id'],'competition':target['competition_name'],'kickoff':target['kickoff']}
        docs=[]; records=[]; concepts={}; reviews={}
        for did in docids:
            d=read(ROOTSRC/'sources'/f'{did}.json'); text=(ROOTSRC/'sources'/d['body_path']).read_text(encoding='utf-8'); reviews[did]=review(d,text)
            # Historical reconstruction requires a parseable capture clock. The
            # source publication clock is retained separately; this synthetic
            # archive clock is provenance metadata, never a live availability
            # claim.
            pub=datetime.fromisoformat(d['source_published_at'].replace('Z','+00:00'))
            docs.append({'url':d['url'],'title':d['title'],'text':text,'captured_at':(pub.replace(microsecond=0)+__import__('datetime').timedelta(hours=1)).isoformat(),'source_published_at':d['source_published_at']})
        case={'fixture':fixture,'documents':docs}; panel=list(_discovery_panel(case)); cutoff=panel[0][-1]['decision_at']
        data=[r for k,r in by.items() if k not in excluded or k==mid]; prior=ReconstructedMatchPrior(data,source_binding={'archives':sources,'excluded_targets':sorted(excluded)}).snapshot(fixture,cutoff)
        assert not set(prior['training_match_ids']) & excluded
        base=bind_prior(panel[0][-1],prior)
        for did,d in zip(docids,docs):
            one=deepcopy(base); one['documents']=[d]; items=EntityEvidenceBank._items(_project(one,text_projection=PROJECTION))
            for channel, vals in items.items():
                for key,pub in vals.items():
                    anns=reviews[did][channel]; claims=[x['claim'] for x in anns]
                    records.append({'input':pub,'response':{'schema':EXTRACTION_VERSION,'input_id':key,'claims':claims},'provenance':{'kind':'explicit_Codex_full_body_review'}})
                    for a in anns: concepts[key,a['claim']['statement']]=a['concept']
        bank=EntityEvidenceBank(records)
        for _,_,_,_,st in panel:
            st=bind_prior(st,prior); projected=bank.project(st); evid=[]
            for iid in projected['used_input_ids']:
                rec=bank.records[iid]
                for cl in rec['response']['claims']: evid.append(rich.evidence_tokens(cl,rec['input']['channel'],fixture,concepts.get((iid,cl['statement']))))
            row={'physical_match_id':mid,'decision_at':st['decision_at'],'result_available_at':None,'outcome':None,'prior':projected['prior'],'evidence':rich.unique_evidence(evid),'state':projected,'used_input_ids':projected['used_input_ids']}
            row['evidence_events']=[]
            for iid in row['used_input_ids']:
                rec=bank.records[iid]
                for cl in rec['response']['claims']:
                    row['evidence_events'].append({'physical_match_id':mid,'input_id':iid,'entity':cl['entity'],'claim':cl,'evidence':rich.evidence_tokens(cl,rec['input']['channel'],fixture,concepts.get((iid,cl['statement']))),'event_id':sha256(json.dumps(cl,sort_keys=True).encode()).hexdigest(),'source_published_at':rec['input'].get('source_published_at')})
            p0=np.asarray(row['prior']); p1=ef.predict(labeler,row); row['prediction_after']=p1.tolist(); row['prediction_before']=p0.tolist(); row['l1_shift']=float(np.abs(p1-p0).sum())
            all_rows.append(row)
        y=0 if target['home_goals']>target['away_goals'] else 1 if target['home_goals']==target['away_goals'] else 2
        body_rows=[r for r in all_rows if r['physical_match_id']==mid]
        reports.append({'physical_match_id':mid,'fixture':name,'outcome':y,'article_count':len(docids),'states':len(body_rows),'modeled_claims':sum(len(r['evidence']) for r in body_rows),'prediction_before':body_rows[0]['prediction_before'],'prediction_after':body_rows[-1]['prediction_after'],'prediction_l1_shift':body_rows[-1]['l1_shift'],'training_candidate':False,'source_overlap_note':('B01 is reused from the earlier Chelsea diagnostic; do not call it independent from that diagnostic.' if mid=='understat:28891' else 'One official lineup body; holdout diagnostic.')})
    write(out/'prediction-input-rows.json',all_rows); write(out/'match-reports.json',reports)
    write(out/'plan.json',{'schema':'fm-g-nonleeds-expansion-plan-v1','created_at':datetime.now(timezone.utc).isoformat(),'matches':list(fixtures),'f_model':str(F_PATH.relative_to(ROOT)),'g_models':{k:str((G_DIR/f'g-{k}.json').relative_to(ROOT)) for k in models},'network':0,'formal_runtime':False,'empirical_g_enabled':False,'source_version_unknown':True,'labels_joined_after_prediction':True})
    print(json.dumps(reports,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
