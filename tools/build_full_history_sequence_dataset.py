"""Build immutable historical sequence inputs from local archived data only.

No network, API, model inference or training. Targets stay in outcomes.jsonl.
All source files remain untouched; uncertain identities and clocks are retained.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from football_research.acquisition.full_history_sequence_metadata import METADATA_NAMES, document_metadata
BASE = Path('E:/FootballMind')
DEFAULT_OUT = BASE/'experiments/full-history-sequential-fg-20261025-v1/data'
PIPE = REPO/'var/corpora/football-news-pipeline-v1'
META = METADATA_NAMES
ALIASES = {'man united':'manchester united','man city':'manchester city',
           "nott'm forest":'nottingham forest','leeds united':'leeds',
           'wolverhampton wanderers':'wolves','tottenham hotspur':'tottenham'}
DIVISIONS = {'premier league':'E0','english premier league':'E0','英超':'E0',
             'serie a':'I1','意甲':'I1','bundesliga':'D1','德甲':'D1',
             'la liga':'SP1','laliga':'SP1','西甲':'SP1','ligue 1':'F1','法甲':'F1',
             'chinese super league':'CHN','中超':'CHN'}


def rows(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        for n,line in enumerate(f,1):
            if line.strip(): yield n,json.loads(line)


def write_json(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def write_rows(path,items):
    with path.open('w',encoding='utf-8',newline='\n') as f:
        for r in items:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')


def digest(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()
def norm(s):
    value=' '.join(str(s or '').casefold().split())
    return ALIASES.get(value,value)
def day(s):
    if not isinstance(s,str):return None
    try:return datetime.fromisoformat(s.replace('Z','+00:00')).date().isoformat()
    except ValueError:return None
def iso_sort(s):
    try:
        d=datetime.fromisoformat(s.replace('Z','+00:00'))
        if d.tzinfo is None:return (d.isoformat(),'timezone_unknown')
        return (d.astimezone(timezone.utc).isoformat(),'timestamp')
    except (ValueError,AttributeError):return ('','unknown')
def valid_prior(p):
    return isinstance(p,(list,tuple)) and len(p)==3 and all(isinstance(v,(int,float)) and math.isfinite(v) and 0<v<1 for v in p) and abs(sum(p)-1)<1e-8
def role(date):
    if not date:return 'unknown_date'
    if date<'2023-01-01':return 'f_fit'
    if date<'2025-01-01':return 'g_fit'
    if date<'2025-07-01':return 'g_dev'
    if date<'2026-01-01':return 'validation'
    return 'test'
def is_synthetic_path(path):
    s=str(path).replace('\\','/').lower()
    return any(t in s for t in ['/tests/','/test_','/pytest','/__pycache__/','/tmp/','/constructed-cases/','/gv2t-','/test-deepseek','/validation-repair-','/verification-pycache/'])


class Builder:
    def __init__(self,out):
        self.out=out;self.matches={};self.aliases={};self.keys=defaultdict(set)
        self.outcomes=defaultdict(list);self.priors=defaultdict(list)
        self.docs={};self.article_docs=defaultdict(set);self.links={};self.inputs=[]
        self.issues=[];self.alias_evidence=[];self.raw_docs=[]
        self.known_exclusions={}
        self.stream_paths=[]

    def inventory(self,p,kind,count):
        self.inputs.append({'path':str(p),'kind':kind,'rows':count,'bytes':p.stat().st_size})

    def add_match(self,r,p,n,provider=None):
        original=r.get('physical_match_id') or r.get('match_id') or r.get('fixture_id')
        if not original:return None
        home=r.get('home_team') or r.get('home') or r.get('home_name')
        away=r.get('away_team') or r.get('away') or r.get('away_name')
        kickoff=r.get('kickoff') or r.get('kickoff_at_utc') or r.get('kickoff_at')
        date=r.get('match_date') or r.get('date_local') or day(kickoff)
        division=r.get('division') or r.get('competition_code')
        competition=r.get('competition') or r.get('competition_name') or r.get('source_competition_name') or r.get('competition_key')
        if isinstance(competition,dict):competition=competition.get('name')
        if not division:division=DIVISIONS.get(str(competition or '').lower())
        if str(original).startswith('fd:'):
            chunks=str(original).split(':')
            if len(chunks)>=6:
                division=division or chunks[1];date=date or chunks[3]
                home=home or chunks[4];away=away or chunks[5]
        key=(division,date,norm(home),norm(away)) if division and date and home and away else None
        canonical=self.aliases.get(str(original))
        if canonical is None and key and len(self.keys.get(key,set()))==1:
            canonical=next(iter(self.keys[key]))
            self.alias_evidence.append({'original_id':str(original),'physical_match_id':canonical,'basis':'same_explicit_competition_date_and_ordered_team_names_with_frozen_aliases','identity_key':list(key),'source':str(p),'row':n})
        if canonical is None:canonical=str(original)
        self.aliases[str(original)]=canonical
        ref={'path':str(p),'row':n,'source_id':str(original),'provider':provider or r.get('provider') or r.get('source')}
        if canonical not in self.matches:
            self.matches[canonical]={'physical_match_id':canonical,'aliases':[str(original)],'match_date':date,'kickoff':kickoff,
                'division':division,'competition':competition,'season':r.get('season'),'home_team':home,'away_team':away,
                'identity_refs':[ref],'split_role':role(date),'identity_status':'explicit_source_identity','prior':None}
        else:
            m=self.matches[canonical]
            if str(original) not in m['aliases']:m['aliases'].append(str(original))
            m['identity_refs'].append(ref)
            for k,v in [('match_date',date),('kickoff',kickoff),('division',division),('home_team',home),('away_team',away)]:
                if m.get(k) is None and v is not None:m[k]=v
            m['split_role']=role(m['match_date'])
        if key:self.keys[key].add(canonical)
        return canonical

    def add_prior(self,mid,p,ref,basis,verified=False):
        mid=self.aliases.get(mid,mid)
        if mid in self.matches and valid_prior(p):self.priors[mid].append({'probabilities':list(p),'source':ref,'basis':basis,'time_verified':verified})

    def add_outcome(self,mid,y,ref):
        mid=self.aliases.get(mid,mid)
        if mid in self.matches and y in (0,1,2):self.outcomes[mid].append({'outcome_class':int(y),'source':ref})

    def structured(self):
        root=BASE/'corpus-v3/structured'
        p=root/'matches.jsonl';rr=list(rows(p));self.inventory(p,'structured_matches',len(rr))
        for n,r in rr:self.add_match(r,p,n,'football-data.co.uk')
        p=root/'market_snapshots.jsonl';rr=list(rows(p));self.inventory(p,'market',len(rr))
        for n,r in rr:
            for book in ['Avg','B365','PS']:
                s=r.get('snapshots',{}).get(book,{})
                if s.get('snapshot_kind')!='closing' and valid_prior(s.get('no_vig_probabilities')):
                    self.add_prior(r['match_id'],s['no_vig_probabilities'],{'path':str(p),'row':n,'book':book,'raw_source':r.get('source')},'source_legacy_nonclosing_odds_unknown_quote_time');break
        p=root/'labels/results.jsonl';rr=list(rows(p));self.inventory(p,'outcome',len(rr))
        for n,r in rr:self.add_outcome(r['match_id'],{'H':0,'D':1,'A':2}.get(r.get('outcome')),{'path':str(p),'row':n,'raw_source':r.get('source')})
        root=REPO/'var/external/football-data-co-uk-20261024-v2/projected'
        p=root/'matches.jsonl';rr=list(rows(p));self.inventory(p,'additional_structured_matches',len(rr))
        for n,r in rr:self.add_match(r,p,n,'football-data.co.uk')
        p=root/'market-snapshots.jsonl';rr=list(rows(p));self.inventory(p,'additional_market',len(rr))
        for n,r in rr:
            for book in ['avg_1x2','b365_1x2']:
                q=r.get(book,{})
                odds=[q.get(k) for k in ['home','draw','away']]
                if all(isinstance(x,(int,float)) and math.isfinite(x) and x>1 for x in odds):
                    inv=[1/x for x in odds];pv=[x/sum(inv) for x in inv]
                    self.add_prior(r['match_id'],pv,{'path':str(p),'row':n,'book':book,'source_file':r.get('source_file'),'source_sha256':r.get('source_sha256')},'legacy_odds_unknown_quote_time');break
        p=root/'labels-results.jsonl';rr=list(rows(p));self.inventory(p,'additional_outcome',len(rr))
        for n,r in rr:self.add_outcome(r['match_id'],r.get('outcome_class'),{'path':str(p),'row':n,'source_file':r.get('source_file'),'source_sha256':r.get('source_sha256')})

    def discover_streams(self):
        paths=[]
        cmd=['rg','--files',str(REPO/'var'),str(REPO/'footballmind-information-stream'),'-g','fixtures.jsonl']
        completed=subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8',check=True)
        for s in completed.stdout.split('\n'):
            if not s.strip():continue
            p=Path(s.strip())
            if is_synthetic_path(p) or '/releases/' in p.as_posix():continue
            paths.append(p)
        for p in sorted(set(paths)):
            rr=list(rows(p));self.inventory(p,'archived_fixture_registry',len(rr))
            for n,r in rr:self.add_match(r,p,n)
            root=p.parent;self.stream_paths.append(root)
            label=root/'labels/results.jsonl'
            if label.exists():
                for n,r in rows(label):
                    mid=r.get('physical_match_id') or r.get('match_id')
                    h=r.get('regulation_home_goals',r.get('home_goals'));a=r.get('regulation_away_goals',r.get('away_goals'))
                    if isinstance(h,int) and isinstance(a,int):self.add_outcome(mid,0 if h>a else 1 if h==a else 2,{'path':str(label),'row':n,'source_url':r.get('result_source_url'),'raw_sha256':r.get('raw_sha256')})
        # Current persisted live catalog is inventory only; its displayed scores are not settlement labels.
        p=REPO/'var/live-coverage-runs/live-g/catalog.json'
        if p.exists():
            rr=json.loads(p.read_text(encoding='utf-8')).get('matches',[]);self.inventory(p,'live_catalog_inventory_only',len(rr))
            for n,r in enumerate(rr,1):self.add_match(r,p,n)

    def add_doc(self,aid,text,meta,ref,body_path=None,expected_sha=None):
        if not isinstance(text,str) or not text.strip():return None
        actual=digest(text)
        if expected_sha and actual!=expected_sha:
            self.issues.append({'article_id':aid,'reason':'body_sha_mismatch','reference':ref,'expected':expected_sha,'actual':actual});return None
        did='sha256:'+actual
        source={'article_id':aid,'source_url':meta.get('url') or meta.get('source_url'),
            'source_domain':meta.get('source_domain') or meta.get('host') or urlsplit(meta.get('url') or meta.get('source_url') or '').hostname,
            'title':meta.get('title'),'published_at':meta.get('source_published_at') or meta.get('published_at'),
            'updated_at':meta.get('source_updated_at') or meta.get('source_modified_at') or meta.get('updated_at') or meta.get('modified_at'),
            'captured_at':meta.get('captured_at') or meta.get('retrieved_at'),
            'source_available_at':meta.get('source_available_at') or meta.get('available_at'),
            'system_available_at':None,'system_available_at_basis':'unknown_historical_availability; capture_record_separate',
            'source_type':meta.get('source_type'),'body_reference':ref,'body_path':str(body_path) if body_path else None,
            'historical_version_unknown':meta.get('historical_version_unknown',True),'information_mode':meta.get('information_mode') or 'historical_reconstruction'}
        if did not in self.docs:
            self.docs[did]={'document_id':did,'text_sha256':actual,'body_characters':len(text),'body_utf8_bytes':len(text.encode()),
                'body_path':str(body_path) if body_path else None,'text_reference':ref,'source_url':source['source_url'],
                'source_domain':source['source_domain'],'published_at':source['published_at'],'captured_at':source['captured_at'],
                'source_available_at':source['source_available_at'],'historical_version_unknown':source['historical_version_unknown'],
                'article_ids':[],'source_versions':[],'existing_claims':None,'body_reference_verified':True}
        doc=self.docs[did]
        if aid not in doc['article_ids']:doc['article_ids'].append(aid)
        if source not in doc['source_versions']:doc['source_versions'].append(source)
        self.article_docs[aid].add(did)
        return did

    def add_link(self,mid,did,aid,meta,ref,binding,accepted=True,claims=None):
        mid=self.aliases.get(mid,mid)
        if mid not in self.matches or did not in self.docs:
            self.issues.append({'reason':'unresolved_match_document_join','physical_match_id':mid,'document_id':did,'article_id':aid,'reference':ref});return
        date=self.matches[mid]['match_date'];pub=meta.get('source_published_at') or meta.get('published_at')
        upd=meta.get('source_updated_at') or meta.get('source_modified_at') or meta.get('updated_at') or meta.get('modified_at')
        pd,ud=day(pub),day(upd);reasons=[]
        if not accepted:reasons.append('association_not_accepted_or_scope_unknown')
        if not date or not pd:reasons.append('publication_or_match_date_unknown')
        elif pd>=date:reasons.append('publication_not_before_match_date')
        if date and ud and ud>=date:reasons.append('updated_on_or_after_match_date')
        rejection=self.known_exclusions.get(aid)
        if rejection:reasons.append('known_full_body_temporal_contamination')
        status='historical_sequence_candidate' if not reasons else 'diagnostic_only'
        elapsed=(datetime.fromisoformat(date)-datetime.fromisoformat(pd)).days if date and pd else None
        source_type=meta.get('source_type');official=source_type=='official_club' or bool(meta.get('publisher') and 'official' in str(meta.get('document_kind','')))
        meta_vec=[math.log1p(max(0,elapsed)) if elapsed is not None else 0.0,float(elapsed is None),float(meta.get('historical_version_unknown',True)),float(official),math.log1p(self.docs[did]['body_characters']),1.0]
        item={'physical_match_id':mid,'document_id':did,'article_id':aid,'published_at':pub,'updated_at':upd,'source_available_at':meta.get('source_available_at') or meta.get('available_at'),
              'captured_at':meta.get('captured_at') or meta.get('retrieved_at'),'binding':binding,'binding_reference':ref,'association_accepted':accepted,
              'source_type':source_type,'historical_version_unknown':meta.get('historical_version_unknown',True),
              'historical_sequence_eligible':not reasons,'exclusion_reasons':reasons,'training_route':status,'known_exclusion':rejection,
              'existing_claims':claims,'time_basis':'source_declared_publication_current_historical_version_not_prospective'}
        key=(mid,did)
        if key not in self.links:self.links[key]=item|{'alternative_link_evidence':[]}
        else:
            old=self.links[key]
            if not old['historical_sequence_eligible'] and item['historical_sequence_eligible']:
                self.links[key]=item|{'alternative_link_evidence':old.get('alternative_link_evidence',[])+[{'article_id':old['article_id'],'reference':old['binding_reference'],'reasons':old['exclusion_reasons']}]}
            else:old['alternative_link_evidence'].append({'article_id':aid,'reference':ref,'reasons':reasons})
        if claims and self.docs[did]['existing_claims'] is None:self.docs[did]['existing_claims']=claims

    def article_corpus(self):
        excluded=BASE/'experiments/postread-clean-expansion-20261025-v1/excluded-review.jsonl'
        self.known_exclusions={r['candidate_text_id']:r for _,r in rows(excluded)}
        p=PIPE/'articles.jsonl';canonical={r['article_id']:r for _,r in rows(p)};self.inventory(p,'canonical_article_metadata',len(canonical))
        wanted={aid:r for aid,r in canonical.items() if r.get('body_status')=='full_text'}
        resolved=set()
        for aid,r in wanted.items():
            bp=Path(r.get('body_path') or '__missing__')
            if bp.is_file():
                try:text=bp.read_bytes().decode('utf-8')
                except UnicodeDecodeError:self.issues.append({'article_id':aid,'reason':'invalid_utf8'});continue
                if self.add_doc(aid,text,r,{'path':str(bp),'kind':'utf8_text'},bp,r.get('recorded_body_sha256')):resolved.add(aid)
        # Only full-text article records are read. No raw HTML or network extraction.
        for bp in sorted((PIPE/'batches').glob('batch-*/articles.jsonl')):
            count=0
            for n,r in rows(bp):
                count+=1;aid=r.get('article_id')
                if aid not in wanted or aid in resolved:continue
                meta={**canonical[aid],**r};text=r.get('body_text')
                expected=canonical[aid].get('recorded_body_sha256')
                if self.add_doc(aid,text,meta,{'path':str(bp),'jsonl_record':n,'article_id':aid,'field':'body_text','body_sha256':expected},expected_sha=expected):resolved.add(aid)
            self.inventory(bp,'inline_article_batch',count)
        for aid,r in canonical.items():
            self.raw_docs.append({'article_id':aid,'body_status':r.get('body_status'),'body_sha256':r.get('recorded_body_sha256'),'document_ids':sorted(self.article_docs.get(aid,set())),
                'resolved_full_body':aid in resolved,'source_domain':r.get('source_domain'),'published_at':r.get('published_at'),
                'route':'body_registry' if aid in resolved else 'missing_or_nonfull_body_diagnostic','source_path':str(p)})
        # Explicit candidate links and automated attribution are retained separately from body semantics.
        decisions={(r['article_id'],r['physical_match_id']):r for _,r in rows(PIPE/'association-decisions.jsonl')}
        lp=PIPE/'article_match_links.jsonl';links=list(rows(lp));self.inventory(lp,'article_match_links',len(links))
        for n,r in links:
            aid=r['article_id'];mid=r['physical_match_id'];decision=decisions.get((aid,mid),{})
            for did in self.article_docs.get(aid,set()):
                self.add_link(mid,did,aid,canonical[aid],{'path':str(lp),'row':n,'association_decision':decision},r.get('binding_method'),bool(decision.get('final_counted')))
        # Legacy E corpus links were copied into pipeline; load exact extra references if not present.
        cr=BASE/'corpus-v3';aa={r['article_id']:r for _,r in rows(cr/'news/articles.jsonl')}
        for n,r in rows(cr/'research/match_article_candidates.jsonl'):
            aid=r['article_id']
            for did in self.article_docs.get(aid,set()):
                self.add_link(r['match_id'],did,aid,aa.get(aid,{}),{'path':str(cr/'research/match_article_candidates.jsonl'),'row':n},r.get('binding'),True)

    def stream_documents(self):
        for root in self.stream_paths:
            p=root/'documents.jsonl'
            if not p.exists():continue
            count=0
            for n,r in rows(p):
                count+=1;aid=r.get('document_id') or r.get('article_id') or f'{p}:{n}'
                body=r.get('body') or r.get('text')
                if not isinstance(body,str):continue
                if any(x in str(r.get('url','')) for x in ['example.com','example.invalid']):continue
                did=self.add_doc(aid,body,r,{'path':str(p),'jsonl_record':n,'field':'body' if 'body' in r else 'text'},expected_sha=r.get('body_sha256'))
                if did:self.add_link(r.get('physical_match_id') or r.get('match_id'),did,aid,r,{'path':str(p),'row':n},r.get('fixture_binding_basis') or 'archived_fixture_document_link',True)
            self.inventory(p,'archived_full_document_records',count)
        # Exact frozen experiment documents add claims-ready texts not necessarily in a canonical stream.
        p=BASE/'experiments/postread-clean-expansion-20261025-v1/documents.jsonl'
        for n,r in rows(p):
            mid=r['physical_match_id'];aid=r.get('semantic_input_id') or r['text_sha256']
            did=self.add_doc(aid,r['text'],r,{'path':str(p),'jsonl_record':n,'field':'text'},expected_sha=r['text_sha256'])
            if did:self.add_link(mid,did,aid,r,{'path':str(p),'row':n},'previous_full_body_reviewed_frozen_dataset',True)

    def finish(self):
        sidecar=BASE/'experiments/postread-semantic-training-ready-20261025-v1/context-sidecar/state-candidate-context.jsonl'
        for _,r in rows(sidecar):
            c=r.get('candidate',{});did='sha256:'+str(c.get('text_sha256'));mid=self.aliases.get(r['physical_match_id'],r['physical_match_id'])
            if did in self.docs:
                self.docs[did]['existing_claims']=c.get('assertions')
                self.docs[did]['semantic_input_ids']=sorted(set(self.docs[did].get('semantic_input_ids',[])+[c['semantic_input_id']]))
                if (mid,did) in self.links:self.links[(mid,did)]['existing_claims']=c.get('assertions')
        outcomes=[]
        for mid,m in self.matches.items():
            ps=self.priors.get(mid,[])
            if ps:
                m.update(prior=ps[0]['probabilities'],prior_source=ps[0]['source'],prior_basis=ps[0]['basis'],prior_time_verified=ps[0]['time_verified'],prior_alternatives=ps[1:])
            else:m.update(prior_source=None,prior_basis=None,prior_time_verified=False,prior_alternatives=[])
            yy=self.outcomes.get(mid,[]);values={r['outcome_class'] for r in yy}
            outcomes.append({'physical_match_id':mid,'outcome_class':next(iter(values)) if len(values)==1 else None,'source_refs':yy,'conflicting_results':len(values)>1})
        ys={r['physical_match_id']:r for r in outcomes}
        # Freeze f's development tail before any learner runs; never use g_fit labels to fit f.
        eligible_body_matches={mid for (mid,_),link in self.links.items() if link['historical_sequence_eligible']}
        fdates=sorted(m['match_date'] for m in self.matches.values() if m['split_role']=='f_fit' and m['physical_match_id'] in eligible_body_matches and valid_prior(m['prior']) and ys[m['physical_match_id']]['outcome_class'] is not None)
        fdev_date=fdates[min(len(fdates)-1,math.floor(len(fdates)*.8))] if fdates else None
        for m in self.matches.values():
            m['fit_role']='f_dev' if m['split_role']=='f_fit' and fdev_date and m['match_date']>=fdev_date else 'f_train' if m['split_role']=='f_fit' else m['split_role']
        old=BASE/'experiments/postread-reversal-corrected-20261025-v2/experiment-plan.json'
        protocols=json.loads(old.read_text(encoding='utf-8'))
        self.out.joinpath('development-protocols').mkdir(exist_ok=True)
        shutil.copyfile(old,self.out/'development-protocols/original-corrected-experiment-plan.json')
        seen=set()
        def find_ids(obj):
            if isinstance(obj,dict):
                for k,v in obj.items():
                    if isinstance(k,str) and (k.startswith('fd:') or k.startswith('understat:')):seen.add(self.aliases.get(k,k))
                    find_ids(v)
            elif isinstance(obj,list):
                for x in obj:find_ids(x)
            elif isinstance(obj,str) and (obj.startswith('fd:') or obj.startswith('understat:')):seen.add(self.aliases.get(obj,obj))
        find_ids(protocols)
        exposure_refs=defaultdict(list)
        for d in sorted((BASE/'experiments').iterdir()):
            if not d.is_dir() or d.name.startswith('full-history-'):continue
            for name in ['protocol.json','split.json','experiment-plan.json','panel.jsonl']:
                p=d/name
                if not p.exists():continue
                before=set(seen)
                if name.endswith('.jsonl'):
                    for _,z in rows(p):find_ids(z.get('match_id') or z.get('physical_match_id'))
                else:
                    try:find_ids(json.loads(p.read_text(encoding='utf-8')))
                    except (UnicodeError,json.JSONDecodeError):continue
                for mid in seen-before:exposure_refs[mid].append(str(p))
        for m in self.matches.values():m['previous_development_exposure']=m['physical_match_id'] in seen
        # Same body across roles is a diagnostic relation, not an unobserved test text.
        bodyroles=defaultdict(set)
        for (mid,did),link in self.links.items():
            if link['historical_sequence_eligible']:bodyroles[did].add(self.matches[mid]['fit_role'])
        crossing={did:sorted(rs) for did,rs in bodyroles.items() if len(rs)>1}
        grouped=defaultdict(list)
        for (mid,did),link in self.links.items():
            link['cross_role_body_overlap']=did in crossing
            if did in crossing:link['historical_sequence_eligible']=False;link['exclusion_reasons'].append('exact_body_shared_across_fit_roles');link['training_route']='diagnostic_only'
            grouped[mid].append(link)
        sequences=[];routes=[]
        for mid,m in sorted(self.matches.items()):
            links=grouped.get(mid,[]);eligible=[r for r in links if r['historical_sequence_eligible']]
            eligible.sort(key=lambda r:(iso_sort(r['published_at'])[0],r['document_id']))
            ids=[r['document_id'] for r in eligible];target=ys[mid]['outcome_class']
            contexts=[{k:r.get(k) for k in ['published_at','updated_at','source_available_at','captured_at','source_type','historical_version_unknown','existing_claims']} for r in eligible]
            doc_objects=[{**self.docs[r['document_id']],**contexts[i]} for i,r in enumerate(eligible)]
            metas=[document_metadata(d,doc_objects[:i],m['match_date']) for i,d in enumerate(doc_objects)]
            for i,r in enumerate(eligible):
                r['document_meta']=metas[i]
                r['document_meta_basis']='frozen_full_sequence_prefix; recompute_for_any_other_adopted_order'
            can_supervise=valid_prior(m['prior']) and target is not None
            seq={'physical_match_id':mid,'split_role':m['split_role'],'fit_role':m['fit_role'],'match_date':m['match_date'],
                'document_ids':ids,'document_meta':metas,
                'documents':[{'document_id':r['document_id'],'text_sha256':self.docs[r['document_id']]['text_sha256'],'vector_key':self.docs[r['document_id']]['text_sha256'],'meta':metas[i],'metadata_context':contexts[i]} for i,r in enumerate(eligible)],
                'prefix_lengths':list(range(len(ids)+1)),'prefix_weights':[1.0/(len(ids)+1)]*(len(ids)+1),
                'order_basis':'source_declared_publication_time_then_text_sha; historical reconstruction',
                'unknown_time_document_ids':[r['document_id'] for r in links if 'publication_or_match_date_unknown' in r['exclusion_reasons']],
                'diagnostic_document_ids':[r['document_id'] for r in links if not r['historical_sequence_eligible']],
                'eligible':can_supervise and bool(ids),'structured_baseline_eligible':can_supervise,
                'historical_assumption':True,'previous_development_exposure':m['previous_development_exposure']}
            sequences.append(seq)
            reasons=[]
            if not valid_prior(m['prior']):reasons.append('prior_missing')
            if target is None:reasons.append('outcome_missing_or_conflicting')
            if not ids:reasons.append('no_time_eligible_linked_body')
            routes.append({'physical_match_id':mid,'split_role':m['split_role'],'fit_role':m['fit_role'],'reasons':reasons,'route':'sequence_supervision' if seq['eligible'] else 'structured_baseline_only' if can_supervise else 'registry_pending','diagnostic_body_count':len(seq['diagnostic_document_ids'])})
        schema={'schema':'footballmind-full-history-sequential-fg-v1','document_meta_names':META,
            'outcome_order':['home','draw','away'],'features_file':'matches.jsonl','labels_file':'outcomes.jsonl',
            'documents_file':'documents.jsonl','sequences_file':'sequences.jsonl','document_id_basis':'sha256:exact_utf8_full_body',
            'text_reference_record_basis':'1-based physical JSONL line; blank lines absent in source batch files',
            'body_text_encoding':'UTF-8 unchanged; use raw bytes decode for text files','prefix_weights':'equal total per match including empty prefix',
            'unknown_time_rule':'retain diagnostic set, do not invent chronological order','all_data_historical_reconstruction_unless_source_asserts_real_observation':True,
            'metadata_unknown_rule':'explicit missing indicators; official=false means not explicitly declared official, not false-source judgment',
            'metadata_rebuild_function':'football_research.acquisition.full_history_sequence_metadata.document_metadata',
            'metadata_context_merge':'documents[document_id] merged with match-specific sequence.documents[].metadata_context; actual selected prefix passed to helper',
            'claim_relation_semantics':'exact claims / availability multistatus only; not truth or contradiction; missing claims unknown; factual_update_unknown always1',
            'no_gain_based_filter':True,'no_claims_requirement':True,'target_fields_never_embedding_input':['outcome_class','gain','p_after'],
            'source_data_unchanged':True,'network_requests':0,'model_calls':0}
        stats={'matches':len(self.matches),'documents':len(self.docs),'article_registry_rows':len(self.raw_docs),'links':len(self.links),
            'sequence_matches':sum(r['eligible'] for r in sequences),'sequence_document_occurrences':sum(len(r['document_ids']) for r in sequences if r['eligible']),
            'sequence_unique_documents':len({d for r in sequences if r['eligible'] for d in r['document_ids']}),
            'structured_baseline_matches':sum(r['structured_baseline_eligible'] for r in sequences),
            'sequence_matches_by_role':dict(Counter(r['fit_role'] for r in sequences if r['eligible'])),
            'structured_matches_by_role':dict(Counter(r['fit_role'] for r in sequences if r['structured_baseline_eligible'])),
            'exact_cross_role_body_groups':len(crossing),'alias_merges':len(self.alias_evidence),'unresolved_issues':len(self.issues),'f_dev_start_date':fdev_date,
            'f_dev_split_basis':'last20percent_in_time_of_f_fit_supervisable_body_matches; same_day_together; fixed_before_training; same_date_for_structured_calibration',
            'f_dev_split_build_adjustment':'initial all-structured tail overallocated sparse body coverage; changed before any model fitting with controller approval',
            'current_semantic_f_not_retrained':True,'network_requests':0,'model_calls':0,'training':0}
        write_rows(self.out/'matches.jsonl',sorted(self.matches.values(),key=lambda r:r['physical_match_id']))
        write_rows(self.out/'outcomes.jsonl',sorted(outcomes,key=lambda r:r['physical_match_id']))
        write_rows(self.out/'documents.jsonl',sorted(self.docs.values(),key=lambda r:r['document_id']))
        write_rows(self.out/'match-document-links.jsonl',self.links.values());write_rows(self.out/'sequences.jsonl',sequences)
        write_rows(self.out/'article-registry.jsonl',self.raw_docs);write_rows(self.out/'missing-routing.jsonl',routes)
        write_rows(self.out/'alias-evidence.jsonl',self.alias_evidence);write_rows(self.out/'issues.jsonl',self.issues)
        write_json(self.out/'source-inventory.json',self.inputs);write_json(self.out/'schema.json',schema)
        write_json(self.out/'split.json',{'schema':'full-history-fg-temporal-split-v1','boundaries':{'f_fit_end_exclusive':'2023-01-01','g_fit_end_exclusive':'2025-01-01','g_dev_end_exclusive':'2025-07-01','validation_end_exclusive':'2026-01-01','f_dev_start_date':fdev_date},'assignments':{m['physical_match_id']:m['fit_role'] for m in self.matches.values()},'old_anchor_protocols':'development-protocols/original-corrected-experiment-plan.json','test_may_have_previous_development_exposure':True})
        write_json(self.out/'duplicate-body-split-audit.json',{'purged_from_temporal_sequences':crossing,'diagnostic_records_retained':True})
        write_json(self.out/'dataset-summary.json',stats)
        write_rows(self.out/'development-exposure.jsonl',[{'physical_match_id':mid,'previous_development_exposure':True,'first_discovery_sources':exposure_refs.get(mid,[]),'initial_corrected_protocol_member':mid in {self.aliases.get(x,x) for x in protocols.get('named_splits',{}).get('P0_original',{})}} for mid in sorted(seen) if mid in self.matches])
        required={d for s in sequences if s['eligible'] for d in s['document_ids']}
        queue=[]
        for did in sorted(required):
            d=self.docs[did]
            queue.append({'document_id':did,'text_sha256':d['text_sha256'],'body_path':d['body_path'],'text_reference':d['text_reference'],
                'body_characters':d['body_characters'],'physical_match_ids':sorted({s['physical_match_id'] for s in sequences if did in s['document_ids']}),'embedding_input':'body_only'})
        write_rows(self.out/'encoding-candidates.jsonl',queue)
        print(json.dumps(stats,ensure_ascii=False,indent=2),flush=True)


def materialize_encoding_inputs(out):
    """Resolve only sequence-required bodies, reusing prepared UTF-8 files first."""
    queue=[r for _,r in rows(out/'encoding-candidates.jsonl')]
    matches={r['physical_match_id']:r for _,r in rows(out/'matches.jsonl')}
    previous=BASE/'experiments/postread-semantic-training-ready-20261025-v1/corpus-queue/texts'
    target=out/'encoding-texts';target.mkdir(exist_ok=True)
    unresolved=defaultdict(list);ready=[]
    def finish(q,text,path):
        actual=digest(text)
        if actual!=q['text_sha256']:raise ValueError(f'Encoding text SHA mismatch: {q["document_id"]}')
        if path is None:
            path=target/(actual+'.txt')
            if not path.exists():path.write_bytes(text.encode('utf-8'))
            elif hashlib.sha256(path.read_bytes()).hexdigest()!=actual:raise ValueError('Existing output body changed')
        mids=q['physical_match_ids'];role_order={'f_train':0,'f_dev':1,'g_fit':2,'g_dev':3,'validation':4,'test':5,'unknown_date':6}
        first=min((matches[mid] for mid in mids),key=lambda m:(role_order.get(m['fit_role'],7),m['match_date'] or '',m['physical_match_id']))
        ready.append({**q,'text_path':str(path),'physical_match_id':first['physical_match_id'],'match_date':first['match_date'],'role':first['fit_role'],'body_verified':True})
    for q in queue:
        cache=previous/(q['text_sha256']+'.txt')
        bp=cache if cache.is_file() else Path(q.get('body_path') or '__missing__')
        if bp.is_file():finish(q,bp.read_bytes().decode('utf-8'),bp)
        else:unresolved[q['text_reference']['path']].append(q)
    for path,qs in unresolved.items():
        byline=defaultdict(list)
        for q in qs:byline[q['text_reference']['jsonl_record']].append(q)
        for n,r in rows(path):
            for q in byline.get(n,[]):
                field=q['text_reference'].get('field','body_text')
                finish(q,r[field],None)
    if len(ready)!=len(queue):raise ValueError('Unresolved sequence body references')
    rank={'f_train':0,'f_dev':1,'g_fit':2,'g_dev':3,'validation':4,'test':5,'unknown_date':6}
    ready.sort(key=lambda r:(rank.get(r['role'],7),r['match_date'] or '',r['physical_match_id'],r['text_sha256']))
    write_rows(out/'eligible-encoding-inputs.jsonl',ready)
    write_json(out/'encoding-input-summary.json',{'texts':len(ready),'roles':dict(Counter(r['role'] for r in ready)),'bytes':sum(Path(r['text_path']).stat().st_size for r in ready),'all_sha_verified':True})
    print(f'Encoding input files ready: {len(ready)}',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--materialize-only',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.materialize_only:materialize_encoding_inputs(args.output);return
    b=Builder(args.output)
    print('Building structured registry...',flush=True);b.structured()
    print('Reading archived fixture registries...',flush=True);b.discover_streams()
    print('Resolving full-text corpus and links...',flush=True);b.article_corpus()
    print('Reading existing canonical full-body streams...',flush=True);b.stream_documents()
    print('Freezing temporal assignments and sequence manifests...',flush=True);b.finish()
    materialize_encoding_inputs(args.output)
    from tools.augment_full_history_target_binding import augment
    augment(args.output)


if __name__=='__main__':main()
