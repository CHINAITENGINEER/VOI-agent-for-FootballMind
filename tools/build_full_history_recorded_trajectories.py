"""Recover archived real execution order without network, refitting, or old-label reuse.

Only whitelisted fields are exported. Provider credentials/configuration envelopes
are never exported. An absent public after is a failure, even if a stale numerical
prediction_after array remains in the run artifact.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
BASE=Path('E:/FootballMind/experiments/full-history-sequential-fg-20261025-v1')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def lines(path):
    with Path(path).open(encoding='utf-8-sig') as f:return [json.loads(s) for s in f if s.strip()]
def seq(value):return value if isinstance(value,list) else []
def obj(value):return value if isinstance(value,dict) else {}
def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
def jsonl(path,rows):
    with Path(path).open('x',encoding='utf8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
def before(a,b):
    if not a or not b:return None
    try:return datetime.fromisoformat(a.replace('Z','+00:00'))<datetime.fromisoformat(b.replace('Z','+00:00'))
    except (ValueError,TypeError):return None
def prob(value):
    import math
    if not isinstance(value,list) or len(value)!=3:return None
    if not all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and 0<x<=1 for x in value):return None
    return value if abs(sum(value)-1)<1e-6 else None


def path_scope(path):
    low=str(path).lower().replace('\\','/')
    if any(s in low for s in ('pytest','/test-','/test_','/tests/','constructed','/mock','/synthetic','/gateway-','/role0/','validation-repair','validation-no','lock-validation')):
        return 'test_or_constructed_excluded'
    if any(s in low for s in ('replay','/parallel/','/verification/','offline-recompute','/audit-case')):
        return 'replay_or_development_diagnostic'
    return 'recorded_execution'


def files(roots):
    result=[]
    for root in roots:
        if not root.exists():continue
        cmd=['rg','--files',str(root)]
        for pat in ('run.json','*runtime-trace*.json','*shadow-trace*.json','*journal*.json','collection-report.json','attempts.jsonl','contexts.jsonl','documents.jsonl','results.jsonl','*settlement*.json','entity-semantic-extractions.json','*provider-receipt*.json','quick-f-report.json','*.zip'):
            cmd+=['-g',pat]
        result.extend(Path(x) for x in subprocess.check_output(cmd,encoding='utf8').splitlines())
    return sorted(set(result))


def model_binding(path,run):
    live=obj(run.get('model_bindings'));binding=live.get('f');version=None;ref=None
    if obj(run.get('model')).get('path'):
        model=run['model'];p=Path(model['path']);actual=sha(p) if p.exists() else None
        return dict(raw_sha256=actual,declared_sha256=model.get('model_sha256'),verified=actual==model.get('model_sha256'),model_version=model.get('model_version'),path=str(p))
    for name in ('context-predictor.json','f.json'):
        p=path.parent/name
        if p.exists():
            raw=read(p);actual=sha(p)
            version=obj(raw).get('model_version');ref=str(p)
            if binding and binding!=actual:return dict(raw_sha256=actual,declared_sha256=binding,verified=False,model_version=version,path=ref)
            return dict(raw_sha256=actual,declared_sha256=binding,verified=True,model_version=version,path=ref)
    return dict(raw_sha256=binding,declared_sha256=binding,verified=False,model_version=version,path=ref)


def body_input_record(inp,document,*,input_ref,claims=None):
    text=inp.get('text',inp.get('body'))
    if not isinstance(text,str) or not text.strip():return None
    digest=hashlib.sha256(text.encode('utf8')).hexdigest()
    return dict(document_id='sha256:'+digest,text_sha256=digest,text=text,body_characters=len(text),
        input_id=inp.get('input_id'),source_url=inp.get('url'),title=inp.get('title'),input_reference=input_ref,
        raw_sha256=document.get('raw_sha256',inp.get('raw_sha256')),
        published_at=document.get('source_published_at'),updated_at=document.get('source_modified_at'),
        source_response_observed_at=document.get('observed_at'),
        source_available_at=None,source_available_at_basis='first_public_availability_not_inferred',
        projected_available_at=document.get('available_at'),projection_provenance=document.get('projection_provenance'),
        fact_event_at=None,existing_claims=claims,historical_version_unknown=False,
        body_origin='archived_semantic_input_exact_text',original_f_consumption_proven=False)


def extract_learned(path,run):
    semantic=path.parent/'entity-semantic-extractions.json';inputs={};fixture={}
    semantic_error=None
    if semantic.exists():
        try:
            archive=read(semantic)
            declared=obj(run.get('semantic_extraction_archive')).get('sha256')
            if declared and sha(semantic)!=declared:raise ValueError('semantic archive SHA mismatch')
            for ei,event in enumerate(seq(archive.get('events'))):
                for ai,attempt in enumerate(seq(event.get('attempts'))):
                    inp=obj(attempt.get('input'));fixture=fixture or obj(inp.get('fixture'))
                    if inp.get('channel')=='body' and inp.get('input_id'):
                        inputs[inp['input_id']]=(inp,dict(path=str(semantic),event=ei,attempt=ai),attempt)
        except (ValueError,OSError) as exc:semantic_error=str(exc)
    state=obj(run.get('final_state'));base=obj(state.get('base'))
    identity=dict(physical_match_id=fixture.get('physical_match_id',state.get('match_id')),
        source_match_id=state.get('match_id'),home_team=fixture.get('home_name'),away_team=fixture.get('away_name'),
        kickoff=fixture.get('kickoff',base.get('kickoff')),prediction_cutoff=state.get('prediction_cutoff'))
    docs={d.get('url'):d for d in seq(run.get('documents')) if isinstance(d,dict)}
    body_records={}
    for iid,(inp,ref,a) in inputs.items():
        body=body_input_record(inp,docs.get(inp.get('url'),{}),input_ref=ref,claims=obj(a.get('response')).get('claims'))
        if body:body_records[iid]=body
    steps=[];adopted=[]
    for ci,c in enumerate(seq(run.get('cycles'))):
        if not isinstance(c,dict):continue
        action=obj(c.get('selected_action'));pred=obj(obj(c.get('public_after')).get('prediction'))
        used=seq(pred.get('used_semantic_input_ids',obj(pred.get('projection')).get('used_input_ids')))
        ready=obj(c.get('semantic_prediction')).get('status')=='ready' and c.get('public_after') is not None
        newly=[i for i in used if i in body_records and i not in adopted] if ready else []
        consumed_before=list(adopted);adopted.extend(newly)
        for iid in newly:body_records[iid]['original_f_consumption_proven']=True
        after=prob(c.get('prediction_after')) if ready else None
        step=dict(step=ci,action_id=action.get('action_id'),action_url=action.get('url'),attempt_id=c.get('attempt_id'),
            selected_propensity=c.get('selected_selection_probability',obj(c.get('exploration_record')).get('selected_p')),
            request_started_at=c.get('request_started_at'),response_observed_at=c.get('response_observed_at'),
            receipt_available_at=c.get('receipt_available_at'),agent_returned_at=c.get('returned_at'),
            f_available_at=obj(c.get('semantic_prediction')).get('available_at'),execution_status=c.get('execution_status'),
            semantic_status=obj(c.get('semantic_prediction')).get('status'),semantic_failure_reason=obj(c.get('semantic_prediction')).get('reason'),
            old_prediction_before=prob(c.get('prediction_before')),old_prediction_after=after,
            stale_saved_after_rejected=after is None and c.get('prediction_after') is not None,
            original_f_used_input_ids=used,adopted_body_input_ids_before=consumed_before,
            newly_adopted_body_input_ids=newly,adopted_body_input_ids_after=list(adopted),
            menu_candidate_count=len(seq(c.get('menu'))),selected_action_present=bool(action),
            old_gain=None,new_f_gain=None,input_reference=dict(path=str(path),cycle=ci))
        steps.append(step)
    prior=prob(run.get('initial_probabilities'))
    market=path.parent/'market-prior.json'
    if market.exists():prior=prob(obj(read(market)).get('probabilities')) or prior
    return identity,prior,steps,list(body_records.values()),semantic_error


def extract_live(path,run):
    f=obj(run.get('fixture'));m=obj(run.get('match'))
    identity=dict(physical_match_id=f.get('physical_match_id',m.get('physical_match_id')),source_match_id=m.get('match_id'),
        home_team=f.get('home_name'),away_team=f.get('away_name'),kickoff=f.get('kickoff'),prediction_cutoff=run.get('prediction_cutoff'))
    documents={d.get('url'):d for d in seq(run.get('documents')) if isinstance(d,dict)}
    steps=[];bodies=[];adopted=[]
    for i,s in enumerate(seq(run.get('steps'))):
        if not isinstance(s,dict):continue
        d=documents.get(s.get('url'),{});inp=dict(s,text=s.get('body'))
        b=body_input_record(inp,d,input_ref=dict(path=str(path),step=i),claims=s.get('claims'))
        proven=s.get('f_executed') is True and bool(s.get('input_id')) and prob(s.get('prediction_after')) is not None
        old=list(adopted)
        if b:
            b['original_f_consumption_proven']=proven;bodies.append(b)
            if proven and b['input_id'] not in adopted:adopted.append(b['input_id'])
        attempts=[a for a in seq(run.get('attempts')) if obj(a.get('action')).get('url')==s.get('url')]
        a=attempts[-1] if attempts else {}
        steps.append(dict(step=i,action_id=s.get('action_id'),action_url=s.get('url'),attempt_id=a.get('request_id'),
            selected_propensity=s.get('propensity'),request_started_at=a.get('requested_at'),
            response_observed_at=obj(obj(a.get('response')).get('document')).get('observed_at'),
            receipt_available_at=a.get('handoff_at'),agent_returned_at=a.get('handoff_at'),f_available_at=s.get('executed_at'),
            execution_status='OK' if proven else 'UNCONFIRMED',semantic_status='ready' if proven else None,
            old_prediction_before=prob(s.get('prediction_before')),old_prediction_after=prob(s.get('prediction_after')) if proven else None,
            original_f_used_input_ids=[s.get('input_id')] if proven else [],adopted_body_input_ids_before=old,
            newly_adopted_body_input_ids=[s.get('input_id')] if proven and s.get('input_id') not in old else [],
            adopted_body_input_ids_after=list(adopted),old_gain=None,new_f_gain=None,input_reference=dict(path=str(path),step=i)))
    # Failed acquisition/provider attempts remain in a separate whitelist ledger,
    # never converted into f update steps or zero labels.
    return identity,prob(run.get('prior')),steps,bodies,None


def extract_control(path,run):
    f=run['fixture'];source=run['source'];sem=run['semantic'];pred=run['prediction']
    identity=dict(physical_match_id=f.get('physical_match_id'),source_match_id=f.get('match_id'),home_team=f.get('home_name'),away_team=f.get('away_name'),kickoff=f.get('kickoff'),prediction_cutoff=obj(run.get('identity')).get('prediction_cutoff'))
    p=path.parent/sem['records_file'];records=read(p);bodies=[]
    for i,a in enumerate(seq(records)):
        inp=obj(a.get('input'))
        if inp.get('channel')!='body':continue
        body=body_input_record(inp,dict(raw_sha256=source.get('body_raw_sha256')),input_ref=dict(path=str(p),index=i),claims=obj(a.get('response')).get('claims'))
        if not body:continue
        if source.get('body_text_sha256')!=body['text_sha256']:raise ValueError('F-control exact body SHA mismatch')
        body['original_f_consumption_proven']=sem.get('body_consumed') is True and body['input_id'] in seq(sem.get('used_input_ids'))
        body['archive_materialization_at']=source.get('fotmob_archive_materialization_at');bodies.append(body)
    adopted=[b['input_id'] for b in bodies if b['original_f_consumption_proven']]
    step=dict(step=0,action_id=None,action_url=source.get('body_url'),selected_propensity=None,actual_agent_selection=False,
        request_started_at=None,response_observed_at=None,receipt_available_at=None,agent_returned_at=None,
        f_available_at=run.get('generated_at'),execution_status='recorded_direct_f_control',semantic_status='ready' if adopted else None,
        old_prediction_before=prob(pred.get('prediction_before')),old_prediction_after=prob(pred.get('prediction_after')) if adopted else None,
        original_f_used_input_ids=sem.get('used_input_ids'),adopted_body_input_ids_before=[],newly_adopted_body_input_ids=adopted,
        adopted_body_input_ids_after=adopted,old_gain=None,new_f_gain=None,input_reference=dict(path=str(path),field='prediction'))
    return identity,prob(obj(run.get('prior')).get('probabilities')),[step],bodies,None


def recover(args):
    if args.output.exists():raise FileExistsError('Independent recovery output already exists')
    paths=files([ROOT/'var',Path('E:/FootballMind')]);inventory=[];candidates=defaultdict(list);settlements={};results=defaultdict(list);control_settlements={}
    matches=lines(args.data/'matches.jsonl');aliases={};match_by={r['physical_match_id']:r for r in matches}
    for r in matches:
        for alias in [r['physical_match_id']]+seq(r.get('aliases')):
            if alias in aliases and aliases[alias]!=r['physical_match_id']:raise ValueError('Conflicting explicit match alias')
            aliases[alias]=r['physical_match_id']
    known_bodies={r['text_sha256']:r['document_id'] for r in lines(args.data/'documents.jsonl')}
    for r in lines(args.data/'outcomes.jsonl'):
        if r.get('outcome_class') is not None:results[r['physical_match_id']].append(dict(outcome_class=r['outcome_class'],evidence=r.get('source_refs'),basis='frozen_data_independent_outcome',path=str(args.data/'outcomes.jsonl')))
    for path in paths:
        scope=path_scope(path);entry=dict(path=str(path),scope=scope,bytes=path.stat().st_size)
        if path.suffix=='.zip':
            entry['record_type']='package_metadata_only_not_new_run'
            try:
                with zipfile.ZipFile(path) as archive:
                    members=archive.infolist();entry['entries']=len(members)
                    entry['relevant_entries']=[dict(name=x.filename,bytes=x.file_size) for x in members if Path(x.filename).name in ('run.json','collection-report.json','runtime-trace.json','shadow-trace.json','manifest.json')]
                    projected=[]
                    for x in members:
                        if Path(x.filename).name not in ('run.json','collection-report.json','runtime-trace.json','shadow-trace.json') or x.file_size>200000:continue
                        try:
                            rr=json.loads(archive.read(x).decode('utf-8-sig'))
                            if isinstance(rr,dict):projected.append(dict(member=x.filename,schema=rr.get('schema',rr.get('schema_version')),run_id=rr.get('run_id'),physical_match_id=rr.get('physical_match_id')))
                        except (ValueError,UnicodeError):pass
                    entry['identity_projection']=projected
            except (zipfile.BadZipFile,OSError) as exc:entry['error']=str(exc)
            inventory.append(entry);continue
        if path.suffix=='.jsonl':
            entry['record_type']='canonical_stream_or_dataset';inventory.append(entry)
            if path.name=='results.jsonl' and scope!='test_or_constructed_excluded':
                try:
                    for r in lines(path):
                        if r.get('independent_outcome_verified') is True and r.get('result_status') in ('finished','final','full_time'):
                            h=r.get('regulation_home_goals');a=r.get('regulation_away_goals');mid=aliases.get(r.get('physical_match_id'),r.get('physical_match_id'))
                            if isinstance(h,int) and isinstance(a,int) and mid:results[mid].append(dict(outcome_class=0 if h>a else 1 if h==a else 2,evidence=r,path=str(path),basis='archived_independent_result'))
                except (ValueError,OSError) as exc:entry['error']=str(exc)
            continue
        if scope=='test_or_constructed_excluded':entry['record_type']='excluded_test_fixture';inventory.append(entry);continue
        try:r=read(path)
        except (ValueError,OSError) as exc:entry['error']=str(exc);inventory.append(entry);continue
        if not isinstance(r,dict):entry['record_type']='non_object';inventory.append(entry);continue
        schema=r.get('schema',r.get('schema_version'));entry['schema']=schema;entry['run_id']=r.get('run_id');entry['sha256']=sha(path);inventory.append(entry)
        if schema=='footballmind-postmatch-settlement-v1' and r.get('settlement_status')=='verified':settlements[r['run_id']]=(path,r)
        if schema=='footballmind-f-control-settlement-status-v1' and r.get('status')=='settled_f_control' and r.get('frozen_files_unchanged') is True:
            y={'home_win':0,'draw':1,'away_win':2}.get(obj(r.get('settlement')).get('outcome'))
            if y is not None:control_settlements[r['frozen_prediction_sha256']]=dict(outcome_class=y,evidence=r.get('observations'),path=str(path),basis='exact_frozen_prediction_SHA_bound_independent_settlement')
        if isinstance(schema,str) and schema.startswith('footballmind-quick-cross-league-f-control-'):
            entry['scope']='recorded_f_control';rid='artifact-sha256:'+entry['sha256'];candidates[rid].append((path,r,entry))
        if isinstance(schema,str) and (schema.startswith('learned-voi-open-execution-') or schema.startswith('footballmind-live-g-experiment-')):
            runid=r.get('run_id')
            if runid:candidates[runid].append((path,r,entry))
    args.output.mkdir(parents=True);(args.output/'bodies').mkdir()
    runs=[];all_steps=[];all_bodies={};replays=[];failures=[];attempt_ledger=[];copies=[]
    for runid,versions in sorted(candidates.items()):
        def rank(item):
            p,r,e=item
            return(e['scope']=='recorded_execution',len(seq(r.get('steps')))+len(seq(r.get('cycles'))),(p.parent/'entity-semantic-extractions.json').exists(),len(str(r.get('documents'))),str(p))
        path,r,entry=max(versions,key=rank);schema=entry['schema'];kind='live' if schema.startswith('footballmind-live') else 'control' if schema.startswith('footballmind-quick') else 'learned'
        copies.append(dict(run_id=runid,chosen=str(path),variants=[dict(path=str(p),sha256=e['sha256'],scope=e['scope']) for p,_,e in versions],same_run_not_independent_matches=True))
        try:identity,prior,steps,bodies,error={'live':extract_live,'learned':extract_learned,'control':extract_control}[kind](path,r)
        except Exception as exc:failures.append(dict(run_id=runid,path=str(path),error_type=type(exc).__name__,reason=str(exc)));continue
        mid=aliases.get(identity.get('physical_match_id'),aliases.get(identity.get('source_match_id'),identity.get('physical_match_id')))
        identity['resolved_physical_match_id']=mid;identity['in_frozen_primary_match_registry']=mid in match_by
        binding=model_binding(path,r);settled=control_settlements.get(entry['sha256']) if kind=='control' else None
        if runid in settlements:
            sp,sr=settlements[runid];rr=sr['result'];y=rr.get('outcome_index')
            if y in (0,1,2) and rr.get('independent_verification'):settled=dict(outcome_class=y,evidence=rr['independent_verification'],path=str(sp),basis='run_bound_verified_settlement')
        if settled is None and mid in results:
            yy={v['outcome_class'] for v in results[mid]}
            if len(yy)==1:settled=dict(outcome_class=next(iter(yy)),evidence=results[mid],basis='matched_exact_identity_result')
            else:failures.append(dict(run_id=runid,physical_match_id=mid,reason='independent_result_conflict',outcome_class=None))
        bmap={b['input_id']:b for b in bodies};adopted=[]
        for s in steps:
            s.update(run_id=runid,physical_match_id=mid,original_f_binding=binding,scope=entry['scope'])
            s['request_before_kickoff']=before(s.get('request_started_at'),identity.get('kickoff'))
            s['f_available_before_kickoff']=before(s.get('f_available_at'),identity.get('kickoff'))
            s['newly_adopted_document_ids']=[bmap[i]['document_id'] for i in s['newly_adopted_body_input_ids']]
            adopted=s['adopted_body_input_ids_after'];all_steps.append(s)
        for b in bodies:
            b.update(run_id=runid,physical_match_id=mid,original_f_binding=binding,in_primary_body_library=b['text_sha256'] in known_bodies)
            text=b.pop('text');target=args.output/'bodies'/(b['text_sha256']+'.txt')
            if not target.exists():target.write_text(text,encoding='utf8',newline='')
            b['body_path']=str(target);all_bodies[(runid,b['input_id'])]=b
        for i,a in enumerate(seq(r.get('attempts'))):
            if not isinstance(a,dict):continue
            action=obj(a.get('action'));response=obj(a.get('response'))
            attempt_ledger.append(dict(run_id=runid,physical_match_id=mid,index=i,attempt_id=a.get('attempt_id',a.get('request_id')),
                action_id=a.get('action_id',action.get('action_id')),tool=action.get('tool'),url=action.get('url'),purpose=a.get('purpose'),
                requested_at=a.get('requested_at',a.get('request_started_at')),handoff_at=a.get('handoff_at'),status=a.get('status',response.get('status')),
                input_reference=dict(path=str(path),attempt=i),not_a_gain_label=True))
        exclusions=[]
        if entry['scope'] not in ('recorded_execution','recorded_f_control'):exclusions.append('replay_or_development_scope')
        if not prior:exclusions.append('actual_prior_missing')
        if not mid:exclusions.append('match_identity_missing')
        if not adopted:exclusions.append('no_proven_full_body_f_consumption')
        if error:exclusions.append('semantic_archive_integrity_error')
        input_ready=not exclusions
        input_sequence=[dict(document_id=bmap[i]['document_id'],text_sha256=bmap[i]['text_sha256'],body_path=bmap[i]['body_path'],
            original_input_id=i,source_url=bmap[i]['source_url'],published_at=bmap[i]['published_at'],source_available_at=None,
            source_response_observed_at=bmap[i]['source_response_observed_at'],fact_event_at=None,existing_claims=bmap[i]['existing_claims'],
            historical_version_unknown=False) for i in adopted]
        beforekickoff=all(s['f_available_before_kickoff'] is True for s in steps if s['newly_adopted_body_input_ids']) if adopted else False
        replay=dict(run_id=runid,physical_match_id=mid,identity=identity,prior=prior,documents=input_sequence,
            actual_adoption_order=[bmap[i]['document_id'] for i in adopted],order_basis='recorded_f_consumption_steps_not_publication_or_fact_event_order',
            new_f_input_replayable=input_ready,new_f_labeled_diagnostic_eligible=input_ready and settled is not None,
            empirical_new_g_training_eligible=False,new_f_executed=False,new_f_gain=None,old_gain_reused=False,
            original_f_binding=binding,independent_outcome=settled,exclusion_reasons=exclusions,
            all_adoption_clocks_before_kickoff=beforekickoff,stratum='recorded_direct_f_control_diagnostic' if kind=='control' else 'recorded_agent_trajectory_frozen_model_replay_diagnostic',
            source_path=str(path),source_sha256=entry['sha256'])
        replays.append(replay)
        runs.append(dict(run_id=runid,scope=entry['scope'],schema=schema,path=str(path),physical_match_id=mid,identity=identity,
            duplicate_artifact_copies=len(versions)-1,source_steps=len(steps),source_attempts=len(seq(r.get('attempts'))),
            body_inputs=len(bodies),consumed_body_inputs=len(adopted),new_f_input_replayable=input_ready,
            independently_settled=settled is not None,new_f_labeled_diagnostic_eligible=replay['new_f_labeled_diagnostic_eligible'],
            created_at=r.get('created_at'),finished_at=r.get('finished_at',r.get('updated_at')),semantic_archive_error=error,
            original_f_binding=binding,exclusion_reasons=exclusions,original_outcome_not_used_as_feature=True))
    jsonl(args.output/'source-inventory.jsonl',inventory);jsonl(args.output/'runs.jsonl',runs);jsonl(args.output/'steps.jsonl',all_steps)
    jsonl(args.output/'body-inputs.jsonl',list(all_bodies.values()));jsonl(args.output/'acquisition-attempts.jsonl',attempt_ledger)
    jsonl(args.output/'new-f-replay-inputs.jsonl',replays);jsonl(args.output/'artifact-copies.jsonl',copies);jsonl(args.output/'failures.jsonl',failures)
    real=[r for r in runs if r['scope']=='recorded_execution'];ready=[r for r in replays if r['new_f_input_replayable']];labelled=[r for r in ready if r['new_f_labeled_diagnostic_eligible']]
    summary=dict(schema='recorded-agent-trajectories-recovery-v1',scanned_files=len(inventory),run_groups=len(runs),recorded_execution_runs=len(real),recorded_direct_f_control_runs=sum(r['scope']=='recorded_f_control' for r in runs),
        recorded_independent_matches=len({r['physical_match_id'] for r in real if r['physical_match_id']}),
        body_input_rows=len(all_bodies),unique_texts=len({b['text_sha256'] for b in all_bodies.values()}),
        replayable_runs=len(ready),replayable_matches=len({r['physical_match_id'] for r in ready}),
        independently_settled_replayable_runs=len(labelled),independently_settled_replayable_matches=len({r['physical_match_id'] for r in labelled}),
        primary_library_missing_adopted_text_sha256=sorted({d['text_sha256'] for r in ready for d in r['documents'] if d['text_sha256'] not in known_bodies}),
        stale_after_arrays_rejected=sum(s.get('stale_saved_after_rejected',False) for s in all_steps),
        exclusions=dict(Counter(x for r in runs for x in r['exclusion_reasons'])),parse_failures=len(failures),
        no_http=True,no_model_training=True,no_f_or_g_execution=True,source_files_modified=False,
        limits=['Archived provider inputs and f-used IDs prove recorded software consumption, not source factual truth.',
            'First public availability and fact-event chronology remain unknown unless explicitly recorded; receipt clocks are separate.',
            'Replay/development paths retained separately; original model outputs/gains are not new-f supervision.',
            'Canonical streams/journal summaries are inventoried; replayable text requires exact body input and actual consumption.',
            'Distinct run artifacts are deduplicated by run_id; multiple runs of one physical match do not increase match count.'])
    save(args.output/'summary.json',summary)
    save(args.output/'manifest.json',{str(p.relative_to(args.output)):sha(p) for p in sorted(args.output.rglob('*')) if p.is_file()})
    print(json.dumps(summary,ensure_ascii=True),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=BASE/'data');p.add_argument('--output',type=Path,default=BASE/'recorded-trajectories')
    recover(p.parse_args())

if __name__=='__main__':main()
