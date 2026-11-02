"""Build label-free explicit state/candidate context from existing local records.

No labels/results file is opened, no f/g or encoder inference is called, and no
old feature/vector/model file is written. Missing historical clocks remain null.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import socket
import sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from football_research.acquisition.entity_semantic_f import _modeled, normalize_entity, predicate_key
from football_research.acquisition.semantic_f import public_input, validate_response, EXTRACTION_VERSION

DATA=Path('E:/FootballMind/experiments/postread-clean-expansion-20261025-v1')
EMB=Path('E:/FootballMind/experiments/postread-qwen-semantic-20261025-v1')
DEFAULT_OUT=Path('E:/FootballMind/experiments/postread-semantic-training-ready-20261025-v1/context-sidecar')
FORBIDDEN={'outcome','outcome_class','outcome_index','p_after','prediction_after','f_prediction_after',
           'realized_gain','signed_gain','gain','loss_before','loss_after','before_loss','after_loss','y_true'}


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def text_sha(t): return hashlib.sha256(t.encode('utf-8')).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p): return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').split('\n') if s.strip()]
def save(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def save_rows(p,rr):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:
        for r in rr:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def no_network(*args,**kwargs):raise RuntimeError('This sidecar builder is local-only')
def keys(claims):return {predicate_key(c) for c in claims if _modeled(c)}
def key_record(k):return dict(team_side=k[0],entity_normalized=k[1],kind=k[2],value=k[3])
def status_sets(ks):
    result=defaultdict(set)
    for team,entity,kind,value in ks:
        if kind=='availability':result[team,entity].add(value)
    return result
def multi_records(sets):
    return [dict(team_side=t,entity_normalized=e,statuses=sorted(v)) for (t,e),v in sorted(sets.items()) if len(v)>1]
def forbidden_paths(value,prefix=''):
    result=[]
    if isinstance(value,dict):
        for k,v in value.items():
            path=prefix+'.'+k if prefix else k
            if k in FORBIDDEN:result.append(path)
            result+=forbidden_paths(v,path)
    elif isinstance(value,list):
        for i,v in enumerate(value):result+=forbidden_paths(v,f'{prefix}[{i}]')
    return result


def quote_offsets(text,quote):
    offsets=[];start=0
    while True:
        start=text.find(quote,start)
        if start<0:break
        offsets.append(dict(start=start,end=start+len(quote)));start+=1
    return offsets


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=DEFAULT_OUT);args=parser.parse_args();out=args.output
    if out.exists() and any(out.iterdir()):raise FileExistsError('Independent empty output directory required')
    socket.socket.connect=socket.create_connection=socket.getaddrinfo=no_network
    out.mkdir(parents=True,exist_ok=True)
    bank_path=ROOT/'var/f-semantic-residual-20261010-v1/assistant-records.json'
    correction_path=Path('E:/FootballMind/experiments/unified-f-postread-g-20261025-v1/review-corrections.jsonl')
    review_paths=[Path('E:/FootballMind/experiments/unified-f-postread-g-expansion-20261025-v1/semantic-reviews.jsonl'),DATA/'semantic-reviews.jsonl']
    fixtures_path=ROOT/'footballmind-information-stream/fixtures.jsonl'
    matches_path=Path('E:/FootballMind/corpus-v3/structured/matches.jsonl')
    source_paths=[DATA/n for n in ('features.jsonl','samples.jsonl','documents.jsonl')]+[bank_path,correction_path,*review_paths,fixtures_path,matches_path,EMB/'embedding-report.json',EMB/'embeddings.npz']
    source_hashes={str(p):sha(p) for p in source_paths}
    features=rows(DATA/'features.jsonl');samples=rows(DATA/'samples.jsonl');documents=rows(DATA/'documents.jsonl')
    sample_by={r['sample_id']:r for r in samples};feature_by={r['sample_id']:r for r in features}
    if len(sample_by)!=len(samples) or len(feature_by)!=len(features) or set(sample_by)!=set(feature_by):raise ValueError('Sample/feature IDs differ')
    docs={(d['physical_match_id'],d['candidate_index']):d for d in documents}
    by_semantic_id={d['semantic_input_id']:d for d in documents}
    mids={r['physical_match_id'] for r in features}
    fixtures={r['physical_match_id']:r for r in rows(fixtures_path) if r['physical_match_id'] in mids}
    matches={r['match_id']:r for r in rows(matches_path) if r['match_id'] in mids}
    records={r['input']['input_id']:r for r in read(bank_path)}
    record_sources={k:str(bank_path) for k in records}
    condition_annotations={}
    for path in review_paths:
        for r in rows(path):
            sid=r['input']['input_id'];records[sid]=r;record_sources[sid]=str(path)
            if r.get('claim_uncertainty'):condition_annotations[sid]={c['claim_index']:c for c in r['claim_uncertainty']}
    corrections={r['semantic_input_id']:r for r in rows(correction_path) if r.get('semantic_input_id')}
    for d in documents:
        sid=d['semantic_input_id']
        if sid in records:continue
        c=corrections[sid];match=matches[d['physical_match_id']]
        body_path=Path(c['body_path']);source_hashes[str(body_path)]=sha(body_path)
        text=body_path.read_bytes().decode('utf-8')
        if sha(body_path)!=c['body_sha256'] or text!=d['text']:raise ValueError('Correction body binding differs')
        fixture=dict(physical_match_id=d['physical_match_id'],home_name=match['home_team'],away_name=match['away_team'],kickoff=c['semantic_fixture_kickoff'])
        public=public_input(fixture,dict(url=d['source_url'],title=d['title'],text=text),'body')
        if public['input_id']!=sid:raise ValueError('Reconstructed label-free semantic input differs from saved ID')
        records[sid]=dict(input=public,response=dict(schema=EXTRACTION_VERSION,input_id=sid,claims=c['claims']))
        record_sources[sid]=str(correction_path)
        condition_annotations[sid]={v['claim_index']:v for v in c.get('claim_uncertainty',[])}
    used_semantic_ids={d['semantic_input_id'] for d in documents}|{k for s in samples for k in s['known_input_ids']}
    for sid in used_semantic_ids:validate_response(records[sid]['input'],records[sid]['response'])
    embedding_report=read(EMB/'embedding-report.json')
    if embedding_report['source_sha256']!=sha(DATA/'features.jsonl') or embedding_report['vectors_file_sha256']!=sha(EMB/'embeddings.npz'):raise ValueError('Frozen vector binding differs')
    vector_keys={r['text_sha256'] for r in embedding_report['records']}
    target_conflicts=[]
    targets={}
    for mid in sorted(mids):
        fixture=fixtures.get(mid);match=matches.get(mid)
        source=fixture or match
        if source is None:raise ValueError('No saved structured fixture: '+mid)
        source_path=fixtures_path if fixture else matches_path
        target=dict(physical_match_id=mid,home_name=source.get('home_team'),away_name=source.get('away_team'),
            competition=source.get('competition'),division=source.get('division'),season=source.get('season'),
            kickoff=source.get('kickoff'),kickoff_local_time=source.get('kickoff_local_time'),
            kickoff_timezone=source.get('kickoff_timezone'),match_date=source.get('match_date'),
            identity_source=dict(path=str(source_path),record_key=mid),
            clock_policy='Preserve stored UTC/local fields; no timezone or kickoff inference')
        if target['match_date'] is None and target['kickoff']:
            target['match_date']=target['kickoff'][:10]
            target['match_date_basis']='date component of stored kickoff; not inferred local calendar'
        else:target['match_date_basis']='stored structured match_date'
        # The model's fixture names can have explicit source spelling differences.
        semantic_fixtures=[]
        for d in documents:
            if d['physical_match_id']!=mid:continue
            sf=records[d['semantic_input_id']]['input']['fixture']
            if sf['physical_match_id']!=mid:raise ValueError('Semantic fixture belongs to another match')
            item={k:sf[k] for k in ('home_name','away_name','kickoff')}
            if item not in semantic_fixtures:semantic_fixtures.append(item)
        target['semantic_fixture_assertions']=semantic_fixtures
        for item in semantic_fixtures:
            if item['home_name']!=target['home_name'] or item['away_name']!=target['away_name']:
                target_conflicts.append(dict(physical_match_id=mid,structured_names=[target['home_name'],target['away_name']],semantic_names=[item['home_name'],item['away_name']],automatic_resolution=False))
        targets[mid]=target
    quote_count=0;condition_count=0
    def evidence(sid,mid):
        nonlocal quote_count,condition_count
        r=records[sid];public=r['input'];body=public['text'];d=by_semantic_id.get(sid)
        if public['fixture']['physical_match_id']!=mid:raise ValueError('Cross-match evidence')
        if d and (text_sha(body)!=d['text_sha256'] or d['text']!=body):raise ValueError('Body SHA differs')
        assertions=[]
        for i,c in enumerate(r['response']['claims']):
            quote_rows=[]
            for q in c['quotes']:
                spans=quote_offsets(body,q)
                if not spans:raise ValueError('Quote not found in source text')
                quote_rows.append(dict(text=q,unicode_codepoint_spans=spans));quote_count+=1
            annotation=condition_annotations.get(sid,{}).get(i)
            # Preserve only the explicitly stored qualifier, never parse a new condition.
            qualifier=annotation.get('qualifier') if annotation else None
            if qualifier is not None:condition_count+=1
            assertions.append(dict(claim_index=i,team_side=c['team'],entity_name=c['entity'],entity_normalized=normalize_entity(c['entity']),
                team_name_by_target_role=targets[mid]['home_name'] if c['team']=='home' else targets[mid]['away_name'] if c['team']=='away' else None,
                kind=c['kind'],value=c['value'],scope=c['scope'],statement=c['statement'],quotes=quote_rows,
                explicit_condition_annotation=qualifier,condition_annotation_missing=qualifier is None,
                temporal_scope_annotation=None,temporal_scope_annotation_missing=True,
                supported_by_frozen_f_schema=bool(_modeled(c)),source_truth_independently_verified=None))
        return dict(semantic_input_id=sid,channel=public['channel'],candidate_text_id=d.get('candidate_text_id') if d else None,
            source_url=public['url'],source_domain=d.get('source_domain') if d else None,title=public['title'],
            text_sha256=text_sha(body),text_reference=dict(path=str(DATA/'documents.jsonl'),record_key=[mid,d['candidate_index']]) if d else dict(path=record_sources[sid],semantic_input_id=sid,field='input.text'),
            source_published_at=d.get('source_published_at') if d else None,source_available_at=d.get('source_available_at') if d else None,
            captured_at=d.get('captured_at') if d else None,historical_version_unknown=d.get('historical_version_unknown') if d else None,
            semantic_record_reference=dict(path=record_sources[sid],semantic_input_id=sid),assertions=assertions)
    output=[];relation_counts=Counter();checks=[]
    for f in features:
        sid=f['sample_id'];s=sample_by[sid];mid=s['physical_match_id'];d=docs[mid,s['candidate_index']]
        cm=d['semantic_input_id'];am=s['acquired_mask'];dm=s['discovered_mask']
        if f['candidate_text']!=d['text'] or not(dm&(1<<s['candidate_index'])) or am&(1<<s['candidate_index']):raise ValueError('Invalid candidate text/mask')
        adopted_ids=[k for k in s['known_input_ids'] if records[k]['input']['channel']=='body']
        title_ids=[k for k in s['known_input_ids'] if records[k]['input']['channel']=='title']
        expected_ids={docs[mid,j]['semantic_input_id'] for j in range(s['candidate_count']) if am&(1<<j)}
        if expected_ids!=set(adopted_ids):raise ValueError('Adopted set differs from mask')
        adopted_text='\n\n'.join(records[k]['input']['text'] for k in adopted_ids)
        if adopted_text!=f['adopted_text']:raise ValueError('Adopted text serialization differs')
        candidate_sha=text_sha(f['candidate_text']);adopted_sha=text_sha(adopted_text) if adopted_text else None
        if candidate_sha not in vector_keys or (adopted_sha and adopted_sha not in vector_keys):raise ValueError('Frozen vector missing')
        known_claims=[c for k in s['known_input_ids'] for c in records[k]['response']['claims']]
        candidate_claims=records[cm]['response']['claims'];kk=keys(known_claims);ck=keys(candidate_claims)
        before_status=status_sets(kk);candidate_status=status_sets(ck);combined_status=status_sets(kk|ck)
        existing=multi_records(before_status);combined=multi_records(combined_status)
        introduced=[r for r in combined if len(before_status.get((r['team_side'],r['entity_normalized']),set()))<=1]
        expanded=[r for r in combined if len(before_status.get((r['team_side'],r['entity_normalized']),set()))>1 and set(r['statuses'])!=before_status[r['team_side'],r['entity_normalized']]]
        repeated=kk&ck;novel=ck-kk
        clocks=s.get('clocks',s)
        row=dict(schema='postread-semantic-context-sidecar-v1',sample_id=sid,physical_match_id=mid,split=f['split'],target=targets[mid],
            prediction_context=dict(current_probabilities=f['state_numeric'][:3],entropy_nats=f['state_numeric'][3],top_two_margin=f['state_numeric'][4],
                prediction_cutoff=clocks.get('prediction_cutoff'),hours_to_kickoff=None if f['state_numeric'][8] else f['state_numeric'][7],
                hours_to_kickoff_unknown=bool(f['state_numeric'][8]),adopted_body_count=len(adopted_ids),remaining_candidate_count=dm.bit_count()-am.bit_count(),
                remaining_adoption_budget=None,remaining_adoption_budget_basis='not_specified_in_saved_sample',
                source_state_feature_reference=dict(path=str(DATA/'features.jsonl'),sample_id=sid)),
            visibility=dict(g_has_read_candidate=bool(s['g_has_read_candidate']),f_has_adopted_candidate=False,
                discovered_mask=dm,acquired_mask=am,candidate_index=s['candidate_index'],
                discovered_body_semantic_ids=[docs[mid,j]['semantic_input_id'] for j in range(s['candidate_count']) if dm&(1<<j)],
                known_g_seen_semantic_ids=s.get('g_seen_input_ids'),complete_g_seen_registry_available=s.get('g_seen_input_ids') is not None,
                f_adopted_body_semantic_ids=adopted_ids,known_title_semantic_ids=title_ids,
                unseen_pool_not_inferred_from_embedding_cache=True),
            candidate=evidence(cm,mid),adopted_evidence=[evidence(k,mid) for k in adopted_ids],known_title_evidence=[evidence(k,mid) for k in title_ids],
            relations=dict(repeated_modeled_predicates=[key_record(k) for k in sorted(repeated)],novel_modeled_predicates=[key_record(k) for k in sorted(novel)],
                repeated_modeled_predicate_count=len(repeated),exact_body_duplicate_with_adopted=any(text_sha(records[k]['input']['text'])==candidate_sha for k in adopted_ids),
                existing_multistatus_entities=existing,existing_multistatus_count=len(existing),
                candidate_only_multistatus_entities=multi_records(candidate_status),combined_multistatus_entities=combined,combined_multistatus_count=len(combined),
                candidate_introduced_multistatus_entities=introduced,candidate_introduced_multistatus_count=len(introduced),
                candidate_expanded_existing_multistatus_entities=expanded,candidate_expanded_existing_multistatus_count=len(expanded),
                multistatus_is_not_logical_contradiction_or_truth_judgment=True),
            frozen_vector_bindings=dict(repository=embedding_report['contract']['repository'],revision=embedding_report['contract']['revision'],
                embeddings_path=str(EMB/'embeddings.npz'),candidate_vector_key=candidate_sha,adopted_vector_key=adopted_sha,
                empty_adopted_evidence_vector='zero' if adopted_sha is None else None,
                canonical_adopted_serialization='known_input_ids body order joined by two LF characters',new_context_text_encoded=False),
            research_provenance=dict(historical_reconstruction=True,source_available_at_unknown_not_an_automatic_research_exclusion=True,
                no_new_f_predictions=True,no_new_embedding_inference=True,condition_annotations_inherited_or_unknown=True))
        if forbidden_paths(row):raise ValueError('Forbidden label field exported: '+str(forbidden_paths(row)))
        if row['candidate']['candidate_text_id'] is None:row['candidate']['candidate_text_id']=s['candidate_text_id']
        output.append(row)
        relation_counts['existing_multistatus_rows']+=bool(existing);relation_counts['candidate_introduced_multistatus_rows']+=bool(introduced)
        relation_counts['candidate_expanded_multistatus_rows']+=bool(expanded);relation_counts['combined_multistatus_rows']+=bool(combined)
        relation_counts['combined_without_candidate_new_or_expanded_rows']+=bool(combined) and not introduced and not expanded
        checks.append(dict(sample_id=sid,candidate_body_sha_ok=True,adopted_mask_ids_ok=True,adopted_serialization_ok=True,
            vector_keys_present=True,label_fields_absent=True,budget_is_null=True))
    source_unchanged=all(sha(Path(p))==h for p,h in source_hashes.items())
    if not source_unchanged:raise ValueError('Input changed during sidecar construction')
    save_rows(out/'state-candidate-context.jsonl',output)
    save(out/'target-identities.json',targets)
    save(out/'identity-spelling-differences.json',target_conflicts)
    save(out/'schema.json',dict(schema='postread-semantic-context-sidecar-v1',type='object',
        required=list(output[0]),join_key='sample_id',physical_group='physical_match_id',
        semantics={'target':'Explicit saved fixture/match metadata; no body/URL inference',
            'prediction_context':'Current probability/state and saved cutoff; no candidate-after prediction',
            'candidate':'Already g-read body and source assertions',
            'adopted_evidence':'Only f-adopted bodies in exact original serialization order',
            'known_title_evidence':'Already observed title assertions, distinct from body adoption',
            'relations':'Exact predicate repeat/novel and availability multi-status sets; no truth arbitration',
            'frozen_vector_bindings':'References unchanged full-text/adopted vectors; no new context embedding'},
        forbidden_label_keys=sorted(FORBIDDEN),nullable_fields=['target.kickoff','target.kickoff_timezone','prediction_context.prediction_cutoff',
            'prediction_context.remaining_adoption_budget','candidate.source_available_at','candidate.assertions[].explicit_condition_annotation',
            'candidate.assertions[].temporal_scope_annotation'],
        remaining_candidate_count_is_not_budget=True,new_model_training_performed=False))
    candidate_claims=[c for r in output for c in r['candidate']['assertions']]
    summary=dict(status='ready_label_free_context_sidecar',sample_rows=len(output),matches=len(mids),documents=len(documents),
        target_home_away_known_matches=sum(bool(t['home_name'] and t['away_name']) for t in targets.values()),
        target_season_known_matches=sum(t['season'] is not None for t in targets.values()),
        target_competition_name_known_matches=sum(t['competition'] is not None for t in targets.values()),
        target_division_code_known_matches=sum(t['division'] is not None for t in targets.values()),
        target_kickoff_known_matches=sum(t['kickoff'] is not None for t in targets.values()),
        target_local_time_known_matches=sum(t['kickoff_local_time'] is not None for t in targets.values()),
        prediction_cutoff_known_rows=sum(r['prediction_context']['prediction_cutoff'] is not None for r in output),
        candidate_source_available_at_known_rows=sum(r['candidate']['source_available_at'] is not None for r in output),
        candidate_published_at_known_rows=sum(r['candidate']['source_published_at'] is not None for r in output),
        candidate_captured_at_known_rows=sum(r['candidate']['captured_at'] is not None for r in output),
        actual_remaining_budget_known_rows=0,candidate_claim_occurrences=len(candidate_claims),
        candidate_claim_explicit_condition_occurrences=sum(c['explicit_condition_annotation'] is not None for c in candidate_claims),
        candidate_claim_condition_unknown_occurrences=sum(c['explicit_condition_annotation'] is None for c in candidate_claims),
        candidate_claim_temporal_scope_unknown_occurrences=len(candidate_claims),
        evidence_quote_occurrences_checked=quote_count,all_claim_quote_offsets_valid=True,
        complete_g_seen_registry_rows=sum(r['visibility']['complete_g_seen_registry_available'] for r in output),
        identity_spelling_differences=len(target_conflicts),relations=dict(relation_counts),
        source_files_unchanged=True,forbidden_label_fields_found=0,network_requests=0,model_inference_calls=0,training_calls=0,
        source_dependencies=list(source_hashes),no_labels_or_results_file_read=True,
        note='Counts of claims/quotes are repeated state occurrences, not independent facts or matches')
    save(out/'checks.json',dict(summary=summary,rows=checks));save(out/'source-hashes.json',source_hashes)
    report=[f"已生成 {len(output)} 条不含标签的状态—候选关系 sidecar，覆盖 {len(mids)} 场比赛、{len(documents)} 篇正文。",
        "sample_id 可连接原 features/向量及后续监督标签，但本构建器没有打开 labels/results 文件。",
        "主客球队、赛季与比赛信息仅来自已保存 fixtures/matches 和 semantic input.fixture；未从正文或 URL 猜测。",
        f"51场主客身份已知={summary['target_home_away_known_matches']}；UTC kickoff已知={summary['target_kickoff_known_matches']}；本地开球时间已知={summary['target_local_time_known_matches']}。",
        f"逐样本 cutoff 已知={summary['prediction_cutoff_known_rows']}；source_available_at 已知={summary['candidate_source_available_at_known_rows']}；published_at已知={summary['candidate_published_at_known_rows']}；captured_at已知={summary['candidate_captured_at_known_rows']}。",
        "所有 remaining_adoption_budget 为 null；remaining_candidate_count 单独保留，未冒充预算。",
        "候选/已采用/已知标题证据的实体名、主客侧、类型、值、scope、statement、原文引句及精确Unicode offset分别保存。没有明确条件注释或时间范围的 claim 保持 null，原 statement/quote 仍保留。",
        "已有多状态、候选新增多状态、候选扩大已有多状态分开；多状态不是已判真伪的逻辑冲突。",
        json.dumps(dict(relation_counts),ensure_ascii=False),
        f"共核对 {quote_count} 次引句引用，全部能在绑定原文找到；该数字包含同一证据在不同状态重复出现，不是独立事实数量。",
        "正文及已采用字符串SHA精确引用原Qwen缓存，未编码新拼接上下文，原features/向量/模型未改。",
        "历史 available_at 未知不会一概排除历史监督研究；仍保持其时间/版本未知属性。",
        "当前sidecar只补显式关系，不自动加入旧模型特征，也未重新训练。"]
    (out/'readable-report.txt').write_text('\n'.join(report)+'\n',encoding='utf-8')
    save(out/'manifest.json',{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='manifest.json'})
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
