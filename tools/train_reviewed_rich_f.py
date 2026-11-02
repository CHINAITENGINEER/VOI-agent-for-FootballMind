"""Offline forward training of a reviewed-language probability head.

Rebuild priors uniformly; exclude all original ABC and panel target results
from base-model fits. Fit residuals only on earlier labels. This remains an
already-reviewed development corpus, not prospective or blind validation.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
import socket
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.reconstruct_panel_priors import ARCHIVES,TARGETS,no_network
from tools.review_f_evidence_panel import state_for,reviewed_mapping
from football_research.acquisition.match_conditioned_f import ReconstructedMatchPrior,bind_prior
from football_research.acquisition.historical_context_training import prepare_historical_cases,_discovery_panel,_project
from football_research.acquisition.news_link_projection import VERSION as NEWS_PROJECTION
from football_research.acquisition.entity_semantic_f import EntityEvidenceBank
from football_research.acquisition.semantic_f import EXTRACTION_VERSION
from football_research.acquisition.state import parse_time
from football_research.acquisition import reviewed_rich_f as rich
from football_research.acquisition import compact_evidence_f as compact

GRID=[None,10.,1.,.1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)


def history():
    rows=[];sources=[]
    for path in ARCHIVES:
        folder=ROOT/path;a=ReconstructedMatchPrior.from_export(folder)
        entry=read(folder/'manifest.json')['files']['id_mappings'];raw=(folder/entry['path']).read_bytes()
        assert sha256(raw).hexdigest()==entry['sha256']
        lookup={}
        for m in map(json.loads,raw.decode().splitlines()):
            if m['source']=='understat' and m['entity_type'] in ('team','competition'):
                k=(m['entity_type'],m['entity_id']);v=f"understat:{m['entity_type']}:{m['source_id']}"
                assert k not in lookup or lookup[k]==v;lookup[k]=v
        for r in a.rows:
            r=deepcopy(r)
            for f,t in [('home_team_id','team'),('away_team_id','team'),('competition_id','competition')]:r[f]=lookup[t,r[f]]
            rows.append(r)
        sources.append({**a.source_binding,'identity_map_sha256':entry['sha256']})
    return rows,sources


def row_for(state,bank,concepts,outcome,result_time):
    projected=bank.project(state);e=[]
    for input_id in projected['used_input_ids']:
        rec=bank.records[input_id]
        for claim in rec['response']['claims']:
            concept=concepts.get((input_id,claim['statement']))
            e.append(rich.evidence_tokens(claim,rec['input']['channel'],state['fixture'],concept))
    return {'physical_match_id':state['fixture']['physical_match_id'],'decision_at':state['decision_at'],
        'result_available_at':result_time,'outcome':outcome,'prior':projected['prior'],
        'evidence':rich.unique_evidence(e),'state':projected,
        'used_input_ids':projected['used_input_ids']}


def train(kind,rows,lam):
    if kind=='rich':return rich.fit(rows,lam)
    beta,info=compact.fit(rows,lam);return {'beta':beta,'optimizer':info}
def predict(kind,model,row):
    return rich.predict(model,row) if kind=='rich' else compact.predict(model['beta'],row['state'])
def evaluate(kind,model,rows):
    groups=defaultdict(list)
    for row in rows:
        p=predict(kind,model,row);y=row['outcome']
        groups[row['physical_match_id']].append((float(-np.log(p[y])),float(np.sum((p-np.eye(3)[y])**2))))
    return [{'physical_match_id':mid,'log_loss':float(np.mean([x[0] for x in values])),
        'brier':float(np.mean([x[1] for x in values]))} for mid,values in sorted(groups.items())]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    socket.socket.connect=no_network;socket.getaddrinfo=no_network
    protocol=read(ROOT/'var/f-entity-semantic-20261011-v2/protocol.json')
    a_ids=protocol['splits']['A'];excluded=set(sum(protocol['splits'].values(),[])+TARGETS)
    write(out/'plan.json',{'schema':rich.VERSION,'frozen_at':datetime.now(timezone.utc).isoformat(),
        'development_ids':a_ids+TARGETS[:4],'post_selection_probe_id':TARGETS[4],
        'base_fit_exclusions':sorted(excluded),'base_sources':ARCHIVES,'prior_regularization':5.,
        'residual_grid':GRID,'warmup_matches':6,'block_size':3,
        'selection':'simplest within one paired standard error, within each model family',
        'families':['no_text','compact_four_parameter','reviewed_rich'],
        'source_semantics':'existing frozen Codex reviews; no provider calls or LLM fine-tuning',
        'time_basis':'historical source versions and reconstructed result clocks; not prospective',
        'previously_reviewed_development':True,'B_C_residual_fit_or_scoring':False,
        'original_B_C_results_excluded_from_new_base_fits':True,'live_activation':False,'g_training':False})
    archive,sources=history();by_id={f"{r['source']}:{r['source_id']}":r for r in archive}
    assert len(by_id)==len(archive)
    snapshots={}
    def bind(state):
        mid=state['fixture']['physical_match_id'];cutoff=state['decision_at']
        if mid not in snapshots:
            data=[r for k,r in by_id.items() if k not in excluded or k==mid]
            generator=ReconstructedMatchPrior(data,source_binding={'archives':sources,'all_excluded_targets':sorted(excluded)})
            snapshots[mid]=generator.snapshot(state['fixture'],cutoff)
            assert not set(snapshots[mid]['training_match_ids'])&excluded
        return bind_prior(state,snapshots[mid])
    cases=read(protocol['source_bindings']['cases']['path'])
    cases,_=prepare_historical_cases([c for c in cases if c['fixture']['physical_match_id'] in a_ids])
    records=[r for r in read(protocol['source_bindings']['records']['path']) if r['input']['fixture']['physical_match_id'] in a_ids]
    bank=EntityEvidenceBank(records);rows=[]
    for case in cases:
        for _,_,_,_,state in _discovery_panel(case):
            state=bind(state);rows.append(row_for(state,bank,{},case['outcome'],case['result_available_at']))
    source=ROOT/'var/work/fm-f-evidence-training-panel-20261014-01'
    panel=[json.loads(x) for x in (source/'samples.jsonl').read_text(encoding='utf-8-sig').splitlines()]
    groups=defaultdict(list)
    for s in panel:
        if s['fixture']['physical_match_id'] in TARGETS:groups[s['fixture']['physical_match_id']].append(s)
    labels={r['physical_match_id']:r for r in map(json.loads,(ROOT/'var/work/fm-f-panel-prior-outcome-handoff-20261014-v1/labels/results.jsonl').read_text(encoding='utf-8').splitlines())}
    for mid,samples in groups.items():
        samples.sort(key=lambda s:parse_time(s['original_clocks']['source_published_at']))
        base=bind(state_for(samples[0]));docs=[];new_records=[];concepts={}
        for s in samples:
            raw=(source/s['body_path']).read_bytes();assert sha256(raw).hexdigest()==s['body_sha256']
            body=raw.decode();observations,claims=reviewed_mapping(s,body)
            doc={'url':s['url'],'title':s['title'],'text':body,'captured_at':s['original_clocks']['captured_at'],
                'source_published_at':s['original_clocks']['source_published_at']}
            one=deepcopy(base);one['documents']=[doc];items=EntityEvidenceBank._items(_project(one,text_projection=NEWS_PROJECTION))
            for channel in ('title','body'):
                for public in items[channel].values():
                    new_records.append({'input':public,'response':{'schema':EXTRACTION_VERSION,'input_id':public['input_id'],
                        'claims':claims if channel=='body' else []},'provenance':{'kind':'existing_Codex_panel_review_mapping','sample_id':s['sample_id']}})
                    if channel=='body':
                        for obs in observations:concepts[public['input_id'],obs['review_claim']['assertion']]=obs['observed_concept']
            docs.append(doc)
        bank_new=EntityEvidenceBank(new_records);target=by_id[mid];lab=labels[mid]
        assert (target['home_goals'],target['away_goals'])==(lab['regulation_home_goals'],lab['regulation_away_goals'])
        outcome=0 if target['home_goals']>target['away_goals'] else 1 if target['home_goals']==target['away_goals'] else 2
        for k in range(len(docs)+1):
            st=deepcopy(base);st['documents']=docs[:k]
            rows.append(row_for(st,bank_new,concepts,outcome,target['result_available_at']))
    write(out/'structured-priors.json',snapshots)
    write(out/'training-rows.json',rows)
    development=[r for r in rows if r['physical_match_id']!=TARGETS[4]]
    facts={r['physical_match_id']:r for r in development}
    ids=sorted(facts,key=lambda mid:(parse_time(facts[mid]['decision_at']),mid))
    assert set(ids)==set(a_ids+TARGETS[:4])
    scores={(kind,lam):[] for kind in ('compact','rich') for lam in GRID};folds=[]
    for start in range(6,len(ids),3):
        valid_ids=ids[start:start+3];cutoff=min(parse_time(facts[mid]['decision_at']) for mid in valid_ids)
        train_ids=[mid for mid in ids[:start] if parse_time(facts[mid]['result_available_at'])<cutoff]
        training=[r for r in development if r['physical_match_id'] in train_ids]
        valid=[r for r in development if r['physical_match_id'] in valid_ids]
        assert len(train_ids)>=3 and not set(train_ids)&set(valid_ids)
        detail={'training_ids':train_ids,'validation_ids':valid_ids,'hypotheses':[]}
        for kind in ('compact','rich'):
            for lam in GRID:
                m=train(kind,training,lam);sc=evaluate(kind,m,valid);scores[kind,lam].extend(sc)
                detail['hypotheses'].append({'family':kind,'regularization':lam,'metrics':sc,'optimizer':m['optimizer']})
                if kind=='rich':assert set(m['encoder']['fit_match_ids'])==set(train_ids)
        folds.append(detail);print(json.dumps({'fold':len(folds),'training_matches':len(train_ids),'validation_matches':len(valid_ids)}),flush=True)
    report={'schema':'fm-reviewed-rich-training-result-v1','development_matches':len(ids),
        'development_states':len(development),'new_non_leeds_training_matches':4,
        'forward_validation_matches':sum(len(f['validation_ids']) for f in folds),'folds':folds,'families':{},
        'validation_team_bias':'All forward validation targets involve Leeds; non-Leeds additions are warmup training. No cross-team superiority established.',
        'text_semantic_capacity':'reviewed assertions + quotes + scope/concept atoms; lexical residual head, not pretrained embedding or LLM fine-tune',
        'source_clock_basis':'historical reconstruction; post-event archives; not independent fresh test',
        'network':0,'provider_calls':0,'g_training':False,'production_ready':False}
    models={}
    for kind in ('compact','rich'):
        matrix=np.asarray([[r['log_loss'] for r in scores[kind,lam]] for lam in GRID])
        best=int(np.argmin(matrix.mean(axis=1)));se=(matrix-matrix[best]).std(axis=1,ddof=1)/np.sqrt(matrix.shape[1])
        chosen=next(i for i in range(len(GRID)) if matrix[i].mean()<=matrix[best].mean()+se[i]+1e-12)
        nonzero=1+int(np.argmin(matrix[1:].mean(axis=1)))
        report['families'][kind]={'selected_regularization':GRID[chosen],'best_nonzero_regularization':GRID[nonzero],
            'candidates':[{'regularization':lam,'log_loss':float(matrix[i].mean()),
                'brier':float(np.mean([r['brier'] for r in scores[kind,lam]])),
                'paired_se_to_best':float(se[i]),'per_match':scores[kind,lam]} for i,lam in enumerate(GRID)]}
        for label,index in [('selected',chosen),('nonzero_candidate',nonzero)]:
            m=train(kind,development,GRID[index]);models[kind,label]=m
            if kind=='rich':
                rich.save(m,out/f'{kind}-{label}.json');restored=rich.load(out/f'{kind}-{label}.json')
                assert np.array_equal(predict(kind,m,development[-1]),predict(kind,restored,development[-1]))
                report['families'][kind][label+'_parameter_count']=int(np.asarray(m['coefficients']).size)
            else:write(out/f'{kind}-{label}.json',m)
    # Chronologically later ticket-only probe: held out from fitting/selection,
    # but already reviewed and not a representative independent performance test.
    probe=[r for r in rows if r['physical_match_id']==TARGETS[4]]
    assert max(parse_time(r['result_available_at']) for r in development)<min(parse_time(r['decision_at']) for r in probe)
    probe_result={}
    for (kind,label),model in models.items():
        preds=[predict(kind,model,r).tolist() for r in probe]
        probe_result[kind+'-'+label]={'predictions':preds,'metrics':evaluate(kind,model,probe),
            'probability_l1_shift':float(np.abs(np.asarray(preds[-1])-preds[0]).sum())}
    report['ticket_probe']=probe_result
    final_predictions=[]
    for r in development:
        if r['evidence']:
            final_predictions.append({'physical_match_id':r['physical_match_id'],'used_input_ids':r['used_input_ids'],
                'prior':r['prior'],'rich_candidate':predict('rich',models['rich','nonzero_candidate'],r).tolist(),
                'scope':'full_fit_training_response_only_not_validation','gain':None})
    write(out/'training-response-diagnostics.json',final_predictions)
    write(out/'report.json',report)
    text=['已实际训练概率残差头；不是GPT微调。历史开发实验，不是前瞻验证。',f'训练比赛={len(ids)}，状态={len(development)}；时间验证比赛={report["forward_validation_matches"]}。',
        report['validation_team_bias'],'']
    for kind,data in report['families'].items():
        text.append(f'{kind}: selected={data["selected_regularization"]}; best_nonzero={data["best_nonzero_regularization"]}')
        for c in data['candidates']:text.append(f'  lambda={c["regularization"]} logloss={c["log_loss"]:.9f} brier={c["brier"]:.9f}')
    text+=['','票务探针：'+json.dumps(probe_result,ensure_ascii=False),'没有运行g训练或启用线上模型。']
    (out/'summary.txt').write_text('\n'.join(text),encoding='utf-8')
    print(json.dumps({k:{a:b for a,b in v.items() if a!='candidates'} for k,v in report['families'].items()}),flush=True)


if __name__=='__main__':main()
