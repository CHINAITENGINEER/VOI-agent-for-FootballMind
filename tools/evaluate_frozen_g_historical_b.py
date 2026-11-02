"""Expand fixed-model evaluation to old B archives, with no model fitting.

Only the numerical no-text prior is reconstructed with the already-declared
excluded-target protocol. B was used elsewhere in the project: not a blind test.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import socket
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.train_reviewed_rich_f import read,write,history,row_for
from tools.train_entity_update_f import enrich
from tools.train_state_candidate_g import before_snapshot,SOURCE
from tools.train_learned_continuation_g import PREVIOUS
from tools.reconstruct_panel_priors import TARGETS,no_network
from football_research.acquisition.match_conditioned_f import ReconstructedMatchPrior,bind_prior
from football_research.acquisition.historical_context_training import prepare_historical_cases,_discovery_panel
from football_research.acquisition.entity_semantic_f import EntityEvidenceBank
from football_research.acquisition import entity_evidence_updates as f
from football_research.acquisition import historical_gain_head as gain
from football_research.acquisition import state_candidate_gain as g1
from football_research.acquisition import learned_continuation_g as g2
from football_research.acquisition import paired_state_gain as paired
from football_research.acquisition.context_predictor import _digest
from football_research.acquisition.historical_policy_comparison import rollout


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--selection-dir',type=Path,required=True);args=ap.parse_args();out=args.output
    out.mkdir(parents=True,exist_ok=False);socket.socket.connect=no_network;socket.getaddrinfo=no_network
    protocol_path=ROOT/'var/f-entity-semantic-20261011-v2/protocol.json'
    protocol=read(protocol_path);ids=protocol['splits']['B'];excluded=set(sum(protocol['splits'].values(),[])+TARGETS)
    selection=read(args.selection_dir/'training-only-selection.json');weight=selection['selected_pair_weight']
    g_path=args.selection_dir/f'g-pair-{int(weight)}.json';f_path=SOURCE/'f-labeler.json'
    h_path=ROOT/'var/reviews/fm-g-learned-two-step-20261014-v1/g-two-step.json'
    paths=[protocol_path,g_path,f_path,h_path,Path(protocol['source_bindings']['cases']['path']),Path(protocol['source_bindings']['records']['path'])]
    bindings={str(p):sha256(p.read_bytes()).hexdigest() for p in paths}
    write(out/'plan.json',{'schema':'frozen-g-historical-b-expansion-v1','created_at':datetime.now(timezone.utc).isoformat(),
        'fixture_ids':ids,'split':'existing project B; not new prospective or project-blind evidence',
        'model_fit_or_hyperparameter_selection_on_B':False,'g_selected_by':'training-only eight-match cross-validation',
        'selected_pair_weight':weight,'source_bindings':bindings,'base_prior_excludes':sorted(excluded),
        'policies':['stop','fixed','random','selected_g','old_two_step'], 'budgets':[1,2,3], 'costs':[0.,.001,.01],
        'C_not_evaluated':True,'clocks':'historical publication/versions and reconstructed result timing assumptions retained',
        'production_ready':False,'online_selection_allowed':False})
    model=paired.load(g_path);labeler=f.load(f_path);two=g2.load(h_path)
    assert not set(ids)&set(model['fit_match_ids'])
    assert not set(ids)&set(labeler['head']['head']['encoder']['fit_match_ids'])
    archive,sources=history();by_id={f"{r['source']}:{r['source_id']}":r for r in archive}
    cases,_=prepare_historical_cases([c for c in read(protocol['source_bindings']['cases']['path']) if c['fixture']['physical_match_id'] in ids])
    assert {c['fixture']['physical_match_id'] for c in cases}==set(ids)
    records=[r for r in read(protocol['source_bindings']['records']['path']) if r['input']['fixture']['physical_match_id'] in ids]
    bank=EntityEvidenceBank(records);record_by_id={r['input']['input_id']:r for r in records}
    snapshots={};rows=[];labels={};clocks={}
    for case in cases:
        mid=case['fixture']['physical_match_id'];labels[mid]=case['outcome']
        for doc in case['documents']:clocks[mid,doc['url']]=doc['source_published_at']
        for _,_,_,_,state in _discovery_panel(case):
            if mid not in snapshots:
                generator=ReconstructedMatchPrior([r for k,r in by_id.items() if k not in excluded or k==mid],
                    source_binding={'archives':sources,'all_excluded_targets':sorted(excluded)})
                snapshots[mid]=generator.snapshot(state['fixture'],state['decision_at'])
                assert not set(snapshots[mid]['training_match_ids'])&excluded
            state=bind_prior(state,snapshots[mid])
            # No target outcome is supplied to the prediction row.
            rows.append(row_for(state,bank,{},None,case['result_available_at']))
    rows=enrich(rows);write(out/'prior-snapshots.json',snapshots);write(out/'prediction-input-rows.json',rows)
    states={};predictions={};row_by_state={}
    for row in rows:
        mid=row['physical_match_id'];used=sorted(row['used_input_ids']);sid=_digest({'match':mid,'inputs':used})
        assert sid not in states
        states[sid]={'physical_match_id':mid,'used_input_ids':used,
            'read_count':sum(record_by_id[k]['input']['channel']=='body' for k in used),
            'title_count':sum(record_by_id[k]['input']['channel']=='title' for k in used)}
        predictions[sid]=f.predict(labeler,row).tolist();row_by_state[sid]=row
    write(out/'fixed-f-predictions.json',predictions)
    transitions=[];labeler_sha=sha256(f_path.read_bytes()).hexdigest()
    for sid,s in states.items():
        mid=s['physical_match_id'];keys=set(s['used_input_ids']);before=row_by_state[sid]
        for tid,after in states.items():
            if after['physical_match_id']!=mid:continue
            target_keys=set(after['used_input_ids']);added=target_keys-keys
            if not keys.issubset(target_keys) or len(added)!=1:continue
            body=next(iter(added));public=record_by_id[body]['input']
            if public['channel']!='body':continue
            titles=[record_by_id[k]['input'] for k in keys if record_by_id[k]['input']['channel']=='title' and record_by_id[k]['input']['url']==public['url']]
            if len(titles)!=1:continue
            p,q=predictions[sid],predictions[tid];outcome=labels[mid]
            age=(f.timestamp(before['decision_at'])-f.timestamp(clocks[mid,public['url']])).total_seconds()/3600
            transitions.append({'physical_match_id':mid,'state_id':sid,'next_state_id':tid,
                'action_id':'read-'+_digest({'url':public['url']})[:20],'body_input_id':body,'body_url':public['url'],
                'query':{'schema':gain.QUERY_SCHEMA,'probabilities':p,'acquired_documents':s['read_count'],
                         'visible_candidates':0,'article_age_hours':age,'title':titles[0]['text']},
                'prediction_before':p,'prediction_after':q,'outcome':outcome,'signed_gain':gain.signed_gain(p,q,outcome),
                'f_labeler_sha256':labeler_sha,'provenance':'enumerated_historical_reconstruction',
                'propensity':None,'prospective_training_eligible':False,'g_split':'B_evaluation_only'})
    graph=defaultdict(list)
    for t in transitions:graph[t['state_id']].append(t)
    for sid,edges in graph.items():
        snap=before_snapshot(row_by_state[sid])
        for t in edges:
            t['query']['visible_candidates']=len(edges);t['query']=g1.build_query(t['query'],snap)
    write(out/'states.json',states);write(out/'transitions.json',transitions)
    assert {t['physical_match_id'] for t in transitions}==set(ids)
    starts={mid:max(((sid,s) for sid,s in states.items() if s['physical_match_id']==mid and s['read_count']==0),
                    key=lambda x:x[1]['title_count'])[0] for mid in ids}
    runs=[]
    for mid,start in starts.items():
        for budget in (1,2,3):
            for cost in (0.,.001,.01):
                for name in ('stop','fixed','random','selected_g','old_two_step'):
                    r=rollout(graph,start,policy='g' if name=='selected_g' else 'menu' if name=='old_two_step' else name,
                        budget=budget,cost=cost,scorer=lambda a:paired.estimate(model,a['query']),
                        menu_selector=lambda menu,left,c:g2.choose(two,menu,left,c))
                    for path in r['paths']:
                        if path['steps']:
                            exact=gain.signed_gain(predictions[start],predictions[path['end_state_id']],labels[mid])
                            assert abs(exact-path['gain'])<1e-12
                    runs.append({**r,'physical_match_id':mid,'comparison_policy':name})
    write(out/'policy-replays.json',runs)
    summary=[]
    for budget in (1,2,3):
        for cost in (0.,.001,.01):
            for name in ('stop','fixed','random','selected_g','old_two_step'):
                rs=[r for r in runs if r['budget']==budget and r['cost_loss_units']==cost and r['comparison_policy']==name]
                summary.append({'budget':budget,'cost':cost,'policy':name,
                    **{k:float(np.mean([r['expected_'+k] for r in rs])) for k in ('reads','gain','net')}})
    gain_scores=[]
    for mid in ids:
        rs=[t for t in transitions if t['physical_match_id']==mid]
        gain_scores.append({'match':mid,'g_mse':float(np.mean([(paired.estimate(model,t['query'])-t['signed_gain'])**2 for t in rs])),
                            'zero_mse':float(np.mean([t['signed_gain']**2 for t in rs]))})
    write(out/'gain-errors.json',gain_scores)
    for path,digest in bindings.items():assert sha256(Path(path).read_bytes()).hexdigest()==digest
    report={'physical_matches':len(ids),'prediction_states':len(states),'transitions':len(transitions),
        'gain_positive':sum(t['signed_gain']>1e-12 for t in transitions),'gain_negative':sum(t['signed_gain']< -1e-12 for t in transitions),
        'gain_zero':sum(abs(t['signed_gain'])<=1e-12 for t in transitions),
        'g_mse':float(np.mean([e['g_mse'] for e in gain_scores])),'zero_mse':float(np.mean([e['zero_mse'] for e in gain_scores])),
        'policy_summary':summary,'policy_runs':len(runs),'model_fit_calls':0,'prior_reconstruction_fits':len(snapshots),
        'network':0,'api_calls':0,'new_project_matches':0,'additional_matches_for_this_g_evaluation':len(ids),
        'blind_test':False,'C_scored':False,'model_bindings_unchanged':True,'production_ready':False,
        'limitations':['All five B matches also involve Leeds; multiteam independence still missing.',
                       'B used elsewhere in project; no claim of a never-seen test set.',
                       'Historical availability/version and prior reconstruction assumptions retained.',
                       'Menus enumerate archived body reads; no live tool failure/discovery dynamics.']}
    write(out/'report.json',report)
    from tools.train_learned_continuation_g import change
    lines=['冻结g的历史B组扩展验证','',
        '按既有B名单全量纳入5场，没有按gain筛场。没有训练f/g/h，只按原排除协议重建无文本prior。',
        '这5场未参与当前g训练/之前5场开发对照，但曾用于项目其他实验，不称项目盲测。C未评分。',
        f'状态{len(states)}，单正文转移{len(transitions)}。',
        f'g MSE={report["g_mse"]}；零预测MSE={report["zero_mse"]}。','',
        '预算 | 成本 | 策略 | 平均读取 | 不计成本 | 计入成本']
    for r in summary:lines.append(f"{r['budget']} | {r['cost']:.3f} | {r['policy']} | {r['reads']:.2f} | {change(r['gain'])} | {change(r['net'])}")
    (out/'研究结果.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import json
    print(json.dumps({k:v for k,v in report.items() if k!='policy_summary'},ensure_ascii=False))
    print(json.dumps([r for r in summary if r['budget']==2 and r['cost']==.001],ensure_ascii=False))


if __name__=='__main__':main()
