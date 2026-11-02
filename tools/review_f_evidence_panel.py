"""Explicit human-readable review mapping and frozen-f diagnostic, no training.

This is a bounded eight-document research review, not an automatic extractor.
The richer observations are sidecar annotations, not changes to frozen schemas.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from football_research.acquisition.entity_semantic_f import EntityEvidenceBank, EntitySemanticF, _predict
from football_research.acquisition.historical_context_training import _project
from football_research.acquisition.news_link_projection import VERSION as NEWS_PROJECTION
from football_research.acquisition.semantic_f import EXTRACTION_VERSION

# claim ID -> observed concept, team, entities, optional coarse availability.
# No probabilities, importance scores, outcomes, or gain labels in this mapping.
MAPPING = {
 'S01-C01': ('match_ambition','away',['Fulham'],None),
 'S01-C02': ('season_ambition','away',['Fulham'],None),
 'S02-C01': ('confirmed_absence','away',['Antonee Robinson'],'unavailable'),
 'S02-C02': ('recovery_not_ready','away',['Antonee Robinson'],None),
 'S02-C03': ('training_participation','away',['Ryan Sessegnon'],None),
 'S02-C04': ('match_decision_pending','away',['Ryan Sessegnon'],'doubtful'),
 'S03-C01': ('suspension','home',['Ryan Gravenberch'],'unavailable'),
 'S03-C02': ('possible_bench','home',['Joe Gomez'],None),
 'S03-C03': ('no_update','home',['Conor Bradley'],None),
 'S03-C04': ('confirmed_absence','away',['Justin Kluivert'],'unavailable'),
 'S03-C05': ('training_participation','away',['Ryan Christie'],None),
 'S03-C06': ('confirmed_absence','away',['Ryan Christie'],'unavailable'),
 'S03-C07': ('club_eligibility_pending','away',['Dango Ouattara'],None),
 'S04-C01': ('confirmed_starter','home',['Alexis Mac Allister'],'available'),
 'S04-C02': ('observed_lineup_change','home',['Liverpool'],None),
 'S04-C03': ('confirmed_bench','home',['Joe Gomez'],'available'),
 'S05-C01': ('collective_availability_with_exception','away',['Joe Willock'],'unavailable'),
 'S05-C02': ('confirmed_squad','away',['Malick Thiaw'],'available'),
 'S05-C03': ('unspecified_unchanged_status','away',['Alexander Isak'],None),
 'S05-C04': ('season_attacking_ambition','away',['Newcastle United'],None),
 'S06-C01': ('confirmed_absence','home',['Destiny Udogie'],'unavailable'),
 'S06-C02': ('collective_availability_with_exception','home',['Destiny Udogie'],'unavailable'),
 'S06-C03': ('confirmed_squad','home',['Yves Bissouma'],'available'),
 'S06-C04': ('partial_training','home',['Manor Solomon'],None),
 'S06-C05': ('ongoing_absence','home',['James Maddison','Dejan Kulusevski','Radu Dragusin'],'unavailable'),
 'S06-C06': ('previous_match_conditions','home',['Tottenham'],None),
 'S07-C01': ('ticket_sales','home',['Fulham'],None),
 'S07-C02': ('ticket_exchange_deadline','home',['Fulham'],None),
 'S08-C01': ('match_preparation','away',['Chelsea'],None),
 'S08-C02': ('training_participation','away',['Jordan Henderson'],None),
 'S08-C03': ('training_participation','away',['Chelsea training group'],None),
}


def reviewed_mapping(sample, body):
    observations, claims = [], []
    for original in sample['review']['claims']:
        c = deepcopy(original)
        assert body[c['start']:c['end']] == c['quote'], c['claim_id']
        ctx = c['context_span']
        assert body[ctx['start']:ctx['end']] == ctx['text'], c['claim_id']
        concept, team, entities, value = MAPPING[c['claim_id']]
        # Preserve the full original context in observations. The protocol's
        # quote-size bound requires a shorter exact context for long paragraphs.
        context = ctx['text'] if len(ctx['text']) <= 2400 else body[max(0,c['start']-400):min(len(body),c['end']+250)]
        observation = {'sample_id':sample['sample_id'], 'review_claim':c,
            'observed_concept':concept, 'team':team, 'entities':entities,
            'legacy_availability':value,
            'mapping_basis':'explicit_codex_full_body_review',
            'interpretation_note':('Confirmed selection entails coarse availability; starter/bench/squad distinction lost.'
                if concept in ('confirmed_starter','confirmed_bench','confirmed_squad') else
                'Only named exception mapped; collective remainder is retained without invented player counts.'
                if concept == 'collective_availability_with_exception' else
                'Observed fact retained; no learned effect or importance assigned.'),
            'information_value':None}
        observations.append(observation)
        for entity in entities:
            claims.append({'team':team, 'entity':entity,
                'kind':'availability' if value else 'other', 'value':value or 'unmodeled',
                'scope':'this_match' if value else ('historical' if concept == 'previous_match_conditions' else 'team_context'),
                'statement':c['assertion'], 'quotes':[c['quote'],context]})
    return observations, claims


def state_for(sample):
    f = sample['fixture']
    reference = (datetime.fromisoformat(f['kickoff'].replace('Z','+00:00'))-timedelta(minutes=1)).isoformat()
    return {'fixture':{'physical_match_id':f['physical_match_id'],'match_id':f['physical_match_id'],
        'home_team_id':f['home_team'],'away_team_id':f['away_team'],
        'home_name':f['home_team'],'away_name':f['away_team'],
        'competition_id':f['competition'],'competition':f['competition'],'kickoff':f['kickoff']},
        'decision_at':reference,'prediction_cutoff':reference,
        'information_mode':'historical_reconstruction',
        'prior_probabilities':[.4,.3,.3], 'prior_basis':'constructed_offline_diagnostic_not_market',
        'documents':[], 'observed_links':[], 'history':[]}


def evaluate(samples, bodies, mapped, model):
    before = state_for(samples[0])
    after = deepcopy(before)
    records, body_ids = [], []
    for s in samples:
        doc={'url':s['url'],'title':s['title'],'text':bodies[s['sample_id']],
             'source_published_at':s['original_clocks']['source_published_at'],
             'captured_at':s['original_clocks']['captured_at']}
        single=deepcopy(before);single['documents']=[doc]
        items=EntityEvidenceBank._items(_project(single,text_projection=NEWS_PROJECTION))
        for channel in ('title','body'):
            for public in items[channel].values():
                if channel=='body':body_ids.append(public['input_id'])
                records.append({'input':public,'response':{'schema':EXTRACTION_VERSION,
                    'input_id':public['input_id'],'claims':mapped[s['sample_id']] if channel=='body' else []},
                    'provenance':{'kind':'explicit_codex_review_mapping','sample_id':s['sample_id'],
                        'provider_execution':False,'title_policy':'no specific individual status encoded from generic titles'}})
        after['documents'].append(doc)
    bank=EntityEvidenceBank(records)
    a,b=bank.project(before),bank.project(after)
    p,q=_predict(model.coefficients,a),_predict(model.coefficients,b)
    assert set(body_ids)<=set(b['used_input_ids'])
    return {'sample_ids':[s['sample_id'] for s in samples],
        'constructed_prior':p.tolist(),'probabilities_after':q.tolist(),
        'probability_l1_shift':float(abs(p-q).sum()), 'body_input_ids':body_ids,
        'body_consumed':True,'projection_after':b,'semantic_records':records,
        'outcome':None,'gain':None,'runtime_policy_executed':False,
        'reference_cutoff_is_actual_execution_clock':False}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    source=ROOT/'var/work/fm-f-evidence-training-panel-20261014-01'
    samples=[json.loads(x) for x in (source/'samples.jsonl').read_text(encoding='utf-8-sig').splitlines()]
    assert {c['claim_id'] for s in samples for c in s['review']['claims']}==set(MAPPING)
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    bodies, mapped, observations={}, {}, []
    for s in samples:
        raw=(source/s['body_path']).read_bytes()
        assert sha256(raw).hexdigest()==s['body_sha256']
        body=raw.decode('utf-8');bodies[s['sample_id']]=body
        obs,claims=reviewed_mapping(s,body);observations.extend(obs);mapped[s['sample_id']]=claims
    model_path=ROOT/'var/f-entity-semantic-f-shadow-materialization-20261011-v1/model.json'
    model_raw=model_path.read_bytes();model=EntitySemanticF.load(model_path)
    rows=[evaluate([s],bodies,mapped,model) for s in samples]
    seq=evaluate([samples[2],samples[3]],bodies,mapped,model)
    seq['before_S04']=rows[2]['probabilities_after']
    seq['incremental_l1']=sum(abs(a-b) for a,b in zip(seq['before_S04'],seq['probabilities_after']))
    seq['ordering_basis']='source publication only; captured historical versions not proven available at original cutoff'
    obs_by_id={o['review_claim']['claim_id']:o for o in observations}
    checks={
        'eight_bodies_consumed':len(rows)==8 and all(r['body_consumed'] for r in rows),
        'all_31_quotes_contexts_valid':len(observations)==31,
        'training_not_promoted_to_available':all(o['legacy_availability'] is None for o in observations if o['observed_concept'] in ('training_participation','partial_training')),
        'christie_training_and_absence_both_retained':obs_by_id['S03-C05']['legacy_availability'] is None and obs_by_id['S03-C06']['legacy_availability']=='unavailable',
        'squad_and_bench_coarse_mapping_explicit':all(obs_by_id[c]['legacy_availability']=='available' for c in ('S04-C01','S04-C03','S05-C02','S06-C03')),
        'collective_remainder_not_invented':all(len(o['entities'])==1 and o['legacy_availability']=='unavailable' for o in observations if o['observed_concept']=='collective_availability_with_exception'),
        's03_has_no_future_confirmed_bench':not any('joe gomez|availability|available' in k for k in rows[2]['projection_after']['novel_body_entity_keys']),
        's04_has_confirmed_bench':any('joe gomez|availability|available' in k for k in rows[3]['projection_after']['novel_body_entity_keys']),
        'ticket_and_training_controls_unchanged':all(rows[i]['probability_l1_shift']<1e-12 for i in (0,6,7)),
        'model_bytes_unchanged':model_path.read_bytes()==model_raw,
    }
    report={'schema':'fm-reviewed-evidence-panel-f-diagnostic-v1','executed_at':datetime.now(timezone.utc).isoformat(),
        'source':str(source),'sample_count':8,'reviewed_claim_count':len(observations),
        'model_version':model.model_version,'model_sha256':sha256(model_raw).hexdigest(),
        'prior_basis':'constructed [.4,.3,.3], not market or real forecasts',
        'semantic_extraction_basis':'manual explicit Codex mapping, not automatic provider validation',
        'historical_version_assumptions_preserved':True,
        'original_clocks':{s['sample_id']:s['original_clocks'] for s in samples},
        'samples':rows,'publication_order_diagnostic':seq,'checks':checks,'passed':all(checks.values()),
        'network_requests':0,'provider_calls':0,'training':False,'g_training_rows':0,
        'zero_or_negative_gain_is_exclusion':False,
        'limitations':['constructed prior','historical version unknown','no outcome used',
            'f response is not accuracy validation','source assertion is not independently verified truth',
            'typed predicates lose selection resolution and uncertainty; hand mapping not extractor automation test']}
    def write(name,obj):
        (out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    write('diagnostic.json',report)
    (out/'reviewed-observations.jsonl').write_text(''.join(json.dumps(o,ensure_ascii=False)+'\n' for o in observations),encoding='utf-8')
    lines=['8篇正文审阅与冻结 f 响应诊断','',
        '所有行使用构造先验 [0.4,0.3,0.3]。不是实际比赛预测；未读取赛果、未训练 f/g。',
        '人工明确映射 -> 冻结语义校验 -> EntityEvidenceBank -> 同字节 shadow f _predict；不是在线自动抽取测试。','']
    for s,r in zip(samples,rows):
        lines += [f"{s['sample_id']} {s['title']}",
            f"正文消费={r['body_consumed']}；建模实体谓词={len(r['projection_after']['novel_body_entity_keys'])}；L1变化={r['probability_l1_shift']:.10f}",
            'after='+str(r['probabilities_after']),
            '特征='+str(r['projection_after']['novel_body_entity_keys']),'']
    lines+=['来源发布时间顺序 S03 -> S04（历史版本假设，不是前瞻轨迹）：',
        'S03='+str(seq['before_S04']),'S03+S04='+str(seq['probabilities_after']),
        '结论：首发/替补/入选名单被粗化成 available；训练不能硬转成 available；集体例外不扩成虚构球员计数。',
        '没有本场计划轮换/战术部署正例，不能对这些特征宣称已验证。',
        '检查='+str(checks)]
    (out/'readable-results.txt').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'passed':report['passed'],'rows':[{k:r[k] for k in ('sample_ids','body_consumed','probability_l1_shift','probabilities_after')} for r in rows]},ensure_ascii=False))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
