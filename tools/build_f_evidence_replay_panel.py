from __future__ import annotations
import hashlib, json, math
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import numpy as np

from football_research.acquisition.entity_semantic_f import (
    EntityEvidenceBank, EntitySemanticF, FEATURES, REPRESENTATION_SHA256,
    _predict, public_input,
)
from football_research.acquisition.semantic_f import EXTRACTION_VERSION
from football_research.acquisition.state import parse_time

ROOT = Path(r'D:\a1-足彩')
SRC = ROOT / 'var' / 'work' / 'fm-f-evidence-training-panel-20261014-01'
OUT = ROOT / 'var' / 'work' / 'fm-f-evidence-replay-panel-20261016-v1'
OUT.mkdir(parents=True, exist_ok=True)
MODEL_PATH = ROOT / 'var' / 'f-entity-semantic-f-shadow-materialization-20261011-v1' / 'model.json'
MODEL = EntitySemanticF.load(MODEL_PATH)
MODEL_BYTES = MODEL_PATH.read_bytes()
MODEL_SHA = hashlib.sha256(MODEL_BYTES).hexdigest()

# Only clear, directly match-bound claims are mapped into this frozen f vocabulary.
# Everything else remains an auditable other/unmodeled claim.
MODELED = {
    'S02-C01': ('away', 'availability', 'unavailable'),
    'S02-C02': ('away', 'availability', 'unavailable'),
    'S02-C04': ('away', 'availability', 'doubtful'),
    'S03-C04': ('away', 'availability', 'unavailable'),
    'S03-C06': ('away', 'availability', 'unavailable'),
    'S05-C01': ('away', 'availability', 'unavailable'),
    'S06-C01': ('home', 'availability', 'unavailable'),
}

def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def empty_state(prior):
    z = [0.0] * len(FEATURES)
    return {'prior': list(prior), 'representation_sha256': REPRESENTATION_SHA256,
            'title': z, 'novel_body': z,
            'title_entity_counts': z, 'novel_body_entity_counts': z}

def role_for_claim(sample, claim_id):
    if claim_id in MODELED:
        return MODELED[claim_id][0]
    # Preserve explicit team attribution where it is unambiguous; otherwise unknown.
    entities = next(c for c in sample['review']['claims'] if c['claim_id'] == claim_id)['entities']
    home, away = sample['fixture']['home_team'].casefold(), sample['fixture']['away_team'].casefold()
    text = ' '.join(entities).casefold()
    if away in text or (sample['fixture']['away_team'] == 'Newcastle United' and 'newcastle' in text):
        return 'away'
    if home in text or 'tottenham' in text and home == 'tottenham':
        return 'home'
    return 'unknown'

def fallback_scope(scope):
    return {'this_match':'this_match', 'current_team_context':'team_context',
            'training_session':'team_context', 'training_week':'team_context',
            'recovery_period':'team_context', 'ongoing_injury':'team_context',
            'target_preparation':'team_context', 'target_match_ticketing':'team_context',
            'ticket_exchange_deadline':'team_context', 'season':'historical',
            'early_season_expectations':'historical', 'previous_match_schedule':'historical',
            'this_match_compared_previous_match':'this_match'}.get(scope, 'team_context')

def build_claim(sample, claim):
    cid = claim['claim_id']
    if cid in MODELED:
        team, kind, value = MODELED[cid]
        scope = 'this_match'
    else:
        team = role_for_claim(sample, cid)
        kind, value, scope = 'other', 'unmodeled', fallback_scope(claim['scope'])
    return {'team': team, 'entity': claim['entities'][0] if claim['entities'] else 'source assertion',
            'kind': kind, 'value': value, 'scope': scope,
            'statement': claim['assertion'], 'quotes': [claim['quote']]}

def read_samples():
    rows=[]
    for line in (SRC/'samples.jsonl').read_text(encoding='utf-8').splitlines():
        sample=json.loads(line)
        body_path = SRC / sample['body_path'] if not Path(sample['body_path']).is_absolute() else Path(sample['body_path'])
        body=body_path.read_text(encoding='utf-8')
        sample['_body']=body
        rows.append(sample)
    return rows

records=[]
rows=[]
for sample in read_samples():
    fx = sample['fixture']
    kickoff = parse_time(fx['kickoff'])
    decision = kickoff - timedelta(hours=1)
    fixture = {
        'physical_match_id': fx['physical_match_id'],
        'match_id': fx['physical_match_id'],
        'home_team_id': fx['home_team'].casefold().replace(' ', '_'),
        'away_team_id': fx['away_team'].casefold().replace(' ', '_'),
        'home_name': fx['home_team'], 'away_name': fx['away_team'],
        'competition_id': 'premier-league', 'competition': fx['competition'],
        'kickoff': parse_time(fx['kickoff']).isoformat(),
        'home_aliases': [fx['home_team']], 'away_aliases': [fx['away_team']],
    }
    clock = sample['original_clocks']
    published = clock.get('source_published_at')
    captured = clock.get('captured_at') or clock.get('response_received_at') or clock.get('observed_at')
    if not published or not captured:
        raise ValueError(f'{sample["sample_id"]}: missing source clocks')
    doc = {'url': sample['url'], 'title': sample['title'], 'text': sample['_body'],
           'source_published_at': published, 'captured_at': captured,
           'source_modified_at': clock.get('source_modified_at')}
    base_state = {
        'fixture': fixture, 'decision_at': decision.isoformat(),
        'prediction_cutoff': decision.isoformat(), 'information_mode': 'historical_reconstruction',
        'prior_probabilities': [0.4, 0.3, 0.3],
        'prior_basis': 'constructed_diagnostic_prior_not_market',
        'allow_unknown_publication_clock': False, 'documents': [doc], 'observed_links': [],
    }
    title_public = public_input(fixture, doc, 'title')
    body_public = public_input(fixture, doc, 'body')
    claims = [build_claim(sample, c) for c in sample['review']['claims']]
    title_record = {'input': title_public,
                    'response': {'schema': EXTRACTION_VERSION, 'input_id': title_public['input_id'], 'claims': []},
                    'provenance': {'origin': 'codex_reviewed_manual_mapping', 'sample_id': sample['sample_id'], 'channel': 'title'}}
    body_record = {'input': body_public,
                   'response': {'schema': EXTRACTION_VERSION, 'input_id': body_public['input_id'], 'claims': claims},
                   'provenance': {'origin': 'codex_reviewed_manual_mapping', 'sample_id': sample['sample_id'], 'channel': 'body',
                                  'review_source': str(SRC/'samples.jsonl')}}
    records.extend([title_record, body_record])
    bank = EntityEvidenceBank([title_record, body_record])
    projected = bank.project(base_state)
    before = _predict(MODEL.coefficients, empty_state(base_state['prior_probabilities']))
    after = _predict(MODEL.coefficients, projected)
    modeled = [c for c in claims if c['kind'] != 'other' and c['scope'] == 'this_match' and c['team'] in ('home','away')]
    row = {
        'schema': 'footballmind-f-evidence-replay-row-v1',
        'sample_id': sample['sample_id'], 'document_id': sample['document_id'],
        'physical_match_id': fx['physical_match_id'], 'fixture': fixture,
        'body_path': sample['body_path'], 'body_sha256': sample['body_sha256'],
        'source_raw_sha256': sample['raw_sha256'], 'source_published_at': published,
        'captured_at': captured, 'historical_version_unknown': bool(sample.get('historical_version_unknown', True)),
        'information_mode': 'historical_reconstruction', 'decision_at': decision.isoformat(),
        'prediction_cutoff': decision.isoformat(), 'prior_probabilities': base_state['prior_probabilities'],
        'prior_basis': 'constructed_diagnostic_prior_not_market', 'body_consumed': body_public['input_id'] in projected['used_input_ids'],
        'title_consumed': title_public['input_id'] in projected['used_input_ids'],
        'used_input_ids': projected['used_input_ids'], 'claim_count': len(claims),
        'modeled_claim_count': len(modeled), 'unmodeled_claim_count': len(projected['unmodeled_claims']),
        'claims': [{'claim_id': c['claim_id'], 'quote': c['quote'], 'mapped': build_claim(sample,c)} for c in sample['review']['claims']],
        'feature_delta': {'title': projected['title'], 'novel_body': projected['novel_body'],
                         'title_entity_counts': projected['title_entity_counts'], 'novel_body_entity_counts': projected['novel_body_entity_counts'],
                         'title_entity_keys': projected['title_entity_keys'], 'novel_body_entity_keys': projected['novel_body_entity_keys']},
        'prediction_before': before.tolist(), 'prediction_after': after.tolist(),
        'l1_shift': float(np.abs(after-before).sum()),
        'prediction_changed': bool(not np.allclose(before, after, rtol=0, atol=1e-15)),
        'extraction_origin': 'codex_reviewed_manual_mapping',
        'extraction_status': projected['extraction_status'],
        'evidence_strength': 'source_assertion_not_independently_truth_verified',
        'production_ready': False, 'training': False, 'g_training_eligible': False,
        'realized_gain': None, 'outcome': None,
        'model_version': MODEL.model_version, 'model_sha256': MODEL_SHA,
        'representation_sha256': REPRESENTATION_SHA256,
    }
    rows.append(row)

# Persist only newly generated artifacts; source/history packages remain untouched.
jsonl=''.join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':'))+'\n' for row in rows)
(OUT/'panel.jsonl').write_text(jsonl, encoding='utf-8')
summary=[]
for row in rows:
    summary.append(f"{row['sample_id']} | {row['fixture']['home_name']}–{row['fixture']['away_name']} | 正文已消费={row['body_consumed']} | 可建模claim={row['modeled_claim_count']} | 未建模={row['unmodeled_claim_count']} | L1变化={row['l1_shift']:.12g} | 概率变化={row['prediction_changed']}")
report = '''# f evidence replay panel（离线诊断）\n\n本面板将已有归档正文与 Codex 逐篇审阅过的精确引文，按冻结 typed-v3 断言协议映射到实体语义 f。它验证正文是否能进入 f、哪些断言被当前特征词表接受，以及接受后 before/after 是否变化。\n\n- 不新增网络请求，不调用 DeepSeek，不运行正式 runtime。\n- prior 为构造的 `[0.4, 0.3, 0.3]`，不是市场先验。\n- `historical_reconstruction`、source assertion、available_at/历史版本边界保留；不生成训练标签或 g 行。\n- 只有直接绑定本场且落入冻结 availability 词表的断言才算 modeled；训练、出镜、愿望、赛季目标、票务、名单背景等均保留为 unmodeled。\n- `prediction_before` 是空证据状态下的同一 shadow f；`prediction_after` 是正文进入后同一 f 的重算。\n\n## 逐篇结果\n\n''' + '\n'.join(summary) + '\n\n## 解释\n\n有建模 claim 的样本用于验证 f 的正向接缝；正文被消费但没有建模 claim 的样本是负控制。任何概率变化都只说明冻结模型对构造 prior 和人工映射断言的响应，不代表真实足球预测收益、校准或因果效果。\n'
(OUT/'readable-report.md').write_text(report, encoding='utf-8')
verification = {
 'schema':'footballmind-f-evidence-replay-panel-validation-v1',
 'status':'offline_diagnostic_only',
 'rows':len(rows), 'body_consumed_rows':sum(r['body_consumed'] for r in rows),
 'modeled_rows':sum(r['modeled_claim_count']>0 for r in rows),
 'negative_control_rows':sum(r['modeled_claim_count']==0 for r in rows),
 'prediction_changed_rows':sum(r['prediction_changed'] for r in rows),
 'all_body_sha_match': all(hashlib.sha256((SRC/r['body_path']).read_bytes()).hexdigest()==r['body_sha256'] for r in rows if r['body_path'].startswith('bodies/')),
 'model_version':MODEL.model_version,'model_sha256':MODEL_SHA,'representation_sha256':REPRESENTATION_SHA256,
 'network_requests':0,'provider_invocations':0,'training_enabled':False,'formal_runtime_enabled':False,
 'production_ready':False,'g_training_eligible_rows':0,
 'prior_basis':'constructed_diagnostic_prior_not_market',
 'extraction_origin':'codex_reviewed_manual_mapping','source_files_unchanged':True,
 'panel_sha256': hashlib.sha256(jsonl.encode()).hexdigest(),
 'rows_summary':[{'sample_id':r['sample_id'],'modeled_claim_count':r['modeled_claim_count'],'l1_shift':r['l1_shift'],'prediction_changed':r['prediction_changed']} for r in rows],
}
(OUT/'verification.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
(OUT/'README.md').write_text('# f evidence replay panel\n\nSee readable-report.md and verification.json. This is an offline, constructed-prior, Codex-reviewed semantic replay. It is not an empirical f/g training set and does not promote any model.\n', encoding='utf-8')
print(json.dumps(verification, ensure_ascii=False, indent=2))
print('OUT',OUT)


