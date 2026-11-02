"""Reconstruct five research priors from existing archives; no text or g fit.

Provider IDs, not local UUIDs or fuzzy names, join seasons. Target results are
deserialized for archived identities but excluded from every model fit. Result
availability remains the export's explicit historical +3h assumption.
"""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import argparse
import json
from pathlib import Path
import socket
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from football_research.acquisition.match_conditioned_f import ReconstructedMatchPrior, bind_prior
from football_research.acquisition.state import parse_time
from football_research.baselines.models import PoissonModel

ARCHIVES=[
 'var/parallel/collectors/round3/data/runs/fc298603-aff8-4f3e-bc4f-6c91db871036/exports/understat-2024-historical_reconstruction-ec78dfa8670887cf',
 'var/parallel/collectors/round2/exports/understat-2025-historical_reconstruction-a3b3d45811fd5cda']
TARGETS=['understat:28778','understat:28779','understat:28780','understat:28782','understat:28942']


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def write(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def no_network(*args,**kwargs):raise RuntimeError('Offline research only')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    socket.socket.connect=no_network;socket.getaddrinfo=no_network
    write(out/'plan.json',{'schema':'fm-panel-reconstructed-priors-plan-v1',
        'frozen_at':datetime.now(timezone.utc).isoformat(),'targets':TARGETS,'sources':ARCHIVES,
        'regularization':5.0,'cutoff':'kickoff minus one hour','training':'all earlier eligible archived results, excluding all five panel targets',
        'identity_join':'exact provider source and entity_type and source_id; reject conflicting mappings',
        'source_availability':'historical reconstruction assumption, never verified original public clock',
        'text_f_refit':False,'outcome_scoring':False,'g_fit':False,
        'old_f_target_dependency_removed':False,'operational_eligible':False})
    combined=[];bindings=[];identity_audit=[]
    for relative in ARCHIVES:
        folder=ROOT/relative
        archive=ReconstructedMatchPrior.from_export(folder)
        manifest=read(folder/'manifest.json');entry=manifest['files']['id_mappings']
        raw=(folder/entry['path']).read_bytes()
        if sha256(raw).hexdigest()!=entry['sha256']:raise ValueError('Identity map digest mismatch')
        maps=[json.loads(x) for x in raw.decode('utf-8').splitlines() if x.strip()]
        lookup={}
        for m in maps:
            if m['source']!='understat' or m['entity_type'] not in ('team','competition'):continue
            key=(m['entity_type'],m['entity_id'])
            value=f"understat:{m['entity_type']}:{m['source_id']}"
            if key in lookup and lookup[key]!=value:raise ValueError('Ambiguous provider identity')
            lookup[key]=value
        for original in archive.rows:
            row=deepcopy(original)
            for field,kind in [('competition_id','competition'),('home_team_id','team'),('away_team_id','team')]:
                row[field]=lookup[(kind,row[field])]
            combined.append(row)
        bindings.append({**archive.source_binding,'identity_mapping_sha256':entry['sha256']})
        identity_audit.append({'archive':relative,'rows':len(archive.rows),'mapping':[
            {'kind':k[0],'local_id':k[1],'provider_id':v} for k,v in sorted(lookup.items())]})
    indexed={f"{r['source']}:{r['source_id']}":r for r in combined}
    if len(indexed)!=len(combined):raise ValueError('Duplicate cross-season matches')
    # Names are a cross-check of exact provider-ID joins, never the join key.
    provider_names={}
    for row in combined:
        for side in ('home','away'):
            provider_names.setdefault(row[side+'_team_id'],set()).add(row[side+'_team_name'])
    if any(len(names)!=1 for names in provider_names.values()):raise ValueError('Cross-season provider/name conflict')
    snapshots={};checks=[]
    for mid in TARGETS:
        target=indexed[mid]
        fixture={'physical_match_id':mid,'match_id':mid,'home_team_id':target['home_team_id'],
            'away_team_id':target['away_team_id'],'home_name':target['home_team_name'],
            'away_name':target['away_team_name'],'competition':target['competition_name'],
            'competition_id':target['competition_id'],'kickoff':parse_time(target['kickoff']).isoformat()}
        cutoff=(parse_time(target['kickoff'])-timedelta(hours=1)).isoformat()
        rows=[r for key,r in indexed.items() if key not in TARGETS or key==mid]
        prior=ReconstructedMatchPrior(rows,source_binding={'archives':bindings,'identity_normalization':'exact Understat IDs','panel_excluded':TARGETS})
        snap=prior.snapshot(fixture,cutoff)
        state={'fixture':fixture,'decision_at':cutoff,'information_mode':'historical_reconstruction'}
        bind_prior(state,snap)
        assert not set(snap['training_match_ids']).intersection(TARGETS)
        assert all(parse_time(indexed[k]['result_available_at'])<parse_time(cutoff) for k in snap['training_match_ids'])
        predicted=PoissonModel.from_state(snap['kernel']).predict_proba(snap['prediction_features'])
        assert max(abs(a-b) for a,b in zip(predicted,snap['probabilities']))<1e-14
        # Same archived identity with arbitrary altered target goals: no leakage.
        altered=deepcopy(rows)
        for r in altered:
            if f"{r['source']}:{r['source_id']}"==mid:r['home_goals'],r['away_goals']=99,98
        perturbed=ReconstructedMatchPrior(altered,source_binding=prior.source_binding).snapshot(fixture,cutoff)
        assert perturbed['probabilities']==snap['probabilities']
        assert perturbed['training_rows_sha256']==snap['training_rows_sha256']
        snapshots[mid]=snap
        checks.append({'physical_match_id':mid,'training_matches':len(snap['training_match_ids']),
            'probabilities':snap['probabilities'],'unknown_team_roles':snap['unknown_team_roles'],
            'target_goal_perturbation_invariant':True,'all_panel_results_excluded':True,
            'history_cutoff_checked':True,'model_reload_identical':True})
    write(out/'structured-priors.json',snapshots)
    write(out/'identity-normalization.json',identity_audit)
    write(out/'validation.json',{'passed':True,'checks':checks,'network':0,'text_f_training':False,'g_training':False,
        'actual_source_observation':'2026 archived versions; not historical live forecasts',
        'old_f_target_dependency_removed':False,'requires_residual_refit_for_independent_evaluation':True})
    lines=['历史重建无文本先验：不是市场报价，不是当时在线预测。',
        '使用2024/2025归档、精确Understat球队ID合并；每场仅拟合cutoff前结果，排除全部5场目标赛果。',
        '结果可用时刻沿用归档kickoff+3小时的研究假设，不是实测。',
        '旧文本f的间接目标依赖未解除；本次不计损失或gain。','']
    for c in checks:lines.append(f"{c['physical_match_id']} | history={c['training_matches']} | p={c['probabilities']} | unknown={c['unknown_team_roles']}")
    (out/'summary.txt').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(checks))


if __name__=='__main__':main()
