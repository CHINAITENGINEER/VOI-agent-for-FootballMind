"""Versioned PIT features over a pinned season snapshot.

Uses the existing Elo/form implementation; never fabricates historical raw
publication timestamps. Reconstruction assumptions are explicit model inputs.
"""
from collections import defaultdict
from datetime import datetime,timedelta
from pathlib import Path
from uuid import uuid4
import json
import math
from psycopg.types.json import Jsonb

from football_spider.features.store import compute,select_matches,PARAMETERS
from football_spider.storage.raw_archive import digest,json_bytes

VERSION='prematch-season-v2.0'
DEFAULT_ASSUMPTIONS={'result_delay_hours':3.0,'xg_delay_hours':24.0,'fixture_lead_hours':168.0,
    'kickoff_timezone_policy':'UTC inferred from archived official calendar JS timezoneOffset conversion; raw strings retained',
    'revision_bias':'Latest frozen source snapshot may contain post-event corrections and rescheduled kickoff times; not historical operational truth'}


def assumptions_checked(values):
    a=dict(values)
    for key in ('result_delay_hours','xg_delay_hours','fixture_lead_hours'):
        v=a.get(key)
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0:
            raise ValueError('Explicit finite nonnegative reconstruction delays required')
    if a['result_delay_hours']<2 or a['xg_delay_hours']<a['result_delay_hours'] or not a.get('kickoff_timezone_policy'):
        raise ValueError('Reconstruction assumptions violate duration/timezone bounds')
    return a


def calculate(target,history,statistics,cutoff,mode,assumptions):
    mid=str(target['match_id'])
    if cutoff.tzinfo is None or cutoff>=target['event_time']:raise ValueError('Prematch aware cutoff required')
    if mode=='historical_reconstruction':
        assumptions=assumptions_checked(assumptions)
        fixture_at=target['event_time']-timedelta(hours=assumptions['fixture_lead_hours'])
    elif mode=='operational':
        fixture_at=max(target['observed_at'],target['ingested_at'])
        if target['status'] not in ('scheduled','unknown'):raise ValueError('No operational prematch fixture')
    else:raise ValueError('Unknown information mode')
    if fixture_at>cutoff:raise ValueError('Fixture not available at cutoff')
    eligible=[];lineage=[]
    for row in history:
        if (str(row['match_id'])==mid or row['competition_id']!=target['competition_id'] or
            row['event_time']>=cutoff or row['record']['status']!='finished' or
            row['record']['score_period']!='regulation' or None in row['record']['score'].values()):continue
        available=max(row['observed_at'],row['ingested_at']) if mode=='operational' else row['event_time']+timedelta(hours=assumptions['result_delay_hours'])
        if available>cutoff:continue
        eligible.append(row)
        lineage.append({'match_id':str(row['match_id']),'version_id':str(row['id']),'kickoff':row['event_time'].isoformat(),
            'result_available_at':available.isoformat(),'observed_at':row['observed_at'].isoformat(),'ingested_at':row['ingested_at'].isoformat(),
            'observation_id':str(row['observation_id'])})
    # Target score and xG are never consulted by compute or the rolling stats.
    values,versions,_=compute(target,eligible,[],cutoff)
    by_version={str(r['id']):r for r in eligible}
    selected_stats={}
    for stat in statistics:
        row=by_version.get(str(stat['match_version_id']))
        if not row or stat['provider']!='understat' or stat['definition_version']!='understat-match-xg-v1':continue
        available=max(stat['observed_at'],stat['ingested_at']) if mode=='operational' else row['event_time']+timedelta(hours=assumptions['xg_delay_hours'])
        if available>cutoff:continue
        if stat.get('source_published_at') and stat['source_published_at']>cutoff:continue
        key=(str(stat['team_id']),str(stat['match_id']),stat['metric'])
        selected_stats[key]=(stat,row,available)
    used_stats=[];xg_lineage=[]
    for side in ('home','away'):
        tid=str(target[side+'_id'])
        rows=[r for r in eligible if tid in (str(r['home_id']),str(r['away_id']))]
        rows.sort(key=lambda r:(r['event_time'],str(r['match_id'])))
        recent=rows[-PARAMETERS['recent_n']:]
        form=values[side+'_form'];count=form['matches']
        for key in ('goals_for','goals_against','points'):
            values[f'{side}_rolling_{key}']=form[key]/count if count else None
        values[f'{side}_rolling_matches']=count
        xvalues={'xg':[],'xga':[]}
        for row in recent:
            for metric in ('xg','xga'):
                value=selected_stats.get((tid,str(row['match_id']),metric))
                if value:
                    stat,_,available=value
                    xvalues[metric].append(stat['value']);used_stats.append(str(stat['id']))
                    xg_lineage.append({'statistic_id':str(stat['id']),'match_id':str(row['match_id']),
                        'xg_available_at':available.isoformat(),'provider':stat['provider'],'definition_version':stat['definition_version']})
        # Complete recent window required; do not quietly average only observed xG rows.
        valid_xg=bool(count) and all(len(v)==count for v in xvalues.values())
        values[f'{side}_rolling_xg_for']=sum(xvalues['xg'])/count if valid_xg else None
        values[f'{side}_rolling_xg_against']=sum(xvalues['xga'])/count if valid_xg else None
        values[f'{side}_rolling_xg_difference']=values[f'{side}_rolling_xg_for']-values[f'{side}_rolling_xg_against'] if valid_xg else None
        values[side+'_form']['coverage']='complete_within_frozen_league_season' if target['record']['coverage']=='verified_complete' else 'unverified_incomplete'
    values['xg']={'provider':'understat','definition_version':'understat-match-xg-v1','window':PARAMETERS['recent_n'],
                 'missing_reason':'no_history_or_incomplete_xg_availability_window',
                 'statistic_ids':sorted(set(used_stats))}
    values['elo']['coverage']='frozen_season_only_no_prior_season_warmup'
    # Every result input time precedes cutoff. Target fixture identity may be an
    # assumption; its later final/result payload is referenced, never projected.
    availability=[fixture_at]+[datetime.fromisoformat(r['result_available_at']) for r in lineage]
    availability += [datetime.fromisoformat(r['xg_available_at']) for r in xg_lineage]
    manifest={'match_id':mid,'prediction_cutoff':cutoff.isoformat(),'feature_version':VERSION,'information_mode':mode,
        'parameters':PARAMETERS,'availability_basis':'max(observed_at,ingested_at)' if mode=='operational' else 'explicit_frozen_snapshot_reconstruction_assumptions',
        'features_available_at':max(availability).isoformat(),'max_input_available_at':max(availability).isoformat(),
        'assumptions':assumptions if mode!='operational' else None,'target_version':str(target['id']),
        'target_projection':'fixture_identity_and_kickoff_only_no_result_no_xg_no_forecast',
        'history_version_ids':sorted(versions),'history_match_ids':sorted(str(r['match_id']) for r in eligible),
        'history_inputs':sorted(lineage,key=lambda r:r['version_id']),'xg_inputs':sorted(xg_lineage,key=lambda r:r['statistic_id']),
        'statistic_ids':sorted(set(used_stats)),'quote_ids':[],
        'observation_ids':sorted({str(target['observation_id'])}|{str(r['observation_id']) for r in eligible}),
        'missing_scope':'no_prior_season_warmup; league_only_rest_days; no_historical_bookmaker_quotes'}
    return values,manifest


class SeasonFeatureStore:
    def __init__(self,base_store):self.base=base_store;self.repo=base_store.repo

    def get_features(self,match_id,as_of,information_mode='operational',*,version_ids=None,assumptions=None):
        if as_of.tzinfo is None:raise ValueError('Aware cutoff required')
        if information_mode=='historical_reconstruction' and (not version_ids or assumptions is None):
            raise ValueError('Reconstruction requires pinned immutable version_ids and assumptions')
        with self.repo.conn.transaction():
            self.repo.conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ')
            if version_ids is not None:
                rows=self.repo.conn.execute('SELECT * FROM fm.match_versions WHERE id=ANY(%s::uuid[])',(list(version_ids),)).fetchall()
                if len(rows)!=len(set(version_ids)):raise ValueError('Pinned version missing')
            else:
                rows=self.repo.conn.execute('SELECT * FROM fm.match_versions WHERE observed_at<=%s AND ingested_at<=%s',(as_of,as_of)).fetchall()
            if information_mode=='operational':rows=[r for r in rows if max(r['observed_at'],r['ingested_at'])<=as_of]
            selected=select_matches(rows);target=selected.get(str(match_id))
            if not target:raise ValueError('No target match known at requested cutoff/mode')
            relevant=[r for r in selected.values() if r['competition_id']==target['competition_id']]
            stats=self.repo.conn.execute('SELECT * FROM fm.team_statistics WHERE match_version_id=ANY(%s::uuid[])',([r['id'] for r in relevant],)).fetchall()
            values,manifest=calculate(target,relevant,stats,as_of,information_mode,assumptions)
            mh=digest(manifest)
            row=self.repo.conn.execute('''INSERT INTO fm.feature_snapshots(id,match_id,prediction_cutoff,feature_version,information_mode,input_manifest_hash,manifest,features)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(match_id,prediction_cutoff,feature_version,information_mode,input_manifest_hash)
                DO UPDATE SET input_manifest_hash=EXCLUDED.input_manifest_hash RETURNING *''',
                (str(uuid4()),match_id,as_of,VERSION,information_mode,mh,Jsonb(manifest),Jsonb(values))).fetchone()
        self.base.export(row)
        return {'snapshot_id':str(row['id']),'manifest_hash':mh,'input_manifest_hash':mh,'manifest':manifest,'features':values,
                'match_id':str(match_id),'prediction_cutoff':as_of.isoformat(),'feature_version':VERSION,'information_mode':information_mode,
                'features_available_at':manifest['features_available_at'],'home_team_id':str(target['home_id']),
                'away_team_id':str(target['away_id']),'competition_id':str(target['competition_id']),'kickoff':target['event_time'].isoformat(),
                'parquet_path':str(self.base.output_dir/f"{row['id']}.parquet")}

    def replay(self,row):
        m=row['manifest']
        if digest(m)!=row['input_manifest_hash']:raise ValueError('Manifest checksum mismatch')
        target=self.repo.conn.execute('SELECT * FROM fm.match_versions WHERE id=%s',(m['target_version'],)).fetchone()
        history=self.repo.conn.execute('SELECT * FROM fm.match_versions WHERE id=ANY(%s::uuid[])',(m['history_version_ids'],)).fetchall()
        stats=self.repo.conn.execute('SELECT * FROM fm.team_statistics WHERE id=ANY(%s::uuid[])',(m['statistic_ids'],)).fetchall()
        if len(history)!=len(m['history_version_ids']) or len(stats)!=len(m['statistic_ids']):raise ValueError('Immutable inputs missing')
        values,manifest=calculate(target,history,stats,row['prediction_cutoff'],row['information_mode'],m['assumptions'])
        if values!=row['features'] or manifest!=m:raise ValueError('Replay mismatch')
        return {'snapshot_id':str(row['id']),'consistent':True,'manifest_hash':digest(m)}
