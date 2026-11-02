from datetime import datetime,timezone
from pathlib import Path
from uuid import UUID
import json

from football_spider.storage.postgres import Repository
from football_spider.storage.raw_archive import RawArchive,json_bytes
from football_spider.models.contracts import Observation
from football_spider.features.store import FeatureStore,select_matches


def utc(value):
    if isinstance(value,str):value=datetime.fromisoformat(value.replace('Z','+00:00'))
    if not isinstance(value,datetime) or value.tzinfo is None:raise ValueError('Explicit timezone required')
    return value.astimezone(timezone.utc)


def plain(value):return json.loads(json_bytes(value))


def match_projection(row):
    r=row['record']
    available=max(row['observed_at'],row['ingested_at'])
    return plain({'match_id':str(row['match_id']),'source':row['source'],'source_id':row['source_id'],
        'version_id':str(row['id']),'competition_id':str(row['competition_id']),'season':r['season'],
        'home_team_id':str(row['home_id']),'away_team_id':str(row['away_id']),
        'home_team_name':r['home']['name'],'away_team_name':r['away']['name'],'competition_name':r['competition']['name'],
        'kickoff':row['event_time'],'status':r['status'],'score_period':r['score_period'],
        'home_goals':r['score']['home'],'away_goals':r['score']['away'],
        'observed_at':row['observed_at'],'ingested_at':row['ingested_at'],'first_observed_at':row['first_observed_at'],
        'fixture_available_at':available,'result_available_at':available if r['status']=='finished' else None,
        'information_mode':'operational','observation_id':str(row['observation_id']),
        'supersedes_id':str(row['supersedes_id']) if row['supersedes_id'] else None,
        'data_version':row['content_hash'],'input_manifest':[f"match_version:{row['id']}",f"observation:{row['observation_id']}|at:{row['observed_at'].isoformat()}"]})


class PlatformReader:
    def __init__(self,database_url,archive_root,features_dir=None):
        self.repo=Repository(database_url)
        self.archive=RawArchive(Path(archive_root))
        self.store=FeatureStore(self.repo,features_dir or self.archive.root/'features')

    def close(self):self.repo.close()
    def __enter__(self):return self
    def __exit__(self,*_):self.close()

    def resolve_id(self,source,source_id,entity_type='match',as_of=None):
        cutoff=utc(as_of or datetime.now(timezone.utc))
        row=self.repo.conn.execute('''SELECT * FROM fm.entity_source_map WHERE source=%s AND source_id=%s AND entity_type=%s
            AND valid_from<=%s AND (valid_to IS NULL OR valid_to>%s) ORDER BY valid_from DESC LIMIT 1''',
            (source,str(source_id),entity_type,cutoff,cutoff)).fetchone()
        return plain(row) if row else None

    def _canonical(self,ident,as_of):
        try:return str(UUID(str(ident)))
        except ValueError:
            source,sid=str(ident).split(':',1)
            if sid.startswith('match:'):sid=sid[6:]
            row=self.resolve_id(source,sid,as_of=as_of)
            return str(row['entity_id']) if row else None

    def list_matches(self,as_of,source=None,competition_id=None,season=None):
        cutoff=utc(as_of)
        rows=self.repo.conn.execute('SELECT * FROM fm.match_versions WHERE observed_at<=%s AND ingested_at<=%s',(cutoff,cutoff)).fetchall()
        rows=[r for r in rows if (source is None or r['source']==source) and
              (competition_id is None or str(r['competition_id'])==str(competition_id)) and
              (season is None or r['record']['season']==str(season))]
        return [match_projection(r) for r in sorted(select_matches(rows).values(),key=lambda x:(x['event_time'],str(x['match_id'])))]

    def get_match(self,match_id,as_of,source=None):
        cutoff=utc(as_of);mid=self._canonical(match_id,cutoff)
        if mid is None:return None
        return next((r for r in self.list_matches(cutoff,source=source) if r['match_id']==mid),None)

    def get_evidence(self,match_id,as_of):
        cutoff=utc(as_of);mid=self._canonical(match_id,cutoff)
        if mid is None:return []
        rows=self.repo.conn.execute('''SELECT v.*,r.metadata FROM fm.match_versions v JOIN fm.raw_observations r ON r.id=v.observation_id
            WHERE v.match_id=%s AND v.observed_at<=%s AND v.ingested_at<=%s ORDER BY v.observed_at,v.ingested_at,v.id''',(mid,cutoff,cutoff)).fetchall()
        return [plain({'version_id':r['id'],'source':r['source'],'observation_id':r['observation_id'],
                       'observed_at':r['observed_at'],'ingested_at':r['ingested_at'],'available_at':max(r['observed_at'],r['ingested_at']),
                       'payload_path':r['metadata']['payload_path'],'content_hash':r['metadata']['content_hash'],
                       'record':r['record']}) for r in rows]

    def read_observation(self,observation_id,as_of):
        cutoff=utc(as_of)
        row=self.repo.conn.execute('SELECT metadata FROM fm.raw_observations WHERE id=%s AND observed_at<=%s AND ingested_at<=%s',
                                   (observation_id,cutoff,cutoff)).fetchone()
        if not row:return None
        obs=Observation.model_validate(row['metadata'])
        return obs,self.archive.read_body(obs)

    def get_features(self,match_id,as_of,feature_version='prematch-v1.1',information_mode='operational',**kwargs):
        cutoff=utc(as_of);mid=self._canonical(match_id,datetime.now(timezone.utc) if information_mode!='operational' else cutoff)
        if mid is None:raise ValueError('No identity mapping at cutoff')
        return self.store.get_features(mid,cutoff,feature_version,information_mode,**kwargs)
