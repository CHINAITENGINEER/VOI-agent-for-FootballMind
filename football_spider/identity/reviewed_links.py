"""Explicit evidence-reviewed cross-source links, not fuzzy name merging."""
from collections import Counter,defaultdict
import json
from football_spider.identity.resolver import stable_id
from football_spider.storage.raw_archive import json_bytes

# Reviewed against multiple independently dated official source observations.
# Each pair must still meet the evidence gate on every execution.
REVIEWED_EPL_TEAMS={'87':'25','73':'348','245':'56','78':'35','92':'46',
                    '89':'27','83':'19','77':'65','80':'24','88':'26'}

def apply_reviewed_epl_links(repo,candidates,understat_observation_id):
    candidates=json.loads(json_bytes(candidates))
    multiplicity=Counter(c['spftll_id'] for c in candidates)
    evidence=defaultdict(list)
    for row in candidates:
        if multiplicity[row['spftll_id']]!=1:continue
        for u,s in zip(row['understat_teams'],row['spftll_teams']):
            evidence[(str(u['id']),s['source_id'])].append(row)
    confirmed={};decisions=[]
    for uid,sid in REVIEWED_EPL_TEAMS.items():
        facts=evidence[(uid,sid)]
        if len({r['kickoff'] for r in facts})<2:raise ValueError('Reviewed team link lost multi-date evidence')
        opponents={str(u['id']) for r in facts for u in r['understat_teams'] if str(u['id'])!=uid}
        if len(opponents)<2:raise ValueError('Reviewed link needs multiple opponents')
        target=stable_id('spftll','team',sid)
        proof={'kind':'reviewed_cross_source_team_link','method':'same_league_kickoff_score_home_away_across_multiple_dates_and_opponents',
               'understat_observation_id':understat_observation_id,'understat_team_id':uid,'spftll_team_id':sid,
               'match_evidence':facts,'not_name_similarity':True}
        repo.identity.confirm('understat','team',uid,target,json_bytes(proof).decode())
        confirmed[uid]=sid;decisions.append(proof)
    # Unconfirmed team identities remain candidates, even when names look close.
    for (uid,sid),facts in evidence.items():
        if uid not in confirmed:
            repo.identity.candidate('understat','team',uid,stable_id('spftll','team',sid),
                'round2_single_fixture_support_requires_review',{'match_evidence':facts,'understat_observation_id':understat_observation_id})
    competitions={str(repo.conn.execute('SELECT competition_id FROM fm.match_versions WHERE id=%s',(r['spftll_version_id'],)).fetchone()['competition_id']) for r in candidates}
    if len(competitions)!=1:raise ValueError('Competition evidence conflict')
    repo.identity.confirm('understat','competition','EPL',next(iter(competitions)),
        json_bytes({'league':'Premier League / 英超','understat_observation_id':understat_observation_id,
                    'basis':'verified 20-team double round robin and reviewed overlapping fixtures','matches':len(candidates)}).decode())
    match_links=[]
    for row in candidates:
        pairs=list(zip(row['understat_teams'],row['spftll_teams']))
        if not all(confirmed.get(str(u['id']))==s['source_id'] for u,s in pairs):continue
        target=stable_id('spftll','match',row['spftll_id'])
        proof={'understat_observation_id':understat_observation_id,'evidence':row,
               'basis':'both_reviewed_team_ids_same_home_away_competition_kickoff_regulation_score'}
        repo.identity.confirm('understat','match',row['understat_id'],target,json_bytes(proof).decode())
        match_links.append(proof)
    return {'confirmed_teams':len(confirmed),'confirmed_matches':len(match_links),'team_decisions':decisions,'match_decisions':match_links,
            'mapping_effect':'subsequent_ingestions_only; no older fact is rewritten'}
