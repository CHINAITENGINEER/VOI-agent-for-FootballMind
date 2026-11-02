"""Understat league XHR normalization, never forecast or season-total features."""
from collections import Counter
from datetime import datetime, timezone
import math

from football_spider.models.contracts import EntityRef, Issue, MatchRecord, NormalizedBatch, Score, TeamStatistic

PARSER_VERSION = 'understat-1.0'
DEFINITION_VERSION = 'understat-match-xg-v1'
LEAGUES = {'EPL': ('Premier League','England',20)}


def audit_season(payload, league, season):
    if league not in LEAGUES:
        raise ValueError('League coverage not verified')
    if not isinstance(payload,dict) or not isinstance(payload.get('dates'),list) or not isinstance(payload.get('teams'),dict):
        raise ValueError('Understat league envelope changed')
    dates, teams = payload['dates'],payload['teams']
    ids=[str(row.get('id')) for row in dates]
    pairs=Counter((str(row.get('h',{}).get('id')),str(row.get('a',{}).get('id'))) for row in dates)
    appearances=Counter(t for pair in pairs for t in pair for _ in range(pairs[pair]))
    expected=LEAGUES[league][2]
    structural_complete=(len(teams)==expected and len(dates)==expected*(expected-1)
                         and len(set(ids))==len(ids) and len(pairs)==expected*(expected-1)
                         and all(h!=a and n==1 and h in teams and a in teams for (h,a),n in pairs.items())
                         and all(n==2*(expected-1) for n in appearances.values()))
    history_checks=0;history_mismatches=[]
    for row in dates:
        for side,other,venue in [('h','a','h'),('a','h','a')]:
            tid=str(row.get(side,{}).get('id'))
            history=[h for h in teams.get(tid,{}).get('history',[]) if h.get('date')==row.get('datetime') and h.get('h_a')==venue]
            if not row.get('isResult'):continue
            if len(history)!=1:
                history_mismatches.append({'match_id':str(row.get('id')),'team_id':tid,'reason':'history_date_venue_nonunique'})
                continue
            h=history[0]
            for field,raw in [('xG',row.get('xG',{}).get(side)),('xGA',row.get('xG',{}).get(other)),
                              ('scored',row.get('goals',{}).get(side)),('missed',row.get('goals',{}).get(other))]:
                try:equal=math.isclose(float(h[field]),float(raw),rel_tol=1e-6,abs_tol=1e-6)
                except (KeyError,TypeError,ValueError):equal=False
                history_checks+=1
                if not equal:history_mismatches.append({'match_id':str(row.get('id')),'team_id':tid,'field':field})
    return {'league':league,'season':str(season),'matches':len(dates),'unique_ids':len(set(ids)),
            'duplicate_ids':[key for key,n in Counter(ids).items() if n>1], 'teams':len(teams),
            'expected_matches':expected*(expected-1),'ordered_team_pairs':len(pairs),
            'team_appearances':dict(appearances),'structural_complete':structural_complete,
            'player_aggregate_rows':len(payload.get('players',[])),
            'history_value_checks':history_checks,'history_mismatches':history_mismatches,
            'first_kickoff_raw':min((r.get('datetime','') for r in dates),default=None),
            'last_kickoff_raw':max((r.get('datetime','') for r in dates),default=None),
            'coverage_basis':'20-team double round robin, distinct home-away pairs and 38 appearances per team; not external score correctness'}


def normalize_understat(obs,payload):
    league=obs.context.get('league')
    season=int(obs.context['season'])
    if obs.context.get('kickoff_timezone')!='UTC':
        raise ValueError('Understat timezone evidence or explicit policy is required')
    audit=audit_season(payload,league,season)
    batch=NormalizedBatch(parser_version=PARSER_VERSION)
    for mismatch in audit['history_mismatches']:
        batch.issues.append(Issue(rule='UNDERSTAT_HISTORY_INCONSISTENT',severity='warning',record_key=mismatch['match_id'],
                                 message='Dates and team history disagree; match dates retained as primary',evidence=mismatch))
    bad_ids=set(audit['duplicate_ids'])
    league_name,country,_=LEAGUES[league]
    for row in payload['dates']:
        mid=str(row.get('id'))
        try:
            if mid in bad_ids:
                raise ValueError('DUPLICATE_SOURCE_MATCH_ID')
            kickoff=datetime.fromisoformat(row['datetime'])
            if kickoff.tzinfo is not None:raise ValueError('SOURCE_TIME_FORMAT_CHANGED')
            kickoff=kickoff.replace(tzinfo=timezone.utc)
            if kickoff.year not in (season,season+1):
                raise ValueError('SEASON_DATE_MISMATCH')
            result=row['isResult']
            if type(result) is not bool:
                raise ValueError('INVALID_RESULT_FLAG')
            refs={side:EntityRef(source_id=str(row[side]['id']),name=row[side]['title'],country=country,category='men_senior') for side in ('h','a')}
            for side in ('h','a'):
                if refs[side].source_id not in payload['teams']:
                    raise ValueError('UNKNOWN_TEAM_REFERENCE')
            scores={}
            for side in ('h','a'):
                raw=row.get('goals',{}).get(side)
                if raw is None:
                    scores[side]=None
                elif isinstance(raw,bool) or float(raw)!=int(float(raw)):
                    raise ValueError('NON_INTEGRAL_SCORE')
                else:
                    scores[side]=int(raw)
            if result and None in scores.values():
                raise ValueError('FINISHED_MISSING_SCORE')
            record=MatchRecord(source_id=mid,competition=EntityRef(source_id=league,name=league_name,country=country,category='men_senior'),
                season=str(season),home=refs['h'],away=refs['a'],kickoff=kickoff,status='finished' if result else 'scheduled',
                score_period='regulation',score=Score(home=scores['h'],away=scores['a']),
                original_time=row['datetime'],source_timezone='UTC',
                coverage='verified_complete' if audit['structural_complete'] else 'incomplete_season',
                missing={'half_score':'source_not_supplied','source_published_at':'not_supplied','lineups':'not_supplied'},
                evidence={'source_path':f'dates[id={mid}]','league':league,'season_start_year':season,
                          'timezone_policy':obs.context.get('timezone_policy'),
                          'forecast_excluded':True,'season_aggregates_excluded':True,
                          'coverage_basis':audit['coverage_basis']})
            pending=[]
            if result:
                for side,other in [('h','a'),('a','h')]:
                    for metric,xside in [('xg',side),('xga',other)]:
                        raw=row.get('xG',{}).get(xside)
                        if raw is None:
                            batch.issues.append(Issue(rule='MISSING_XG',severity='warning',record_key=mid,message=f'{side}:{metric} absent'))
                            continue
                        if isinstance(raw,bool):
                            raise ValueError('INVALID_XG')
                        value=float(raw)
                        if not math.isfinite(value) or value<0:
                            raise ValueError('INVALID_XG')
                        pending.append(TeamStatistic(source_match_id=mid,source_team_id=refs[side].source_id,
                            metric=metric,value=value,provider='understat',definition_version=DEFINITION_VERSION,
                            evidence={'source_path':f'dates[id={mid}].xG.{xside}',
                                      'xga_definition':'opponent_match_xg','published_at_known':False}))
            batch.matches.append(record)
            batch.statistics.extend(pending)
        except (KeyError,ValueError,TypeError,OverflowError) as exc:
            batch.issues.append(Issue(rule='UNDERSTAT_RECORD_INVALID',severity='error',record_key=mid,
                                     message=type(exc).__name__,evidence={'reason':str(exc)[:300]}))
    return batch
