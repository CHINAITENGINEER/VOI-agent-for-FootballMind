"""Raw-backed official league results for the registered 2026 Finnish pair.

The official championship schedule supplies native event identity and kickoff.
The match follow-up supplies its played state, ordered final scores and the
observed two-half timeline. Availability is the real receipt/verification clock;
no final-whistle instant is fabricated from a calendar or an elapsed-time rule.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
from hashlib import sha256
import json
from pathlib import Path
import re
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup

from football_research.acquisition.state import parse_time
from football_spider.storage.raw_archive import RawArchive
from .finland_fixture import VERSION as IDENTITY_VERSION, _digest
from .settlement import export_results
from .types import ResearchDataset, ResearchMatch

VERSION = 'finland-independent-regulation-settlement-v1'
PARSER_VERSION = 'veikkausliiga-official-finished-two-halves-v1'
RULE_VERSION = 'veikkausliiga-2026-mestaruussarja-observed-two-halves-v1'
GROUP3_RULE_VERSION = 'veikkausliiga-2026-karsintasarja-observed-two-halves-v1'
RESULTS_URL = 'https://www.veikkausliiga.com/tilastot/2026/veikkausliiga/ottelut/'
RESULTS_PARAMS = {'group': '2'}
FORMAT_URL = 'https://www.veikkausliiga.com/veikkausliiga/sarjajarjestelma'
LAW7_URL = 'https://www.theifab.com/laws/latest/the-duration-of-the-match/'
LAW10_URL = 'https://www.theifab.com/laws/latest/determining-the-outcome-of-a-match/'
RULE_URLS = (FORMAT_URL, LAW7_URL, LAW10_URL)
RULE_PACKAGE_ROOT = Path(__file__).resolve().parents[2] / 'config' / 'source-contracts' / 'finland-2026'
RULE_PACKAGE_MANIFEST_SHA256 = '8af2fa02baba42f61af72007887610699e87898d77ba32cd4daf62ac40d3e827'
OFFICIAL_IDENTITY_VERSION = 'official-fixture-evidence-identity-v1'
_MATCH_URL = re.compile(re.escape(RESULTS_URL) + r'([1-9][0-9]*)/seuranta/')
_PHASES = ('Ottelu päättyi', 'Puoliaika päättyi', 'Puoliaika alkoi',
           'Puoliaika päättyi', 'Puoliaika alkoi', 'Ottelu alkoi')


def _group(value):
    if type(value) is not str or value not in {'2', '3'}:
        raise ValueError('The registered Finnish schedule group must be explicit string 2 or 3')
    return value


def _stage(schedule_group):
    return 'Mestaruussarja' if _group(schedule_group) == '2' else 'Karsintasarja'


def _rule_version(schedule_group):
    return RULE_VERSION if _group(schedule_group) == '2' else GROUP3_RULE_VERSION


def result_schedule_group(binding):
    """Select only the stage explicitly bound by the original fixture identity."""
    group = _group(binding.get('schedule_group', '2'))
    if group == '3' and binding.get('schema') != OFFICIAL_IDENTITY_VERSION:
        raise ValueError('Group 3 settlement requires an explicit official identity scope')
    if binding['canonical'].get('stage') != _stage(group):
        raise ValueError('Official identity schedule group and stage scope disagree')
    return group


def _text(node):
    return ' '.join(node.get_text(' ', strip=True).split())


def _answer(status, reason, **values):
    return {'status': status, 'reason': reason, 'parser_version': PARSER_VERSION, **values}


def _reference(observation):
    return {'observation_id': observation.id, 'raw_sha256': observation.content_hash,
            'url': observation.endpoint, 'params': deepcopy(observation.params),
            'observed_at': observation.observed_at.isoformat()}


def _receipt(observation, body, *, as_of):
    if sha256(body).hexdigest() != observation.content_hash:
        raise ValueError('Official raw checksum mismatch')
    if observation.observed_at > parse_time(as_of):
        raise ValueError('Official observation was unavailable at verification')
    requested = observation.context.get('requested_at')
    if requested is None or parse_time(requested) > observation.observed_at:
        raise ValueError('Official request and actual receipt clocks are required')
    if observation.http_status != 200 or observation.error_class:
        return 'official_response_unavailable'
    if observation.source != 'open_page' or len(body) > 2_000_000 or observation.context.get('response_complete') is False:
        return 'official_response_incomplete_or_unsupported'
    return None


def _rule_projection(observation, body, *, as_of, schedule_group='2'):
    schedule_group = _group(schedule_group)
    if observation.endpoint not in RULE_URLS:
        return _answer('IGNORED', 'not_registered_rule_source')
    problem = _receipt(observation, body, as_of=as_of)
    if problem or observation.params:
        return _answer('UNCONFIRMED', problem or 'unexpected_rule_parameters')
    soup = BeautifulSoup(body, 'html.parser')
    main = soup.find('main')
    if main is None:
        return _answer('UNCONFIRMED', 'official_rule_body_missing')
    for node in main.select('script, style, nav, footer'):
        node.decompose()
    text = _text(main)
    if observation.endpoint == FORMAT_URL:
        checks = (r'Veikkausliiga pelataan kaudella 2026 seuraavasti:',
                  r'Kuuden joukkueen kaksinkertainen M\s*estaruussarja \(30 ottelua\)',
                  r'Karsinnat eivät ole osa Veikkausliigaa',
                  r'runkosarjassa saadut pisteet ja maalit seuraavat mukana')
        meaning = '2026_double_round_robin_championship_distinct_from_relegation_playoffs'
        if schedule_group == '3':
            # The league's six-team lower stage is not the later two-legged
            # Palloliitto playoff. Re-project those exact archived statements;
            # the immutable championship package is not silently re-labelled.
            checks = (r'Veikkausliiga pelataan kaudella 2026 seuraavasti:',
                      r'Kahdentoista joukkueen kaksinkertainen runkosarja \(132 ottelua\)',
                      r'Kuuden joukkueen kaksinkertainen M\s*estaruussarja \(30 ottelua\)',
                      r'Kuuden joukkueen yksinkertainen Karsintasarja \(15 ottelua\)',
                      r'Yhteensä Veikkausliigassa pelataan 177 ottelua',
                      r'Lisäksi pelataan kaksi ottelua Veikkausliigan karsintoja',
                      r'Karsinnat eivät ole osa Veikkausliigaa, vaan niissä kilpailujärjestäjänä toimii Palloliitto',
                      r'runkosarjassa saadut pisteet ja maalit seuraavat mukana')
            meaning = '2026_single_round_robin_karsintasarja_within_177_league_matches_distinct_from_relegation_playoffs'
    elif observation.endpoint == LAW7_URL:
        checks = (r'Law 7', r'two equal halves of 45 minutes',
                  r'Allowance is made by the referee in each half',
                  r'The additional time may be increased by the referee but not reduced')
        meaning = 'two_45_minute_halves_with_referee_added_time'
    else:
        checks = (r'Law 10', r'If both teams score no goals or an equal number of goals, the match is drawn',
                  r'two equal periods of extra time not exceeding 15 minutes each',
                  r'Penalties \(penalty shoot-out\) are taken after the match has ended')
        meaning = 'draws_extra_time_and_penalty_shootout_are_distinct_outcomes'
    if not all(re.search(pattern, text) for pattern in checks):
        return _answer('UNCONFIRMED', 'registered_rule_semantics_not_reproduced')
    return _answer('RULE', 'official_rule_semantics_observed', meaning=meaning)


def verify_rule_package(*, as_of, package_root=None, schedule_group='2'):
    """Verify the portable immutable rule snapshot without any network access."""
    schedule_group = _group(schedule_group)
    root = Path(package_root or RULE_PACKAGE_ROOT).resolve()
    raw = (root/'manifest.json').read_bytes()
    if sha256(raw).hexdigest() != RULE_PACKAGE_MANIFEST_SHA256:
        raise ValueError('Finland rule package manifest checksum mismatch')
    manifest = json.loads(raw)
    archive = RawArchive(root)
    all_items = archive.observations()
    by_id = {item.id:item for item in all_items}
    declared = {row['observation_id'] for row in manifest['observations']}
    if len(by_id) != len(all_items) or set(by_id) != declared:
        raise ValueError('Finland rule package observations changed')
    unavailable = []
    for reference in manifest['observations']:
        for key in ('metadata', 'payload'):
            path = (root/reference[key+'_path']).resolve()
            if not path.is_relative_to(root) or sha256(path.read_bytes()).hexdigest() != reference[key+'_sha256']:
                raise ValueError('Finland rule package file checksum mismatch')
        observation = by_id[reference['observation_id']]
        body = archive.read_body(observation)
        if (observation.endpoint != reference['url'] or observation.content_hash != reference['raw_sha256']
                or observation.observed_at.isoformat() != reference['observed_at']):
            raise ValueError('Finland rule snapshot provenance changed')
        if observation.observed_at > parse_time(as_of):
            unavailable.append(observation.id)
        elif _rule_projection(observation, body, as_of=as_of, schedule_group=schedule_group)['status'] != 'RULE':
            raise ValueError('Finland rule snapshot semantics no longer reproduce')
    return {'status':'PENDING' if unavailable else 'VERIFIED', 'rule_version':_rule_version(schedule_group),
        'reason':'rule_snapshot_not_yet_available' if unavailable else 'raw_rule_snapshot_reverified',
        'archive_root':str(root), 'manifest_sha256':RULE_PACKAGE_MANIFEST_SHA256,
        'observation_ids':[] if unavailable else sorted(declared),
        'not_available_observation_ids':sorted(unavailable),
        'available_at':max(row['observed_at'] for row in manifest['observations'])}


def copy_rule_observations(destination, *, as_of, schedule_group='2'):
    """Copy verified original receipt identities and clocks, charging no HTTP."""
    from .prospective_text import _copy_observations
    check = verify_rule_package(as_of=as_of, schedule_group=schedule_group)
    if check['status'] != 'VERIFIED':
        return {**check, 'copied_observation_ids':[], 'http_requests':0}
    source = RawArchive(RULE_PACKAGE_ROOT)
    target = destination if isinstance(destination, RawArchive) else RawArchive(Path(destination))
    _copy_observations(source, target, observation_ids=check['observation_ids'])
    return {**check, 'copied_observation_ids':check['observation_ids'], 'http_requests':0}


def _event_id(canonical):
    native = re.fullmatch(r'(?:palloliitto:match|veikkausliiga:2026:match):([1-9][0-9]*)',
                          canonical.get('source_event_id', ''))
    if native is None:
        raise ValueError('A verified native Finnish event ID is required')
    return native.group(1)


def match_followup_url(binding):
    """Return the observed official URL pattern for a bound native event."""
    if (binding.get('schema') not in {IDENTITY_VERSION, 'finland-stage-fixture-identity-v1', OFFICIAL_IDENTITY_VERSION}
            or binding.get('status') != 'CONFIRMED'):
        raise ValueError('A confirmed Finland identity binding is required')
    group = result_schedule_group(binding)
    if binding.get('schema') == OFFICIAL_IDENTITY_VERSION:
        canonical = binding['canonical']
        if (binding.get('match_id') != canonical.get('source_event_id')
                or any(canonical.get(k) != v for k,v in {'season':'2026','competition':'Veikkausliiga',
                    'category':'senior_men_club','stage':_stage(group)}.items())):
            raise ValueError('Official identity must preserve its native Finnish event and competition scope')
    return RESULTS_URL + _event_id(binding['canonical']) + '/seuranta/'


def project_championship_schedule(observation, body, canonical, *, as_of, schedule_group='2'):
    """Require exact native event, group, ordered teams and explicit local clock."""
    schedule_group = _group(schedule_group)
    stage = _stage(schedule_group)
    name = 'championship' if schedule_group == '2' else 'karsintasarja'
    if observation.endpoint != RESULTS_URL or observation.params != {'group': schedule_group}:
        return _answer('IGNORED', 'not_registered_' + name + '_schedule')
    if canonical.get('stage') != stage:
        return _answer('UNCONFIRMED', 'canonical_schedule_stage_conflict')
    problem = _receipt(observation, body, as_of=as_of)
    if problem:
        return _answer('UNCONFIRMED', problem)
    soup = BeautifulSoup(body, 'html.parser')
    for selector, value, title in [('select#season option[selected]', '2026', '2026'),
                                   ('select#category option[selected]', 'veikkausliiga', 'Veikkausliiga'),
                                   ('select[name=group] option[selected]', schedule_group, stage)]:
        nodes = soup.select(selector)
        if len(nodes) != 1 or nodes[0].get('value') != value or _text(nodes[0]) != title:
            return _answer('UNCONFIRMED', 'official_schedule_season_competition_or_stage_missing')
    native = _event_id(canonical)
    path = '/tilastot/2026/veikkausliiga/ottelut/' + native + '/'
    targets = [a for a in soup.find_all('a', href=True) if a.get('href') == path]
    if len(targets) != 1 or targets[0].find_parent('tr') is None:
        return _answer('UNCONFIRMED', 'native_schedule_event_missing_or_ambiguous')
    row = targets[0].find_parent('tr')
    cells = row.find_all('td', recursive=False)
    local = parse_time(canonical['kickoff']).astimezone(ZoneInfo('Europe/Helsinki'))
    if (len(cells) != 8 or row.get('data-time') != local.strftime('%Y%m%d')
            or _text(cells[2]) != local.strftime('%H:%M')
            or _text(targets[0]) != canonical['home_name'] + ' - ' + canonical['away_name']):
        return _answer('DISPUTED', 'native_schedule_order_or_kickoff_conflict')
    link = path + 'seuranta/'
    if not any(a.get('href') == link for a in cells[5].find_all('a', href=True)):
        return _answer('UNCONFIRMED', 'observed_native_followup_link_missing')
    shared = {'source_event_id': canonical['source_event_id'], 'kickoff': canonical['kickoff'],
              'followup_url': urljoin(RESULTS_URL, link), 'stage': stage, 'season': '2026'}
    score = _text(cells[6])
    if score in ('', '-', '–', '—'):
        return _answer('PENDING', 'official_' + name + '_score_not_available', **shared)
    pair = re.fullmatch(r'(0|[1-9][0-9]*)\s*[-–—]\s*(0|[1-9][0-9]*)', score)
    if pair is None:
        return _answer('UNCONFIRMED', 'official_score_not_plain_ordered_integers', **shared)
    if observation.observed_at < parse_time(canonical['kickoff']) + timedelta(minutes=90):
        return _answer('DISPUTED', 'score_receipt_precedes_possible_regulation_finish', **shared)
    return _answer('SCORE', 'native_' + name + '_ordered_score_observed',
                   home_goals=int(pair.group(1)), away_goals=int(pair.group(2)), **shared)


def project_league_match(observation, body, canonical, *, as_of):
    """Project an actual follow-up page; this alone is not a training label."""
    match = _MATCH_URL.fullmatch(observation.endpoint)
    if match is None or match.group(1) != _event_id(canonical):
        return _answer('IGNORED', 'not_target_native_league_followup')
    problem = _receipt(observation, body, as_of=as_of)
    if problem or observation.params:
        return _answer('UNCONFIRMED', problem or 'unexpected_match_parameters')
    soup = BeautifulSoup(body, 'html.parser')
    links = soup.select('meta[property="og:url"]')
    heads = soup.select('main .game-header .head')
    if len(links) != 1 or links[0].get('content') != observation.endpoint:
        return _answer('DISPUTED', 'embedded_native_match_url_conflict')
    if len(heads) != 1:
        return _answer('UNCONFIRMED', 'official_match_header_missing_or_ambiguous')
    head = heads[0]
    for side in ('home', 'away'):
        names = head.select('.' + side + ' .team-name a')
        if len(names) != 1 or _text(names[0]) != canonical[side+'_name']:
            return _answer('DISPUTED', 'official_followup_ordered_teams_conflict')
    if 'finished' not in head.get('class', []):
        return _answer('PENDING', 'official_finished_state_not_observed')
    local = parse_time(canonical['kickoff']).astimezone(ZoneInfo('Europe/Helsinki'))
    states = head.select('.game-state')
    if not states or any(_text(node) != 'Pelattu ' + local.strftime('%d.%m.%Y') for node in states):
        return _answer('DISPUTED', 'official_played_date_or_state_conflict')
    if observation.observed_at < parse_time(canonical['kickoff']) + timedelta(minutes=90):
        return _answer('DISPUTED', 'finished_receipt_precedes_possible_regulation_finish')
    score = []
    for side in ('home', 'away'):
        nodes = head.select('#' + side + '-goals')
        if len(nodes) != 1 or re.fullmatch(r'0|[1-9][0-9]*', _text(nodes[0])) is None:
            return _answer('UNCONFIRMED', 'explicit_final_integer_score_missing')
        score.append(int(_text(nodes[0])))
    panels = soup.select('main .game-events')
    if not panels:
        return _answer('UNCONFIRMED', 'finished_match_period_evidence_missing')
    for panel in panels:
        phases = tuple(_text(p) for p in panel.select('p.info.center'))
        if phases != _PHASES:
            return _answer('UNCONFIRMED', 'complete_two_half_terminal_sequence_not_observed')
        for node in panel.select('.regular-event.time'):
            event_time = re.fullmatch(r'([0-9]{1,3})\s*\x27(?:\s*\+\s*([0-9]+))?', _text(node))
            if event_time is None:
                return _answer('UNCONFIRMED', 'unrecognised_event_period_clock')
            minute, added = int(event_time.group(1)), event_time.group(2)
            if minute > 90 or added is not None and minute not in (45, 90):
                return _answer('DISPUTED', 'nonregulation_event_period_detected')
    return _answer('FINISHED_SCORE', 'official_finished_score_and_two_regulation_halves_agree',
        home_goals=score[0], away_goals=score[1], score_period='regulation',
        full_time_at=None, finished_available_at=observation.observed_at.isoformat(),
        clock_basis='actual_finished_response_receipt_not_imputed_final_whistle',
        terminal_sequence=list(_PHASES), event_id=canonical['source_event_id'])


def verify_finland_settlement(binding, *, identity_archive_root, result_archive_root,
                              observation_ids, as_of):
    """Re-read official identity, rule snapshots, group and terminal result bytes."""
    from .verified_fixtures import verify_identity_binding
    as_of = parse_time(as_of)
    identity = verify_identity_binding(binding, archive_root=identity_archive_root, as_of=as_of)
    report = {'schema': VERSION, 'match_id': binding['match_id'],
        'identity_binding_sha256': binding['binding_sha256'], 'status': 'UNCONFIRMED',
        'reason': None, 'verified_at': as_of.isoformat(), 'observations': [], 'projections': [],
        'ignored': [], 'settlement_rows': [], 'regulation_rule': None}

    def finish(status, reason):
        report.update(status=status, reason=reason)
        report['report_sha256'] = _digest(report)
        return report

    if identity['status'] != 'CONFIRMED':
        return finish('UNCONFIRMED', 'fixture_identity_not_confirmed')
    canonical = identity['canonical']
    group = result_schedule_group(identity)
    stage = _stage(group)
    rule_version = _rule_version(group)
    if any(canonical.get(k) != v for k, v in {'season':'2026', 'competition':'Veikkausliiga',
            'stage':stage, 'category':'senior_men_club'}.items()):
        return finish('UNCONFIRMED', 'registered_regulation_scope_not_met')
    if not observation_ids:
        return finish('PENDING', 'independent_result_observations_missing')
    if (not isinstance(observation_ids, (list, tuple)) or len(set(observation_ids)) != len(observation_ids)
            or any(not isinstance(x, str) or not x for x in observation_ids)):
        raise ValueError('Distinct explicit result observations are required')
    archive = RawArchive(Path(result_archive_root))
    items = archive.observations()
    by_id = {item.id:item for item in items}
    if len(by_id) != len(items) or any(ident not in by_id for ident in observation_ids):
        raise ValueError('Result observations missing or ambiguous')
    rules, schedules, matches = {}, [], []
    package = verify_rule_package(as_of=as_of, schedule_group=group)
    package_observations = {o.id:o for o in RawArchive(RULE_PACKAGE_ROOT).observations()}
    for ident in sorted(observation_ids):
        observation = by_id[ident]
        reference = _reference(observation)
        body = archive.read_body(observation)
        if observation.observed_at > as_of:
            raise ValueError('Result observation was unavailable at verification')
        if observation.endpoint in RULE_URLS:
            expected = package_observations.get(observation.id)
            if (package['status'] != 'VERIFIED' or expected is None
                    or observation.model_dump() != expected.model_dump()):
                projection = _answer('UNCONFIRMED', 'rule_observation_not_available_or_not_registered_snapshot')
            else:
                projection = _rule_projection(observation, body, as_of=as_of, schedule_group=group)
            if projection['status'] == 'RULE':
                rules.setdefault(observation.endpoint, []).append({**reference, **projection})
        elif observation.endpoint == RESULTS_URL and observation.params == {'group': group}:
            projection = project_championship_schedule(observation, body, canonical, as_of=as_of,
                                                       schedule_group=group)
            schedules.append({**reference, **projection})
        else:
            projection = project_league_match(observation, body, canonical, as_of=as_of)
            if projection['status'] != 'IGNORED':
                matches.append({**reference, **projection})
        if projection['status'] == 'IGNORED':
            report['ignored'].append({**reference, 'reason':projection['reason']})
        else:
            report['observations'].append(reference)
            report['projections'].append({**reference, **projection})
    if any(row['status'] == 'DISPUTED' for row in schedules + matches):
        return finish('DISPUTED', 'official_target_identity_time_or_period_conflict')
    if all(url in rules for url in RULE_URLS):
        rule_inputs = [min(rules[url], key=lambda row:(row['observed_at'],row['observation_id'])) for url in RULE_URLS]
        report['regulation_rule'] = {'version':rule_version, 'period':'regulation',
            'meaning':'90_minutes_plus_stoppage_excludes_extra_time_penalties',
            'scope':'2026_senior_men_veikkausliiga_' + stage.lower() + '_observed_two_halves',
            'basis':'official_2026_format_and_ifab_laws7_10_plus_match_terminal_period_evidence',
            'references':[{key:row[key] for key in ('observation_id','url','raw_sha256','observed_at')} for row in rule_inputs],
            'available_at':max(row['observed_at'] for row in rule_inputs)}
    else:
        report['missing_rule_sources'] = [url for url in RULE_URLS if url not in rules]
        return finish('PENDING', 'independent_regulation_rule_bytes_missing_or_unrecognised')
    usable_matches = [row for row in matches if row['status'] in ('FINISHED_SCORE', 'PENDING')]
    finals = [row for row in usable_matches if row['status']=='FINISHED_SCORE']
    if not finals:
        return finish('PENDING', 'target_finished_two_half_result_not_observed')
    newest = max(row['observed_at'] for row in usable_matches)
    if any(row['observed_at']==newest and row['status']!='FINISHED_SCORE' for row in usable_matches):
        return finish('DISPUTED', 'finished_result_reverted_without_resolution')
    scores = [row for row in schedules if row['status']=='SCORE']
    if not scores:
        name = 'championship' if group == '2' else 'karsintasarja'
        return finish('UNCONFIRMED', 'native_' + name + '_score_not_observed')
    if len({(row['home_goals'],row['away_goals']) for row in finals+scores}) != 1:
        return finish('DISPUTED', 'official_schedule_and_terminal_score_revisions_disagree')
    final = min(finals, key=lambda row:(row['observed_at'],row['observation_id']))
    score = min(scores, key=lambda row:(row['observed_at'],row['observation_id']))
    used = sorted([final,score]+rule_inputs, key=lambda row:row['observation_id'])
    manifest = tuple('observation:'+row['observation_id']+'|sha256:'+row['raw_sha256']+'|at:'+row['observed_at'] for row in used)
    if binding['schema'] == OFFICIAL_IDENTITY_VERSION:
        # This evidence-only entry has no quote or initial forecasting manifest.
        # Its verified canonical entities supply the result projection directly.
        if binding['match_id'] != canonical['source_event_id']:
            raise ValueError('Official result identity must be its verified native event')
        fixture = {'match_id':binding['match_id'],
            'competition_id':'veikkausliiga:competition:2026', 'season':canonical['season'],
            'home_team_id':canonical['home_id'], 'away_team_id':canonical['away_id']}
    else:
        fixture = binding['initialization_manifest']['market_initialization']['fixture']
    received = max(parse_time(row['observed_at']) for row in used)
    result = ResearchMatch(match_id=fixture['match_id'], competition_id=fixture['competition_id'],
        season=fixture['season'], home_team_id=fixture['home_team_id'], away_team_id=fixture['away_team_id'],
        kickoff=parse_time(canonical['kickoff']), home_goals=final['home_goals'], away_goals=final['away_goals'],
        result_available_at=as_of, fixture_available_at=parse_time(binding['verified_at']),
        source='veikkausliiga_official_independent', input_manifest=manifest, score_period='regulation',
        status='finished', time_basis='official_finished_receipt_and_verification_no_imputed_final_whistle',
        competition_name=canonical['competition'], home_team_name=canonical['home_name'],
        away_team_name=canonical['away_name'], observed_at=received, ingested_at=as_of, operational_available_at=as_of)
    dataset = ResearchDataset((result,), (), VERSION+':'+_digest({'identity':binding['binding_sha256'],
        'inputs':manifest, 'rule':report['regulation_rule']}), manifest, {})
    rows, audit = export_results(dataset)
    if len(rows)!=1:
        report['export_audit']=audit
        return finish('UNCONFIRMED', 'settlement_export_rejected_clocks_or_semantics')
    rows[0].update(identity_binding_sha256=binding['binding_sha256'], regulation_rule_version=rule_version,
                   verification_schema=VERSION)
    report['settlement_rows']=rows
    return finish('VERIFIED', 'official_native_group_identity_terminal_two_halves_and_scores_agree')


__all__ = ['VERSION', 'PARSER_VERSION', 'RULE_VERSION', 'GROUP3_RULE_VERSION', 'RULE_URLS', 'RESULTS_URL', 'RESULTS_PARAMS',
           'RULE_PACKAGE_ROOT', 'verify_rule_package', 'copy_rule_observations',
           'result_schedule_group', 'match_followup_url', 'project_championship_schedule', 'project_league_match',
           'verify_finland_settlement']
