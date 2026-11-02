"""Constructed raw-backed role f/B/g/C mechanism; never a real effect claim.

The only substitutions are a declared simulated clock and source transport.
No predictor, query-value estimate, signed target or policy choice is patched.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import importlib
import json
import math
from pathlib import Path
import socket

import pytest

from football_agent.news_discovery import NewsDiscoveryTools
from football_agent.learned_voi_replay import replay_saved_run
from football_research.acquisition.visible_state import encode_visible_state
from football_research.data_adapters.official_context_registration import prepare_official_context_registration
from football_research.data_adapters.official_context_collection import collect_official_context
from football_research.data_adapters.official_fixture import prepare_official_fixture
from football_research.data_adapters.finland_result_capture import capture_finland_result
from football_research.data_adapters.prospective_learning import FACTS
from football_spider.storage.raw_archive import RawArchive, digest
from test_official_context_gain import Clock, OfficialSource, read, save
from test_context_gain import exposure

ROOT = Path(__file__).resolve().parents[3]
UTC = timezone.utc
ARTICLE = 'https://www.veikkausliiga.com/uutiset/2026/10/10/hjk-selection-update/'
UPDATE = 'HJK striker is available for selection after recovery and full training.'
BALANCED = 'HJK and VPS prepare balanced squads for their evenly matched meeting.'
SENTINEL = 'UNREAD_ROLE_BODY_SENTINEL'


def install_role_simulation(patch):
    def deny(*args, **kwargs):
        raise AssertionError('Constructed role chain must never access the network')
    for owner, key in ((socket, 'getaddrinfo'), (socket, 'create_connection'),
                       (socket.socket, 'connect'), (socket.socket, 'connect_ex')):
        patch.setattr(owner, key, deny)
    for name in ('football_agent.learned_voi_runtime', 'football_agent.inquiry_tools',
                 'football_research.acquisition.role_context_predictor',
                 'football_research.acquisition.role_context_value',
                 'football_research.data_adapters.official_context_registration',
                 'football_research.data_adapters.official_context_collection',
                 'football_research.data_adapters.official_context_gain',
                 'football_research.data_adapters.settlement'):
        module = importlib.import_module(name)
        if hasattr(module, 'datetime'):
            patch.setattr(module, 'datetime', Clock)
    calls = []

    def execute(self, prepared, *, run_id, request_id):
        calls.append(deepcopy(prepared))
        if prepared['tool'] == 'search_web':
            body = b'<rss><channel></channel></rss>'
        elif prepared['endpoint'] == ARTICLE:
            body = ('<html><script type="application/ld+json">'+json.dumps({
                '@type': 'NewsArticle', 'headline': 'HJK selection update',
                'datePublished': '2026-10-10T11:00:00+00:00',
                'articleBody': UPDATE+' '+SENTINEL})+'</script></html>').encode()
        else:
            body = (f'<html><main>Official news.<a href="{ARTICLE}">{UPDATE}</a>'
                    '</main></html>').encode()
        at = Clock.now(UTC)
        observation = self.archive.save(prepared['source'], prepared['endpoint'], prepared['params'], body,
            status=200, observed_at=at, context={'requested_at': at.isoformat(), 'inquiry_run_id': run_id})
        return {'document': self.project(observation, prepared),
            'observation': observation.model_dump(mode='json'), 'http_attempts': 1,
            'money': None, 'money_status': 'not_measured', 'latency_seconds': .01}

    patch.setattr(NewsDiscoveryTools, 'execute', execute)
    return calls


def prepare_source(folder, index, day, *, outcome=0, swap=False, teams=None):
    source = OfficialSource(index, day, outcome)
    if teams:
        source.home, source.away = teams
    if swap:
        source.home, source.away = source.away, source.home
    result = prepare_official_fixture('veikkausliiga:2026:match:'+source.native,
        output=folder/'prepare', clock=lambda: Clock.now(UTC), transport=source)
    assert result['status'] == 'ready', result
    return source, result, read(result['identity_path'])


def raw_A_records(root, *, identity_offset=0):
    """Constructed A views are backed by official identity, body and score raws.

    These are constructed before/after snapshots, not an observed acquisition
    policy trajectory. Its class pattern is part of the declared mechanism.
    """
    from football_research.acquisition.fixture_roles import bind_fixture_roles
    from football_research.data_adapters.fixture_role_identity import resolve_fixture_role_identity

    sources, records, bindings = [], [], {}
    for index, outcome in enumerate([0]*4+[1]*4+[2]*4):
        folder = root/'A'/f'{index:02}'
        source, prepared, binding = prepare_source(folder, identity_offset+2000+index, 8+index%2,
            outcome=outcome, swap=outcome == 2)
        physical = binding['canonical']['source_event_id']
        bindings[physical] = binding
        resolver = resolve_fixture_role_identity(binding, archive_root=prepared['identity_archive'], as_of=Clock.now(UTC))
        at = Clock.now(UTC)
        body = BALANCED if outcome == 1 else UPDATE
        archive = RawArchive(folder/'news-archive')
        url = f'https://www.veikkausliiga.com/uutiset/2026/10/07/constructed-A-{index}/'
        html = ('<html><script type="application/ld+json">'+json.dumps({
            '@type': 'NewsArticle', 'headline': 'Observed squad update',
            'datePublished': '2026-10-07T11:00:00+00:00', 'articleBody': body})+'</script></html>').encode()
        obs = archive.save('open_page', url, {}, html, status=200, observed_at=at,
                           context={'requested_at': at.isoformat()})
        news = NewsDiscoveryTools(archive, ['www.veikkausliiga.com'], projection_version=6)
        prepared_news = news.prepare({'tool': 'read_url', 'url': url},
                                     {'task': {'seed_urls': [url]}, 'documents': []})
        document = news.project(obs, prepared_news)
        public = {'state': {'match_id': binding['match_id'], 'prediction_cutoff': binding['prediction_cutoff'],
            'base': {'home_team_id': binding['canonical']['home_id'],
                'away_team_id': binding['canonical']['away_id'],
                'competition_id': binding['canonical']['competition'], 'kickoff': binding['canonical']['kickoff']},
            'action_history': []}, 'decision_at': Clock.now(UTC).isoformat(),
            'prediction': {'probabilities': None}, 'documents': [], 'attempts': []}
        for prefix in (0, 1):
            snapshot = deepcopy(public)
            snapshot['documents'] = [] if prefix == 0 else [document]
            encoded = bind_fixture_roles(encode_visible_state(snapshot), resolver)
            records.append({'physical_match_id': physical, 'run_id': 'constructed-A:'+str(index),
                'prefix_index': prefix, 'visible_state': encoded})
        save(folder/'public.json', public)
        save(folder/'document.json', document)
        sources.append((source, folder, binding))
    Clock.at = datetime(2026, 10, 10, 10, tzinfo=UTC)
    for source, folder, binding in sources:
        result = capture_finland_result(binding, identity_archive_root=folder/'prepare/identity-archive',
            output=folder/'results', clock=lambda: Clock.now(UTC), transport=source)
        assert result['status'] == 'VERIFIED', result
        label = result['settlement_rows'][0]
        outcome = 0 if label['home_goals'] > label['away_goals'] else 2 if label['home_goals'] < label['away_goals'] else 1
        for row in records:
            if row['physical_match_id'] == binding['canonical']['source_event_id']:
                row.update(outcome=outcome, result_available_at=label['available_at'])
    return records, bindings


def identity_probe(root, f, g, fpath, *, identity_offset=0):
    """Same actual parent label and clock; only verified fixture scope differs."""
    from football_agent.evidence_only_policy import EvidenceOnlyPolicy
    from football_agent.learned_voi_web import _web_action
    from football_research.acquisition.fixture_roles import bind_fixture_roles, role_state
    from football_research.acquisition.role_context_query import query_input
    from football_research.acquisition.role_context_value import _public
    from football_research.data_adapters.fixture_role_identity import resolve_fixture_role_identity

    Clock.at = datetime(2026, 10, 15, 14, tzinfo=UTC)
    setups = []
    for i, (label, teams) in enumerate((('home', ('HJK', 'VPS')), ('away', ('VPS', 'HJK')),
                                      ('unrelated', ('KuPS', 'FC Inter')))):
        folder = root/'identity-probe'/label
        _, prepared, binding = prepare_source(folder, identity_offset+5000+i, 16, teams=teams)
        identity = resolve_fixture_role_identity(binding, archive_root=prepared['identity_archive'], as_of=Clock.now(UTC))
        setups.append((label, binding, identity))
    raw = RawArchive(root/'identity-probe/news-archive')
    at = Clock.now(UTC)
    url = 'https://www.veikkausliiga.com/uutiset/'
    body = (f'<html><main><a href="{ARTICLE}">{UPDATE}</a></main></html>').encode()
    obs = raw.save('open_page', url, {}, body, status=200, observed_at=at, context={'requested_at': at.isoformat()})
    tools = NewsDiscoveryTools(raw, ['www.veikkausliiga.com'], index_urls=[url], projection_version=6)
    doc = tools.project(obs, {'tool': 'read_url', 'endpoint': url, 'params': {}, 'source': 'open_page'})
    action = _web_action('read_url', ARTICLE, cost={'money': .01, 'requests': 1.})
    when = Clock.now(UTC).isoformat()
    outputs, inputs = {}, {}
    vectors = {}
    for label, binding, identity in setups:
        public = {'state': EvidenceOnlyPolicy(binding).initial_state, 'decision_at': when,
            'documents': [deepcopy(doc)], 'attempts': [], 'prediction': {'probabilities': None}}
        visible = bind_fixture_roles(encode_visible_state(public, [action]), identity)
        forecast = f.predict(visible, physical_match_id=binding['canonical']['source_event_id'])
        query = query_input(visible, action, forecast)
        fv = f._vector(role_state(visible))[0].tolist()
        gv = g._vector(_public(query, predictor=f))[0].tolist()
        value = g.predict(query, physical_match_id=binding['canonical']['source_event_id'],
                          predictor_sha256=sha256(fpath.read_bytes()).hexdigest())
        inputs[label] = {'visible': visible, 'action': action, 'forecast': forecast, 'query': query}
        outputs[label] = {'probabilities': forecast['probabilities'], 'role_counts': forecast['role_counts'],
            'query_value': value, 'f_vector_sha256': digest(fv), 'g_vector_sha256': digest(gv)}
        vectors[label] = {'f': fv, 'g': gv}
    output = {'evidence_origin': 'constructed_mechanism', 'same_observed_label': UPDATE,
        'counterfactual_identity_method': 'separate original-byte-verified fixtures, identical source document and decision clock',
        'outputs': outputs, 'vector_max_abs_deltas': {key: {
            model: max(abs(a-b) for a,b in zip(vectors['home'][model], vectors[key][model]))
            for model in ('f', 'g')} for key in ('away', 'unrelated')},
        'claim': 'representation distinguishes fixture roles; estimated utility can still be UNKNOWN'}
    save(root/'identity-probe/inputs.json', inputs)
    save(root/'identity-probe/report.json', output)
    return output


def build_role_chain(root, patch, *, b_physical=12):
    from football_research.acquisition.role_context_predictor import RoleContextPredictor
    from football_research.acquisition.role_context_value import RoleContextValueModel
    from football_research.data_adapters.role_context_gain import build_role_context_gain_from_runs

    root = Path(root)
    if b_physical not in (12, 24):
        raise ValueError('Only the predeclared B12 undercoverage and B24 role-coverage mechanisms are supported')
    identity_offset = 10000 if b_physical == 24 else 0
    root.mkdir(parents=True, exist_ok=False)
    Clock.at = datetime(2026, 10, 7, 12, tzinfo=UTC)
    calls = install_role_simulation(patch)
    save(root/'scope.json', {'evidence_origin': 'constructed_mechanism', 'real_network_requests': 0,
        'simulated_clock': '2026-10-07 through 2026-10-15 UTC, including future dates',
        'A_design': '12 raw-identity/body/result-backed constructed fixtures, 4 of each outcome',
        'B_design': f'{b_physical} disjoint actual collector fixtures, 2 query dates, signed targets computed by same f',
        'B_label_design': 'fixed each block of 12: 8 outcomes aligned with HJK role, 4 opposite; no fitting target overrides',
        'B_role_support_design': '6 per home/away role (undercoverage)' if b_physical == 12 else '12 per home/away role (existing gate coverage)',
        'C_design': 'one disjoint constructed fixture; no effectiveness assertion',
        'cost_design': 'all four loss-cost weights zero; finite request/resource caps',
        'claim': 'entity-relative representation and raw execution mechanism only, no real football benefit'})
    data, a_bindings = raw_A_records(root, identity_offset=identity_offset)
    save(root/'A-records.json', data)
    Clock.at = datetime(2026, 10, 10, 12, tzinfo=UTC)
    f = RoleContextPredictor().fit(data, as_of='2026-10-10T11:00:00+00:00',
        dataset_version=f'round203-constructed-raw-role-A-B{b_physical}', evidence_origin='constructed_mechanism')
    fpath = root/'f.json'
    f.save(fpath)
    previous = [root/'A-exposure.json']
    plan = exposure(f)
    plan['created_at'] = '2026-10-10T10:30:00+00:00'
    for row in plan['matches']:
        binding = a_bindings[row['source_event_id']]
        row['fixture_facts'] = {key: binding['canonical'][key] for key in FACTS}
        row['kickoff_at'] = binding['canonical']['kickoff']
    plan['cohort_id'] = plan['schema']+':'+digest({key:value for key,value in plan.items() if key != 'cohort_id'})[:24]
    save(previous[0], plan)
    settings = {'requests_per_match': 4, 'seed': 198, 'route': 'direct',
        'news_route_overrides': None, 'news_projection_version': 6,
        'lambda_cost': {'money': 0., 'requests': 0., 'latency_ms': 0., 'tokens': 0.}}
    sources, runs, archives = [], [], {}
    for i in range(b_physical):
        Clock.at = datetime(2026, 10, 10+i//(b_physical//2), 13, tzinfo=UTC)+timedelta(minutes=i)
        folder = root/'B'/f'{i:02}'
        swap = i % 2 == 1
        outcome = (2 if swap else 0) if i % 12 < 8 else (0 if swap else 2)
        source, prepared, _ = prepare_source(folder, identity_offset+3000+i, 12+i%2, outcome=outcome, swap=swap)
        plan = folder/'registration.json'
        prepare_official_context_registration(prepared['identity_path'], identity_archive=prepared['identity_archive'],
            previous_plan_paths=previous, predictor_path=fpath, output=plan, purpose='calibration', settings=settings)
        report = collect_official_context(plan, output=folder/'run')
        save(folder/'collection-report.json', report)
        previous.append(plan)
        runs.append(folder/'run')
        sources.append(source)
    Clock.at = datetime(2026, 10, 14, 22, tzinfo=UTC)
    for source, run in zip(sources, runs):
        binding = read(run/'identity.json')
        archive = run.parent/'results'
        result = capture_finland_result(binding, identity_archive_root=run/'identity-archive', output=archive,
            clock=lambda: Clock.now(UTC), transport=source)
        assert result['status'] == 'VERIFIED', result
        archives[binding['canonical']['source_event_id']] = archive
    Clock.at = datetime(2026, 10, 15, 12, tzinfo=UTC)
    cutoff = Clock.now(UTC).isoformat()
    gain = build_role_context_gain_from_runs(runs, archives, root/'gain', predictor_path=fpath, as_of=cutoff)
    g = RoleContextValueModel().fit([root/'gain'], predictor_path=fpath, as_of=cutoff,
        dataset_version=f'round203-constructed-role-actual-B{b_physical}-returns')
    gpath = root/'g.json'
    g.save(gpath)
    Clock.at = datetime(2026, 10, 15, 13, tzinfo=UTC)
    _, prepared, _ = prepare_source(root/'C', identity_offset+4000, 16)
    prepare_official_context_registration(prepared['identity_path'], identity_archive=prepared['identity_archive'],
        previous_plan_paths=previous, predictor_path=fpath, value_model_path=gpath,
        output=root/'C/registration.json', purpose='confirmation', settings=settings)
    c = collect_official_context(root/'C/registration.json', output=root/'C/run')
    save(root/'C/collection-report.json', c)
    replay = replay_saved_run(root/'C/run', output=root/'C/replay', project_root=ROOT)
    run = read(root/'C/run/run.json')
    basis = Counter(cycle.get('selection_basis') for cycle in run['cycles'] if cycle.get('selected_action'))
    probe = identity_probe(root, f, g, fpath, identity_offset=identity_offset)
    report = {'scope': read(root/'scope.json'), 'gain': gain,
        'A_physical': len(f.metadata['training_physical_match_ids']),
        'B_physical': len(g.metadata['training_physical_match_ids']), 'C_physical': 1,
        'actor_returns': len(calls), 'C_returns': c['http_requests'], 'C_selection_basis': dict(basis),
        'C_initial': run['initial_probabilities'], 'C_final': run['final_probabilities'],
        'C_terminal': run['terminal_selection'], 'C_exact_replay': replay,
        'f_model_version': f.model_version, 'g_model_version': g.model_version,
        'identity_probe': probe,
        'real_model_ready': False, 'real_network_requests': 0, 'production_ready': False}
    save(root/'report.json', report)
    return report


@pytest.fixture(scope='module')
def role_chain(tmp_path_factory):
    root = tmp_path_factory.mktemp('role')/'m'
    with pytest.MonkeyPatch.context() as patch:
        yield root, build_role_chain(root, patch)


def test_actual_role_f_signed_B_gain_g_C_and_full_raw_replay(role_chain):
    root, report = role_chain
    assert report['A_physical'] == report['B_physical'] == 12 and report['C_physical'] == 1
    assert report['C_exact_replay']['identical_replay']
    assert report['C_exact_replay']['replay_network_requests'] == 0
    targets = read(root/'gain/targets.json')
    assert targets and any(row['signed_gain'] < 0 for row in targets)
    assert any(row['signed_gain'] > 0 for row in targets)
    for row in targets:
        y = row['outcome']
        assert math.isclose(row['signed_gain'], math.log(row['prediction_after'][y])-math.log(row['prediction_before'][y]), abs_tol=1e-14)
    assert report['C_initial'] == [1/3]*3 and report['C_final'] != report['C_initial']
    assert report['real_model_ready'] is report['production_ready'] is False


def test_new_role_models_cannot_be_loaded_as_legacy_optional_models(role_chain):
    from football_research.acquisition.optional_prior_predictor import OptionalPriorContextPredictor
    from football_research.acquisition.optional_context_value import OptionalContextValueModel
    root, _ = role_chain
    with pytest.raises(ValueError):
        OptionalPriorContextPredictor.load(root/'f.json')
    with pytest.raises(ValueError):
        OptionalContextValueModel.load(root/'g.json', predictor_path=root/'f.json')


def test_physical_A_B_C_exposure_stays_disjoint_and_C_cannot_be_gain_B(role_chain, tmp_path):
    from football_research.data_adapters.role_context_gain import build_role_context_gain_from_runs
    root, _ = role_chain
    f, g = read(root/'f.json'), read(root/'g.json')
    a = set(f['metadata']['training_physical_match_ids'])
    b = set(g['metadata']['training_physical_match_ids'])
    c = read(root/'C/run/identity.json')['canonical']['source_event_id']
    assert not a & b and c not in a | b
    with pytest.raises(ValueError, match='Only registered independent B'):
        build_role_context_gain_from_runs([root/'C/run'], {}, tmp_path/'wrong-phase',
            predictor_path=root/'f.json', as_of=Clock.now(UTC))


def test_unread_article_body_never_appears_in_pre_query_visible_candidates(role_chain):
    from football_research.acquisition.fixture_roles import bind_fixture_roles
    from football_research.acquisition.role_context_predictor import RoleContextPredictor
    from football_research.acquisition.role_context_query import query_input
    root, _ = role_chain
    probe = read(root/'identity-probe/inputs.json')['home']
    baseline, action = probe['visible'], probe['action']
    assert SENTINEL not in json.dumps(baseline)
    injected = deepcopy(baseline)
    injected['future_returns'] = {'body': SENTINEL, 'outcome': 2, 'gain': 999}
    injected['candidates'][action['action_id']]['future_body'] = SENTINEL
    # The encoder whitelists the pre-read actual menu; unknown payloads never
    # become role text. A changed already-bound sidecar is rejected outright.
    with pytest.raises(ValueError, match='roles|projection|binding'):
        bind_fixture_roles(injected, baseline['fixture_roles']['identity'])
    f = RoleContextPredictor.load(root/'f.json')
    forecast = f.predict(baseline, physical_match_id=baseline['fixture_roles']['identity_scope']['physical_match_id'])
    rebuilt = query_input(baseline, action, forecast)
    assert rebuilt == probe['query'] and SENTINEL not in json.dumps(rebuilt)


def test_same_text_verified_home_away_and_unrelated_fixtures_change_shared_f_g_vectors(role_chain):
    _, report = role_chain
    probe = report['identity_probe']
    for kind in ('away', 'unrelated'):
        assert probe['vector_max_abs_deltas'][kind]['f'] > 0
        assert probe['vector_max_abs_deltas'][kind]['g'] > 0
    assert probe['outputs']['home']['probabilities'] != probe['outputs']['away']['probabilities']
    assert probe['outputs']['home']['role_counts']['home'] == 1
    assert probe['outputs']['away']['role_counts']['away'] == 1
    assert probe['outputs']['unrelated']['role_counts']['unattributed'] == 1
