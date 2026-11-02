from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from football_agent.learned_voi_runtime import LearnedVOIRuntime
from football_agent.learned_voi_web import _web_action
from football_research.data_adapters.prospective_text import (
    MarketPriorPolicy, select_initializations, collect_prepared_batch, _save)
from football_research.data_adapters.types import ResearchDataset, ResearchMatch, ResearchOddsSnapshot

NOW = datetime(2026, 10, 6, 10, tzinfo=timezone.utc)


def objects(mid='m1', minutes=180):
    match = ResearchMatch(mid, 'league', None, 'home', 'away', NOW + timedelta(minutes=minutes),
        fixture_available_at=NOW-timedelta(minutes=2), source='spftll',
        input_manifest=('observation:fixture',), status='scheduled',
        home_team_name='Home', away_team_name='Away')
    quote = ResearchOddsSnapshot(mid, NOW-timedelta(minutes=1), 2., 3., 4.,
                                 input_manifest=('observation:quote',))
    return match, quote


def make_policy():
    from football_research.data_adapters.types import json_record
    match, quote = objects()
    return MarketPriorPolicy(json_record(match), json_record(quote), frozen_at=NOW,
                             prediction_cutoff=match.kickoff-timedelta(minutes=1))


def test_market_prior_unknown_score_period_not_promoted():
    p = make_policy()
    assert p.model.fixture['score_period'] == 'unknown'
    assert p.model.quote['period'] == 'regulation'
    assert p.model.probabilities == pytest.approx([6/13, 4/13, 3/13])
    assert p.prior_binding()['probability_basis'] == 'observed_market_snapshot'
    assert p.prior_binding()['training_match_ids'] == []
    assert p.prior_binding()['fitted_parameters'] is False
    restored = MarketPriorPolicy.from_manifest(json.loads(json.dumps(p.manifest)))
    assert restored.manifest == p.manifest
    assert restored.model.predict_partial(p.model.initial) == p.model.predict_partial(p.model.initial)


@pytest.mark.parametrize('change', ['probabilities', 'quote_sha256', 'normalization', 'model_version'])
def test_manifest_does_not_accept_unbound_initial_probabilities(change):
    m = deepcopy(make_policy().manifest)
    if change == 'model_version':
        m[change] = 'forged'
    else:
        m['market_initialization'][change] = [0.1, 0.2, 0.7] if change == 'probabilities' else 'forged'
    with pytest.raises(ValueError, match='changed'):
        MarketPriorPolicy.from_manifest(m)


@pytest.mark.parametrize('bad', ['future_quote', 'old_quote', 'historical', 'known_outcome', 'wrong_match', 'future_fixture'])
def test_operational_p0_rejects_unobserved_stale_mixed_or_labelled_initialization(bad):
    p = make_policy()
    m, q = deepcopy(p.model.fixture), deepcopy(p.model.quote)
    if bad == 'future_quote': q['observed_at'] = (NOW+timedelta(seconds=1)).isoformat()
    if bad == 'old_quote': q['observed_at'] = (NOW-timedelta(hours=2)).isoformat()
    if bad == 'historical': m['information_mode'] = 'historical_reconstruction'
    if bad == 'known_outcome': m['home_goals'] = m['away_goals'] = 0
    if bad == 'wrong_match': q['match_id'] = 'different'
    if bad == 'future_fixture': m['fixture_available_at'] = (NOW+timedelta(seconds=1)).isoformat()
    with pytest.raises(ValueError):
        MarketPriorPolicy(m, q, frozen_at=NOW, prediction_cutoff=p.model.cutoff)


def test_selection_is_chronological_not_quote_strength_and_keeps_missing():
    m1, q1 = objects('m1', 180)
    m2, q2 = objects('m2', 120)
    m3, _ = objects('m3', 100)
    data = ResearchDataset((m1, m2, m3), (q1, q2), 'real-capture', (), {})
    selected = select_initializations(data, as_of=NOW, limit=2)
    assert [p['market_initialization']['fixture']['match_id'] for p in selected['selected']] == ['m2', 'm1']
    assert selected['excluded'] == [{'match_id': 'm3', 'reason': 'complete_current_regulation_quote_missing'}]
    assert selected['labels'] == []


def test_selection_supports_21_and_42_day_windows_with_predeclared_fixture_scope():
    early, early_quote = objects('early', 20*24*60)
    later, later_quote = objects('later', 40*24*60)
    data = ResearchDataset((later, early), (later_quote, early_quote), 'real-capture', (), {})
    window21 = select_initializations(data, as_of=NOW, horizon_hours=504,
                                      fixture_ids=['later', 'early', 'missing'])
    assert [p['market_initialization']['fixture']['match_id'] for p in window21['selected']] == ['early']
    assert window21['selection_scope']['fixture_ids'] == ['early', 'later', 'missing']
    assert window21['selection_scope']['missing_fixture_ids'] == ['missing']
    assert window21['selection_scope']['window_end'] == (NOW+timedelta(days=21)).isoformat()
    assert window21['selection_scope']['outcome_used_for_selection'] is False
    window42 = select_initializations(data, as_of=NOW, horizon_hours=1008, fixture_ids=['later'])
    assert [p['market_initialization']['fixture']['match_id'] for p in window42['selected']] == ['later']
    with pytest.raises(ValueError, match='Bounded future horizon'):
        select_initializations(data, as_of=NOW, horizon_hours=1009)


@pytest.mark.parametrize('fixture_ids', [[], '', [''], [None]])
def test_selection_rejects_ambiguous_empty_fixture_scope(fixture_ids):
    match, quote = objects()
    data = ResearchDataset((match,), (quote,), 'real-capture', (), {})
    with pytest.raises(ValueError, match='fixture_ids'):
        select_initializations(data, as_of=NOW, fixture_ids=fixture_ids)


def test_prepare_cli_forwards_bounded_horizon_and_fixture_scope(tmp_path, monkeypatch):
    from football_research.data_adapters import prospective_text as module
    from football_research.data_adapters import spftll
    captured = []
    monkeypatch.setattr(spftll, 'load_spftll_archive', lambda *a, **kw: 'loaded')
    monkeypatch.setattr(module, 'select_initializations',
                        lambda *a, **kw: captured.append((a, kw)) or {'selected': []})
    output = tmp_path/'new-plan.json'
    module.main(['prepare', '--archive', str(tmp_path/'archive'), '--output', str(output),
                 '--horizon-hours', '504', '--fixture-id', 'spftll:match:101',
                 '--fixture-id', 'spftll:match:102'])
    assert captured[0][1]['horizon_hours'] == 504
    assert captured[0][1]['fixture_ids'] == ['spftll:match:101', 'spftll:match:102']
    assert json.loads(output.read_text()) == {'selected': []}


def test_runtime_preserves_unknown_and_failure_without_fake_probability_update(tmp_path):
    p = make_policy()
    a = _web_action('search_web', 'Home Away', cost={'money': .01, 'requests': 1})
    state = p.model.initial
    plan = p.plan(state, [a], {'limits': {'money': .01, 'requests': 1}, 'spent': {}})
    assert plan['candidates'][0]['value']['net_voi'] is None
    clock_times = iter([NOW+timedelta(seconds=i) for i in range(1, 20)])
    run = LearnedVOIRuntime(p, seed=179, exploration_limits={'requests': 1, 'money': .01},
        max_exploration_actions=1).run(state, run_id='test-prospective',
        limits={'requests': 1, 'money': .01}, propose=lambda public: [a],
        execute=lambda action, public: {'status': 'FAILED',
            'actual_cost': {'requests': 1., 'money': None, 'latency_ms': 2., 'tokens': 0.},
            'receipt': {'failure': 'source_unavailable'}}, output=tmp_path/'run',
        provider_version='fixture', generator_version='fixture', clock=lambda: next(clock_times),
        evidence_match={'match_id': 'm1', 'home_team': 'Home', 'away_team': 'Away',
            'kickoff_at': p.model.fixture['kickoff'], 'prediction_cutoff': p.model.cutoff})
    assert len(run['attempts']) == 1
    assert run['exploration_actions'] == 1
    assert run['initial_probabilities'] == run['final_probabilities']
    assert run['outcome'] is None
    assert run['measured_resources']['requests'] == 1
    assert run['cycles'][0]['selection_basis'] != 'computed_positive_voi'


def test_structured_reveal_cannot_change_frozen_market_prior():
    p = make_policy()
    s = deepcopy(p.model.initial)
    s['base']['home_team_id'] = 'different'
    with pytest.raises(ValueError): p.model.predict_partial(s)
    with pytest.raises(ValueError):
        p.plan(p.model.initial, [{'tool': 'frozen_evidence_projection', 'group_ids': ['elo']}], {})


def test_duplicate_match_batch_rejected_before_output_or_network(tmp_path):
    p = make_policy()
    plan = tmp_path/'plan.json'
    _save(plan, {'schema': 'prospective-text-initializations-v1', 'selected': [p.manifest, p.manifest]})
    with pytest.raises(ValueError, match='Duplicate'):
        collect_prepared_batch(plan, output=tmp_path/'out', archive_root=tmp_path/'absent',
                               allowed_hosts=['www.uefa.com'])
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('version', [0, 3, True, 2])
def test_source_profile_opt_in_rejects_invalid_or_unverified_before_io(tmp_path, version):
    with pytest.raises(ValueError, match='[Ss]ource profile'):
        collect_prepared_batch(tmp_path/'absent-plan', output=tmp_path/'out',
            archive_root=tmp_path/'absent', allowed_hosts=[], source_profile_version=version)


def test_evidence_projection_rejects_unknown_version_before_input_or_network(tmp_path):
    with pytest.raises(ValueError, match='Unsupported match evidence'):
        collect_prepared_batch(tmp_path/'absent-plan', output=tmp_path/'out',
            archive_root=tmp_path/'absent', allowed_hosts=[], evidence_version='unknown')
    assert not (tmp_path/'out').exists()


def test_collection_cli_forwards_explicit_article_evidence_projection(tmp_path, monkeypatch):
    from football_research.data_adapters import prospective_text as module
    captured = []
    monkeypatch.setattr(module, 'collect_prepared_batch', lambda *a, **kw: captured.append(kw) or {})
    module.main(['collect', '--archive', str(tmp_path/'absent'), '--plan', 'not-read.json',
        '--output', str(tmp_path/'new-output'), '--allowed-host', 'www.uefa.com',
        '--news-provider', 'bing_rss', '--news-projection-version', '2',
        '--evidence-version', 'match-evidence-extraction-v3'])
    assert captured[0]['evidence_version'] == 'match-evidence-extraction-v3'
    assert captured[0]['news_projection_version'] == 2
    assert not (tmp_path/'out').exists()


def test_future_initialization_rejected_before_output_or_network(tmp_path):
    from football_research.data_adapters.types import json_record
    future = datetime.now(timezone.utc)+timedelta(days=2)
    m = ResearchMatch('future', 'l', None, 'h', 'a', future+timedelta(hours=3),
        fixture_available_at=future-timedelta(minutes=1), input_manifest=('m',), status='scheduled')
    q = ResearchOddsSnapshot('future', future-timedelta(minutes=1), 2., 3., 4., input_manifest=('q',))
    p = MarketPriorPolicy(json_record(m), json_record(q), frozen_at=future,
                         prediction_cutoff=future+timedelta(hours=2))
    plan = tmp_path/'plan.json'
    _save(plan, {'schema': 'prospective-text-initializations-v1', 'selected': [p.manifest]})
    with pytest.raises(ValueError, match='future'):
        collect_prepared_batch(plan, output=tmp_path/'out', archive_root=tmp_path/'absent', allowed_hosts=[])
    assert not (tmp_path/'out').exists()


def test_direct_runtime_rejects_market_freeze_after_actual_start(tmp_path):
    p = make_policy()
    def forbidden(*args):
        pytest.fail('Future p0 must be rejected before callbacks')
    runtime = LearnedVOIRuntime(p, seed=179, exploration_limits={'requests': 1, 'money': .01},
        max_exploration_actions=1)
    with pytest.raises(ValueError, match='market'):
        runtime.run(p.model.initial, run_id='future-freeze', limits={'requests': 1, 'money': .01},
            propose=forbidden, execute=forbidden, output=tmp_path/'out',
            provider_version='test', generator_version='test', clock=lambda: NOW-timedelta(seconds=1))
    assert not (tmp_path/'out').exists()


@pytest.fixture
def archived_market(tmp_path):
    """A synthetic source fixture validates full raw -> p0 -> response joining."""
    import shutil
    from football_spider.storage.raw_archive import RawArchive
    from football_agent.inquiry_tools import OpenWebTools
    from football_agent.learned_voi_web import MatchDocumentCandidates
    from football_agent.match_analysis_report import write_analysis_report
    from football_research.data_adapters.spftll import load_spftll_archive
    prior = RawArchive(tmp_path/'prior-source')
    source_row = {'matchId': 101, 'leagueId': 1864, 'leagueChs': 'fixture league',
        'homeId': 11, 'awayId': 12, 'homeChs': 'Home', 'awayChs': 'Away',
        'footMatchState': 0, 'matchTime': '2026-10-06T21:00:00',
        'spfWinFoot': 2., 'spfEqualFoot': 3., 'spfLoseFoot': 4.}
    prior.save('spftll', 'https://slb.haoyun999.cn/api/Game/GetSimpleMatchsAll', {},
        json.dumps({'code': 0, 'data': [{'list': [source_row]}]}).encode(), status=200,
        observed_at=NOW-timedelta(minutes=1))
    data = load_spftll_archive(prior.root/'raw', as_of=NOW)
    plan = select_initializations(data, as_of=NOW, limit=1)
    p = MarketPriorPolicy.from_manifest(plan['selected'][0])
    folder = tmp_path/'original'
    archive = RawArchive(folder/'source-archive')
    clock = lambda: NOW+timedelta(seconds=2)
    tools = OpenWebTools(archive, ['club.example'], extraction_version=3, clock=clock)
    generator = MatchDocumentCandidates(match_terms=['Home', 'Away'],
        seed_urls=['https://club.example/news/squad'], allowed_hosts=['club.example'],
        numeric_actions=[], web_cost={'requests': 1., 'money': .01})
    identity = {'match_id': p.model.fixture['match_id'], 'home_team': 'Home', 'away_team': 'Away',
                'kickoff_at': p.model.fixture['kickoff'], 'prediction_cutoff': p.model.cutoff}
    def execute(action, public):
        prepared = tools.prepare(action, {'task': {'seed_urls': generator.seed_urls}, 'documents': public['documents']})
        body = (b'<rss><channel><item><title>Home squad</title><link>https://club.example/news/squad</link>'
                b'<description>Home information</description></item></channel></rss>'
                if prepared['tool'] == 'search_web' else b'<html><p>Home squad news</p></html>')
        obs = archive.save(prepared['source'], prepared['endpoint'], prepared['params'], body,
            status=200, observed_at=clock(), context={'requested_at': (NOW+timedelta(seconds=1)).isoformat()})
        return {'status': 'OK', 'document': tools.project(obs, prepared),
            'actual_cost': {'requests': 1., 'money': None, 'latency_ms': 0., 'tokens': 0.},
            'receipt': {'observation': obs.model_dump(mode='json'), 'prepared': prepared}}
    moments = iter([NOW+timedelta(seconds=i) for i in (1, 1, 3, 4)])
    run = LearnedVOIRuntime(p, seed=179, exploration_limits={'money': .01, 'requests': 1},
        max_exploration_actions=1).run(p.model.initial, run_id='synthetic-market-episode',
        limits={'money': .01, 'requests': 1}, propose=generator, execute=execute, output=folder,
        clock=lambda: next(moments), provider_version='synthetic-source', generator_version=generator.version,
        run_metadata={'generator': generator.configuration(), 'transport': tools.configuration()},
        evidence_match=identity)
    shutil.copytree(prior.root, folder/'prior-archive')
    write_analysis_report(run, match=identity, output=folder)
    label = {'match_id': p.model.fixture['match_id'], 'kickoff': p.model.fixture['kickoff'],
        'status': 'final', 'score_period': 'regulation', 'home_goals': 1, 'away_goals': 0,
        'available_at': (NOW+timedelta(hours=6)).isoformat(), 'observed_at': (NOW+timedelta(hours=6)).isoformat(),
        'source': 'independent-test-result-fixture', 'source_reference': 'independent-test-result:1',
        'input_manifest': ['independent-test-result-digest'], 'revision_id': 'test-only-result-revision',
        'information_mode': 'operational'}
    return folder, p, label


def test_independent_settlement_completes_market_episode_without_changing_unknown_fixture(archived_market):
    from football_research.data_adapters.prospective_text import join_settlements
    folder, p, label = archived_market
    args = dict(prior_bindings={p.model.model_version: p.prior_binding()}, as_of=NOW+timedelta(days=1))
    assert p.model.fixture['score_period'] == 'unknown'
    pending = join_settlements([folder], settlements=[], **args)
    assert pending['training_records'] == []
    qualified = join_settlements([folder], settlements=[label], **args)
    assert len(qualified['training_records']) == 1
    assert qualified['training_records'][0]['prior_probabilities'] == p.model.probabilities
    assert qualified['training_records'][0]['outcome'] == 0
    label['score_period'] = 'extra_time'
    assert join_settlements([folder], settlements=[label], **args)['training_records'] == []


@pytest.mark.parametrize('tamper', ['p0', 'raw'])
def test_market_episode_rejects_changed_p0_or_original_quote_bytes(archived_market, tamper):
    from football_research.data_adapters.prospective_text import join_settlements
    folder, p, label = archived_market
    if tamper == 'p0':
        run = json.loads((folder/'run.json').read_text(encoding='utf-8'))
        run['initial_probabilities'] = [.1, .2, .7]
        _save(folder/'run.json', run)
    else:
        import gzip
        payload = next((folder/'prior-archive'/'raw').rglob('*.body.gz'))
        payload.write_bytes(gzip.compress(b'{"code":0,"data":[]}'))
    result = join_settlements([folder], settlements=[label],
        prior_bindings={p.model.model_version: p.prior_binding()}, as_of=NOW+timedelta(days=1))
    assert result['training_records'] == []
    expected = 'observed_market_prior_archive_invalid' if tamper == 'raw' else 'observed_market_prior_binding_invalid'
    assert expected in result['dataset']['episodes'][0]['training_exclusion_reasons']


def test_market_episode_generic_replay_uses_raw_quote_without_historical_model(archived_market, tmp_path):
    from football_agent.learned_voi_replay import replay_saved_run
    folder, _, _ = archived_market
    report = replay_saved_run(folder, output=tmp_path/'replay', project_root=tmp_path/'no-historical-model')
    assert report['identical_replay']
    assert report['raw_responses_reprojected'] == 1
    assert report['replay_network_requests'] == 0


def test_cli_audit_defaults_to_no_labels_and_preserves_existing_output(archived_market, tmp_path):
    from football_research.data_adapters.prospective_text import main
    import shutil
    folder, p, _ = archived_market
    batch = tmp_path/'batch'
    target = batch/'match-101'
    shutil.copytree(folder, target)
    _save(target/'prior-bindings.json', {p.model.model_version: p.prior_binding()})
    output = tmp_path/'audit.json'
    main(['audit', '--batch', str(batch), '--output', str(output)])
    result = json.loads(output.read_text(encoding='utf-8'))
    assert result['training_records'] == []
    assert result['dataset']['audit']['unique_matches'] == 1
    with pytest.raises(FileExistsError):
        main(['audit', '--batch', str(batch), '--output', str(output)])
