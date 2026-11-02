"""Full constructed A/B/C mechanism through public collectors and raw results.

Only source transport and the explicitly simulated clock are replaced. The
predictor, gains, value fit, candidate discovery, policy and replay are real.
This is deliberately not a real-observational effectiveness experiment.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import importlib
import json
import math
from pathlib import Path
import socket

import pytest

from football_agent.news_discovery import NewsDiscoveryTools
from football_agent.learned_voi_replay import replay_saved_run
from football_research.acquisition.optional_prior_predictor import OptionalPriorContextPredictor
from football_research.acquisition.optional_context_value import OptionalContextValueModel
from football_research.data_adapters.official_context_registration import prepare_official_context_registration
from football_research.data_adapters.official_context_collection import collect_official_context
from football_research.data_adapters.official_context_gain import build_official_context_gain_from_runs
from football_research.data_adapters.official_fixture import prepare_official_fixture
from football_research.data_adapters.finland_result_capture import capture_finland_result
from test_context_predictor import visible, WORDS, AS_OF
from test_optional_prior_predictor import missing
from test_context_gain import exposure
from test_official_fixture_results import Source
from test_finland_stage_fixture import schedule

ROOT = Path(__file__).resolve().parents[3]
UTC = timezone.utc
ARTICLE = 'https://www.veikkausliiga.com/uutiset/2026/10/07/hjk-vps-squad-preview/'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)+'\n')


class Clock(datetime):
    at = datetime(2026, 10, 7, 12, tzinfo=UTC)

    @classmethod
    def now(cls, tz=None):
        cls.at += timedelta(milliseconds=10)
        return cls.at.astimezone(tz or UTC)


class OfficialSource(Source):
    def __init__(self, index, day, outcome=0):
        super().__init__()
        self.native, self.day, self.outcome = str(9876000+index), day, outcome
        self.score = ('2 — 0', '1 — 1', '0 — 2')[outcome]

    def schedule(self, score='-'):
        return schedule(home=self.home, away=self.away, native=self.native,
                        date=f'202610{self.day:02}', clock='19:00', score=score)

    def terminal(self):
        text = super().terminal().decode().replace('05.10.2026', f'{self.day:02}.10.2026')
        home, away = ((2, 0), (1, 1), (0, 2))[self.outcome]
        return text.replace('id="home-goals">2', f'id="home-goals">{home}').replace(
            'id="away-goals">0', f'id="away-goals">{away}').encode()


def install_simulation(patch):
    def blocked(*args, **kwargs):
        raise AssertionError('Constructed full-chain mechanism must never use the network')
    for owner, name in ((socket, 'getaddrinfo'), (socket, 'create_connection'),
                        (socket.socket, 'connect'), (socket.socket, 'connect_ex')):
        patch.setattr(owner, name, blocked)
    for name in ('football_agent.learned_voi_runtime', 'football_agent.inquiry_tools',
                 'football_research.acquisition.optional_prior_predictor',
                 'football_research.acquisition.optional_context_value',
                 'football_research.data_adapters.official_context_registration',
                 'football_research.data_adapters.official_context_collection',
                 'football_research.data_adapters.official_context_gain',
                 'football_research.data_adapters.settlement'):
        patch.setattr(importlib.import_module(name), 'datetime', Clock)
    calls = []

    def execute(self, prepared, *, run_id, request_id):
        calls.append(deepcopy(prepared))
        if prepared['tool'] == 'search_web':
            body = b'<rss><channel></channel></rss>'
        elif prepared['endpoint'] == ARTICLE:
            body = ('<html><script type="application/ld+json">'+json.dumps({
                '@type': 'NewsArticle', 'headline': 'HJK VPS squad preview',
                'datePublished': '2026-10-07T11:00:00+00:00',
                'articleBody': WORDS[0]+' HJK VPS. UNREAD_BODY_SENTINEL.'})+'</script></html>').encode()
        else:
            body = (f'<html><main>Official news.<a href="{ARTICLE}">HJK VPS {WORDS[0]}</a>'
                    '</main></html>').encode()
        at = Clock.now(UTC)
        obs = self.archive.save(prepared['source'], prepared['endpoint'], prepared['params'], body,
            status=200, observed_at=at, context={'requested_at': at.isoformat(), 'inquiry_run_id': run_id})
        return {'document': self.project(obs, prepared), 'observation': obs.model_dump(mode='json'),
                'http_attempts': 1, 'money': None, 'money_status': 'not_measured', 'latency_seconds': .01}

    patch.setattr(NewsDiscoveryTools, 'execute', execute)
    return calls


def build_chain(root, patch):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    Clock.at = datetime(2026, 10, 7, 12, tzinfo=UTC)
    calls = install_simulation(patch)
    save(root/'scope.json', {'evidence_origin': 'constructed_mechanism',
        'simulated_clock': '2026-10-07 through 2026-10-10 UTC', 'real_network_requests': 0,
        'training_A': '12 constructed physical matches; labels 6 home / 3 draw / 3 away',
        'B_design': '12 constructed different native identities over 2 dates; 10 home / 2 away',
        'cost_design': 'all loss-conversion weights zero; request and monetary caps enforced',
        'claim': 'public raw same-f B/g/C mechanism only; no real accuracy or VoI benefit'})
    data = []
    for i, outcome in enumerate([0]*6+[1]*3+[2]*3):
        for text in (None, WORDS[outcome]):
            state = missing(visible(f'constructed:A:{i}', day=i+1, text=text,
                                    source='www.veikkausliiga.com'))
            state['before']['current_probabilities'] = None
            data.append({'physical_match_id': f'constructed:physical:A:{i}', 'visible_state': state,
                         'outcome': outcome, 'result_available_at': f'2025-01-{i+1:02}T22:00:00+00:00'})
    save(root/'A-records.json', data)
    f = OptionalPriorContextPredictor().fit(data, as_of=AS_OF, dataset_version='round201-constructed-A',
                                          evidence_origin='constructed_mechanism')
    fpath = root/'f.json'
    f.save(fpath)
    previous = [root/'A-exposure.json']
    save(previous[0], exposure(f))
    settings = {'requests_per_match': 4, 'seed': 198, 'route': 'direct',
                'news_route_overrides': None, 'news_projection_version': 6,
                'lambda_cost': {'money': 0., 'requests': 0., 'latency_ms': 0., 'tokens': 0.}}
    sources, runs, result_archives = [], [], {}
    Clock.at = datetime(2026, 10, 7, 13, tzinfo=UTC)
    for i in range(12):
        folder = root/'B'/f'{i:02}'
        source = OfficialSource(i, 8+i%2, 0 if i < 10 else 2)
        prepared = prepare_official_fixture('veikkausliiga:2026:match:'+source.native,
            output=folder/'prepare', clock=lambda: Clock.now(UTC), transport=source)
        assert prepared['status'] == 'ready', prepared
        plan = folder/'registration.json'
        prepare_official_context_registration(prepared['identity_path'],
            identity_archive=prepared['identity_archive'], previous_plan_paths=previous,
            predictor_path=fpath, output=plan, purpose='calibration', settings=settings)
        report = collect_official_context(plan, output=folder/'run')
        save(folder/'collection-report.json', report)
        previous.append(plan)
        runs.append(folder/'run')
        sources.append(source)
    Clock.at = datetime(2026, 10, 9, 22, tzinfo=UTC)
    for source, run in zip(sources, runs):
        binding = read(run/'identity.json')
        archive = run.parent/'results'
        verified = capture_finland_result(binding, identity_archive_root=run/'identity-archive',
            output=archive, clock=lambda: Clock.now(UTC), transport=source)
        assert verified['status'] == 'VERIFIED', verified
        result_archives[binding['canonical']['source_event_id']] = archive
    Clock.at = datetime(2026, 10, 10, 12, tzinfo=UTC)
    cutoff = Clock.now(UTC).isoformat()
    gain = build_official_context_gain_from_runs(runs, result_archives, root/'gain',
                                               predictor_path=fpath, as_of=cutoff)
    g = OptionalContextValueModel().fit([root/'gain'], predictor_path=fpath, as_of=cutoff,
                                       dataset_version='round201-actual-constructed-B-returns')
    gpath = root/'g.json'
    g.save(gpath)
    Clock.at = datetime(2026, 10, 10, 13, tzinfo=UTC)
    source = OfficialSource(1000, 11)
    prepared = prepare_official_fixture('veikkausliiga:2026:match:'+source.native,
        output=root/'C/prepare', clock=lambda: Clock.now(UTC), transport=source)
    assert prepared['status'] == 'ready', prepared
    prepare_official_context_registration(prepared['identity_path'],
        identity_archive=prepared['identity_archive'], previous_plan_paths=previous, predictor_path=fpath,
        value_model_path=gpath, output=root/'C/registration.json', purpose='confirmation', settings=settings)
    c = collect_official_context(root/'C/registration.json', output=root/'C/run')
    save(root/'C/collection-report.json', c)
    replay = replay_saved_run(root/'C/run', output=root/'C/replay', project_root=ROOT)
    run = read(root/'C/run/run.json')
    basis = Counter(cycle.get('selection_basis') for cycle in run['cycles'] if cycle.get('selected_action'))
    report = {'scope': read(root/'scope.json'), 'gain': gain,
        'A_physical': len(f.metadata['training_physical_match_ids']),
        'B_physical': len(g.metadata['training_physical_match_ids']), 'C_physical': 1,
        'actor_returns': len(calls), 'C_returns': c['http_requests'], 'C_selection_basis': dict(basis),
        'C_initial': run['initial_probabilities'], 'C_final': run['final_probabilities'],
        'C_terminal': run['terminal_selection'], 'C_exact_replay': replay,
        'f_model_version': f.model_version, 'g_model_version': g.model_version,
        'real_model_ready': False, 'real_network_requests': 0, 'production_ready': False}
    save(root/'report.json', report)
    return report


@pytest.fixture(scope='module')
def chain(tmp_path_factory):
    root = tmp_path_factory.mktemp('official-full-chain')/'mechanism'
    with pytest.MonkeyPatch.context() as patch:
        result = build_chain(root, patch)
        yield root, result


def test_full_raw_B_to_same_f_signed_gain_g_independent_C_and_exact_replay(chain):
    root, report = chain
    assert report['A_physical'] == report['B_physical'] == 12 and report['C_physical'] == 1
    assert report['gain']['supervised_physical_matches'] == 12
    assert report['gain']['negative_targets'] > 0 and report['gain']['positive_targets'] > 0
    assert report['C_exact_replay']['identical_replay']
    assert report['C_exact_replay']['replay_network_requests'] == 0
    assert report['C_initial'] == [.5, .25, .25] and report['C_final'] != report['C_initial']
    assert any('positive' in str(key) for key in report['C_selection_basis']), report['C_selection_basis']
    for row in read(root/'gain/targets.json'):
        y = row['outcome']
        assert math.isclose(row['signed_gain'], math.log(row['prediction_after'][y])-math.log(row['prediction_before'][y]), abs_tol=1e-14)
    assert all(row['inputs']['state']['prior'] is None and row['inputs']['forecast']['prior_probabilities'] is None
               for row in read(root/'gain/features.json'))


def test_pending_results_never_become_zero_gain_labels(chain, tmp_path):
    root, _ = chain
    report = build_official_context_gain_from_runs([root/'B/00/run'], {}, tmp_path/'pending',
        predictor_path=root/'f.json', as_of=Clock.now(UTC))
    assert report['supervised_transitions'] == 0 and report['pending_prefixes'] > 0
    assert report['actual_transitions'] == len(report['excluded_transitions']) > 0
    assert {row['reason'] for row in report['excluded_transitions']} == {'independent_settlement_missing'}
    assert read(tmp_path/'pending/targets.json') == []


def test_confirmation_cannot_be_reused_as_B_gain(chain, tmp_path):
    root, _ = chain
    with pytest.raises(ValueError, match='Only registered independent B'):
        build_official_context_gain_from_runs([root/'C/run'], {}, tmp_path/'invalid',
                                             predictor_path=root/'f.json', as_of=Clock.now(UTC))


def test_report_distinguishes_training_base_and_learned_query_gain(chain):
    root, _ = chain
    report = read(root/'C/run/analysis.json')
    assert report['analysis_mode'] == 'official_optional_context'
    p = report['probabilities']
    assert p['original_prior'] is None and p['model_base'] == [.5, .25, .25]
    assert p['initial_probability_basis'] == 'independent-training-fixture-label-frequency-v1'
    assert p['base_source']['physical_matches'] == 12 and not p['calibration_established']
    serialized = json.dumps(report['decisions'])
    assert 'conditional_empirical_signed_gain_minus_declared_cost' in serialized
    assert 'expected_predictor_gain' in serialized
    assert '本次没有市场先验' in (root/'C/run/report.md').read_text(encoding='utf-8')
