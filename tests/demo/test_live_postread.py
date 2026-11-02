"""Online control flow, without external requests or learned-model substitutions in production."""
from copy import deepcopy
from hashlib import sha256

import numpy as np
import pytest

from demo import live_postread as live

STAMP = '2026-10-25T10:00:00+00:00'
KICK = '2026-10-25T19:00:00+00:00'
BASE = 'https://www.tottenhamhotspur.com/news/2026/october/'


class Model:
    head = live.DEFAULT_HEAD
    threshold = .01
    bindings = {'f_model_sha256': live.F_SHA, 'g_model_sha256': live.G_SHA}
    def __init__(self, *, stop=False, fail_encode=False, fail_after=False):
        self.stop, self.fail_encode, self.fail_after = stop, fail_encode, fail_after
        self.calls = []
    def initial_prediction(self, fixture, prior):
        self.calls.append(('initial', []))
        return {'probabilities': [.6, .25, .15]}
    def encode_document(self, doc, *, max_input_tokens=None):
        self.calls.append(('encode', doc['document_id']))
        if self.fail_encode:
            raise ValueError('test encoding failure')
        v = np.zeros(1024, dtype=np.float32); v[0] = 1
        return {'document': deepcopy(doc), 'vector': v, 'encoding': {'text_sha256': doc['text_sha256'],
            'vector_file_sha256': 'a'*64, 'cache_hit': False, 'inference_seconds': .01}}
    def evaluate(self, fixture, prior, docs, vectors, adopted, budget, cost_loss_units):
        self.calls.append(('score', [d['document_id'] for d in docs]))
        remaining = [i for i in range(len(docs)) if i not in adopted]
        scores = [{'index': i, 'score': -.1 if self.stop else .1, 'document_id': docs[i]['document_id']} for i in remaining]
        reason = 'budget_exhausted' if len(adopted) >= budget else 'candidate_pool_exhausted' if not remaining else 'predicted_gain_below_frozen_threshold' if self.stop else None
        return {'selected_index': remaining[0] if reason is None else None, 'candidate_scores': scores,
            'stop_reason': reason, 'prediction': {'probabilities': [.6+.01*len(adopted), .25-.01*len(adopted), .15]}}
    def predict(self, fixture, prior, docs, vectors, adopted):
        self.calls.append(('adopt', list(adopted)))
        if self.fail_after:
            raise ValueError('test f failure')
        return {'probabilities': [.6+.01*len(adopted), .25-.01*len(adopted), .15],
            'used_input_ids': [docs[i]['document_id'] for i in adopted],
            'f_model_sha256': live.F_SHA, 'f_executed': True}


class Tools:
    def __init__(self, backend): self.backend = backend
    def prepare(self, action, public): return action
    def execute(self, action, **kw):
        self.backend.requests.append(action)
        body = action.get('url') in [BASE+'team-news-a', BASE+'team-news-b']
        text = 'Different observed body ' + action['url'] if body else 'Index'
        return {'document': {'document_id': action.get('url', 'search')+str(len(self.backend.requests)),
            'status': 'OK', 'url': action.get('url', 'https://www.bing.com/search'),
            'observed_at': STAMP, 'text': text, 'raw_sha256': sha256(text.encode()).hexdigest(),
            'source_published_at': STAMP, 'structured_news': {'role': 'article' if body else 'index'},
            'links': [] if body else [{'url': BASE+'team-news-'+s, 'title': 'Tottenham first team news '+s,
                'source_published_at': STAMP} for s in ['a', 'b']]}}


class Backend:
    def __init__(self): self.requests = []
    def initialize(self, match, folder):
        return {'market_initialization': {'prediction_cutoff': '2026-10-25T18:59:00+00:00', 'probabilities': [.6, .25, .15]}}
    def identity(self, manifest, folder, expected):
        return {'canonical': {'source_event_id': 'test-match', 'home_name': 'Tottenham Hotspur',
            'away_name': 'Coventry City', 'kickoff': KICK}, 'aliases': {'home': ['Tottenham Hotspur'], 'away': ['Coventry City']}}
    def tools(self, *args): return Tools(self)
    def semantics(self, *args): pytest.fail('No external semantic API in the new vector pipeline')


def application(tmp_path, model=None):
    app = live.LivePostreadApplication(tmp_path, backend=Backend(), clock=lambda: STAMP, model=model or Model())
    live.save(tmp_path/'catalog.json', {'status': 'ready', 'updated_at': STAMP, 'matches': [{
        'match_id': 'fixture-test', 'fixture_id': 'fixture-test', 'source_match_id': 'test',
        'home': 'Tottenham Hotspur', 'away': 'Coventry City', 'kickoff': KICK, 'supported': True}]})
    return app


def run(app, **payload):
    record = app.start({'match_id': 'fixture-test', **payload})
    app.workers[record['run_id']].join(10)
    assert not app.workers[record['run_id']].is_alive()
    return app.get(record['run_id'])


def test_all_bodies_encoded_before_g_and_only_selected_enter_f(tmp_path, monkeypatch):
    monkeypatch.setattr('demo.live_g.key_from_environment', lambda: None)
    app = application(tmp_path)
    result = run(app, max_reads=1)
    assert result['state'] == 'STOPPED', result
    assert result['counts']['encoded'] == 2
    assert len(result['steps']) == 1
    calls = [kind for kind, _ in app.model.calls]
    assert max(i for i, kind in enumerate(calls) if kind == 'encode') < calls.index('score') < calls.index('adopt')
    assert result['steps'][0]['input_mode'] == 'qwen_fulltext_embedding'
    assert result['steps'][0]['input_id'] == sha256(result['steps'][0]['body'].encode()).hexdigest()
    assert result['counts']['provider'] == 0
    assert sum(a['purpose'] == 'search_discovery' for a in result['attempts']) == 5
    assert result['outcome'] is None and result['gain'] is None
    assert app.catalog()['pipeline_mode'] == live.MODE
    assert app.catalog()['provider_required'] is False


def test_g_stop_keeps_acquired_bodies_without_f_after(tmp_path):
    result = run(application(tmp_path, Model(stop=True)))
    assert result['counts']['encoded'] == 2 and len(result['candidate_pool']) == 2
    assert result['steps'] == [] and result['current'] == result['before']
    assert all(p['prediction_after'] is None and not p['f_adopted'] for p in result['candidate_pool'])
    assert result['reason'] == 'predicted_gain_below_frozen_threshold'


@pytest.mark.parametrize('flag', ['fail_encode', 'fail_after'])
def test_failed_computation_never_commits_fake_after(tmp_path, flag):
    result = run(application(tmp_path, Model(**{flag: True})))
    assert result['steps'] == [] and result['used_input_ids'] == []
    assert result['current'] == result['before']
    assert all(p['prediction_after'] is None for p in result['candidate_pool'])
    assert result['gain'] is None
    assert result['state'] == ('FAILED' if flag == 'fail_after' else 'STOPPED')


def test_same_f_prefix_value_checked_before_next_commit(tmp_path):
    result = run(application(tmp_path), max_reads=20)
    assert len(result['steps']) == 2
    assert result['steps'][1]['prediction_before'] == result['steps'][0]['prediction_after']
    assert len(result['used_input_ids']) == 2
    assert result['effective_adoption_limit'] == 2


def test_missing_consumption_ids_cannot_commit_an_after(tmp_path):
    class MissingIDs(Model):
        def predict(self, *args):
            value = super().predict(*args)
            value.pop('used_input_ids')
            return value
    result = run(application(tmp_path, MissingIDs()))
    assert result['state'] == 'FAILED'
    assert result['steps'] == [] and result['current'] == result['before']
    assert result['gain'] is None


def test_publication_conflict_is_not_filled_from_directory(tmp_path, monkeypatch):
    original = Tools.execute
    def execute(self, action, **kwargs):
        value = original(self, action, **kwargs)
        d = value['document']
        if d['structured_news']['role'] == 'article':
            d.update(source_published_at=None, publication_status='source_publication_date_conflict',
                source_publication_date='2026-10-24',
                live_dutch_body_projection={'source_metadata_published_at': STAMP})
        return value
    monkeypatch.setattr(Tools, 'execute', execute)
    result = run(application(tmp_path))
    assert result['counts']['encoded'] == 2
    for row in result['candidate_pool']:
        assert row['published_at'] is None and row['directory_published_at'] == STAMP
        assert row['source_publication_metadata'] == STAMP


def test_category_navigation_and_other_squad_are_kept_as_rejected_menu(tmp_path):
    app = application(tmp_path)
    record = {'fixture': {'home_name': 'AZ Alkmaar', 'away_name': 'FC Twente'}, 'menus': []}
    links = [
        {'url': 'https://www.az.nl/inside-az/categorieen/laatste-nieuws', 'title': 'Laatste Nieuws'},
        {'url': 'https://www.az.nl/inside-az/nieuws/2026/oktober/jong-az', 'title': 'Jong AZ tegen Helmond'},
        {'url': 'https://www.az.nl/inside-az/nieuws/2026/oktober/preview-az-twente', 'title': 'AZ - FC Twente'}]
    candidates = []
    app._admit_links(record, {'links': links}, ['https://www.az.nl/inside-az/nieuws'], set(), candidates)
    assert len(record['menus']) == 3
    assert [r['admission_reason'] for r in record['menus'][:2]] == ['directory_category_navigation', 'directory_other_squad_label']
    assert [r['url'] for r in candidates] == [links[2]['url']]


def test_verified_ceuta_card_accepts_root_slug_but_not_other_squad(tmp_path):
    app = application(tmp_path)
    record = {'fixture': {'home_name': 'Cádiz', 'away_name': 'Ceuta'}, 'menus': []}
    links = [{'url': 'https://adceutafc.com/andres-fuentes-dirigira-el-cadiz-cf-ad-ceuta-fc/',
        'title': 'Andrés Fuentes dirigirá el Cádiz CF – AD Ceuta FC',
        'directory_metadata_adapter': 'footballmind-live-ceuta-observed-news-cards-v1'},
        {'url': 'https://adceutafc.com/ceuta-femenino-team-news/',
        'title': 'La AD Ceuta Femenino logra un punto',
        'directory_metadata_adapter': 'footballmind-live-ceuta-observed-news-cards-v1'}]
    candidates = []
    app._admit_links(record, {'links': links}, ['https://adceutafc.com/noticias/'], set(), candidates)
    assert len(candidates) == 1 and candidates[0]['url'] == links[0]['url']
    assert record['menus'][1]['admission_reason'] == 'directory_other_squad_label'


def test_five_minute_start_margin_keeps_market_and_kickoff_checks():
    from demo.live_coverage import availability
    match = {'status': 'scheduled', 'supported': True, 'kickoff': KICK, 'quote_observed': True}
    assert availability(match, '2026-10-25T18:40:00+00:00')[0] is False
    assert availability(match, '2026-10-25T18:40:00+00:00', minimum_lead_minutes=5)[0] is True
    assert availability(match, '2026-10-25T18:56:00+00:00', minimum_lead_minutes=5)[0] is False
    match['quote_observed'] = False
    assert availability(match, '2026-10-25T18:40:00+00:00', minimum_lead_minutes=5)[0] is False


@pytest.mark.parametrize('limit', [0, 21, 1.2, True])
def test_invalid_limits_do_not_start_collection(tmp_path, limit):
    app = application(tmp_path)
    with pytest.raises(ValueError): app.start({'match_id': 'fixture-test', 'max_reads': limit})
    assert app.workers == {} and app.backend.requests == []
