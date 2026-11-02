"""Independent arithmetic checks of constructed g weights and runtime cost."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone, timedelta
from hashlib import sha256
import json
import socket

import numpy as np
import pytest

from football_agent.context_voi_policy import ContextVOIPolicy
from football_agent.learned_voi_web import _web_action
from football_research.acquisition.context_gain import _query_input, _digest
from football_research.acquisition.context_value import ContextValueModel, _bundles, PARAMETERS
from football_research.acquisition.visible_state import encode_visible_state
from football_research.data_adapters.prospective_text import MarketPriorPolicy
from test_context_value import models, G_AS_OF, CONTEXTS, ConstructedFitClock
from test_prospective_training_integration import _source_inputs


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        pytest.fail('Independent value checks must not use HTTP or DNS')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'getaddrinfo', denied)
    monkeypatch.setattr('football_research.acquisition.context_value.datetime', ConstructedFitClock)


def test_physical_weight_and_ridge_normal_equation_independent_recalculation(models, tmp_path):
    _, fpath, g, root = models
    rows, _, _, _, _ = _bundles(root/'gains', predictor_path=fpath, as_of=G_AS_OF)
    counts = Counter(row['physical_match_id'] for row in rows)
    weights = np.array([1./(len(counts)*counts[row['physical_match_id']]) for row in rows])
    matrix = np.array([g._vector(row['context'])[0] for row in rows])
    gain = np.array([row['gain'] for row in rows])
    penalty = np.eye(matrix.shape[1])*PARAMETERS['regularization']
    penalty[0, 0] = 0.
    beta = np.array(g.coefficients)
    np.testing.assert_allclose(matrix.T@(weights*(gain-matrix@beta)), penalty@beta, rtol=0, atol=1e-12)
    objective = float(np.sum(weights*(gain-matrix@beta)**2)+beta@penalty@beta)
    assert objective == pytest.approx(g.metadata['objective'], abs=1e-12)

    # Distinct logged transition IDs of one match must share its one match mass.
    # This is a constructed source only; real records require the raw replay bridge.
    folder = tmp_path/'unequal-trajectories'
    folder.mkdir()
    for path in (root/'gains').glob('*.json'):
        (folder/path.name).write_bytes(path.read_bytes())
    features = json.loads((folder/'features.json').read_text(encoding='utf-8'))
    targets = json.loads((folder/'targets.json').read_text(encoding='utf-8'))
    for index in range(7):
        a, b = deepcopy(features[0]), deepcopy(targets[0])
        a['sample_id'] = b['sample_id'] = a['sample_id']+':another-actual-transition:'+str(index)
        features.append(a)
        targets.append(b)
    report = json.loads((folder/'report.json').read_text(encoding='utf-8'))
    report.update(features_sha256=_digest(features), targets_sha256=_digest(targets))
    for name, value in [('features', features), ('targets', targets), ('report', report)]:
        (folder/(name+'.json')).write_text(json.dumps(value), encoding='utf-8')
    alternate = ContextValueModel().fit(folder, predictor_path=fpath, as_of=G_AS_OF,
                                      dataset_version='constructed-independent-unequal-trajectories')
    assert alternate.metadata['readiness']['physical_matches'] == 24
    assert alternate.metadata['readiness']['transitions'] == 31
    assert alternate.encoder == g.encoder
    np.testing.assert_allclose(alternate.coefficients, g.coefficients, rtol=0, atol=1e-12)


def test_policy_subtracts_resource_price_once_from_same_f_empirical_gain(models, tmp_path):
    f, fpath, g, _ = models
    at = datetime(2027, 2, 10, 10, tzinfo=timezone.utc)
    source = _source_inputs(tmp_path/'source', at=at, local_id=101, event_id=2050992)
    original = MarketPriorPolicy.from_manifest(source['plan']['selected'][0])
    base = MarketPriorPolicy(original.model.fixture, original.model.quote,
        frozen_at=original.model.frozen_at, prediction_cutoff=original.model.cutoff,
        lambda_cost={'money': 7., 'requests': .003})
    policy = ContextVOIPolicy(base, f, physical_match_id='heldout-cost-check', value_model=g,
                             predictor_artifact=fpath.read_bytes())
    text = CONTEXTS['positive']
    doc = {'document_id': 'known-article', 'raw_sha256': 'a'*64,
        'url': 'https://www.uefa.com/news/known/', 'status': 'OK', 'kind': 'article',
        'text': text, 'links': [], 'source_published_at': None,
        'observed_at': at.isoformat(), 'available_at': at.isoformat(),
        'structured_news': {'version': 'structured-news-discovery-v3', 'role': 'article',
            'segments': [{'text': text, 'start': 0, 'end': len(text),
                'content_source': 'json_ld_article_body', 'json_pointer': 'script[0].articleBody'}]}}
    budget = {'limits': {'money': 1., 'requests': 3.}, 'spent': {}}
    public = {'state': policy.public_state(base.model.initial), 'documents': [doc], 'attempts': [],
        'decision_at': (at+timedelta(seconds=1)).isoformat(), 'budget': budget,
        'prediction': base.model.predict_partial(base.model.initial)}
    public['prediction'] = policy.predict_public(public)
    action = _web_action('read_url', 'https://www.uefa.com/news/actual/',
                         cost={'money': .01, 'requests': 1.})
    query = _query_input(encode_visible_state(public, [action]), action,
                         public['prediction']['probabilities'])
    expected = g.predict(query, physical_match_id='heldout-cost-check',
                         predictor_sha256=sha256(fpath.read_bytes()).hexdigest())
    assert expected['reason'] == 'supported_empirical_context'
    assert expected['support_status'] == 'supported' and expected['cost_deducted'] is False
    result = policy.plan_public(public, [action], budget)
    value = result['candidates'][0]['value']
    assert value['expected_signed_gain'] == expected['expected_signed_gain']
    assert value['cost'] == pytest.approx(.073, abs=1e-15)
    assert value['net_voi'] == pytest.approx(expected['expected_signed_gain']-.073, abs=1e-15)
    assert value['gross_evsi'] is None and result['selection']['stop_certificate'] is None
