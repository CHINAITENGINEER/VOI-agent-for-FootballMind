from copy import deepcopy
import json

import numpy as np
import pytest

from football_research.acquisition import neural_state_gain as neural
from football_research.acquisition import state_candidate_gain as linear
from football_research.acquisition import historical_gain_head as base


def rows():
    snapshot = {'physical_match_id': 'observed', 'used_input_ids': [], 'claims': []}
    result = []
    for i, (title, gain) in enumerate([
        ('injury update striker doubtful', .03), ('tickets travel supporter news', -.02),
        ('official starting lineup team news', 0.), ('injury midfielder returns', .015),
    ]):
        public = {'schema': base.QUERY_SCHEMA, 'probabilities': [.4, .3, .3],
                  'acquired_documents': 0, 'visible_candidates': 4,
                  'article_age_hours': 1., 'title': title}
        result.append({'physical_match_id': str(i), 'query': linear.build_query(public, snapshot),
                       'signed_gain': gain})
    return result


@pytest.fixture(scope='module')
def trained():
    data = rows()
    representation = linear.fit(data)
    return data, representation, neural.fit(data, representation)


def test_backprop_matches_numeric_gradient():
    rng = np.random.default_rng(42)
    dimension = 2
    count = dimension*16+16+16*8+8+8+1
    theta = rng.normal(0, .1, count)
    x, y = rng.normal(size=(5, dimension)), rng.normal(size=5)
    weights = np.array([.1, .1, .2, .3, .3])
    loss, gradient = neural.objective(theta, x, y, weights, .01)
    assert np.isfinite(loss)
    numeric = np.zeros(count)
    for i in range(count):
        plus, minus = theta.copy(), theta.copy()
        plus[i] += 1e-6
        minus[i] -= 1e-6
        numeric[i] = (neural.objective(plus, x, y, weights, .01)[0]
                      -neural.objective(minus, x, y, weights, .01)[0])/2e-6
    np.testing.assert_allclose(gradient, numeric, atol=1e-8, rtol=1e-5)


def test_54_input_parameter_count():
    arrays = neural.unpack(np.zeros(1025), 54)
    assert [a.shape for a in arrays] == [(54, 16), (16,), (16, 8), (8,), (8, 1), (1,)]
    with pytest.raises(ValueError):
        neural.unpack(np.zeros(1026), 54)


def test_signed_output_roundtrip_and_tamper(trained, tmp_path):
    data, _, model = trained
    assert neural.estimate(model, data[0]['query']) > 0
    assert neural.estimate(model, data[1]['query']) < 0
    assert model['online_selection_allowed'] is False
    path = tmp_path/'neural.json'
    neural.save(model, path)
    restored = neural.load(path)
    assert [neural.estimate(model, r['query']) for r in data] == [neural.estimate(restored, r['query']) for r in data]
    payload = json.loads(path.read_text())
    payload['parameters'][0] += 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='digest'):
        neural.load(path)


@pytest.mark.parametrize('field', ['body', 'prediction_after', 'outcome', 'signed_gain'])
def test_hidden_query_fields_rejected(trained, field):
    data, _, model = trained
    q = deepcopy(data[0]['query'])
    q[field] = 'hidden'
    with pytest.raises(ValueError):
        neural.estimate(model, q)
    q = deepcopy(data[0]['query'])
    q['public'][field] = 'hidden'
    with pytest.raises(ValueError):
        neural.estimate(model, q)


def test_training_encoder_cohort_must_match(trained):
    data, representation, _ = trained
    with pytest.raises(ValueError, match='same training matches'):
        neural.fit(data[1:], representation)


def test_match_weighting_and_determinism(trained):
    data, representation, model = trained
    repeated = neural.fit([data[0]]*7+data[1:], representation)
    np.testing.assert_allclose(model['target_mean'], repeated['target_mean'], atol=1e-15)
    np.testing.assert_allclose(model['target_scale'], repeated['target_scale'], atol=1e-15)
    np.testing.assert_allclose([neural.estimate(model, r['query']) for r in data],
                               [neural.estimate(repeated, r['query']) for r in data], atol=1e-6)
    same = neural.fit(data, representation)
    np.testing.assert_array_equal(same['parameters'], model['parameters'])


def test_zero_targets_not_removed_or_invented():
    data = rows()
    for row in data:
        row['signed_gain'] = 0.
    model = neural.fit(data, linear.fit(data))
    assert model['fit_match_ids'] == ['0', '1', '2', '3']
    assert max(abs(neural.estimate(model, r['query'])) for r in data) < 1e-12


@pytest.mark.parametrize('value', [float('nan'), float('inf'), True])
def test_invalid_targets_rejected(trained, value):
    data, representation, _ = trained
    invalid = deepcopy(data)
    invalid[0]['signed_gain'] = value
    with pytest.raises(ValueError, match='Finite signed gains'):
        neural.fit(invalid, representation)
