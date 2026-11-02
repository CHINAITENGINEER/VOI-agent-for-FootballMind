import inspect
import numpy as np
import pytest

from football_research.acquisition.corpus_rolling_g import (
    subsets, match_weights, prequery_features, fit_gain, predict_gain, choose_action)
from tools.train_e_drive_rolling_g import purge


def query(**changes):
    args = dict(probabilities=[.5, .3, .2], read_count=0, menu_count=3,
                age_days=2, source_side=0, host_index=0, host_count=4,
                title_vector=np.ones(384)/np.sqrt(384), read_title_vectors=[])
    args.update(changes)
    return prequery_features(**args)


def test_all_edges_no_gain_gate():
    ss = subsets(['a', 'b', 'c'])
    edges = [(s, a) for s in ss for a in ['a', 'b', 'c'] if a not in s]
    assert len(ss) == 8 and len(edges) == 12
    assert (('a', 'b'), 'c') in edges
    with pytest.raises(ValueError):
        subsets(['a', 'a'])


def test_equal_match_weight():
    w = match_weights(['a', 'b', 'b', 'b'])
    assert w[0] == pytest.approx(.5)
    assert w[1:].sum() == pytest.approx(.5)


def test_query_contract_excludes_outcome_body_after():
    params = inspect.signature(prequery_features).parameters
    assert not {'body_vector', 'outcome', 'prediction_after', 'gain'}.intersection(params)
    assert query().shape == (399,)
    with pytest.raises(TypeError):
        prequery_features(body_vector=np.ones(384))


@pytest.mark.parametrize('changes', [
    {'probabilities': [50, 30, 20]}, {'probabilities': [.5, .3, float('nan')]},
    {'read_count': 3}, {'age_days': -1}, {'title_vector': np.ones(385)},
    {'read_count': 1, 'read_title_vectors': []}])
def test_invalid_query(changes):
    with pytest.raises(ValueError):
        query(**changes)


def test_negative_zero_positive_are_trainable_and_replayable():
    x = np.array([[-1., 0.], [0., 0.], [1., 0.]])
    y = np.array([-.1, 0., .1])
    model = fit_gain(x, y, ['a', 'b', 'c'], alpha=.1)
    p = predict_gain(model, x)
    assert p[0] < 0 and p[2] > 0
    assert np.mean((p-y)**2) < np.mean(y**2)
    selected, _ = choose_action(model, ['bad', 'good'], x[[0, 2]], 0.)
    assert selected == 'good'
    selected, _ = choose_action(model, ['bad'], x[[0]], 0.)
    assert selected is None
    selected, _ = choose_action(model, ['good'], x[[2]], 1.)
    assert selected is None


def test_body_cluster_purge_removes_full_training_match():
    history = [{'match_id': 'a', 'candidates': [{'body_sha256': 'same'}, {'body_sha256': 'other'}]},
               {'match_id': 'b', 'candidates': [{'body_sha256': 'safe'}]}]
    target = [{'candidates': [{'body_sha256': 'same'}]}]
    assert [p['match_id'] for p in purge(history, target)] == ['b']
