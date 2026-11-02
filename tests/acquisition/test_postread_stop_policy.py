import numpy as np
import pytest
from tools.train_postread_stop_policy import choose_threshold, temporal_folds, run_policy


def test_one_se_prefers_zero_when_improvement_uncertain():
    selected, best, _ = choose_threshold({0.: [1., 1.], -.01: [.5, 1.4]})
    assert best['threshold'] == -.01
    assert selected['threshold'] == 0.


def test_threshold_changes_when_gain_consistent():
    selected, _, _ = choose_threshold({0.: [1., 1.], -.01: [.8, .8]})
    assert selected['threshold'] == -.01
    with pytest.raises(ValueError):
        choose_threshold({0.: [np.nan, 1.]})


def test_forward_dates_and_duplicate_purge():
    panels = [{'match_id': str(i), 'date': f'2024-01-{i+1:02}',
               'candidates': [{'body_sha256': 'shared' if i in (0, 3) else str(i)}]} for i in range(12)]
    for history, heldout in temporal_folds(panels):
        assert max(p['date'] for p in history) < min(p['date'] for p in heldout)
        a = {c['body_sha256'] for p in history for c in p['candidates']}
        b = {c['body_sha256'] for p in heldout for c in p['candidates']}
        assert not a & b


def test_threshold_only_changes_stop_not_gain_or_pool(monkeypatch):
    monkeypatch.setattr('tools.train_postread_stop_policy.score', lambda model, x: np.array([-.005]))
    class Inputs:
        def vector(self, *args):
            return np.array([1.])
    panel = [{'match_id': 'm', 'candidates': [{'article_id': 'a'}]}]
    states = {'m': {(): np.array([.4, .3, .3]), ('a',): np.array([.5, .2, .3])}}
    stop = run_policy(Inputs(), panel, states, {}, 0.)[0]
    keep = run_policy(Inputs(), panel, states, {}, -.01)[0]
    assert stop['f_adopted_ids'] == []
    assert keep['f_adopted_ids'] == ['a']
    assert stop['trace'][0]['estimated_gains'] == keep['trace'][0]['estimated_gains'] == [-.005]
    assert stop['trace'][0]['g_seen_ids'] == ['a']
