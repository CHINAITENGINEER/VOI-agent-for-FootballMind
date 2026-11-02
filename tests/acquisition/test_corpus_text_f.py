import numpy as np
import pytest
from scipy.optimize import check_grad

from football_research.acquisition.corpus_text_f import (
    scope_exclusion, make_panel, objective, fit_residual, predict_residual, loss_each,
)


@pytest.mark.parametrize('title', ['U18s preview: Arsenal v Tottenham', 'Liverpool U21s appearance',
    'Youth internationals', "Women's team news", 'Under-18s preview'])
def test_other_team_scope_is_not_mens_match_evidence(title):
    assert scope_exclusion({'title': title})


def test_regular_preview_retained():
    assert scope_exclusion({'title': 'Liverpool v Tottenham: Team news'}) is None


def test_shared_article_across_splits_and_same_day_removed_without_labels():
    matches = {'a': {'match_date': '2025-06-30'}, 'b': {'match_date': '2025-07-02'},
               'c': {'match_date': '2026-02-01'}, 'd': {'match_date': '2026-02-02'}}
    articles = {k: {'title': 'Team news', 'body_sha256': sha, 'published_at': '2025-06-29'}
                for k, sha in [('x', 'shared'), ('y', 'shared'), ('z', 'separate'), ('w', 'today')]}
    links = [{'match_id': mid, 'article_id': aid, 'days_before_match_date': days}
             for mid, aid, days in [('a', 'x', 1), ('b', 'y', 3), ('c', 'z', 1), ('d', 'w', 0)]]
    panel, rejected = make_panel(matches, links, articles)
    assert [p['match_id'] for p in panel] == ['c']
    assert [r['reason'] for r in rejected].count('shared_body_cluster_crosses_split') == 2
    assert any(r['reason'] == 'same_day_kickoff_order_unknown' for r in rejected)


def test_neural_objective_gradient():
    rng = np.random.default_rng(19)
    x = rng.normal(size=(5, 3)); base = np.tile([.4, .3, .3], (5, 1))
    y = np.array([0, 1, 2, 0, 2]); w = np.ones(5)/5
    theta = rng.normal(0, .1, size=3*2+2+2*3+3)
    args = (x, base, y, w, np.array([1., 1., 0., 1., 1.]), 2, .01)
    error = check_grad(lambda v: objective(v, *args)[0], lambda v: objective(v, *args)[1], theta)
    assert error < 1e-6


def test_model_learns_response_and_empty_evidence_preserves_prior_exactly():
    # Constructed behavior test only; does not enter the real training corpus.
    x = np.array([[-1.], [-.9], [0.], [.1], [.9], [1.]])
    y = np.array([0, 0, 1, 1, 2, 2]); base = np.tile([.4, .3, .3], (6, 1))
    m = fit_residual(x, base, y, np.ones(6)/6, alpha=.001, hidden=4)
    p = predict_residual(m, x, base, np.ones(6))
    assert loss_each(p, y).mean() < loss_each(base, y).mean()
    assert np.array_equal(predict_residual(m, x, base, np.zeros(6)), base)
    assert np.allclose(p.sum(1), 1)
    with pytest.raises(ValueError):
        predict_residual(m, x, base, [-1]*6)
