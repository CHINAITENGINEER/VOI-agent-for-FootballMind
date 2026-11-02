from copy import deepcopy
import numpy as np
import pytest
from football_research.acquisition.compact_evidence_f import features, predict, fit
from football_research.acquisition.entity_semantic_f import REPRESENTATION_SHA256
from football_research.acquisition.semantic_f import FEATURES


def state(mid='m', prior=None):
    return {'physical_match_id': mid, 'representation_sha256': REPRESENTATION_SHA256,
            'prior': prior or [.4, .3, .3],
            **{k: [0.] * len(FEATURES) for k in
               ('title', 'novel_body', 'title_entity_counts', 'novel_body_entity_counts')}}


def mention(s, key, channel='novel_body', count=1):
    i = FEATURES.index(key)
    s[channel][i] = 1.
    s[channel + '_entity_counts'][i] = float(count)


def test_no_evidence_is_exact_prior_with_any_coefficients():
    assert np.allclose(predict([2, -3, 4, 5], state()), [.4, .3, .3])


def test_home_away_equivariance():
    a = state(prior=[.6, .25, .15]); b = state(prior=[.15, .25, .6])
    mention(a, 'home:availability:unavailable'); mention(b, 'away:availability:unavailable')
    assert np.allclose(predict([.4, .2, -.1, .3], a)[::-1], predict([.4, .2, -.1, .3], b))


def test_duplicate_titles_and_entities_do_not_inflate_presence():
    a = state(); mention(a, 'away:availability:doubtful')
    b = deepcopy(a); mention(b, 'away:availability:doubtful', count=8)
    mention(b, 'away:availability:doubtful', channel='title')
    assert np.array_equal(features(a), features(b))


def test_equal_match_weights_are_invariant_to_whole_match_row_replication():
    a, b = state('a'), state('b')
    mention(a, 'home:availability:unavailable'); mention(b, 'away:availability:unavailable')
    r1 = {'physical_match_id': 'a', 'state': a, 'outcome': 2}
    r2 = {'physical_match_id': 'b', 'state': b, 'outcome': 0}
    beta, _ = fit([r1, r2], 1.)
    more, _ = fit([r1] * 9 + [r2], 1.)
    assert np.allclose(beta, more, atol=1e-8)
    assert predict(beta, a)[2] > a['prior'][2]


def test_conflicting_labels_and_invalid_coefficients_rejected():
    s = state()
    with pytest.raises(ValueError, match='Consistent'):
        fit([{'physical_match_id': 'm', 'state': s, 'outcome': y} for y in (0, 1)], 1.)
    with pytest.raises(ValueError, match='finite'):
        predict([float('nan')] * 4, s)


def test_base_only_and_unsupported_tactics():
    s = state(); mention(s, 'home:tactical_plan:attacking')
    beta, _ = fit([{'physical_match_id': 'm', 'state': s, 'outcome': 0}], None)
    assert beta == [0.] * 4
    assert not features(s).any()
