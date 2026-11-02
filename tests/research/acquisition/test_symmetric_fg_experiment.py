"""Boundary tests for the paired training experiment, not performance promises."""
from copy import deepcopy

import numpy as np
import pytest

from tools.train_symmetric_fg import public_row, scores


def sample(mid='m', y=0):
    return {'physical_match_id': mid, 'prior': [.5, .3, .2],
            'decision_at': '2025-01-01T12:00:00Z', 'used_input_ids': [],
            'evidence_events': [], 'outcome': y, 'result_available_at': 'later',
            'signed_gain': 999, 'prediction_after': [0, 0, 1]}


def test_external_labels_join_after_prediction_without_entering_predictor():
    seen = []
    r = sample(y=None)

    def predict(public):
        assert set(public) == {'physical_match_id', 'prior', 'decision_at',
                               'used_input_ids', 'evidence_events'}
        seen.append(deepcopy(public))
        return np.asarray(public['prior'])

    first = scores([r], predict, {'m': 0})
    second = scores([r], predict, {'m': 2})
    assert seen[0] == seen[1]
    assert first['loss'] == pytest.approx(-np.log(.5))
    assert second['loss'] == pytest.approx(-np.log(.2))
    assert r['outcome'] is None


def test_evaluation_weights_matches_not_expanded_state_count():
    predict = lambda r: np.asarray(r['prior'])
    assert scores([sample(), sample('n', 2)], predict) == scores(
        [sample()]*10 + [sample('n', 2)], predict)


def test_unknown_outcome_is_not_silently_a_zero_gain_label():
    with pytest.raises(ValueError, match='Settled outcome'):
        scores([sample(y=None)], lambda r: np.asarray(r['prior']))
    with pytest.raises(KeyError):
        scores([sample()], lambda r: np.asarray(r['prior']), {})


def test_public_inputs_are_not_mutable_aliases_of_training_record():
    original = sample()
    projected = public_row(original)
    projected['prior'][0] = .1
    assert original['prior'] == [.5, .3, .2]
