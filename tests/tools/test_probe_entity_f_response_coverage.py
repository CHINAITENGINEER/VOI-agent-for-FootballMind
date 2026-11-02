import numpy as np
import pytest

from football_research.acquisition.entity_semantic_f import EntitySemanticF
from football_research.acquisition.semantic_f import FEATURES
from tools.probe_entity_f_response_coverage import probe_model


def model():
    return np.zeros((8 * len(FEATURES), 3))


def test_zero_model_and_common_logit_shifts_have_no_probability_response():
    weights = model()
    weights[2 * len(FEATURES)] = [2., 2., 2.]
    report = probe_model(EntitySemanticF(weights.tolist(), {}))
    assert report['body_features_with_class_contrast'] == 0
    assert report['body_features_responding_in_probes'] == 0
    assert all(r['feature_vector_changed'] for r in report['probes'])


def test_count_probe_detects_effect_hidden_by_first_entity_cancellation():
    weights = model()
    n = len(FEATURES)
    weights[2*n] = [.3, -.1, -.2]
    weights[3*n] = [-.3, .1, .2]
    report = probe_model(EntitySemanticF(weights.tolist(), {}))
    relevant = [r for r in report['probes'] if r['feature'] == FEATURES[0]]
    assert report['body_features_responding_in_probes'] == 1
    assert all(r['max_probability_change'] == pytest.approx(0.)
               for r in relevant if r['entity_count_after'] == 1)
    assert all(r['max_probability_change'] > .01
               for r in relevant if r['entity_count_after'] == 2)


def test_prior_interaction_is_not_missed_by_neutral_prior():
    weights = model()
    weights[6 * len(FEATURES)] = [.2, -.1, -.1]
    report = probe_model(EntitySemanticF(weights.tolist(), {}))
    relevant = [r for r in report['probes'] if r['feature'] == FEATURES[0]
                and r['entity_count_after'] == 1]
    assert relevant[0]['max_probability_change'] == pytest.approx(0.)
    assert relevant[1]['max_probability_change'] > 0
    assert relevant[2]['max_probability_change'] > 0


def test_probe_does_not_mutate_f_or_claim_gain():
    frozen = EntitySemanticF(model().tolist(), {'scope': 'test'})
    payload = frozen.payload()
    report = probe_model(frozen)
    assert frozen.payload() == payload
    assert report['constructed_probe_count'] == len(FEATURES) * 3 * 2
    assert not any('gain' in row or 'outcome' in row for row in report['probes'])
