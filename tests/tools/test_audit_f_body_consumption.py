from copy import deepcopy

import numpy as np
import pytest

from football_research.acquisition.entity_semantic_f import EntityEvidenceBank, EntitySemanticF
from football_research.acquisition.semantic_f import FEATURES
from tests.research.acquisition.test_semantic_f import state, article, record
from tools.audit_f_body_consumption import audit_pair


def run_pair(*, scope='this_match', empty=False, weights=True, duplicate=False):
    before = state()
    doc = article()
    records = [record(before, doc, 'title', empty=True),
               record(before, doc, 'body', scope=scope, empty=empty)]
    after = deepcopy(before)
    after['documents'] = [doc]
    if duplicate:
        before['documents'] = [doc]
    coefficients = np.zeros((8*len(FEATURES), 3))
    if weights:
        coefficients[2*len(FEATURES)+FEATURES.index('home:availability:unavailable'), 0] = -.7
    return audit_pair(EntitySemanticF(coefficients.tolist(), {}), EntityEvidenceBank(records), before, after)


def test_body_consumption_is_distinct_from_effect_and_gain():
    result = run_pair(scope='team_context')
    assert result['bodies'][0]['semantic_input_consumed'] is True
    assert result['bodies'][0]['claim_count'] == 1
    assert result['bodies'][0]['supported_claim_count'] == 0
    assert result['reason'] == 'body_consumed_but_outside_f_feature_support'
    assert result['gain'] is None and not result['optimality_established']


def test_supported_novel_body_reaches_f_and_changes_prediction():
    result = run_pair()
    assert result['reason'] == 'body_changed_features_and_prediction'
    assert result['prediction_after'][0] < result['prediction_before'][0]
    assert result['g_training_rows_emitted'] == 0


def test_zero_coefficients_are_not_misreported_as_missing_body():
    assert run_pair(weights=False)['reason'] == 'features_changed_but_model_response_negligible'


def test_empty_extraction_and_repeat_have_distinct_reasons():
    assert run_pair(empty=True)['reason'] == 'body_consumed_no_extracted_claims'
    assert run_pair(duplicate=True)['reason'] == 'no_new_body'


def test_missing_extraction_fails_instead_of_becoming_zero_gain():
    before = state()
    after = deepcopy(before)
    after['documents'] = [article()]
    model = EntitySemanticF(np.zeros((8*len(FEATURES), 3)).tolist(), {})
    with pytest.raises(ValueError, match='SEMANTIC_EXTRACTION_MISSING'):
        audit_pair(model, EntityEvidenceBank([]), before, after)


def test_changed_prior_is_rejected():
    before = state()
    after = deepcopy(before)
    after['prior_probabilities'] = [.5, .2, .3]
    model = EntitySemanticF(np.zeros((8*len(FEATURES), 3)).tolist(), {})
    with pytest.raises(ValueError, match='fixed-prior'):
        audit_pair(model, EntityEvidenceBank([]), before, after)
