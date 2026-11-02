import numpy as np
from tools.check_entity_f_body_controls import run_controls
from football_research.acquisition.entity_semantic_f import EntitySemanticF
from football_research.acquisition.semantic_f import FEATURES


def test_controls_distinguish_missing_response_and_protocol_errors():
    n = len(FEATURES)
    weights = np.zeros((8*n, 3))
    weights[2*n + FEATURES.index('home:availability:unavailable')] = [-.02, 0., .02]
    model = EntitySemanticF(weights.tolist(), {})
    payload = model.payload()
    report = run_controls(model)
    assert report['passed']
    assert all(r['body_consumed'] for r in report['positive_and_negative_controls'])
    assert all(r['gain'] is None and not r['g_training_eligible'] for r in report['positive_and_negative_controls'])
    assert all(r['rejected'] and r['correct_rejection'] for r in report['rejection_controls'])
    assert model.payload() == payload


def test_all_zero_model_does_not_pass_positive_control():
    report = run_controls(EntitySemanticF(np.zeros((8*len(FEATURES), 3)).tolist(), {}))
    assert not report['passed']
    positive = report['positive_and_negative_controls'][0]
    assert positive['body_consumed'] and positive['changed_body_features']
    assert not positive['check_passed']
