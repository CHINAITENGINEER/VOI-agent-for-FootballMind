from copy import deepcopy
import json

import pytest

from footballmind.reproduction import REFERENCE, verify_predictions
from footballmind.__main__ import main


@pytest.fixture
def replay(tmp_path):
    rows = deepcopy(json.loads(REFERENCE.read_text(encoding='utf-8'))['predictions'])
    path = tmp_path / 'predictions.jsonl'
    def save(values):
        path.write_text(''.join(json.dumps(v) + '\n' for v in values), encoding='utf-8')
        return path
    save(rows)
    return rows, path, save


def test_reference_matches_and_tolerates_small_float_rounding(replay):
    rows, path, save = replay
    rows[0]['prediction']['p_after'][0] += 1e-8
    rows[0]['prediction']['p_after'][1] -= 1e-8
    report = verify_predictions(save(rows))
    assert report['status'] == 'PASS' and report['matched_predictions'] == 3


def test_changed_adoption_rejected_even_with_identical_probabilities(replay):
    rows, path, save = replay
    rows[0]['prediction']['selected_ids'] = ['different-article']
    assert verify_predictions(save(rows))['status'] == 'FAIL'
    assert main(['verify', str(path)]) == 1


def test_changed_model_binding_rejected(replay):
    rows, _, save = replay
    rows[0]['prediction']['model_bindings']['f_sha256'] = '0' * 64
    assert verify_predictions(save(rows))['status'] == 'FAIL'


def test_nonfinite_probability_rejected(replay):
    rows, _, save = replay
    rows[0]['prediction']['p_after'][0] = float('nan')
    assert verify_predictions(save(rows))['status'] == 'FAIL'


def test_missing_strategy_and_duplicate_records_rejected(replay):
    rows, _, save = replay
    assert verify_predictions(save(rows[:-1]))['status'] == 'FAIL'
    with pytest.raises(ValueError, match='Duplicate'):
        verify_predictions(save([*rows, rows[0]]))


def test_extra_matches_are_counted_but_not_claimed_as_verified(replay):
    rows, _, save = replay
    extra = deepcopy(rows[0]); extra['case_id'] = 'another-match'
    report = verify_predictions(save([*rows, extra]))
    assert report['status'] == 'PASS'
    assert report['reference_predictions'] == 3 and report['additional_rows_not_compared'] == 1
