import json
from copy import deepcopy

import pytest

from football_research.acquisition.structured_source_handoff import validate_jsonl, validate_record


@pytest.fixture
def inputs(tmp_path):
    quote = 'The source reports probabilities 20%, 30%, 50%.'
    body = 'Heading\r\n'+quote+'\r\n'
    path = tmp_path/'body.txt'
    path.write_bytes(body.encode('utf-8'))
    row = dict(schema='footballmind-offline-source-assertion-handoff-v1',
               article_id='test:source', source_url='https://example.test/a',
               body_path=str(path), inherited_body_sha256='inherited-not-reverified',
               target_physical_match_id='test:fixture', target_kickoff='2026-10-18T15:30:00Z',
               target_prediction_cutoff='2026-10-18T14:30:00Z',
               prospective_training_eligibility_claimed=False, retain_in_research_pool=True,
               assertions=[dict(assertion_id='test:probability', type='third_party_probability',
                   provenance='source_assertion_not_independent_truth',
                   values=dict(home=.2, draw=.3, away=.5), source_model_calibration='unknown',
                   probability_quote_time=None, not_shadow_f_prediction=True,
                   evidence=dict(quote=quote, start=body.index(quote), end=body.index(quote)+len(quote),
                                 offset_unit='unicode_codepoints'))])
    return row, tmp_path


def test_crlf_preserved_and_record_is_not_modified(inputs):
    row, root = inputs
    original = deepcopy(row)
    path = root/'rows.jsonl'
    path.write_text(json.dumps(row)+'\n', encoding='utf-8')
    assert validate_jsonl(path, body_root=root) == [original]
    assert row == original


def test_quote_offsets_cannot_be_modified(inputs):
    row, root = inputs
    row['assertions'][0]['evidence']['start'] += 1
    with pytest.raises(ValueError, match='does not match'):
        validate_record(row, body_root=root)


def test_third_party_probability_cannot_be_shadow_prediction(inputs):
    row, root = inputs
    row['assertions'][0]['not_shadow_f_prediction'] = False
    with pytest.raises(ValueError, match='shadow f forecast'):
        validate_record(row, body_root=root)


def test_team_context_cannot_be_promoted(inputs):
    row, root = inputs
    row['assertions'][0].update(type='injury', scope='team_context',
                              this_match_assertion=True, target_fixture_status='unknown')
    with pytest.raises(ValueError, match='team context'):
        validate_record(row, body_root=root)


@pytest.mark.parametrize('gain', [0, .2])
def test_even_zero_gain_is_a_label_not_an_empty_value(inputs, gain):
    row, root = inputs
    row['gain'] = gain
    with pytest.raises(ValueError, match='Training field'):
        validate_record(row, body_root=root)


def test_body_path_cannot_escape_root(inputs):
    row, root = inputs
    row['body_path'] = str(root.parent/'outside.txt')
    with pytest.raises(ValueError, match='escapes'):
        validate_record(row, body_root=root)


def test_evidence_check_cannot_be_skipped(inputs):
    row, _ = inputs
    with pytest.raises(ValueError, match='cannot be skipped'):
        validate_record(row, body_root=None)


@pytest.mark.parametrize('value', ['2026-10-18T15:30:00Z', '2026-10-18T14:00:00'])
def test_invalid_cutoff_is_not_accepted(inputs, value):
    row, root = inputs
    row['target_prediction_cutoff'] = value
    with pytest.raises(ValueError):
        validate_record(row, body_root=root)
