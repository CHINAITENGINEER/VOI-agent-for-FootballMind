import pytest
from tools.evaluate_nonleeds_g_probe import reviewed_claim
from tools.verify_nonleeds_g_probe import change


def test_review_requires_unique_verbatim_quote_and_context():
    text = 'Palmer is available for Arsenal. He may start.'
    row = reviewed_claim(text, 'Cole Palmer', 'home', 'confirmed_available',
        'Palmer is available for Arsenal.', 'Can be selected, starting is not confirmed.',
        value='available', context=text)
    assert all(text[s['start']:s['end']] == s['quote'] for s in row['spans'])
    assert row['information_value'] is None
    assert row['automatic_truth_label'] is False
    with pytest.raises(ValueError, match='one exact location'):
        reviewed_claim(text, 'Cole Palmer', 'home', 'confirmed_available',
                       'Palmer will start.', 'Unsupported paraphrase as quotation.')
    with pytest.raises(ValueError, match='one exact location'):
        reviewed_claim(text+' '+text, 'Cole Palmer', 'home', 'confirmed_available',
                       text, 'Ambiguous occurrence.')
    with pytest.raises(ValueError, match='context'):
        reviewed_claim(text, 'Cole Palmer', 'home', 'confirmed_available',
                       'He may start.', 'Possible start.', context='He definitely starts.')


def test_selection_does_not_implicitly_become_medical_availability():
    row = reviewed_claim('Cole Palmer is on the bench.', 'Cole Palmer', 'home',
                         'confirmed_bench', 'Cole Palmer is on the bench.', 'Confirmed selection.')
    assert row['claim']['kind'] == 'other'
    assert row['claim']['value'] == 'unmodeled'
    assert row['concept'] == 'confirmed_bench'


@pytest.mark.parametrize('value,expected', [(.01, '损失降低 0.010000'),
    (-.01, '损失增加 0.010000'), (1e-16, '不变')])
def test_readable_gain_direction(value, expected):
    assert change(value) == expected
